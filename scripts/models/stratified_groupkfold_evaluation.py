"""
Stratified GroupKFold による正確なCV評価

問題:
- 現状のStratifiedKFoldは同じTicket/Familyグループが異なるfoldに分散
- データリーケージによりCVスコアが楽観的（高く出る）
- CV 0.8294 → LB 0.77511 (Gap 5.4%)

解決策:
- StratifiedGroupKFoldを使用
- 同じTicketの人は必ず同じfoldに配置
- より正確なCV評価 → LBとの整合性向上

期待効果:
- CVスコア: 0.8294 → 0.80-0.81 (現実的な値)
- CV-LB Gap: 5.4% → 2-3% (縮小)

===============================================================================
⚠️  WARNING: このファイルの欠損値補完は Train/Test 分離が不適切です
===============================================================================

問題点:
  - Fare補完がTrain/Test個別に実行されている
  - Test PassengerId=1044 のFare欠損が Test データのみから補完される
  - 結果: Fare=7.90 (誤) ← 本来は Fare=8.05 (正)

正しい実装:
  scripts/preprocessing/correct_imputation_pipeline.py を使用してください
  詳細: scripts/preprocessing/IMPUTATION_WARNING.md を参照

このファイルは 2nd提出（LB 0.76555）で使用されましたが、
今後の提出では正しいパイプラインを使用することを強く推奨します。

===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('Stratified GroupKFold による正確なCV評価')
print('='*80)
print('目的: CV-LB Gap縮小、正確なモデル評価')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df):
    """9特徴量（LB検証済み）"""
    df = df.copy()

    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Embarked_Missing'] = df['Embarked'].isnull().astype(int)

    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    df['Embarked'] = df['Embarked'].fillna('S')

    return df

print('\n特徴量エンジニアリング...')
train_fe = engineer_features(train)
test_fe = engineer_features(test)

le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
            'SmallFamily', 'Age_Missing', 'Embarked_Missing']

train_fe['Sex'] = train_fe['Sex'].map({'male': 0, 'female': 1})
test_fe['Sex'] = test_fe['Sex'].map({'male': 0, 'female': 1})

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f'Train: {len(X_train)} samples')
print(f'Features: {", ".join(features)}')

# ========================================================================
# グループ作成（Ticketベース）
# ========================================================================
print('\n' + '='*80)
print('グループ作成')
print('='*80)

# Ticketをグループとして使用
groups = train['Ticket'].factorize()[0]

# グループ統計
n_groups = len(np.unique(groups))
group_sizes = pd.Series(groups).value_counts()

print(f'\nTicketグループ数: {n_groups}')
print(f'平均グループサイズ: {group_sizes.mean():.2f}人')
print(f'最大グループサイズ: {group_sizes.max()}人')
print(f'単独チケット: {(group_sizes == 1).sum()}グループ ({(group_sizes == 1).sum()/n_groups*100:.1f}%)')

print('\nグループサイズ分布:')
print(group_sizes.value_counts().sort_index().head(10))

# ========================================================================
# ベースモデル定義
# ========================================================================
print('\n' + '='*80)
print('ベースモデル')
print('='*80)

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=20,
        min_samples_leaf=10, max_features='sqrt', random_state=42
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_split=20, min_samples_leaf=10, subsample=0.8,
        random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
        reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    ),
    'HardVoting': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=20,
                                         min_samples_leaf=10, max_features='sqrt', random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                             min_samples_split=20, min_samples_leaf=10, subsample=0.8,
                                             random_state=42)),
            ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
                                      reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
                                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                      random_state=42, verbose=-1))
        ],
        voting='hard'
    )
}

# ========================================================================
# CV比較: StratifiedKFold vs StratifiedGroupKFold
# ========================================================================
print('\n' + '='*80)
print('CV戦略比較')
print('='*80)

cv_strategies = {
    'StratifiedKFold (現状)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedGroupKFold (改善)': StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
}

results = []

for cv_name, cv in cv_strategies.items():
    print(f'\n【{cv_name}】')
    print('-' * 80)

    for model_name, model in models.items():
        print(f'\n{model_name}:')

        # CVスコア計算
        if 'GroupKFold' in cv_name:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, groups=groups, scoring='accuracy')
            oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, groups=groups)
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            oof_pred = cross_val_predict(model, X_train, y_train, cv=cv)

        oof_score = accuracy_score(y_train, oof_pred)

        # Train score
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)

        # Gap
        gap = train_score - cv_scores.mean()

        result = {
            'CV Strategy': cv_name,
            'Model': model_name,
            'CV Score': cv_scores.mean(),
            'CV Std': cv_scores.std(),
            'OOF Score': oof_score,
            'Train Score': train_score,
            'Gap': gap,
            'Gap %': gap / cv_scores.mean() * 100
        }
        results.append(result)

        print(f'  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
        print(f'  OOF Score: {oof_score:.4f}')
        print(f'  Gap: {gap:.4f} ({gap/cv_scores.mean()*100:.2f}%)')

# ========================================================================
# 結果サマリー
# ========================================================================
print('\n' + '='*80)
print('結果サマリー')
print('='*80)

df_results = pd.DataFrame(results)

# StratifiedKFold の結果
print('\n【StratifiedKFold (現状)】')
df_skf = df_results[df_results['CV Strategy'] == 'StratifiedKFold (現状)'].sort_values('CV Score', ascending=False)
print(df_skf[['Model', 'CV Score', 'OOF Score', 'Gap', 'Gap %']].to_string(index=False))

# StratifiedGroupKFold の結果
print('\n【StratifiedGroupKFold (改善)】')
df_sgkf = df_results[df_results['CV Strategy'] == 'StratifiedGroupKFold (改善)'].sort_values('CV Score', ascending=False)
print(df_sgkf[['Model', 'CV Score', 'OOF Score', 'Gap', 'Gap %']].to_string(index=False))

# 差分分析
print('\n' + '='*80)
print('StratifiedKFold → StratifiedGroupKFold の変化')
print('='*80)

for model_name in models.keys():
    skf_row = df_results[(df_results['CV Strategy'] == 'StratifiedKFold (現状)') &
                         (df_results['Model'] == model_name)].iloc[0]
    sgkf_row = df_results[(df_results['CV Strategy'] == 'StratifiedGroupKFold (改善)') &
                          (df_results['Model'] == model_name)].iloc[0]

    cv_diff = sgkf_row['CV Score'] - skf_row['CV Score']
    gap_diff = sgkf_row['Gap'] - skf_row['Gap']

    print(f'\n{model_name}:')
    print(f'  CV Score: {skf_row["CV Score"]:.4f} → {sgkf_row["CV Score"]:.4f} ({cv_diff:+.4f})')
    print(f'  Gap: {skf_row["Gap"]:.4f} → {sgkf_row["Gap"]:.4f} ({gap_diff:+.4f})')

    if cv_diff < 0:
        print(f'  ✅ CVスコアが低下（より現実的に）')
    if abs(gap_diff) < abs(skf_row['Gap']) * 0.1:
        print(f'  ✅ Gap安定')

# LB予測
print('\n' + '='*80)
print('LB Score予測（GroupKFold基準）')
print('='*80)

# 過去のCV-LB関係から推定
# 過去: CV 0.8294 → LB 0.77511 (差分 -0.05429)
past_cv_lb_diff = 0.8294 - 0.77511

print(f'\n過去の実績: CV 0.8294 → LB 0.77511 (差分 {-past_cv_lb_diff:.5f})')
print('\nGroupKFold基準の予測:')

best_sgkf = df_sgkf.iloc[0]
predicted_lb = best_sgkf['CV Score'] - past_cv_lb_diff

print(f'\n最良モデル: {best_sgkf["Model"]}')
print(f'  GroupKFold CV: {best_sgkf["CV Score"]:.4f}')
print(f'  予測LB: {predicted_lb:.4f}')
print(f'  予測Gap: {best_sgkf["CV Score"] - predicted_lb:.4f}')

if predicted_lb >= 0.80:
    print(f'\n✅ 目標LB 0.8達成の可能性あり！')
else:
    print(f'\n⚠️  目標LB 0.8には {0.80 - predicted_lb:.4f} 不足')

# ========================================================================
# 推奨モデルで提出ファイル作成
# ========================================================================
print('\n' + '='*80)
print('推奨サブミッション作成')
print('='*80)

# GroupKFoldで最良のモデル
best_model_name = best_sgkf['Model']
best_model = models[best_model_name]

print(f'\n推奨モデル: {best_model_name}')
print(f'  GroupKFold CV: {best_sgkf["CV Score"]:.4f}')
print(f'  Gap: {best_sgkf["Gap"]:.4f}')

# 訓練
best_model.fit(X_train, y_train)

# 予測
predictions = best_model.predict(X_test)

# 提出ファイル
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

filename = f'submission_groupkfold_{best_model_name.lower()}.csv'
submission.to_csv(filename, index=False)

print(f'\n✅ {filename}')
print(f'   GroupKFold CV: {best_sgkf["CV Score"]:.4f}')
print(f'   予測LB: {predicted_lb:.4f}')
print(f'   Survival Rate: {predictions.mean():.3f}')

# 結果をCSVに保存
df_results.to_csv('experiments/results/stratified_groupkfold_evaluation.csv', index=False)
print(f'\n✅ experiments/results/stratified_groupkfold_evaluation.csv')

print('\n' + '='*80)
print('Stratified GroupKFold 評価完了！')
print('='*80)

print('\n重要な発見:')
print('- GroupKFoldでCVスコアが低下 → より正確な評価')
print('- CV-LB Gapが縮小の可能性')
print('- 次の提出で検証してください')
