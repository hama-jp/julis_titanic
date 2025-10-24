"""
Stacking Ensemble - メタ学習による高精度アンサンブル

現状: Hard Voting OOF 0.8350
目標: Stacking OOF 0.840 (+0.5%)

戦略:
1. ベースモデル: LightGBM, GradientBoosting, RandomForest
2. メタモデル: LogisticRegression（シンプル、過学習しにくい）
3. Out-of-Fold予測でメタ特徴量生成
4. リーケージ防止（testデータの情報は使わない）
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('STACKING ENSEMBLE - メタ学習による高精度モデル')
print('='*80)
print('目標: Hard Voting OOF 0.8350 → Stacking OOF 0.840 (+0.5%)')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df):
    """9特徴量の基本セット（Deck_Safe除外、LB検証済み）"""
    df = df.copy()

    # Missing flags BEFORE imputation
    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Embarked_Missing'] = df['Embarked'].isnull().astype(int)

    # Title extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    # Age imputation by Title + Pclass
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fare imputation and binning
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    return df

print('\n特徴量エンジニアリング...')
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Encoding
from sklearn.preprocessing import LabelEncoder
le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

# 9特徴量（LB 0.77751で検証済み）
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
            'SmallFamily', 'Age_Missing', 'Embarked_Missing']

# Sex encoding
train_fe['Sex'] = train_fe['Sex'].map({'male': 0, 'female': 1})
test_fe['Sex'] = test_fe['Sex'].map({'male': 0, 'female': 1})

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f'\nTraining data: {len(X_train)} samples')
print(f'Features ({len(features)}): {", ".join(features)}')
print(f'Test data: {len(X_test)} samples')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========================================================================
# ベースモデル定義（既存の最良パラメータ）
# ========================================================================
print('\n' + '='*80)
print('ベースモデル定義')
print('='*80)

base_models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=2,
        num_leaves=4,
        min_child_samples=30,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_split_gain=0.01,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    )
}

print('\n1. LightGBM (Conservative_V1)')
print('   CV Score (記録): 0.8317, Gap: 0.0090')
print('   Parameters: max_depth=2, num_leaves=4, strong regularization')

print('\n2. GradientBoosting (Conservative)')
print('   CV Score (記録): 0.8316, Gap: 0.0314')
print('   Parameters: max_depth=3, learning_rate=0.05')

print('\n3. RandomForest (Conservative)')
print('   CV Score (記録): 0.8294, Gap: 0.0090')
print('   Parameters: max_depth=4, min_samples_leaf=10')

# ========================================================================
# ベースモデルのCV評価（確認）
# ========================================================================
print('\n' + '='*80)
print('ベースモデルのCV評価（確認）')
print('='*80)

base_cv_scores = {}
for name, model in base_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    base_cv_scores[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }
    print(f'\n{name}:')
    print(f'  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')

# ========================================================================
# Hard Voting（ベースライン）
# ========================================================================
print('\n' + '='*80)
print('ベースライン: Hard Voting')
print('='*80)

from sklearn.ensemble import VotingClassifier

voting_hard = VotingClassifier(
    estimators=[
        ('lgb', base_models['LightGBM']),
        ('gb', base_models['GradientBoosting']),
        ('rf', base_models['RandomForest'])
    ],
    voting='hard'
)

# OOF予測でHard Votingを評価
oof_pred_hard = cross_val_predict(voting_hard, X_train, y_train, cv=cv)
hard_oof_score = accuracy_score(y_train, oof_pred_hard)

print(f'\nHard Voting OOF Score: {hard_oof_score:.4f}')
print('（期待: 0.8350前後）')

# ========================================================================
# Stacking Ensemble
# ========================================================================
print('\n' + '='*80)
print('Stacking Ensemble')
print('='*80)

# メタモデルの候補
meta_models = {
    'LogisticRegression_C01': LogisticRegression(C=0.1, random_state=42, max_iter=1000),
    'LogisticRegression_C05': LogisticRegression(C=0.5, random_state=42, max_iter=1000),
    'LogisticRegression_C10': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
}

stacking_results = []

for meta_name, meta_model in meta_models.items():
    print(f'\n【Stacking with {meta_name}】')

    stacking = StackingClassifier(
        estimators=[
            ('lgb', base_models['LightGBM']),
            ('gb', base_models['GradientBoosting']),
            ('rf', base_models['RandomForest'])
        ],
        final_estimator=meta_model,
        cv=5,  # 内部CVでOOF予測生成
        passthrough=False,  # 元の特徴量は使わない（メタ特徴量のみ）
        n_jobs=-1
    )

    # OOF予測
    oof_pred_stacking = cross_val_predict(stacking, X_train, y_train, cv=cv)
    stacking_oof_score = accuracy_score(y_train, oof_pred_stacking)

    # CV評価
    cv_scores = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='accuracy')

    result = {
        'meta_model': meta_name,
        'oof_score': stacking_oof_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    stacking_results.append(result)

    print(f'  OOF Score: {stacking_oof_score:.4f}')
    print(f'  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
    print(f'  vs Hard Voting: {stacking_oof_score - hard_oof_score:+.4f}')

# ========================================================================
# Stacking with Passthrough（元の特徴量も使用）
# ========================================================================
print('\n' + '='*80)
print('Stacking with Passthrough（元特徴量 + メタ特徴量）')
print('='*80)

stacking_passthrough = StackingClassifier(
    estimators=[
        ('lgb', base_models['LightGBM']),
        ('gb', base_models['GradientBoosting']),
        ('rf', base_models['RandomForest'])
    ],
    final_estimator=LogisticRegression(C=0.1, random_state=42, max_iter=1000),
    cv=5,
    passthrough=True,  # 元の特徴量も使う
    n_jobs=-1
)

oof_pred_passthrough = cross_val_predict(stacking_passthrough, X_train, y_train, cv=cv)
passthrough_oof_score = accuracy_score(y_train, oof_pred_passthrough)

cv_scores_passthrough = cross_val_score(stacking_passthrough, X_train, y_train, cv=cv, scoring='accuracy')

print(f'\nOOF Score: {passthrough_oof_score:.4f}')
print(f'CV Score: {cv_scores_passthrough.mean():.4f} (±{cv_scores_passthrough.std():.4f})')
print(f'vs Hard Voting: {passthrough_oof_score - hard_oof_score:+.4f}')

# ========================================================================
# 結果サマリー
# ========================================================================
print('\n' + '='*80)
print('結果サマリー')
print('='*80)

all_results = pd.DataFrame([
    {'Model': 'Hard Voting (Baseline)', 'OOF Score': hard_oof_score, 'Improvement': 0.0},
    {'Model': 'Stacking (C=0.1)', 'OOF Score': stacking_results[0]['oof_score'],
     'Improvement': stacking_results[0]['oof_score'] - hard_oof_score},
    {'Model': 'Stacking (C=0.5)', 'OOF Score': stacking_results[1]['oof_score'],
     'Improvement': stacking_results[1]['oof_score'] - hard_oof_score},
    {'Model': 'Stacking (C=1.0)', 'OOF Score': stacking_results[2]['oof_score'],
     'Improvement': stacking_results[2]['oof_score'] - hard_oof_score},
    {'Model': 'Stacking + Passthrough', 'OOF Score': passthrough_oof_score,
     'Improvement': passthrough_oof_score - hard_oof_score}
]).sort_values('OOF Score', ascending=False)

print('\n【OOF Score Ranking】')
print(all_results.to_string(index=False))

best_model = all_results.iloc[0]
print(f'\n🏆 Best Model: {best_model["Model"]}')
print(f'   OOF Score: {best_model["OOF Score"]:.4f}')
print(f'   Improvement: {best_model["Improvement"]:+.4f} ({best_model["Improvement"]/hard_oof_score*100:+.2f}%)')

# 目標達成確認
target_oof = 0.840
if best_model['OOF Score'] >= target_oof:
    print(f'\n✅ 目標達成！ OOF {target_oof:.3f}以上達成')
else:
    print(f'\n⚠️  目標未達: OOF {best_model["OOF Score"]:.4f} < 目標 {target_oof:.3f}')
    print(f'   差分: {target_oof - best_model["OOF Score"]:.4f}')

# ========================================================================
# 最良モデルで提出ファイル生成
# ========================================================================
print('\n' + '='*80)
print('提出ファイル生成')
print('='*80)

# 最良のStackingモデルを再訓練
if best_model['Model'] == 'Hard Voting (Baseline)':
    # Hard Votingが最良の場合
    final_stacking = voting_hard
    print('最良モデル: Hard Voting（Stackingより優秀）')
elif best_model['Model'] == 'Stacking + Passthrough':
    final_stacking = stacking_passthrough
else:
    # メタモデルのC値を特定
    if 'C=' in best_model['Model']:
        c_value = float(best_model['Model'].split('C=')[1].rstrip(')'))
    else:
        c_value = 0.1
    final_stacking = StackingClassifier(
        estimators=[
            ('lgb', base_models['LightGBM']),
            ('gb', base_models['GradientBoosting']),
            ('rf', base_models['RandomForest'])
        ],
        final_estimator=LogisticRegression(C=c_value, random_state=42, max_iter=1000),
        cv=5,
        passthrough=False,
        n_jobs=-1
    )

print('\n訓練中...')
final_stacking.fit(X_train, y_train)

# 予測
predictions = final_stacking.predict(X_test)

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

filename = 'submission_stacking_ensemble.csv'
submission.to_csv(filename, index=False)

survival_rate = predictions.mean()
print(f'\n✅ {filename}')
print(f'   Model: {best_model["Model"]}')
print(f'   OOF Score: {best_model["OOF Score"]:.4f}')
print(f'   Survival Rate: {survival_rate:.3f}')

# ========================================================================
# 比較用: Hard Votingも生成
# ========================================================================
print('\n比較用: Hard Voting提出ファイル')
voting_hard.fit(X_train, y_train)
voting_pred = voting_hard.predict(X_test)

submission_voting = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': voting_pred
})

filename_voting = 'submission_hard_voting_baseline.csv'
submission_voting.to_csv(filename_voting, index=False)

print(f'✅ {filename_voting}')
print(f'   OOF Score: {hard_oof_score:.4f}')
print(f'   Survival Rate: {voting_pred.mean():.3f}')

# 結果をCSVに保存
results_df = all_results.copy()
results_df.to_csv('experiments/results/stacking_ensemble_results.csv', index=False)
print(f'\n✅ 結果を保存: experiments/results/stacking_ensemble_results.csv')

print('\n' + '='*80)
print('STACKING ENSEMBLE完了！')
print('='*80)

print('\n推奨提出順序:')
print(f'1. {filename} (Stacking - 最良OOF)')
print(f'2. {filename_voting} (Hard Voting - ベースライン)')
print('\nKaggleで検証してください！')
