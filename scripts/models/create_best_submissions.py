"""
有望なサブミッションファイル作成

提出回数5回を大切に使うため、最も有望なバリエーションを厳選して作成：
1. Hard Voting (LGB + GB + RF)
2. Soft Voting (確率平均)
3. Weighted Soft Voting (CVスコア比例)
4. Best Single Model (RandomForest)

各モデルのOOFスコアを計算して推奨順位を決定
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('有望なサブミッションファイル作成')
print('='*80)
print('提出回数5回を厳選して使用')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df):
    """9特徴量（LB 0.77751検証済み）"""
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

print(f'Train: {len(X_train)} samples, Test: {len(X_test)} samples')
print(f'Features: {", ".join(features)}')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========================================================================
# ベースモデル定義
# ========================================================================
print('\n' + '='*80)
print('ベースモデル（既存最良パラメータ）')
print('='*80)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
    reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_samples_split=20, min_samples_leaf=10, subsample=0.8,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_split=20,
    min_samples_leaf=10, max_features='sqrt', random_state=42
)

print('✓ LightGBM (CV 0.8317記録)')
print('✓ GradientBoosting (CV 0.8316記録)')
print('✓ RandomForest (CV 0.8294記録)')

# ========================================================================
# サブミッション候補作成
# ========================================================================
print('\n' + '='*80)
print('サブミッション候補を作成')
print('='*80)

submissions = []

# ========================================================================
# 1. Best Single Model (RandomForest)
# ========================================================================
print('\n【1. Best Single Model - RandomForest】')
print('   理由: 単一モデルで最高CV、シンプルで安定')

oof_rf = cross_val_predict(rf_model, X_train, y_train, cv=cv)
oof_score_rf = accuracy_score(y_train, oof_rf)

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

submissions.append({
    'name': 'Best_Single_RF',
    'filename': 'submission_best_single_rf.csv',
    'model': 'RandomForest',
    'oof_score': oof_score_rf,
    'predictions': pred_rf,
    'survival_rate': pred_rf.mean(),
    'description': '単一モデル最良、シンプル'
})

print(f'   OOF Score: {oof_score_rf:.4f}')
print(f'   Survival Rate: {pred_rf.mean():.3f}')

# ========================================================================
# 2. Hard Voting
# ========================================================================
print('\n【2. Hard Voting (LGB + GB + RF)】')
print('   理由: 現状の実績モデル、多数決で安定')

voting_hard = VotingClassifier(
    estimators=[('lgb', lgb_model), ('gb', gb_model), ('rf', rf_model)],
    voting='hard'
)

oof_hard = cross_val_predict(voting_hard, X_train, y_train, cv=cv)
oof_score_hard = accuracy_score(y_train, oof_hard)

voting_hard.fit(X_train, y_train)
pred_hard = voting_hard.predict(X_test)

submissions.append({
    'name': 'Hard_Voting',
    'filename': 'submission_hard_voting.csv',
    'model': 'Hard Voting (LGB+GB+RF)',
    'oof_score': oof_score_hard,
    'predictions': pred_hard,
    'survival_rate': pred_hard.mean(),
    'description': '多数決アンサンブル、安定性高'
})

print(f'   OOF Score: {oof_score_hard:.4f}')
print(f'   Survival Rate: {pred_hard.mean():.3f}')

# ========================================================================
# 3. Soft Voting
# ========================================================================
print('\n【3. Soft Voting (LGB + GB + RF)】')
print('   理由: 確率平均でHard Votingより柔軟')

voting_soft = VotingClassifier(
    estimators=[('lgb', lgb_model), ('gb', gb_model), ('rf', rf_model)],
    voting='soft'
)

oof_soft = cross_val_predict(voting_soft, X_train, y_train, cv=cv)
oof_score_soft = accuracy_score(y_train, oof_soft)

voting_soft.fit(X_train, y_train)
pred_soft = voting_soft.predict(X_test)

submissions.append({
    'name': 'Soft_Voting',
    'filename': 'submission_soft_voting.csv',
    'model': 'Soft Voting (LGB+GB+RF)',
    'oof_score': oof_score_soft,
    'predictions': pred_soft,
    'survival_rate': pred_soft.mean(),
    'description': '確率平均アンサンブル、柔軟性高'
})

print(f'   OOF Score: {oof_score_soft:.4f}')
print(f'   Survival Rate: {pred_soft.mean():.3f}')

# ========================================================================
# 4. Weighted Soft Voting (OOFスコア比例)
# ========================================================================
print('\n【4. Weighted Soft Voting】')
print('   理由: 各モデルの性能に応じた重み付け')

# 各モデルのOOFスコア取得
oof_lgb = cross_val_predict(lgb_model, X_train, y_train, cv=cv)
oof_gb = cross_val_predict(gb_model, X_train, y_train, cv=cv)
# oof_rf は既に計算済み

oof_score_lgb = accuracy_score(y_train, oof_lgb)
oof_score_gb = accuracy_score(y_train, oof_gb)
# oof_score_rf は既に計算済み

# 重みを正規化
total_score = oof_score_lgb + oof_score_gb + oof_score_rf
weights = {
    'lgb': oof_score_lgb / total_score,
    'gb': oof_score_gb / total_score,
    'rf': oof_score_rf / total_score
}

print(f'   Weights: LGB={weights["lgb"]:.3f}, GB={weights["gb"]:.3f}, RF={weights["rf"]:.3f}')

# 重み付きSoft Voting
voting_weighted = VotingClassifier(
    estimators=[('lgb', lgb_model), ('gb', gb_model), ('rf', rf_model)],
    voting='soft',
    weights=[weights['lgb'], weights['gb'], weights['rf']]
)

oof_weighted = cross_val_predict(voting_weighted, X_train, y_train, cv=cv)
oof_score_weighted = accuracy_score(y_train, oof_weighted)

voting_weighted.fit(X_train, y_train)
pred_weighted = voting_weighted.predict(X_test)

submissions.append({
    'name': 'Weighted_Soft_Voting',
    'filename': 'submission_weighted_soft_voting.csv',
    'model': 'Weighted Soft Voting',
    'oof_score': oof_score_weighted,
    'predictions': pred_weighted,
    'survival_rate': pred_weighted.mean(),
    'description': f'重み付きアンサンブル (LGB:{weights["lgb"]:.2f}, GB:{weights["gb"]:.2f}, RF:{weights["rf"]:.2f})'
})

print(f'   OOF Score: {oof_score_weighted:.4f}')
print(f'   Survival Rate: {pred_weighted.mean():.3f}')

# ========================================================================
# 5. LightGBM単体（最良個別モデル候補）
# ========================================================================
print('\n【5. LightGBM Single】')
print('   理由: 記録上最高個別CV 0.8317、Gap最小 0.0090')

oof_lgb_single = cross_val_predict(lgb_model, X_train, y_train, cv=cv)
oof_score_lgb_single = accuracy_score(y_train, oof_lgb_single)

lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_test)

submissions.append({
    'name': 'LightGBM_Single',
    'filename': 'submission_lightgbm_single.csv',
    'model': 'LightGBM',
    'oof_score': oof_score_lgb_single,
    'predictions': pred_lgb,
    'survival_rate': pred_lgb.mean(),
    'description': '最良個別モデル、過学習最小'
})

print(f'   OOF Score: {oof_score_lgb_single:.4f}')
print(f'   Survival Rate: {pred_lgb.mean():.3f}')

# ========================================================================
# 結果サマリーと推奨順位
# ========================================================================
print('\n' + '='*80)
print('サブミッション候補サマリー')
print('='*80)

# DataFrameに整理
df_submissions = pd.DataFrame(submissions)
df_submissions = df_submissions.sort_values('oof_score', ascending=False).reset_index(drop=True)

print('\n【OOFスコア順ランキング】')
for idx, row in df_submissions.iterrows():
    print(f"\n{idx+1}位: {row['name']}")
    print(f"     OOF Score: {row['oof_score']:.4f}")
    print(f"     Model: {row['model']}")
    print(f"     Survival Rate: {row['survival_rate']:.3f}")
    print(f"     説明: {row['description']}")

# ファイル生成
print('\n' + '='*80)
print('サブミッションファイル生成')
print('='*80)

for idx, row in df_submissions.iterrows():
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': row['predictions']
    })
    submission.to_csv(row['filename'], index=False)
    print(f"✅ {row['filename']}")

# 推奨順位を保存
recommendation = df_submissions[['name', 'filename', 'model', 'oof_score', 'survival_rate', 'description']].copy()
recommendation.to_csv('submission_recommendations.csv', index=False)
print(f"\n✅ submission_recommendations.csv (推奨順位)")

# ========================================================================
# 提出推奨
# ========================================================================
print('\n' + '='*80)
print('📊 提出推奨（5回を厳選）')
print('='*80)

print('\n【推奨提出順序】\n')

best = df_submissions.iloc[0]
second = df_submissions.iloc[1]
third = df_submissions.iloc[2]

print(f"🥇 1st提出: {best['filename']}")
print(f"   モデル: {best['model']}")
print(f"   OOF: {best['oof_score']:.4f} (最高スコア)")
print(f"   理由: OOF最高、最も有望")

print(f"\n🥈 2nd提出: {second['filename']}")
print(f"   モデル: {second['model']}")
print(f"   OOF: {second['oof_score']:.4f}")
print(f"   理由: OOF 2位、{best['name']}と比較検証")

print(f"\n🥉 3rd提出: {third['filename']}")
print(f"   モデル: {third['model']}")
print(f"   OOF: {third['oof_score']:.4f}")
print(f"   理由: OOF 3位、多様性確保")

# 予測の一致率を確認
print('\n' + '='*80)
print('予測の一致率（多様性確認）')
print('='*80)

for i in range(len(df_submissions)):
    for j in range(i+1, len(df_submissions)):
        name1 = df_submissions.iloc[i]['name']
        name2 = df_submissions.iloc[j]['name']
        pred1 = df_submissions.iloc[i]['predictions']
        pred2 = df_submissions.iloc[j]['predictions']

        agreement = (pred1 == pred2).mean()
        print(f"{name1} vs {name2}: {agreement:.1%} 一致")

print('\n' + '='*80)
print('サブミッション作成完了！')
print('='*80)
print('\n✅ 5つの候補ファイルを作成しました')
print('✅ OOFスコア順に推奨順位を決定しました')
print('✅ submission_recommendations.csv で詳細確認できます')
print('\n慎重に提出してください！残り5回です。')
