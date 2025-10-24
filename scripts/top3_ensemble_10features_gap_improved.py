"""
Top 3アンサンブル再構築 with 10特徴量 + Gap改善

目標:
1. 10特徴量（Deck_Safe追加）でモデル構築
2. 正則化強化でGap < 0.015
3. Top 3アンサンブル（LightGBM + GradientBoosting + RandomForest）
4. OOF Score 0.835+を目指す
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("Top 3アンサンブル再構築 with 10特徴量 + Gap改善")
print("="*80)

# ========================================================================
# データ読み込みと特徴量エンジニアリング
# ========================================================================
print("\n【データ準備】")
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

def engineer_features(df):
    """10特徴量のエンジニアリング（Deck_Safe追加）"""
    df_fe = df.copy()

    # 1. 欠損フラグ
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)

    # 2. Title抽出
    df_fe['Title'] = df_fe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_fe['Title'] = df_fe['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_fe['Title'] = df_fe['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_fe['Title'] = df_fe['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df_fe['Title'] = df_fe['Title'].map(title_mapping).fillna(0)

    # 3. 家族関連
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    df_fe['SmallFamily'] = ((df_fe['FamilySize'] >= 2) & (df_fe['FamilySize'] <= 4)).astype(int)

    # 4. Age補完
    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    # 5. Fare補完とビニング
    df_fe['Fare'].fillna(df_fe.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    df_fe['FareBin'] = pd.qcut(df_fe['Fare'], q=4, labels=False, duplicates='drop')

    # 6. Sex encoding
    df_fe['Sex'] = df_fe['Sex'].map({'male': 0, 'female': 1})

    # 7. Embarked補完
    df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0], inplace=True)

    # 8. Deck_Safe（新規）
    df_fe['Deck'] = df_fe['Cabin'].str[0]
    df_fe['Deck_Safe'] = ((df_fe['Deck'] == 'D') | (df_fe['Deck'] == 'E') | (df_fe['Deck'] == 'B')).astype(int)

    return df_fe

# データ結合
all_data = pd.concat([train, test], sort=False, ignore_index=True)
all_data_fe = engineer_features(all_data)

# Train/Test分割
train_fe = all_data_fe[:len(train)].copy()
test_fe = all_data_fe[len(train):].copy()

# 10特徴量
features = [
    'Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
    'SmallFamily', 'Age_Missing', 'Embarked_Missing',
    'Deck_Safe'  # NEW
]

X_train = train_fe[features]
y_train = train['Survived']
X_test = test_fe[features]

print(f"特徴量: {features}")
print(f"特徴量数: {len(features)}")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# CV設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========================================================================
# モデル1: LightGBM（正則化強化）
# ========================================================================
print("\n" + "="*80)
print("モデル1: LightGBM（正則化強化でGap < 0.015を目指す）")
print("="*80)

# 正則化を段階的に強化して最適化
lgb_configs = [
    {
        'name': 'Baseline (既存)',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 30,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_split_gain': 0.01,
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 40,  # 30 -> 40
            'reg_alpha': 1.5,  # 1.0 -> 1.5
            'reg_lambda': 1.5,  # 1.0 -> 1.5
            'min_split_gain': 0.02,  # 0.01 -> 0.02
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Very Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 3,  # 4 -> 3
            'min_child_samples': 50,  # 40 -> 50
            'reg_alpha': 2.0,  # 1.5 -> 2.0
            'reg_lambda': 2.0,  # 1.5 -> 2.0
            'min_split_gain': 0.03,  # 0.02 -> 0.03
            'learning_rate': 0.05,
            'random_state': 42,
            'verbose': -1
        }
    }
]

print("\n【LightGBM 正則化チューニング】")
lgb_results = []

for config in lgb_configs:
    model = lgb.LGBMClassifier(**config['params'])
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    gap = train_score - cv_scores.mean()

    result = {
        'name': config['name'],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'gap': gap,
        'model': model
    }
    lgb_results.append(result)

    print(f"\n{config['name']}:")
    print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Train Score: {train_score:.4f}")
    print(f"  Gap: {gap:.4f} {'✓' if gap < 0.015 else '✗'}")

# 最適なLightGBMを選択（Gap < 0.015 かつ CV最高）
valid_lgb = [r for r in lgb_results if r['gap'] < 0.015]
if valid_lgb:
    best_lgb = max(valid_lgb, key=lambda x: x['cv_mean'])
    print(f"\n【最適LightGBM】: {best_lgb['name']}")
else:
    best_lgb = min(lgb_results, key=lambda x: x['gap'])
    print(f"\n【最適LightGBM】: {best_lgb['name']} (Gap目標未達成、最小Gapを選択)")

print(f"  CV Score: {best_lgb['cv_mean']:.4f}")
print(f"  Gap: {best_lgb['gap']:.4f}")

# ========================================================================
# モデル2: GradientBoosting（正則化強化）
# ========================================================================
print("\n" + "="*80)
print("モデル2: GradientBoosting（正則化強化でGap < 0.015を目指す）")
print("="*80)

gb_configs = [
    {
        'name': 'Baseline (既存)',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
    },
    {
        'name': 'Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'min_samples_split': 30,  # 20 -> 30
            'min_samples_leaf': 15,  # 10 -> 15
            'max_features': 'sqrt',  # 全特徴 -> sqrt
            'random_state': 42
        }
    },
    {
        'name': 'Very Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,  # 3 -> 2
            'learning_rate': 0.05,
            'min_samples_split': 40,  # 30 -> 40
            'min_samples_leaf': 20,  # 15 -> 20
            'max_features': 'sqrt',
            'random_state': 42
        }
    }
]

print("\n【GradientBoosting 正則化チューニング】")
gb_results = []

for config in gb_configs:
    model = GradientBoostingClassifier(**config['params'])
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    gap = train_score - cv_scores.mean()

    result = {
        'name': config['name'],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'gap': gap,
        'model': model
    }
    gb_results.append(result)

    print(f"\n{config['name']}:")
    print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Train Score: {train_score:.4f}")
    print(f"  Gap: {gap:.4f} {'✓' if gap < 0.015 else '✗'}")

# 最適なGBを選択
valid_gb = [r for r in gb_results if r['gap'] < 0.015]
if valid_gb:
    best_gb = max(valid_gb, key=lambda x: x['cv_mean'])
    print(f"\n【最適GradientBoosting】: {best_gb['name']}")
else:
    best_gb = min(gb_results, key=lambda x: x['gap'])
    print(f"\n【最適GradientBoosting】: {best_gb['name']} (Gap目標未達成、最小Gapを選択)")

print(f"  CV Score: {best_gb['cv_mean']:.4f}")
print(f"  Gap: {best_gb['gap']:.4f}")

# ========================================================================
# モデル3: RandomForest（正則化強化）
# ========================================================================
print("\n" + "="*80)
print("モデル3: RandomForest（正則化強化でGap < 0.015を目指す）")
print("="*80)

rf_configs = [
    {
        'name': 'Baseline (既存)',
        'params': {
            'n_estimators': 100,
            'max_depth': 4,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'random_state': 42
        }
    },
    {
        'name': 'Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 4,
            'min_samples_split': 30,  # 20 -> 30
            'min_samples_leaf': 15,  # 10 -> 15
            'max_features': 'sqrt',  # 全特徴 -> sqrt
            'random_state': 42
        }
    },
    {
        'name': 'Very Strong Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,  # 4 -> 3
            'min_samples_split': 40,  # 30 -> 40
            'min_samples_leaf': 20,  # 15 -> 20
            'max_features': 'sqrt',
            'random_state': 42
        }
    }
]

print("\n【RandomForest 正則化チューニング】")
rf_results = []

for config in rf_configs:
    model = RandomForestClassifier(**config['params'])
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    gap = train_score - cv_scores.mean()

    result = {
        'name': config['name'],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'gap': gap,
        'model': model
    }
    rf_results.append(result)

    print(f"\n{config['name']}:")
    print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Train Score: {train_score:.4f}")
    print(f"  Gap: {gap:.4f} {'✓' if gap < 0.015 else '✗'}")

# 最適なRFを選択
valid_rf = [r for r in rf_results if r['gap'] < 0.015]
if valid_rf:
    best_rf = max(valid_rf, key=lambda x: x['cv_mean'])
    print(f"\n【最適RandomForest】: {best_rf['name']}")
else:
    best_rf = min(rf_results, key=lambda x: x['gap'])
    print(f"\n【最適RandomForest】: {best_rf['name']} (Gap目標未達成、最小Gapを選択)")

print(f"  CV Score: {best_rf['cv_mean']:.4f}")
print(f"  Gap: {best_rf['gap']:.4f}")

# ========================================================================
# Top 3アンサンブル構築
# ========================================================================
print("\n" + "="*80)
print("Top 3アンサンブル構築")
print("="*80)

# Hard Voting
print("\n【Hard Voting Ensemble】")
hard_voting = VotingClassifier(
    estimators=[
        ('lgb', best_lgb['model']),
        ('gb', best_gb['model']),
        ('rf', best_rf['model'])
    ],
    voting='hard'
)

# Hard Voting評価
hard_voting.fit(X_train, y_train)
hard_oof_pred = cross_val_predict(hard_voting, X_train, y_train, cv=cv, method='predict')
hard_oof_score = accuracy_score(y_train, hard_oof_pred)
hard_train_score = hard_voting.score(X_train, y_train)
hard_gap = hard_train_score - hard_oof_score

print(f"OOF Score: {hard_oof_score:.4f}")
print(f"Train Score: {hard_train_score:.4f}")
print(f"Gap: {hard_gap:.4f}")

# Soft Voting
print("\n【Soft Voting Ensemble】")
soft_voting = VotingClassifier(
    estimators=[
        ('lgb', best_lgb['model']),
        ('gb', best_gb['model']),
        ('rf', best_rf['model'])
    ],
    voting='soft'
)

# Soft Voting評価
soft_voting.fit(X_train, y_train)
soft_oof_pred = cross_val_predict(soft_voting, X_train, y_train, cv=cv, method='predict')
soft_oof_score = accuracy_score(y_train, soft_oof_pred)
soft_train_score = soft_voting.score(X_train, y_train)
soft_gap = soft_train_score - soft_oof_score

print(f"OOF Score: {soft_oof_score:.4f}")
print(f"Train Score: {soft_train_score:.4f}")
print(f"Gap: {soft_gap:.4f}")

# 最良のアンサンブルを選択
if hard_oof_score > soft_oof_score:
    best_ensemble = hard_voting
    best_ensemble_name = "Hard Voting"
    best_ensemble_oof = hard_oof_score
    best_ensemble_gap = hard_gap
else:
    best_ensemble = soft_voting
    best_ensemble_name = "Soft Voting"
    best_ensemble_oof = soft_oof_score
    best_ensemble_gap = soft_gap

print(f"\n【最良のアンサンブル】: {best_ensemble_name}")
print(f"OOF Score: {best_ensemble_oof:.4f}")
print(f"Gap: {best_ensemble_gap:.4f}")

# ========================================================================
# 総合比較
# ========================================================================
print("\n" + "="*80)
print("総合比較")
print("="*80)

comparison = pd.DataFrame({
    'モデル': [
        best_lgb['name'] + ' (LightGBM)',
        best_gb['name'] + ' (GB)',
        best_rf['name'] + ' (RF)',
        'Hard Voting',
        'Soft Voting'
    ],
    'CV/OOF Score': [
        best_lgb['cv_mean'],
        best_gb['cv_mean'],
        best_rf['cv_mean'],
        hard_oof_score,
        soft_oof_score
    ],
    'Train Score': [
        best_lgb['train_score'],
        best_gb['train_score'],
        best_rf['train_score'],
        hard_train_score,
        soft_train_score
    ],
    'Gap': [
        best_lgb['gap'],
        best_gb['gap'],
        best_rf['gap'],
        hard_gap,
        soft_gap
    ],
    'Gap達成': [
        '✓' if best_lgb['gap'] < 0.015 else '✗',
        '✓' if best_gb['gap'] < 0.015 else '✗',
        '✓' if best_rf['gap'] < 0.015 else '✗',
        '✓' if hard_gap < 0.015 else '✗',
        '✓' if soft_gap < 0.015 else '✗'
    ]
})

print("\n")
print(comparison.to_string(index=False))

# 最高スコア
best_score_idx = comparison['CV/OOF Score'].idxmax()
print(f"\n【最高スコア】: {comparison.iloc[best_score_idx]['モデル']}")
print(f"Score: {comparison.iloc[best_score_idx]['CV/OOF Score']:.4f}")

# ========================================================================
# OOF詳細評価
# ========================================================================
print("\n" + "="*80)
print("OOF詳細評価（最良アンサンブル）")
print("="*80)

# 最良アンサンブルのOOF予測を使用
if best_ensemble_name == "Hard Voting":
    oof_pred = hard_oof_pred
else:
    oof_pred = soft_oof_pred

# Confusion Matrix
print("\n【Confusion Matrix】")
cm = confusion_matrix(y_train, oof_pred)
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# Classification Report
print("\n【Classification Report】")
print(classification_report(y_train, oof_pred, target_names=['Died', 'Survived']))

# ========================================================================
# テストデータ予測とSubmission作成
# ========================================================================
print("\n" + "="*80)
print("テストデータ予測とSubmission作成")
print("="*80)

# 個別モデルのSubmission
print("\n【個別モデルのSubmission作成】")

# LightGBM
lgb_pred = best_lgb['model'].predict(X_test)
lgb_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': lgb_pred
})
lgb_file = f'submission_top3_lgb_{best_lgb["name"].replace(" ", "_").lower()}_10features.csv'
lgb_submission.to_csv(lgb_file, index=False)
print(f"✓ LightGBM: {lgb_file}")

# GradientBoosting
gb_pred = best_gb['model'].predict(X_test)
gb_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': gb_pred
})
gb_file = f'submission_top3_gb_{best_gb["name"].replace(" ", "_").lower()}_10features.csv'
gb_submission.to_csv(gb_file, index=False)
print(f"✓ GradientBoosting: {gb_file}")

# RandomForest
rf_pred = best_rf['model'].predict(X_test)
rf_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': rf_pred
})
rf_file = f'submission_top3_rf_{best_rf["name"].replace(" ", "_").lower()}_10features.csv'
rf_submission.to_csv(rf_file, index=False)
print(f"✓ RandomForest: {rf_file}")

# Hard Voting Ensemble
hard_pred = hard_voting.predict(X_test)
hard_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': hard_pred
})
hard_file = 'submission_top3_hardvoting_10features.csv'
hard_submission.to_csv(hard_file, index=False)
print(f"✓ Hard Voting: {hard_file}")

# Soft Voting Ensemble
soft_pred = soft_voting.predict(X_test)
soft_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': soft_pred
})
soft_file = 'submission_top3_softvoting_10features.csv'
soft_submission.to_csv(soft_file, index=False)
print(f"✓ Soft Voting: {soft_file}")

# ========================================================================
# まとめ
# ========================================================================
print("\n" + "="*80)
print("まとめ")
print("="*80)

print(f"""
【使用特徴量】
- 10特徴量（Deck_Safe追加）: {features}

【個別モデル最適化結果】
1. LightGBM ({best_lgb['name']}):
   CV Score: {best_lgb['cv_mean']:.4f}
   Gap: {best_lgb['gap']:.4f} {'✓ 目標達成' if best_lgb['gap'] < 0.015 else '✗ 目標未達'}

2. GradientBoosting ({best_gb['name']}):
   CV Score: {best_gb['cv_mean']:.4f}
   Gap: {best_gb['gap']:.4f} {'✓ 目標達成' if best_gb['gap'] < 0.015 else '✗ 目標未達'}

3. RandomForest ({best_rf['name']}):
   CV Score: {best_rf['cv_mean']:.4f}
   Gap: {best_rf['gap']:.4f} {'✓ 目標達成' if best_rf['gap'] < 0.015 else '✗ 目標未達'}

【アンサンブル結果】
- Hard Voting OOF: {hard_oof_score:.4f} (Gap: {hard_gap:.4f})
- Soft Voting OOF: {soft_oof_score:.4f} (Gap: {soft_gap:.4f})

【最良モデル】: {best_ensemble_name}
- OOF Score: {best_ensemble_oof:.4f}
- Gap: {best_ensemble_gap:.4f}

【目標達成状況】
- Gap < 0.015: {'✓ 達成' if best_ensemble_gap < 0.015 else '✗ 未達成'}
- OOF Score 0.835+: {'✓ 達成' if best_ensemble_oof >= 0.835 else '✗ 未達成'}

【作成されたSubmission】
- {lgb_file}
- {gb_file}
- {rf_file}
- {hard_file}
- {soft_file}
""")

print("="*80)
print("完了")
print("="*80)
