"""
Gap Reduction Tuning - 過学習削減に特化したハイパーパラメータチューニング

目的: CV ScoreとTrain Scoreの差（Gap）を最小化
現状:
  - LightGBM: Gap 0.0090
  - RandomForest: Gap 0.0112
  - GradientBoosting: Gap 0.0314
目標: Gap < 0.005 を目指す

戦略:
  1. 正則化の強化（reg_alpha, reg_lambda, C値）
  2. モデルの複雑度削減（max_depth, num_leaves）
  3. サンプルサイズの増加（min_samples_leaf, min_child_samples）
  4. Early Stopping
  5. Dropout/Subsample調整
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('GAP REDUCTION TUNING - 過学習削減特化型チューニング')
print('='*80)
print('目標: Gap < 0.005 を達成')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df):
    """9特徴量の基本セット（Deck_Safe除外）"""
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

# 評価関数
def evaluate_gap(model, X, y, cv, model_name):
    """Gapを重視した評価"""
    # CV score
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Train score
    model.fit(X, y)
    train_score = model.score(X, y)

    # Gap計算
    gap = train_score - cv_mean

    return {
        'model': model_name,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_score': train_score,
        'gap': gap,
        'gap_percentage': gap / cv_mean * 100
    }

print('\n' + '='*80)
print('現状のGap評価（ベースライン）')
print('='*80)

baseline_models = {
    'RF_Current': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    ),
    'GB_Current': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    'LGB_Current': lgb.LGBMClassifier(
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
    )
}

baseline_results = []
for name, model in baseline_models.items():
    result = evaluate_gap(model, X_train, y_train, cv, name)
    baseline_results.append(result)
    print(f"\n{name}:")
    print(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
    print(f"  Train Score: {result['train_score']:.4f}")
    print(f"  Gap: {result['gap']:.4f} ({result['gap_percentage']:.2f}%)")

# ========================================================================
# Gap削減チューニング
# ========================================================================

print('\n' + '='*80)
print('Gap削減チューニング開始')
print('='*80)

tuned_results = []

# ========================================================================
# 1. Random Forest - 超保守的設定
# ========================================================================
print('\n【1. Random Forest - 超保守的設定】')
print('戦略: max_depth削減、min_samples_leaf増加')

rf_configs = [
    {
        'name': 'RF_UltraConservative_D3',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,  # 4 → 3
            'min_samples_split': 30,  # 20 → 30
            'min_samples_leaf': 15,  # 10 → 15
            'max_features': 'sqrt',
            'random_state': 42
        }
    },
    {
        'name': 'RF_UltraConservative_D2',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,  # さらに削減
            'min_samples_split': 40,
            'min_samples_leaf': 20,
            'max_features': 'sqrt',
            'random_state': 42
        }
    },
    {
        'name': 'RF_MinComplexity',
        'params': {
            'n_estimators': 50,  # 木の数も削減
            'max_depth': 3,
            'min_samples_split': 25,
            'min_samples_leaf': 12,
            'max_features': 'sqrt',
            'min_impurity_decrease': 0.001,  # 分岐の最小改善
            'random_state': 42
        }
    }
]

for config in rf_configs:
    model = RandomForestClassifier(**config['params'])
    result = evaluate_gap(model, X_train, y_train, cv, config['name'])
    tuned_results.append(result)
    print(f"\n{config['name']}:")
    print(f"  CV: {result['cv_mean']:.4f}, Gap: {result['gap']:.4f} ({result['gap_percentage']:.2f}%)")

# ========================================================================
# 2. Gradient Boosting - 強力な正則化
# ========================================================================
print('\n【2. Gradient Boosting - 強力な正則化】')
print('戦略: learning_rate削減、max_depth削減、subsample削減')

gb_configs = [
    {
        'name': 'GB_StrongReg_V1',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,  # 3 → 2
            'learning_rate': 0.03,  # 0.05 → 0.03
            'min_samples_split': 30,  # 20 → 30
            'min_samples_leaf': 15,  # 10 → 15
            'subsample': 0.7,  # 0.8 → 0.7
            'max_features': 'sqrt',
            'random_state': 42
        }
    },
    {
        'name': 'GB_StrongReg_V2',
        'params': {
            'n_estimators': 150,  # 数を増やして学習率を下げる
            'max_depth': 2,
            'learning_rate': 0.02,  # さらに削減
            'min_samples_split': 35,
            'min_samples_leaf': 20,
            'subsample': 0.6,  # さらに削減
            'max_features': 0.7,
            'random_state': 42
        }
    },
    {
        'name': 'GB_MinDepth',
        'params': {
            'n_estimators': 100,
            'max_depth': 1,  # 最小深度
            'learning_rate': 0.05,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'random_state': 42
        }
    }
]

for config in gb_configs:
    model = GradientBoostingClassifier(**config['params'])
    result = evaluate_gap(model, X_train, y_train, cv, config['name'])
    tuned_results.append(result)
    print(f"\n{config['name']}:")
    print(f"  CV: {result['cv_mean']:.4f}, Gap: {result['gap']:.4f} ({result['gap_percentage']:.2f}%)")

# ========================================================================
# 3. LightGBM - 最強正則化 + Dropout
# ========================================================================
print('\n【3. LightGBM - 最強正則化 + Dropout】')
print('戦略: reg_alpha/lambda増加、dropout追加、min_child_samples増加')

lgb_configs = [
    {
        'name': 'LGB_StrongReg_V1',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 40,  # 30 → 40
            'reg_alpha': 2.0,  # 1.0 → 2.0
            'reg_lambda': 2.0,  # 1.0 → 2.0
            'min_split_gain': 0.02,  # 0.01 → 0.02
            'learning_rate': 0.05,
            'subsample': 0.7,  # 0.8 → 0.7
            'colsample_bytree': 0.7,  # 0.8 → 0.7
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'LGB_StrongReg_V2_Dropout',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 50,  # さらに増加
            'reg_alpha': 3.0,  # さらに増加
            'reg_lambda': 3.0,
            'min_split_gain': 0.03,
            'learning_rate': 0.05,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'drop_rate': 0.1,  # Dropout追加
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'LGB_MinLeaves',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 3,  # 4 → 3（最小化）
            'min_child_samples': 35,
            'reg_alpha': 1.5,
            'reg_lambda': 1.5,
            'min_split_gain': 0.015,
            'learning_rate': 0.05,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'random_state': 42,
            'verbose': -1
        }
    }
]

for config in lgb_configs:
    model = lgb.LGBMClassifier(**config['params'])
    result = evaluate_gap(model, X_train, y_train, cv, config['name'])
    tuned_results.append(result)
    print(f"\n{config['name']}:")
    print(f"  CV: {result['cv_mean']:.4f}, Gap: {result['gap']:.4f} ({result['gap_percentage']:.2f}%)")

# ========================================================================
# 4. Logistic Regression - 最強正則化
# ========================================================================
print('\n【4. Logistic Regression - 最強正則化】')
print('戦略: C値削減（正則化強化）')

lr_configs = [
    {
        'name': 'LR_StrongReg_C01',
        'params': {
            'C': 0.1,  # 0.5 → 0.1
            'penalty': 'l2',
            'max_iter': 1000,
            'random_state': 42
        }
    },
    {
        'name': 'LR_StrongReg_C005',
        'params': {
            'C': 0.05,  # さらに削減
            'penalty': 'l2',
            'max_iter': 1000,
            'random_state': 42
        }
    },
    {
        'name': 'LR_ElasticNet',
        'params': {
            'C': 0.1,
            'penalty': 'elasticnet',
            'solver': 'saga',
            'l1_ratio': 0.5,  # L1とL2の混合
            'max_iter': 2000,
            'random_state': 42
        }
    }
]

for config in lr_configs:
    model = LogisticRegression(**config['params'])
    result = evaluate_gap(model, X_train, y_train, cv, config['name'])
    tuned_results.append(result)
    print(f"\n{config['name']}:")
    print(f"  CV: {result['cv_mean']:.4f}, Gap: {result['gap']:.4f} ({result['gap_percentage']:.2f}%)")

# ========================================================================
# 結果のまとめ
# ========================================================================

print('\n' + '='*80)
print('結果サマリー')
print('='*80)

# 全結果を結合
all_results = baseline_results + tuned_results
results_df = pd.DataFrame(all_results)

# Gapでソート
results_df_sorted = results_df.sort_values('gap')

print('\n【Gap順にランキング（上位10）】')
print(results_df_sorted[['model', 'cv_mean', 'train_score', 'gap', 'gap_percentage']].head(10).to_string(index=False))

print('\n【Gap < 0.005を達成したモデル】')
gap_achieved = results_df_sorted[results_df_sorted['gap'] < 0.005]
if len(gap_achieved) > 0:
    print(gap_achieved[['model', 'cv_mean', 'gap', 'gap_percentage']].to_string(index=False))
else:
    print('該当なし（目標未達成）')
    print(f'最小Gap: {results_df_sorted.iloc[0]["gap"]:.4f} ({results_df_sorted.iloc[0]["model"]})')

print('\n【CV Scoreを維持しつつGapを削減したモデル（CV > 0.82）】')
best_balance = results_df_sorted[(results_df_sorted['cv_mean'] > 0.82) & (results_df_sorted['gap'] < 0.01)]
if len(best_balance) > 0:
    print(best_balance[['model', 'cv_mean', 'gap']].to_string(index=False))
else:
    print('該当なし')

# ベストモデル選定
print('\n' + '='*80)
print('推奨モデル選定')
print('='*80)

best_gap_model = results_df_sorted.iloc[0]
print(f"\n🏆 最小Gap達成モデル: {best_gap_model['model']}")
print(f"   CV Score: {best_gap_model['cv_mean']:.4f} (±{best_gap_model['cv_std']:.4f})")
print(f"   Train Score: {best_gap_model['train_score']:.4f}")
print(f"   Gap: {best_gap_model['gap']:.4f} ({best_gap_model['gap_percentage']:.2f}%)")

# CV Scoreも考慮したバランス型
results_df_sorted['score'] = results_df_sorted['cv_mean'] - results_df_sorted['gap'] * 2  # Gapにペナルティ
best_balanced = results_df_sorted.sort_values('score', ascending=False).iloc[0]

print(f"\n⭐ バランス型ベストモデル: {best_balanced['model']}")
print(f"   CV Score: {best_balanced['cv_mean']:.4f}")
print(f"   Gap: {best_balanced['gap']:.4f}")
print(f"   総合スコア: {best_balanced['score']:.4f}")

# 結果をCSVに保存
results_df_sorted.to_csv('experiments/results/gap_reduction_tuning_results.csv', index=False)
print(f"\n✅ 結果を保存: experiments/results/gap_reduction_tuning_results.csv")

print('\n' + '='*80)
print('GAP REDUCTION TUNING完了！')
print('='*80)
