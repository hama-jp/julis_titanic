"""
Optunaによるハイパーパラメータチューニング

対象モデル:
1. LightGBM
2. GradientBoosting
3. RandomForest

評価方法:
- Stratified K-Fold CV (5-fold)
- 評価指標: OOF Accuracy
- 探索回数: 各モデル50 trials

目的:
- 現在のパラメータより良いものを見つける
- 10シードアンサンブルで使用する最適パラメータを決定
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("Optunaによるハイパーパラメータチューニング")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

# 正しい補完パイプライン
imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

def engineer_features(df):
    """特徴量エンジニアリング"""
    df_new = df.copy()
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})

    df_new['Title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Mme': 2,
        'Don': 4, 'Dona': 4, 'Lady': 4, 'Countess': 4, 'Jonkheer': 4, 'Sir': 4,
        'Capt': 4, 'Ms': 1
    }
    df_new['Title'] = df_new['Title'].map(title_mapping).fillna(4)

    df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
    df_new['SmallFamily'] = ((df_new['FamilySize'] >= 2) & (df_new['FamilySize'] <= 4)).astype(int)

    bins = [0, 18, 60, 100]
    labels = [0, 1, 2]
    df_new['AgeGroup'] = pd.cut(df_new['Age'], bins=bins, labels=labels, include_lowest=True)
    df_new['AgeGroup'] = df_new['AgeGroup'].astype(int)

    return df_new

train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize', 'SmallFamily'] + embarked_cols

y_train = train['Survived']

# 現在のベースラインパラメータ
baseline_params = {
    'LightGBM': {
        'n_estimators': 100,
        'max_depth': 2,
        'num_leaves': 4,
        'min_child_samples': 30,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'min_split_gain': 0.01,
        'learning_rate': 0.05
    },
    'GradientBoosting': {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.05
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5
    }
}

# ベースラインOOFスコアを計算
def calculate_baseline_oof(model_class, params, features):
    """ベースラインのOOFスコアを計算"""
    X_train = train[features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = model_class(**params, random_state=42)

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        val_pred = model.predict(X_fold_val)
        oof_pred[val_idx] = val_pred

    return accuracy_score(y_train, oof_pred)

print("\n" + "=" * 80)
print("ベースラインOOFスコア")
print("=" * 80)

baseline_scores = {}

print("\nLightGBM (AgeGroup使用):")
lgb_features = base_features + ['AgeGroup']
baseline_scores['LightGBM'] = calculate_baseline_oof(
    lgb.LGBMClassifier,
    {**baseline_params['LightGBM'], 'verbose': -1},
    lgb_features
)
print(f"  OOF Score: {baseline_scores['LightGBM']:.4f}")

print("\nGradientBoosting (AgeGroup使用):")
gb_features = base_features + ['AgeGroup']
baseline_scores['GradientBoosting'] = calculate_baseline_oof(
    GradientBoostingClassifier,
    baseline_params['GradientBoosting'],
    gb_features
)
print(f"  OOF Score: {baseline_scores['GradientBoosting']:.4f}")

print("\nRandomForest (Age使用):")
rf_features = base_features + ['Age']
baseline_scores['RandomForest'] = calculate_baseline_oof(
    RandomForestClassifier,
    baseline_params['RandomForest'],
    rf_features
)
print(f"  OOF Score: {baseline_scores['RandomForest']:.4f}")

# Optunaチューニング
optuna.logging.set_verbosity(optuna.logging.WARNING)

results = {}

# 1. LightGBM
print("\n" + "=" * 80)
print("LightGBM ハイパーパラメータチューニング")
print("=" * 80)

def objective_lgb(trial):
    """LightGBM目的関数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'num_leaves': trial.suggest_int('num_leaves', 4, 31),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }

    X_train = train[lgb_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = lgb.LGBMClassifier(**params)

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        val_pred = model.predict(X_fold_val)
        oof_pred[val_idx] = val_pred

    return accuracy_score(y_train, oof_pred)

study_lgb = optuna.create_study(direction='maximize', study_name='LightGBM')
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)

results['LightGBM'] = {
    'best_params': study_lgb.best_params,
    'best_score': study_lgb.best_value,
    'baseline_score': baseline_scores['LightGBM'],
    'improvement': study_lgb.best_value - baseline_scores['LightGBM']
}

print(f"\nベストスコア: {study_lgb.best_value:.4f}")
print(f"ベースライン: {baseline_scores['LightGBM']:.4f}")
print(f"改善: {results['LightGBM']['improvement']:+.4f}")
print(f"\nベストパラメータ:")
for key, value in study_lgb.best_params.items():
    print(f"  {key}: {value}")

# 2. GradientBoosting
print("\n" + "=" * 80)
print("GradientBoosting ハイパーパラメータチューニング")
print("=" * 80)

def objective_gb(trial):
    """GradientBoosting目的関数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    X_train = train[gb_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = GradientBoostingClassifier(**params)

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        val_pred = model.predict(X_fold_val)
        oof_pred[val_idx] = val_pred

    return accuracy_score(y_train, oof_pred)

study_gb = optuna.create_study(direction='maximize', study_name='GradientBoosting')
study_gb.optimize(objective_gb, n_trials=50, show_progress_bar=True)

results['GradientBoosting'] = {
    'best_params': study_gb.best_params,
    'best_score': study_gb.best_value,
    'baseline_score': baseline_scores['GradientBoosting'],
    'improvement': study_gb.best_value - baseline_scores['GradientBoosting']
}

print(f"\nベストスコア: {study_gb.best_value:.4f}")
print(f"ベースライン: {baseline_scores['GradientBoosting']:.4f}")
print(f"改善: {results['GradientBoosting']['improvement']:+.4f}")
print(f"\nベストパラメータ:")
for key, value in study_gb.best_params.items():
    print(f"  {key}: {value}")

# 3. RandomForest
print("\n" + "=" * 80)
print("RandomForest ハイパーパラメータチューニング")
print("=" * 80)

def objective_rf(trial):
    """RandomForest目的関数"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }

    X_train = train[rf_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = RandomForestClassifier(**params)

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        model.fit(X_fold_train, y_fold_train)
        val_pred = model.predict(X_fold_val)
        oof_pred[val_idx] = val_pred

    return accuracy_score(y_train, oof_pred)

study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=True)

results['RandomForest'] = {
    'best_params': study_rf.best_params,
    'best_score': study_rf.best_value,
    'baseline_score': baseline_scores['RandomForest'],
    'improvement': study_rf.best_value - baseline_scores['RandomForest']
}

print(f"\nベストスコア: {study_rf.best_value:.4f}")
print(f"ベースライン: {baseline_scores['RandomForest']:.4f}")
print(f"改善: {results['RandomForest']['improvement']:+.4f}")
print(f"\nベストパラメータ:")
for key, value in study_rf.best_params.items():
    print(f"  {key}: {value}")

# 結果サマリー
print("\n" + "=" * 80)
print("チューニング結果サマリー")
print("=" * 80)

results_df = pd.DataFrame([
    {
        'Model': name,
        'Baseline OOF': res['baseline_score'],
        'Best OOF': res['best_score'],
        'Improvement': res['improvement']
    }
    for name, res in results.items()
])

print(f"\n{results_df.to_string(index=False)}")

total_improvement = sum([res['improvement'] for res in results.values()])
print(f"\n合計改善: {total_improvement:+.4f}")

# 最適パラメータを保存
import json

output_file = '/home/user/julis_titanic/experiments/results/optuna_best_params.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n最適パラメータを保存: {output_file}")

print("\n" + "=" * 80)
print("次のステップ")
print("=" * 80)

print("\n最適パラメータで10シードアンサンブルを再実装してください:")
print("  → scripts/models/final_ensemble_10seeds_optimized.py")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
