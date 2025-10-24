"""
最終アンサンブル: モデル別最適特徴量 + Optuna + 10シード

ステップ:
1. 各モデルに最適な特徴量セットを特定
2. Optunaでハイパーパラメータ最適化
3. 最適パラメータで10シード訓練
4. アンサンブル予測
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("最終アンサンブル: モデル別最適特徴量 + Optuna + 10シード")
print("=" * 80)

# データ読み込みと前処理
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')
test_ids = test['PassengerId'].copy()

imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

def extract_ticket_prefix(ticket):
    ticket = str(ticket).strip()
    if ticket.replace('.', '').isdigit():
        return 'NUMERIC'
    match = re.match(r'^([A-Za-z./\s]+)', ticket)
    if match:
        prefix = match.group(1).strip()
        prefix = re.sub(r'[./\s]+', '_', prefix)
        prefix = prefix.strip('_')
        return prefix.upper()
    return 'OTHER'

def add_ticket_features(df_train, df_test):
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    combined = pd.concat([df_train_copy, df_test_copy], axis=0, ignore_index=True)

    combined['Ticket_Prefix'] = combined['Ticket'].apply(extract_ticket_prefix)
    ticket_counts = combined.groupby('Ticket').size()
    combined['Ticket_GroupSize'] = combined['Ticket'].map(ticket_counts)
    combined['Ticket_Fare_Per_Person'] = combined['Fare'] / combined['Ticket_GroupSize']

    high_survival_prefixes = ['F_C_C', 'PC']
    low_survival_prefixes = ['SOTON_OQ', 'SOTON_O_Q', 'W_C', 'A', 'CA', 'S_O_C']

    def categorize_prefix(prefix):
        if prefix in high_survival_prefixes:
            return 2
        elif prefix in low_survival_prefixes:
            return 0
        else:
            return 1

    combined['Ticket_Prefix_Category'] = combined['Ticket_Prefix'].apply(categorize_prefix)
    combined['Ticket_Is_Numeric'] = (combined['Ticket_Prefix'] == 'NUMERIC').astype(int)
    combined['Ticket_SmallGroup'] = ((combined['Ticket_GroupSize'] >= 2) &
                                      (combined['Ticket_GroupSize'] <= 4)).astype(int)

    train_len = len(df_train_copy)
    df_train_new = combined.iloc[:train_len].copy()
    df_test_new = combined.iloc[train_len:].copy()

    return df_train_new, df_test_new

train, test = add_ticket_features(train, test)

def engineer_features(df):
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
    return df_new

train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ===== 1. モデル別最適特徴量セット =====
print("\n" + "=" * 80)
print("1. モデル別最適特徴量セット（分析結果から）")
print("=" * 80)

base_core = ['Pclass', 'Sex', 'Title'] + embarked_cols

MODEL_FEATURE_CONFIGS = {
    'GradientBoosting': {
        'features': base_core + ['FamilySize', 'SmallFamily', 'Ticket_GroupSize',
                                 'Ticket_Fare_Per_Person', 'Ticket_Is_Numeric',
                                 'Ticket_Prefix_Category'],
        'reason': '分析結果: FamilySize+SmallFamily+Ticket_GroupSize が最良（0.8451）'
    },
    'RandomForest': {
        'features': base_core + ['Ticket_GroupSize', 'Ticket_Fare_Per_Person',
                                 'Ticket_Is_Numeric', 'Ticket_Prefix_Category'],
        'reason': '分析結果: Ticket_GroupSize中心が最良（0.8339）、家族特徴量は不要'
    },
    'LightGBM': {
        'features': base_core + ['Ticket_GroupSize', 'Ticket_SmallGroup',
                                 'Ticket_Fare_Per_Person', 'Ticket_Is_Numeric',
                                 'Ticket_Prefix_Category'],
        'reason': '分析結果: Ticket_GroupSize+Ticket_SmallGroup が最良（0.8328）'
    },
    'LogisticRegression': {
        'features': base_core + ['SibSp', 'Parch', 'FamilySize', 'SmallFamily', 'Fare'],
        'reason': '分析結果: Ticket特徴量なし（多重共線性により悪化）'
    }
}

for model_name, config in MODEL_FEATURE_CONFIGS.items():
    print(f"\n{model_name}:")
    print(f"  特徴量数: {len(config['features'])}")
    print(f"  理由: {config['reason']}")
    print(f"  特徴量: {config['features']}")

y = train['Survived']

# ===== 2. Optuna ハイパーパラメータ最適化 =====
print("\n" + "=" * 80)
print("2. Optuna ハイパーパラメータ最適化")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }

    X = train[MODEL_FEATURE_CONFIGS['GradientBoosting']['features']]
    model = GradientBoostingClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    X = train[MODEL_FEATURE_CONFIGS['RandomForest']['features']]
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42,
        'verbose': -1
    }

    X = train[MODEL_FEATURE_CONFIGS['LightGBM']['features']]
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def objective_lr(trial):
    params = {
        'C': trial.suggest_float('C', 0.001, 100, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'saga',
        'max_iter': 1000,
        'random_state': 42
    }

    X = train[MODEL_FEATURE_CONFIGS['LogisticRegression']['features']]
    model = LogisticRegression(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)

best_params = {}

print("\nGradientBoosting最適化...")
study_gb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_gb.optimize(objective_gb, n_trials=100, show_progress_bar=True)
best_params['GradientBoosting'] = study_gb.best_params
print(f"  Best score: {study_gb.best_value:.4f}")
print(f"  Best params: {study_gb.best_params}")

print("\nRandomForest最適化...")
study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=100, show_progress_bar=True)
best_params['RandomForest'] = study_rf.best_params
print(f"  Best score: {study_rf.best_value:.4f}")
print(f"  Best params: {study_rf.best_params}")

print("\nLightGBM最適化...")
study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=100, show_progress_bar=True)
best_params['LightGBM'] = study_lgb.best_params
print(f"  Best score: {study_lgb.best_value:.4f}")
print(f"  Best params: {study_lgb.best_params}")

print("\nLogisticRegression最適化...")
study_lr = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lr.optimize(objective_lr, n_trials=50, show_progress_bar=True)
# solver and max_iter are fixed in objective, so add them to best_params
best_params['LogisticRegression'] = {**study_lr.best_params, 'solver': 'saga', 'max_iter': 1000}
print(f"  Best score: {study_lr.best_value:.4f}")
print(f"  Best params: {best_params['LogisticRegression']}")

# パラメータ保存
import json
optuna_results = {
    'GradientBoosting': {
        'best_params': best_params['GradientBoosting'],
        'best_score': float(study_gb.best_value),
        'features': MODEL_FEATURE_CONFIGS['GradientBoosting']['features']
    },
    'RandomForest': {
        'best_params': best_params['RandomForest'],
        'best_score': float(study_rf.best_value),
        'features': MODEL_FEATURE_CONFIGS['RandomForest']['features']
    },
    'LightGBM': {
        'best_params': best_params['LightGBM'],
        'best_score': float(study_lgb.best_value),
        'features': MODEL_FEATURE_CONFIGS['LightGBM']['features']
    },
    'LogisticRegression': {
        'best_params': best_params['LogisticRegression'],
        'best_score': float(study_lr.best_value),
        'features': MODEL_FEATURE_CONFIGS['LogisticRegression']['features']
    }
}

with open('/home/user/julis_titanic/experiments/results/final_optuna_params.json', 'w') as f:
    json.dump(optuna_results, f, indent=2)

print("\nOptuna最適化完了！パラメータ保存: experiments/results/final_optuna_params.json")

# ===== 3. 10シードアンサンブル =====
print("\n" + "=" * 80)
print("3. 10シードアンサンブル訓練")
print("=" * 80)

SEEDS = [42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]

model_predictions = {
    'GradientBoosting': [],
    'RandomForest': [],
    'LightGBM': [],
    'LogisticRegression': []
}

oof_scores = {model: [] for model in model_predictions.keys()}

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n[Seed {seed_idx}/10] Random seed: {seed}")

    # GradientBoosting
    X_gb = train[MODEL_FEATURE_CONFIGS['GradientBoosting']['features']]
    X_gb_test = test[MODEL_FEATURE_CONFIGS['GradientBoosting']['features']]
    params_gb = {**best_params['GradientBoosting'], 'random_state': seed}
    model_gb = GradientBoostingClassifier(**params_gb)

    cv_scores = cross_val_score(model_gb, X_gb, y, cv=cv, scoring='accuracy')
    oof_scores['GradientBoosting'].append(cv_scores.mean())

    model_gb.fit(X_gb, y)
    pred_gb = model_gb.predict_proba(X_gb_test)[:, 1]
    model_predictions['GradientBoosting'].append(pred_gb)
    print(f"  GB   OOF: {cv_scores.mean():.4f}")

    # RandomForest
    X_rf = train[MODEL_FEATURE_CONFIGS['RandomForest']['features']]
    X_rf_test = test[MODEL_FEATURE_CONFIGS['RandomForest']['features']]
    params_rf = {**best_params['RandomForest'], 'random_state': seed}
    model_rf = RandomForestClassifier(**params_rf)

    cv_scores = cross_val_score(model_rf, X_rf, y, cv=cv, scoring='accuracy')
    oof_scores['RandomForest'].append(cv_scores.mean())

    model_rf.fit(X_rf, y)
    pred_rf = model_rf.predict_proba(X_rf_test)[:, 1]
    model_predictions['RandomForest'].append(pred_rf)
    print(f"  RF   OOF: {cv_scores.mean():.4f}")

    # LightGBM
    X_lgb = train[MODEL_FEATURE_CONFIGS['LightGBM']['features']]
    X_lgb_test = test[MODEL_FEATURE_CONFIGS['LightGBM']['features']]
    params_lgb = {**best_params['LightGBM'], 'random_state': seed}
    model_lgb = lgb.LGBMClassifier(**params_lgb)

    cv_scores = cross_val_score(model_lgb, X_lgb, y, cv=cv, scoring='accuracy')
    oof_scores['LightGBM'].append(cv_scores.mean())

    model_lgb.fit(X_lgb, y)
    pred_lgb = model_lgb.predict_proba(X_lgb_test)[:, 1]
    model_predictions['LightGBM'].append(pred_lgb)
    print(f"  LGB  OOF: {cv_scores.mean():.4f}")

    # LogisticRegression
    X_lr = train[MODEL_FEATURE_CONFIGS['LogisticRegression']['features']]
    X_lr_test = test[MODEL_FEATURE_CONFIGS['LogisticRegression']['features']]
    params_lr = {**best_params['LogisticRegression'], 'random_state': seed}
    model_lr = LogisticRegression(**params_lr)

    cv_scores = cross_val_score(model_lr, X_lr, y, cv=cv, scoring='accuracy')
    oof_scores['LogisticRegression'].append(cv_scores.mean())

    model_lr.fit(X_lr, y)
    pred_lr = model_lr.predict_proba(X_lr_test)[:, 1]
    model_predictions['LogisticRegression'].append(pred_lr)
    print(f"  LR   OOF: {cv_scores.mean():.4f}")

# ===== 4. 予測の平均化 =====
print("\n" + "=" * 80)
print("4. 予測の平均化とアンサンブル")
print("=" * 80)

# 各モデルの10シード平均
averaged_predictions = {}
for model_name in model_predictions.keys():
    averaged_predictions[model_name] = np.mean(model_predictions[model_name], axis=0)
    avg_oof = np.mean(oof_scores[model_name])
    std_oof = np.std(oof_scores[model_name])
    print(f"\n{model_name}:")
    print(f"  10シード平均 OOF: {avg_oof:.4f} (± {std_oof:.4f})")

# モデル間アンサンブル
print("\n" + "=" * 80)
print("5. 最終アンサンブル")
print("=" * 80)

# Equal weight ensemble
ensemble_equal = np.mean([
    averaged_predictions['GradientBoosting'],
    averaged_predictions['RandomForest'],
    averaged_predictions['LightGBM'],
    averaged_predictions['LogisticRegression']
], axis=0)

# Weighted by OOF score
weights = {}
for model_name in model_predictions.keys():
    weights[model_name] = np.mean(oof_scores[model_name])

total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

print("\nモデル重み（OOFスコアベース）:")
for model_name, weight in weights.items():
    print(f"  {model_name}: {weight:.4f}")

ensemble_weighted = (
    averaged_predictions['GradientBoosting'] * weights['GradientBoosting'] +
    averaged_predictions['RandomForest'] * weights['RandomForest'] +
    averaged_predictions['LightGBM'] * weights['LightGBM'] +
    averaged_predictions['LogisticRegression'] * weights['LogisticRegression']
)

# Top3 ensemble (GB + RF + LGB)
ensemble_top3 = np.mean([
    averaged_predictions['GradientBoosting'],
    averaged_predictions['RandomForest'],
    averaged_predictions['LightGBM']
], axis=0)

# ===== 6. Submission作成 =====
print("\n" + "=" * 80)
print("6. Submission作成")
print("=" * 80)

submissions = {
    'gb_10seeds': (averaged_predictions['GradientBoosting'] > 0.5).astype(int),
    'rf_10seeds': (averaged_predictions['RandomForest'] > 0.5).astype(int),
    'lgb_10seeds': (averaged_predictions['LightGBM'] > 0.5).astype(int),
    'lr_10seeds': (averaged_predictions['LogisticRegression'] > 0.5).astype(int),
    'ensemble_equal_4models': (ensemble_equal > 0.5).astype(int),
    'ensemble_weighted_4models': (ensemble_weighted > 0.5).astype(int),
    'ensemble_top3_models': (ensemble_top3 > 0.5).astype(int)
}

import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')

for name, predictions in submissions.items():
    submission = pd.DataFrame({
        'PassengerId': test_ids,
        'Survived': predictions
    })
    filename = f'/home/user/julis_titanic/submissions/{timestamp}_{name}.csv'
    submission.to_csv(filename, index=False)
    print(f"  ✓ {filename}")

# ===== 7. サマリー =====
print("\n" + "=" * 80)
print("7. 最終サマリー")
print("=" * 80)

print("\n【モデル別OOFスコア（10シード平均）】")
for model_name in model_predictions.keys():
    avg_oof = np.mean(oof_scores[model_name])
    std_oof = np.std(oof_scores[model_name])
    min_oof = np.min(oof_scores[model_name])
    max_oof = np.max(oof_scores[model_name])
    print(f"{model_name:20s}: {avg_oof:.4f} (± {std_oof:.4f}) [{min_oof:.4f} - {max_oof:.4f}]")

print("\n【推奨Submission】")
best_model = max(weights.keys(), key=lambda k: weights[k])
print(f"1. 単体モデル: {timestamp}_{best_model.lower()}_10seeds.csv")
print(f"   期待精度: {np.mean(oof_scores[best_model]):.4f}")
print(f"\n2. アンサンブル: {timestamp}_ensemble_weighted_4models.csv")
print(f"   期待精度: 単体より高い可能性")
print(f"\n3. Top3アンサンブル: {timestamp}_ensemble_top3_models.csv")
print(f"   期待精度: 高い安定性")

print("\n" + "=" * 80)
print("完了！")
print("=" * 80)
