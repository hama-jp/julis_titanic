"""
最終アンサンブル: Optuna最適化 + 10シード平均化

実装:
- 4モデル（LightGBM + GradientBoosting + RandomForest + SVM_RBF）
- Optuna最適パラメータ適用（LightGBM, GradientBoosting, RandomForest）
- 各モデルを10個の異なる乱数シードで学習
- 全データ（100%）で再学習
- Test予測を平均（バギング効果）

期待効果:
- Optunaチューニング: +0.0640 OOF改善
- 10シード平均: 予測の安定性向上
- 分散（variance）の減少
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("最終アンサンブル: Optuna最適化 + 4モデル × 10シード = 40モデル平均")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

# 正しい補完パイプライン
print("\n正しい補完パイプラインで前処理...")
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

print("特徴量エンジニアリング...")
train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize', 'SmallFamily'] + embarked_cols

# Optuna最適パラメータ読み込み
print("\nOptuna最適パラメータ読み込み...")
with open('/home/user/julis_titanic/experiments/results/optuna_best_params.json', 'r') as f:
    optuna_results = json.load(f)

# モデル設定（Optuna最適化パラメータ適用）
model_configs = {
    'LightGBM': {
        'model_class': lgb.LGBMClassifier,
        'params': {
            **optuna_results['LightGBM']['best_params'],
            'verbose': -1
        },
        'features': base_features + ['AgeGroup'],
        'baseline_score': optuna_results['LightGBM']['baseline_score'],
        'optimized_score': optuna_results['LightGBM']['best_score']
    },
    'GradientBoosting': {
        'model_class': GradientBoostingClassifier,
        'params': optuna_results['GradientBoosting']['best_params'],
        'features': base_features + ['AgeGroup'],
        'baseline_score': optuna_results['GradientBoosting']['baseline_score'],
        'optimized_score': optuna_results['GradientBoosting']['best_score']
    },
    'RandomForest': {
        'model_class': RandomForestClassifier,
        'params': optuna_results['RandomForest']['best_params'],
        'features': base_features + ['Age'],
        'baseline_score': optuna_results['RandomForest']['baseline_score'],
        'optimized_score': optuna_results['RandomForest']['best_score']
    },
    'SVM_RBF': {
        'model_class': SVC,
        'params': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True
        },
        'features': base_features + ['Age'],
        'use_scaling': True,
        'baseline_score': None,  # Not tuned with Optuna
        'optimized_score': None
    }
}

# Optunaでの改善サマリー表示
print("\n" + "=" * 80)
print("Optunaチューニング結果")
print("=" * 80)
print(f"\n{'モデル':>20s} {'ベースOOF':>12s} {'最適OOF':>12s} {'改善':>12s}")
print("-" * 80)

total_improvement = 0
for name, config in model_configs.items():
    if config['baseline_score'] is not None:
        baseline = config['baseline_score']
        optimized = config['optimized_score']
        improvement = optimized - baseline
        total_improvement += improvement
        print(f"{name:>20s} {baseline:>12.4f} {optimized:>12.4f} {improvement:>+12.4f}")
    else:
        print(f"{name:>20s} {'(not tuned)':>12s} {'(baseline)':>12s} {'N/A':>12s}")

print("-" * 80)
print(f"{'合計改善':>20s} {'':<12s} {'':<12s} {total_improvement:>+12.4f}")

y_train = train['Survived']

# 10個の乱数シード
n_seeds = 10
seeds = [42 + i * 10 for i in range(n_seeds)]

print(f"\n使用シード数: {n_seeds}個")
print(f"シード: {seeds}")

print("\n" + "=" * 80)
print("ステップ1: OOF評価（単一シード42での検証）")
print("=" * 80)

# 単一シード（42）でOOF評価
single_seed_oof_probabilities = {}

for name, config in model_configs.items():
    print(f"\n{name} (seed=42):")

    model_class = config['model_class']
    params = config['params'].copy()
    params['random_state'] = 42
    features = config['features']
    use_scaling = config.get('use_scaling', False)

    X_train = train[features]

    oof_proba = np.zeros(len(X_train))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model = model_class(**params)

        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        if use_scaling:
            scaler = StandardScaler()
            X_fold_train = pd.DataFrame(
                scaler.fit_transform(X_fold_train),
                columns=X_fold_train.columns,
                index=X_fold_train.index
            )
            X_fold_val = pd.DataFrame(
                scaler.transform(X_fold_val),
                columns=X_fold_val.columns,
                index=X_fold_val.index
            )

        model.fit(X_fold_train, y_fold_train)
        val_proba = model.predict_proba(X_fold_val)[:, 1]
        oof_proba[val_idx] = val_proba

    oof_pred = (oof_proba >= 0.5).astype(int)
    oof_score = accuracy_score(y_train, oof_pred)
    single_seed_oof_probabilities[name] = oof_proba

    print(f"  OOF Score: {oof_score:.4f}")

    # Optunaでの改善を表示
    if config['baseline_score'] is not None:
        improvement = oof_score - config['baseline_score']
        print(f"  Optuna改善: {improvement:+.4f} (ベースライン: {config['baseline_score']:.4f})")

# 4モデルアンサンブル（単一シード）
ensemble_oof_proba = (
    single_seed_oof_probabilities['LightGBM'] +
    single_seed_oof_probabilities['GradientBoosting'] +
    single_seed_oof_probabilities['RandomForest'] +
    single_seed_oof_probabilities['SVM_RBF']
) / 4
ensemble_oof_pred = (ensemble_oof_proba >= 0.5).astype(int)
ensemble_oof_score = accuracy_score(y_train, ensemble_oof_pred)

print(f"\n4モデルアンサンブル（Optuna最適化、単一シード）:")
print(f"  OOF Score: {ensemble_oof_score:.4f}")

# 旧バージョン（最適化前）との比較
old_ensemble_oof = 0.8316  # scripts/models/final_ensemble_10seeds.py の結果
ensemble_improvement = ensemble_oof_score - old_ensemble_oof
print(f"  vs 最適化前: {ensemble_improvement:+.4f}")

print("\n" + "=" * 80)
print(f"ステップ2: 全データで再学習（{n_seeds}シード × 4モデル = {n_seeds*4}モデル）")
print("=" * 80)

# 各シードでTest予測を取得
all_test_probabilities = []

for seed_idx, seed in enumerate(seeds, 1):
    print(f"\n--- シード {seed_idx}/{n_seeds} (random_state={seed}) ---")

    seed_test_probabilities = {}

    for name, config in model_configs.items():
        model_class = config['model_class']
        params = config['params'].copy()
        params['random_state'] = seed
        features = config['features']
        use_scaling = config.get('use_scaling', False)

        X_train = train[features]
        X_test = test[features]

        model = model_class(**params)

        if use_scaling:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

            model.fit(X_train_scaled, y_train)
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            test_proba = model.predict_proba(X_test)[:, 1]

        seed_test_probabilities[name] = test_proba

    # 4モデルアンサンブル（このシード）
    ensemble_test_proba = (
        seed_test_probabilities['LightGBM'] +
        seed_test_probabilities['GradientBoosting'] +
        seed_test_probabilities['RandomForest'] +
        seed_test_probabilities['SVM_RBF']
    ) / 4

    all_test_probabilities.append(ensemble_test_proba)

    print(f"  ✅ 4モデル学習完了")

print("\n" + "=" * 80)
print(f"ステップ3: {n_seeds}シードの予測を平均")
print("=" * 80)

# 全シードの予測を平均
final_test_proba = np.mean(all_test_probabilities, axis=0)
final_test_pred = (final_test_proba >= 0.5).astype(int)

# 予測の標準偏差（安定性の指標）
test_proba_std = np.std(all_test_probabilities, axis=0)
mean_std = np.mean(test_proba_std)
max_std = np.max(test_proba_std)

print(f"\n予測の安定性:")
print(f"  平均標準偏差: {mean_std:.4f} (低いほど安定)")
print(f"  最大標準偏差: {max_std:.4f}")

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_test_pred
})

output_file = '/home/user/julis_titanic/submissions/submission_final_optimized_10seeds.csv'
submission.to_csv(output_file, index=False)

print("\n" + "=" * 80)
print("結果サマリー")
print("=" * 80)

print(f"\n✅ 提出ファイル作成: {output_file}")
print(f"\n構成:")
print(f"  - モデル数: 4種類")
print(f"  - シード数: {n_seeds}個")
print(f"  - 合計: {n_seeds * 4}モデルの平均")
print(f"\n改善要素:")
print(f"  1. Optunaハイパーパラメータ最適化")
print(f"     - LightGBM: {optuna_results['LightGBM']['improvement']:+.4f}")
print(f"     - GradientBoosting: {optuna_results['GradientBoosting']['improvement']:+.4f}")
print(f"     - RandomForest: {optuna_results['RandomForest']['improvement']:+.4f}")
print(f"     - 合計: {total_improvement:+.4f}")
print(f"  2. 10シード平均化（バギング効果）")
print(f"  3. Soft Voting（確率平均）")

print(f"\nOOF Score:")
print(f"  - 最適化前（10シード）: {old_ensemble_oof:.4f}")
print(f"  - 最適化後（単一シード）: {ensemble_oof_score:.4f} ({ensemble_improvement:+.4f})")

print(f"\n予測の安定性:")
print(f"  - 平均std: {mean_std:.4f}")
print(f"  - 最大std: {max_std:.4f}")

# 期待LB
fourth_submission_oof = 0.8249
fourth_submission_lb = 0.78468
fourth_submission_gap = (fourth_submission_oof - fourth_submission_lb) / fourth_submission_oof

expected_lb = ensemble_oof_score * (1 - fourth_submission_gap)
improvement_vs_4th = expected_lb - fourth_submission_lb

print(f"\n期待LB予測（Gap {fourth_submission_gap*100:.2f}%仮定）:")
print(f"  - 4th提出: LB {fourth_submission_lb:.5f} (OOF {fourth_submission_oof:.4f})")
print(f"  - 期待LB: {expected_lb:.5f}")
print(f"  - 期待改善: {improvement_vs_4th:+.5f}")
print(f"  - 目標0.80まで: 残り {0.80 - expected_lb:.5f}")

if expected_lb > fourth_submission_lb:
    print(f"\n✅ 推奨: この提出ファイルを使用（+{improvement_vs_4th:.5f}の改善期待）")
else:
    print(f"\n⚠️  注意: 期待LBは4th提出より低い可能性")

print("\n比較:")
print(f"  4th提出: LB 0.78468 ← 現在のベスト")
print(f"  最適化版: LB {expected_lb:.5f}（期待値）")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
