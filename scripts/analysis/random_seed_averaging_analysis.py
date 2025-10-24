"""
複数乱数シード平均化の効果分析

目的:
- 異なる乱数シード数（5, 10, 20, 30個）で予測の安定性を比較
- 最適なシード数を決定
- 計算時間とのトレードオフを評価

理論:
- 複数シードで学習 → モデルの初期化が異なる
- 予測を平均 → 分散（variance）が減少
- バギング（Bootstrap Aggregating）に似た効果
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("複数乱数シード平均化の効果分析")
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

# 4モデル構成
model_configs = {
    'LightGBM': {
        'model_class': lgb.LGBMClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'num_leaves': 4,
            'min_child_samples': 30,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_split_gain': 0.01,
            'learning_rate': 0.05,
            'verbose': -1
        },
        'features': base_features + ['AgeGroup']
    },
    'GradientBoosting': {
        'model_class': GradientBoostingClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05
        },
        'features': base_features + ['AgeGroup']
    },
    'RandomForest': {
        'model_class': RandomForestClassifier,
        'params': {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        },
        'features': base_features + ['Age']
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
        'use_scaling': True
    }
}

y_train = train['Survived']

# テストするシード数
seed_counts = [1, 5, 10, 20, 30]

print("\n" + "=" * 80)
print("実験: 異なるシード数での予測安定性")
print("=" * 80)

results = []

for n_seeds in seed_counts:
    print(f"\n{'='*80}")
    print(f"シード数: {n_seeds}個")
    print(f"{'='*80}")

    start_time = time.time()

    # 各シードで4モデルアンサンブルのOOF予測を取得
    all_seeds_oof_proba = []

    seeds = [42 + i * 10 for i in range(n_seeds)]

    for seed_idx, seed in enumerate(seeds, 1):
        if n_seeds > 1:
            print(f"\nシード {seed_idx}/{n_seeds} (random_state={seed})")

        # 各モデルのOOF予測
        model_oof_probabilities = {}

        for name, config in model_configs.items():
            model_class = config['model_class']
            params = config['params'].copy()
            params['random_state'] = seed
            features = config['features']
            use_scaling = config.get('use_scaling', False)

            X_train = train[features]

            oof_proba = np.zeros(len(X_train))

            # CV（簡略版: random_stateを変えて評価）
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

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

            model_oof_probabilities[name] = oof_proba

        # 4モデルアンサンブル（Soft Voting）
        ensemble_oof_proba = (
            model_oof_probabilities['LightGBM'] +
            model_oof_probabilities['GradientBoosting'] +
            model_oof_probabilities['RandomForest'] +
            model_oof_probabilities['SVM_RBF']
        ) / 4

        all_seeds_oof_proba.append(ensemble_oof_proba)

    # シード平均
    avg_oof_proba = np.mean(all_seeds_oof_proba, axis=0)
    avg_oof_pred = (avg_oof_proba >= 0.5).astype(int)
    avg_oof_score = accuracy_score(y_train, avg_oof_pred)

    # 予測の標準偏差（安定性の指標）
    std_oof_proba = np.std(all_seeds_oof_proba, axis=0)
    mean_std = np.mean(std_oof_proba)
    max_std = np.max(std_oof_proba)

    elapsed_time = time.time() - start_time

    results.append({
        'n_seeds': n_seeds,
        'oof_score': avg_oof_score,
        'mean_std': mean_std,
        'max_std': max_std,
        'time_sec': elapsed_time
    })

    print(f"\n結果:")
    print(f"  OOF Score: {avg_oof_score:.4f}")
    print(f"  予測の平均標準偏差: {mean_std:.4f} (低いほど安定)")
    print(f"  予測の最大標準偏差: {max_std:.4f}")
    print(f"  計算時間: {elapsed_time:.1f}秒")

# 結果サマリー
print("\n" + "=" * 80)
print("結果サマリー")
print("=" * 80)

results_df = pd.DataFrame(results)

print(f"\n{'シード数':>8s} {'OOF Score':>12s} {'平均std':>10s} {'最大std':>10s} {'時間(秒)':>10s} {'時間比':>10s}")
print("-" * 80)

base_time = results_df.iloc[0]['time_sec']
for _, row in results_df.iterrows():
    time_ratio = row['time_sec'] / base_time
    print(f"{int(row['n_seeds']):>8d} {row['oof_score']:>12.4f} {row['mean_std']:>10.4f} {row['max_std']:>10.4f} {row['time_sec']:>10.1f} {time_ratio:>10.2f}x")

# 改善幅の分析
print("\n" + "=" * 80)
print("改善幅の分析")
print("=" * 80)

baseline_score = results_df.iloc[0]['oof_score']
baseline_std = results_df.iloc[0]['mean_std']

print(f"\nベースライン（シード1個）:")
print(f"  OOF Score: {baseline_score:.4f}")
print(f"  平均std: {baseline_std:.4f}")

print(f"\n{'シード数':>8s} {'OOF改善':>12s} {'std改善':>12s} {'推奨':>10s}")
print("-" * 80)

for _, row in results_df.iterrows():
    if row['n_seeds'] == 1:
        continue

    score_improve = row['oof_score'] - baseline_score
    std_improve = baseline_std - row['mean_std']

    # 推奨判定
    if row['n_seeds'] <= 10:
        recommend = "✅"
    elif row['n_seeds'] <= 20:
        recommend = "⭐"
    else:
        recommend = "△"

    print(f"{int(row['n_seeds']):>8d} {score_improve:>+12.4f} {std_improve:>+12.4f} {recommend:>10s}")

print("\n" + "=" * 80)
print("推奨")
print("=" * 80)

# 最適なシード数を推定
best_balance_idx = 1  # デフォルトは5個
if len(results_df) > 2:
    # std改善と時間のバランスが良いポイントを探す
    # std改善が頭打ちになる前のポイント
    std_improvements = [baseline_std - r['mean_std'] for r in results]

    # 1つ前との改善幅が小さくなるポイント
    for i in range(2, len(std_improvements)):
        prev_gain = std_improvements[i-1] - std_improvements[i-2] if i >= 2 else std_improvements[i-1]
        curr_gain = std_improvements[i] - std_improvements[i-1]

        # 収穫逓減が見られたら
        if prev_gain > 0 and curr_gain < prev_gain * 0.5:
            best_balance_idx = i - 1
            break

best_n_seeds = results_df.iloc[best_balance_idx]['n_seeds']

print(f"\n推奨シード数: {best_n_seeds}個")
print(f"\n理由:")
print(f"  - 計算時間: {results_df.iloc[best_balance_idx]['time_sec']:.1f}秒 ({results_df.iloc[best_balance_idx]['time_sec']/base_time:.1f}x)")
print(f"  - OOF改善: {results_df.iloc[best_balance_idx]['oof_score'] - baseline_score:+.4f}")
print(f"  - 安定性改善: {baseline_std - results_df.iloc[best_balance_idx]['mean_std']:+.4f}")

print(f"\n一般的な推奨:")
print(f"  - 5-10個: 通常はこれで十分 ✅")
print(f"  - 10-20個: より慎重に精度を追求する場合 ⭐")
print(f"  - 20+個: 時間に余裕があり、最大限の安定性を求める場合 △")

print("\n次のステップ:")
print(f"  → 推奨シード数（{best_n_seeds}個）で最終アンサンブルを実装")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)

# 結果をファイルに保存
results_df.to_csv('/home/user/julis_titanic/experiments/results/random_seed_averaging_analysis.csv', index=False)
print(f"\n結果を保存: experiments/results/random_seed_averaging_analysis.csv")
