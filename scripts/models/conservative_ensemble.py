"""
保守的な特徴量セットでアンサンブル
- Ticket特徴量は最小限
- 強い正則化
- CV-LBギャップを最小化
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import sys
from datetime import datetime

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline
from scripts.feature_engineering.ticket_feature_evaluation import add_ticket_features, engineer_features

print("=" * 80)
print("保守的特徴量セット: CV-LBギャップ最小化")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')
passenger_ids = test['PassengerId']

# 補完パイプライン
imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

# 基本特徴量のみ（Ticket特徴量を追加する前の状態）
train = engineer_features(train)
test = engineer_features(test)

# Embarked
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)
embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

y = train['Survived']

# 保守的な特徴量セット（前回の成功パターン）
BASE_FEATURES = [
    'Pclass', 'Sex', 'Title',
    'SibSp', 'Parch', 'FamilySize', 'SmallFamily',
    'Fare',
    'Embarked_Missing', 'Embarked_Q', 'Embarked_S'
]

print("\n" + "=" * 80)
print("特徴量セット比較")
print("=" * 80)

# Strategy 1: 完全に保守的（Ticket特徴量なし）
conservative_features = BASE_FEATURES.copy()

print(f"\n[Strategy 1] Conservative (No Ticket features)")
print(f"  Features ({len(conservative_features)}): {conservative_features}")

# Strategy 2: Ticket_Fare_Per_Person のみ追加（最も有望な1つだけ）
# Ticket特徴量を追加（copyせずに直接）
train, test = add_ticket_features(train, test)

moderate_features = BASE_FEATURES + ['Ticket_Fare_Per_Person']

print(f"\n[Strategy 2] Moderate (Only Ticket_Fare_Per_Person)")
print(f"  Features ({len(moderate_features)}): {moderate_features}")

# Strategy 3: FamilySize除外（Ticket_GroupSizeと冗長）
# Ticket_GroupSizeとTicket_Fare_Per_Personのみ

minimal_ticket_features = [
    'Pclass', 'Sex', 'Title',
    'SibSp', 'Parch', 'SmallFamily',  # FamilySizeを除外
    'Ticket_GroupSize',  # FamilySizeの代わり
    'Ticket_Fare_Per_Person',
    'Embarked_Missing', 'Embarked_Q', 'Embarked_S'
]

print(f"\n[Strategy 3] Minimal Ticket (GroupSize + FarePP only)")
print(f"  Features ({len(minimal_ticket_features)}): {minimal_ticket_features}")

# 評価関数
def evaluate_feature_set(features, name):
    print(f"\n{'=' * 80}")
    print(f"評価: {name}")
    print(f"{'=' * 80}")

    X = train[features]
    X_test = test[features]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # LightGBM（保守的パラメータ）
    model_lgb = lgb.LGBMClassifier(
        n_estimators=100,  # Optunaの194→100に削減
        max_depth=4,       # Optunaの6→4に削減
        learning_rate=0.05,  # Optunaの0.083→0.05に削減
        num_leaves=20,     # Optunaの56→20に削減
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,  # Optunaの25→30に増加（正則化強化）
        random_state=42,
        verbose=-1
    )
    scores_lgb = cross_val_score(model_lgb, X, y, cv=cv, scoring='accuracy')
    results['LightGBM'] = scores_lgb.mean()
    print(f"  LightGBM: {scores_lgb.mean():.4f} (+/- {scores_lgb.std():.4f})")

    # GradientBoosting（保守的パラメータ）
    model_gb = GradientBoostingClassifier(
        n_estimators=100,  # Optunaの215→100に削減
        max_depth=3,       # Optunaの5→3に削減
        learning_rate=0.05,  # Optunaの0.038→0.05に微増
        subsample=0.8,
        min_samples_split=15,  # Optunaの12→15に増加
        min_samples_leaf=5,    # Optunaの2→5に増加
        random_state=42
    )
    scores_gb = cross_val_score(model_gb, X, y, cv=cv, scoring='accuracy')
    results['GradientBoosting'] = scores_gb.mean()
    print(f"  GradientBoosting: {scores_gb.mean():.4f} (+/- {scores_gb.std():.4f})")

    # RandomForest（保守的パラメータ）
    model_rf = RandomForestClassifier(
        n_estimators=100,  # Optunaの239→100に削減
        max_depth=5,       # Optunaの6→5に削減
        min_samples_split=15,  # Optunaの9→15に増加
        min_samples_leaf=5,    # Optunaの1→5に増加
        max_features='sqrt',   # OptunaのNone→sqrtに変更
        random_state=42
    )
    scores_rf = cross_val_score(model_rf, X, y, cv=cv, scoring='accuracy')
    results['RandomForest'] = scores_rf.mean()
    print(f"  RandomForest: {scores_rf.mean():.4f} (+/- {scores_rf.std():.4f})")

    # Soft Voting Ensemble
    model_lgb.fit(X, y)
    model_gb.fit(X, y)
    model_rf.fit(X, y)

    pred_lgb = model_lgb.predict_proba(X_test)[:, 1]
    pred_gb = model_gb.predict_proba(X_test)[:, 1]
    pred_rf = model_rf.predict_proba(X_test)[:, 1]

    # Equal weight soft voting
    ensemble_pred = (pred_lgb + pred_gb + pred_rf) / 3

    avg_oof = np.mean(list(results.values()))
    print(f"\n  平均OOF: {avg_oof:.4f}")
    print(f"  期待LBギャップ: ~4-5% (保守的パラメータ)")

    return ensemble_pred, results, avg_oof

# 各戦略を評価
results_all = {}

pred_conservative, res_conservative, oof_conservative = evaluate_feature_set(
    conservative_features, "Conservative (No Ticket)"
)
results_all['conservative'] = (pred_conservative, res_conservative, oof_conservative)

pred_moderate, res_moderate, oof_moderate = evaluate_feature_set(
    moderate_features, "Moderate (FarePP only)"
)
results_all['moderate'] = (pred_moderate, res_moderate, oof_moderate)

pred_minimal, res_minimal, oof_minimal = evaluate_feature_set(
    minimal_ticket_features, "Minimal Ticket"
)
results_all['minimal'] = (pred_minimal, res_minimal, oof_minimal)

# 提出ファイル生成
print("\n" + "=" * 80)
print("提出ファイル生成")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")

for strategy_name, (pred, res, oof) in results_all.items():
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': (pred >= 0.5).astype(int)
    })
    filename = f"submissions/{timestamp}_conservative_{strategy_name}.csv"
    submission.to_csv(filename, index=False)
    print(f"✓ {filename}")
    print(f"  OOF: {oof:.4f}")
    print(f"  Survived: {(pred >= 0.5).sum()}/418 ({(pred >= 0.5).sum()/418*100:.1f}%)")

# 最終推奨
print("\n" + "=" * 80)
print("推奨提出順序")
print("=" * 80)

strategies_sorted = sorted(results_all.items(),
                          key=lambda x: x[2],
                          reverse=True)

for rank, (strategy_name, (_, res, oof)) in enumerate(strategies_sorted, 1):
    print(f"{rank}. {timestamp}_conservative_{strategy_name}.csv (OOF: {oof:.4f})")

print("\n完了！")
