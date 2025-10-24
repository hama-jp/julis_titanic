"""
4モデルアンサンブル（LightGBM + GradientBoosting + RandomForest + SVM_RBF）

多様性分析の結果:
- SVMはRFと高相関（0.9654）だが、全体として微増効果あり
- 4モデル Soft Voting: OOF 0.8316 (+0.0022 vs 3モデル)
- 期待LB: 0.79110 (+0.00642 vs 4th提出)

懸念点:
- SVMのGap（6.78%）が4thのGap（3.9%）より大きい
- アンサンブル全体のGapが悪化する可能性あり

目的:
- 正確な4モデル Soft Votingの提出ファイルを生成
- 明日の提出枠復活に向けた実験
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
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("4モデルアンサンブル提出ファイル生成")
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

    # AgeGroup
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

# モデル設定
base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize', 'SmallFamily'] + embarked_cols

model_configs = {
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=2,
            num_leaves=4,
            min_child_samples=30,
            reg_alpha=1.0,
            reg_lambda=1.0,
            min_split_gain=0.01,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        ),
        'features': base_features + ['AgeGroup']
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        ),
        'features': base_features + ['AgeGroup']
    },
    'RandomForest': {
        'model': RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        'features': base_features + ['Age']
    },
    'SVM_RBF': {
        'model': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True),
        'features': base_features + ['Age'],
        'use_scaling': True
    }
}

y_train = train['Survived']

print("\n" + "=" * 80)
print("ステップ1: Cross-Validation で OOF予測を取得")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = {}
oof_probabilities = {}
test_probabilities = {}

for name, config in model_configs.items():
    print(f"\n{name}:")

    model = config['model']
    features = config['features']
    use_scaling = config.get('use_scaling', False)

    X_train = train[features]
    X_test = test[features]

    oof_pred = np.zeros(len(X_train))
    oof_proba = np.zeros(len(X_train))
    test_proba_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
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
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        model.fit(X_fold_train, y_fold_train)

        val_pred = model.predict(X_fold_val)
        val_proba = model.predict_proba(X_fold_val)[:, 1]

        oof_pred[val_idx] = val_pred
        oof_proba[val_idx] = val_proba

        if use_scaling:
            test_proba_list.append(model.predict_proba(X_test_scaled)[:, 1])
        else:
            test_proba_list.append(model.predict_proba(X_test)[:, 1])

    oof_score = accuracy_score(y_train, oof_pred)
    oof_predictions[name] = oof_pred
    oof_probabilities[name] = oof_proba

    # Test予測（CV fold の平均）
    test_probabilities[name] = np.mean(test_proba_list, axis=0)

    print(f"  OOF Score: {oof_score:.4f}")

print("\n" + "=" * 80)
print("ステップ2: アンサンブル評価")
print("=" * 80)

# 3モデルアンサンブル（Soft Voting）
ensemble_3_soft_proba = (
    oof_probabilities['LightGBM'] +
    oof_probabilities['GradientBoosting'] +
    oof_probabilities['RandomForest']
) / 3
ensemble_3_soft_pred = (ensemble_3_soft_proba >= 0.5).astype(int)
ensemble_3_soft_score = accuracy_score(y_train, ensemble_3_soft_pred)

# 4モデルアンサンブル（Soft Voting）
ensemble_4_soft_proba = (
    oof_probabilities['LightGBM'] +
    oof_probabilities['GradientBoosting'] +
    oof_probabilities['RandomForest'] +
    oof_probabilities['SVM_RBF']
) / 4
ensemble_4_soft_pred = (ensemble_4_soft_proba >= 0.5).astype(int)
ensemble_4_soft_score = accuracy_score(y_train, ensemble_4_soft_pred)

print(f"\n{'アンサンブル':30s} {'OOF Score':>12s} {'改善':>10s}")
print("-" * 60)
print(f"{'3モデル Soft Voting':30s} {ensemble_3_soft_score:>12.4f} {'-':>10s}")
print(f"{'4モデル Soft Voting':30s} {ensemble_4_soft_score:>12.4f} {ensemble_4_soft_score - ensemble_3_soft_score:>+10.4f}")

# 期待LB
fourth_submission_oof = 0.8249
fourth_submission_lb = 0.78468
fourth_submission_gap = (fourth_submission_oof - fourth_submission_lb) / fourth_submission_oof

expected_lb_4model = ensemble_4_soft_score * (1 - fourth_submission_gap)
improvement = expected_lb_4model - fourth_submission_lb

print(f"\n期待LB予測（Gap {fourth_submission_gap*100:.2f}%仮定）:")
print(f"  4モデル Soft: OOF {ensemble_4_soft_score:.4f} → 期待LB {expected_lb_4model:.5f} ({improvement:+.5f})")

print("\n" + "=" * 80)
print("ステップ3: Test予測生成")
print("=" * 80)

# 4モデル Soft Voting（Test）
test_ensemble_proba = (
    test_probabilities['LightGBM'] +
    test_probabilities['GradientBoosting'] +
    test_probabilities['RandomForest'] +
    test_probabilities['SVM_RBF']
) / 4
test_ensemble_pred = (test_ensemble_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_ensemble_pred
})

output_file = '/home/user/julis_titanic/submissions/submission_4model_ensemble_soft_voting.csv'
submission.to_csv(output_file, index=False)

print(f"\n✅ 提出ファイル作成: {output_file}")
print(f"   モデル: LightGBM + GradientBoosting + RandomForest + SVM_RBF")
print(f"   手法: Soft Voting（確率平均 >= 0.5）")
print(f"   OOF Score: {ensemble_4_soft_score:.4f}")
print(f"   期待LB: {expected_lb_4model:.5f} ({improvement:+.5f})")

print("\n" + "=" * 80)
print("リスク分析")
print("=" * 80)

print("\n✅ 潜在的メリット:")
print("  - OOF改善: +0.0022 (vs 3モデル)")
print("  - 期待LB改善: +0.00642")
print("  - アンサンブルの多様性が若干増加")

print("\n⚠️ 懸念点:")
print("  - SVMのGap（6.78%）が4thのGap（3.9%）より大きい")
print("  - RandomForest vs SVM の相関が非常に高い（0.9654）")
print("  - アンサンブル全体のGapが悪化する可能性あり")

print("\n推奨:")
print("  - 明日の提出枠復活を待つ")
print("  - 4モデルアンサンブルを試す価値はあるが、慎重に")
print("  - 期待改善は +0.00642（目標0.80まで残り+0.00890）")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
