"""
XGBoost追加4モデルアンサンブル

目的:
- 現在の3モデル（LightGBM, GB, RF）にXGBoostを追加
- 4モデルのHard Votingアンサンブル
- LB 0.78以上を目指す

ベースライン:
- 9特徴量 3モデルHard Voting
- OOF: 0.8384, LB: 0.77751
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBoost追加4モデルアンサンブル")
print("=" * 80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 特徴量エンジニアリング関数
def add_title_feature(df):
    """Titleカラム追加"""
    df_new = df.copy()
    df_new['Title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Mme': 2,
        'Don': 4, 'Dona': 4, 'Lady': 4, 'Countess': 4, 'Jonkheer': 4, 'Sir': 4,
        'Capt': 4, 'Ms': 1
    }
    df_new['Title'] = df_new['Title'].map(title_mapping).fillna(4)
    return df_new

def add_familysize_feature(df):
    """FamilySizeカラム追加"""
    df_new = df.copy()
    df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
    return df_new

def preprocess_data(df):
    """前処理"""
    df_new = df.copy()
    df_new['Age'] = df_new['Age'].fillna(df_new['Age'].median())
    df_new['Fare'] = df_new['Fare'].fillna(df_new['Fare'].median())
    df_new['Embarked'] = df_new['Embarked'].fillna('S')
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})
    return df_new

# 特徴量追加
print("\n特徴量エンジニアリング...")
train = preprocess_data(train)
train = add_title_feature(train)
train = add_familysize_feature(train)
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)

test = preprocess_data(test)
test = add_title_feature(test)
test = add_familysize_feature(test)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# 9特徴量
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']
embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
feature_cols.extend(embarked_cols)

print(f"\n使用する特徴量（{len(feature_cols)}個）:")
for i, col in enumerate(feature_cols, 1):
    print(f"  {i}. {col}")

X_train = train[feature_cols]
y_train = train['Survived']
X_test = test[feature_cols]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ========================================
# ベースライン: 3モデルアンサンブル
# ========================================
print("\n" + "=" * 80)
print("ベースライン: 3モデル（LightGBM + GB + RF）")
print("=" * 80)

lgb_model_3 = lgb.LGBMClassifier(
    n_estimators=100, max_depth=3, num_leaves=8,
    learning_rate=0.05, random_state=42, verbose=-1
)
gb_model_3 = GradientBoostingClassifier(
    n_estimators=100, max_depth=3,
    learning_rate=0.05, random_state=42
)
rf_model_3 = RandomForestClassifier(
    n_estimators=100, max_depth=None,
    min_samples_split=10, min_samples_leaf=4, random_state=42
)

voting_3models = VotingClassifier(
    estimators=[('lgb', lgb_model_3), ('gb', gb_model_3), ('rf', rf_model_3)],
    voting='hard'
)

# CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_3models = np.zeros(len(X_train))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]

    voting_3models.fit(X_fold_train, y_fold_train)
    val_pred = voting_3models.predict(X_fold_val)
    oof_3models[val_idx] = val_pred

oof_score_3models = accuracy_score(y_train, oof_3models)
print(f"\n3モデルアンサンブル OOF Score: {oof_score_3models:.4f}")

# ========================================
# 新規: 4モデルアンサンブル（XGBoost追加）
# ========================================
print("\n" + "=" * 80)
print("新規: 4モデル（LightGBM + GB + RF + XGBoost）")
print("=" * 80)

# モデル定義
print("\nモデル設定:")

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
print("  - LightGBM: n_estimators=100, max_depth=3, num_leaves=8")

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
print("  - GradientBoosting: n_estimators=100, max_depth=3")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
print("  - RandomForest: n_estimators=100, min_samples_split=10")

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
print("  - XGBoost: n_estimators=100, max_depth=3")

# 4モデルHard Voting
voting_4models = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('gb', gb_model),
        ('rf', rf_model),
        ('xgb', xgb_model)
    ],
    voting='hard'
)
print("  - Voting: Hard（4モデル）")

# Stratified K-Fold CV
print("\nStratified 5-Fold Cross-Validation開始...")

oof_predictions = np.zeros(len(X_train))
test_predictions_list = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold}/5 ---")

    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # モデル訓練
    voting_4models.fit(X_fold_train, y_fold_train)

    # Validation予測
    val_pred = voting_4models.predict(X_fold_val)
    oof_predictions[val_idx] = val_pred

    # Fold精度
    fold_acc = accuracy_score(y_fold_val, val_pred)
    fold_scores.append(fold_acc)
    print(f"Fold {fold} Accuracy: {fold_acc:.4f}")

    # Test予測
    test_pred = voting_4models.predict(X_test)
    test_predictions_list.append(test_pred)

# OOF Score計算
oof_score_4models = accuracy_score(y_train, oof_predictions)

print("\n" + "=" * 80)
print("結果サマリー")
print("=" * 80)

print(f"\n各Foldの精度:")
for fold, score in enumerate(fold_scores, 1):
    print(f"  Fold {fold}: {score:.4f}")

print(f"\nCV平均: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"4モデルアンサンブル OOF Score: {oof_score_4models:.4f}")

# ベースラインとの比較
oof_improvement = oof_score_4models - oof_score_3models

print(f"\n--- ベースライン比較 ---")
print(f"ベースライン（3モデル）: {oof_score_3models:.4f}")
print(f"XGBoost追加（4モデル）: {oof_score_4models:.4f}")
print(f"OOF改善: {oof_improvement:+.4f} ({oof_improvement*100:+.2f}%)")

if oof_improvement > 0:
    print("✅ OOFスコア改善")
else:
    print("❌ OOFスコア悪化")

# 期待LB予測
baseline_lb = 0.77751
expected_gap = 0.0609  # 9特徴量モデルのGap
expected_lb = oof_score_4models - expected_gap

print(f"\n--- LB予測（Gap {expected_gap*100:.2f}%仮定） ---")
print(f"期待LB Score: {expected_lb:.5f}")
print(f"ベースラインLB: {baseline_lb:.5f}")
print(f"予想LB改善: {expected_lb - baseline_lb:+.5f}")

# Test予測（多数決）
print("\nTest予測生成...")
test_predictions_array = np.array(test_predictions_list)
final_test_predictions = np.round(test_predictions_array.mean(axis=0)).astype(int)

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_test_predictions
})

output_file = 'submission_xgboost_4model_ensemble.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 提出ファイル作成: {output_file}")

print("\n" + "=" * 80)
print("評価結果サマリー")
print("=" * 80)

print(f"\n1. OOFスコア: {oof_score_4models:.4f} ({oof_improvement:+.4f})")
print(f"2. 期待LB: {expected_lb:.5f} ({expected_lb - baseline_lb:+.5f})")

print("\n次のアクション:")
if oof_improvement > 0:
    print(f"✅ OOF改善 → Kaggle LBで検証推奨")
    print(f"   提出ファイル: {output_file}")
    print(f"   期待LB: {expected_lb:.5f}")
elif oof_improvement >= -0.001:  # ほぼ変わらず
    print(f"⚠️ OOF改善なし（±0.1%以内） → LB検証は任意")
else:
    print(f"❌ OOF悪化 → この改善は不採用")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)

# 結果をファイルに保存
summary_file = 'results/xgboost_4model_ensemble_evaluation.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("XGBoost追加4モデルアンサンブル評価結果\n")
    f.write("=" * 80 + "\n\n")

    f.write("使用した特徴量（9特徴）:\n")
    for i, col in enumerate(feature_cols, 1):
        f.write(f"  {i}. {col}\n")

    f.write(f"\n--- OOFスコア ---\n")
    f.write(f"ベースライン（3モデル）: {oof_score_3models:.4f}\n")
    f.write(f"XGBoost追加（4モデル）: {oof_score_4models:.4f}\n")
    f.write(f"改善: {oof_improvement:+.4f} ({oof_improvement*100:+.2f}%)\n")

    f.write(f"\n--- 期待LB（Gap {expected_gap*100:.2f}%仮定） ---\n")
    f.write(f"期待LB Score: {expected_lb:.5f}\n")
    f.write(f"ベースラインLB: {baseline_lb:.5f}\n")
    f.write(f"予想改善: {expected_lb - baseline_lb:+.5f}\n")

print(f"\n詳細結果を保存: {summary_file}")
