"""
9特徴量Hard VotingモデルのOOFスコア再計算

目的:
- 9特徴量モデルの正確なOOFスコアを計算
- CV-LB Gapを正確に測定
- LB Score: 0.77751との比較

特徴量:
1. Pclass
2. Sex
3. Age
4. SibSp
5. Parch
6. Fare
7. Embarked
8. Title
9. FamilySize
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("9特徴量Hard VotingモデルのOOFスコア計算")
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

    # タイトルのグループ化
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

    # 欠損値処理
    df_new['Age'] = df_new['Age'].fillna(df_new['Age'].median())
    df_new['Fare'] = df_new['Fare'].fillna(df_new['Fare'].median())
    df_new['Embarked'] = df_new['Embarked'].fillna('S')

    # Embarkedをダミー変数化
    df_new = pd.get_dummies(df_new, columns=['Embarked'], drop_first=True)

    # Sexを数値化
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})

    return df_new

# 特徴量追加
print("\n特徴量エンジニアリング...")
train = add_title_feature(train)
train = add_familysize_feature(train)
train = preprocess_data(train)

test = add_title_feature(test)
test = add_familysize_feature(test)
test = preprocess_data(test)

# 9特徴量を選択
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']

# Embarkedダミー変数を追加
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

# モデル定義（過去のTop 3モデルと同じ設定）
print("\nモデル設定:")

# LightGBM（ベースライン）
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
print("  - LightGBM: n_estimators=100, max_depth=3, num_leaves=8")

# GradientBoosting（ベースライン）
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
print("  - GradientBoosting: n_estimators=100, max_depth=3")

# RandomForest（ベースライン）
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
print("  - RandomForest: n_estimators=100, min_samples_split=10")

# Hard Voting Ensemble
hard_voting = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    voting='hard'
)
print("  - Voting: Hard")

# Stratified K-Fold Cross-Validation
print("\nStratified 5-Fold Cross-Validation開始...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    hard_voting.fit(X_fold_train, y_fold_train)

    # Validation予測
    val_pred = hard_voting.predict(X_fold_val)
    oof_predictions[val_idx] = val_pred

    # Fold精度
    fold_acc = accuracy_score(y_fold_val, val_pred)
    fold_scores.append(fold_acc)
    print(f"Fold {fold} Accuracy: {fold_acc:.4f}")

    # Test予測（アンサンブル用）
    test_pred = hard_voting.predict(X_test)
    test_predictions_list.append(test_pred)

# OOF Score計算
oof_score = accuracy_score(y_train, oof_predictions)

print("\n" + "=" * 80)
print("結果サマリー")
print("=" * 80)

print(f"\n各Foldの精度:")
for fold, score in enumerate(fold_scores, 1):
    print(f"  Fold {fold}: {score:.4f}")

print(f"\nCV平均: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"OOF Score: {oof_score:.4f}")

# LB Scoreとの比較
lb_score = 0.77751
cv_lb_gap = oof_score - lb_score
cv_lb_gap_pct = cv_lb_gap * 100

print(f"\n--- CV-LB Gap分析 ---")
print(f"OOF Score (CV): {oof_score:.4f}")
print(f"LB Score (Kaggle): {lb_score:.5f}")
print(f"CV-LB Gap: {cv_lb_gap:.5f} ({cv_lb_gap_pct:.2f}%)")

# Gap評価
if cv_lb_gap_pct < 3:
    gap_eval = "✅ 優秀"
elif cv_lb_gap_pct < 5:
    gap_eval = "✅ 許容範囲"
elif cv_lb_gap_pct < 8:
    gap_eval = "⚠️ 要注意"
else:
    gap_eval = "❌ 深刻"

print(f"Gap評価: {gap_eval}")

# 10特徴量モデルとの比較
print(f"\n--- 10特徴量モデルとの比較 ---")
print(f"10特徴（+Deck_Safe）:")
print(f"  OOF: 0.8305")
print(f"  LB: 0.76794")
print(f"  Gap: 0.06256 (6.26%)")
print(f"\n9特徴（Deck_Safe除外）:")
print(f"  OOF: {oof_score:.4f}")
print(f"  LB: {lb_score:.5f}")
print(f"  Gap: {cv_lb_gap:.5f} ({cv_lb_gap_pct:.2f}%)")

oof_diff = oof_score - 0.8305
lb_diff = lb_score - 0.76794
gap_improvement = 6.26 - cv_lb_gap_pct

print(f"\n変化:")
print(f"  OOF: {oof_diff:+.4f} ({'改善' if oof_diff > 0 else '悪化'})")
print(f"  LB: {lb_diff:+.5f} ({'改善' if lb_diff > 0 else '悪化'})")
print(f"  Gap: {gap_improvement:+.2f}% ({'改善' if gap_improvement > 0 else '悪化'})")

# Test予測（多数決）
print("\nTest予測生成...")
test_predictions_array = np.array(test_predictions_list)
final_test_predictions = np.round(test_predictions_array.mean(axis=0)).astype(int)

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_test_predictions
})

output_file = 'submission_9features_hardvoting_recalculated.csv'
submission.to_csv(output_file, index=False)
print(f"提出ファイル作成: {output_file}")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)

# 結果をファイルに保存
summary_file = 'results/9features_oof_analysis.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("9特徴量Hard VotingモデルのOOFスコア分析\n")
    f.write("=" * 80 + "\n\n")

    f.write("使用した特徴量:\n")
    for i, col in enumerate(feature_cols, 1):
        f.write(f"  {i}. {col}\n")

    f.write(f"\n各Foldの精度:\n")
    for fold, score in enumerate(fold_scores, 1):
        f.write(f"  Fold {fold}: {score:.4f}\n")

    f.write(f"\nCV平均: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}\n")
    f.write(f"OOF Score: {oof_score:.4f}\n")

    f.write(f"\n--- CV-LB Gap分析 ---\n")
    f.write(f"OOF Score (CV): {oof_score:.4f}\n")
    f.write(f"LB Score (Kaggle): {lb_score:.5f}\n")
    f.write(f"CV-LB Gap: {cv_lb_gap:.5f} ({cv_lb_gap_pct:.2f}%)\n")
    f.write(f"Gap評価: {gap_eval}\n")

    f.write(f"\n--- 10特徴量モデルとの比較 ---\n")
    f.write(f"10特徴（+Deck_Safe）:\n")
    f.write(f"  OOF: 0.8305\n")
    f.write(f"  LB: 0.76794\n")
    f.write(f"  Gap: 0.06256 (6.26%)\n")
    f.write(f"\n9特徴（Deck_Safe除外）:\n")
    f.write(f"  OOF: {oof_score:.4f}\n")
    f.write(f"  LB: {lb_score:.5f}\n")
    f.write(f"  Gap: {cv_lb_gap:.5f} ({cv_lb_gap_pct:.2f}%)\n")

    f.write(f"\n変化:\n")
    f.write(f"  OOF: {oof_diff:+.4f}\n")
    f.write(f"  LB: {lb_diff:+.5f}\n")
    f.write(f"  Gap: {gap_improvement:+.2f}%\n")

print(f"\n結果を保存: {summary_file}")
