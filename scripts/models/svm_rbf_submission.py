"""
SVM RBF 単体モデルの提出ファイル生成

目的:
- Age × Pclass 交互作用評価で、SVM_RBFがOOF 0.8316を記録
- 参考用に提出ファイルを生成（5th提出の候補の一つ）
- ⚠️ ただし、OOFが高い ≠ LBが高い という教訓あり

OOF Score: 0.8316
期待LB: 不明（リスク中～高）
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("SVM RBF 単体モデルの提出ファイル生成")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 正しい補完パイプライン
imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

def engineer_features(df):
    """特徴量エンジニアリング"""
    df_new = df.copy()

    # Sex
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})

    # Title
    df_new['Title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Mme': 2,
        'Don': 4, 'Dona': 4, 'Lady': 4, 'Countess': 4, 'Jonkheer': 4, 'Sir': 4,
        'Capt': 4, 'Ms': 1
    }
    df_new['Title'] = df_new['Title'].map(title_mapping).fillna(4)

    # FamilySize
    df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1

    # SmallFamily
    df_new['SmallFamily'] = ((df_new['FamilySize'] >= 2) & (df_new['FamilySize'] <= 4)).astype(int)

    return df_new

print("\n特徴量エンジニアリング...")
train = engineer_features(train)
test = engineer_features(test)

# Embarkedダミー変数化
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# 列の一致確認
embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# 特徴量
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title',
            'FamilySize', 'SmallFamily'] + embarked_cols

print(f"\n使用特徴量（{len(features)}個）:")
for i, col in enumerate(features, 1):
    print(f"  {i}. {col}")

X_train = train[features]
y_train = train['Survived']
X_test = test[features]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# SVM RBF モデル
print("\n" + "=" * 80)
print("SVM RBF 訓練")
print("=" * 80)

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)

# Stratified K-Fold Cross-Validation
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

    # スケーリング（SVM必須）
    scaler = StandardScaler()
    X_fold_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_fold_train),
        columns=X_fold_train.columns,
        index=X_fold_train.index
    )
    X_fold_val_scaled = pd.DataFrame(
        scaler.transform(X_fold_val),
        columns=X_fold_val.columns,
        index=X_fold_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # 訓練
    model.fit(X_fold_train_scaled, y_fold_train)

    # Validation予測
    val_pred = model.predict(X_fold_val_scaled)
    oof_predictions[val_idx] = val_pred

    # Fold精度
    fold_acc = accuracy_score(y_fold_val, val_pred)
    fold_scores.append(fold_acc)
    print(f"Fold {fold} Accuracy: {fold_acc:.4f}")

    # Test予測
    test_pred = model.predict(X_test_scaled)
    test_predictions_list.append(test_pred)

# OOF Score
oof_score = accuracy_score(y_train, oof_predictions)

print("\n" + "=" * 80)
print("結果サマリー")
print("=" * 80)

print(f"\n各Foldの精度:")
for fold, score in enumerate(fold_scores, 1):
    print(f"  Fold {fold}: {score:.4f}")

print(f"\nCV平均: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"OOF Score: {oof_score:.4f}")

# 過去のベストとの比較
print(f"\n--- 過去のベストとの比較 ---")
print(f"4th提出 Soft Voting: OOF 0.8249, LB 0.78468")
print(f"3rd提出 Hard Voting: OOF 0.8294, LB 0.77751")
print(f"SVM_RBF（新）: OOF {oof_score:.4f}, LB ?")

print(f"\n⚠️ 注意: OOFが高い ≠ LBが高い")
print(f"   Hard Voting: OOF 0.8294 → LB 0.77751")
print(f"   Soft Voting: OOF 0.8249 → LB 0.78468（より低いOOFで高いLB）")

# Test予測（多数決）
test_predictions_array = np.array(test_predictions_list)
final_test_predictions = np.round(test_predictions_array.mean(axis=0)).astype(int)

# 提出ファイル作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': final_test_predictions
})

output_file = '/home/user/julis_titanic/submissions/submission_svm_rbf.csv'
submission.to_csv(output_file, index=False)

print(f"\n✅ 提出ファイル作成: {output_file}")
print(f"OOF Score: {oof_score:.4f}")

print("\n" + "=" * 80)
print("推奨")
print("=" * 80)

print("\n⚠️ このファイルは参考用です")
print("5th提出の推奨: 4th提出ファイル再提出（submission_4th_soft_voting_LB_0.78468.csv）")
print("理由:")
print("  1. OOFが高くてもLBが低い前例あり")
print("  2. SVM単体 < アンサンブル（理論的優位性）")
print("  3. 最後の1回で改悪を避けるべき")
print("\nSVM_RBFを試す場合: リスク中～高を承知の上で")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
