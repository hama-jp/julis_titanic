"""
Sex×Pclass交互作用特徴量の評価

目的:
- 9特徴量（ベースライン: OOF 0.8384, LB 0.77751）に
  Sex×Pclass交互作用を追加して効果を検証

理論的根拠:
- 1等女性: 生存率 ~96%
- 3等男性: 生存率 ~14%
- 性別と客室クラスの組み合わせは強いシグナル

評価項目:
1. OOFスコア改善（ベースライン比）
2. VIF（多重共線性チェック）
3. Kaggle LB提出用ファイル作成
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("Sex×Pclass交互作用特徴量の評価")
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

def add_sex_pclass_interaction(df):
    """Sex×Pclass交互作用追加"""
    df_new = df.copy()

    # Sex×Pclassの組み合わせを作成
    # 0=Male, 1=Female なので、文字列に変換
    sex_label = df_new['Sex'].map({0: 'Male', 1: 'Female'})
    df_new['Sex_Pclass'] = sex_label + '_' + df_new['Pclass'].astype(str)

    # ダミー変数化（drop_first=Trueで1つ削除して多重共線性回避）
    sex_pclass_dummies = pd.get_dummies(df_new['Sex_Pclass'], prefix='SexPclass', drop_first=True)
    df_new = pd.concat([df_new, sex_pclass_dummies], axis=1)

    # 元のSex_Pclassカラムは削除
    df_new = df_new.drop('Sex_Pclass', axis=1)

    return df_new

def preprocess_data(df):
    """前処理"""
    df_new = df.copy()

    # 欠損値処理
    df_new['Age'] = df_new['Age'].fillna(df_new['Age'].median())
    df_new['Fare'] = df_new['Fare'].fillna(df_new['Fare'].median())
    df_new['Embarked'] = df_new['Embarked'].fillna('S')

    # Sexを数値化（交互作用追加前に実施）
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})

    return df_new

def calculate_vif(X, feature_names):
    """VIF（分散拡大係数）を計算"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_values = []

    for i in range(len(feature_names)):
        try:
            vif = variance_inflation_factor(X.values, i)
            if np.isinf(vif) or np.isnan(vif):
                vif = 999999  # inf/nanは大きな値に置き換え
            vif_values.append(vif)
        except:
            vif_values.append(999999)

    vif_data["VIF"] = vif_values
    return vif_data.sort_values('VIF', ascending=False)

# 特徴量追加
print("\n特徴量エンジニアリング...")
train = preprocess_data(train)
train = add_title_feature(train)
train = add_familysize_feature(train)
train = add_sex_pclass_interaction(train)  # Sex×Pclass交互作用追加

test = preprocess_data(test)
test = add_title_feature(test)
test = add_familysize_feature(test)
test = add_sex_pclass_interaction(test)  # Sex×Pclass交互作用追加

# Embarkedダミー変数化
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# ベース9特徴量
base_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']

# Embarkedダミー変数を追加
embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
base_features.extend(embarked_cols)

# Sex×Pclass交互作用ダミー変数を追加
sex_pclass_cols = [col for col in train.columns if col.startswith('SexPclass_')]
interaction_features = base_features + sex_pclass_cols

print(f"\nベース特徴量（{len(base_features)}個）:")
for i, col in enumerate(base_features, 1):
    print(f"  {i}. {col}")

print(f"\nSex×Pclass交互作用特徴量（{len(sex_pclass_cols)}個）:")
for i, col in enumerate(sex_pclass_cols, 1):
    print(f"  {i}. {col}")

print(f"\n合計特徴量数: {len(interaction_features)}個")

X_train = train[interaction_features]
y_train = train['Survived']
X_test = test[interaction_features]

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# VIF計算（多重共線性チェック）
print("\n" + "=" * 80)
print("VIF（多重共線性）評価")
print("=" * 80)

vif_df = calculate_vif(X_train, interaction_features)
print("\nVIF結果（上位10件）:")
print(vif_df.head(10).to_string(index=False))

max_vif = vif_df['VIF'].max()
problematic_features = vif_df[vif_df['VIF'] >= 10]

print(f"\n最大VIF: {max_vif:.2f}")
if len(problematic_features) > 0:
    print(f"⚠️ VIF≥10の特徴量: {len(problematic_features)}個")
    print(problematic_features.to_string(index=False))
else:
    print("✅ すべての特徴量でVIF < 10（多重共線性なし）")

# モデル定義
print("\n" + "=" * 80)
print("モデル訓練")
print("=" * 80)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

# GradientBoosting
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

# RandomForest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Hard Voting Ensemble
hard_voting = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    voting='hard'
)

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

    # Test予測
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

# ベースライン（9特徴量）との比較
baseline_oof = 0.8384
oof_improvement = oof_score - baseline_oof

print(f"\n--- ベースライン比較 ---")
print(f"ベースライン（9特徴）: {baseline_oof:.4f}")
print(f"Sex×Pclass追加: {oof_score:.4f}")
print(f"OOF改善: {oof_improvement:+.4f} ({oof_improvement*100:+.2f}%)")

if oof_improvement > 0:
    print("✅ OOFスコア改善")
else:
    print("❌ OOFスコア悪化")

# 期待LB予測（Gap 6.09%を仮定）
expected_gap = 0.0609
expected_lb = oof_score - expected_gap
baseline_lb = 0.77751

print(f"\n--- LB予測（Gap 6.09%仮定） ---")
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

output_file = 'submission_sex_pclass_interaction.csv'
submission.to_csv(output_file, index=False)
print(f"✅ 提出ファイル作成: {output_file}")

# Sex×Pclass組み合わせごとの生存率を表示
print("\n" + "=" * 80)
print("Sex×Pclass組み合わせごとの生存率（Trainデータ）")
print("=" * 80)

train_original = pd.read_csv('data/raw/train.csv')
survival_by_sex_pclass = train_original.groupby(['Sex', 'Pclass'])['Survived'].agg([
    ('生存率', lambda x: f"{x.mean():.1%}"),
    ('サンプル数', 'count'),
    ('生存者数', 'sum')
])
print(survival_by_sex_pclass)

print("\n" + "=" * 80)
print("評価結果サマリー")
print("=" * 80)

print(f"\n1. OOFスコア: {oof_score:.4f} ({oof_improvement:+.4f})")
print(f"2. 最大VIF: {max_vif:.2f} ({'✅ OK' if max_vif < 10 else '⚠️ 要注意'})")
print(f"3. 期待LB: {expected_lb:.5f} ({expected_lb - baseline_lb:+.5f})")

print("\n次のアクション:")
if oof_improvement > 0 and max_vif < 10:
    print(f"✅ OOF改善 & VIF良好 → Kaggle LBで検証推奨")
    print(f"   提出ファイル: {output_file}")
elif oof_improvement > 0 and max_vif >= 10:
    print(f"⚠️ OOF改善したがVIF問題あり → 慎重にLB検証")
elif oof_improvement <= 0:
    print(f"❌ OOF改善なし → この特徴量は不採用")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)

# 結果をファイルに保存
summary_file = 'results/sex_pclass_interaction_evaluation.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Sex×Pclass交互作用特徴量の評価結果\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"使用した特徴量（{len(interaction_features)}個）:\n")
    f.write("\nベース特徴量:\n")
    for i, col in enumerate(base_features, 1):
        f.write(f"  {i}. {col}\n")
    f.write("\nSex×Pclass交互作用:\n")
    for i, col in enumerate(sex_pclass_cols, 1):
        f.write(f"  {i}. {col}\n")

    f.write(f"\n--- OOFスコア ---\n")
    f.write(f"ベースライン（9特徴）: {baseline_oof:.4f}\n")
    f.write(f"Sex×Pclass追加: {oof_score:.4f}\n")
    f.write(f"改善: {oof_improvement:+.4f} ({oof_improvement*100:+.2f}%)\n")

    f.write(f"\n--- VIF（多重共線性） ---\n")
    f.write(f"最大VIF: {max_vif:.2f}\n")
    f.write(f"\nVIF上位10件:\n")
    f.write(vif_df.head(10).to_string(index=False))

    f.write(f"\n\n--- 期待LB（Gap 6.09%仮定） ---\n")
    f.write(f"期待LB Score: {expected_lb:.5f}\n")
    f.write(f"ベースラインLB: {baseline_lb:.5f}\n")
    f.write(f"予想改善: {expected_lb - baseline_lb:+.5f}\n")

print(f"\n詳細結果を保存: {summary_file}")
