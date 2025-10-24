"""
Age × Pclass 交互作用特徴量の評価

目的:
- ブースティング決定木系以外のモデルで、Age × Pclass 交互作用が効果的か検証
- 対象モデル: ロジスティック回帰、RandomForest、SVM
- ブースティング系（LightGBM, GradientBoosting）は自動的に交互作用を学習できるため除外

理論的根拠:
- 子供（Age < 18）の1等客: 生存率 ~100%
- 子供（Age < 18）の3等客: 生存率 ~30-40%
- Age × Pclass の組み合わせは強いシグナル
- ロジスティック回帰やSVMは明示的に交互作用項が必要

評価項目:
1. OOFスコア改善（ベースライン比）
2. VIF（多重共線性チェック）
3. モデル別の効果比較
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
import warnings
warnings.filterwarnings('ignore')

# VIFは参考情報なので、statsmodelsなしでもスキップ可能
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️ statsmodelsが利用できません。VIF計算をスキップします。")

# 正しい補完パイプラインをインポート
sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("Age × Pclass 交互作用特徴量の評価（ブースティング以外）")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 正しい補完パイプラインで前処理
print("\n正しい補完パイプラインで前処理...")
imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

print("✅ Train/Test分離を維持した補完完了")

def engineer_base_features(df):
    """ベース特徴量エンジニアリング（9特徴量）"""
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

    # SmallFamily (2-4人の家族)
    df_new['SmallFamily'] = ((df_new['FamilySize'] >= 2) & (df_new['FamilySize'] <= 4)).astype(int)

    return df_new

def add_age_pclass_interaction(df):
    """Age × Pclass 交互作用を追加"""
    df_new = df.copy()

    # 連続値 × カテゴリの交互作用
    # Age × Pclass_1, Age × Pclass_2, Age × Pclass_3
    df_new['Age_Pclass_1'] = df_new['Age'] * (df_new['Pclass'] == 1).astype(int)
    df_new['Age_Pclass_2'] = df_new['Age'] * (df_new['Pclass'] == 2).astype(int)
    df_new['Age_Pclass_3'] = df_new['Age'] * (df_new['Pclass'] == 3).astype(int)

    return df_new

# 特徴量エンジニアリング
print("\n特徴量エンジニアリング...")
train = engineer_base_features(train)
test = engineer_base_features(test)

# Embarkedダミー変数化
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# テスト列の一致確認
embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ベース特徴量（9個）
base_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title',
                 'FamilySize', 'SmallFamily'] + embarked_cols

print(f"\nベース特徴量（{len(base_features)}個）:")
for i, col in enumerate(base_features, 1):
    print(f"  {i}. {col}")

# Age × Pclass 交互作用を追加
train_interaction = add_age_pclass_interaction(train)
test_interaction = add_age_pclass_interaction(test)

interaction_cols = ['Age_Pclass_1', 'Age_Pclass_2', 'Age_Pclass_3']
interaction_features = base_features + interaction_cols

print(f"\nAge × Pclass 交互作用特徴量（{len(interaction_cols)}個）:")
for i, col in enumerate(interaction_cols, 1):
    print(f"  {i}. {col}")

print(f"\n合計特徴量数（交互作用あり）: {len(interaction_features)}個")

# データ準備
X_train_base = train[base_features]
X_train_interaction = train_interaction[interaction_features]
y_train = train['Survived']

X_test_base = test[base_features]
X_test_interaction = test_interaction[interaction_features]

print(f"\nX_train_base shape: {X_train_base.shape}")
print(f"X_train_interaction shape: {X_train_interaction.shape}")

# VIF計算（多重共線性チェック）
vif_df = None
max_vif = None

if HAS_STATSMODELS:
    print("\n" + "=" * 80)
    print("VIF（多重共線性）評価 - Age × Pclass 交互作用")
    print("=" * 80)

    def calculate_vif(X, feature_names):
        """VIF（分散拡大係数）を計算"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = feature_names
        vif_values = []

        for i in range(len(feature_names)):
            try:
                vif = variance_inflation_factor(X.values, i)
                if np.isinf(vif) or np.isnan(vif):
                    vif = 999999
                vif_values.append(vif)
            except:
                vif_values.append(999999)

        vif_data["VIF"] = vif_values
        return vif_data.sort_values('VIF', ascending=False)

    vif_df = calculate_vif(X_train_interaction, interaction_features)
    print("\nVIF結果（上位10件）:")
    print(vif_df.head(10).to_string(index=False))

    max_vif = vif_df['VIF'].max()
    print(f"\n最大VIF: {max_vif:.2f}")
else:
    print("\n⚠️ VIF計算をスキップ（statsmodels未インストール）")

# モデル定義
print("\n" + "=" * 80)
print("モデル評価（ブースティング以外）")
print("=" * 80)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    'SVM_RBF': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
}

# 評価関数
def evaluate_model(model, X_train, y_train, X_test, model_name, use_scaling=False):
    """モデルを評価"""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X_train))
    test_predictions_list = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # スケーリング（ロジスティック回帰、SVMで必要）
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

        # 訓練
        model.fit(X_fold_train, y_fold_train)

        # Validation予測
        val_pred = model.predict(X_fold_val)
        oof_predictions[val_idx] = val_pred

        # Fold精度
        fold_acc = accuracy_score(y_fold_val, val_pred)
        fold_scores.append(fold_acc)

        # Test予測
        if use_scaling:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            test_pred = model.predict(X_test_scaled)
        else:
            test_pred = model.predict(X_test)

        test_predictions_list.append(test_pred)

    # OOF Score計算
    oof_score = accuracy_score(y_train, oof_predictions)
    cv_mean = np.mean(fold_scores)
    cv_std = np.std(fold_scores)

    # Test予測（多数決）
    test_predictions_array = np.array(test_predictions_list)
    final_test_predictions = np.round(test_predictions_array.mean(axis=0)).astype(int)

    return {
        'oof_score': oof_score,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'test_pred': final_test_predictions
    }

# 結果格納
results = {}

print("\n--- ベースライン（交互作用なし）評価 ---\n")

for name, model in models.items():
    print(f"{name}:")
    use_scaling = name in ['LogisticRegression', 'SVM_RBF']

    result = evaluate_model(
        model, X_train_base, y_train, X_test_base,
        name, use_scaling=use_scaling
    )

    results[f'{name}_Base'] = result
    print(f"  OOF Score: {result['oof_score']:.4f}")
    print(f"  CV Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print()

print("\n--- Age × Pclass 交互作用あり評価 ---\n")

for name, model in models.items():
    print(f"{name}:")
    use_scaling = name in ['LogisticRegression', 'SVM_RBF']

    result = evaluate_model(
        model, X_train_interaction, y_train, X_test_interaction,
        name, use_scaling=use_scaling
    )

    results[f'{name}_Interaction'] = result
    print(f"  OOF Score: {result['oof_score']:.4f}")
    print(f"  CV Mean: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print()

# 結果サマリー
print("\n" + "=" * 80)
print("結果サマリー - Age × Pclass 交互作用の効果")
print("=" * 80)

print(f"\n{'モデル':<25s} {'ベース OOF':>12s} {'交互作用 OOF':>12s} {'改善':>10s} {'判定':>8s}")
print("-" * 80)

for model_name in ['LogisticRegression', 'RandomForest', 'SVM_RBF']:
    base_oof = results[f'{model_name}_Base']['oof_score']
    inter_oof = results[f'{model_name}_Interaction']['oof_score']
    improvement = inter_oof - base_oof
    judgment = "✅ 改善" if improvement > 0 else "❌ 悪化" if improvement < 0 else "→ 同じ"

    print(f"{model_name:<25s} {base_oof:>12.4f} {inter_oof:>12.4f} {improvement:>+10.4f} {judgment:>8s}")

# ベストモデルの特定
print("\n" + "=" * 80)
print("ベストモデル")
print("=" * 80)

best_base = max(
    [(name, res['oof_score']) for name, res in results.items() if 'Base' in name],
    key=lambda x: x[1]
)
best_interaction = max(
    [(name, res['oof_score']) for name, res in results.items() if 'Interaction' in name],
    key=lambda x: x[1]
)

print(f"\nベースライン最高: {best_base[0]} (OOF: {best_base[1]:.4f})")
print(f"交互作用最高: {best_interaction[0]} (OOF: {best_interaction[1]:.4f})")

# 提出ファイル生成（交互作用で改善があった場合のみ）
improvements = {
    model_name: results[f'{model_name}_Interaction']['oof_score'] - results[f'{model_name}_Base']['oof_score']
    for model_name in ['LogisticRegression', 'RandomForest', 'SVM_RBF']
}

best_improvement_model = max(improvements.items(), key=lambda x: x[1])

print("\n" + "=" * 80)
print("交互作用の効果分析")
print("=" * 80)

print(f"\n最大改善モデル: {best_improvement_model[0]}")
print(f"改善幅: {best_improvement_model[1]:+.4f}")

if best_improvement_model[1] > 0:
    print(f"\n✅ Age × Pclass 交互作用は {best_improvement_model[0]} で効果あり")

    # 最良モデルの提出ファイル生成
    best_model_key = f"{best_improvement_model[0]}_Interaction"
    test_pred = results[best_model_key]['test_pred']

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred
    })

    output_file = '/home/user/julis_titanic/submissions/submission_age_pclass_interaction.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n提出ファイル作成: {output_file}")
    print(f"モデル: {best_improvement_model[0]}")
    print(f"OOF Score: {results[best_model_key]['oof_score']:.4f}")
else:
    print(f"\n❌ Age × Pclass 交互作用は全モデルで効果なし")
    print("提出ファイルは生成しません")

# 詳細結果をファイルに保存
summary_file = '/home/user/julis_titanic/experiments/results/age_pclass_interaction_evaluation.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Age × Pclass 交互作用特徴量の評価結果（ブースティング以外）\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"使用した特徴量:\n")
    f.write(f"\nベース特徴量（{len(base_features)}個）:\n")
    for i, col in enumerate(base_features, 1):
        f.write(f"  {i}. {col}\n")

    f.write(f"\nAge × Pclass 交互作用（{len(interaction_cols)}個）:\n")
    for i, col in enumerate(interaction_cols, 1):
        f.write(f"  {i}. {col}\n")

    if HAS_STATSMODELS and vif_df is not None:
        f.write(f"\n--- VIF（多重共線性） ---\n")
        f.write(f"最大VIF: {max_vif:.2f}\n")
        f.write(f"\nVIF上位10件:\n")
        f.write(vif_df.head(10).to_string(index=False))
    else:
        f.write(f"\n--- VIF（多重共線性） ---\n")
        f.write("VIF計算スキップ（statsmodels未インストール）\n")

    f.write(f"\n\n--- 結果サマリー ---\n\n")
    f.write(f"{'モデル':<25s} {'ベース OOF':>12s} {'交互作用 OOF':>12s} {'改善':>10s}\n")
    f.write("-" * 80 + "\n")

    for model_name in ['LogisticRegression', 'RandomForest', 'SVM_RBF']:
        base_oof = results[f'{model_name}_Base']['oof_score']
        inter_oof = results[f'{model_name}_Interaction']['oof_score']
        improvement = inter_oof - base_oof

        f.write(f"{model_name:<25s} {base_oof:>12.4f} {inter_oof:>12.4f} {improvement:>+10.4f}\n")

    f.write(f"\n--- 結論 ---\n")
    if best_improvement_model[1] > 0:
        f.write(f"✅ Age × Pclass 交互作用は {best_improvement_model[0]} で効果あり\n")
        f.write(f"改善幅: {best_improvement_model[1]:+.4f}\n")
    else:
        f.write(f"❌ Age × Pclass 交互作用は効果なし\n")

print(f"\n詳細結果を保存: {summary_file}")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
