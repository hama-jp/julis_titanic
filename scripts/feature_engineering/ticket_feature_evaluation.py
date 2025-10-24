"""
Ticket特徴量の実装と評価

発見された有効なTicketパターンを特徴量として実装し、
既存モデルと比較して効果を検証する
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import json
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("Ticket特徴量の実装と評価")
print("=" * 80)

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# 正しい補完パイプライン
print("\n補完パイプライン実行...")
imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

# ===== Ticket特徴量の実装 =====
print("\n" + "=" * 80)
print("Ticket特徴量の実装")
print("=" * 80)

def extract_ticket_prefix(ticket):
    """チケットのプレフィックス（文字部分）を抽出"""
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

def extract_ticket_number(ticket):
    """チケットの数値部分を抽出"""
    ticket = str(ticket).strip()
    match = re.search(r'(\d+)', ticket)
    if match:
        return int(match.group(1))
    return 0

def add_ticket_features(df_train, df_test):
    """Ticket特徴量を追加"""
    # Train + Test を結合してグループサイズを計算
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()

    combined = pd.concat([df_train_copy, df_test_copy], axis=0, ignore_index=True)

    # 基本的なTicket特徴量
    combined['Ticket_Prefix'] = combined['Ticket'].apply(extract_ticket_prefix)
    combined['Ticket_Number'] = combined['Ticket'].apply(extract_ticket_number)
    combined['Ticket_Length'] = combined['Ticket'].astype(str).str.len()

    # Ticketグループサイズ（全データで計算）
    ticket_counts = combined.groupby('Ticket').size()
    combined['Ticket_GroupSize'] = combined['Ticket'].map(ticket_counts)

    # カテゴリ化
    # 分析結果から
    high_survival_prefixes = ['F_C_C', 'PC']
    low_survival_prefixes = ['SOTON_OQ', 'SOTON_O_Q', 'W_C', 'A', 'CA', 'S_O_C']

    def categorize_prefix(prefix):
        if prefix in high_survival_prefixes:
            return 2  # HIGH
        elif prefix in low_survival_prefixes:
            return 0  # LOW
        else:
            return 1  # MEDIUM

    combined['Ticket_Prefix_Category'] = combined['Ticket_Prefix'].apply(categorize_prefix)

    # 数値のみのチケットかどうか
    combined['Ticket_Is_Numeric'] = (combined['Ticket_Prefix'] == 'NUMERIC').astype(int)

    # 一人あたりの運賃（Fare per person）
    combined['Ticket_Fare_Per_Person'] = combined['Fare'] / combined['Ticket_GroupSize']

    # 小グループ（2-4人）フラグ
    combined['Ticket_SmallGroup'] = ((combined['Ticket_GroupSize'] >= 2) &
                                      (combined['Ticket_GroupSize'] <= 4)).astype(int)

    # Train と Test に分割
    train_len = len(df_train_copy)
    df_train_new = combined.iloc[:train_len].copy()
    df_test_new = combined.iloc[train_len:].copy()

    return df_train_new, df_test_new

print("Ticket特徴量を追加中...")
train, test = add_ticket_features(train, test)

print("\n追加されたTicket特徴量:")
ticket_features = ['Ticket_Prefix', 'Ticket_GroupSize', 'Ticket_Length',
                   'Ticket_Prefix_Category', 'Ticket_Is_Numeric',
                   'Ticket_Fare_Per_Person', 'Ticket_SmallGroup']
print(ticket_features)

print("\nサンプルデータ:")
print(train[['Ticket'] + ticket_features].head(10))

# ===== 基本特徴量エンジニアリング =====
print("\n" + "=" * 80)
print("基本特徴量エンジニアリング")
print("=" * 80)

def engineer_features(df):
    """基本的な特徴量エンジニアリング"""
    df_new = df.copy()
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
    df_new['SmallFamily'] = ((df_new['FamilySize'] >= 2) & (df_new['FamilySize'] <= 4)).astype(int)

    # AgeGroup
    bins = [0, 18, 60, 100]
    labels = [0, 1, 2]
    df_new['AgeGroup'] = pd.cut(df_new['Age'], bins=bins, labels=labels, include_lowest=True)
    df_new['AgeGroup'] = df_new['AgeGroup'].astype(int)

    return df_new

train = engineer_features(train)
test = engineer_features(test)

# Embarked
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ===== モデル評価 =====
print("\n" + "=" * 80)
print("モデル評価（Ticket特徴量あり vs なし）")
print("=" * 80)

# ベースライン特徴量（既存の9特徴量）
base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title',
                 'FamilySize', 'SmallFamily'] + embarked_cols

# Ticket特徴量（数値のみ - カテゴリカルは除外）
ticket_numeric_features = ['Ticket_GroupSize', 'Ticket_Length', 'Ticket_Prefix_Category',
                           'Ticket_Is_Numeric', 'Ticket_Fare_Per_Person', 'Ticket_SmallGroup']

# 3つのセットで評価
feature_sets = {
    'Baseline (9 features)': base_features,
    'Baseline + All Ticket': base_features + ticket_numeric_features,
    'Baseline + Top3 Ticket': base_features + ['Ticket_GroupSize', 'Ticket_Fare_Per_Person', 'Ticket_SmallGroup']
}

X = train.copy()
y = train['Survived']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# モデル定義
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
}

results = []

for feature_name, features in feature_sets.items():
    print(f"\n{'=' * 60}")
    print(f"Feature Set: {feature_name}")
    print(f"Features ({len(features)}): {features}")
    print(f"{'=' * 60}")

    X_features = X[features]

    for model_name, model in models.items():
        scores = cross_val_score(model, X_features, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()

        print(f"{model_name:20s}: {mean_score:.4f} (+/- {std_score:.4f})")

        results.append({
            'Feature_Set': feature_name,
            'Model': model_name,
            'CV_Mean': mean_score,
            'CV_Std': std_score,
            'Num_Features': len(features)
        })

# ===== 結果の集計 =====
print("\n" + "=" * 80)
print("結果の集計")
print("=" * 80)

results_df = pd.DataFrame(results)

# ベースラインとの比較
baseline_results = results_df[results_df['Feature_Set'] == 'Baseline (9 features)'].set_index('Model')

print("\nベースラインとの改善率:")
print("-" * 60)
for feature_set in ['Baseline + All Ticket', 'Baseline + Top3 Ticket']:
    print(f"\n{feature_set}:")
    feature_results = results_df[results_df['Feature_Set'] == feature_set].set_index('Model')

    for model in models.keys():
        baseline_score = baseline_results.loc[model, 'CV_Mean']
        new_score = feature_results.loc[model, 'CV_Mean']
        improvement = new_score - baseline_score
        improvement_pct = (improvement / baseline_score) * 100

        print(f"  {model:20s}: {baseline_score:.4f} -> {new_score:.4f} "
              f"({improvement:+.4f}, {improvement_pct:+.2f}%)")

# ===== 最良モデルの詳細評価 =====
print("\n" + "=" * 80)
print("最良モデルの詳細評価")
print("=" * 80)

best_idx = results_df['CV_Mean'].idxmax()
best_result = results_df.iloc[best_idx]

print(f"\n最良結果:")
print(f"  Feature Set: {best_result['Feature_Set']}")
print(f"  Model: {best_result['Model']}")
print(f"  CV Score: {best_result['CV_Mean']:.4f} (+/- {best_result['CV_Std']:.4f})")
print(f"  Num Features: {best_result['Num_Features']}")

best_features = feature_sets[best_result['Feature_Set']]
best_model = models[best_result['Model']]

# 特徴量重要度
if hasattr(best_model, 'feature_importances_'):
    X_best = X[best_features]
    best_model.fit(X_best, y)

    importances = pd.DataFrame({
        'Feature': best_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n特徴量重要度（Top 15）:")
    print(importances.head(15).to_string(index=False))

# ===== 結果の保存 =====
print("\n" + "=" * 80)
print("結果の保存")
print("=" * 80)

results_df.to_csv('/home/user/julis_titanic/experiments/results/ticket_feature_evaluation.csv', index=False)
print("評価結果を保存: experiments/results/ticket_feature_evaluation.csv")

# Summary
summary = {
    'best_feature_set': best_result['Feature_Set'],
    'best_model': best_result['Model'],
    'best_cv_score': float(best_result['CV_Mean']),
    'best_cv_std': float(best_result['CV_Std']),
    'baseline_score': float(baseline_results.loc[best_result['Model'], 'CV_Mean']),
    'improvement': float(best_result['CV_Mean'] - baseline_results.loc[best_result['Model'], 'CV_Mean']),
    'ticket_features': ticket_numeric_features,
    'recommended_features': best_features
}

with open('/home/user/julis_titanic/experiments/results/ticket_feature_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("サマリーを保存: experiments/results/ticket_feature_summary.json")

print("\n" + "=" * 80)
print("評価完了！")
print("=" * 80)

# 最終サマリー
print("\n【結論】")
if summary['improvement'] > 0:
    print(f"✓ Ticket特徴量により精度が改善: {summary['improvement']:+.4f} ({summary['improvement']/summary['baseline_score']*100:+.2f}%)")
    print(f"✓ 推奨特徴量セット: {best_result['Feature_Set']}")
    print(f"✓ 最良モデル: {best_result['Model']}")
else:
    print(f"✗ Ticket特徴量は有意な改善をもたらしませんでした")
    print(f"  変化: {summary['improvement']:+.4f}")
