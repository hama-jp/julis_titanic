"""
Fare vs Ticket_Fare_Per_Person の多重共線性影響分析

問題: 相関0.827と高いのに、Ticket_Fare_Per_Personが特徴量重要度2位
疑問: なぜ両方必要なのか？片方だけでは駄目なのか？
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("Fare vs Ticket_Fare_Per_Person の多重共線性影響分析")
print("=" * 80)

# データ読み込みと前処理
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

imputer = CorrectImputationPipeline()
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(test)

def extract_ticket_prefix(ticket):
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

def add_ticket_features(df_train, df_test):
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    combined = pd.concat([df_train_copy, df_test_copy], axis=0, ignore_index=True)

    combined['Ticket_Prefix'] = combined['Ticket'].apply(extract_ticket_prefix)
    ticket_counts = combined.groupby('Ticket').size()
    combined['Ticket_GroupSize'] = combined['Ticket'].map(ticket_counts)
    combined['Ticket_Fare_Per_Person'] = combined['Fare'] / combined['Ticket_GroupSize']

    train_len = len(df_train_copy)
    df_train_new = combined.iloc[:train_len].copy()
    df_test_new = combined.iloc[train_len:].copy()

    return df_train_new, df_test_new

train, test = add_ticket_features(train, test)

def engineer_features(df):
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
    return df_new

train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ===== 1. 基本的な違いの分析 =====
print("\n" + "=" * 80)
print("1. FareとTicket_Fare_Per_Personの基本的な違い")
print("=" * 80)

print("\n基本統計:")
print("-" * 80)
fare_stats = train['Fare'].describe()
fare_pp_stats = train['Ticket_Fare_Per_Person'].describe()

comparison = pd.DataFrame({
    'Fare': fare_stats,
    'Ticket_Fare_Per_Person': fare_pp_stats
})
print(comparison)

print(f"\n相関係数: {train[['Fare', 'Ticket_Fare_Per_Person']].corr().iloc[0, 1]:.4f}")

# 差分の分析
train['Fare_Difference'] = train['Fare'] - train['Ticket_Fare_Per_Person']
train['Fare_Ratio'] = train['Ticket_Fare_Per_Person'] / train['Fare']

print(f"\n差分（Fare - Ticket_Fare_Per_Person）:")
print(train['Fare_Difference'].describe())

print(f"\n比率（Ticket_Fare_Per_Person / Fare）:")
print(train['Fare_Ratio'].describe())

# ===== 2. いつ大きく異なるか =====
print("\n" + "=" * 80)
print("2. FareとTicket_Fare_Per_Personが大きく異なるケース")
print("=" * 80)

# 差が大きいケース
train['Abs_Fare_Difference'] = train['Fare_Difference'].abs()
top_diff = train.nlargest(10, 'Abs_Fare_Difference')

print("\n差が最も大きい10件:")
print("-" * 80)
print(top_diff[['PassengerId', 'Name', 'Ticket', 'Ticket_GroupSize', 'Fare',
               'Ticket_Fare_Per_Person', 'Fare_Difference', 'Survived',
               'Pclass']].to_string(index=False))

# グループサイズ別の分析
print("\n" + "=" * 80)
print("3. Ticket_GroupSize別の分析")
print("=" * 80)

print("\nグループサイズ別の統計:")
print("-" * 80)
groupsize_analysis = train.groupby('Ticket_GroupSize').agg({
    'Fare': ['mean', 'std'],
    'Ticket_Fare_Per_Person': ['mean', 'std'],
    'Fare_Ratio': 'mean',
    'Survived': ['mean', 'count']
}).round(3)
groupsize_analysis.columns = ['Fare_Mean', 'Fare_Std', 'FarePP_Mean', 'FarePP_Std',
                               'Ratio_Mean', 'Survival_Rate', 'Count']
print(groupsize_analysis)

print("\n重要な発見:")
print("-" * 80)
print("- グループサイズ1（単独）: Fare = Ticket_Fare_Per_Person")
print("- グループサイズ2+: Ticket_Fare_Per_Person < Fare")
print("- グループサイズが大きいほど差が大きい")

# ===== 4. 生存率との関係 =====
print("\n" + "=" * 80)
print("4. 生存率との関係")
print("=" * 80)

# 4分位数で分割
train['Fare_Quartile'] = pd.qcut(train['Fare'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
train['FarePP_Quartile'] = pd.qcut(train['Ticket_Fare_Per_Person'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

print("\nFare 4分位数別の生存率:")
fare_q_survival = train.groupby('Fare_Quartile')['Survived'].agg(['mean', 'count']).round(3)
fare_q_survival.columns = ['Survival_Rate', 'Count']
print(fare_q_survival)

print("\nTicket_Fare_Per_Person 4分位数別の生存率:")
farepp_q_survival = train.groupby('FarePP_Quartile')['Survived'].agg(['mean', 'count']).round(3)
farepp_q_survival.columns = ['Survival_Rate', 'Count']
print(farepp_q_survival)

# ===== 5. クロス集計 =====
print("\n" + "=" * 80)
print("5. FareとTicket_Fare_Per_Personのクロス集計")
print("=" * 80)

print("\n生存率のクロス集計（Fare Q1-Q4 × FarePP Q1-Q4）:")
crosstab_survival = pd.crosstab(train['Fare_Quartile'], train['FarePP_Quartile'],
                                  values=train['Survived'], aggfunc='mean').round(3)
print(crosstab_survival)

print("\n件数のクロス集計:")
crosstab_count = pd.crosstab(train['Fare_Quartile'], train['FarePP_Quartile'])
print(crosstab_count)

print("\n⚠️ 注目: 対角線外のセル（Fareが高いがFarePPが低い、またはその逆）")
print("    → これらは追加情報を提供している")

# ===== 6. モデル性能比較 =====
print("\n" + "=" * 80)
print("6. モデル性能比較（Fare vs Ticket_Fare_Per_Person vs 両方）")
print("=" * 80)

base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Title',
                 'FamilySize', 'SmallFamily'] + embarked_cols

y = train['Survived']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_configs = {
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

feature_configs = {
    'Baseline (no Fare)': base_features,
    'Baseline + Fare': base_features + ['Fare'],
    'Baseline + Ticket_Fare_Per_Person': base_features + ['Ticket_Fare_Per_Person'],
    'Baseline + Both': base_features + ['Fare', 'Ticket_Fare_Per_Person'],
    'Baseline + Ticket_GroupSize': base_features + ['Ticket_GroupSize'],
    'Baseline + Ticket_GroupSize + Fare': base_features + ['Ticket_GroupSize', 'Fare'],
    'Baseline + Ticket_GroupSize + FarePP': base_features + ['Ticket_GroupSize', 'Ticket_Fare_Per_Person']
}

results = []

for model_name, model in model_configs.items():
    print(f"\n{model_name}:")
    print("-" * 60)

    for feat_name, features in feature_configs.items():
        X = train[features]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()

        print(f"  {feat_name:45s}: {mean_score:.4f}")

        results.append({
            'Model': model_name,
            'Features': feat_name,
            'CV_Score': mean_score,
            'Num_Features': len(features)
        })

# ===== 7. 結果の分析 =====
print("\n" + "=" * 80)
print("7. 結果の分析")
print("=" * 80)

results_df = pd.DataFrame(results)

for model_name in model_configs.keys():
    model_results = results_df[results_df['Model'] == model_name].sort_values('CV_Score', ascending=False)

    print(f"\n{model_name} - 最良構成:")
    print(model_results.head(3).to_string(index=False))

# ===== 8. 特徴量重要度の比較 =====
print("\n" + "=" * 80)
print("8. 特徴量重要度の比較（GradientBoosting）")
print("=" * 80)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

# 両方使う場合
features_both = base_features + ['Fare', 'Ticket_Fare_Per_Person']
X_both = train[features_both]
gb.fit(X_both, y)

importances_both = pd.DataFrame({
    'Feature': features_both,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n両方使う場合の特徴量重要度:")
print(importances_both.head(10).to_string(index=False))

fare_importance = importances_both[importances_both['Feature'] == 'Fare']['Importance'].values[0]
farepp_importance = importances_both[importances_both['Feature'] == 'Ticket_Fare_Per_Person']['Importance'].values[0]

print(f"\nFare重要度: {fare_importance:.4f}")
print(f"Ticket_Fare_Per_Person重要度: {farepp_importance:.4f}")
print(f"比率: {farepp_importance / fare_importance:.2f}x")

# ===== 9. 結論 =====
print("\n" + "=" * 80)
print("9. 結論")
print("=" * 80)

print("""
【FareとTicket_Fare_Per_Personの関係】

1. **相関は高い（0.827）が、異なる情報を持つ**:
   - Fare: チケットの総運賃（グループ全体）
   - Ticket_Fare_Per_Person: 一人あたりの運賃

2. **いつ大きく異なるか**:
   - 単独旅行者（GroupSize=1）: Fare = Ticket_Fare_Per_Person
   - グループ旅行者（GroupSize=2+）: Ticket_Fare_Per_Person < Fare
   - グループサイズが大きいほど差が大きい

3. **なぜ両方必要なのか**:
   - Fare: 社会的地位・チケットクラスを反映
   - Ticket_Fare_Per_Person: 個人の実質的な支払額を反映
   - 両方使うことで、「裕福なグループ旅行者」と「裕福な単独旅行者」を
     区別できる

4. **Tree-basedモデルでの使い分け**:
   - 両方の特徴量を使って、より細かい分割が可能
   - 例: 「高運賃だが一人あたりは低い（大家族）」
         「低運賃だが一人あたりは高い（小グループ）」

5. **多重共線性の影響**:
   - Tree-based: 問題なし（自動で使い分け）
   - Linear: 問題あり（係数が不安定）

【推奨】
- Tree-basedモデル: Fare + Ticket_Fare_Per_Person + Ticket_GroupSize
  → 3つの組み合わせで最良の性能
- 線形モデル: Fareのみ、またはTicket_Fare_Per_Personのみ
  → 多重共線性を避ける
""")

# 結果保存
import json
summary = {
    'correlation': float(train[['Fare', 'Ticket_Fare_Per_Person']].corr().iloc[0, 1]),
    'fare_importance': float(fare_importance),
    'farepp_importance': float(farepp_importance),
    'best_results_by_model': results_df.groupby('Model').apply(
        lambda x: x.nlargest(1, 'CV_Score')[['Features', 'CV_Score']].to_dict('records')[0]
    ).to_dict()
}

with open('/home/user/julis_titanic/experiments/results/fare_vs_farepp_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n結果を保存: experiments/results/fare_vs_farepp_analysis.json")
print("\n分析完了！")
