"""
SibSp/Parch/FamilySize vs Ticket_GroupSize 比較分析

仮説: Ticket_GroupSizeだけで十分？SibSp/Parchは冗長？
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("SibSp/Parch/FamilySize vs Ticket_GroupSize 比較分析")
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

    high_survival_prefixes = ['F_C_C', 'PC']
    low_survival_prefixes = ['SOTON_OQ', 'SOTON_O_Q', 'W_C', 'A', 'CA', 'S_O_C']

    def categorize_prefix(prefix):
        if prefix in high_survival_prefixes:
            return 2
        elif prefix in low_survival_prefixes:
            return 0
        else:
            return 1

    combined['Ticket_Prefix_Category'] = combined['Ticket_Prefix'].apply(categorize_prefix)
    combined['Ticket_Is_Numeric'] = (combined['Ticket_Prefix'] == 'NUMERIC').astype(int)
    combined['Ticket_SmallGroup'] = ((combined['Ticket_GroupSize'] >= 2) &
                                      (combined['Ticket_GroupSize'] <= 4)).astype(int)

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

# ===== 1. 基本的な関係の分析 =====
print("\n" + "=" * 80)
print("1. SibSp/Parch/FamilySize と Ticket_GroupSize の関係")
print("=" * 80)

print("\n基本統計:")
print("-" * 80)
family_features = ['SibSp', 'Parch', 'FamilySize', 'Ticket_GroupSize']
print(train[family_features].describe())

print("\n相関行列:")
print(train[family_features].corr().round(3))

# ===== 2. 一致度の分析 =====
print("\n" + "=" * 80)
print("2. FamilySize と Ticket_GroupSize の一致度")
print("=" * 80)

train['Family_Ticket_Match'] = (train['FamilySize'] == train['Ticket_GroupSize']).astype(int)
train['Family_Ticket_Diff'] = train['FamilySize'] - train['Ticket_GroupSize']

match_rate = train['Family_Ticket_Match'].mean()
print(f"\n一致率: {match_rate:.2%} ({train['Family_Ticket_Match'].sum()}/{len(train)})")

print(f"\n差分の分布:")
print(train['Family_Ticket_Diff'].value_counts().sort_index())

print(f"\n不一致のケース（差が大きい順、上位15件）:")
mismatch = train[train['Family_Ticket_Match'] == 0].copy()
mismatch['Abs_Diff'] = mismatch['Family_Ticket_Diff'].abs()
mismatch = mismatch.nlargest(15, 'Abs_Diff')
print(mismatch[['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'FamilySize',
               'Ticket_GroupSize', 'Family_Ticket_Diff', 'Survived']].to_string(index=False))

# ===== 3. いつ異なるか？ =====
print("\n" + "=" * 80)
print("3. FamilySize と Ticket_GroupSize が異なるパターン")
print("=" * 80)

print("\nパターン1: FamilySize < Ticket_GroupSize（家族以外と同じチケット）")
pattern1 = train[train['Family_Ticket_Diff'] < 0].copy()
pattern1['Abs_Diff'] = pattern1['Family_Ticket_Diff'].abs()
print(f"件数: {len(pattern1)} ({len(pattern1)/len(train)*100:.1f}%)")
print(f"生存率: {pattern1['Survived'].mean():.3f}")
print("\n例:")
print(pattern1.nlargest(5, 'Abs_Diff')[['Name', 'Ticket', 'SibSp', 'Parch',
      'FamilySize', 'Ticket_GroupSize', 'Survived']].to_string(index=False))

print("\nパターン2: FamilySize > Ticket_GroupSize（家族が複数チケットに分散）")
pattern2 = train[train['Family_Ticket_Diff'] > 0].copy()
pattern2['Abs_Diff'] = pattern2['Family_Ticket_Diff'].abs()
print(f"件数: {len(pattern2)} ({len(pattern2)/len(train)*100:.1f}%)")
print(f"生存率: {pattern2['Survived'].mean():.3f}")
print("\n例:")
print(pattern2.nlargest(5, 'Abs_Diff')[['Name', 'Ticket', 'SibSp', 'Parch',
      'FamilySize', 'Ticket_GroupSize', 'Survived']].to_string(index=False))

# ===== 4. 生存率との関係 =====
print("\n" + "=" * 80)
print("4. 生存率との関係")
print("=" * 80)

print("\nFamilySize別の生存率:")
familysize_survival = train.groupby('FamilySize')['Survived'].agg(['mean', 'count']).round(3)
familysize_survival.columns = ['Survival_Rate', 'Count']
print(familysize_survival)

print("\nTicket_GroupSize別の生存率:")
ticketsize_survival = train.groupby('Ticket_GroupSize')['Survived'].agg(['mean', 'count']).round(3)
ticketsize_survival.columns = ['Survival_Rate', 'Count']
print(ticketsize_survival)

print("\nSurvivedとの相関:")
correlations = train[family_features + ['Survived']].corr()['Survived'].sort_values(ascending=False)
print(correlations)

# ===== 5. モデル性能比較 =====
print("\n" + "=" * 80)
print("5. モデル性能比較（様々な組み合わせ）")
print("=" * 80)

base_features_core = ['Pclass', 'Sex', 'Title', 'Ticket_Fare_Per_Person'] + embarked_cols

feature_configs = {
    # ベースライン
    'Baseline (SibSp+Parch+FamilySize+SmallFamily)':
        base_features_core + ['SibSp', 'Parch', 'FamilySize', 'SmallFamily'],

    # Ticket_GroupSize のみ
    'Ticket_GroupSize only':
        base_features_core + ['Ticket_GroupSize'],

    'Ticket_GroupSize + Ticket_SmallGroup':
        base_features_core + ['Ticket_GroupSize', 'Ticket_SmallGroup'],

    # 組み合わせ
    'SibSp + Parch + Ticket_GroupSize':
        base_features_core + ['SibSp', 'Parch', 'Ticket_GroupSize'],

    'FamilySize + Ticket_GroupSize':
        base_features_core + ['FamilySize', 'Ticket_GroupSize'],

    'FamilySize + SmallFamily + Ticket_GroupSize':
        base_features_core + ['FamilySize', 'SmallFamily', 'Ticket_GroupSize'],

    # Ticket優先
    'Ticket_GroupSize + Ticket_SmallGroup (NO Family)':
        base_features_core + ['Ticket_GroupSize', 'Ticket_SmallGroup'],

    # すべて
    'All (SibSp+Parch+FamilySize+Ticket_GroupSize)':
        base_features_core + ['SibSp', 'Parch', 'FamilySize', 'SmallFamily', 'Ticket_GroupSize'],
}

y = train['Survived']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
}

results = []

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)

    for feat_name, features in feature_configs.items():
        X = train[features]
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()

        print(f"  {feat_name:50s}: {mean_score:.4f}")

        results.append({
            'Model': model_name,
            'Features': feat_name,
            'CV_Score': mean_score,
            'Num_Features': len(features)
        })

# ===== 6. 結果の分析 =====
print("\n" + "=" * 80)
print("6. 結果の分析")
print("=" * 80)

results_df = pd.DataFrame(results)

for model_name in models.keys():
    model_results = results_df[results_df['Model'] == model_name].sort_values('CV_Score', ascending=False)

    print(f"\n{model_name} - Top 5:")
    print(model_results.head(5).to_string(index=False))

# ベースラインとの比較
print("\n" + "=" * 80)
print("7. ベースラインとの比較")
print("=" * 80)

baseline_name = 'Baseline (SibSp+Parch+FamilySize+SmallFamily)'
baseline_results = results_df[results_df['Features'] == baseline_name].set_index('Model')

print("\n改善率:")
print("-" * 60)

comparison_configs = [
    'Ticket_GroupSize only',
    'Ticket_GroupSize + Ticket_SmallGroup',
    'Ticket_GroupSize + Ticket_SmallGroup (NO Family)'
]

for feat_name in comparison_configs:
    print(f"\n{feat_name}:")
    feat_results = results_df[results_df['Features'] == feat_name].set_index('Model')

    for model in models.keys():
        baseline_score = baseline_results.loc[model, 'CV_Score']
        new_score = feat_results.loc[model, 'CV_Score']
        improvement = new_score - baseline_score
        improvement_pct = (improvement / baseline_score) * 100

        print(f"  {model:20s}: {baseline_score:.4f} -> {new_score:.4f} "
              f"({improvement:+.4f}, {improvement_pct:+.2f}%)")

# ===== 8. 特徴量重要度 =====
print("\n" + "=" * 80)
print("8. 特徴量重要度（GradientBoosting）")
print("=" * 80)

gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

# ベースライン
baseline_features = base_features_core + ['SibSp', 'Parch', 'FamilySize', 'SmallFamily']
X_baseline = train[baseline_features]
gb.fit(X_baseline, y)

importances_baseline = pd.DataFrame({
    'Feature': baseline_features,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nベースライン（SibSp+Parch+FamilySize+SmallFamily）:")
print(importances_baseline.to_string(index=False))

# Ticket_GroupSize only
ticket_features = base_features_core + ['Ticket_GroupSize', 'Ticket_SmallGroup']
X_ticket = train[ticket_features]
gb.fit(X_ticket, y)

importances_ticket = pd.DataFrame({
    'Feature': ticket_features,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTicket_GroupSize + Ticket_SmallGroup:")
print(importances_ticket.to_string(index=False))

# ===== 9. 結論 =====
print("\n" + "=" * 80)
print("9. 結論")
print("=" * 80)

# 最良構成を特定
best_overall = results_df.loc[results_df['CV_Score'].idxmax()]

print(f"\n最良構成（全体）:")
print(f"  モデル: {best_overall['Model']}")
print(f"  特徴量: {best_overall['Features']}")
print(f"  CV Score: {best_overall['CV_Score']:.4f}")
print(f"  特徴量数: {best_overall['Num_Features']}")

print("\n【結論】")

# GradientBoostingでの最良
gb_results = results_df[results_df['Model'] == 'GradientBoosting'].sort_values('CV_Score', ascending=False)
gb_best = gb_results.iloc[0]
gb_baseline = gb_results[gb_results['Features'] == baseline_name].iloc[0]

improvement = gb_best['CV_Score'] - gb_baseline['CV_Score']

print(f"\nGradientBoostingでの推奨:")
print(f"  最良: {gb_best['Features']}")
print(f"  スコア: {gb_best['CV_Score']:.4f}")
print(f"  ベースラインからの改善: {improvement:+.4f} ({improvement/gb_baseline['CV_Score']*100:+.2f}%)")

if improvement > 0.001:
    print(f"\n✓ {gb_best['Features']} を推奨")
    print(f"  SibSp/Parch/FamilySizeの代わりにTicket_GroupSizeを使うことで性能向上")
elif abs(improvement) < 0.001:
    print(f"\n= 性能はほぼ同等")
    print(f"  シンプルさの観点からTicket_GroupSizeのみを推奨")
else:
    print(f"\n✗ ベースラインの方が良い")
    print(f"  SibSp/Parch/FamilySizeを保持すべき")

# 結果保存
import json
summary = {
    'match_rate': float(match_rate),
    'correlation': {
        'SibSp_Ticket_GroupSize': float(train[['SibSp', 'Ticket_GroupSize']].corr().iloc[0, 1]),
        'Parch_Ticket_GroupSize': float(train[['Parch', 'Ticket_GroupSize']].corr().iloc[0, 1]),
        'FamilySize_Ticket_GroupSize': float(train[['FamilySize', 'Ticket_GroupSize']].corr().iloc[0, 1])
    },
    'best_config': {
        'model': best_overall['Model'],
        'features': best_overall['Features'],
        'score': float(best_overall['CV_Score']),
        'num_features': int(best_overall['Num_Features'])
    },
    'gb_baseline_score': float(gb_baseline['CV_Score']),
    'gb_best_score': float(gb_best['CV_Score']),
    'gb_improvement': float(improvement)
}

with open('/home/user/julis_titanic/experiments/results/family_vs_ticket_group_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n結果を保存: experiments/results/family_vs_ticket_group_analysis.json")
print("\n分析完了！")
