"""
Ticket特徴量の重要度分析

各Ticket特徴量の重要度と単独での効果を詳細に分析
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("Ticket特徴量の重要度分析")
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

def extract_ticket_number(ticket):
    ticket = str(ticket).strip()
    match = re.search(r'(\d+)', ticket)
    if match:
        return int(match.group(1))
    return 0

def add_ticket_features(df_train, df_test):
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()
    combined = pd.concat([df_train_copy, df_test_copy], axis=0, ignore_index=True)

    combined['Ticket_Prefix'] = combined['Ticket'].apply(extract_ticket_prefix)
    combined['Ticket_Number'] = combined['Ticket'].apply(extract_ticket_number)
    combined['Ticket_Length'] = combined['Ticket'].astype(str).str.len()

    ticket_counts = combined.groupby('Ticket').size()
    combined['Ticket_GroupSize'] = combined['Ticket'].map(ticket_counts)

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
    combined['Ticket_Fare_Per_Person'] = combined['Fare'] / combined['Ticket_GroupSize']
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
    bins = [0, 18, 60, 100]
    labels = [0, 1, 2]
    df_new['AgeGroup'] = pd.cut(df_new['Age'], bins=bins, labels=labels, include_lowest=True)
    df_new['AgeGroup'] = df_new['AgeGroup'].astype(int)
    return df_new

train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ===== 1. 特徴量重要度（モデル別） =====
print("\n" + "=" * 80)
print("1. 特徴量重要度（モデル別）")
print("=" * 80)

base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title',
                 'FamilySize', 'SmallFamily'] + embarked_cols

ticket_features = ['Ticket_GroupSize', 'Ticket_Length', 'Ticket_Prefix_Category',
                   'Ticket_Is_Numeric', 'Ticket_Fare_Per_Person', 'Ticket_SmallGroup']

all_features = base_features + ticket_features

X = train[all_features]
y = train['Survived']

models_with_importance = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1)
}

for model_name, model in models_with_importance.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    model.fit(X, y)

    importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n全特徴量（Top 20）:")
    print(importances.head(20).to_string(index=False))

    print(f"\nTicket特徴量のみ:")
    ticket_importances = importances[importances['Feature'].isin(ticket_features)]
    print(ticket_importances.to_string(index=False))

# ===== 2. 単独特徴量評価 =====
print("\n" + "=" * 80)
print("2. 各Ticket特徴量の単独効果")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ベースラインスコア
X_baseline = train[base_features]
baseline_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
baseline_scores = cross_val_score(baseline_model, X_baseline, y, cv=cv, scoring='accuracy')
baseline_mean = baseline_scores.mean()

print(f"\nベースライン（{len(base_features)} features）: {baseline_mean:.4f}")

# 各Ticket特徴量を1つずつ追加
print(f"\n各Ticket特徴量を追加した効果:")
print("-" * 60)

single_feature_results = []

for ticket_feature in ticket_features:
    features_with_ticket = base_features + [ticket_feature]
    X_with_ticket = train[features_with_ticket]

    scores = cross_val_score(baseline_model, X_with_ticket, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    improvement = mean_score - baseline_mean

    print(f"{ticket_feature:30s}: {mean_score:.4f} ({improvement:+.4f})")

    single_feature_results.append({
        'Feature': ticket_feature,
        'CV_Score': mean_score,
        'Improvement': improvement
    })

# ===== 3. 特徴量の組み合わせ分析 =====
print("\n" + "=" * 80)
print("3. Ticket特徴量の組み合わせ分析")
print("=" * 80)

# 上位3つのTicket特徴量を特定
single_df = pd.DataFrame(single_feature_results).sort_values('Improvement', ascending=False)
top_features = single_df.head(3)['Feature'].tolist()

print(f"\n単独で効果が高いTop3特徴量:")
for i, feature in enumerate(top_features, 1):
    improvement = single_df[single_df['Feature'] == feature]['Improvement'].values[0]
    print(f"  {i}. {feature}: {improvement:+.4f}")

# 組み合わせ評価
print(f"\n組み合わせ評価:")
print("-" * 60)

combinations = [
    ('Top1', top_features[:1]),
    ('Top2', top_features[:2]),
    ('Top3', top_features[:3]),
    ('All Ticket', ticket_features)
]

for combo_name, combo_features in combinations:
    features_combo = base_features + combo_features
    X_combo = train[features_combo]

    scores = cross_val_score(baseline_model, X_combo, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    improvement = mean_score - baseline_mean

    print(f"{combo_name:15s} ({len(combo_features):2d} ticket features): {mean_score:.4f} ({improvement:+.4f})")

# ===== 4. 相関分析 =====
print("\n" + "=" * 80)
print("4. Ticket特徴量間の相関")
print("=" * 80)

ticket_corr = train[ticket_features].corr()
print("\nTicket特徴量間の相関行列:")
print(ticket_corr.round(3))

# 高相関のペアを特定
print("\n高相関のペア（|相関| > 0.5）:")
high_corr_pairs = []
for i in range(len(ticket_features)):
    for j in range(i+1, len(ticket_features)):
        corr_value = ticket_corr.iloc[i, j]
        if abs(corr_value) > 0.5:
            print(f"  {ticket_features[i]:30s} <-> {ticket_features[j]:30s}: {corr_value:.3f}")
            high_corr_pairs.append((ticket_features[i], ticket_features[j], corr_value))

# ===== 5. 既存特徴量との相関 =====
print("\n" + "=" * 80)
print("5. Ticket特徴量と既存特徴量の相関")
print("=" * 80)

key_base_features = ['Pclass', 'Sex', 'Fare', 'FamilySize', 'SmallFamily']
cross_corr = train[ticket_features + key_base_features].corr()

print("\nTicket特徴量と既存特徴量の相関:")
for ticket_feature in ticket_features:
    print(f"\n{ticket_feature}:")
    for base_feature in key_base_features:
        corr_value = cross_corr.loc[ticket_feature, base_feature]
        if abs(corr_value) > 0.3:
            print(f"  {base_feature:20s}: {corr_value:+.3f}")

# ===== 6. 最終推奨 =====
print("\n" + "=" * 80)
print("6. 最終推奨")
print("=" * 80)

print("\n【推奨事項】")

# 最良の組み合わせ
best_combo_idx = max(range(len(combinations)),
                      key=lambda i: baseline_mean + single_feature_results[0]['Improvement'])

print(f"\n1. 効果が確認されたTicket特徴量:")
for i, row in single_df[single_df['Improvement'] > 0].iterrows():
    print(f"   - {row['Feature']:30s}: {row['Improvement']:+.4f}")

if len(single_df[single_df['Improvement'] > 0]) == 0:
    print("   （効果的な特徴量なし）")

print(f"\n2. 推奨する追加特徴量:")
if single_df['Improvement'].max() > 0.001:
    print(f"   {', '.join(top_features[:3])}")
    print(f"   期待改善: +{single_df.head(3)['Improvement'].sum():.4f}")
else:
    print("   Ticket特徴量の追加は推奨されません（改善が微小）")

print(f"\n3. 注意点:")
if len(high_corr_pairs) > 0:
    print(f"   - {len(high_corr_pairs)}組の高相関ペアが存在（多重共線性の可能性）")
    for pair in high_corr_pairs:
        print(f"     {pair[0]} <-> {pair[1]}: {pair[2]:.3f}")
else:
    print(f"   - 高相関ペアなし（多重共線性のリスク低）")

print("\n分析完了！")
