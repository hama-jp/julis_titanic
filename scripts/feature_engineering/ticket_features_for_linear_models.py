"""
LogisticRegression用のTicket特徴量最適化

問題: 全Ticket特徴量を追加するとLogisticRegressionで悪化（-1.53%）
原因: 多重共線性
解決: VIF分析と特徴量選択で線形モデル用の最適セットを作成
"""

import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/user/julis_titanic')
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

print("=" * 80)
print("LogisticRegression用のTicket特徴量最適化")
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
    combined['Ticket_Length'] = combined['Ticket'].astype(str).str.len()

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
    return df_new

train = engineer_features(train)
test = engineer_features(test)

train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

embarked_cols = [col for col in train.columns if col.startswith('Embarked_')]
for col in embarked_cols:
    if col not in test.columns:
        test[col] = 0

# ===== 1. VIF分析（多重共線性の診断） =====
print("\n" + "=" * 80)
print("1. VIF（Variance Inflation Factor）分析")
print("=" * 80)

base_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Title',
                 'FamilySize', 'SmallFamily'] + embarked_cols

ticket_features = ['Ticket_GroupSize', 'Ticket_Length', 'Ticket_Prefix_Category',
                   'Ticket_Is_Numeric', 'Ticket_Fare_Per_Person', 'Ticket_SmallGroup']

all_features = base_features + ticket_features

X_all = train[all_features]
y = train['Survived']

# 標準化（VIF計算のため）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features)

print("\nVIF値（高いほど多重共線性が強い）:")
print("基準: VIF > 10 は問題あり、VIF > 5 は要注意")
print("-" * 80)

vif_data = pd.DataFrame()
vif_data['Feature'] = all_features
vif_data['VIF'] = [variance_inflation_factor(X_scaled_df.values, i)
                   for i in range(len(all_features))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(vif_data.to_string(index=False))

print("\n⚠️ 高VIF特徴量（VIF > 5）:")
high_vif = vif_data[vif_data['VIF'] > 5]
print(high_vif.to_string(index=False))

# ===== 2. 相関行列（詳細） =====
print("\n" + "=" * 80)
print("2. 相関分析（絶対値 > 0.6）")
print("=" * 80)

corr_matrix = train[all_features].corr()

print("\n高相関ペア:")
print("-" * 80)
high_corr_pairs = []
for i in range(len(all_features)):
    for j in range(i+1, len(all_features)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > 0.6:
            feat1 = all_features[i]
            feat2 = all_features[j]
            is_ticket_pair = (feat1 in ticket_features or feat2 in ticket_features)
            print(f"{feat1:30s} <-> {feat2:30s}: {corr_value:+.3f} {'[Ticket関連]' if is_ticket_pair else ''}")
            high_corr_pairs.append((feat1, feat2, corr_value))

# ===== 3. 特徴量選択戦略 =====
print("\n" + "=" * 80)
print("3. 特徴量選択戦略")
print("=" * 80)

print("\n【問題のある特徴量ペア】")
print("-" * 80)

strategies = {
    'Strategy 1: 高VIF特徴量を除外': {
        'remove': ['FamilySize', 'Fare'],  # VIF最高
        'keep_ticket': ticket_features
    },
    'Strategy 2: Ticket重複除外': {
        'remove': [],
        'keep_ticket': ['Ticket_Is_Numeric', 'Ticket_Prefix_Category', 'Ticket_Length']
        # GroupSize/SmallGroup/FarePerPersonを除外（既存特徴量と重複）
    },
    'Strategy 3: 最小構成': {
        'remove': ['FamilySize'],  # Ticket_GroupSizeと重複
        'keep_ticket': ['Ticket_Is_Numeric', 'Ticket_Prefix_Category']
        # 最も独立性の高いTicket特徴量のみ
    },
    'Strategy 4: Fare系統一': {
        'remove': ['Fare'],  # Ticket_Fare_Per_Personと交換
        'keep_ticket': ['Ticket_Fare_Per_Person', 'Ticket_Is_Numeric',
                        'Ticket_Prefix_Category', 'Ticket_SmallGroup']
    }
}

for strategy_name, strategy in strategies.items():
    print(f"\n{strategy_name}:")
    remove_features = strategy['remove']
    ticket_keep = strategy['keep_ticket']

    if remove_features:
        print(f"  除外: {', '.join(remove_features)}")
    print(f"  追加Ticket特徴量: {', '.join(ticket_keep)}")

# ===== 4. 各戦略の評価 =====
print("\n" + "=" * 80)
print("4. LogisticRegressionでの評価（各戦略）")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ベースライン
X_baseline = train[base_features]
lr_baseline = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
baseline_scores = cross_val_score(lr_baseline, X_baseline, y, cv=cv, scoring='accuracy')
baseline_mean = baseline_scores.mean()
baseline_std = baseline_scores.std()

print(f"\n{'Strategy':<40s} {'Features':>4s} {'CV Score':>10s} {'Change':>10s}")
print("-" * 80)
print(f"{'Baseline (existing features)':<40s} {len(base_features):>4d} {baseline_mean:>10.4f} {'-':>10s}")

results = []

for strategy_name, strategy in strategies.items():
    # 特徴量セット構築
    remove_features = strategy['remove']
    ticket_keep = strategy['keep_ticket']

    features_to_use = [f for f in base_features if f not in remove_features] + ticket_keep

    X_strategy = train[features_to_use]

    # 評価
    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    scores = cross_val_score(lr_model, X_strategy, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    improvement = mean_score - baseline_mean

    print(f"{strategy_name:<40s} {len(features_to_use):>4d} {mean_score:>10.4f} {improvement:>+10.4f}")

    results.append({
        'Strategy': strategy_name,
        'Features': features_to_use,
        'Num_Features': len(features_to_use),
        'CV_Mean': mean_score,
        'CV_Std': std_score,
        'Improvement': improvement
    })

# ===== 5. 正則化強度の調整 =====
print("\n" + "=" * 80)
print("5. 正則化強度の調整（全Ticket特徴量使用）")
print("=" * 80)

X_all_ticket = train[all_features]

print(f"\n{'C (inverse regularization)':<30s} {'CV Score':>10s} {'Change':>10s}")
print("-" * 80)

C_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for C in C_values:
    lr_reg = LogisticRegression(max_iter=1000, random_state=42, C=C)
    scores = cross_val_score(lr_reg, X_all_ticket, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    improvement = mean_score - baseline_mean

    print(f"C={C:<27.2f} {mean_score:>10.4f} {improvement:>+10.4f}")

# ===== 6. 最良戦略の詳細 =====
print("\n" + "=" * 80)
print("6. 最良戦略の詳細")
print("=" * 80)

best_idx = max(range(len(results)), key=lambda i: results[i]['CV_Mean'])
best_strategy = results[best_idx]

print(f"\n最良戦略: {best_strategy['Strategy']}")
print(f"CV Score: {best_strategy['CV_Mean']:.4f} (± {best_strategy['CV_Std']:.4f})")
print(f"改善: {best_strategy['Improvement']:+.4f} ({best_strategy['Improvement']/baseline_mean*100:+.2f}%)")
print(f"特徴量数: {best_strategy['Num_Features']}")

print(f"\n使用特徴量:")
for feat in best_strategy['Features']:
    if feat in ticket_features:
        print(f"  - {feat} [Ticket]")
    else:
        print(f"  - {feat}")

# VIFチェック
X_best = train[best_strategy['Features']]
X_best_scaled = scaler.fit_transform(X_best)
X_best_scaled_df = pd.DataFrame(X_best_scaled, columns=best_strategy['Features'])

vif_best = pd.DataFrame()
vif_best['Feature'] = best_strategy['Features']
vif_best['VIF'] = [variance_inflation_factor(X_best_scaled_df.values, i)
                   for i in range(len(best_strategy['Features']))]
vif_best = vif_best.sort_values('VIF', ascending=False)

print(f"\nVIF（最良戦略）:")
print(vif_best.to_string(index=False))

max_vif = vif_best['VIF'].max()
print(f"\n最大VIF: {max_vif:.2f}")
if max_vif > 10:
    print("⚠️ 依然として多重共線性の問題あり（VIF > 10）")
elif max_vif > 5:
    print("⚠️ 要注意レベルの多重共線性（VIF > 5）")
else:
    print("✓ 多重共線性は許容範囲内（VIF < 5）")

# ===== 7. 推奨事項 =====
print("\n" + "=" * 80)
print("7. 推奨事項")
print("=" * 80)

print("\n【LogisticRegression用の推奨構成】")

if best_strategy['Improvement'] > 0:
    print(f"\n✓ {best_strategy['Strategy']} を推奨")
    print(f"  - 改善: {best_strategy['Improvement']:+.4f}")
    print(f"  - 特徴量数: {best_strategy['Num_Features']}")
    print(f"  - 追加Ticket特徴量: {[f for f in best_strategy['Features'] if f in ticket_features]}")
else:
    print("\n✗ Ticket特徴量はLogisticRegressionでは推奨されません")
    print(f"  最良でも改善なし: {best_strategy['Improvement']:+.4f}")

print("\n【モデル別の推奨】")
print("  - GradientBoosting/LightGBM: 全Ticket特徴量使用（+1.08%）")
print("  - RandomForest: 全Ticket特徴量使用（+0.27%）")
if best_strategy['Improvement'] > 0:
    print(f"  - LogisticRegression: {best_strategy['Strategy']}")
else:
    print("  - LogisticRegression: Ticket特徴量なし（ベースライン）")

# 結果保存
import json
summary = {
    'baseline_score': float(baseline_mean),
    'best_strategy': best_strategy['Strategy'],
    'best_score': float(best_strategy['CV_Mean']),
    'improvement': float(best_strategy['Improvement']),
    'recommended_features': best_strategy['Features'],
    'all_strategies': [{
        'name': r['Strategy'],
        'score': float(r['CV_Mean']),
        'improvement': float(r['Improvement']),
        'num_features': r['Num_Features']
    } for r in results]
}

with open('/home/user/julis_titanic/experiments/results/ticket_features_linear_model_optimization.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n結果を保存: experiments/results/ticket_features_linear_model_optimization.json")
print("\n分析完了！")
