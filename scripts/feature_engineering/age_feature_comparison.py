"""
年齢特徴量の比較実験

目的: 以下の3つのケースで精度を比較
  1. 年齢の生値のみ（Age）
  2. 年齢層ラベルのみ（AgeGroup_DomainKnowledge）
  3. 両方を使用（Age + AgeGroup_DomainKnowledge）

年齢層のビニング:
  bins = [0, 18, 60, 80]
  labels = ['Child', 'Adult', 'Elderly']
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("年齢特徴量の比較実験")
print("="*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df, use_raw_age=True, use_age_group=False):
    """
    特徴量エンジニアリング

    Parameters:
    -----------
    df : DataFrame
        入力データ
    use_raw_age : bool
        年齢の生値を使用するか
    use_age_group : bool
        年齢層ラベルを使用するか
    """
    df_fe = df.copy()

    # 欠損値フラグ
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)

    # Title抽出
    df_fe['Title'] = df_fe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_fe['Title'] = df_fe['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_fe['Title'] = df_fe['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_fe['Title'] = df_fe['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df_fe['Title'] = df_fe['Title'].map(title_mapping).fillna(0)

    # 家族サイズ関連
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    df_fe['SmallFamily'] = ((df_fe['FamilySize'] >= 2) & (df_fe['FamilySize'] <= 4)).astype(int)

    # 年齢の欠損値補完（Title x Pclassの中央値）
    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    # 年齢層の作成（ドメイン知識ベース）
    if use_age_group:
        bins_domain_knowledge = [0, 18, 60, 80]
        labels_domain_knowledge = ['Child', 'Adult', 'Elderly']
        df_fe['AgeGroup_DomainKnowledge'] = pd.cut(
            df_fe['Age'],
            bins=bins_domain_knowledge,
            labels=labels_domain_knowledge,
            right=False
        )
        # カテゴリカル変数を数値に変換
        age_group_mapping = {'Child': 0, 'Adult': 1, 'Elderly': 2}
        df_fe['AgeGroup_DomainKnowledge'] = df_fe['AgeGroup_DomainKnowledge'].map(age_group_mapping)

    # Fare関連
    df_fe['Fare'].fillna(df_fe.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    df_fe['FareBin'] = pd.qcut(df_fe['Fare'], q=4, labels=False, duplicates='drop')

    # Sex
    df_fe['Sex'] = df_fe['Sex'].map({'male': 0, 'female': 1})

    # Embarked
    df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0], inplace=True)

    return df_fe

# データ結合
all_data = pd.concat([train, test], sort=False, ignore_index=True)

# モデル設定（2種類で比較）
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
    reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
    learning_rate=0.05, random_state=42, verbose=-1
)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_split=10,
    min_samples_leaf=5, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 基本特徴量（年齢関連を除く）
base_features = ['Pclass', 'Sex', 'Title', 'IsAlone', 'FareBin',
                 'SmallFamily', 'Age_Missing', 'Embarked_Missing']

# ========================================================================
# ケース1: 年齢の生値のみ（Age）
# ========================================================================
print("\n" + "="*80)
print("【ケース1: 年齢の生値のみ（Age）】")
print("="*80)

all_data_raw_age = engineer_features(all_data, use_raw_age=True, use_age_group=False)
train_raw_age = all_data_raw_age[:len(train)]
y_train = train['Survived']

raw_age_features = base_features + ['Age']
X_raw_age = train_raw_age[raw_age_features]

print(f"\n特徴量数: {len(raw_age_features)}")
print(f"特徴量: {raw_age_features}")

# LightGBM
print("\n--- LightGBM ---")
cv_scores_lgb_raw = cross_val_score(lgb_model, X_raw_age, y_train, cv=cv, scoring='accuracy')
lgb_model.fit(X_raw_age, y_train)
train_score_lgb_raw = lgb_model.score(X_raw_age, y_train)

print(f"CV Score: {cv_scores_lgb_raw.mean():.5f} (±{cv_scores_lgb_raw.std():.5f})")
print(f"Train Score: {train_score_lgb_raw:.5f}")
print(f"Gap: {train_score_lgb_raw - cv_scores_lgb_raw.mean():.5f}")

# Random Forest
print("\n--- Random Forest ---")
cv_scores_rf_raw = cross_val_score(rf_model, X_raw_age, y_train, cv=cv, scoring='accuracy')
rf_model.fit(X_raw_age, y_train)
train_score_rf_raw = rf_model.score(X_raw_age, y_train)

print(f"CV Score: {cv_scores_rf_raw.mean():.5f} (±{cv_scores_rf_raw.std():.5f})")
print(f"Train Score: {train_score_rf_raw:.5f}")
print(f"Gap: {train_score_rf_raw - cv_scores_rf_raw.mean():.5f}")

# ========================================================================
# ケース2: 年齢層ラベルのみ（AgeGroup_DomainKnowledge）
# ========================================================================
print("\n" + "="*80)
print("【ケース2: 年齢層ラベルのみ（AgeGroup_DomainKnowledge）】")
print("="*80)
print("年齢層: Child (0-18), Adult (18-60), Elderly (60-80)")

all_data_age_group = engineer_features(all_data, use_raw_age=False, use_age_group=True)
train_age_group = all_data_age_group[:len(train)]

age_group_features = base_features + ['AgeGroup_DomainKnowledge']
X_age_group = train_age_group[age_group_features]

print(f"\n特徴量数: {len(age_group_features)}")
print(f"特徴量: {age_group_features}")

# LightGBM
print("\n--- LightGBM ---")
cv_scores_lgb_group = cross_val_score(lgb_model, X_age_group, y_train, cv=cv, scoring='accuracy')
lgb_model.fit(X_age_group, y_train)
train_score_lgb_group = lgb_model.score(X_age_group, y_train)

print(f"CV Score: {cv_scores_lgb_group.mean():.5f} (±{cv_scores_lgb_group.std():.5f})")
print(f"Train Score: {train_score_lgb_group:.5f}")
print(f"Gap: {train_score_lgb_group - cv_scores_lgb_group.mean():.5f}")

# Random Forest
print("\n--- Random Forest ---")
cv_scores_rf_group = cross_val_score(rf_model, X_age_group, y_train, cv=cv, scoring='accuracy')
rf_model.fit(X_age_group, y_train)
train_score_rf_group = rf_model.score(X_age_group, y_train)

print(f"CV Score: {cv_scores_rf_group.mean():.5f} (±{cv_scores_rf_group.std():.5f})")
print(f"Train Score: {train_score_rf_group:.5f}")
print(f"Gap: {train_score_rf_group - cv_scores_rf_group.mean():.5f}")

# ========================================================================
# ケース3: 両方を使用（Age + AgeGroup_DomainKnowledge）
# ========================================================================
print("\n" + "="*80)
print("【ケース3: 両方を使用（Age + AgeGroup_DomainKnowledge）】")
print("="*80)

all_data_both = engineer_features(all_data, use_raw_age=True, use_age_group=True)
train_both = all_data_both[:len(train)]

both_features = base_features + ['Age', 'AgeGroup_DomainKnowledge']
X_both = train_both[both_features]

print(f"\n特徴量数: {len(both_features)}")
print(f"特徴量: {both_features}")

# LightGBM
print("\n--- LightGBM ---")
cv_scores_lgb_both = cross_val_score(lgb_model, X_both, y_train, cv=cv, scoring='accuracy')
lgb_model.fit(X_both, y_train)
train_score_lgb_both = lgb_model.score(X_both, y_train)

print(f"CV Score: {cv_scores_lgb_both.mean():.5f} (±{cv_scores_lgb_both.std():.5f})")
print(f"Train Score: {train_score_lgb_both:.5f}")
print(f"Gap: {train_score_lgb_both - cv_scores_lgb_both.mean():.5f}")

# Random Forest
print("\n--- Random Forest ---")
cv_scores_rf_both = cross_val_score(rf_model, X_both, y_train, cv=cv, scoring='accuracy')
rf_model.fit(X_both, y_train)
train_score_rf_both = rf_model.score(X_both, y_train)

print(f"CV Score: {cv_scores_rf_both.mean():.5f} (±{cv_scores_rf_both.std():.5f})")
print(f"Train Score: {train_score_rf_both:.5f}")
print(f"Gap: {train_score_rf_both - cv_scores_rf_both.mean():.5f}")

# ========================================================================
# 総合比較
# ========================================================================
print("\n" + "="*80)
print("総合比較")
print("="*80)

# LightGBM比較
print("\n【LightGBM】")
comparison_lgb = pd.DataFrame({
    'ケース': ['年齢の生値のみ', '年齢層ラベルのみ', '両方を使用'],
    '特徴量数': [len(raw_age_features), len(age_group_features), len(both_features)],
    'CV Score': [cv_scores_lgb_raw.mean(), cv_scores_lgb_group.mean(), cv_scores_lgb_both.mean()],
    'CV Std': [cv_scores_lgb_raw.std(), cv_scores_lgb_group.std(), cv_scores_lgb_both.std()],
    'Train Score': [train_score_lgb_raw, train_score_lgb_group, train_score_lgb_both],
    'Gap': [
        train_score_lgb_raw - cv_scores_lgb_raw.mean(),
        train_score_lgb_group - cv_scores_lgb_group.mean(),
        train_score_lgb_both - cv_scores_lgb_both.mean()
    ],
    'CV変化': [
        0,
        cv_scores_lgb_group.mean() - cv_scores_lgb_raw.mean(),
        cv_scores_lgb_both.mean() - cv_scores_lgb_raw.mean()
    ]
})

print(comparison_lgb.to_string(index=False))

best_lgb_idx = comparison_lgb['CV Score'].idxmax()
print(f"\n【最良のケース (LightGBM)】")
print(f"{comparison_lgb.iloc[best_lgb_idx]['ケース']}")
print(f"CV Score: {comparison_lgb.iloc[best_lgb_idx]['CV Score']:.5f}")

# Random Forest比較
print("\n" + "="*80)
print("【Random Forest】")
comparison_rf = pd.DataFrame({
    'ケース': ['年齢の生値のみ', '年齢層ラベルのみ', '両方を使用'],
    '特徴量数': [len(raw_age_features), len(age_group_features), len(both_features)],
    'CV Score': [cv_scores_rf_raw.mean(), cv_scores_rf_group.mean(), cv_scores_rf_both.mean()],
    'CV Std': [cv_scores_rf_raw.std(), cv_scores_rf_group.std(), cv_scores_rf_both.std()],
    'Train Score': [train_score_rf_raw, train_score_rf_group, train_score_rf_both],
    'Gap': [
        train_score_rf_raw - cv_scores_rf_raw.mean(),
        train_score_rf_group - cv_scores_rf_group.mean(),
        train_score_rf_both - cv_scores_rf_both.mean()
    ],
    'CV変化': [
        0,
        cv_scores_rf_group.mean() - cv_scores_rf_raw.mean(),
        cv_scores_rf_both.mean() - cv_scores_rf_raw.mean()
    ]
})

print(comparison_rf.to_string(index=False))

best_rf_idx = comparison_rf['CV Score'].idxmax()
print(f"\n【最良のケース (Random Forest)】")
print(f"{comparison_rf.iloc[best_rf_idx]['ケース']}")
print(f"CV Score: {comparison_rf.iloc[best_rf_idx]['CV Score']:.5f}")

# 相互作用の評価（LightGBM）
print("\n" + "="*80)
print("【相互作用の評価 (LightGBM)】")
print("="*80)

expected_improvement_lgb = (cv_scores_lgb_group.mean() - cv_scores_lgb_raw.mean())
actual_improvement_lgb = cv_scores_lgb_both.mean() - cv_scores_lgb_raw.mean()
interaction_effect_lgb = actual_improvement_lgb - expected_improvement_lgb

print(f"年齢層ラベルのみの改善: {expected_improvement_lgb:+.5f}")
print(f"両方使用時の改善: {actual_improvement_lgb:+.5f}")
print(f"相互作用効果: {interaction_effect_lgb:+.5f}")

if interaction_effect_lgb < -0.001:
    print("→ 負の相互作用あり（特徴量が互いに干渉している）")
elif interaction_effect_lgb > 0.001:
    print("→ 正の相互作用あり（特徴量が相乗効果を発揮）")
else:
    print("→ 相互作用なし（独立）")

# 相互作用の評価（Random Forest）
print("\n" + "="*80)
print("【相互作用の評価 (Random Forest)】")
print("="*80)

expected_improvement_rf = (cv_scores_rf_group.mean() - cv_scores_rf_raw.mean())
actual_improvement_rf = cv_scores_rf_both.mean() - cv_scores_rf_raw.mean()
interaction_effect_rf = actual_improvement_rf - expected_improvement_rf

print(f"年齢層ラベルのみの改善: {expected_improvement_rf:+.5f}")
print(f"両方使用時の改善: {actual_improvement_rf:+.5f}")
print(f"相互作用効果: {interaction_effect_rf:+.5f}")

if interaction_effect_rf < -0.001:
    print("→ 負の相互作用あり（特徴量が互いに干渉している）")
elif interaction_effect_rf > 0.001:
    print("→ 正の相互作用あり（特徴量が相乗効果を発揮）")
else:
    print("→ 相互作用なし（独立）")

print("\n" + "="*80)
print("分析完了")
print("="*80)
