#!/usr/bin/env python3
"""
既存コードから本番パイプラインへの移行例

Before: 各スクリプトに埋め込まれた補完ロジック
After: ProductionImputerを使用した統一パイプライン
"""

import sys
sys.path.append('scripts/preprocessing')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from production_imputation_pipeline import ProductionImputer

print('='*80)
print('MIGRATION EXAMPLE: BEFORE vs AFTER')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# ============================================================================
# BEFORE: 既存の方法（各スクリプトに埋め込み）
# ============================================================================
print('\n' + '='*80)
print('BEFORE: Embedded imputation in each script')
print('='*80)

def old_way_feature_engineering(df):
    """
    既存方法: 各スクリプトに埋め込まれた特徴量エンジニアリング
    （model_specific_features_pipeline.pyから抜粋）
    """
    df_fe = df.copy()

    # Title extraction
    df_fe['Title'] = df_fe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df_fe['Title'] = df_fe['Title'].map(title_mapping)

    # Family features
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)

    # Age imputation (Title + Pclass別)
    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    # Fare imputation (Pclass別)
    df_fe['Fare'] = df_fe.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # Embarked (最頻値)
    df_fe['Embarked'] = df_fe['Embarked'].fillna('S')

    # Cabin
    df_fe['Has_Cabin'] = df_fe['Cabin'].notna().astype(int)

    return df_fe

print("\nProcessing with OLD way...")
train_old = old_way_feature_engineering(train.copy())
test_old = old_way_feature_engineering(test.copy())

# Encoding
train_old['Sex'] = train_old['Sex'].map({'male': 0, 'female': 1})
test_old['Sex'] = test_old['Sex'].map({'male': 0, 'female': 1})

le_title = LabelEncoder()
train_old['Title'] = le_title.fit_transform(train_old['Title'])
test_old['Title'] = le_title.transform(test_old['Title'])

# Features
features_old = ['Pclass', 'Sex', 'Age', 'Title', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'Has_Cabin']

X_train_old = train_old[features_old]
y_train = train['Survived']

print(f"Features: {len(features_old)}")
print(f"Feature list: {', '.join(features_old)}")

# ============================================================================
# AFTER: 本番パイプライン（ProductionImputer）
# ============================================================================
print('\n' + '='*80)
print('AFTER: Production imputation pipeline')
print('='*80)

def new_way_feature_engineering(df):
    """
    新方法: 補完済みデータに対する特徴量エンジニアリング
    （補完ロジックはProductionImputerに分離）
    """
    df_fe = df.copy()

    # 補完とTitle抽出は既にProductionImputerで完了

    # Family features
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)

    # Has_Cabinも既にProductionImputerで作成済み

    return df_fe

print("\nProcessing with NEW way...")

# 1. 補完（分離されたパイプライン）
imputer = ProductionImputer(verbose=False)
train_imputed = imputer.fit_transform(train.copy())
test_imputed = imputer.transform(test.copy())

# 2. 特徴量エンジニアリング（補完ロジックを含まない）
train_new = new_way_feature_engineering(train_imputed)
test_new = new_way_feature_engineering(test_imputed)

# Encoding
train_new['Sex'] = train_new['Sex'].map({'male': 0, 'female': 1})
test_new['Sex'] = test_new['Sex'].map({'male': 0, 'female': 1})

le_title_new = LabelEncoder()
train_new['Title'] = le_title_new.fit_transform(train_new['Title'])
test_new['Title'] = le_title_new.transform(test_new['Title'])

# Features（同じ）
features_new = ['Pclass', 'Sex', 'Age', 'Title', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'Has_Cabin']

X_train_new = train_new[features_new]

print(f"Features: {len(features_new)}")
print(f"Feature list: {', '.join(features_new)}")

# ============================================================================
# COMPARISON: CV Scores
# ============================================================================
print('\n' + '='*80)
print('COMPARISON: Cross-Validation Scores')
print('='*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                     learning_rate=0.05, random_state=42),
}

results = []

for model_name, model in models.items():
    # Old way
    old_scores = cross_val_score(model, X_train_old, y_train, cv=cv, scoring='accuracy')
    old_mean = old_scores.mean()
    old_std = old_scores.std()

    # New way
    new_scores = cross_val_score(model, X_train_new, y_train, cv=cv, scoring='accuracy')
    new_mean = new_scores.mean()
    new_std = new_scores.std()

    # Improvement
    improvement = new_mean - old_mean

    results.append({
        'Model': model_name,
        'Old_Mean': old_mean,
        'Old_Std': old_std,
        'New_Mean': new_mean,
        'New_Std': new_std,
        'Improvement': improvement
    })

    print(f"\n{model_name}:")
    print(f"  OLD:  {old_mean:.5f} (±{old_std:.5f})")
    print(f"  NEW:  {new_mean:.5f} (±{new_std:.5f})")
    print(f"  Δ:    {improvement:+.5f} {'✅' if improvement > 0 else '❌' if improvement < 0 else '='}")

# ============================================================================
# SUMMARY
# ============================================================================
print('\n' + '='*80)
print('SUMMARY')
print('='*80)

results_df = pd.DataFrame(results)
avg_improvement = results_df['Improvement'].mean()

print(f"\nAverage improvement: {avg_improvement:+.5f}")

if avg_improvement > 0:
    print("\n✅ The new production pipeline shows IMPROVEMENT")
elif avg_improvement < 0:
    print("\n⚠️ The new production pipeline shows slight DECREASE")
    print("   (However, it handles test data missing values better)")
else:
    print("\n= The results are EQUIVALENT")

print("\n" + "-"*80)
print("BENEFITS OF NEW PIPELINE:")
print("-"*80)
print("""
1. ✅ Code Reusability
   - No need to copy-paste imputation logic
   - Use ProductionImputer across all scripts

2. ✅ Maintainability
   - Imputation logic in ONE place
   - Easy to update and improve

3. ✅ Consistency
   - All models use the SAME imputation method
   - No discrepancies between scripts

4. ✅ Test Data Handling
   - Properly handles Fare missing in test data
   - Pclass + Embarked based imputation

5. ✅ Leak Prevention
   - Clear train/test separation
   - imputer.fit(train) → imputer.transform(test)

6. ✅ Experimentation
   - Easy to compare different strategies
   - Validated with multiple experiments
""")

print("\n" + "="*80)
print("MIGRATION RECOMMENDATION")
print("="*80)
print("""
Recommended migration approach:

1. Start with NEW models
   - Use ProductionImputer for all new development

2. Gradually migrate EXISTING models
   - Replace embedded imputation with ProductionImputer
   - Validate that scores are maintained or improved

3. Keep BASELINE for comparison
   - Maintain 1-2 existing scripts as baseline

4. UPDATE documentation
   - Mark ProductionImputer as the standard approach
""")

print("\n" + "="*80)
print("EXAMPLE COMPLETED!")
print("="*80)
