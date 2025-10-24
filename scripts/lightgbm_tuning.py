"""
LightGBM Ultra-Conservative Tuning

Analyze overfitting causes and test progressively more conservative
parameters to reduce overfitting gap from 0.0427 to < 0.01

Current issue: LightGBM gap 0.0427 (target: < 0.01)

Overfitting reduction strategies:
1. Reduce model complexity (max_depth, num_leaves)
2. Increase regularization (reg_alpha, reg_lambda, min_gain_to_split)
3. Increase minimum samples (min_child_samples)
4. Reduce learning rate
5. Add more constraints (min_split_gain, max_bin)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIGHTGBM ULTRA-CONSERVATIVE TUNING")
print("=" * 80)
print("Goal: Reduce overfitting gap from 0.0427 to < 0.01")
print("=" * 80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print("\nEngineering features...")

def engineer_features(df):
    """Engineer features for both train and test"""
    df = df.copy()

    # Create missing value flags BEFORE imputation
    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Cabin_Missing'] = df['Cabin'].isnull().astype(int)
    df['Embarked_Missing'] = df['Embarked'].isnull().astype(int)

    # Title extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                        'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df['Title'] = df['Title'].map(title_mapping).fillna(0)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    # Impute Age by Title and Pclass median
    for title in df['Title'].unique():
        for pclass in df['Pclass'].unique():
            mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isnull())
            median_age = df[(df['Title'] == title) & (df['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df.loc[mask, 'Age'] = median_age

    # Fill any remaining missing Age with overall median
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fare binning
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=False, duplicates='drop')

    # Sex encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Embarked (fill missing with mode, not used as feature but needed for consistency)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    return df

# Engineer features
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Select 9 features (safe approach)
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'SmallFamily',
            'Age_Missing', 'Embarked_Missing']

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f"\nTraining data: {len(X_train)} samples")
print(f"Features ({len(features)}): {', '.join(features)}")
print(f"Test data: {len(X_test)} samples")

print("\n" + "=" * 80)
print("OVERFITTING ANALYSIS")
print("=" * 80)

print("\nCurrent LightGBM parameters (causing gap 0.0427):")
print("  n_estimators: 100")
print("  max_depth: 3")
print("  num_leaves: 8")
print("  learning_rate: 0.05")
print("  min_child_samples: 20")
print("  reg_alpha: 0.5, reg_lambda: 0.5")

print("\nPossible overfitting causes:")
print("  1. num_leaves too high (8 leaves = complex trees)")
print("  2. Regularization too weak (0.5 for small dataset)")
print("  3. min_child_samples too low (20 samples per leaf)")
print("  4. No min_split_gain constraint")

print("\n" + "=" * 80)
print("TESTING CONSERVATIVE CONFIGURATIONS")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

configurations = [
    {
        'name': 'Current',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'num_leaves': 8,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Conservative_V1',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,  # Shallower trees
            'learning_rate': 0.05,
            'num_leaves': 4,  # Fewer leaves (2^2 = 4)
            'min_child_samples': 30,  # More samples per leaf
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,  # Stronger L1
            'reg_lambda': 1.0,  # Stronger L2
            'min_split_gain': 0.01,  # Minimum gain to split
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Conservative_V2',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'learning_rate': 0.03,  # Slower learning
            'num_leaves': 4,
            'min_child_samples': 40,  # Even more samples
            'subsample': 0.7,  # Less data per tree
            'colsample_bytree': 0.7,  # Fewer features per tree
            'reg_alpha': 1.5,  # Even stronger regularization
            'reg_lambda': 1.5,
            'min_split_gain': 0.02,
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Ultra_Conservative',
        'params': {
            'n_estimators': 50,  # Fewer trees
            'max_depth': 2,
            'learning_rate': 0.03,
            'num_leaves': 3,  # Very few leaves
            'min_child_samples': 50,  # Very high minimum
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 2.0,  # Maximum regularization
            'reg_lambda': 2.0,
            'min_split_gain': 0.05,  # High minimum gain
            'random_state': 42,
            'verbose': -1
        }
    },
    {
        'name': 'Extreme_Conservative',
        'params': {
            'n_estimators': 50,
            'max_depth': 2,
            'learning_rate': 0.02,  # Very slow
            'num_leaves': 3,
            'min_child_samples': 60,  # Extreme minimum
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 3.0,  # Extreme regularization
            'reg_lambda': 3.0,
            'min_split_gain': 0.1,  # Very high gain requirement
            'random_state': 42,
            'verbose': -1
        }
    }
]

results = []

for config in configurations:
    name = config['name']
    params = config['params']

    print(f"\n{name}:")
    print(f"  Testing parameters...")

    # Create model
    model = lgb.LGBMClassifier(**params)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Training score
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    # Calculate metrics
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_score - cv_mean

    results.append({
        'Config': name,
        'CV_Score': cv_mean,
        'CV_Std': cv_std,
        'Train_Score': train_score,
        'Gap': gap,
        'max_depth': params['max_depth'],
        'num_leaves': params['num_leaves'],
        'min_child_samples': params['min_child_samples'],
        'reg_alpha': params['reg_alpha']
    })

    print(f"  CV: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  Train: {train_score:.4f}")
    print(f"  Gap: {gap:.4f}")

    if gap < 0.01:
        print(f"  ✅ EXCELLENT - Target achieved!")
    elif gap < 0.02:
        print(f"  ✓ GOOD - Close to target")
    else:
        print(f"  ⚠️  Still high overfitting")

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print("\nAll Configurations:")
print(results_df[['Config', 'CV_Score', 'Gap']].to_string(index=False))

print("\n" + "=" * 80)
print("BEST CONFIGURATION ANALYSIS")
print("=" * 80)

# Find best by gap
best_gap_idx = results_df['Gap'].idxmin()
best_gap_config = results_df.loc[best_gap_idx]

print(f"\nLowest Overfitting Gap:")
print(f"  Config: {best_gap_config['Config']}")
print(f"  CV Score: {best_gap_config['CV_Score']:.4f}")
print(f"  Gap: {best_gap_config['Gap']:.4f}")

# Find best by CV among low-gap configs
low_gap_configs = results_df[results_df['Gap'] < 0.02]
if len(low_gap_configs) > 0:
    best_cv_idx = low_gap_configs['CV_Score'].idxmax()
    best_config = low_gap_configs.loc[best_cv_idx]

    print(f"\nBest CV among Low-Gap Configs (gap < 0.02):")
    print(f"  Config: {best_config['Config']}")
    print(f"  CV Score: {best_config['CV_Score']:.4f}")
    print(f"  Gap: {best_config['Gap']:.4f}")
else:
    print(f"\n⚠️  No configuration achieved gap < 0.02")
    # Find best compromise
    results_df['Score'] = results_df['CV_Score'] - results_df['Gap'] * 2  # Penalize gap
    best_idx = results_df['Score'].idxmax()
    best_config = results_df.loc[best_idx]

    print(f"\nBest Compromise (CV - 2*Gap):")
    print(f"  Config: {best_config['Config']}")
    print(f"  CV Score: {best_config['CV_Score']:.4f}")
    print(f"  Gap: {best_config['Gap']:.4f}")

print("\n" + "=" * 80)
print("COMPARISON WITH OTHER MODELS")
print("=" * 80)

print("\nBest LightGBM vs Current Models:")
print(f"  LightGBM ({best_config['Config']}): CV {best_config['CV_Score']:.4f}, Gap {best_config['Gap']:.4f}")
print(f"  GB: CV 0.8316, Gap 0.0314")
print(f"  RF: CV 0.8294, Gap 0.0112")
print(f"  LR: CV 0.8193, Gap 0.0022")

cv_diff_gb = best_config['CV_Score'] - 0.8316
cv_diff_rf = best_config['CV_Score'] - 0.8294

print(f"\nDifference from current models:")
print(f"  vs GB: {cv_diff_gb:+.4f}")
print(f"  vs RF: {cv_diff_rf:+.4f}")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

if best_config['CV_Score'] > 0.8316 and best_config['Gap'] < 0.02:
    print(f"\n🎯 RECOMMENDED: Replace GB with tuned LightGBM")
    print(f"   Ensemble: RF + LightGBM ({best_config['Config']}) + LR")
    print(f"   Reason: Better CV than GB with low overfitting")
    recommendation = f"Use LightGBM ({best_config['Config']})"
elif best_config['CV_Score'] > 0.8294 and best_config['Gap'] < 0.02:
    print(f"\n🎯 RECOMMENDED: Consider LightGBM as alternative to GB")
    print(f"   LightGBM better than RF, lower overfitting than GB")
    recommendation = f"Consider LightGBM ({best_config['Config']})"
elif best_config['Gap'] < 0.015:
    print(f"\n✓ LightGBM tuning successful (gap reduced)")
    print(f"   But CV still below current models")
    print(f"   Gap improved: 0.0427 → {best_config['Gap']:.4f}")
    recommendation = "Gap reduced but keep current ensemble"
else:
    print(f"\n❌ LightGBM still not suitable")
    print(f"   Gap: {best_config['Gap']:.4f} (target: < 0.01)")
    print(f"   CV: {best_config['CV_Score']:.4f} (below GB and RF)")
    recommendation = "Keep current ensemble (RF + GB + LR)"

print(f"\n📊 Final Recommendation: {recommendation}")

print("\n" + "=" * 80)
print("PARAMETER INSIGHTS")
print("=" * 80)

print("\nKey parameters for reducing overfitting:")
print("  1. num_leaves: Lower is better (8 → 3-4)")
print("  2. max_depth: Shallow trees (3 → 2)")
print("  3. min_child_samples: Higher is better (20 → 40-60)")
print("  4. Regularization: Stronger is better (0.5 → 2.0-3.0)")
print("  5. min_split_gain: Add constraint (0 → 0.02-0.1)")

print("\nGap reduction achieved:")
original_gap = results_df[results_df['Config'] == 'Current']['Gap'].values[0]
best_gap = best_config['Gap']
improvement = original_gap - best_gap
improvement_pct = (improvement / original_gap) * 100

print(f"  Original: {original_gap:.4f}")
print(f"  Best: {best_gap:.4f}")
print(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")

print("\n" + "=" * 80)
print("TUNING COMPLETE!")
print("=" * 80)
