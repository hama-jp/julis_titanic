"""
LightGBM vs GradientBoosting Comparison

Compare LightGBM with current models to decide if it should be included
in the ensemble.

LightGBM Advantages:
- Faster training than sklearn GradientBoosting
- Better handling of categorical features
- More efficient memory usage
- Often achieves better performance on tabular data
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("LIGHTGBM COMPARISON - 9 FEATURES")
print("=" * 80)
print("Comparing LightGBM with current models")
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
print("MODEL CONFIGURATIONS")
print("=" * 80)

# Define models with conservative parameters
model_configs = {
    'RF': {
        'model': RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42
        ),
        'description': 'Random Forest (Conservative)'
    },
    'GB': {
        'model': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
        'description': 'Gradient Boosting (Conservative)'
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=8,  # 2^3 = 8 (conservative for max_depth=3)
            min_child_samples=20,  # Similar to min_samples_leaf
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,  # L1 regularization
            reg_lambda=0.5,  # L2 regularization
            random_state=42,
            verbose=-1
        ),
        'description': 'LightGBM (Conservative)'
    },
    'LR': {
        'model': LogisticRegression(
            C=0.5,
            penalty='l2',
            max_iter=1000,
            random_state=42
        ),
        'description': 'Logistic Regression (Conservative)'
    }
}

for name, config in model_configs.items():
    print(f"\n{name}: {config['description']}")

print("\n" + "=" * 80)
print("CROSS-VALIDATION EVALUATION")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, config in model_configs.items():
    model = config['model']

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Training score
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    # Overfitting gap
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_score - cv_mean

    results.append({
        'Model': name,
        'CV_Score': cv_mean,
        'CV_Std': cv_std,
        'Train_Score': train_score,
        'Overfitting_Gap': gap
    })

    print(f"\n{name}:")
    print(f"  CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  Train Score: {train_score:.4f}")
    print(f"  Overfitting Gap: {gap:.4f}")

results_df = pd.DataFrame(results).sort_values('CV_Score', ascending=False)

print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print("\nRanked by CV Score:")
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("LIGHTGBM ANALYSIS")
print("=" * 80)

lgb_row = results_df[results_df['Model'] == 'LightGBM'].iloc[0]
gb_row = results_df[results_df['Model'] == 'GB'].iloc[0]

cv_diff = lgb_row['CV_Score'] - gb_row['CV_Score']
gap_diff = lgb_row['Overfitting_Gap'] - gb_row['Overfitting_Gap']

print(f"\nLightGBM vs GradientBoosting:")
print(f"  CV Score: {lgb_row['CV_Score']:.4f} vs {gb_row['CV_Score']:.4f}")
print(f"  Difference: {cv_diff:+.4f}")
print(f"  Overfitting Gap: {lgb_row['Overfitting_Gap']:.4f} vs {gb_row['Overfitting_Gap']:.4f}")
print(f"  Gap Difference: {gap_diff:+.4f}")

if cv_diff > 0.001:
    print(f"\n✅ LightGBM is BETTER than GradientBoosting (CV +{cv_diff:.4f})")
    recommendation = "Include LightGBM"
elif cv_diff < -0.001:
    print(f"\n❌ LightGBM is WORSE than GradientBoosting (CV {cv_diff:.4f})")
    recommendation = "Keep GradientBoosting"
else:
    print(f"\n⚖️  LightGBM and GradientBoosting are SIMILAR (CV {cv_diff:+.4f})")
    if gap_diff < 0:
        print(f"   But LightGBM has LOWER overfitting ({gap_diff:+.4f})")
        recommendation = "Prefer LightGBM (lower overfitting)"
    else:
        recommendation = "Either model is acceptable"

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

print(f"\n🎯 {recommendation}")

# Find best model
best_row = results_df.iloc[0]
print(f"\n📊 Best Model: {best_row['Model']}")
print(f"   CV Score: {best_row['CV_Score']:.4f}")
print(f"   Overfitting Gap: {best_row['Overfitting_Gap']:.4f}")

# Check overfitting
max_gap = results_df['Overfitting_Gap'].max()
print(f"\n🔍 Overfitting Check:")
print(f"   Maximum gap: {max_gap:.4f}")
if max_gap < 0.01:
    print(f"   ✅ EXCELLENT - Very low overfitting!")
elif max_gap < 0.03:
    print(f"   ✓ GOOD - Low overfitting")
else:
    print(f"   ⚠️  WARNING - Some overfitting detected")

print("\n" + "=" * 80)
print("ENSEMBLE STRATEGY")
print("=" * 80)

print("\nOption 1: Replace GB with LightGBM")
print("  Ensemble: RF + LightGBM + LR")
print("  Reason: If LightGBM is better than GB")

print("\nOption 2: Add LightGBM to existing ensemble")
print("  Ensemble: RF + GB + LightGBM + LR")
print("  Reason: More diversity, but more complex")

print("\nOption 3: Keep current ensemble")
print("  Ensemble: RF + GB + LR")
print("  Reason: If LightGBM doesn't add value")

# Determine best ensemble
if cv_diff > 0.001:
    print(f"\n🎯 RECOMMENDED: Option 1 (Replace GB with LightGBM)")
    print(f"   LightGBM has better CV score (+{cv_diff:.4f})")
    best_ensemble = "RF + LightGBM + LR"
elif abs(cv_diff) <= 0.001 and gap_diff < 0:
    print(f"\n🎯 RECOMMENDED: Option 1 (Replace GB with LightGBM)")
    print(f"   LightGBM has lower overfitting ({gap_diff:+.4f})")
    best_ensemble = "RF + LightGBM + LR"
else:
    print(f"\n🎯 RECOMMENDED: Option 3 (Keep current ensemble)")
    print(f"   GB is sufficient, no significant improvement from LightGBM")
    best_ensemble = "RF + GB + LR"

print(f"\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

print(f"\nBest Ensemble Configuration: {best_ensemble}")
print("\n1. Create ensemble with recommended configuration")
print("2. Generate submission files")
print("3. Compare with existing submissions when daily limit resets")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
