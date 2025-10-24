"""
XGBoost Comparison with All Models

Compare XGBoost with RF, GB, LightGBM, and LR to find the best
ensemble configuration.

XGBoost Advantages:
- Regularization built-in (L1/L2)
- Tree pruning (max_depth then prune back)
- Parallel processing
- Strong performance on tabular data
- Often wins Kaggle competitions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBOOST COMPARISON - 9 FEATURES")
print("=" * 80)
print("Comparing XGBoost with all models")
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
    'XGBoost': {
        'model': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,  # Similar to min_samples_leaf
            reg_alpha=0.5,  # L1 regularization
            reg_lambda=0.5,  # L2 regularization
            gamma=0.1,  # Minimum loss reduction for split
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        'description': 'XGBoost (Conservative)'
    },
    'LightGBM': {
        'model': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
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
print("XGBOOST ANALYSIS")
print("=" * 80)

xgb_row = results_df[results_df['Model'] == 'XGBoost'].iloc[0]
gb_row = results_df[results_df['Model'] == 'GB'].iloc[0]
rf_row = results_df[results_df['Model'] == 'RF'].iloc[0]

print(f"\nXGBoost vs GradientBoosting:")
cv_diff_gb = xgb_row['CV_Score'] - gb_row['CV_Score']
gap_diff_gb = xgb_row['Overfitting_Gap'] - gb_row['Overfitting_Gap']
print(f"  CV Score: {xgb_row['CV_Score']:.4f} vs {gb_row['CV_Score']:.4f}")
print(f"  Difference: {cv_diff_gb:+.4f}")
print(f"  Overfitting Gap: {xgb_row['Overfitting_Gap']:.4f} vs {gb_row['Overfitting_Gap']:.4f}")
print(f"  Gap Difference: {gap_diff_gb:+.4f}")

print(f"\nXGBoost vs Random Forest:")
cv_diff_rf = xgb_row['CV_Score'] - rf_row['CV_Score']
gap_diff_rf = xgb_row['Overfitting_Gap'] - rf_row['Overfitting_Gap']
print(f"  CV Score: {xgb_row['CV_Score']:.4f} vs {rf_row['CV_Score']:.4f}")
print(f"  Difference: {cv_diff_rf:+.4f}")
print(f"  Overfitting Gap: {xgb_row['Overfitting_Gap']:.4f} vs {rf_row['Overfitting_Gap']:.4f}")
print(f"  Gap Difference: {gap_diff_rf:+.4f}")

# Determine XGBoost quality
xgb_gap = xgb_row['Overfitting_Gap']
print(f"\nXGBoost Overfitting Assessment:")
print(f"  Gap: {xgb_gap:.4f}")
if xgb_gap < 0.01:
    print(f"  ✅ EXCELLENT - Very low overfitting!")
    overfitting_status = "excellent"
elif xgb_gap < 0.03:
    print(f"  ✓ GOOD - Low overfitting")
    overfitting_status = "good"
else:
    print(f"  ⚠️  WARNING - Significant overfitting")
    overfitting_status = "poor"

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find top 3 models by CV score
top3 = results_df.head(3)

print("\nTop 3 Models by CV Score:")
for idx, row in top3.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['CV_Score']:.4f} (gap: {row['Overfitting_Gap']:.4f})")

# Recommendation logic
xgb_rank = results_df[results_df['Model'] == 'XGBoost'].index[0] + 1

if xgb_rank == 1 and xgb_gap < 0.03:
    recommendation = "Use XGBoost as primary model"
    print(f"\n🎯 {recommendation}")
    print(f"   XGBoost is the best model with acceptable overfitting")
elif xgb_rank <= 2 and xgb_gap < 0.03:
    recommendation = "Include XGBoost in ensemble"
    print(f"\n🎯 {recommendation}")
    print(f"   XGBoost is top-2 with acceptable overfitting")
elif xgb_gap < 0.01:
    recommendation = "Consider XGBoost (very low overfitting)"
    print(f"\n🎯 {recommendation}")
    print(f"   XGBoost has excellent generalization despite lower CV")
else:
    recommendation = "Do not use XGBoost"
    print(f"\n❌ {recommendation}")
    print(f"   XGBoost does not provide sufficient benefit")

print("\n" + "=" * 80)
print("ENSEMBLE STRATEGY")
print("=" * 80)

# Get low-overfitting models (gap < 0.02)
low_overfitting = results_df[results_df['Overfitting_Gap'] < 0.02]
print(f"\nModels with Low Overfitting (gap < 0.02):")
for idx, row in low_overfitting.iterrows():
    print(f"  ✓ {row['Model']}: CV {row['CV_Score']:.4f}, gap {row['Overfitting_Gap']:.4f}")

# Get high CV models (CV > 0.82)
high_cv = results_df[results_df['CV_Score'] > 0.82]
print(f"\nModels with High CV Score (> 0.82):")
for idx, row in high_cv.iterrows():
    print(f"  ✓ {row['Model']}: CV {row['CV_Score']:.4f}, gap {row['Overfitting_Gap']:.4f}")

# Determine best ensemble
xgb_in_top3 = 'XGBoost' in top3['Model'].values
xgb_low_overfitting = xgb_gap < 0.02

print("\n" + "=" * 80)
print("BEST ENSEMBLE CONFIGURATION")
print("=" * 80)

if xgb_rank == 1 and xgb_low_overfitting:
    ensemble = "RF + XGBoost + LR"
    reason = "XGBoost is best model with low overfitting"
elif xgb_in_top3 and xgb_low_overfitting:
    # Check if XGBoost is better than GB
    if cv_diff_gb > 0:
        ensemble = "RF + XGBoost + LR"
        reason = "XGBoost better than GB, low overfitting"
    else:
        ensemble = "RF + GB + XGBoost + LR"
        reason = "Include both GB and XGBoost for diversity"
elif not xgb_low_overfitting:
    ensemble = "RF + GB + LR"
    reason = "XGBoost overfitting too high, exclude"
else:
    ensemble = "RF + GB + LR"
    reason = "Current ensemble is optimal"

print(f"\n🎯 RECOMMENDED ENSEMBLE: {ensemble}")
print(f"   Reason: {reason}")

# Show expected performance
ensemble_models = ensemble.replace(' + ', ', ').replace('+', ',').split(', ')
ensemble_models = [m.strip() for m in ensemble_models]
ensemble_avg_cv = results_df[results_df['Model'].isin(ensemble_models)]['CV_Score'].mean()
ensemble_max_gap = results_df[results_df['Model'].isin(ensemble_models)]['Overfitting_Gap'].max()

print(f"\n📊 Expected Ensemble Performance:")
print(f"   Average CV: {ensemble_avg_cv:.4f}")
print(f"   Max Overfitting Gap: {ensemble_max_gap:.4f}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)

if 'XGBoost' in ensemble:
    print("\n✅ Create ensemble including XGBoost")
    print("   Generate submission files with XGBoost")
else:
    print("\n✓ Keep current ensemble (RF + GB + LR)")
    print("   XGBoost does not improve ensemble")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE!")
print("=" * 80)
