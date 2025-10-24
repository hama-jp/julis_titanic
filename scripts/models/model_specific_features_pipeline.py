"""
Model-Specific Feature Pipeline

Based on age feature comparison experiment results:
- LightGBM: Use age group labels (better stability, less overfitting)
- Random Forest: Use raw age values (better accuracy with deeper trees)
- Logistic Regression: Use age group labels (linear model benefits from categories)

This pipeline allows each model to use its optimal feature set.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print('='*80)
print('MODEL-SPECIFIC FEATURE PIPELINE')
print('='*80)
print('Strategy: Each model uses its optimal feature set')
print('  - LightGBM: Age group labels (Child/Adult/Elderly)')
print('  - Random Forest: Raw age values')
print('  - Gradient Boosting: Age group labels')
print('  - Logistic Regression: Age group labels')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """
    Create all possible features.
    Model-specific feature selection happens later.
    """
    df_fe = df.copy()

    # Missing flags (BEFORE imputation)
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)

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
    df_fe['SmallFamily'] = ((df_fe['FamilySize'] >= 2) & (df_fe['FamilySize'] <= 4)).astype(int)

    # Age imputation (AFTER creating missing flag)
    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    # Age group (domain knowledge based)
    bins_domain_knowledge = [0, 18, 60, 100]  # Extended upper bound to 100
    labels_domain_knowledge = [0, 1, 2]  # Child, Adult, Elderly
    df_fe['AgeGroup'] = pd.cut(
        df_fe['Age'],
        bins=bins_domain_knowledge,
        labels=labels_domain_knowledge,
        right=False
    ).astype(int)

    # Fare imputation and binning
    df_fe['Fare'] = df_fe.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df_fe['FareBin'] = pd.qcut(df_fe['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df_fe['FareBin'] = df_fe['FareBin'].astype(int)

    # Embarked (AFTER creating missing flag)
    df_fe['Embarked'] = df_fe['Embarked'].fillna('S')

    # Sex encoding
    df_fe['Sex'] = df_fe['Sex'].map({'male': 0, 'female': 1})

    # Title encoding
    le_title = LabelEncoder()
    df_fe['Title'] = le_title.fit_transform(df_fe['Title'])

    return df_fe

print('\nEngineering features...')
train_fe = engineer_features(train)
test_fe = engineer_features(test)

y_train = train['Survived']

print(f'Training data: {len(train_fe)} samples')
print(f'Test data: {len(test_fe)} samples')

# ============================================================================
# MODEL CONFIGURATIONS WITH SPECIFIC FEATURES
# ============================================================================
print('\n' + '='*80)
print('MODEL CONFIGURATIONS')
print('='*80)

# Base features (common to all models)
base_features = ['Pclass', 'Sex', 'Title', 'IsAlone', 'FareBin',
                 'SmallFamily', 'Age_Missing', 'Embarked_Missing']

# Model-specific configurations
model_configs = [
    {
        'name': 'LightGBM',
        'model': lgb.LGBMClassifier(
            n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
            reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
            learning_rate=0.05, random_state=42, verbose=-1
        ),
        'features': base_features + ['AgeGroup'],  # Use age group labels
        'description': 'LightGBM with age group labels (shallow trees)',
        'age_feature': 'AgeGroup'
    },
    {
        'name': 'RandomForest',
        'model': RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10,
            min_samples_leaf=5, random_state=42
        ),
        'features': base_features + ['Age'],  # Use raw age
        'description': 'Random Forest with raw age values (deeper trees)',
        'age_feature': 'Age'
    },
    {
        'name': 'GradientBoosting',
        'model': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10,
            subsample=0.8, random_state=42
        ),
        'features': base_features + ['AgeGroup'],  # Use age group labels
        'description': 'Gradient Boosting with age group labels (shallow trees)',
        'age_feature': 'AgeGroup'
    },
    {
        'name': 'LogisticRegression',
        'model': LogisticRegression(
            C=0.5, penalty='l2', max_iter=1000, random_state=42
        ),
        'features': base_features + ['AgeGroup'],  # Use age group labels
        'description': 'Logistic Regression with age group labels (linear model)',
        'age_feature': 'AgeGroup'
    },
]

print('\nModel-Specific Feature Sets:')
for i, config in enumerate(model_configs, 1):
    print(f'\n{i}. {config["name"]}:')
    print(f'   {config["description"]}')
    print(f'   Age feature: {config["age_feature"]}')
    print(f'   Total features: {len(config["features"])}')
    print(f'   Features: {", ".join(config["features"])}')

# ============================================================================
# CROSS-VALIDATION FOR EACH MODEL
# ============================================================================
print('\n' + '='*80)
print('CROSS-VALIDATION RESULTS')
print('='*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for config in model_configs:
    print(f'\n{config["name"]}:')
    print(f'  Features: {len(config["features"])} ({config["age_feature"]} for age)')

    # Get model-specific features
    X_train_model = train_fe[config['features']]
    X_test_model = test_fe[config['features']]

    model = config['model']

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Train on all data
    model.fit(X_train_model, y_train)
    train_pred = model.predict(X_train_model)
    train_acc = accuracy_score(y_train, train_pred)

    # Predict on test
    test_pred = model.predict(X_test_model)
    test_pred_proba = model.predict_proba(X_test_model)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate overfitting gap
    overfit_gap = train_acc - cv_mean

    results.append({
        'name': config['name'],
        'age_feature': config['age_feature'],
        'n_features': len(config['features']),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_acc': train_acc,
        'overfit_gap': overfit_gap,
        'model': model,
        'predictions': test_pred,
        'predictions_proba': test_pred_proba,
        'features': config['features']
    })

    print(f'  CV Score: {cv_mean:.5f} (±{cv_std:.5f})')
    print(f'  Train Accuracy: {train_acc:.5f}')
    print(f'  Overfitting Gap: {overfit_gap:.5f}')

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print('\n' + '='*80)
print('MODEL COMPARISON')
print('='*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_mean', ascending=False)

print('\nRanked by CV Score:')
print(results_df[['name', 'age_feature', 'n_features', 'cv_mean', 'cv_std', 'overfit_gap']].to_string(index=False))

# Best model
best_idx = results_df['cv_mean'].idxmax()
best_model = results_df.loc[best_idx]

print(f'\n【Best Individual Model】')
print(f'Model: {best_model["name"]}')
print(f'Age Feature: {best_model["age_feature"]}')
print(f'CV Score: {best_model["cv_mean"]:.5f} (±{best_model["cv_std"]:.5f})')
print(f'Overfitting Gap: {best_model["overfit_gap"]:.5f}')

# ============================================================================
# ENSEMBLE PREDICTIONS
# ============================================================================
print('\n' + '='*80)
print('ENSEMBLE PREDICTIONS')
print('='*80)

# Extract predictions from each model
lgb_pred = results_df[results_df['name'] == 'LightGBM']['predictions'].values[0]
rf_pred = results_df[results_df['name'] == 'RandomForest']['predictions'].values[0]
gb_pred = results_df[results_df['name'] == 'GradientBoosting']['predictions'].values[0]
lr_pred = results_df[results_df['name'] == 'LogisticRegression']['predictions'].values[0]

# Extract probabilities for soft voting
lgb_proba = results_df[results_df['name'] == 'LightGBM']['predictions_proba'].values[0]
rf_proba = results_df[results_df['name'] == 'RandomForest']['predictions_proba'].values[0]
gb_proba = results_df[results_df['name'] == 'GradientBoosting']['predictions_proba'].values[0]
lr_proba = results_df[results_df['name'] == 'LogisticRegression']['predictions_proba'].values[0]

# Method 1: Hard Voting (majority vote)
hard_voting_pred = np.round((lgb_pred + rf_pred + gb_pred + lr_pred) / 4).astype(int)

print('\n1. Hard Voting (Majority Vote):')
print(f'   All 4 models contribute equally')
print(f'   Survival Rate: {hard_voting_pred.mean():.3f}')

# Method 2: Soft Voting (average probabilities)
soft_voting_proba = (lgb_proba + rf_proba + gb_proba + lr_proba) / 4
soft_voting_pred = (soft_voting_proba > 0.5).astype(int)

print('\n2. Soft Voting (Average Probabilities):')
print(f'   Average of predicted probabilities')
print(f'   Survival Rate: {soft_voting_pred.mean():.3f}')

# Method 3: Weighted by CV score
weights = results_df['cv_mean'].values / results_df['cv_mean'].sum()
weighted_pred = np.round(
    lgb_pred * weights[0] + rf_pred * weights[1] +
    gb_pred * weights[2] + lr_pred * weights[3]
).astype(int)

print('\n3. Weighted Voting (CV Score Based):')
print(f'   Weights: LGB={weights[0]:.3f}, RF={weights[1]:.3f}, GB={weights[2]:.3f}, LR={weights[3]:.3f}')
print(f'   Survival Rate: {weighted_pred.mean():.3f}')

# ============================================================================
# CREATE SUBMISSION FILES
# ============================================================================
print('\n' + '='*80)
print('CREATING SUBMISSION FILES')
print('='*80)

timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')

# Individual model submissions
for result in results:
    filename = f'submissions/{timestamp}_{result["name"].lower()}_modelspecific.csv'
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': result['predictions']
    })
    submission.to_csv(filename, index=False)

    print(f'\n{filename}')
    print(f'  Model: {result["name"]}')
    print(f'  Age Feature: {result["age_feature"]}')
    print(f'  CV Score: {result["cv_mean"]:.5f}')
    print(f'  Overfitting Gap: {result["overfit_gap"]:.5f}')

# Ensemble submissions
ensemble_submissions = [
    (f'submissions/{timestamp}_ensemble_hard_voting_modelspecific.csv',
     hard_voting_pred, 'Hard Voting'),
    (f'submissions/{timestamp}_ensemble_soft_voting_modelspecific.csv',
     soft_voting_pred, 'Soft Voting'),
    (f'submissions/{timestamp}_ensemble_weighted_modelspecific.csv',
     weighted_pred, 'Weighted Voting'),
]

for filename, predictions, method in ensemble_submissions:
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv(filename, index=False)

    print(f'\n{filename}')
    print(f'  Method: {method}')
    print(f'  Survival Rate: {predictions.mean():.3f}')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)

print(f"""
🎯 MODEL-SPECIFIC FEATURE ENGINEERING APPLIED

Key Innovation:
  - Each model uses its optimal age feature representation
  - LightGBM: AgeGroup (CV: {results_df[results_df['name'] == 'LightGBM']['cv_mean'].values[0]:.5f})
  - RandomForest: Age (CV: {results_df[results_df['name'] == 'RandomForest']['cv_mean'].values[0]:.5f})
  - GradientBoosting: AgeGroup (CV: {results_df[results_df['name'] == 'GradientBoosting']['cv_mean'].values[0]:.5f})
  - LogisticRegression: AgeGroup (CV: {results_df[results_df['name'] == 'LogisticRegression']['cv_mean'].values[0]:.5f})

Benefits:
  1. ✅ Each model uses features suited to its architecture
  2. ✅ Increased model diversity for better ensembles
  3. ✅ Based on empirical evidence from age feature experiment
  4. ✅ Maintains stability with conservative parameters

Best Individual Model:
  - {best_model['name']} with {best_model['age_feature']}
  - CV Score: {best_model['cv_mean']:.5f} (±{best_model['cv_std']:.5f})
  - Overfitting Gap: {best_model['overfit_gap']:.5f}

Recommended Submissions (in order):
  1. Ensemble Soft Voting (leverages probability diversity)
  2. Ensemble Hard Voting (simple majority vote)
  3. {best_model['name']} (best individual model)

Files Generated:
  - 4 individual model submissions
  - 3 ensemble submissions
  - Total: 7 submission files in submissions/

Next Steps:
  1. Submit ensemble_soft_voting first (expected best performance)
  2. Compare with previous submissions
  3. Consider adding more model-specific optimizations
""")

print('\n' + '='*80)
print('PIPELINE COMPLETE!')
print('='*80)
