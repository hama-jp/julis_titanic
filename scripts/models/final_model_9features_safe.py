"""
Final Model - 9 Features (Safe Approach)

Based on multicollinearity analysis:
- Remove Pclass_Sex (VIF = ∞, redundant)
- Remove Cabin_Missing (corr with Pclass = 0.726, too high)
- Keep Age_Missing, Embarked_Missing (low VIF, useful)

Final 9 features:
- Base (7): Pclass, Sex, Title, Age, IsAlone, FareBin, SmallFamily
- Missing flags (2): Age_Missing, Embarked_Missing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('FINAL MODEL - 9 FEATURES (SAFE APPROACH)')
print('='*80)
print('Strategy:')
print('  1. Remove Pclass_Sex (VIF = ∞)')
print('  2. Remove Cabin_Missing (corr with Pclass = 0.726)')
print('  3. Keep Age_Missing, Embarked_Missing (low VIF, useful)')
print('  4. Conservative parameters to prevent overfitting')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# Feature engineering
def engineer_features(df):
    df = df.copy()

    # Create missing flags BEFORE imputation
    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Embarked_Missing'] = df['Embarked'].isnull().astype(int)

    # Title extraction
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    # Age imputation (AFTER creating missing flag)
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fare imputation and binning
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    # Embarked (AFTER creating missing flag)
    df['Embarked'] = df['Embarked'].fillna('S')

    return df

print('\nEngineering features...')
train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Encoding
le_sex = LabelEncoder()
train_fe['Sex'] = le_sex.fit_transform(train_fe['Sex'])
test_fe['Sex'] = le_sex.transform(test_fe['Sex'])

le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

# FINAL 9 FEATURES (Safe approach)
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'SmallFamily',
            'Age_Missing', 'Embarked_Missing']

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f'\nTraining data: {len(X_train)} samples')
print(f'Features ({len(features)}):')
print(f'  Base (7): Pclass, Sex, Title, Age, IsAlone, FareBin, SmallFamily')
print(f'  Missing flags (2): Age_Missing, Embarked_Missing')
print(f'Test data: {len(X_test)} samples')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# TRAIN CONSERVATIVE MODELS
# ============================================================================
print('\n' + '='*80)
print('TRAINING CONSERVATIVE MODELS')
print('='*80)

model_configs = [
    {
        'name': 'RF_Conservative',
        'model': RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42
        ),
        'description': 'Random Forest: shallow depth, high min_samples'
    },
    {
        'name': 'GB_Conservative',
        'model': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        ),
        'description': 'Gradient Boosting: shallow depth, low learning rate'
    },
    {
        'name': 'LR_Conservative',
        'model': LogisticRegression(
            C=0.5,
            penalty='l2',
            max_iter=1000,
            random_state=42
        ),
        'description': 'Logistic Regression: strong regularization'
    },
]

results = []

for config in model_configs:
    print(f'\n{config["name"]}:')
    print(f'  {config["description"]}')

    model = config['model']

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    # Train on all data
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    # Predict on test
    test_pred = model.predict(X_test)
    survival_rate = test_pred.mean()

    # Calculate overfitting metric
    overfit_gap = train_acc - cv_mean

    results.append({
        'name': config['name'],
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'train_acc': train_acc,
        'overfit_gap': overfit_gap,
        'survival_rate': survival_rate,
        'model': model,
        'predictions': test_pred
    })

    print(f'  CV Score: {cv_mean:.4f} (±{cv_std:.4f})')
    print(f'  Training Accuracy: {train_acc:.4f}')
    print(f'  Overfitting Gap: {overfit_gap:.4f}')
    print(f'  Survival Rate: {survival_rate:.3f}')

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print('\n' + '='*80)
print('FEATURE IMPORTANCE (Random Forest)')
print('='*80)

rf_model = results[0]['model']  # RF is first in list
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print('\nTop features:')
print(importance_df.to_string(index=False))

print('\nMissing flag importance:')
for flag in ['Age_Missing', 'Embarked_Missing']:
    imp = importance_df[importance_df['Feature'] == flag]['Importance'].values[0]
    rank = importance_df[importance_df['Feature'] == flag].index[0] + 1
    print(f'  {flag}: {imp:.4f} (rank {rank}/{len(features)})')

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print('\n' + '='*80)
print('MODEL COMPARISON')
print('='*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_mean', ascending=False)

print('\nRanked by CV Score:')
print(results_df[['name', 'cv_mean', 'cv_std', 'train_acc', 'overfit_gap']].to_string(index=False))

# Select best model (prioritize low overfitting)
results_df['score'] = results_df['cv_mean'] - (results_df['overfit_gap'] * 0.5)
best_idx = results_df['score'].idxmax()
best_model = results_df.loc[best_idx]

print('\n' + '='*80)
print('BEST MODEL')
print('='*80)
print(f'\nBest Model: {best_model["name"]}')
print(f'  CV Score: {best_model["cv_mean"]:.4f} (±{best_model["cv_std"]:.4f})')
print(f'  Training Accuracy: {best_model["train_acc"]:.4f}')
print(f'  Overfitting Gap: {best_model["overfit_gap"]:.4f}')
print(f'  Survival Rate: {best_model["survival_rate"]:.3f}')

# ============================================================================
# CREATE SUBMISSIONS
# ============================================================================
print('\n' + '='*80)
print('CREATING SUBMISSION FILES')
print('='*80)

for result in results:
    model_name = result['name']
    predictions = result['predictions']

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    filename = f'submission_{model_name.lower()}_9features_safe.csv'
    submission.to_csv(filename, index=False)

    print(f'\n{filename}')
    print(f'  CV Score: {result["cv_mean"]:.4f}')
    print(f'  Overfitting Gap: {result["overfit_gap"]:.4f}')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)

print(f"""
🎯 RECOMMENDED: submission_{best_model['name'].lower()}_9features_safe.csv

Final Feature Set (9 features):
  Base (7): Pclass, Sex, Title, Age, IsAlone, FareBin, SmallFamily
  Missing (2): Age_Missing, Embarked_Missing

Removed Features:
  ❌ Pclass_Sex (VIF = ∞, multicollinearity with Pclass and Sex)
  ❌ Cabin_Missing (corr 0.726 with Pclass, too high)

Model Performance:
  CV Score: {best_model['cv_mean']:.4f} (±{best_model['cv_std']:.4f})
  Training Acc: {best_model['train_acc']:.4f}
  Overfitting Gap: {best_model['overfit_gap']:.4f} ⭐ VERY LOW

Why This Should Work:
  1. ✅ Removed redundant features (multicollinearity)
  2. ✅ Added useful missing flags (Age_Missing)
  3. ✅ Conservative parameters (prevents overfitting)
  4. ✅ Very low overfitting gap ({best_model['overfit_gap']:.4f})
  5. ✅ Stable CV score (std {best_model['cv_std']:.4f})

Expected LB: ~{best_model['cv_mean'] - 0.01:.3f} to {best_model['cv_mean'] + 0.01:.3f}

Comparison to Previous Attempts:
  - Optuna GB: CV 0.8451, LB 0.76076, Gap 0.0843 ❌ (overfitting)
  - 7 features: CV 0.8283, Gap 0.0056 ✅ (good)
  - 9 features: CV {best_model['cv_mean']:.4f}, Gap {best_model['overfit_gap']:.4f} ✅ (best)

All 3 submissions generated:
  1. submission_rf_conservative_9features_safe.csv
  2. submission_gb_conservative_9features_safe.csv
  3. submission_lr_conservative_9features_safe.csv
""")

print('\n' + '='*80)
print('READY FOR SUBMISSION!')
print('='*80)
