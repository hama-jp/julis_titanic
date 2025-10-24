"""
Ensemble Models - 9 Features (Safe Approach)

Combine predictions from RF, GB, and LR models to create
more robust predictions through ensemble learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('ENSEMBLE MODELS - 9 FEATURES (SAFE APPROACH)')
print('='*80)
print('Strategy: Combine RF, GB, and LR predictions for robust results')
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

    # Age imputation
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fare imputation and binning
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    # Embarked
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

# 9 FEATURES (Safe approach)
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'SmallFamily',
            'Age_Missing', 'Embarked_Missing']

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f'\nTraining data: {len(X_train)} samples')
print(f'Features ({len(features)}): {", ".join(features)}')
print(f'Test data: {len(X_test)} samples')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# DEFINE MODELS
# ============================================================================
print('\n' + '='*80)
print('DEFINING BASE MODELS')
print('='*80)

# Conservative parameters (same as before)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42
)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

lr_model = LogisticRegression(
    C=0.5,
    penalty='l2',
    max_iter=1000,
    random_state=42
)

print('\n1. Random Forest: max_depth=4, min_samples_leaf=10')
print('2. Gradient Boosting: max_depth=3, learning_rate=0.05')
print('3. Logistic Regression: C=0.5')

# ============================================================================
# INDIVIDUAL MODEL PERFORMANCE
# ============================================================================
print('\n' + '='*80)
print('INDIVIDUAL MODEL CV SCORES')
print('='*80)

models = {
    'RF': rf_model,
    'GB': gb_model,
    'LR': lr_model
}

individual_scores = {}
individual_preds = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    oof_preds = cross_val_predict(model, X_train, y_train, cv=cv)

    individual_scores[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'oof': oof_preds
    }

    print(f'{name}: {scores.mean():.4f} (±{scores.std():.4f})')

# ============================================================================
# ENSEMBLE METHODS
# ============================================================================
print('\n' + '='*80)
print('ENSEMBLE METHODS')
print('='*80)

ensemble_results = []

# Method 1: Simple Average (Equal weights)
print('\n1. Simple Average (Equal Weights)')
print('   Averaging predictions from all 3 models')

oof_avg = np.round((individual_scores['RF']['oof'] +
                    individual_scores['GB']['oof'] +
                    individual_scores['LR']['oof']) / 3).astype(int)
avg_score = accuracy_score(y_train, oof_avg)

ensemble_results.append({
    'method': 'Simple Average',
    'score': avg_score,
    'description': 'Equal weights (1/3 each)'
})

print(f'   OOF Score: {avg_score:.4f}')

# Method 2: Weighted Average (CV score based)
print('\n2. Weighted Average (CV Score Based)')
print('   Weights proportional to CV performance')

# Calculate weights based on CV scores
total_score = sum([individual_scores[name]['mean'] for name in models.keys()])
weights = {name: individual_scores[name]['mean'] / total_score for name in models.keys()}

print(f'   Weights: RF={weights["RF"]:.3f}, GB={weights["GB"]:.3f}, LR={weights["LR"]:.3f}')

oof_weighted = np.round(
    individual_scores['RF']['oof'] * weights['RF'] +
    individual_scores['GB']['oof'] * weights['GB'] +
    individual_scores['LR']['oof'] * weights['LR']
).astype(int)
weighted_score = accuracy_score(y_train, oof_weighted)

ensemble_results.append({
    'method': 'Weighted Average',
    'score': weighted_score,
    'description': f'RF:{weights["RF"]:.2f}, GB:{weights["GB"]:.2f}, LR:{weights["LR"]:.2f}'
})

print(f'   OOF Score: {weighted_score:.4f}')

# Method 3: Voting Classifier (Hard Voting)
print('\n3. Hard Voting Classifier')
print('   Majority vote from 3 models')

voting_hard = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)],
    voting='hard'
)

voting_hard_scores = cross_val_score(voting_hard, X_train, y_train, cv=cv, scoring='accuracy')
voting_hard_oof = cross_val_predict(voting_hard, X_train, y_train, cv=cv)
voting_hard_score = accuracy_score(y_train, voting_hard_oof)

ensemble_results.append({
    'method': 'Hard Voting',
    'score': voting_hard_score,
    'description': 'Majority vote'
})

print(f'   OOF Score: {voting_hard_score:.4f} (±{voting_hard_scores.std():.4f})')

# Method 4: Voting Classifier (Soft Voting)
print('\n4. Soft Voting Classifier')
print('   Average of predicted probabilities')

voting_soft = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('lr', lr_model)],
    voting='soft'
)

voting_soft_scores = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
voting_soft_oof = cross_val_predict(voting_soft, X_train, y_train, cv=cv)
voting_soft_score = accuracy_score(y_train, voting_soft_oof)

ensemble_results.append({
    'method': 'Soft Voting',
    'score': voting_soft_score,
    'description': 'Average probabilities'
})

print(f'   OOF Score: {voting_soft_score:.4f} (±{voting_soft_scores.std():.4f})')

# ============================================================================
# COMPARE ENSEMBLE METHODS
# ============================================================================
print('\n' + '='*80)
print('ENSEMBLE COMPARISON')
print('='*80)

ensemble_df = pd.DataFrame(ensemble_results).sort_values('score', ascending=False)
print('\nEnsemble Methods Ranked by OOF Score:')
print(ensemble_df.to_string(index=False))

best_ensemble = ensemble_df.iloc[0]
print(f'\n✅ Best Ensemble: {best_ensemble["method"]} (OOF: {best_ensemble["score"]:.4f})')

# ============================================================================
# TRAIN FINAL MODELS AND CREATE PREDICTIONS
# ============================================================================
print('\n' + '='*80)
print('TRAINING FINAL MODELS ON ALL DATA')
print('='*80)

# Train individual models
print('\nTraining individual models...')
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Get individual predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

print('  ✓ RF trained')
print('  ✓ GB trained')
print('  ✓ LR trained')

# Train voting classifiers
print('\nTraining ensemble models...')
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

voting_hard_pred = voting_hard.predict(X_test)
voting_soft_pred = voting_soft.predict(X_test)

print('  ✓ Hard Voting trained')
print('  ✓ Soft Voting trained')

# Create ensemble predictions
avg_pred = np.round((rf_pred + gb_pred + lr_pred) / 3).astype(int)
weighted_pred = np.round(
    rf_pred * weights['RF'] +
    gb_pred * weights['GB'] +
    lr_pred * weights['LR']
).astype(int)

# ============================================================================
# CREATE SUBMISSION FILES
# ============================================================================
print('\n' + '='*80)
print('CREATING SUBMISSION FILES')
print('='*80)

submissions = [
    ('submission_ensemble_simple_avg_9features.csv', avg_pred, 'Simple Average'),
    ('submission_ensemble_weighted_avg_9features.csv', weighted_pred, 'Weighted Average'),
    ('submission_ensemble_hard_voting_9features.csv', voting_hard_pred, 'Hard Voting'),
    ('submission_ensemble_soft_voting_9features.csv', voting_soft_pred, 'Soft Voting'),
]

for filename, predictions, method in submissions:
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv(filename, index=False)

    survival_rate = predictions.mean()

    # Find corresponding OOF score
    oof_score = ensemble_df[ensemble_df['method'].str.contains(method.split()[0])]['score'].values
    oof_score_str = f'{oof_score[0]:.4f}' if len(oof_score) > 0 else 'N/A'

    print(f'\n{filename}')
    print(f'  Method: {method}')
    print(f'  OOF Score: {oof_score_str}')
    print(f'  Survival Rate: {survival_rate:.3f}')

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================
print('\n' + '='*80)
print('FINAL RECOMMENDATION')
print('='*80)

print(f"""
🎯 RECOMMENDED ENSEMBLE: submission_ensemble_{best_ensemble['method'].lower().replace(' ', '_')}_9features.csv

Best Ensemble Method: {best_ensemble['method']}
  - OOF Score: {best_ensemble['score']:.4f}
  - Description: {best_ensemble['description']}

Why Ensemble Works:
  1. ✅ Combines strengths of different algorithms
  2. ✅ RF captures non-linear patterns
  3. ✅ GB captures sequential patterns
  4. ✅ LR provides linear baseline
  5. ✅ Averaging reduces individual model errors

Comparison to Individual Models:
  - RF alone: {individual_scores['RF']['mean']:.4f}
  - GB alone: {individual_scores['GB']['mean']:.4f}
  - LR alone: {individual_scores['LR']['mean']:.4f}
  - Best Ensemble: {best_ensemble['score']:.4f}

All 4 Ensemble Submissions Generated:
  1. submission_ensemble_simple_avg_9features.csv
  2. submission_ensemble_weighted_avg_9features.csv
  3. submission_ensemble_hard_voting_9features.csv
  4. submission_ensemble_soft_voting_9features.csv

Try in order:
  1st: Best ensemble ({best_ensemble['method']})
  2nd: Other ensemble methods
  3rd: Best individual model (RF)
""")

print('\n' + '='*80)
print('ENSEMBLE MODELS READY!')
print('='*80)
