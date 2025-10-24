"""
Advanced Improvement Strategies - Post Submission #3

Current Best Results:
- Submission #2: LB 0.7799 (OOF 0.8283, Gap 0.0484) - with Fare
- Submission #3: LB 0.77272 (OOF 0.7946, Gap 0.0219) - without Fare

Key Insights:
1. Removing Fare reduced gap by 55% (0.048 → 0.022)
2. No-Fare models: LB ≈ OOF - 0.022 (very predictable)
3. Need: OOF ~0.805 to achieve LB ~0.78

Strategy:
- Test multiple approaches to find OOF ~0.80-0.82 with gap <0.03
- Focus on stable features (no raw Fare)
- Try different model complexities and ensembles
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('ADVANCED IMPROVEMENT STRATEGIES')
print('='*80)
print('Target: OOF ~0.80-0.82 with gap <0.03')
print('Expected LB: ~0.78-0.80')
print('='*80)

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Feature engineering
def engineer_features(df):
    df = df.copy()

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

    # Age imputation and features
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['IsElderly'] = (df['Age'] >= 60).astype(int)

    # Age binning
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 16, 32, 48, 100], labels=[0, 1, 2, 3])
    df['AgeGroup'] = df['AgeGroup'].cat.codes

    # Fare (for binning experiments)
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].cat.codes

    # Embarked
    df['Embarked'] = df['Embarked'].fillna('S')

    # Interaction features
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
    df['Title_Pclass'] = df['Title'].astype(str) + '_' + df['Pclass'].astype(str)

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# Encoding
le_sex = LabelEncoder()
train_fe['Sex'] = le_sex.fit_transform(train_fe['Sex'])
test_fe['Sex'] = le_sex.transform(test_fe['Sex'])

le_embarked = LabelEncoder()
train_fe['Embarked'] = le_embarked.fit_transform(train_fe['Embarked'])
test_fe['Embarked'] = le_embarked.transform(test_fe['Embarked'])

le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

le_pclass_sex = LabelEncoder()
train_fe['Pclass_Sex'] = le_pclass_sex.fit_transform(train_fe['Pclass_Sex'])
test_fe['Pclass_Sex'] = le_pclass_sex.transform(test_fe['Pclass_Sex'])

le_title_pclass = LabelEncoder()
train_fe['Title_Pclass'] = le_title_pclass.fit_transform(train_fe['Title_Pclass'])
test_fe['Title_Pclass'] = le_title_pclass.transform(test_fe['Title_Pclass'])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\n' + '='*80)
print('STRATEGY 1: Optimized Random Forest (No Fare)')
print('='*80)
print('Goal: OOF ~0.80-0.82 with same stable features as #3')

# Test different RF configurations
rf_configs = {
    'RF_Depth5': {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 4},
    'RF_Depth6': {'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 4},
    'RF_Depth4_Loose': {'max_depth': 4, 'min_samples_split': 10, 'min_samples_leaf': 3},
    'RF_Balanced': {'max_depth': 5, 'min_samples_split': 15, 'min_samples_leaf': 5},
}

# Use proven stable features from #3
stable_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone']
X_train = train_fe[stable_features]
y_train = train_fe['Survived']
X_test = test_fe[stable_features]

rf_results = []
rf_models = {}

print('\nTesting RF configurations with stable features:', stable_features)
print()

for name, params in rf_configs.items():
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        **params
    )

    scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    oof = cross_val_predict(rf, X_train, y_train, cv=cv, method='predict')
    oof_acc = accuracy_score(y_train, oof)

    # Expected LB based on #3 pattern (gap ~0.025 for no-Fare models)
    expected_lb = oof_acc - 0.025
    expected_gap = 0.025

    rf_results.append({
        'Model': name,
        'OOF': oof_acc,
        'CV_Std': scores.std(),
        'Expected_LB': expected_lb,
        'Expected_Gap': expected_gap
    })

    rf_models[name] = (rf, oof_acc)

    print(f'{name:20s} | OOF: {oof_acc:.4f} | Std: {scores.std():.4f} | Expected LB: {expected_lb:.4f}')

print('\n' + '='*80)
print('STRATEGY 2: Gradient Boosting (No Fare)')
print('='*80)
print('Goal: Alternative high-capacity model')

gb_configs = {
    'GB_Conservative': {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05},
    'GB_Medium': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
    'GB_Depth4': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.03},
}

gb_results = []
gb_models = {}

print('\nTesting GB configurations:')
print()

for name, params in gb_configs.items():
    gb = GradientBoostingClassifier(
        random_state=42,
        min_samples_split=15,
        **params
    )

    scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='accuracy')
    oof = cross_val_predict(gb, X_train, y_train, cv=cv, method='predict')
    oof_acc = accuracy_score(y_train, oof)

    expected_lb = oof_acc - 0.025
    expected_gap = 0.025

    gb_results.append({
        'Model': name,
        'OOF': oof_acc,
        'CV_Std': scores.std(),
        'Expected_LB': expected_lb,
        'Expected_Gap': expected_gap
    })

    gb_models[name] = (gb, oof_acc)

    print(f'{name:20s} | OOF: {oof_acc:.4f} | Std: {scores.std():.4f} | Expected LB: {expected_lb:.4f}')

print('\n' + '='*80)
print('STRATEGY 3: Enhanced Feature Sets')
print('='*80)
print('Goal: Test carefully selected additional features')

enhanced_feature_sets = {
    'Enhanced_V1': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'IsChild'],
    'Enhanced_V2': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'SmallFamily'],
    'Enhanced_V3': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'Pclass_Sex'],
    'Enhanced_V4': ['Pclass', 'Sex', 'Title', 'AgeGroup', 'IsAlone', 'FareBin'],  # Binned Fare
}

enhanced_results = []
enhanced_models = {}

print('\nTesting enhanced feature sets:')
print()

for name, features in enhanced_feature_sets.items():
    X = train_fe[features]
    y = train_fe['Survived']

    # Use RF with moderate parameters
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )

    scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    oof = cross_val_predict(rf, X, y, cv=cv, method='predict')
    oof_acc = accuracy_score(y, oof)

    # If using FareBin, expect slightly higher gap
    expected_gap = 0.030 if 'FareBin' in features else 0.025
    expected_lb = oof_acc - expected_gap

    enhanced_results.append({
        'Model': name,
        'Features': len(features),
        'OOF': oof_acc,
        'CV_Std': scores.std(),
        'Expected_LB': expected_lb,
        'Expected_Gap': expected_gap,
        'feature_list': features
    })

    enhanced_models[name] = (rf, oof_acc, features)

    print(f'{name:20s} | Features: {len(features)} | OOF: {oof_acc:.4f} | Expected LB: {expected_lb:.4f}')

print('\n' + '='*80)
print('STRATEGY 4: Stacking Ensemble')
print('='*80)
print('Goal: Combine multiple models for better stability')

# Base models
base_models = [
    ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5,
                                   min_samples_split=10, min_samples_leaf=4,
                                   random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                       learning_rate=0.05, random_state=42))
]

# Meta-learner
meta_learner = LogisticRegression(C=0.5, random_state=42)

# Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5
)

print('\nTesting Stacking Classifier:')
scores = cross_val_score(stacking, X_train, y_train, cv=cv, scoring='accuracy')
oof = cross_val_predict(stacking, X_train, y_train, cv=cv, method='predict')
oof_acc = accuracy_score(y_train, oof)

expected_lb = oof_acc - 0.025
print(f'Stacking OOF: {oof_acc:.4f} | CV Std: {scores.std():.4f} | Expected LB: {expected_lb:.4f}')

stacking_model = (stacking, oof_acc)

print('\n' + '='*80)
print('SUMMARY: Top Candidates for Submission #4+')
print('='*80)

all_results = []

# Add RF results
for r in rf_results:
    all_results.append({
        'Model': r['Model'],
        'Type': 'RF (No Fare)',
        'OOF': r['OOF'],
        'Expected_LB': r['Expected_LB'],
        'Expected_Gap': r['Expected_Gap'],
        'features': stable_features
    })

# Add GB results
for r in gb_results:
    all_results.append({
        'Model': r['Model'],
        'Type': 'GB (No Fare)',
        'OOF': r['OOF'],
        'Expected_LB': r['Expected_LB'],
        'Expected_Gap': r['Expected_Gap'],
        'features': stable_features
    })

# Add enhanced results
for r in enhanced_results:
    all_results.append({
        'Model': r['Model'],
        'Type': 'RF (Enhanced)',
        'OOF': r['OOF'],
        'Expected_LB': r['Expected_LB'],
        'Expected_Gap': r['Expected_Gap'],
        'features': r['feature_list']
    })

# Add stacking
all_results.append({
    'Model': 'Stacking',
    'Type': 'Ensemble',
    'OOF': oof_acc,
    'Expected_LB': expected_lb,
    'Expected_Gap': 0.025,
    'features': stable_features
})

# Sort by expected LB
all_results_df = pd.DataFrame(all_results)
all_results_df = all_results_df.sort_values('Expected_LB', ascending=False)

print('\nTop 10 Models by Expected LB Score:')
print(all_results_df.head(10)[['Model', 'Type', 'OOF', 'Expected_LB', 'Expected_Gap']].to_string(index=False))

print('\n' + '='*80)
print('CREATING SUBMISSION FILES')
print('='*80)

# Create submission for top 3 models
submissions = []

for i, row in all_results_df.head(3).iterrows():
    model_name = row['Model']
    features = row['features']

    # Get model
    if model_name in rf_models:
        model, _ = rf_models[model_name]
    elif model_name in gb_models:
        model, _ = gb_models[model_name]
    elif model_name in enhanced_models:
        model, _, _ = enhanced_models[model_name]
    elif model_name == 'Stacking':
        model, _ = stacking_model
    else:
        continue

    # Train and predict
    X_tr = train_fe[features]
    y_tr = train_fe['Survived']
    X_te = test_fe[features]

    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)

    # Create submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    filename = f'submission_Advanced_{model_name}.csv'
    submission.to_csv(filename, index=False)

    survival_rate = predictions.mean()

    submissions.append({
        'filename': filename,
        'model': model_name,
        'oof': row['OOF'],
        'expected_lb': row['Expected_LB'],
        'survival_rate': survival_rate
    })

    print(f"\n{filename}")
    print(f"  OOF: {row['OOF']:.4f} | Expected LB: {row['Expected_LB']:.4f} | Survival Rate: {survival_rate:.3f}")

print('\n' + '='*80)
print('RECOMMENDATIONS')
print('='*80)

best = all_results_df.iloc[0]
second = all_results_df.iloc[1]
third = all_results_df.iloc[2]

print(f"""
Based on analysis of Submission #3 results:

🥇 FIRST CHOICE: {best['Model']}
   - Type: {best['Type']}
   - OOF: {best['OOF']:.4f}
   - Expected LB: {best['Expected_LB']:.4f}
   - Expected Gap: {best['Expected_Gap']:.4f}
   - File: submission_Advanced_{best['Model']}.csv

🥈 SECOND CHOICE: {second['Model']}
   - Type: {second['Type']}
   - OOF: {second['OOF']:.4f}
   - Expected LB: {second['Expected_LB']:.4f}
   - File: submission_Advanced_{second['Model']}.csv

🥉 THIRD CHOICE: {third['Model']}
   - Type: {third['Type']}
   - OOF: {third['OOF']:.4f}
   - Expected LB: {third['Expected_LB']:.4f}
   - File: submission_Advanced_{third['Model']}.csv

KEY STRATEGY:
1. All models use stable features (minimal or no raw Fare)
2. Expected gaps are <0.03 (much better than Submission #2's 0.048)
3. Expected LB scores are {best['Expected_LB']:.3f}-{third['Expected_LB']:.3f}
4. Should improve over Submission #3's LB 0.77272

SUBMISSION PLAN:
- Submission #4: Use FIRST CHOICE
- If #4 successful (LB ≥ 0.78): Try minor variations
- If #4 disappointing: Try SECOND or THIRD CHOICE
- Reserve 6 submissions for final optimization
""")

print('\n' + '='*80)
print('DONE!')
print('='*80)
