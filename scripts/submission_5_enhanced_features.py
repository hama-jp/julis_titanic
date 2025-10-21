"""
Submission #5: Enhanced Features Model

Based on Submission #4 results:
- #4: LB 0.77751, OOF 0.8058, Gap 0.0283
- Formula confirmed: LB ≈ OOF - 0.028 (for no-Fare RF)

Strategy:
- Add FareBin (binned Fare) instead of raw Fare
- Add Pclass_Sex interaction feature
- Target OOF: 0.820-0.825
- Expected gap: 0.032-0.035
- Expected LB: 0.787-0.792 (beats current best 0.7799!)

Features: Pclass, Sex, Title, Age, IsAlone, FareBin, Pclass_Sex (7)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('SUBMISSION #5: ENHANCED FEATURES MODEL')
print('='*80)
print('Goal: Beat current best (LB 0.7799)')
print('Strategy: Add FareBin + Pclass_Sex interaction')
print('Expected: OOF 0.820-0.825, LB 0.787-0.792')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

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

    # Age imputation
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['IsElderly'] = (df['Age'] >= 60).astype(int)

    # Fare imputation and binning
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # FareBin: Quartile-based binning (more robust than raw Fare)
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

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
print('TESTING ENHANCED FEATURE SETS')
print('='*80)

# Feature set configurations
feature_configs = {
    'Enhanced_V1': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin'],
        'description': 'Base + FareBin'
    },
    'Enhanced_V2': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'Pclass_Sex'],
        'description': 'Base + Pclass_Sex interaction'
    },
    'Enhanced_V3': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex'],
        'description': 'Base + FareBin + Pclass_Sex (RECOMMENDED)'
    },
    'Enhanced_V4': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex', 'SmallFamily'],
        'description': 'V3 + SmallFamily'
    },
    'Enhanced_V5': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex', 'IsChild'],
        'description': 'V3 + IsChild'
    },
}

# Test configurations with different RF depths
rf_configs = [
    {'name': 'Depth4', 'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 5},
    {'name': 'Depth5', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 4},
    {'name': 'Depth6', 'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 4},
]

results = []

for feat_name, feat_config in feature_configs.items():
    features = feat_config['features']
    X = train_fe[features]
    y = train_fe['Survived']

    print(f'\n--- Testing {feat_name}: {feat_config["description"]} ---')
    print(f'Features ({len(features)}): {", ".join(features)}')

    for rf_config in rf_configs:
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf']
        )

        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
        oof = cross_val_predict(rf, X, y, cv=cv, method='predict')
        oof_acc = accuracy_score(y, oof)

        # Estimate gap based on #4 pattern
        # No Fare: gap ~0.025-0.028
        # With FareBin: gap ~0.030-0.035
        has_fare = 'FareBin' in features
        expected_gap = 0.033 if has_fare else 0.028
        expected_lb = oof_acc - expected_gap

        results.append({
            'Config': f'{feat_name}_{rf_config["name"]}',
            'FeatureSet': feat_name,
            'Depth': rf_config['max_depth'],
            'NumFeatures': len(features),
            'OOF': oof_acc,
            'CV_Std': scores.std(),
            'Expected_Gap': expected_gap,
            'Expected_LB': expected_lb,
            'features': features
        })

        print(f'  {rf_config["name"]:8s} | OOF: {oof_acc:.4f} | Std: {scores.std():.4f} | '
              f'Expected Gap: {expected_gap:.3f} | Expected LB: {expected_lb:.4f}')

print('\n' + '='*80)
print('TOP 10 MODELS BY EXPECTED LB')
print('='*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Expected_LB', ascending=False)

print(results_df.head(10)[['Config', 'NumFeatures', 'OOF', 'Expected_Gap', 'Expected_LB']].to_string(index=False))

print('\n' + '='*80)
print('SELECTING BEST MODEL FROM CV')
print('='*80)

# Select best model based on expected LB
best_row = results_df.iloc[0]
config_name = best_row['Config']
features = best_row['features']
depth = best_row['Depth']

print(f"Best Model: {config_name}")
print(f"OOF Score: {best_row['OOF']:.4f}")
print(f"Expected LB: {best_row['Expected_LB']:.4f}")
print(f"Features ({len(features)}): {', '.join(features)}")

# Get hyperparameters for best model
if depth == 4:
    params = {'max_depth': 4, 'min_samples_split': 15, 'min_samples_leaf': 5}
elif depth == 5:
    params = {'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 4}
else:
    params = {'max_depth': 6, 'min_samples_split': 10, 'min_samples_leaf': 4}

print(f"\nHyperparameters: {params}")

print('\n' + '='*80)
print('TRAINING MODELS ON ALL CV FOLDS')
print('='*80)

# Train models on each CV fold with best configuration
X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f"Training {cv.n_splits} models on CV folds...")
print(f"Features: {len(features)}, Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Store models and predictions from each fold
fold_models = []
fold_test_preds = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    # Train model on this fold's training data
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42, **params)
    model.fit(X_fold_train, y_fold_train)

    # Predict on test data
    test_pred = model.predict(X_test)
    fold_test_preds.append(test_pred)
    fold_models.append(model)

    print(f"  Fold {fold_idx}/{cv.n_splits} completed")

print(f"\n{cv.n_splits} models trained successfully")

# Ensemble predictions: Average predictions from all folds (majority voting)
fold_test_preds_array = np.array(fold_test_preds)
predictions = np.round(fold_test_preds_array.mean(axis=0)).astype(int)

print(f"\nEnsemble method: Averaging predictions from {cv.n_splits} fold models")
print(f"Individual fold predictions shape: {fold_test_preds_array.shape}")

# Also train a model on all data for comparison
print('\n' + '='*80)
print('TRAINING FINAL MODEL ON ALL DATA (for comparison)')
print('='*80)

final_rf = RandomForestClassifier(n_estimators=100, random_state=42, **params)
final_rf.fit(X_train, y_train)

train_pred = final_rf.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
all_data_predictions = final_rf.predict(X_test)

print(f"Training Accuracy (all data model): {train_acc:.4f}")
print(f"Predictions difference: {np.sum(predictions != all_data_predictions)} / {len(predictions)}")

# Create submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

filename = f'submission_{config_name}.csv'
submission.to_csv(filename, index=False)

survival_rate = predictions.mean()

print('\n' + '='*80)
print('SUBMISSION FILE CREATED')
print('='*80)
print(f"\n{filename}")
print(f"  Method: {cv.n_splits}-fold CV ensemble (averaging fold predictions)")
print(f"  OOF Score (from CV): {best_row['OOF']:.4f}")
print(f"  Training Accuracy (all data model): {train_acc:.4f}")
print(f"  Expected LB: {best_row['Expected_LB']:.4f}")
print(f"  Expected Gap: {best_row['Expected_Gap']:.3f}")
print(f"  Survival Rate (ensemble): {survival_rate:.3f}")
print(f"  Survival Rate (all data): {all_data_predictions.mean():.3f}")
print(f"  Prediction differences: {np.sum(predictions != all_data_predictions)} samples")
print(f"  Features ({len(features)}): {', '.join(features)}")

# Store submission info
submissions = [{
    'filename': filename,
    'config': config_name,
    'oof': best_row['OOF'],
    'train_acc': train_acc,
    'expected_lb': best_row['Expected_LB'],
    'expected_gap': best_row['Expected_Gap'],
    'survival_rate': survival_rate,
    'features': features
}]

best = submissions[0]

print('\n' + '='*80)
print('FINAL RECOMMENDATION FOR SUBMISSION #5')
print('='*80)

best = submissions[0]

print(f"""
🥇 FINAL MODEL: {best['filename']}

Model Details:
- Configuration: {best['config']}
- Method: {cv.n_splits}-fold CV ensemble (averaging predictions)
- OOF Score (from CV): {best['oof']:.4f}
- Training Accuracy (all data): {best['train_acc']:.4f}
- Expected Gap: {best['expected_gap']:.3f}
- Expected LB: {best['expected_lb']:.4f}
- Survival Rate (ensemble): {best['survival_rate']:.3f}

Features ({len(best['features'])}):
{', '.join(best['features'])}

Model Training Process:
1. Cross-Validation: Tested {len(results_df)} different configurations
2. Best Config Selected: {best['config']} (highest expected LB)
3. CV Fold Training: Trained {cv.n_splits} models (one per fold, each on ~80% data)
4. Ensemble Prediction: Averaged predictions from all {cv.n_splits} fold models
5. Test Predictions: Generated predictions for {len(X_test)} test samples

Why This Ensemble Approach:
1. Uses predictions from {cv.n_splits} diverse models (different train subsets)
2. Reduces overfitting through model averaging
3. More robust predictions than single model trained on all data
4. FareBin (binned Fare) reduces distribution sensitivity vs raw Fare
5. Pclass_Sex captures strong interaction effect
6. Hyperparameters optimized via CV on {cv.n_splits}-fold validation
7. Based on validated formula: LB ≈ OOF - {best['expected_gap']:.3f}

Expected Outcomes:
- Optimistic: LB 0.79-0.792 (new best!)
- Realistic: LB 0.785-0.79 (beats 0.7799)
- Pessimistic: LB 0.78-0.785 (still improvement)

Success Criteria:
- ✅ Excellent: LB ≥ 0.79
- 🟡 Good: LB = 0.78-0.79
- 🟠 Acceptable: LB ≥ 0.7799 (beats current best)
- ❌ Rethink: LB <0.7799

Next Steps After #5:
- If LB ≥ 0.79: Try minor variations (more features, different depths)
- If LB 0.78-0.79: Try ensemble or higher capacity
- If LB <0.78: May need to try GB or different approach
""")

print('\n' + '='*80)
print('DONE! Ready for submission.')
print('='*80)
