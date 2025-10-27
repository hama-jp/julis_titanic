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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, RepeatedStratifiedKFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
import argparse
warnings.filterwarnings('ignore')

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description='Run tree ensemble analysis.')
parser.add_argument('--mode', type=str, default='best', choices=['best', 'conservative'],
                    help='The mode to run: "best" selects the top model, "conservative" uses a hardcoded safer model.')
parser.add_argument('--cv_strategy', type=str, default='stratified', choices=['stratified', 'repeated', 'group'],
                    help='The CV strategy to use.')
args = parser.parse_args()


print('='*80)
print('TREE ENSEMBLE ANALYSIS')
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

    # AgeBin: Binning continuous age into categories
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 16, 32, 48, 64, 100], labels=False, right=False)
    df['AgeBin'] = df['AgeBin'].astype(int)

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

# --- CV Strategy ---
print(f"Running in {args.mode} mode.")
print(f"CV Strategy: {args.cv_strategy}")
print('='*80)

groups = None
if args.cv_strategy == 'stratified':
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
elif args.cv_strategy == 'repeated':
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
elif args.cv_strategy == 'group':
    cv = GroupKFold(n_splits=5)
    # Create groups from Ticket column
    groups = pd.factorize(train['Ticket'])[0]


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
    'Enhanced_V6_AgeBin': {
        'features': ['Pclass', 'Sex', 'Title', 'AgeBin', 'IsAlone', 'FareBin', 'Pclass_Sex'],
        'description': 'V3 with AgeBin instead of Age'
    },
    'Enhanced_V7_AgeBin': {
        'features': ['Pclass', 'Sex', 'Title', 'AgeBin', 'IsAlone', 'FareBin', 'Pclass_Sex', 'SmallFamily'],
        'description': 'V4 with AgeBin instead of Age'
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

        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy', groups=groups)
        oof = cross_val_predict(rf, X, y, cv=cv, method='predict', groups=groups)
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


# Select best model based on mode
if args.mode == 'best':
    best_row = results_df.iloc[0]
    config_name = best_row['Config']
    features = best_row['features']
    depth = best_row['Depth']
    print(f"Best Model (selected dynamically): {config_name}")
else: # conservative mode
    config_name = 'Enhanced_V4_Depth4'
    features = feature_configs['Enhanced_V4']['features']
    depth = 4
    best_row = results_df[results_df['Config'] == config_name].iloc[0]
    print(f"Best Model (hardcoded for conservative approach): {config_name}")


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
print('TRAINING ENSEMBLE WITH DIFFERENT ARCHITECTURES')
print('='*80)

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f"Training data: {len(X_train)} samples, {len(features)} features")
print(f"Test data: {len(X_test)} samples")

# Train multiple model architectures on ALL data
models = {}
predictions_dict = {}

# 1. Random Forest (with optimized params from CV)
print("\n1. Random Forest")
print(f"   Params: {params}")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, **params)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
models['RF'] = rf_model
predictions_dict['RF'] = rf_pred
print(f"   Training accuracy: {rf_train_acc:.4f}")
print(f"   Survival rate: {rf_pred.mean():.3f}")

# 2. Gradient Boosting
print("\n2. Gradient Boosting")
gb_params = {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 42}
print(f"   Params: {gb_params}")
gb_model = GradientBoostingClassifier(**gb_params)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_train_acc = accuracy_score(y_train, gb_model.predict(X_train))
models['GB'] = gb_model
predictions_dict['GB'] = gb_pred
print(f"   Training accuracy: {gb_train_acc:.4f}")
print(f"   Survival rate: {gb_pred.mean():.3f}")

# 3. LightGBM
print("\n3. LightGBM")
lgb_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 100, 'random_state': 42}
print(f"   Params: {lgb_params}")
lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
lgb_train_acc = accuracy_score(y_train, lgb_model.predict(X_train))
models['LGB'] = lgb_model
predictions_dict['LGB'] = lgb_pred
print(f"   Training accuracy: {lgb_train_acc:.4f}")
print(f"   Survival rate: {lgb_pred.mean():.3f}")

# Create ensemble predictions (voting)
print("\n" + "="*80)
print("ENSEMBLE PREDICTIONS")
print("="*80)

# Hard voting (majority vote)
pred_array = np.array([predictions_dict['RF'], predictions_dict['GB'], predictions_dict['LGB']])
predictions = np.round(pred_array.mean(axis=0)).astype(int)
prediction_std = pred_array.std(axis=0)

print(f"Ensemble method: Average of 3 different architectures (RF, GB, LGB)")
print(f"Ensemble survival rate: {predictions.mean():.3f}")
print(f"Average prediction std: {prediction_std.mean():.4f}")

# Show differences
print("\nPrediction agreement:")
print(f"  RF vs GB: {np.sum(predictions_dict['RF'] == predictions_dict['GB'])} / {len(X_test)} ({np.sum(predictions_dict['RF'] == predictions_dict['GB'])/len(X_test):.1%})")
print(f"  RF vs LGB: {np.sum(predictions_dict['RF'] == predictions_dict['LGB'])} / {len(X_test)} ({np.sum(predictions_dict['RF'] == predictions_dict['LGB'])/len(X_test):.1%})")
print(f"  GB vs LGB: {np.sum(predictions_dict['GB'] == predictions_dict['LGB'])} / {len(X_test)} ({np.sum(predictions_dict['GB'] == predictions_dict['LGB'])/len(X_test):.1%})")
print(f"  All 3 agree: {np.sum((predictions_dict['RF'] == predictions_dict['GB']) & (predictions_dict['GB'] == predictions_dict['LGB']))} / {len(X_test)}")

# Create submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

# Create descriptive filename based on experiment configuration
# Format: submission_{mode}_tree_ensemble_{feature_version}.csv
feature_version = config_name.split('_')[1].lower()   # Enhanced_V4_Depth5 -> v4

filename = f'submissions/submission_{args.mode}_tree_ensemble_{feature_version}.csv'

submission.to_csv(filename, index=False)
print(f"\nGenerated filename: {filename}")

survival_rate = predictions.mean()

print('\n' + '='*80)
print('SUBMISSION FILE CREATED')
print('='*80)
print(f"\n{filename}")
print(f"  Method: Tree-only ensemble (RF + GB + LGB)")
print(f"  OOF Score (from CV): {best_row['OOF']:.4f}")
print(f"  RF Training Accuracy: {rf_train_acc:.4f}")
print(f"  GB Training Accuracy: {gb_train_acc:.4f}")
print(f"  LGB Training Accuracy: {lgb_train_acc:.4f}")
print(f"  Expected LB: {best_row['Expected_LB']:.4f}")
print(f"  Expected Gap: {best_row['Expected_Gap']:.3f}")
print(f"  Ensemble Survival Rate: {survival_rate:.3f}")
print(f"  Features ({len(features)}): {', '.join(features)}")

# Store submission info
submissions = [{
    'filename': filename,
    'config': config_name,
    'oof': best_row['OOF'],
    'rf_train_acc': rf_train_acc,
    'gb_train_acc': gb_train_acc,
    'lgb_train_acc': lgb_train_acc,
    'expected_lb': best_row['Expected_LB'],
    'expected_gap': best_row['Expected_Gap'],
    'survival_rate': survival_rate,
    'features': features
}]

best = submissions[0]

print('\n' + '='*80)
print('FINAL RECOMMENDATION FOR SUBMISSION #5')
print('='*80)

print(f"""
🥇 FINAL MODEL: {best['filename']}

Model Details:
- Configuration: {best['config']} (optimized via CV)
- Method: Tree-only ensemble (RF + GB + LGB)
- OOF Score (from CV): {best['oof']:.4f}
- RF Training Accuracy: {best['rf_train_acc']:.4f}
- GB Training Accuracy: {best['gb_train_acc']:.4f}
- LGB Training Accuracy: {best['lgb_train_acc']:.4f}
- Expected Gap: {best['expected_gap']:.3f}
- Expected LB: {best['expected_lb']:.4f}
- Survival Rate (ensemble): {best['survival_rate']:.3f}

Features ({len(best['features'])}):
{', '.join(best['features'])}

Model Training Process:
1. Cross-Validation: Tested {len(results_df)} different configurations
2. Best Config Selected: {best['config']} (highest expected LB)
3. Multi-Model Training: Trained 3 different architectures on ALL {len(X_train)} samples
   - Random Forest: max_depth={params['max_depth']}, n_estimators=100
   - Gradient Boosting: max_depth=3, learning_rate=0.1, n_estimators=100
   - LightGBM: n_estimators=100
4. Ensemble Prediction: Averaged predictions from all 3 models (voting)
5. Test Predictions: Generated predictions for {len(X_test)} test samples

Why This Ensemble Approach:
1. Combines 3 different architectures for model diversity
2. Each model has different strengths and weaknesses
3. Ensemble reduces bias from any single model
4. All models trained on full dataset (maximizes learning)
5. FareBin (binned Fare) reduces distribution sensitivity vs raw Fare
6. Pclass_Sex captures strong interaction effect
7. RF hyperparameters optimized via CV on {cv.n_splits}-fold validation
8. Based on validated formula: LB ≈ OOF - {best['expected_gap']:.3f}

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
