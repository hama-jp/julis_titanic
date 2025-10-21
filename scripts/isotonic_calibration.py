"""
Isotonic Calibration for Model Predictions

Isotonic Calibration improves probability estimates by mapping predicted
probabilities to actual probabilities using a monotonic (isotonic) function.

Benefits:
- Better probability estimates (more reliable confidence)
- Can improve ranking and threshold-based decisions
- Reduces overconfidence or underconfidence in predictions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ISOTONIC CALIBRATION - 9 FEATURES")
print("=" * 80)
print("Calibrating prediction probabilities for better reliability")
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

# Define base models with conservative parameters
model_configs = {
    'RF': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    ),
    'GB': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    'LR': LogisticRegression(
        C=0.5,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
}

print("\n" + "=" * 80)
print("EVALUATING CALIBRATION")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
calibration_results = []
oof_predictions_uncalib = {}
oof_predictions_calib = {}

for name, base_model in model_configs.items():
    print(f"\n{name} Model:")
    print("-" * 40)

    # Out-of-fold predictions for uncalibrated model
    oof_uncalib = np.zeros((len(X_train), 2))
    oof_calib = np.zeros((len(X_train), 2))

    # Train calibrated and uncalibrated models
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Uncalibrated model
        uncalib_model = base_model.__class__(**base_model.get_params())
        uncalib_model.fit(X_tr, y_tr)
        oof_uncalib[val_idx] = uncalib_model.predict_proba(X_val)

        # Calibrated model (isotonic)
        # Use cv=2 for internal calibration (splits training into train/calibration)
        calib_model = CalibratedClassifierCV(
            base_model.__class__(**base_model.get_params()),
            method='isotonic',
            cv=2
        )
        calib_model.fit(X_tr, y_tr)
        oof_calib[val_idx] = calib_model.predict_proba(X_val)

    # Store predictions
    oof_predictions_uncalib[name] = oof_uncalib[:, 1]
    oof_predictions_calib[name] = oof_calib[:, 1]

    # Compute metrics
    uncalib_acc = accuracy_score(y_train, (oof_uncalib[:, 1] >= 0.5).astype(int))
    calib_acc = accuracy_score(y_train, (oof_calib[:, 1] >= 0.5).astype(int))

    uncalib_brier = brier_score_loss(y_train, oof_uncalib[:, 1])
    calib_brier = brier_score_loss(y_train, oof_calib[:, 1])

    uncalib_logloss = log_loss(y_train, oof_uncalib[:, 1])
    calib_logloss = log_loss(y_train, oof_calib[:, 1])

    print(f"  Uncalibrated - Accuracy: {uncalib_acc:.4f}, Brier: {uncalib_brier:.4f}, LogLoss: {uncalib_logloss:.4f}")
    print(f"  Calibrated   - Accuracy: {calib_acc:.4f}, Brier: {calib_brier:.4f}, LogLoss: {calib_logloss:.4f}")

    brier_improvement = uncalib_brier - calib_brier
    logloss_improvement = uncalib_logloss - calib_logloss

    print(f"  Improvement  - Brier: {brier_improvement:+.4f}, LogLoss: {logloss_improvement:+.4f}")

    calibration_results.append({
        'Model': name,
        'Uncalib_Acc': uncalib_acc,
        'Calib_Acc': calib_acc,
        'Uncalib_Brier': uncalib_brier,
        'Calib_Brier': calib_brier,
        'Brier_Improvement': brier_improvement,
        'Uncalib_LogLoss': uncalib_logloss,
        'Calib_LogLoss': calib_logloss,
        'LogLoss_Improvement': logloss_improvement
    })

print("\n" + "=" * 80)
print("CALIBRATION COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(calibration_results)
print("\nMetrics Comparison:")
print(results_df.to_string(index=False))

print("\n" + "=" * 80)
print("TRAINING FINAL CALIBRATED MODELS")
print("=" * 80)

# Train final calibrated models on all data
final_models = {}
final_predictions = {}

for name, base_model in model_configs.items():
    print(f"\nTraining calibrated {name}...")

    # Use CalibratedClassifierCV with cv=5 for isotonic calibration
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method='isotonic',
        cv=5
    )

    calibrated_model.fit(X_train, y_train)
    final_models[name] = calibrated_model

    # Predict on test set
    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    final_predictions[name] = {
        'proba': test_proba,
        'pred': test_pred
    }

    survival_rate = test_pred.mean()
    print(f"  ✓ {name} calibrated")
    print(f"    Test survival rate: {survival_rate:.3f}")

print("\n" + "=" * 80)
print("CREATING CALIBRATED SUBMISSIONS")
print("=" * 80)

# Individual calibrated submissions
for name in model_configs.keys():
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': final_predictions[name]['pred']
    })

    filename = f'submission_{name.lower()}_calibrated_9features.csv'
    submission.to_csv(filename, index=False)

    survival_rate = final_predictions[name]['pred'].mean()
    print(f"\n{filename}")
    print(f"  Model: {name} (Isotonic Calibrated)")
    print(f"  Survival Rate: {survival_rate:.3f}")

# Calibrated ensemble (simple average)
ensemble_proba = (final_predictions['RF']['proba'] +
                  final_predictions['GB']['proba'] +
                  final_predictions['LR']['proba']) / 3
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensemble_pred
})

filename = 'submission_ensemble_calibrated_9features.csv'
submission.to_csv(filename, index=False)

survival_rate = ensemble_pred.mean()
print(f"\n{filename}")
print(f"  Model: Ensemble (RF+GB+LR, Isotonic Calibrated)")
print(f"  Survival Rate: {survival_rate:.3f}")

print("\n" + "=" * 80)
print("CALIBRATION ANALYSIS")
print("=" * 80)

print("\n📊 What is Isotonic Calibration?")
print("-" * 40)
print("Isotonic calibration maps predicted probabilities to actual probabilities")
print("using a non-parametric, monotonic (isotonic) function.")
print()
print("Benefits:")
print("  ✓ Better probability estimates (more reliable confidence)")
print("  ✓ Reduces overconfidence or underconfidence")
print("  ✓ Improves Brier score and log loss")
print("  ✓ More flexible than Platt scaling (sigmoid)")
print()
print("When to use:")
print("  - When predicted probabilities don't match actual frequencies")
print("  - For ensemble models that may be poorly calibrated")
print("  - When you need reliable probability estimates")

print("\n📈 Results Summary:")
print("-" * 40)
for result in calibration_results:
    name = result['Model']
    brier_imp = result['Brier_Improvement']
    logloss_imp = result['LogLoss_Improvement']

    brier_status = "✓ Better" if brier_imp > 0 else "✗ Worse"
    logloss_status = "✓ Better" if logloss_imp > 0 else "✗ Worse"

    print(f"{name}:")
    print(f"  Brier Score: {brier_imp:+.4f} {brier_status}")
    print(f"  Log Loss: {logloss_imp:+.4f} {logloss_status}")

print("\n" + "=" * 80)
print("RECOMMENDED SUBMISSION ORDER")
print("=" * 80)

# Find best calibrated model by Brier score improvement
best_idx = results_df['Brier_Improvement'].idxmax()
best_model = results_df.loc[best_idx, 'Model']
best_improvement = results_df.loc[best_idx, 'Brier_Improvement']

print(f"\n1st Priority: submission_{best_model.lower()}_calibrated_9features.csv")
print(f"   Best Brier improvement: {best_improvement:+.4f}")
print(f"\n2nd Priority: submission_ensemble_calibrated_9features.csv")
print(f"   Combines RF+GB+LR (all calibrated)")
print(f"\n3rd Priority: Compare with uncalibrated versions")
print(f"   Test if calibration improves LB score")

print("\n" + "=" * 80)
print("CALIBRATED MODELS READY!")
print("=" * 80)
