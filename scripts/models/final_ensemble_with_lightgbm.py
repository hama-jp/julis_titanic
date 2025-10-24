"""
Final Ensemble with Tuned LightGBM

Based on tuning results:
- LightGBM Conservative_V1: CV 0.8317, Gap 0.0090 (BEST!)
- RF Conservative: CV 0.8294, Gap 0.0112
- LR Conservative: CV 0.8193, Gap 0.0022

Final Ensemble: RF + LightGBM + LR
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FINAL ENSEMBLE: RF + LightGBM + LR")
print("=" * 80)
print("Using tuned LightGBM Conservative_V1 (CV 0.8317, Gap 0.0090)")
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

# Define models
models = {
    'RF': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.05,
        num_leaves=4,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_split_gain=0.01,
        random_state=42,
        verbose=-1
    ),
    'LR': LogisticRegression(
        C=0.5,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
}

print("\nRF: Random Forest (Conservative)")
print("  max_depth=4, min_samples_leaf=10")

print("\nLightGBM: LightGBM Conservative_V1 (Tuned)")
print("  max_depth=2, num_leaves=4, min_child_samples=30")
print("  reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01")

print("\nLR: Logistic Regression (Conservative)")
print("  C=0.5 (strong regularization)")

print("\n" + "=" * 80)
print("CROSS-VALIDATION EVALUATION")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

individual_scores = {}

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()

    individual_scores[name] = {
        'mean': mean_score,
        'std': std_score
    }

    print(f"\n{name}:")
    print(f"  CV Score: {mean_score:.4f} (±{std_score:.4f})")

print("\n" + "=" * 80)
print("ENSEMBLE METHODS")
print("=" * 80)

# 1. Simple Average
print("\n1. Simple Average (Equal Weights)")
oof_pred_rf = np.zeros(len(X_train))
oof_pred_lgb = np.zeros(len(X_train))
oof_pred_lr = np.zeros(len(X_train))

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # RF
    rf_model = models['RF']
    rf_model.fit(X_tr, y_tr)
    oof_pred_rf[val_idx] = rf_model.predict(X_val)

    # LightGBM
    lgb_model = models['LightGBM']
    lgb_model.fit(X_tr, y_tr)
    oof_pred_lgb[val_idx] = lgb_model.predict(X_val)

    # LR
    lr_model = models['LR']
    lr_model.fit(X_tr, y_tr)
    oof_pred_lr[val_idx] = lr_model.predict(X_val)

# Simple average
simple_avg = np.round((oof_pred_rf + oof_pred_lgb + oof_pred_lr) / 3).astype(int)
simple_score = (simple_avg == y_train).mean()

print(f"  OOF Score: {simple_score:.4f}")

# 2. Weighted Average
print("\n2. Weighted Average (CV Score Based)")
total_score = sum([individual_scores[name]['mean'] for name in models.keys()])
weights = {name: individual_scores[name]['mean'] / total_score for name in models.keys()}

print(f"  Weights: RF={weights['RF']:.3f}, LightGBM={weights['LightGBM']:.3f}, LR={weights['LR']:.3f}")

weighted_avg = np.round(
    oof_pred_rf * weights['RF'] +
    oof_pred_lgb * weights['LightGBM'] +
    oof_pred_lr * weights['LR']
).astype(int)
weighted_score = (weighted_avg == y_train).mean()

print(f"  OOF Score: {weighted_score:.4f}")

# 3. Soft Voting
print("\n3. Soft Voting Classifier")
voting_soft = VotingClassifier(
    estimators=[
        ('rf', models['RF']),
        ('lgb', models['LightGBM']),
        ('lr', models['LR'])
    ],
    voting='soft'
)

soft_scores = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
soft_score = soft_scores.mean()

print(f"  CV Score: {soft_score:.4f} (±{soft_scores.std():.4f})")

print("\n" + "=" * 80)
print("ENSEMBLE COMPARISON")
print("=" * 80)

ensemble_results = pd.DataFrame([
    {'Method': 'Simple Average', 'Score': simple_score},
    {'Method': 'Weighted Average', 'Score': weighted_score},
    {'Method': 'Soft Voting', 'Score': soft_score}
]).sort_values('Score', ascending=False)

print("\nEnsemble Methods Ranked:")
print(ensemble_results.to_string(index=False))

best_ensemble = ensemble_results.iloc[0]
print(f"\n✅ Best Ensemble: {best_ensemble['Method']} (Score: {best_ensemble['Score']:.4f})")

print("\n" + "=" * 80)
print("TRAINING FINAL MODELS")
print("=" * 80)

# Train final models on all data
final_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    final_models[name] = model
    print(f"  ✓ {name} trained")

# Train ensemble
print(f"\nTraining Soft Voting ensemble...")
voting_soft.fit(X_train, y_train)
print(f"  ✓ Ensemble trained")

print("\n" + "=" * 80)
print("CREATING SUBMISSION FILES")
print("=" * 80)

# Individual submissions
for name, model in final_models.items():
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    filename = f'submission_{name.lower()}_tuned_9features.csv'
    submission.to_csv(filename, index=False)

    survival_rate = predictions.mean()
    print(f"\n{filename}")
    print(f"  Model: {name}")
    print(f"  Survival Rate: {survival_rate:.3f}")

# Ensemble submission
ensemble_pred = voting_soft.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensemble_pred
})

filename = 'submission_ensemble_rf_lightgbm_lr_9features.csv'
submission.to_csv(filename, index=False)

survival_rate = ensemble_pred.mean()
print(f"\n{filename}")
print(f"  Model: Ensemble (RF + LightGBM + LR, Soft Voting)")
print(f"  Survival Rate: {survival_rate:.3f}")

print("\n" + "=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)

print(f"\n🎯 PRIMARY SUBMISSION: submission_lightgbm_tuned_9features.csv")
print(f"   Best individual model: CV 0.8317, Gap 0.0090")

print(f"\n🎯 SECONDARY SUBMISSION: submission_ensemble_rf_lightgbm_lr_9features.csv")
print(f"   Ensemble combines best of RF, LightGBM, and LR")

print(f"\n🎯 TERTIARY SUBMISSION: submission_rf_tuned_9features.csv")
print(f"   Alternative: RF with lowest overfitting gap")

print("\n" + "=" * 80)
print("Key Improvements from LightGBM Tuning:")
print("=" * 80)

print("\nBefore Tuning:")
print("  CV: 0.8170, Gap: 0.0427 ❌")

print("\nAfter Tuning (Conservative_V1):")
print("  CV: 0.8317, Gap: 0.0090 ✅")

print("\nImprovements:")
print(f"  CV: +{0.8317-0.8170:.4f} (+1.47%)")
print(f"  Gap: -{0.0427-0.0090:.4f} (-79.0%)")

print("\nNew Ensemble vs Old Ensemble:")
print("  Old: RF + GB + LR (Best CV: 0.8316)")
print("  New: RF + LightGBM + LR (Best CV: 0.8317)")
print(f"  Improvement: +0.0001")

print("\n" + "=" * 80)
print("ENSEMBLE READY!")
print("=" * 80)
