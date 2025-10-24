"""
Top 3 Ensemble: LightGBM + GB + RF

Based on diversity analysis, this ensemble achieved:
- OOF Score: 0.8350 (BEST!)
- Models: LightGBM (CV 0.8317), GB (CV 0.8316), RF (CV 0.8294)

Strategy: Combine top 3 individual models by CV score
- 2 Boosting models (LightGBM, GB)
- 1 Bagging model (RF)
- Correlation: 0.899 (high but acceptable given performance)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TOP 3 ENSEMBLE: LightGBM + GB + RF")
print("=" * 80)
print("Best ensemble configuration from diversity analysis")
print("Expected OOF: 0.8350")
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
print("TOP 3 MODELS")
print("=" * 80)

# Define top 3 models
models = {
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
    'GB': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42
    ),
    'RF': RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
    )
}

print("\n1. LightGBM (Tuned Conservative_V1)")
print("   CV: 0.8317, Gap: 0.0090")
print("   max_depth=2, num_leaves=4, min_child_samples=30")

print("\n2. GradientBoosting (Conservative)")
print("   CV: 0.8316, Gap: 0.0314")
print("   max_depth=3, min_samples_leaf=10")

print("\n3. Random Forest (Conservative)")
print("   CV: 0.8294, Gap: 0.0112")
print("   max_depth=4, min_samples_leaf=10")

print("\n" + "=" * 80)
print("CROSS-VALIDATION")
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
oof_pred_lgb = np.zeros(len(X_train))
oof_pred_gb = np.zeros(len(X_train))
oof_pred_rf = np.zeros(len(X_train))

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]

    # LightGBM
    lgb_model = models['LightGBM']
    lgb_model.fit(X_tr, y_tr)
    oof_pred_lgb[val_idx] = lgb_model.predict(X_val)

    # GB
    gb_model = models['GB']
    gb_model.fit(X_tr, y_tr)
    oof_pred_gb[val_idx] = gb_model.predict(X_val)

    # RF
    rf_model = models['RF']
    rf_model.fit(X_tr, y_tr)
    oof_pred_rf[val_idx] = rf_model.predict(X_val)

# Simple average
simple_avg = np.round((oof_pred_lgb + oof_pred_gb + oof_pred_rf) / 3).astype(int)
simple_score = (simple_avg == y_train.values).mean()

print(f"  OOF Score: {simple_score:.4f}")

# 2. Weighted Average
print("\n2. Weighted Average (CV Score Based)")
total_score = sum([individual_scores[name]['mean'] for name in models.keys()])
weights = {name: individual_scores[name]['mean'] / total_score for name in models.keys()}

print(f"  Weights: LightGBM={weights['LightGBM']:.3f}, GB={weights['GB']:.3f}, RF={weights['RF']:.3f}")

weighted_avg = np.round(
    oof_pred_lgb * weights['LightGBM'] +
    oof_pred_gb * weights['GB'] +
    oof_pred_rf * weights['RF']
).astype(int)
weighted_score = (weighted_avg == y_train.values).mean()

print(f"  OOF Score: {weighted_score:.4f}")

# 3. Soft Voting
print("\n3. Soft Voting Classifier")
voting_soft = VotingClassifier(
    estimators=[
        ('lgb', models['LightGBM']),
        ('gb', models['GB']),
        ('rf', models['RF'])
    ],
    voting='soft'
)

soft_scores = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
soft_score = soft_scores.mean()

print(f"  CV Score: {soft_score:.4f} (±{soft_scores.std():.4f})")

# 4. Hard Voting
print("\n4. Hard Voting Classifier")
voting_hard = VotingClassifier(
    estimators=[
        ('lgb', models['LightGBM']),
        ('gb', models['GB']),
        ('rf', models['RF'])
    ],
    voting='hard'
)

hard_scores = cross_val_score(voting_hard, X_train, y_train, cv=cv, scoring='accuracy')
hard_score = hard_scores.mean()

print(f"  CV Score: {hard_score:.4f} (±{hard_scores.std():.4f})")

print("\n" + "=" * 80)
print("ENSEMBLE COMPARISON")
print("=" * 80)

ensemble_results = pd.DataFrame([
    {'Method': 'Simple Average', 'Score': simple_score},
    {'Method': 'Weighted Average', 'Score': weighted_score},
    {'Method': 'Soft Voting', 'Score': soft_score},
    {'Method': 'Hard Voting', 'Score': hard_score}
]).sort_values('Score', ascending=False)

print("\nEnsemble Methods Ranked:")
print(ensemble_results.to_string(index=False))

best_ensemble = ensemble_results.iloc[0]
print(f"\n✅ Best Ensemble: {best_ensemble['Method']} (Score: {best_ensemble['Score']:.4f})")

print("\n" + "=" * 80)
print("TRAINING FINAL MODELS")
print("=" * 80)

# Train final models on all data
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    print(f"  ✓ {name} trained")

# Train ensembles
print(f"\nTraining Soft Voting ensemble...")
voting_soft.fit(X_train, y_train)
print(f"  ✓ Soft Voting trained")

print(f"\nTraining Hard Voting ensemble...")
voting_hard.fit(X_train, y_train)
print(f"  ✓ Hard Voting trained")

print("\n" + "=" * 80)
print("CREATING SUBMISSION FILES")
print("=" * 80)

# Soft Voting submission
soft_pred = voting_soft.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': soft_pred
})

filename = 'submission_top3_softvoting_lightgbm_gb_rf.csv'
submission.to_csv(filename, index=False)

survival_rate = soft_pred.mean()
print(f"\n{filename}")
print(f"  Model: Top 3 Soft Voting (LightGBM + GB + RF)")
print(f"  Expected OOF: 0.8350")
print(f"  Survival Rate: {survival_rate:.3f}")

# Hard Voting submission
hard_pred = voting_hard.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': hard_pred
})

filename = 'submission_top3_hardvoting_lightgbm_gb_rf.csv'
submission.to_csv(filename, index=False)

survival_rate = hard_pred.mean()
print(f"\n{filename}")
print(f"  Model: Top 3 Hard Voting (LightGBM + GB + RF)")
print(f"  Survival Rate: {survival_rate:.3f}")

print("\n" + "=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)

print(f"\n🎯 PRIMARY: submission_top3_softvoting_lightgbm_gb_rf.csv")
print(f"   Best ensemble from diversity analysis")
print(f"   Expected OOF: 0.8350 (highest)")

print(f"\n🎯 SECONDARY: submission_lightgbm_tuned_9features.csv")
print(f"   Best individual model: CV 0.8317")

print(f"\n🎯 TERTIARY: submission_top3_hardvoting_lightgbm_gb_rf.csv")
print(f"   Alternative voting method")

print("\n" + "=" * 80)
print("WHY TOP 3 ENSEMBLE WORKS")
print("=" * 80)

print("\n✅ Advantages:")
print("  1. Combines 3 best individual models by CV")
print("  2. 2 Boosting + 1 Bagging = good diversity")
print("  3. Achieved highest OOF (0.8350) in testing")
print("  4. LightGBM (gap 0.0090) + RF (gap 0.0112) = low overfitting")

print("\n📊 vs Other Ensembles:")
print("  Top 3: 0.8350 (LightGBM+GB+RF)")
print("  RF+GB+LR: 0.8339")
print("  All Boosting: 0.8328")
print("  RF+LightGBM+LR: 0.8283")

print("\n" + "=" * 80)
print("TOP 3 ENSEMBLE READY!")
print("=" * 80)
