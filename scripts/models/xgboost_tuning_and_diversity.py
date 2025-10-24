"""
XGBoost Tuning and Ensemble Diversity Analysis

Two key questions:
1. Can XGBoost be tuned to match/beat LightGBM performance?
2. Do multiple boosting models (GB, LightGBM, XGBoost) provide diversity?

Diversity Metrics:
- Prediction correlation between models
- Disagreement rate
- Q-statistic (measures diversity)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("XGBOOST TUNING & ENSEMBLE DIVERSITY ANALYSIS")
print("=" * 80)
print("Q1: Can XGBoost be tuned conservatively like LightGBM?")
print("Q2: Do multiple boosting models provide diversity?")
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
print("PART 1: XGBOOST CONSERVATIVE TUNING")
print("=" * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_configs = [
    {
        'name': 'XGB_Original',
        'params': {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
            'gamma': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    },
    {
        'name': 'XGB_Conservative_V1',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,  # Shallower
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 20,  # More samples
            'reg_alpha': 1.0,  # Stronger L1
            'reg_lambda': 1.0,  # Stronger L2
            'gamma': 0.2,  # Higher minimum loss reduction
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    },
    {
        'name': 'XGB_Conservative_V2',
        'params': {
            'n_estimators': 100,
            'max_depth': 2,
            'learning_rate': 0.03,  # Slower
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 30,
            'reg_alpha': 1.5,
            'reg_lambda': 1.5,
            'gamma': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    },
    {
        'name': 'XGB_Ultra_Conservative',
        'params': {
            'n_estimators': 50,
            'max_depth': 2,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 40,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'gamma': 0.5,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    }
]

xgb_results = []

for config in xgb_configs:
    name = config['name']
    params = config['params']

    print(f"\n{name}:")

    model = xgb.XGBClassifier(**params)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Training score
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_score - cv_mean

    xgb_results.append({
        'Config': name,
        'CV_Score': cv_mean,
        'CV_Std': cv_std,
        'Train_Score': train_score,
        'Gap': gap
    })

    print(f"  CV: {cv_mean:.4f} (±{cv_std:.4f})")
    print(f"  Train: {train_score:.4f}")
    print(f"  Gap: {gap:.4f}")

    if gap < 0.01:
        print(f"  ✅ EXCELLENT - Target achieved!")
    elif gap < 0.02:
        print(f"  ✓ GOOD - Close to target")

xgb_df = pd.DataFrame(xgb_results)

print("\n" + "=" * 80)
print("XGBOOST TUNING RESULTS")
print("=" * 80)

print("\nAll XGBoost Configurations:")
print(xgb_df[['Config', 'CV_Score', 'Gap']].to_string(index=False))

# Find best XGBoost
low_gap_xgb = xgb_df[xgb_df['Gap'] < 0.02]
if len(low_gap_xgb) > 0:
    best_xgb_idx = low_gap_xgb['CV_Score'].idxmax()
    best_xgb = low_gap_xgb.loc[best_xgb_idx]
else:
    best_xgb_idx = xgb_df['CV_Score'].idxmax()
    best_xgb = xgb_df.loc[best_xgb_idx]

print(f"\n✅ Best XGBoost: {best_xgb['Config']}")
print(f"   CV: {best_xgb['CV_Score']:.4f}, Gap: {best_xgb['Gap']:.4f}")

print("\n" + "=" * 80)
print("PART 2: ALL BOOSTING MODELS COMPARISON")
print("=" * 80)

# Define all models including tuned versions
all_models = {
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
    'XGBoost': xgb.XGBClassifier(**xgb_configs[best_xgb_idx]['params']),
    'LR': LogisticRegression(
        C=0.5,
        penalty='l2',
        max_iter=1000,
        random_state=42
    )
}

print("\nEvaluating all models with CV...")

all_results = []
oof_predictions = {}

for name, model in all_models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

    # Get OOF predictions for diversity analysis
    oof_pred = np.zeros(len(X_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]

        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        oof_pred[val_idx] = model_clone.predict(X_val)

    oof_predictions[name] = oof_pred

    # Training score
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    gap = train_score - cv_mean

    all_results.append({
        'Model': name,
        'CV_Score': cv_mean,
        'CV_Std': cv_std,
        'Gap': gap,
        'Architecture': 'Boosting' if name in ['GB', 'LightGBM', 'XGBoost'] else ('Bagging' if name == 'RF' else 'Linear')
    })

    print(f"\n{name}:")
    print(f"  CV: {cv_mean:.4f} (±{cv_std:.4f}), Gap: {gap:.4f}")

all_df = pd.DataFrame(all_results).sort_values('CV_Score', ascending=False)

print("\n" + "=" * 80)
print("ALL MODELS RANKED")
print("=" * 80)

print("\n")
print(all_df[['Model', 'CV_Score', 'Gap', 'Architecture']].to_string(index=False))

print("\n" + "=" * 80)
print("PART 3: DIVERSITY ANALYSIS")
print("=" * 80)

print("\nAnalyzing prediction diversity between models...")

def calculate_diversity_metrics(pred1, pred2, y_true):
    """Calculate diversity metrics between two models"""
    # Agreement/Disagreement
    agreement = (pred1 == pred2).mean()
    disagreement = 1 - agreement

    # Both correct, both wrong, etc.
    both_correct = ((pred1 == y_true) & (pred2 == y_true)).sum()
    both_wrong = ((pred1 != y_true) & (pred2 != y_true)).sum()
    model1_correct = ((pred1 == y_true) & (pred2 != y_true)).sum()
    model2_correct = ((pred1 != y_true) & (pred2 == y_true)).sum()

    # Q-statistic (measures diversity: -1 to +1, lower is more diverse)
    n = len(y_true)
    if (both_correct + model2_correct) * (both_wrong + model2_correct) * \
       (both_correct + model1_correct) * (both_wrong + model1_correct) > 0:
        Q = (both_correct * both_wrong - model1_correct * model2_correct) / \
            (both_correct * both_wrong + model1_correct * model2_correct)
    else:
        Q = np.nan

    # Correlation
    correlation = np.corrcoef(pred1, pred2)[0, 1]

    return {
        'disagreement': disagreement,
        'Q_statistic': Q,
        'correlation': correlation,
        'both_correct': both_correct,
        'both_wrong': both_wrong
    }

# Analyze diversity between all pairs
diversity_results = []

for model1, model2 in combinations(all_models.keys(), 2):
    metrics = calculate_diversity_metrics(
        oof_predictions[model1],
        oof_predictions[model2],
        y_train.values
    )

    diversity_results.append({
        'Model1': model1,
        'Model2': model2,
        'Disagreement': metrics['disagreement'],
        'Correlation': metrics['correlation'],
        'Q_Statistic': metrics['Q_statistic']
    })

diversity_df = pd.DataFrame(diversity_results)

print("\nPrediction Correlation Matrix:")
print("(Lower correlation = higher diversity)")
print()

# Create correlation matrix
models_list = list(all_models.keys())
corr_matrix = np.eye(len(models_list))
for i, m1 in enumerate(models_list):
    for j, m2 in enumerate(models_list):
        if i != j:
            corr = diversity_df[
                ((diversity_df['Model1'] == m1) & (diversity_df['Model2'] == m2)) |
                ((diversity_df['Model1'] == m2) & (diversity_df['Model2'] == m1))
            ]['Correlation'].values[0]
            corr_matrix[i, j] = corr

corr_df = pd.DataFrame(corr_matrix, index=models_list, columns=models_list)
print(corr_df.round(3).to_string())

print("\n" + "=" * 80)
print("BOOSTING MODELS DIVERSITY")
print("=" * 80)

boosting_pairs = diversity_df[
    (diversity_df['Model1'].isin(['GB', 'LightGBM', 'XGBoost'])) &
    (diversity_df['Model2'].isin(['GB', 'LightGBM', 'XGBoost']))
].sort_values('Correlation')

print("\nDiversity between Boosting Models:")
print(boosting_pairs[['Model1', 'Model2', 'Correlation', 'Disagreement']].to_string(index=False))

avg_boosting_corr = boosting_pairs['Correlation'].mean()
print(f"\nAverage correlation between boosting models: {avg_boosting_corr:.4f}")

# Compare with diverse architectures
diverse_pairs = diversity_df[
    ((diversity_df['Model1'] == 'RF') & (diversity_df['Model2'].isin(['LightGBM', 'GB', 'XGBoost']))) |
    ((diversity_df['Model1'].isin(['LightGBM', 'GB', 'XGBoost'])) & (diversity_df['Model2'] == 'LR'))
].sort_values('Correlation')

avg_diverse_corr = diverse_pairs['Correlation'].mean()
print(f"Average correlation between diverse architectures: {avg_diverse_corr:.4f}")

print("\n" + "=" * 80)
print("ENSEMBLE COMPARISON")
print("=" * 80)

print("\nTesting different ensemble configurations...")

ensemble_configs = [
    {
        'name': 'All Boosting (GB + LightGBM + XGBoost)',
        'models': ['GB', 'LightGBM', 'XGBoost']
    },
    {
        'name': 'Diverse (RF + LightGBM + LR)',
        'models': ['RF', 'LightGBM', 'LR']
    },
    {
        'name': 'Diverse (RF + XGBoost + LR)',
        'models': ['RF', 'XGBoost', 'LR']
    },
    {
        'name': 'Diverse (RF + GB + LR)',
        'models': ['RF', 'GB', 'LR']
    },
    {
        'name': 'Top 3 by CV',
        'models': all_df.head(3)['Model'].tolist()
    }
]

ensemble_scores = []

for config in ensemble_configs:
    name = config['name']
    model_names = config['models']

    # Simple average ensemble
    ensemble_pred = np.zeros(len(X_train))
    for model_name in model_names:
        ensemble_pred += oof_predictions[model_name]
    ensemble_pred = np.round(ensemble_pred / len(model_names)).astype(int)

    score = (ensemble_pred == y_train.values).mean()

    # Calculate average correlation within ensemble
    ensemble_pairs = diversity_df[
        (diversity_df['Model1'].isin(model_names)) &
        (diversity_df['Model2'].isin(model_names))
    ]
    avg_corr = ensemble_pairs['Correlation'].mean() if len(ensemble_pairs) > 0 else np.nan

    ensemble_scores.append({
        'Ensemble': name,
        'Models': ' + '.join(model_names),
        'OOF_Score': score,
        'Avg_Correlation': avg_corr
    })

    print(f"\n{name}:")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  OOF Score: {score:.4f}")
    print(f"  Avg Correlation: {avg_corr:.4f}")

ensemble_df = pd.DataFrame(ensemble_scores).sort_values('OOF_Score', ascending=False)

print("\n" + "=" * 80)
print("ENSEMBLE RESULTS RANKED")
print("=" * 80)

print("\n")
print(ensemble_df[['Ensemble', 'OOF_Score', 'Avg_Correlation']].to_string(index=False))

print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

best_ensemble = ensemble_df.iloc[0]

print(f"\n✅ Best Ensemble: {best_ensemble['Ensemble']}")
print(f"   OOF Score: {best_ensemble['OOF_Score']:.4f}")
print(f"   Avg Correlation: {best_ensemble['Avg_Correlation']:.4f}")

print("\n📊 Key Findings:")

print("\n1. XGBoost Tuning:")
print(f"   Original: CV {xgb_df.iloc[0]['CV_Score']:.4f}, Gap {xgb_df.iloc[0]['Gap']:.4f}")
print(f"   Best: CV {best_xgb['CV_Score']:.4f}, Gap {best_xgb['Gap']:.4f}")
improvement = best_xgb['CV_Score'] - xgb_df.iloc[0]['CV_Score']
print(f"   Improvement: {improvement:+.4f}")

print("\n2. Boosting Models Diversity:")
print(f"   Avg correlation between boosting models: {avg_boosting_corr:.4f}")
print(f"   Avg correlation between diverse models: {avg_diverse_corr:.4f}")
if avg_boosting_corr > 0.9:
    print(f"   ⚠️  Very high correlation - limited diversity!")
elif avg_boosting_corr > 0.8:
    print(f"   ⚠️  High correlation - some redundancy")
else:
    print(f"   ✓ Moderate correlation - good diversity")

print("\n3. Ensemble Strategy:")
if best_ensemble['Ensemble'].find('Boosting') >= 0:
    print(f"   🎯 Multiple boosting models work best despite high correlation")
    print(f"   Reason: Very high individual performance outweighs redundancy")
else:
    print(f"   🎯 Diverse architectures work best")
    print(f"   Reason: Lower correlation improves ensemble robustness")

print("\n" + "=" * 80)
print("FINAL RECOMMENDATION")
print("=" * 80)

print(f"\n🎯 Use: {best_ensemble['Models']}")
print(f"   Expected OOF: {best_ensemble['OOF_Score']:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
