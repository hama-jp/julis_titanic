"""
Missing Value Impact Analysis

Check if the presence/absence of missing values in features
is correlated with survival rate.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('MISSING VALUE IMPACT ANALYSIS')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print(f'\nTraining data: {len(train)} samples')
print(f'Test data: {len(test)} samples')

# ============================================================================
# ANALYZE MISSING VALUES IN TRAINING DATA
# ============================================================================
print('\n' + '='*80)
print('MISSING VALUES IN TRAINING DATA')
print('='*80)

missing_info = []
for col in train.columns:
    n_missing = train[col].isnull().sum()
    pct_missing = n_missing / len(train) * 100

    if n_missing > 0:
        missing_info.append({
            'Feature': col,
            'Missing_Count': n_missing,
            'Missing_Pct': pct_missing
        })

if missing_info:
    missing_df = pd.DataFrame(missing_info).sort_values('Missing_Count', ascending=False)
    print('\nFeatures with missing values:')
    print(missing_df.to_string(index=False))
else:
    print('\nNo missing values found.')

# ============================================================================
# CORRELATION WITH SURVIVAL
# ============================================================================
print('\n' + '='*80)
print('MISSING VALUE FLAGS vs SURVIVAL RATE')
print('='*80)

survival_impact = []

for col in train.columns:
    if col == 'Survived':
        continue

    n_missing = train[col].isnull().sum()

    if n_missing > 0 and n_missing < len(train):  # Has missing values but not all
        # Create missing flag
        is_missing = train[col].isnull()

        # Calculate survival rates
        survival_when_missing = train[is_missing]['Survived'].mean()
        survival_when_present = train[~is_missing]['Survived'].mean()

        # Difference
        diff = survival_when_missing - survival_when_present
        abs_diff = abs(diff)

        # Count
        n_missing_samples = is_missing.sum()
        n_present_samples = (~is_missing).sum()

        # Statistical significance (chi-square test)
        from scipy.stats import chi2_contingency
        contingency_table = pd.crosstab(is_missing, train['Survived'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        survival_impact.append({
            'Feature': col,
            'Missing_Count': n_missing_samples,
            'Present_Count': n_present_samples,
            'Survival_Missing': survival_when_missing,
            'Survival_Present': survival_when_present,
            'Difference': diff,
            'Abs_Difference': abs_diff,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

if survival_impact:
    impact_df = pd.DataFrame(survival_impact).sort_values('Abs_Difference', ascending=False)
    print('\nSurvival Rate by Missing Status:')
    print(impact_df.to_string(index=False))

    # Highlight significant ones
    print('\n' + '='*80)
    print('STATISTICALLY SIGNIFICANT FEATURES (p < 0.05)')
    print('='*80)

    significant = impact_df[impact_df['Significant'] == 'Yes']
    if len(significant) > 0:
        print('\nThese features have missing values that correlate with survival:')
        for _, row in significant.iterrows():
            direction = "higher" if row['Difference'] > 0 else "lower"
            print(f"\n{row['Feature']}:")
            print(f"  Missing ({row['Missing_Count']} samples): {row['Survival_Missing']:.3f} survival rate")
            print(f"  Present ({row['Present_Count']} samples): {row['Survival_Present']:.3f} survival rate")
            print(f"  → Missing values have {direction} survival rate (diff: {row['Difference']:+.3f})")
            print(f"  → p-value: {row['P_Value']:.4f}")
    else:
        print('\nNo statistically significant features found.')
else:
    print('\nNo features with partial missing values.')

# ============================================================================
# CREATE MISSING VALUE FLAGS
# ============================================================================
print('\n' + '='*80)
print('CREATING MISSING VALUE FLAG FEATURES')
print('='*80)

def add_missing_flags(df, significant_features):
    df = df.copy()
    added_features = []

    for feature in significant_features:
        if feature in df.columns:
            flag_name = f'{feature}_Missing'
            df[flag_name] = df[feature].isnull().astype(int)
            added_features.append(flag_name)
            print(f'  Added: {flag_name}')

    return df, added_features

# Get significant features
significant_features = []
if len(survival_impact) > 0:
    impact_df = pd.DataFrame(survival_impact)
    # Use features with abs difference > 0.05 or p-value < 0.05
    significant_features = impact_df[
        (impact_df['Abs_Difference'] > 0.05) | (impact_df['P_Value'] < 0.05)
    ]['Feature'].tolist()

print(f'\nFeatures to create missing flags for: {significant_features if significant_features else "None"}')

if significant_features:
    train_with_flags, added_train_flags = add_missing_flags(train, significant_features)
    test_with_flags, added_test_flags = add_missing_flags(test, significant_features)

    # ============================================================================
    # FEATURE ENGINEERING WITH MISSING FLAGS
    # ============================================================================
    print('\n' + '='*80)
    print('FEATURE ENGINEERING WITH MISSING FLAGS')
    print('='*80)

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

        # Fare imputation and binning
        df['Fare'] = df.groupby('Pclass')['Fare'].transform(
            lambda x: x.fillna(x.median())
        )
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
        df['FareBin'] = df['FareBin'].astype(int)

        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')

        return df

    train_fe = engineer_features(train_with_flags)
    test_fe = engineer_features(test_with_flags)

    # Encoding
    le_sex = LabelEncoder()
    train_fe['Sex'] = le_sex.fit_transform(train_fe['Sex'])
    test_fe['Sex'] = le_sex.transform(test_fe['Sex'])

    le_title = LabelEncoder()
    train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
    test_fe['Title'] = le_title.transform(test_fe['Title'])

    # Base features (without missing flags)
    base_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'SmallFamily']

    # Features with missing flags
    features_with_flags = base_features + added_train_flags

    print(f'\nBase features ({len(base_features)}): {", ".join(base_features)}')
    print(f'Added missing flags ({len(added_train_flags)}): {", ".join(added_train_flags)}')
    print(f'Total features ({len(features_with_flags)}): {", ".join(features_with_flags)}')

    # ============================================================================
    # EVALUATE WITH MISSING FLAGS
    # ============================================================================
    print('\n' + '='*80)
    print('MODEL EVALUATION: WITH vs WITHOUT MISSING FLAGS')
    print('='*80)

    X_train = train_fe[base_features]
    X_train_with_flags = train_fe[features_with_flags]
    y_train = train_fe['Survived']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Test both configurations
    rf_params = {
        'n_estimators': 100,
        'max_depth': 4,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'max_features': 'sqrt',
        'random_state': 42
    }

    # Without missing flags
    rf_base = RandomForestClassifier(**rf_params)
    scores_base = cross_val_score(rf_base, X_train, y_train, cv=cv, scoring='accuracy')

    # With missing flags
    rf_flags = RandomForestClassifier(**rf_params)
    scores_flags = cross_val_score(rf_flags, X_train_with_flags, y_train, cv=cv, scoring='accuracy')

    print('\nRandom Forest CV Scores:')
    print(f'  Without missing flags: {scores_base.mean():.4f} (±{scores_base.std():.4f})')
    print(f'  With missing flags:    {scores_flags.mean():.4f} (±{scores_flags.std():.4f})')
    print(f'  Improvement:           {scores_flags.mean() - scores_base.mean():+.4f}')

    # Train final models
    rf_base.fit(X_train, y_train)
    rf_flags.fit(X_train_with_flags, y_train)

    # Feature importance with flags
    if len(added_train_flags) > 0:
        print('\n' + '='*80)
        print('FEATURE IMPORTANCE (WITH MISSING FLAGS)')
        print('='*80)

        importance_df = pd.DataFrame({
            'Feature': features_with_flags,
            'Importance': rf_flags.feature_importances_
        }).sort_values('Importance', ascending=False)

        print('\nTop features:')
        print(importance_df.to_string(index=False))

        # Highlight missing flag features
        print('\nMissing flag importance:')
        for flag in added_train_flags:
            imp = importance_df[importance_df['Feature'] == flag]['Importance'].values[0]
            rank = importance_df[importance_df['Feature'] == flag].index[0] + 1
            print(f'  {flag}: {imp:.4f} (rank {rank}/{len(features_with_flags)})')

    # ============================================================================
    # RECOMMENDATION
    # ============================================================================
    print('\n' + '='*80)
    print('RECOMMENDATION')
    print('='*80)

    improvement = scores_flags.mean() - scores_base.mean()

    if improvement > 0.001:
        print(f'\n✅ RECOMMENDED: Add missing value flags')
        print(f'   Improvement: {improvement:+.4f} in CV score')
        print(f'   Features to add: {", ".join(added_train_flags)}')
        print(f'\n   Missing value flags are useful for prediction!')
    else:
        print(f'\n❌ NOT RECOMMENDED: Missing value flags do not improve performance')
        print(f'   Change in CV score: {improvement:+.4f}')
        print(f'\n   Keep base features only.')

else:
    print('\n❌ No significant missing value patterns found.')
    print('   Missing value flags are not necessary.')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
