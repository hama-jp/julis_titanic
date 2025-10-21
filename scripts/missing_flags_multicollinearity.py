"""
Missing Flags Multicollinearity Check

Check if missing value flags are correlated with base features,
which could cause multicollinearity issues.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('MISSING FLAGS MULTICOLLINEARITY CHECK')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')

# Feature engineering
def engineer_features(df):
    df = df.copy()

    # Create missing flags BEFORE imputation
    df['Age_Missing'] = df['Age'].isnull().astype(int)
    df['Cabin_Missing'] = df['Cabin'].isnull().astype(int)
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

train_fe = engineer_features(train)

# Encoding
le_sex = LabelEncoder()
train_fe['Sex'] = le_sex.fit_transform(train_fe['Sex'])

le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])

# Define feature sets
base_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'SmallFamily']
missing_flags = ['Age_Missing', 'Cabin_Missing', 'Embarked_Missing']
all_features = base_features + missing_flags

X = train_fe[all_features]

print(f'\nBase features ({len(base_features)}): {", ".join(base_features)}')
print(f'Missing flags ({len(missing_flags)}): {", ".join(missing_flags)}')

# ============================================================================
# CORRELATION MATRIX
# ============================================================================
print('\n' + '='*80)
print('CORRELATION MATRIX - Missing Flags vs Base Features')
print('='*80)

# Calculate correlation between missing flags and base features
corr_matrix = X.corr()

# Extract correlations with missing flags
print('\nCorrelations with Missing Flags:')
for flag in missing_flags:
    print(f'\n{flag}:')
    flag_corr = corr_matrix[flag][base_features].sort_values(key=abs, ascending=False)
    for feature, corr in flag_corr.items():
        if abs(corr) > 0.1:  # Show correlations > 0.1
            print(f'  {feature}: {corr:+.3f}')

# ============================================================================
# HIGH CORRELATION PAIRS
# ============================================================================
print('\n' + '='*80)
print('HIGH CORRELATION PAIRS (|r| > 0.3)')
print('='*80)

high_corr_pairs = []
for flag in missing_flags:
    for feature in base_features:
        corr = corr_matrix.loc[flag, feature]
        if abs(corr) > 0.3:
            high_corr_pairs.append({
                'Missing_Flag': flag,
                'Base_Feature': feature,
                'Correlation': corr
            })

if high_corr_pairs:
    corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print('\nFound high correlations:')
    print(corr_df.to_string(index=False))
else:
    print('\nNo high correlations found (all |r| < 0.3)')

# ============================================================================
# VIF ANALYSIS
# ============================================================================
print('\n' + '='*80)
print('VARIANCE INFLATION FACTOR (VIF) - WITH MISSING FLAGS')
print('='*80)

def calculate_vif(df):
    vif_data = []
    for i, col in enumerate(df.columns):
        X_other = df.drop(columns=[col])
        y_feature = df[col]

        lr = LinearRegression()
        lr.fit(X_other, y_feature)
        r_squared = lr.score(X_other, y_feature)

        if r_squared < 0.9999:
            vif = 1 / (1 - r_squared)
        else:
            vif = float('inf')

        vif_data.append({'Feature': col, 'VIF': vif})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

vif_df = calculate_vif(X)
print('\nVIF values:')
print(vif_df.to_string(index=False))

# Highlight high VIF
print('\n' + '='*80)
print('VIF SUMMARY')
print('='*80)

high_vif = vif_df[vif_df['VIF'] > 5]
if len(high_vif) > 0:
    print(f'\n⚠️  Features with high VIF (>5):')
    print(high_vif.to_string(index=False))
else:
    print(f'\n✅ All features have acceptable VIF (<5)')

# Check specifically for missing flags
print('\nMissing Flag VIF:')
for flag in missing_flags:
    vif = vif_df[vif_df['Feature'] == flag]['VIF'].values[0]
    status = '⚠️' if vif > 5 else '✅'
    print(f'  {status} {flag}: {vif:.2f}')

# ============================================================================
# CROSSTAB ANALYSIS
# ============================================================================
print('\n' + '='*80)
print('CROSSTAB ANALYSIS - Missing Flags vs Categorical Features')
print('='*80)

# Check Cabin_Missing vs Pclass (most likely to be correlated)
print('\nCabin_Missing vs Pclass:')
crosstab = pd.crosstab(train_fe['Cabin_Missing'], train_fe['Pclass'], margins=True, normalize='columns')
print(crosstab.round(3))

print('\nInterpretation:')
for pclass in [1, 2, 3]:
    pct_missing = train_fe[train_fe['Pclass'] == pclass]['Cabin_Missing'].mean()
    print(f'  Pclass {pclass}: {pct_missing:.1%} have missing Cabin')

# ============================================================================
# RECOMMENDATION
# ============================================================================
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

# Check for problematic flags
problematic_flags = []

# Check high VIF
for flag in missing_flags:
    vif = vif_df[vif_df['Feature'] == flag]['VIF'].values[0]
    if vif > 5:
        problematic_flags.append((flag, f'High VIF: {vif:.2f}'))

# Check high correlation
for pair in high_corr_pairs:
    if abs(pair['Correlation']) > 0.5:
        flag = pair['Missing_Flag']
        feature = pair['Base_Feature']
        corr = pair['Correlation']
        problematic_flags.append((flag, f'High correlation with {feature}: {corr:+.3f}'))

if problematic_flags:
    print('\n⚠️  ISSUES FOUND:')
    for flag, issue in problematic_flags:
        print(f'  {flag}: {issue}')

    # Group by flag
    flags_to_remove = list(set([flag for flag, _ in problematic_flags]))

    print(f'\n❌ Consider REMOVING these flags: {", ".join(flags_to_remove)}')

    remaining_flags = [f for f in missing_flags if f not in flags_to_remove]
    if remaining_flags:
        print(f'✅ Keep these flags: {", ".join(remaining_flags)}')
        recommended_features = base_features + remaining_flags
    else:
        print(f'✅ Use base features only (no missing flags)')
        recommended_features = base_features

else:
    print('\n✅ NO MULTICOLLINEARITY ISSUES DETECTED')
    print(f'   All missing flags have acceptable VIF (<5)')
    print(f'   All correlations are moderate (|r| < 0.5)')
    print(f'\n✅ RECOMMENDED: Use all 10 features')
    recommended_features = all_features

print(f'\nRecommended feature set ({len(recommended_features)} features):')
print(f'  {", ".join(recommended_features)}')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
