"""
Feature Multicollinearity Check

Analyze feature correlations and VIF to identify redundant features
that may be causing overfitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('FEATURE MULTICOLLINEARITY ANALYSIS')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# Feature engineering (same as before)
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

# Current feature set (Enhanced_V4)
current_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex', 'SmallFamily']

# All available features
all_features = ['Pclass', 'Sex', 'Title', 'Age', 'SibSp', 'Parch', 'FamilySize',
                'IsAlone', 'SmallFamily', 'IsChild', 'IsElderly', 'FareBin',
                'Embarked', 'Pclass_Sex', 'Title_Pclass']

print('\n' + '='*80)
print('CORRELATION MATRIX - Current Features')
print('='*80)

X_current = train_fe[current_features]
y = train_fe['Survived']

# Calculate correlation matrix
corr_matrix = X_current.corr()
print('\nCorrelation Matrix:')
print(corr_matrix.round(3))

# Find highly correlated feature pairs
print('\n' + '='*80)
print('HIGHLY CORRELATED FEATURE PAIRS (|correlation| > 0.5)')
print('='*80)

high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.5:
            high_corr_pairs.append({
                'Feature1': corr_matrix.columns[i],
                'Feature2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    corr_df = pd.DataFrame(high_corr_pairs)
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
    print(corr_df.to_string(index=False))
else:
    print('No highly correlated pairs found.')

# Calculate VIF (Variance Inflation Factor)
print('\n' + '='*80)
print('VARIANCE INFLATION FACTOR (VIF)')
print('='*80)
print('VIF > 10: High multicollinearity')
print('VIF 5-10: Moderate multicollinearity')
print('VIF < 5: Low multicollinearity\n')

from sklearn.linear_model import LinearRegression

def calculate_vif(df):
    vif_data = []
    for i, col in enumerate(df.columns):
        # Use other features to predict this feature
        X_other = df.drop(columns=[col])
        y_feature = df[col]

        lr = LinearRegression()
        lr.fit(X_other, y_feature)
        r_squared = lr.score(X_other, y_feature)

        # VIF = 1 / (1 - R²)
        if r_squared < 0.9999:  # Avoid division by zero
            vif = 1 / (1 - r_squared)
        else:
            vif = float('inf')

        vif_data.append({'Feature': col, 'VIF': vif})

    return pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

vif_df = calculate_vif(X_current)
print(vif_df.to_string(index=False))

# Analyze feature importance
print('\n' + '='*80)
print('FEATURE IMPORTANCE ANALYSIS')
print('='*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train a simple RF to get feature importance
rf_simple = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_simple.fit(X_current, y)

importance_df = pd.DataFrame({
    'Feature': current_features,
    'Importance': rf_simple.feature_importances_
}).sort_values('Importance', ascending=False)

print('\nRandom Forest Feature Importance:')
print(importance_df.to_string(index=False))

# Suggest feature subsets
print('\n' + '='*80)
print('SUGGESTED FEATURE SUBSETS')
print('='*80)

# Subset 1: Remove features with high VIF
high_vif_features = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
subset1 = [f for f in current_features if f not in high_vif_features]

# Subset 2: Top N most important features
top_n = 5
subset2 = importance_df.head(top_n)['Feature'].tolist()

# Subset 3: Manual selection based on domain knowledge
# Remove interaction features and keep base features
subset3 = ['Pclass', 'Sex', 'Age', 'FareBin', 'IsAlone']

# Subset 4: Remove SmallFamily (keep IsAlone which is simpler)
subset4 = [f for f in current_features if f != 'SmallFamily']

# Subset 5: Remove Pclass_Sex (keep Pclass and Sex separately)
subset5 = [f for f in current_features if f != 'Pclass_Sex']

subsets = {
    'Subset 1 (Low VIF)': subset1,
    'Subset 2 (Top 5 Important)': subset2,
    'Subset 3 (Base Features)': subset3,
    'Subset 4 (No SmallFamily)': subset4,
    'Subset 5 (No Pclass_Sex)': subset5,
}

print('\nFeature Subsets:')
for name, features in subsets.items():
    print(f'\n{name} ({len(features)} features):')
    print(f'  {", ".join(features)}')

# Test each subset with simple models
print('\n' + '='*80)
print('EVALUATING FEATURE SUBSETS')
print('='*80)

results = []

# Add current features as baseline
subsets['Current (Baseline)'] = current_features

for subset_name, features in subsets.items():
    X_subset = train_fe[features]

    # Test with Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
    rf_scores = cross_val_score(rf, X_subset, y, cv=cv, scoring='accuracy')
    rf_mean = rf_scores.mean()
    rf_std = rf_scores.std()

    # Test with Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    gb_scores = cross_val_score(gb, X_subset, y, cv=cv, scoring='accuracy')
    gb_mean = gb_scores.mean()
    gb_std = gb_scores.std()

    # Test with Logistic Regression
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_subset, y, cv=cv, scoring='accuracy')
    lr_mean = lr_scores.mean()
    lr_std = lr_scores.std()

    results.append({
        'Subset': subset_name,
        'N_Features': len(features),
        'RF_Mean': rf_mean,
        'RF_Std': rf_std,
        'GB_Mean': gb_mean,
        'GB_Std': gb_std,
        'LR_Mean': lr_mean,
        'LR_Std': lr_std,
        'Avg_Mean': (rf_mean + gb_mean + lr_mean) / 3
    })

results_df = pd.DataFrame(results).sort_values('Avg_Mean', ascending=False)

print('\nCV Scores by Subset:')
print(results_df[['Subset', 'N_Features', 'RF_Mean', 'GB_Mean', 'LR_Mean', 'Avg_Mean']].to_string(index=False))

# Show best subset
print('\n' + '='*80)
print('RECOMMENDATION')
print('='*80)

best_subset = results_df.iloc[0]
best_subset_name = best_subset['Subset']
best_features = subsets[best_subset_name]

print(f'\nBest Subset: {best_subset_name}')
print(f'Number of Features: {len(best_features)}')
print(f'Features: {", ".join(best_features)}')
print(f'\nCV Scores:')
print(f'  Random Forest: {best_subset["RF_Mean"]:.4f} (±{best_subset["RF_Std"]:.4f})')
print(f'  Gradient Boosting: {best_subset["GB_Mean"]:.4f} (±{best_subset["GB_Std"]:.4f})')
print(f'  Logistic Regression: {best_subset["LR_Mean"]:.4f} (±{best_subset["LR_Std"]:.4f})')
print(f'  Average: {best_subset["Avg_Mean"]:.4f}')

# Compare to current baseline
baseline = results_df[results_df['Subset'] == 'Current (Baseline)'].iloc[0]
improvement = best_subset['Avg_Mean'] - baseline['Avg_Mean']

print(f'\nComparison to Current Baseline:')
print(f'  Current Avg: {baseline["Avg_Mean"]:.4f}')
print(f'  Best Avg: {best_subset["Avg_Mean"]:.4f}')
print(f'  Improvement: {improvement:+.4f}')

if best_subset_name != 'Current (Baseline)':
    print(f'\n✅ Recommendation: Use {best_subset_name}')
    removed_features = set(current_features) - set(best_features)
    if removed_features:
        print(f'   Removed: {", ".join(removed_features)}')
    added_features = set(best_features) - set(current_features)
    if added_features:
        print(f'   Added: {", ".join(added_features)}')
else:
    print(f'\n✅ Current feature set is already optimal')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
