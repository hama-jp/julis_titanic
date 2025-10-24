#!/usr/bin/env python3
"""
項目別の補完方法比較実験

Age, Fare, Embarkedそれぞれについて、
既存の方法と新しい方法を比較して最適な組み合わせを見つける
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('FEATURE-BY-FEATURE IMPUTATION COMPARISON')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# ============================================================================
# AGE IMPUTATION COMPARISON
# ============================================================================
print('\n' + '='*80)
print('AGE IMPUTATION COMPARISON')
print('='*80)

def impute_age_title_pclass(df_train, df_test):
    """既存方法: Title + Pclass別の中央値"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Title抽出
    for df in [train_copy, test_copy]:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
            'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
            'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
        }
        df['Title'] = df['Title'].map(title_mapping)

    # Title + Pclass別の中央値で補完（trainから学習）
    age_dict = {}
    for title in train_copy['Title'].unique():
        for pclass in train_copy['Pclass'].unique():
            mask = (train_copy['Title'] == title) & (train_copy['Pclass'] == pclass)
            median_age = train_copy.loc[mask, 'Age'].median()
            if pd.notna(median_age):
                age_dict[(title, pclass)] = median_age

    # 全体の中央値（フォールバック）
    overall_median = train_copy['Age'].median()

    # Train補完
    for idx in train_copy[train_copy['Age'].isnull()].index:
        title = train_copy.loc[idx, 'Title']
        pclass = train_copy.loc[idx, 'Pclass']
        train_copy.loc[idx, 'Age'] = age_dict.get((title, pclass), overall_median)

    # Test補完
    for idx in test_copy[test_copy['Age'].isnull()].index:
        title = test_copy.loc[idx, 'Title']
        pclass = test_copy.loc[idx, 'Pclass']
        test_copy.loc[idx, 'Age'] = age_dict.get((title, pclass), overall_median)

    return train_copy, test_copy

def impute_age_pclass_sex(df_train, df_test):
    """新方法: Pclass + Sex別の中央値"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Pclass + Sex別の中央値（trainから学習）
    age_dict = {}
    for pclass in train_copy['Pclass'].unique():
        for sex in train_copy['Sex'].unique():
            mask = (train_copy['Pclass'] == pclass) & (train_copy['Sex'] == sex)
            median_age = train_copy.loc[mask, 'Age'].median()
            age_dict[(pclass, sex)] = median_age

    # 全体の中央値（フォールバック）
    overall_median = train_copy['Age'].median()

    # Train補完
    for idx in train_copy[train_copy['Age'].isnull()].index:
        pclass = train_copy.loc[idx, 'Pclass']
        sex = train_copy.loc[idx, 'Sex']
        train_copy.loc[idx, 'Age'] = age_dict.get((pclass, sex), overall_median)

    # Test補完
    for idx in test_copy[test_copy['Age'].isnull()].index:
        pclass = test_copy.loc[idx, 'Pclass']
        sex = test_copy.loc[idx, 'Sex']
        test_copy.loc[idx, 'Age'] = age_dict.get((pclass, sex), overall_median)

    return train_copy, test_copy

# ============================================================================
# FARE IMPUTATION COMPARISON
# ============================================================================
print('\n' + '='*80)
print('FARE IMPUTATION COMPARISON')
print('='*80)

def impute_fare_pclass(df_train, df_test):
    """既存方法: Pclass別の中央値"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Pclass別の中央値
    fare_dict = {}
    for pclass in train_copy['Pclass'].unique():
        median_fare = train_copy[train_copy['Pclass'] == pclass]['Fare'].median()
        fare_dict[pclass] = median_fare

    # Train補完
    for idx in train_copy[train_copy['Fare'].isnull()].index:
        pclass = train_copy.loc[idx, 'Pclass']
        train_copy.loc[idx, 'Fare'] = fare_dict[pclass]

    # Test補完
    for idx in test_copy[test_copy['Fare'].isnull()].index:
        pclass = test_copy.loc[idx, 'Pclass']
        test_copy.loc[idx, 'Fare'] = fare_dict[pclass]

    return train_copy, test_copy

def impute_fare_pclass_embarked(df_train, df_test):
    """新方法: Pclass + Embarked別の中央値"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Pclass + Embarked別の中央値
    fare_dict = {}
    for pclass in train_copy['Pclass'].unique():
        for embarked in train_copy['Embarked'].dropna().unique():
            mask = (train_copy['Pclass'] == pclass) & (train_copy['Embarked'] == embarked)
            median_fare = train_copy.loc[mask, 'Fare'].median()
            fare_dict[(pclass, embarked)] = median_fare

    # Pclass別（フォールバック）
    for pclass in train_copy['Pclass'].unique():
        median_fare = train_copy[train_copy['Pclass'] == pclass]['Fare'].median()
        fare_dict[(pclass, None)] = median_fare

    # Train補完
    for idx in train_copy[train_copy['Fare'].isnull()].index:
        pclass = train_copy.loc[idx, 'Pclass']
        embarked = train_copy.loc[idx, 'Embarked']
        if pd.notna(embarked) and (pclass, embarked) in fare_dict:
            train_copy.loc[idx, 'Fare'] = fare_dict[(pclass, embarked)]
        else:
            train_copy.loc[idx, 'Fare'] = fare_dict[(pclass, None)]

    # Test補完
    for idx in test_copy[test_copy['Fare'].isnull()].index:
        pclass = test_copy.loc[idx, 'Pclass']
        embarked = test_copy.loc[idx, 'Embarked']
        if pd.notna(embarked) and (pclass, embarked) in fare_dict:
            test_copy.loc[idx, 'Fare'] = fare_dict[(pclass, embarked)]
        else:
            test_copy.loc[idx, 'Fare'] = fare_dict[(pclass, None)]

    return train_copy, test_copy

# ============================================================================
# EMBARKED IMPUTATION COMPARISON
# ============================================================================
print('\n' + '='*80)
print('EMBARKED IMPUTATION COMPARISON')
print('='*80)

def impute_embarked_mode(df_train, df_test):
    """既存方法: 最頻値（'S'）"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    mode_embarked = train_copy['Embarked'].mode()[0]

    train_copy['Embarked'].fillna(mode_embarked, inplace=True)
    test_copy['Embarked'].fillna(mode_embarked, inplace=True)

    return train_copy, test_copy

def impute_embarked_fare_based(df_train, df_test):
    """新方法: Fare + Pclassから推定"""
    train_copy = df_train.copy()
    test_copy = df_test.copy()

    # Pclass + Embarked別のFare範囲を学習
    fare_ranges = {}
    for pclass in train_copy['Pclass'].unique():
        pclass_data = train_copy[train_copy['Pclass'] == pclass]
        for embarked in pclass_data['Embarked'].dropna().unique():
            embarked_data = pclass_data[pclass_data['Embarked'] == embarked]
            fare_25 = embarked_data['Fare'].quantile(0.25)
            fare_75 = embarked_data['Fare'].quantile(0.75)
            fare_ranges[(pclass, embarked)] = (fare_25, fare_75)

    mode_embarked = train_copy['Embarked'].mode()[0]

    def estimate_embarked(pclass, fare):
        if pd.isna(fare):
            return mode_embarked

        best_match = None
        best_score = float('inf')

        for (pc, emb), (fare_25, fare_75) in fare_ranges.items():
            if pc == pclass:
                if fare_25 <= fare <= fare_75:
                    score = 0
                else:
                    score = min(abs(fare - fare_25), abs(fare - fare_75))

                if score < best_score:
                    best_score = score
                    best_match = emb

        return best_match if best_match else mode_embarked

    # Train補完
    for idx in train_copy[train_copy['Embarked'].isnull()].index:
        pclass = train_copy.loc[idx, 'Pclass']
        fare = train_copy.loc[idx, 'Fare']
        train_copy.loc[idx, 'Embarked'] = estimate_embarked(pclass, fare)

    # Test補完
    for idx in test_copy[test_copy['Embarked'].isnull()].index:
        pclass = test_copy.loc[idx, 'Pclass']
        fare = test_copy.loc[idx, 'Fare']
        test_copy.loc[idx, 'Embarked'] = estimate_embarked(pclass, fare)

    return train_copy, test_copy

# ============================================================================
# 特徴量エンジニアリング（共通）
# ============================================================================
def engineer_features(df):
    """基本的な特徴量エンジニアリング"""
    df = df.copy()

    # Sexエンコード
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Embarkedエンコード
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin flag
    df['Has_Cabin'] = df['Cabin'].notna().astype(int)

    # 基本特徴量
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'Embarked', 'FamilySize', 'IsAlone', 'Has_Cabin']

    return df[features]

# ============================================================================
# 補完方法の全組み合わせを評価
# ============================================================================
print('\n' + '='*80)
print('TESTING ALL COMBINATIONS')
print('='*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# 組み合わせのリスト
combinations = [
    {
        'name': 'Existing_Best',
        'age_method': 'Title+Pclass',
        'fare_method': 'Pclass',
        'embarked_method': 'Mode',
        'age_fn': impute_age_title_pclass,
        'fare_fn': impute_fare_pclass,
        'embarked_fn': impute_embarked_mode
    },
    {
        'name': 'New_Pipeline',
        'age_method': 'Pclass+Sex',
        'fare_method': 'Pclass+Embarked',
        'embarked_method': 'Fare-based',
        'age_fn': impute_age_pclass_sex,
        'fare_fn': impute_fare_pclass_embarked,
        'embarked_fn': impute_embarked_fare_based
    },
    {
        'name': 'Best_Age_Existing',
        'age_method': 'Title+Pclass',
        'fare_method': 'Pclass+Embarked',
        'embarked_method': 'Fare-based',
        'age_fn': impute_age_title_pclass,
        'fare_fn': impute_fare_pclass_embarked,
        'embarked_fn': impute_embarked_fare_based
    },
    {
        'name': 'Best_Age_New',
        'age_method': 'Pclass+Sex',
        'fare_method': 'Pclass+Embarked',
        'embarked_method': 'Fare-based',
        'age_fn': impute_age_pclass_sex,
        'fare_fn': impute_fare_pclass_embarked,
        'embarked_fn': impute_embarked_fare_based
    },
]

for combo in combinations:
    print(f'\n{"-"*60}')
    print(f'Testing: {combo["name"]}')
    print(f'  Age: {combo["age_method"]}')
    print(f'  Fare: {combo["fare_method"]}')
    print(f'  Embarked: {combo["embarked_method"]}')
    print(f'{"-"*60}')

    # データコピー
    train_proc = train.copy()
    test_proc = test.copy()

    # Embarked補完（Fareが必要な場合があるので最初）
    train_proc, test_proc = combo['embarked_fn'](train_proc, test_proc)

    # Fare補完
    train_proc, test_proc = combo['fare_fn'](train_proc, test_proc)

    # Age補完
    train_proc, test_proc = combo['age_fn'](train_proc, test_proc)

    # 特徴量エンジニアリング
    X_train = engineer_features(train_proc)
    y_train = train['Survived']

    # LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='accuracy')
    lr_mean = lr_scores.mean()
    lr_std = lr_scores.std()

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    rf_mean = rf_scores.mean()
    rf_std = rf_scores.std()

    print(f'\nResults:')
    print(f'  LogisticRegression: {lr_mean:.5f} (±{lr_std:.5f})')
    print(f'  RandomForest:       {rf_mean:.5f} (±{rf_std:.5f})')

    results.append({
        'name': combo['name'],
        'age_method': combo['age_method'],
        'fare_method': combo['fare_method'],
        'embarked_method': combo['embarked_method'],
        'lr_mean': lr_mean,
        'lr_std': lr_std,
        'rf_mean': rf_mean,
        'rf_std': rf_std,
        'avg_score': (lr_mean + rf_mean) / 2
    })

# ============================================================================
# 結果サマリー
# ============================================================================
print('\n' + '='*80)
print('RESULTS SUMMARY')
print('='*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('avg_score', ascending=False)

print('\nRanked by Average Score:')
print(results_df[['name', 'age_method', 'fare_method', 'embarked_method',
                   'lr_mean', 'rf_mean', 'avg_score']].to_string(index=False))

# ベストの組み合わせ
best = results_df.iloc[0]

print('\n' + '='*80)
print('BEST COMBINATION')
print('='*80)
print(f'\nConfiguration: {best["name"]}')
print(f'  Age補完:      {best["age_method"]}')
print(f'  Fare補完:     {best["fare_method"]}')
print(f'  Embarked補完: {best["embarked_method"]}')
print(f'\nScores:')
print(f'  LogisticRegression: {best["lr_mean"]:.5f} (±{best["lr_std"]:.5f})')
print(f'  RandomForest:       {best["rf_mean"]:.5f} (±{best["rf_std"]:.5f})')
print(f'  Average:            {best["avg_score"]:.5f}')

# 結果を保存
results_df.to_csv('experiments/results/20241024_feature_comparison.csv', index=False)
print('\n' + '='*80)
print('Results saved to: experiments/results/20241024_feature_comparison.csv')
print('='*80)
