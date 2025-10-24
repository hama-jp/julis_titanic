#!/usr/bin/env python3
"""
欠損値の詳細分析スクリプト
train/testデータの欠損値の状況を確認し、補完戦略を検討する
"""

import pandas as pd
import numpy as np

def check_missing_values():
    """train/testデータの欠損値を詳細にチェック"""

    # データ読み込み
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    print("=" * 80)
    print("TRAIN DATA - Missing Values Analysis")
    print("=" * 80)
    print(f"Total rows: {len(train)}")
    print("\nMissing values count and percentage:")
    train_missing = train.isnull().sum()
    train_missing_pct = (train_missing / len(train) * 100).round(2)
    train_missing_df = pd.DataFrame({
        'Column': train_missing.index,
        'Missing_Count': train_missing.values,
        'Missing_Percentage': train_missing_pct.values
    })
    train_missing_df = train_missing_df[train_missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    print(train_missing_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("TEST DATA - Missing Values Analysis")
    print("=" * 80)
    print(f"Total rows: {len(test)}")
    print("\nMissing values count and percentage:")
    test_missing = test.isnull().sum()
    test_missing_pct = (test_missing / len(test) * 100).round(2)
    test_missing_df = pd.DataFrame({
        'Column': test_missing.index,
        'Missing_Count': test_missing.values,
        'Missing_Percentage': test_missing_pct.values
    })
    test_missing_df = test_missing_df[test_missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    print(test_missing_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("MISSING VALUES COMPARISON")
    print("=" * 80)

    # testにあってtrainにない欠損値を特定
    train_missing_cols = set(train_missing_df['Column'].tolist())
    test_missing_cols = set(test_missing_df['Column'].tolist())

    only_in_test = test_missing_cols - train_missing_cols
    only_in_train = train_missing_cols - test_missing_cols
    common = train_missing_cols & test_missing_cols

    if common:
        print("\nColumns with missing values in BOTH train and test:")
        for col in common:
            train_cnt = train_missing[col]
            test_cnt = test_missing[col]
            print(f"  - {col}: train={train_cnt} ({train_missing_pct[col]}%), test={test_cnt} ({test_missing_pct[col]}%)")

    if only_in_train:
        print("\nColumns with missing values ONLY in train:")
        for col in only_in_train:
            print(f"  - {col}: {train_missing[col]} ({train_missing_pct[col]}%)")

    if only_in_test:
        print("\nColumns with missing values ONLY in test:")
        for col in only_in_test:
            print(f"  - {col}: {test_missing[col]} ({test_missing_pct[col]}%)")

    # Fareの詳細分析（testで欠損がある場合）
    if 'Fare' in test_missing_cols:
        print("\n" + "=" * 80)
        print("FARE MISSING VALUE DETAILED ANALYSIS")
        print("=" * 80)

        # testのFare欠損値の位置
        fare_missing_idx = test[test['Fare'].isnull()].index
        print(f"\nTest data - Fare missing at index: {fare_missing_idx.tolist()}")

        # 欠損値のある行の詳細
        print("\nPassenger details with missing Fare:")
        missing_fare_passengers = test[test['Fare'].isnull()]
        print(missing_fare_passengers[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].to_string())

        # Fareの基本統計
        print("\n" + "-" * 40)
        print("Fare statistics in train data:")
        print(train['Fare'].describe())

        print("\nFare statistics in test data (excluding missing):")
        print(test['Fare'].describe())

        # PclassとEmbarkedごとのFare統計（補完戦略のため）
        print("\n" + "-" * 40)
        print("Fare statistics by Pclass and Embarked in TRAIN data:")
        for pclass in sorted(train['Pclass'].unique()):
            for embarked in sorted(train['Embarked'].dropna().unique()):
                subset = train[(train['Pclass'] == pclass) & (train['Embarked'] == embarked)]['Fare']
                if len(subset) > 0:
                    print(f"  Pclass={pclass}, Embarked={embarked}: "
                          f"mean={subset.mean():.2f}, median={subset.median():.2f}, "
                          f"count={len(subset)}")

        # 欠損値の乗客のPclass/Embarkedに基づく推奨補完値
        print("\n" + "-" * 40)
        print("Recommended imputation for missing Fare passengers:")
        for idx, row in missing_fare_passengers.iterrows():
            pclass = row['Pclass']
            embarked = row['Embarked']
            if pd.notna(embarked):
                similar_fares = train[(train['Pclass'] == pclass) & (train['Embarked'] == embarked)]['Fare']
                if len(similar_fares) > 0:
                    median_fare = similar_fares.median()
                    mean_fare = similar_fares.mean()
                    print(f"  PassengerId {row['PassengerId']} (Pclass={pclass}, Embarked={embarked}): "
                          f"median={median_fare:.2f}, mean={mean_fare:.2f}")
                else:
                    print(f"  PassengerId {row['PassengerId']} (Pclass={pclass}, Embarked={embarked}): "
                          f"No matching records")
            else:
                print(f"  PassengerId {row['PassengerId']} (Pclass={pclass}, Embarked=NaN): "
                      f"Need to handle Embarked first")

if __name__ == '__main__':
    check_missing_values()
