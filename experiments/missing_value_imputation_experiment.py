#!/usr/bin/env python3
"""
欠損値補完戦略の比較実験

複数の補完方法を試して、モデル精度への影響を評価する
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('scripts/preprocessing')
from missing_value_imputation import MissingValueImputer


def prepare_features(df, is_train=True):
    """
    特徴量エンジニアリング（基本的な変換のみ）

    Parameters:
    -----------
    df : pd.DataFrame
        入力データ
    is_train : bool
        訓練データかどうか

    Returns:
    --------
    pd.DataFrame : 変換後のデータ
    """
    df = df.copy()

    # Sex をエンコード
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    # Embarked をエンコード（欠損値がある場合は先に補完が必要）
    if df['Embarked'].isnull().sum() == 0:
        df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])

    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 使用する特徴量を選択
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']

    if 'Has_Cabin' in df.columns:
        feature_cols.append('Has_Cabin')

    if 'Embarked' in df.columns:
        feature_cols.append('Embarked')

    return df[feature_cols]


def strategy_1_group_median(train, test):
    """
    Strategy 1: Group-based Median Imputation
    Pclass + Embarked/Sex別の中央値で補完
    """
    print("\n" + "="*80)
    print("Strategy 1: Group-based Median Imputation")
    print("="*80)

    imputer = MissingValueImputer()
    train_imputed = imputer.fit_transform(train.copy())
    test_imputed = imputer.transform(test.copy())

    # 特徴量変換
    X_train = prepare_features(train_imputed, is_train=True)
    y_train = train_imputed['Survived']

    return X_train, y_train, train_imputed, test_imputed


def strategy_2_knn_imputation(train, test):
    """
    Strategy 2: KNN Imputation
    K最近傍法で欠損値を補完
    """
    print("\n" + "="*80)
    print("Strategy 2: KNN Imputation")
    print("="*80)

    train_df = train.copy()
    test_df = test.copy()

    # まず、Embarked（少数）を最頻値で補完
    if train_df['Embarked'].isnull().sum() > 0:
        mode_embarked = train_df['Embarked'].mode()[0]
        train_df['Embarked'].fillna(mode_embarked, inplace=True)
        print(f"Imputed {train['Embarked'].isnull().sum()} Embarked values with mode: {mode_embarked}")

    # Fare（testの1件）を中央値で補完
    if test_df['Fare'].isnull().sum() > 0:
        median_fare = train_df['Fare'].median()
        test_df['Fare'].fillna(median_fare, inplace=True)
        print(f"Imputed {test['Fare'].isnull().sum()} Fare values with median: {median_fare:.2f}")

    # Cabinフラグ化
    train_df['Has_Cabin'] = train_df['Cabin'].notna().astype(int)
    test_df['Has_Cabin'] = test_df['Cabin'].notna().astype(int)

    # Sex, Embarked をエンコード
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    train_df['Sex'] = le_sex.fit_transform(train_df['Sex'])
    test_df['Sex'] = le_sex.transform(test_df['Sex'])

    train_df['Embarked'] = le_embarked.fit_transform(train_df['Embarked'])
    test_df['Embarked'] = le_embarked.transform(test_df['Embarked'])

    # KNN Imputationに使用する特徴量
    impute_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # KNN Imputer (n_neighbors=5)
    knn_imputer = KNNImputer(n_neighbors=5)

    # Train
    train_impute_data = train_df[impute_features].copy()
    train_imputed_values = knn_imputer.fit_transform(train_impute_data)
    train_df[impute_features] = train_imputed_values

    # Test
    test_impute_data = test_df[impute_features].copy()
    test_imputed_values = knn_imputer.transform(test_impute_data)
    test_df[impute_features] = test_imputed_values

    print(f"Imputed Age using KNN (n_neighbors=5)")

    # 特徴量追加
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
    test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

    # 特徴量選択
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'IsAlone', 'Has_Cabin']

    X_train = train_df[feature_cols]
    y_train = train_df['Survived']

    return X_train, y_train, train_df, test_df


def strategy_3_iterative_imputation(train, test):
    """
    Strategy 3: Iterative Imputation (MICE-like)
    他の特徴量からモデルベースで補完
    """
    print("\n" + "="*80)
    print("Strategy 3: Iterative Imputation (MICE)")
    print("="*80)

    train_df = train.copy()
    test_df = test.copy()

    # まず、Embarked（少数）を最頻値で補完
    if train_df['Embarked'].isnull().sum() > 0:
        mode_embarked = train_df['Embarked'].mode()[0]
        train_df['Embarked'].fillna(mode_embarked, inplace=True)
        print(f"Imputed {train['Embarked'].isnull().sum()} Embarked values with mode: {mode_embarked}")

    # Fare（testの1件）を中央値で補完
    if test_df['Fare'].isnull().sum() > 0:
        median_fare = train_df['Fare'].median()
        test_df['Fare'].fillna(median_fare, inplace=True)
        print(f"Imputed {test['Fare'].isnull().sum()} Fare values with median: {median_fare:.2f}")

    # Cabinフラグ化
    train_df['Has_Cabin'] = train_df['Cabin'].notna().astype(int)
    test_df['Has_Cabin'] = test_df['Cabin'].notna().astype(int)

    # Sex, Embarked をエンコード
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    train_df['Sex'] = le_sex.fit_transform(train_df['Sex'])
    test_df['Sex'] = le_sex.transform(test_df['Sex'])

    train_df['Embarked'] = le_embarked.fit_transform(train_df['Embarked'])
    test_df['Embarked'] = le_embarked.transform(test_df['Embarked'])

    # Iterative Imputationに使用する特徴量
    impute_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin']

    # Iterative Imputer
    iter_imputer = IterativeImputer(random_state=42, max_iter=10)

    # Train
    train_impute_data = train_df[impute_features].copy()
    train_imputed_values = iter_imputer.fit_transform(train_impute_data)
    train_df[impute_features] = train_imputed_values

    # Test
    test_impute_data = test_df[impute_features].copy()
    test_imputed_values = iter_imputer.transform(test_impute_data)
    test_df[impute_features] = test_imputed_values

    print(f"Imputed Age using Iterative Imputation (MICE)")

    # 特徴量追加
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    train_df['IsAlone'] = (train_df['FamilySize'] == 1).astype(int)

    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
    test_df['IsAlone'] = (test_df['FamilySize'] == 1).astype(int)

    # 特徴量選択
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'IsAlone', 'Has_Cabin']

    X_train = train_df[feature_cols]
    y_train = train_df['Survived']

    return X_train, y_train, train_df, test_df


def evaluate_strategy(X_train, y_train, strategy_name):
    """
    補完戦略を評価

    Parameters:
    -----------
    X_train : pd.DataFrame
        訓練特徴量
    y_train : pd.Series
        訓練ターゲット
    strategy_name : str
        戦略名

    Returns:
    --------
    dict : 評価結果
    """
    print(f"\n{'-'*60}")
    print(f"Evaluating {strategy_name}")
    print(f"{'-'*60}")

    # クロスバリデーション設定
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Logistic Regression
    print("\n1. Logistic Regression")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='accuracy')
    lr_mean = lr_scores.mean()
    lr_std = lr_scores.std()
    results['LogisticRegression'] = {'mean': lr_mean, 'std': lr_std, 'scores': lr_scores}
    print(f"   Accuracy: {lr_mean:.4f} (+/- {lr_std:.4f})")

    # Random Forest
    print("2. Random Forest")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    rf_mean = rf_scores.mean()
    rf_std = rf_scores.std()
    results['RandomForest'] = {'mean': rf_mean, 'std': rf_std, 'scores': rf_scores}
    print(f"   Accuracy: {rf_mean:.4f} (+/- {rf_std:.4f})")

    return results


def main():
    """
    メイン処理: 複数の補完戦略を比較
    """
    print("="*80)
    print("MISSING VALUE IMPUTATION STRATEGY COMPARISON")
    print("="*80)

    # データ読み込み
    print("\nLoading data...")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    all_results = {}

    # Strategy 1: Group-based Median
    try:
        X_train_1, y_train_1, train_imp_1, test_imp_1 = strategy_1_group_median(train, test)
        results_1 = evaluate_strategy(X_train_1, y_train_1, "Strategy 1: Group-based Median")
        all_results['Strategy_1_GroupMedian'] = results_1
    except Exception as e:
        print(f"Strategy 1 failed: {e}")

    # Strategy 2: KNN Imputation
    try:
        X_train_2, y_train_2, train_imp_2, test_imp_2 = strategy_2_knn_imputation(train, test)
        results_2 = evaluate_strategy(X_train_2, y_train_2, "Strategy 2: KNN Imputation")
        all_results['Strategy_2_KNN'] = results_2
    except Exception as e:
        print(f"Strategy 2 failed: {e}")

    # Strategy 3: Iterative Imputation
    try:
        X_train_3, y_train_3, train_imp_3, test_imp_3 = strategy_3_iterative_imputation(train, test)
        results_3 = evaluate_strategy(X_train_3, y_train_3, "Strategy 3: Iterative Imputation")
        all_results['Strategy_3_Iterative'] = results_3
    except Exception as e:
        print(f"Strategy 3 failed: {e}")

    # 結果サマリー
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    summary_data = []
    for strategy_name, results in all_results.items():
        for model_name, metrics in results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Model': model_name,
                'Mean_Accuracy': metrics['mean'],
                'Std_Accuracy': metrics['std']
            })

    summary_df = pd.DataFrame(summary_data)
    print("\n", summary_df.to_string(index=False))

    # ベスト戦略の特定
    print("\n" + "-"*80)
    print("BEST STRATEGIES")
    print("-"*80)

    for model in ['LogisticRegression', 'RandomForest']:
        model_results = summary_df[summary_df['Model'] == model].sort_values('Mean_Accuracy', ascending=False)
        best = model_results.iloc[0]
        print(f"\nBest for {model}:")
        print(f"  Strategy: {best['Strategy']}")
        print(f"  Accuracy: {best['Mean_Accuracy']:.4f} (+/- {best['Std_Accuracy']:.4f})")

    # 結果をファイルに保存
    print("\n" + "="*80)
    print("Saving results...")
    summary_df.to_csv('experiments/results/20241024_imputation_comparison.csv', index=False)
    print("  Saved to experiments/results/20241024_imputation_comparison.csv")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
