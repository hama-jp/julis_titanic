#!/usr/bin/env python3
"""
本番パイプラインのCV評価

ProductionImputerを使って複数のモデルで評価
"""

import sys
sys.path.append('scripts/preprocessing')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from production_imputation_pipeline import ProductionImputer

print('='*80)
print('PRODUCTION PIPELINE EVALUATION')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 本番パイプラインで補完
print("\nApplying production imputation pipeline...")
imputer = ProductionImputer(verbose=False)
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

print(f"Train shape: {train_imputed.shape}")
print(f"Test shape: {test_imputed.shape}")

# 特徴量エンジニアリング
def engineer_features(df):
    """基本的な特徴量エンジニアリング"""
    df = df.copy()

    # Sexエンコード
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Titleエンコード
    le_title = LabelEncoder()
    df['Title'] = le_title.fit_transform(df['Title'])

    # Embarkedエンコード
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    # Age binning
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 60, 100], labels=[0, 1, 2])
    df['AgeGroup'] = df['AgeGroup'].astype(int)

    # Fare binning
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    return df

print("\nEngineering features...")
train_fe = engineer_features(train_imputed)
test_fe = engineer_features(test_imputed)

y_train = train['Survived']

# 特徴量セット定義
feature_sets = {
    'Basic': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Has_Cabin'],
    'With_Family': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                    'FamilySize', 'IsAlone', 'Has_Cabin'],
    'With_Bins': ['Pclass', 'Sex', 'AgeGroup', 'SibSp', 'Parch', 'FareBin', 'Embarked',
                  'FamilySize', 'IsAlone', 'Has_Cabin'],
    'With_Title': ['Pclass', 'Sex', 'Age', 'Title', 'SibSp', 'Parch', 'Fare', 'Embarked',
                   'FamilySize', 'IsAlone', 'SmallFamily', 'Has_Cabin'],
    'Full': ['Pclass', 'Sex', 'Age', 'AgeGroup', 'Title', 'SibSp', 'Parch', 'Fare', 'FareBin',
             'Embarked', 'FamilySize', 'IsAlone', 'SmallFamily', 'Has_Cabin'],
}

# モデル定義
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                     learning_rate=0.05, random_state=42),
}

# CV評価
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)

    for fs_name, features in feature_sets.items():
        X_train = train_fe[features]

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        results.append({
            'Model': model_name,
            'Features': fs_name,
            'N_Features': len(features),
            'CV_Mean': cv_mean,
            'CV_Std': cv_std
        })

        print(f"  {fs_name:15s} ({len(features):2d} features): "
              f"{cv_mean:.5f} (±{cv_std:.5f})")

# 結果サマリー
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

# モデル別ベスト
print("\nBest feature set for each model:")
for model_name in models.keys():
    model_results = results_df[results_df['Model'] == model_name]
    best = model_results.loc[model_results['CV_Mean'].idxmax()]
    print(f"\n{model_name}:")
    print(f"  Best: {best['Features']} ({best['N_Features']} features)")
    print(f"  Score: {best['CV_Mean']:.5f} (±{best['CV_Std']:.5f})")

# 全体ベスト
print("\n" + "="*80)
print("OVERALL BEST CONFIGURATION")
print("="*80)

best_overall = results_df.loc[results_df['CV_Mean'].idxmax()]
print(f"\nModel: {best_overall['Model']}")
print(f"Features: {best_overall['Features']} ({best_overall['N_Features']} features)")
print(f"CV Score: {best_overall['CV_Mean']:.5f} (±{best_overall['CV_Std']:.5f})")

# 結果を保存
results_df.to_csv('experiments/results/20241024_production_pipeline_evaluation.csv', index=False)
print("\n" + "="*80)
print("Results saved to: experiments/results/20241024_production_pipeline_evaluation.csv")
print("="*80)
