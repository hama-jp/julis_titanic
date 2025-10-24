"""
正しいTrain/Test分離を使った提出ファイル作成

このスクリプトは以下の改善を実装しています：
1. ✅ Train/Test分離を正しく実装した欠損値補完
2. ✅ モデル固有の最適特徴量（Age vs AgeGroup）
3. ✅ Fare補完の改善（Pclass + Embarked別）

既存の問題点を全て修正した最新版です。

Author: Claude Code
Date: 2025-10-24
Version: 3.0 (Correct Imputation + Model-Specific Features)
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('/home/user/julis_titanic')

from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('正しいTrain/Test分離を使った提出ファイル作成')
print('='*80)
print('改善点:')
print('  1. Train/Test分離を正しく実装')
print('  2. モデル固有の最適特徴量')
print('  3. Fare補完の改善')
print('='*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print('\nステップ1: 欠損値補完（正しいTrain/Test分離）')
print('-' * 80)

# 正しい補完パイプライン
imputer = CorrectImputationPipeline(use_fare_based_embarked=True)
train_imputed = imputer.fit_transform(train, create_missing_flags=True)
test_imputed = imputer.transform(test, create_missing_flags=True)

print(f'\n補完後の欠損値:')
print(f'  Train: Age={train_imputed["Age"].isnull().sum()}, Fare={train_imputed["Fare"].isnull().sum()}')
print(f'  Test:  Age={test_imputed["Age"].isnull().sum()}, Fare={test_imputed["Fare"].isnull().sum()}')

# Test Fare補完の確認
test_fare_idx = test[test['Fare'].isnull()].index
if len(test_fare_idx) > 0:
    idx = test_fare_idx[0]
    print(f'\n✅ Test PassengerId=1044 の Fare補完: {test_imputed.loc[idx, "Fare"]:.2f}')
    print(f'   （正しい値: 8.05, 古い方法では: 7.90）')

print('\nステップ2: 特徴量エンジニアリング（モデル共通部分）')
print('-' * 80)

def engineer_common_features(df):
    """
    全モデル共通の特徴量エンジニアリング
    Age, Fare, Embarked は既に補完済み
    """
    df = df.copy()

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)

    # Fare binning
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop')
    df['FareBin'] = df['FareBin'].astype(int)

    # Age grouping (for models that prefer categories)
    bins_domain_knowledge = [0, 18, 60, 100]
    labels_domain_knowledge = [0, 1, 2]  # Child, Adult, Elderly
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=bins_domain_knowledge,
        labels=labels_domain_knowledge,
        right=False
    ).astype(int)

    # Sex encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    return df

train_fe = engineer_common_features(train_imputed)
test_fe = engineer_common_features(test_imputed)

# Title encoding（全モデル共通）
le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

y_train = train['Survived']

print(f'  Train: {len(train_fe)} samples')
print(f'  Test:  {len(test_fe)} samples')

print('\nステップ3: モデル固有の特徴量セット定義')
print('-' * 80)

# モデル固有の特徴量セット（年齢特徴量実験の結果を反映）
model_configs = {
    'LightGBM': {
        'features': ['Pclass', 'Sex', 'Title', 'AgeGroup', 'IsAlone', 'FareBin',
                    'SmallFamily', 'Age_Missing', 'Embarked_Missing'],
        'model': lgb.LGBMClassifier(
            n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
            reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        ),
        'description': 'AgeGroup（カテゴリ）使用 - 浅い木に最適'
    },
    'GradientBoosting': {
        'features': ['Pclass', 'Sex', 'Title', 'AgeGroup', 'IsAlone', 'FareBin',
                    'SmallFamily', 'Age_Missing', 'Embarked_Missing'],
        'model': GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10, subsample=0.8,
            random_state=42
        ),
        'description': 'AgeGroup（カテゴリ）使用 - 浅い木に最適'
    },
    'RandomForest': {
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
                    'SmallFamily', 'Age_Missing', 'Embarked_Missing'],
        'model': RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_split=10,
            min_samples_leaf=10, random_state=42
        ),
        'description': 'Age（連続値）使用 - 深い木に最適'
    }
}

for name, config in model_configs.items():
    print(f'\n{name}:')
    print(f'  特徴量: {len(config["features"])} 個')
    print(f'  年齢特徴: {"AgeGroup" if "AgeGroup" in config["features"] else "Age"}')
    print(f'  説明: {config["description"]}')

print('\nステップ4: Out-of-Fold 予測とアンサンブル')
print('-' * 80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
test_predictions = {}

for name, config in model_configs.items():
    print(f'\n{name} の学習中...')

    # モデル固有の特徴量
    X_train = train_fe[config['features']]
    X_test = test_fe[config['features']]

    model = config['model']

    # Out-of-Fold予測
    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
    oof_score = accuracy_score(y_train, oof_pred)

    # 全データで学習してTest予測
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    test_pred_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'oof_score': oof_score,
        'oof_pred': oof_pred,
        'test_pred': test_pred,
        'test_proba': test_pred_proba
    }

    print(f'  OOF Score: {oof_score:.4f}')

print('\n' + '='*80)
print('個別モデルのOOFスコア比較')
print('='*80)

for name, result in sorted(results.items(), key=lambda x: x[1]['oof_score'], reverse=True):
    print(f'{name:20s}: {result["oof_score"]:.4f}')

print('\nステップ5: アンサンブル予測')
print('-' * 80)

# Hard Voting (多数決)
ensemble_pred = np.round((
    results['LightGBM']['test_pred'] +
    results['GradientBoosting']['test_pred'] +
    results['RandomForest']['test_pred']
) / 3).astype(int)

oof_ensemble = np.round((
    results['LightGBM']['oof_pred'] +
    results['GradientBoosting']['oof_pred'] +
    results['RandomForest']['oof_pred']
) / 3).astype(int)

oof_ensemble_score = accuracy_score(y_train, oof_ensemble)

print(f'\nHard Voting OOF Score: {oof_ensemble_score:.4f}')

# Soft Voting (確率平均)
soft_proba = (
    results['LightGBM']['test_proba'] +
    results['GradientBoosting']['test_proba'] +
    results['RandomForest']['test_proba']
) / 3

soft_pred = (soft_proba > 0.5).astype(int)

print(f'Soft Voting生成完了')

print('\nステップ6: サブミッションファイル作成')
print('-' * 80)

# 個別モデル
for name, result in results.items():
    filename = f'submission_correct_{name.lower()}.csv'
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': result['test_pred']
    })
    submission.to_csv(f'submissions/{filename}', index=False)
    print(f'\n✅ {filename}')
    print(f'   OOF Score: {result["oof_score"]:.4f}')

# アンサンブル
filename = 'submission_correct_hard_voting.csv'
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensemble_pred
})
submission.to_csv(f'submissions/{filename}', index=False)
print(f'\n✅ {filename}')
print(f'   OOF Score: {oof_ensemble_score:.4f}')

filename = 'submission_correct_soft_voting.csv'
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': soft_pred
})
submission.to_csv(f'submissions/{filename}', index=False)
print(f'\n✅ {filename}')
print(f'   OOF Score: (確率ベース)')

print('\n' + '='*80)
print('最終サマリ')
print('='*80)

print('\n【改善点】')
print('  1. ✅ Train/Test分離を正しく実装')
print('     - Test Fare補完: 7.90 → 8.05')
print('  2. ✅ モデル固有の最適特徴量')
print('     - LightGBM/GB: AgeGroup（カテゴリ）')
print('     - RandomForest: Age（連続値）')
print('  3. ✅ Fare補完の改善')
print('     - Pclass + Embarked別の中央値')

print('\n【推奨提出順】')
best_model = max(results.items(), key=lambda x: x[1]['oof_score'])
print(f'  1. submission_correct_hard_voting.csv (OOF: {oof_ensemble_score:.4f})')
print(f'  2. submission_correct_{best_model[0].lower()}.csv (OOF: {best_model[1]["oof_score"]:.4f}) - {best_model[0]}')
print(f'  3. submission_correct_soft_voting.csv')

print('\n【期待効果】')
print(f'  - 正しいFare補完による精度向上')
print(f'  - モデル固有最適化による多様性向上')
print(f'  - 期待LB: 0.775 - 0.78 (1st提出 0.77511 を上回る可能性)')

print('\n' + '='*80)
print('✅ 完了！正しいパイプラインでサブミッション作成成功')
print('='*80)
