"""
Deep Analysis to Avoid Wasted Submissions
Goal: Identify the most reliable model before submitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('='*80)
print('DEEP ANALYSIS: Finding the Most Reliable Model')
print('='*80)

# 特徴量エンジニアリング
def engineer_features(df):
    df = df.copy()

    # Title抽出
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # FamilySize & IsAlone
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Age補完
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fare補完
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # Embarked補完
    df['Embarked'] = df['Embarked'].fillna('S')

    # 追加特徴量
    df['IsChild'] = (df['Age'] < 16).astype(int)
    df['FareLevel'] = pd.cut(df['Fare'], bins=[0, 10, 30, 100, 1000],
                              labels=[0, 1, 2, 3])
    df['FareLevel'] = df['FareLevel'].cat.codes

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)

# エンコーディング
le_sex = LabelEncoder()
train_fe['Sex'] = le_sex.fit_transform(train_fe['Sex'])
test_fe['Sex'] = le_sex.transform(test_fe['Sex'])

le_embarked = LabelEncoder()
train_fe['Embarked'] = le_embarked.fit_transform(train_fe['Embarked'])
test_fe['Embarked'] = le_embarked.transform(test_fe['Embarked'])

le_title = LabelEncoder()
train_fe['Title'] = le_title.fit_transform(train_fe['Title'])
test_fe['Title'] = le_title.transform(test_fe['Title'])

print('\n' + '='*80)
print('STEP 1: Train/Test Distribution Comparison')
print('='*80)

# 数値特徴量の分布比較
numeric_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']

print('\n--- Distribution Comparison (Train vs Test) ---')
dist_comparison = []

for feat in numeric_features:
    train_mean = train_fe[feat].mean()
    test_mean = test_fe[feat].mean()
    train_std = train_fe[feat].std()
    test_std = test_fe[feat].std()

    mean_diff = abs(train_mean - test_mean)
    std_diff = abs(train_std - test_std)

    dist_comparison.append({
        'Feature': feat,
        'Train_Mean': train_mean,
        'Test_Mean': test_mean,
        'Mean_Diff': mean_diff,
        'Train_Std': train_std,
        'Test_Std': test_std,
        'Std_Diff': std_diff
    })

dist_df = pd.DataFrame(dist_comparison)
print(dist_df.to_string(index=False))

# 大きな分布の違いを特定
print('\n--- Features with Large Distribution Differences ---')
large_diff = dist_df[dist_df['Mean_Diff'] > 0.1]
if len(large_diff) > 0:
    print(large_diff[['Feature', 'Train_Mean', 'Test_Mean', 'Mean_Diff']])
    print('\n⚠️  WARNING: Large distribution differences detected!')
    print('    This may explain CV/LB gap.')
else:
    print('✓ No large distribution differences found.')

print('\n' + '='*80)
print('STEP 2: CV Stability Analysis (Fold-by-Fold Variance)')
print('='*80)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 複数のモデルでfold間のばらつきを分析
models_to_test = {
    'LR_V5': {
        'model': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'features': ['Pclass', 'Sex', 'Title', 'Age', 'Embarked']
    },
    'LR_V6': {
        'model': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'features': ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title']
    },
    'RF_Conservative': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=4,
                                       min_samples_split=15, min_samples_leaf=5,
                                       random_state=42),
        'features': ['Pclass', 'Sex', 'Title', 'Fare']
    },
    'GB_Conservative': {
        'model': GradientBoostingClassifier(n_estimators=50, max_depth=3,
                                            learning_rate=0.05, min_samples_split=15,
                                            random_state=42),
        'features': ['Pclass', 'Sex', 'Title', 'Fare']
    }
}

stability_results = []

for model_name, config in models_to_test.items():
    X = train_fe[config['features']]
    y = train_fe['Survived']

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = config['model']
        model.fit(X_train, y_train)
        score = accuracy_score(y_val, model.predict(X_val))
        fold_scores.append(score)

    stability_results.append({
        'Model': model_name,
        'Mean': np.mean(fold_scores),
        'Std': np.std(fold_scores),
        'Min': np.min(fold_scores),
        'Max': np.max(fold_scores),
        'Range': np.max(fold_scores) - np.min(fold_scores),
        'Fold_Scores': fold_scores
    })

stability_df = pd.DataFrame(stability_results)
print('\n--- CV Stability (Lower Std = More Stable) ---')
print(stability_df[['Model', 'Mean', 'Std', 'Min', 'Max', 'Range']].to_string(index=False))

# 各foldのスコアを詳細表示
print('\n--- Detailed Fold Scores ---')
for result in stability_results:
    print(f"\n{result['Model']}:")
    for i, score in enumerate(result['Fold_Scores'], 1):
        print(f"  Fold {i}: {score:.4f}")

# 最も安定したモデルを特定
most_stable = stability_df.loc[stability_df['Std'].idxmin()]
print(f"\n✓ Most Stable Model: {most_stable['Model']} (Std: {most_stable['Std']:.4f})")

print('\n' + '='*80)
print('STEP 3: Out-of-Fold (OOF) Predictions Analysis')
print('='*80)

# OOF予測で実際のCV性能を確認
print('\n--- Generating OOF Predictions ---')

oof_results = []

for model_name, config in models_to_test.items():
    X = train_fe[config['features']].values
    y = train_fe['Survived'].values

    # OOF予測を生成
    oof_predictions = cross_val_predict(
        config['model'], X, y, cv=cv, method='predict'
    )

    oof_accuracy = accuracy_score(y, oof_predictions)

    # Train全体での予測（オーバーフィッティング確認）
    model = config['model']
    model.fit(X, y)
    train_predictions = model.predict(X)
    train_accuracy = accuracy_score(y, train_predictions)

    overfit_gap = train_accuracy - oof_accuracy

    oof_results.append({
        'Model': model_name,
        'OOF_Accuracy': oof_accuracy,
        'Train_Accuracy': train_accuracy,
        'Overfit_Gap': overfit_gap
    })

oof_df = pd.DataFrame(oof_results)
print(oof_df.to_string(index=False))

print('\n--- Overfitting Analysis ---')
high_overfit = oof_df[oof_df['Overfit_Gap'] > 0.05]
if len(high_overfit) > 0:
    print('⚠️  Models with high overfitting (gap > 0.05):')
    print(high_overfit[['Model', 'Overfit_Gap']])
else:
    print('✓ All models show acceptable overfitting levels.')

print('\n' + '='*80)
print('STEP 4: Feature Importance & Leak Detection')
print('='*80)

# PassengerIdがリークしていないか確認
print('\n--- Checking for Data Leaks ---')
print('PassengerId ranges:')
print(f"  Train: {train['PassengerId'].min()} - {train['PassengerId'].max()}")
print(f"  Test:  {test['PassengerId'].min()} - {test['PassengerId'].max()}")

# PassengerIdと目的変数の相関
correlation = train['PassengerId'].corr(train['Survived'])
print(f'\nCorrelation between PassengerId and Survived: {correlation:.4f}')
if abs(correlation) > 0.1:
    print('⚠️  WARNING: Potential leak detected!')
else:
    print('✓ No obvious leak from PassengerId.')

print('\n' + '='*80)
print('STEP 5: Final Model Recommendation')
print('='*80)

# 総合評価
print('\n--- Comprehensive Model Ranking ---')

final_scores = []
for i, row in stability_df.iterrows():
    model_name = row['Model']
    oof_row = oof_df[oof_df['Model'] == model_name].iloc[0]

    # スコアリング基準：
    # 1. OOF Accuracy (高いほど良い)
    # 2. Stability (Stdが低いほど良い)
    # 3. Overfitting (Gapが小さいほど良い)

    stability_score = 1 / (1 + row['Std'])  # 低いStd = 高いスコア
    overfit_score = 1 / (1 + oof_row['Overfit_Gap'])  # 低いGap = 高いスコア

    # 総合スコア（重み付け）
    composite_score = (
        oof_row['OOF_Accuracy'] * 0.5 +
        stability_score * 0.3 +
        overfit_score * 0.2
    )

    final_scores.append({
        'Model': model_name,
        'OOF_Acc': oof_row['OOF_Accuracy'],
        'CV_Std': row['Std'],
        'Overfit_Gap': oof_row['Overfit_Gap'],
        'Composite_Score': composite_score
    })

final_df = pd.DataFrame(final_scores).sort_values('Composite_Score', ascending=False)
print(final_df.to_string(index=False))

print('\n' + '='*80)
print('RECOMMENDATION FOR NEXT SUBMISSION')
print('='*80)

best_model = final_df.iloc[0]
second_best = final_df.iloc[1]

print(f"""
Based on comprehensive analysis:

🥇 PRIMARY RECOMMENDATION: {best_model['Model']}
   - OOF Accuracy: {best_model['OOF_Acc']:.4f}
   - CV Stability (Std): {best_model['CV_Std']:.4f}
   - Overfitting Gap: {best_model['Overfit_Gap']:.4f}
   - Composite Score: {best_model['Composite_Score']:.4f}

   Why: {'Low CV variance (stable)' if best_model['CV_Std'] < 0.015 else 'Good balance'},
        {'minimal overfitting' if best_model['Overfit_Gap'] < 0.03 else 'acceptable overfitting'}

🥈 SECONDARY RECOMMENDATION: {second_best['Model']}
   - OOF Accuracy: {second_best['OOF_Acc']:.4f}
   - CV Stability (Std): {second_best['CV_Std']:.4f}
   - Overfitting Gap: {second_best['Overfit_Gap']:.4f}
   - Composite Score: {second_best['Composite_Score']:.4f}

STRATEGY:
1. Submit the PRIMARY model first
2. If LB score is within 0.02 of OOF score → Good alignment, continue refining
3. If LB score is >0.03 below OOF → Try SECONDARY model (different approach)
4. Reserve remaining submissions for fine-tuning the best performer
""")

# 推奨モデルでテストデータの予測を生成
print('\n' + '='*80)
print('Generating Prediction for Recommended Model')
print('='*80)

best_model_name = best_model['Model']
best_config = models_to_test[best_model_name]

X_train = train_fe[best_config['features']]
y_train = train_fe['Survived']
X_test = test_fe[best_config['features']]

# モデル訓練
final_model = best_config['model']
final_model.fit(X_train, y_train)
predictions = final_model.predict(X_test)

# 提出ファイル作成
output_filename = f'submission_RECOMMENDED_{best_model_name}.csv'
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv(output_filename, index=False)

print(f'\n✓ Created: {output_filename}')
print(f'  Features: {best_config["features"]}')
print(f'  Expected LB: {best_model["OOF_Acc"]:.4f} ± 0.02')
print(f'  Survival rate: {predictions.mean():.3f}')

print('\n' + '='*80)
print('KEY INSIGHTS')
print('='*80)

print(f"""
1. Train/Test Distribution:
   {'⚠️  Some differences detected - may affect LB performance' if len(large_diff) > 0 else '✓ Distributions are similar'}

2. CV Stability:
   Most stable model has Std={most_stable['Std']:.4f}
   {'✓ Very stable predictions' if most_stable['Std'] < 0.015 else '⚠️  Some variance between folds'}

3. Overfitting:
   Best model overfit gap: {best_model['Overfit_Gap']:.4f}
   {'✓ Minimal overfitting' if best_model['Overfit_Gap'] < 0.03 else '⚠️  Some overfitting present'}

4. Expected LB Performance:
   Based on OOF accuracy: {best_model['OOF_Acc']:.4f}
   Realistic LB range: {best_model['OOF_Acc'] - 0.02:.4f} - {best_model['OOF_Acc'] + 0.01:.4f}

5. Comparison to Current Best:
   Current LB: 0.7703
   Expected improvement: {best_model['OOF_Acc'] - 0.7703:.4f}
   {'✓ Expected to improve' if best_model['OOF_Acc'] > 0.7703 else '⚠️  May not improve significantly'}
""")
