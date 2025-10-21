"""
Next Generation Models - Based on Submission #2 Results

Current Status:
- Submission #1: CV 0.8373, LB 0.7703 (gap 0.067)
- Submission #2: OOF 0.8283, LB 0.7799 (gap 0.048)
- Gap reduced by 27.8% ✓
- Need to close remaining 0.048 gap

Strategy:
1. Test even simpler models (LR with stable features)
2. Remove Fare (largest distribution gap)
3. Focus on features with good Train/Test alignment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('='*80)
print('NEXT GENERATION MODELS - Closing the Gap')
print('='*80)
print('Current Best: LB 0.7799 (Submission #2)')
print('Gap to Close: 0.048')
print('Remaining Submissions: 8')
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

    # Family features
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
    df['IsElderly'] = (df['Age'] >= 60).astype(int)

    # Fare binning (より粗く)
    df['FareBin'] = pd.qcut(df['Fare'], q=3, labels=[0, 1, 2], duplicates='drop')
    df['FareBin'] = df['FareBin'].cat.codes

    # Age binning (よりシンプルに)
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 16, 32, 48, 100],
                             labels=[0, 1, 2, 3])
    df['AgeGroup'] = df['AgeGroup'].cat.codes

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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('\n' + '='*80)
print('STRATEGY 1: Remove Fare (Largest Distribution Gap)')
print('='*80)

# Fareを除外した特徴量セット
feature_sets_no_fare = {
    'NoFare_V1': ['Pclass', 'Sex', 'Title', 'Age'],
    'NoFare_V2': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone'],
    'NoFare_V3': ['Pclass', 'Sex', 'Title', 'Age', 'Embarked'],
    'NoFare_V4': ['Pclass', 'Sex', 'Title', 'AgeGroup', 'IsAlone'],
}

print('\n--- Testing Feature Sets WITHOUT Fare ---')

results = []

for fs_name, features in feature_sets_no_fare.items():
    X = train_fe[features]
    y = train_fe['Survived']

    # Logistic Regression (最も安定)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    oof_lr = cross_val_predict(lr, X, y, cv=cv, method='predict')
    oof_lr_acc = accuracy_score(y, oof_lr)

    # RandomForest (保守的)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=15,
                                 min_samples_leaf=5, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    oof_rf = cross_val_predict(rf, X, y, cv=cv, method='predict')
    oof_rf_acc = accuracy_score(y, oof_rf)

    results.append({
        'FeatureSet': fs_name,
        'Features': len(features),
        'LR_CV': lr_scores.mean(),
        'LR_Std': lr_scores.std(),
        'LR_OOF': oof_lr_acc,
        'RF_CV': rf_scores.mean(),
        'RF_Std': rf_scores.std(),
        'RF_OOF': oof_rf_acc,
        'feature_list': features
    })

results_df = pd.DataFrame(results).sort_values('LR_OOF', ascending=False)
print(results_df[['FeatureSet', 'Features', 'LR_OOF', 'LR_Std', 'RF_OOF', 'RF_Std']].to_string(index=False))

print('\n' + '='*80)
print('STRATEGY 2: Ultra-Simple Models (Target CV ~0.79-0.80)')
print('='*80)

# 非常にシンプルな特徴量セット
simple_sets = {
    'Minimal_3': ['Pclass', 'Sex', 'Title'],
    'Minimal_4': ['Pclass', 'Sex', 'Title', 'Age'],
    'Minimal_5': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone'],
}

print('\n--- Ultra-Simple Feature Sets ---')

simple_results = []

for fs_name, features in simple_sets.items():
    X = train_fe[features]
    y = train_fe['Survived']

    # Logistic Regression (強い正則化)
    lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')
    oof_lr = cross_val_predict(lr, X, y, cv=cv, method='predict')
    oof_lr_acc = accuracy_score(y, oof_lr)

    simple_results.append({
        'FeatureSet': fs_name,
        'Features': len(features),
        'LR_OOF': oof_lr_acc,
        'LR_Std': lr_scores.std(),
        'feature_list': features
    })

simple_df = pd.DataFrame(simple_results).sort_values('LR_OOF', ascending=False)
print(simple_df.to_string(index=False))

print('\n' + '='*80)
print('STRATEGY 3: Weighted Ensemble')
print('='*80)

# 最良の特徴量セットでアンサンブル
best_features = results_df.iloc[0]['feature_list']
X_train = train_fe[best_features]
y_train = train_fe['Survived']
X_test = test_fe[best_features]

print(f'\nUsing features: {best_features}')

# Weighted Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4,
                                      min_samples_split=15, min_samples_leaf=5,
                                      random_state=42)),
    ],
    voting='soft',
    weights=[2, 1]  # LRに重み
)

ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
oof_ensemble = cross_val_predict(ensemble, X_train, y_train, cv=cv, method='predict')
oof_ensemble_acc = accuracy_score(y_train, oof_ensemble)

print(f'Ensemble OOF: {oof_ensemble_acc:.4f} (+/- {ensemble_scores.std():.4f})')

print('\n' + '='*80)
print('CREATING SUBMISSIONS')
print('='*80)

submissions = []

# 1. Best LR model (No Fare)
best_lr_set = results_df.iloc[0]
X_train = train_fe[best_lr_set['feature_list']]
y_train = train_fe['Survived']
X_test = test_fe[best_lr_set['feature_list']]

lr_final = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
lr_final.fit(X_train, y_train)
pred_lr = lr_final.predict(X_test)

submissions.append({
    'filename': f'submission_NextGen_LR_{best_lr_set["FeatureSet"]}.csv',
    'predictions': pred_lr,
    'oof': best_lr_set['LR_OOF'],
    'description': f'LR with {best_lr_set["feature_list"]}',
    'expected_lb': f'{best_lr_set["LR_OOF"] - 0.02:.4f}-{best_lr_set["LR_OOF"] + 0.01:.4f}'
})

# 2. Best RF model (No Fare)
rf_final = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=15,
                                   min_samples_leaf=5, random_state=42)
rf_final.fit(X_train, y_train)
pred_rf = rf_final.predict(X_test)

submissions.append({
    'filename': f'submission_NextGen_RF_{best_lr_set["FeatureSet"]}.csv',
    'predictions': pred_rf,
    'oof': best_lr_set['RF_OOF'],
    'description': f'RF with {best_lr_set["feature_list"]}',
    'expected_lb': f'{best_lr_set["RF_OOF"] - 0.02:.4f}-{best_lr_set["RF_OOF"] + 0.01:.4f}'
})

# 3. Ultra-simple LR
simple_best = simple_df.iloc[0]
X_train = train_fe[simple_best['feature_list']]
y_train = train_fe['Survived']
X_test = test_fe[simple_best['feature_list']]

lr_simple = LogisticRegression(C=0.05, max_iter=1000, random_state=42)
lr_simple.fit(X_train, y_train)
pred_simple = lr_simple.predict(X_test)

submissions.append({
    'filename': f'submission_NextGen_Simple_{simple_best["FeatureSet"]}.csv',
    'predictions': pred_simple,
    'oof': simple_best['LR_OOF'],
    'description': f'Simple LR with {simple_best["feature_list"]}',
    'expected_lb': f'{simple_best["LR_OOF"] - 0.02:.4f}-{simple_best["LR_OOF"] + 0.01:.4f}'
})

# 4. Ensemble
ensemble.fit(X_train, y_train)
pred_ensemble = ensemble.predict(X_test)

submissions.append({
    'filename': 'submission_NextGen_Ensemble_LR_RF.csv',
    'predictions': pred_ensemble,
    'oof': oof_ensemble_acc,
    'description': f'Weighted Ensemble (LR:2, RF:1) with {best_lr_set["feature_list"]}',
    'expected_lb': f'{oof_ensemble_acc - 0.02:.4f}-{oof_ensemble_acc + 0.01:.4f}'
})

# ファイル保存
print('\n--- Creating Submission Files ---\n')

for i, sub in enumerate(submissions, 1):
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': sub['predictions']
    })
    submission.to_csv(sub['filename'], index=False)

    survival_rate = sub['predictions'].mean()
    print(f"{i}. {sub['filename']}")
    print(f"   OOF: {sub['oof']:.4f} | Expected LB: {sub['expected_lb']}")
    print(f"   Survival Rate: {survival_rate:.3f}")
    print(f"   Description: {sub['description']}")
    print()

print('=' * 80)
print('RECOMMENDATION FOR SUBMISSION #3')
print('=' * 80)

# OOFスコアでソート
submissions.sort(key=lambda x: abs(x['oof'] - 0.7799), reverse=False)  # 現在のLBに近いものを優先

best_next = submissions[0]
second_best = submissions[1]

print(f"""
Based on Submission #2 results (LB 0.7799, gap 0.048):

🥇 PRIMARY RECOMMENDATION: {best_next['filename']}
   - OOF: {best_next['oof']:.4f}
   - Expected LB: {best_next['expected_lb']}
   - {best_next['description']}

   Why: OOF score closest to current LB (0.7799)
        Likely to have smallest gap
        Safe, predictable improvement

🥈 SECONDARY RECOMMENDATION: {second_best['filename']}
   - OOF: {second_best['oof']:.4f}
   - Expected LB: {second_best['expected_lb']}
   - {second_best['description']}

STRATEGY:
1. Submit PRIMARY first → establish stable baseline
2. If gap is small (<0.03) → continue with similar approach
3. If gap is still large → try ensemble or feature engineering
4. Reserve 7 submissions for optimization

KEY INSIGHT:
- Removed Fare (largest distribution gap)
- May improve Train/Test alignment
- Focus on stable, reliable features
""")

print('\n' + '=' * 80)
print('NEXT STEPS')
print('=' * 80)
print("""
1. Submit the PRIMARY recommendation
2. Record LB score in RESULTS_LOG.md
3. Calculate new gap
4. If gap < 0.03: Continue optimizing this approach
5. If gap >= 0.03: Try different strategy
6. Target: Close gap to <0.02 by submission #5
""")
