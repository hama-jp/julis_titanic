"""
Titanic Model Improvement Strategy
Current: CV=0.837, LB=0.770 (gap=0.067)
Remaining submissions: 9

Strategy:
1. Reduce overfitting (gap too large)
2. Try simpler models
3. Ensemble methods
4. Feature engineering refinement
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('='*80)
print('TITANIC MODEL IMPROVEMENT - STRATEGIC APPROACH')
print('='*80)
print(f'Current Status: CV=0.837, LB=0.770 (Overfitting gap: 0.067)')
print(f'Remaining submissions: 9')
print('='*80)

# 特徴量エンジニアリング関数
def engineer_features(df, is_train=True):
    """特徴量エンジニアリング（汎化性重視）"""
    df = df.copy()

    # 1. Title抽出
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    df['Title'] = df['Title'].map(title_mapping)

    # 2. FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 3. IsAlone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 4. Age補完（TitleとPclassごとの中央値）
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )

    # 5. Fare補完
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )

    # 6. Embarked補完
    df['Embarked'] = df['Embarked'].fillna('S')

    # 7. Age binning (シンプルに)
    df['IsChild'] = (df['Age'] < 16).astype(int)

    # 8. Fare binning (シンプルに)
    df['FareLevel'] = pd.cut(df['Fare'], bins=[0, 10, 30, 100, 1000],
                              labels=[0, 1, 2, 3])
    df['FareLevel'] = df['FareLevel'].cat.codes

    return df

# 特徴量エンジニアリング適用
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

# 交差検証設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 特徴量セットの定義
feature_sets = {
    # より汎化性の高い特徴量セット
    'V1_Minimal': ['Pclass', 'Sex', 'Title'],
    'V2_WithAge': ['Pclass', 'Sex', 'Title', 'Age'],
    'V3_WithFare': ['Pclass', 'Sex', 'Title', 'Fare'],
    'V4_Balanced': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone'],
    'V5_WithEmbarked': ['Pclass', 'Sex', 'Title', 'Age', 'Embarked'],
    'V6_Original': ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title'],
    'V7_Simple': ['Pclass', 'Sex', 'Title', 'IsChild', 'FareLevel'],
    'V8_Family': ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone'],
}

print('\n' + '='*80)
print('STEP 1: Testing Different Feature Sets with Multiple Models')
print('='*80)

results = []

for fs_name, features in feature_sets.items():
    X_train = train_fe[features]
    y_train = train_fe['Survived']

    # モデル1: Logistic Regression (シンプル、汎化性高い)
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='accuracy')

    # モデル2: Random Forest (保守的パラメータ)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=15,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

    # モデル3: Gradient Boosting (保守的パラメータ)
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        min_samples_split=15, random_state=42
    )
    gb_scores = cross_val_score(gb, X_train, y_train, cv=cv, scoring='accuracy')

    results.append({
        'Features': fs_name,
        'Num_Features': len(features),
        'LR_mean': lr_scores.mean(),
        'LR_std': lr_scores.std(),
        'RF_mean': rf_scores.mean(),
        'RF_std': rf_scores.std(),
        'GB_mean': gb_scores.mean(),
        'GB_std': gb_scores.std(),
        'Avg_Score': (lr_scores.mean() + rf_scores.mean() + gb_scores.mean()) / 3,
        'feature_list': features
    })

results_df = pd.DataFrame(results).sort_values('Avg_Score', ascending=False)
print('\n--- Model Performance Comparison ---')
print(results_df[['Features', 'Num_Features', 'LR_mean', 'RF_mean', 'GB_mean', 'Avg_Score']].to_string(index=False))

print('\n' + '='*80)
print('STEP 2: Creating Submission Files for Top Candidates')
print('='*80)

# 提出用モデルを作成
submissions_to_create = []

# Top 3特徴量セットを選択
top_feature_sets = results_df.head(3)

for idx, row in top_feature_sets.iterrows():
    features = row['feature_list']
    fs_name = row['Features']

    X_train = train_fe[features]
    y_train = train_fe['Survived']
    X_test = test_fe[features]

    # Logistic Regression
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    submissions_to_create.append({
        'name': f'submission_{fs_name}_LR.csv',
        'predictions': pred_lr,
        'cv_score': row['LR_mean'],
        'description': f'{fs_name} with Logistic Regression'
    })

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=15,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    submissions_to_create.append({
        'name': f'submission_{fs_name}_RF.csv',
        'predictions': pred_rf,
        'cv_score': row['RF_mean'],
        'description': f'{fs_name} with Random Forest'
    })

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        min_samples_split=15, random_state=42
    )
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)

    submissions_to_create.append({
        'name': f'submission_{fs_name}_GB.csv',
        'predictions': pred_gb,
        'cv_score': row['GB_mean'],
        'description': f'{fs_name} with Gradient Boosting'
    })

print('\n' + '='*80)
print('STEP 3: Creating Ensemble Models')
print('='*80)

# アンサンブル1: ベスト特徴量セットでVoting
best_features = top_feature_sets.iloc[0]['feature_list']
X_train = train_fe[best_features]
y_train = train_fe['Survived']
X_test = test_fe[best_features]

# Soft Voting
voting_soft = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=15,
                                      min_samples_leaf=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                          min_samples_split=15, random_state=42))
    ],
    voting='soft'
)

voting_soft_scores = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
print(f'\nVoting Ensemble (Soft) CV Score: {voting_soft_scores.mean():.4f} (+/- {voting_soft_scores.std():.4f})')

voting_soft.fit(X_train, y_train)
pred_voting_soft = voting_soft.predict(X_test)

submissions_to_create.append({
    'name': 'submission_Ensemble_VotingSoft.csv',
    'predictions': pred_voting_soft,
    'cv_score': voting_soft_scores.mean(),
    'description': 'Voting Ensemble (LR+RF+GB, Soft)'
})

# Hard Voting
voting_hard = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=15,
                                      min_samples_leaf=5, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                          min_samples_split=15, random_state=42))
    ],
    voting='hard'
)

voting_hard_scores = cross_val_score(voting_hard, X_train, y_train, cv=cv, scoring='accuracy')
print(f'Voting Ensemble (Hard) CV Score: {voting_hard_scores.mean():.4f} (+/- {voting_hard_scores.std():.4f})')

voting_hard.fit(X_train, y_train)
pred_voting_hard = voting_hard.predict(X_test)

submissions_to_create.append({
    'name': 'submission_Ensemble_VotingHard.csv',
    'predictions': pred_voting_hard,
    'cv_score': voting_hard_scores.mean(),
    'description': 'Voting Ensemble (LR+RF+GB, Hard)'
})

print('\n' + '='*80)
print('STEP 4: Saving Submission Files')
print('='*80)

# CV scoreでソート
submissions_to_create.sort(key=lambda x: x['cv_score'], reverse=True)

# 上位候補を保存
print('\nTop Submission Candidates (sorted by CV score):')
print('-' * 80)

for i, sub in enumerate(submissions_to_create[:10], 1):  # Top 10を表示
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': sub['predictions']
    })
    submission.to_csv(sub['name'], index=False)

    survival_rate = sub['predictions'].mean()
    print(f"{i:2d}. {sub['name']:40s} | CV: {sub['cv_score']:.4f} | Survival: {survival_rate:.3f}")
    print(f"    {sub['description']}")

print('\n' + '='*80)
print('SUBMISSION STRATEGY RECOMMENDATION')
print('='*80)
print("""
Recommended submission order (use 9 remaining submissions):

PHASE 1 - Test Simpler Models (Submissions 2-4):
- Submit top 3 models with lowest CV variance (most stable)
- Focus on Logistic Regression and simple ensembles
- Goal: Find model with better CV/LB alignment

PHASE 2 - Ensemble Exploration (Submissions 5-7):
- Submit voting ensembles
- Try weighted combinations if Phase 1 shows patterns
- Goal: Improve through ensemble diversity

PHASE 3 - Final Optimization (Submissions 8-10):
- Based on LB feedback, fine-tune best approach
- Try minor variations of best performer
- Reserve last submission for the absolute best model

Key Insights:
1. Current gap (CV 0.837 vs LB 0.770) suggests overfitting
2. Simpler models may perform better on LB despite lower CV
3. Watch for models with low CV variance (more stable)
4. Ensemble methods can reduce variance and improve generalization
""")

print('\n' + '='*80)
print('NEXT STEPS')
print('='*80)
print("""
1. Review the submissions created above
2. Choose 2-3 candidates for next submission:
   - Pick models with CV ~0.78-0.80 (closer to current LB)
   - Prefer simpler models (fewer features, simpler algorithms)
   - Consider ensemble models for stability

3. Track results in a spreadsheet:
   - Model name | Features | Algorithm | CV Score | LB Score | Gap

4. Adjust strategy based on LB feedback
""")
