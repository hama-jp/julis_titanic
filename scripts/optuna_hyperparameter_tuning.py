"""
Optuna Hyperparameter Tuning

Strategy:
- Use Optuna to optimize hyperparameters for RF, GB, and LR
- Optimize based on cross-validation score
- Train final models with best parameters on all data
- Generate predictions from single best model and ensemble
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('OPTUNA HYPERPARAMETER TUNING')
print('='*80)
print('Optimizing: Random Forest, Gradient Boosting, Logistic Regression')
print('='*80)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# Feature engineering
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

print('\nEngineering features...')
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

# Use Enhanced_V4 features (best from previous experiments)
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex', 'SmallFamily']

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

print(f'Training data: {len(X_train)} samples, {len(features)} features')
print(f'Features: {", ".join(features)}')

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# 1. OPTIMIZE RANDOM FOREST
# ============================================================================
print('\n' + '='*80)
print('OPTIMIZING RANDOM FOREST')
print('='*80)

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    return score

optuna.logging.set_verbosity(optuna.logging.WARNING)
study_rf = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_rf.optimize(objective_rf, n_trials=100, show_progress_bar=True)

print(f'\nBest RF Score: {study_rf.best_value:.4f}')
print(f'Best RF Params: {study_rf.best_params}')

# ============================================================================
# 2. OPTIMIZE GRADIENT BOOSTING
# ============================================================================
print('\n' + '='*80)
print('OPTIMIZING GRADIENT BOOSTING')
print('='*80)

def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }

    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    return score

study_gb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_gb.optimize(objective_gb, n_trials=100, show_progress_bar=True)

print(f'\nBest GB Score: {study_gb.best_value:.4f}')
print(f'Best GB Params: {study_gb.best_params}')

# ============================================================================
# 3. OPTIMIZE LOGISTIC REGRESSION
# ============================================================================
print('\n' + '='*80)
print('OPTIMIZING LOGISTIC REGRESSION')
print('='*80)

def objective_lr(trial):
    params = {
        'C': trial.suggest_float('C', 0.001, 10.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l2']),
        'max_iter': 1000,
        'random_state': 42
    }

    model = LogisticRegression(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    return score

study_lr = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lr.optimize(objective_lr, n_trials=50, show_progress_bar=True)

print(f'\nBest LR Score: {study_lr.best_value:.4f}')
print(f'Best LR Params: {study_lr.best_params}')

# ============================================================================
# TRAIN FINAL MODELS WITH OPTIMIZED PARAMETERS
# ============================================================================
print('\n' + '='*80)
print('TRAINING FINAL MODELS WITH OPTIMIZED PARAMETERS')
print('='*80)

models = {}
predictions_dict = {}
scores = {}

# Train Random Forest
print('\n1. Random Forest')
rf_model = RandomForestClassifier(**study_rf.best_params)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
models['RF'] = rf_model
predictions_dict['RF'] = rf_pred
scores['RF'] = study_rf.best_value
print(f'   CV Score: {study_rf.best_value:.4f}')
print(f'   Training Accuracy: {rf_train_acc:.4f}')
print(f'   Survival Rate: {rf_pred.mean():.3f}')

# Train Gradient Boosting
print('\n2. Gradient Boosting')
gb_model = GradientBoostingClassifier(**study_gb.best_params)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_train_acc = accuracy_score(y_train, gb_model.predict(X_train))
models['GB'] = gb_model
predictions_dict['GB'] = gb_pred
scores['GB'] = study_gb.best_value
print(f'   CV Score: {study_gb.best_value:.4f}')
print(f'   Training Accuracy: {gb_train_acc:.4f}')
print(f'   Survival Rate: {gb_pred.mean():.3f}')

# Train Logistic Regression
print('\n3. Logistic Regression')
lr_model = LogisticRegression(**study_lr.best_params)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
models['LR'] = lr_model
predictions_dict['LR'] = lr_pred
scores['LR'] = study_lr.best_value
print(f'   CV Score: {study_lr.best_value:.4f}')
print(f'   Training Accuracy: {lr_train_acc:.4f}')
print(f'   Survival Rate: {lr_pred.mean():.3f}')

# ============================================================================
# SELECT BEST SINGLE MODEL
# ============================================================================
print('\n' + '='*80)
print('MODEL COMPARISON')
print('='*80)

best_model_name = max(scores, key=scores.get)
best_model_score = scores[best_model_name]

print(f'\nCV Scores:')
for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f'  {name}: {score:.4f}')

print(f'\nBest Single Model: {best_model_name} (CV: {best_model_score:.4f})')

# ============================================================================
# CREATE SUBMISSIONS
# ============================================================================
print('\n' + '='*80)
print('CREATING SUBMISSION FILES')
print('='*80)

# 1. Best single model submission
best_pred = predictions_dict[best_model_name]
submission_best = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': best_pred
})
filename_best = f'submission_optuna_{best_model_name.lower()}_single.csv'
submission_best.to_csv(filename_best, index=False)

print(f'\n1. Best Single Model: {filename_best}')
print(f'   Model: {best_model_name}')
print(f'   CV Score: {best_model_score:.4f}')
print(f'   Survival Rate: {best_pred.mean():.3f}')

# 2. Ensemble submission (average of all 3)
ensemble_pred = np.round(np.array([predictions_dict['RF'], predictions_dict['GB'], predictions_dict['LR']]).mean(axis=0)).astype(int)
submission_ensemble = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': ensemble_pred
})
filename_ensemble = 'submission_optuna_ensemble_rf_gb_lr.csv'
submission_ensemble.to_csv(filename_ensemble, index=False)

print(f'\n2. Ensemble Model: {filename_ensemble}')
print(f'   Models: RF + GB + LR')
print(f'   Average CV Score: {np.mean(list(scores.values())):.4f}')
print(f'   Survival Rate: {ensemble_pred.mean():.3f}')

# Show prediction agreement
print('\nPrediction Agreement:')
print(f'  RF vs GB: {np.sum(predictions_dict["RF"] == predictions_dict["GB"])} / {len(X_test)} ({np.sum(predictions_dict["RF"] == predictions_dict["GB"])/len(X_test):.1%})')
print(f'  RF vs LR: {np.sum(predictions_dict["RF"] == predictions_dict["LR"])} / {len(X_test)} ({np.sum(predictions_dict["RF"] == predictions_dict["LR"])/len(X_test):.1%})')
print(f'  GB vs LR: {np.sum(predictions_dict["GB"] == predictions_dict["LR"])} / {len(X_test)} ({np.sum(predictions_dict["GB"] == predictions_dict["LR"])/len(X_test):.1%})')
print(f'  All 3 agree: {np.sum((predictions_dict["RF"] == predictions_dict["GB"]) & (predictions_dict["GB"] == predictions_dict["LR"]))} / {len(X_test)}')

print('\n' + '='*80)
print('OPTIMIZATION COMPLETE!')
print('='*80)
print(f'\nGenerated Files:')
print(f'  1. {filename_best} (Best single model: {best_model_name})')
print(f'  2. {filename_ensemble} (Ensemble of all 3 models)')
print('\nRecommendation: Try the best single model first, then ensemble if needed.')
print('='*80)
