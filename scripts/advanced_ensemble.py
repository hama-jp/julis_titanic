"""
Advanced Ensemble Model (Stacking)

This script implements a stacking ensemble to improve the LB score.

Strategy:
- Use a diverse set of base models (LGBM, GB, RF, LR).
- Generate out-of-fold (OOF) predictions for each base model.
- Train a meta-model (Logistic Regression) on the OOF predictions.
- The meta-model learns to combine the base model predictions optimally.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# --- Stacking Configuration ---
NFOLDS = 5
SEED = 42

# --- Model Definitions ---
# Base models (with tuned hyperparameters)
rf_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'criterion': 'gini',
    'random_state': SEED,
    'n_jobs': -1
}
gb_params = {
    'n_estimators': 200,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'random_state': SEED
}
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'num_leaves': 20,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': SEED
}
lr_params = {
    'C': 0.1,
    'solver': 'liblinear',
    'random_state': SEED
}

rf = RandomForestClassifier(**rf_params)
gb = GradientBoostingClassifier(**gb_params)
lgb_clf = lgb.LGBMClassifier(**lgb_params)
lr = LogisticRegression(**lr_params)

base_models = {
    'rf': rf,
    'gb': gb,
    'lgb': lgb_clf,
    'lr': lr
}

# Meta-model
meta_model = LogisticRegression(C=1.0, random_state=SEED)


def get_oof_predictions(clf, x_train, y_train, x_test, n_folds=NFOLDS, seed=SEED):
    """
    Generates out-of-fold prediction probabilities for a given classifier.
    """
    oof_train = np.zeros((len(x_train),))
    oof_test = np.zeros((len(x_test),))
    oof_test_skf = np.empty((n_folds, len(x_test)))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for i, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_te = x_train.iloc[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict_proba(x_te)[:, 1]
        oof_test_skf[i, :] = clf.predict_proba(x_test)[:, 1]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def main():
    print('='*80)
    print('ADVANCED ENSEMBLE (STACKING)')
    print('='*80)

    # Load data
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    # Feature engineering (re-using the function from the previous script)
    def engineer_features(df):
        df = df.copy()
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master', 'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare',
            'Major': 'Rare', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare', 'Lady': 'Rare',
            'Countess': 'Rare', 'Jonkheer': 'Rare', 'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
        }
        df['Title'] = df['Title'].map(title_mapping)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
        df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
        df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
        df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)
        df['Embarked'] = df['Embarked'].fillna('S')
        df['Pclass_Sex'] = df['Pclass'].astype(str) + '_' + df['Sex']
        return df

    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    # Encoding
    for col in ['Sex', 'Embarked', 'Title', 'Pclass_Sex']:
        le = LabelEncoder()
        train_fe[col] = le.fit_transform(train_fe[col])
        test_fe[col] = le.transform(test_fe[col])

    features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin', 'Pclass_Sex', 'SmallFamily']
    X_train = train_fe[features]
    y_train = train_fe['Survived']
    X_test = test_fe[features]

    print("Generating OOF predictions for base models...")
    oof_train_list = []
    oof_test_list = []

    for name, model in base_models.items():
        print(f"  - Training {name}...")
        oof_train, oof_test = get_oof_predictions(model, X_train, y_train, X_test)
        oof_train_list.append(oof_train)
        oof_test_list.append(oof_test)
        print(f"    OOF accuracy for {name}: {accuracy_score(y_train, oof_train.round())}")


    # Create new feature set for meta-model
    X_train_meta = np.concatenate(oof_train_list, axis=1)
    X_test_meta = np.concatenate(oof_test_list, axis=1)

    print("\nTraining meta-model...")
    meta_model.fit(X_train_meta, y_train)

    # Make final predictions
    meta_predictions_proba = meta_model.predict_proba(X_test_meta)[:, 1]
    predictions = (meta_predictions_proba > 0.5).astype(int)

    print("\n" + "="*80)
    print("SUBMISSION FILE CREATED")
    print("="*80)

    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': predictions
    })

    filename = 'submissions/submission_advanced_ensemble.csv'
    submission.to_csv(filename, index=False)
    print(f"Generated filename: {filename}")
    print(f"Ensemble survival rate: {predictions.mean():.3f}")

if __name__ == '__main__':
    main()
