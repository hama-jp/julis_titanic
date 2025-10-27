"""
最終版ベースラインモデル - Title特徴量を削除
TitleはAge補完にのみ使用し、特徴量としては使わない
（TitleはSexとAgeから派生した冗長な情報）
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """データを読み込む"""
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    return train, test


def create_final_features(df, train_df=None):
    """最終版特徴量作成（Titleは補完にのみ使用）"""
    data = df.copy()

    # 1. Sex
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # 2. Title抽出（Age補完用のみ）
    if train_df is not None:
        all_data = pd.concat([train_df, df], sort=False, ignore_index=True)
    else:
        all_data = data.copy()

    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    all_data['Title'] = all_data['Title'].map(title_mapping).fillna('Rare')

    # Pclass x Title別の中央値を計算
    age_pclass_title_median = all_data.groupby(['Pclass', 'Title'])['Age'].median()
    # Title別の中央値（フォールバック用）
    age_title_median = all_data.groupby('Title')['Age'].median()
    # 全体の中央値（最終フォールバック）
    age_overall_median = all_data['Age'].median()

    # data用にTitleを抽出（補完用）
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

    # 3. Age補完（Pclass x Title → Title → 全体の3段階フォールバック）
    def impute_age(row):
        if pd.notna(row['Age']):
            return row['Age']

        # まずPclass x Titleで試す
        pclass_title_key = (row['Pclass'], row['Title'])
        if pclass_title_key in age_pclass_title_median.index:
            return age_pclass_title_median[pclass_title_key]

        # 次にTitleで試す
        if row['Title'] in age_title_median.index:
            return age_title_median[row['Title']]

        # 最後に全体の中央値
        return age_overall_median

    data['Age_fillna'] = data.apply(impute_age, axis=1)

    # Titleはここで削除（特徴量として使わない）
    data.drop('Title', axis=1, inplace=True)

    # 4. FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 5. IsAlone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 6. Fare
    data['Fare_fillna'] = data['Fare'].fillna(data['Fare'].median())
    data.loc[data['Fare_fillna'] == 0, 'Fare_fillna'] = data['Fare'].median()

    # 7. Embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked_C'] = (data['Embarked'] == 'C').astype(int)
    data['Embarked_Q'] = (data['Embarked'] == 'Q').astype(int)

    # 8. FarePerPerson
    data['FarePerPerson'] = data['Fare_fillna'] / data['FamilySize']

    # 9. TicketGroupSize
    if train_df is not None:
        all_tickets = pd.concat([train_df['Ticket'], df['Ticket']], ignore_index=True)
        ticket_counts = all_tickets.value_counts()
    else:
        ticket_counts = data['Ticket'].value_counts()
    data['TicketGroupSize'] = data['Ticket'].map(ticket_counts)

    return data


def get_feature_columns():
    """使用する特徴量（Title関連を削除）"""
    features = [
        'Pclass',
        'Sex_male',
        'Age_fillna',
        'FamilySize',
        'IsAlone',
        'Fare_fillna',
        'Embarked_C',
        'Embarked_Q',
        'FarePerPerson',
        'TicketGroupSize'
    ]
    return features


def train_model_cv(X, y, model, model_name, n_folds=5, random_state=42):
    """クロスバリデーションでモデルを学習"""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_predictions = np.zeros(len(X))
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_train, y_train)

        # OOF予測
        oof_predictions[valid_idx] = model.predict_proba(X_valid)[:, 1]

        # Fold score
        fold_score = accuracy_score(y_valid, model.predict(X_valid))
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.4f}")

    # OOF全体のスコア
    oof_pred_binary = (oof_predictions > 0.5).astype(int)
    oof_accuracy = accuracy_score(y, oof_pred_binary)
    oof_auc = roc_auc_score(y, oof_predictions)

    print(f"{model_name} OOF Accuracy: {oof_accuracy:.4f}, AUC: {oof_auc:.4f}")
    print(f"{model_name} CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    return oof_predictions, oof_accuracy


def train_and_predict(X_train, y_train, X_test, model, n_folds=5, random_state=42):
    """クロスバリデーションで学習してテストデータを予測"""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    test_predictions = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(X_tr, y_tr)

        # テストデータの予測を累積
        test_predictions += model.predict_proba(X_test)[:, 1] / n_folds

    return test_predictions


def main():
    print("=" * 80)
    print("最終版ベースラインモデル - Title特徴量を削除")
    print("=" * 80)

    # データ読み込み
    print("\n[1] データ読み込み")
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # 特徴量作成
    print("\n[2] 特徴量作成（Titleを削除）")
    train_features = create_final_features(train)
    test_features = create_final_features(test, train_df=train)

    feature_cols = get_feature_columns()
    print(f"使用する特徴量: {len(feature_cols)}個（14個→10個に削減）")
    print("\n削除した特徴量:")
    print("  - Title_Mr, Title_Miss, Title_Mrs, Title_Master")
    print("  理由: TitleはSexとAgeから派生した冗長な情報")
    print("\n残した特徴量:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    print("\nTitleはAge補完にのみ使用（Pclass x Title別中央値）")

    X_train = train_features[feature_cols]
    y_train = train['Survived']
    X_test = test_features[feature_cols]

    # モデル定義
    print("\n[3] モデル定義")
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.3,
            max_iter=1000,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.04,
            num_leaves=20,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    }

    # 各モデルを学習してOOFスコアを取得
    print("\n[4] クロスバリデーション")
    oof_predictions_dict = {}
    test_predictions_dict = {}

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        oof_pred, oof_acc = train_model_cv(X_train, y_train, model, model_name)
        oof_predictions_dict[model_name] = oof_pred

        # テストデータの予測も取得
        test_pred = train_and_predict(X_train, y_train, X_test, model)
        test_predictions_dict[model_name] = test_pred

    # アンサンブル
    print("\n[5] アンサンブル")
    oof_ensemble = np.mean(list(oof_predictions_dict.values()), axis=0)
    oof_ensemble_binary = (oof_ensemble > 0.5).astype(int)
    oof_ensemble_acc = accuracy_score(y_train, oof_ensemble_binary)
    oof_ensemble_auc = roc_auc_score(y_train, oof_ensemble)

    print(f"Ensemble OOF Accuracy: {oof_ensemble_acc:.4f}, AUC: {oof_ensemble_auc:.4f}")

    # テストデータのアンサンブル予測
    test_ensemble = np.mean(list(test_predictions_dict.values()), axis=0)
    test_ensemble_binary = (test_ensemble > 0.5).astype(int)

    # 提出ファイル作成
    print("\n[6] 提出ファイル作成")

    # 全モデル平均
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_ensemble_binary
    })
    submission_path = 'submissions/baseline_final_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"全モデル平均: {submission_path}")

    # 各モデルの個別提出ファイル
    for model_name, test_pred in test_predictions_dict.items():
        test_pred_binary = (test_pred > 0.5).astype(int)
        submission_model = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': test_pred_binary
        })
        model_path = f'submissions/baseline_final_{model_name.lower()}.csv'
        submission_model.to_csv(model_path, index=False)
        print(f"  {model_name}: {model_path}")

    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print(f"\n比較:")
    print(f"  Title特徴量あり (14特徴): OOF 0.8474, LB 0.75119")
    print(f"  Title特徴量なし (10特徴): OOF {oof_ensemble_acc:.4f}, LB ???")
    print(f"\n冗長な特徴量を削除し、過学習を軽減")


if __name__ == '__main__':
    main()
