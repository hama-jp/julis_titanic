"""
シンプルなベースラインモデル - ゼロからLB 0.8を目指す
基本的な特徴量のみを使用したクリーンなアプローチ
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


def create_simple_features(df):
    """シンプルな特徴量を作成"""
    # コピーして作業
    data = df.copy()

    # 1. Sex: male=1, female=0
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # 2. Pclass: そのまま使用（1, 2, 3）
    # data['Pclass'] はそのまま

    # 3. Age: 欠損値を中央値で補完
    data['Age_fillna'] = data['Age'].fillna(data['Age'].median())

    # 4. FamilySize: SibSp + Parch + 1
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 5. IsAlone: FamilySize == 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 6. Fare: 欠損値を中央値で補完、0の場合も中央値で補完
    data['Fare_fillna'] = data['Fare'].fillna(data['Fare'].median())
    data.loc[data['Fare_fillna'] == 0, 'Fare_fillna'] = data['Fare'].median()

    # 7. Embarked: 欠損値を最頻値(S)で補完、One-hot encoding
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked_C'] = (data['Embarked'] == 'C').astype(int)
    data['Embarked_Q'] = (data['Embarked'] == 'Q').astype(int)
    data['Embarked_S'] = (data['Embarked'] == 'S').astype(int)

    # 8. Title: Nameから敬称を抽出
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Titleをグループ化
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Lady': 'Rare',
        'Countess': 'Rare',
        'Jonkheer': 'Rare',
        'Sir': 'Rare',
        'Capt': 'Rare',
        'Ms': 'Miss'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna('Rare')

    # Title One-hot encoding
    data['Title_Mr'] = (data['Title'] == 'Mr').astype(int)
    data['Title_Miss'] = (data['Title'] == 'Miss').astype(int)
    data['Title_Mrs'] = (data['Title'] == 'Mrs').astype(int)
    data['Title_Master'] = (data['Title'] == 'Master').astype(int)
    data['Title_Rare'] = (data['Title'] == 'Rare').astype(int)

    return data


def get_feature_columns():
    """使用する特徴量のリストを返す"""
    features = [
        'Pclass',
        'Sex_male',
        'Age_fillna',
        'SibSp',
        'Parch',
        'FamilySize',
        'IsAlone',
        'Fare_fillna',
        'Embarked_C',
        'Embarked_Q',
        'Embarked_S',
        'Title_Mr',
        'Title_Miss',
        'Title_Mrs',
        'Title_Master',
        'Title_Rare'
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
    print("シンプルなベースラインモデル - ゼロからのスタート")
    print("=" * 80)

    # データ読み込み
    print("\n[1] データ読み込み")
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # 特徴量作成
    print("\n[2] 特徴量作成")
    train_features = create_simple_features(train)
    test_features = create_simple_features(test)

    feature_cols = get_feature_columns()
    print(f"使用する特徴量 ({len(feature_cols)}個):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")

    X_train = train_features[feature_cols]
    y_train = train['Survived']
    X_test = test_features[feature_cols]

    # モデル定義
    print("\n[3] モデル定義")
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=20,
            min_child_samples=10,
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

    # アンサンブル（平均）
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
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_ensemble_binary
    })

    submission_path = 'submissions/baseline_simple_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"提出ファイル保存: {submission_path}")

    # 各モデルの個別提出ファイルも作成
    for model_name, test_pred in test_predictions_dict.items():
        test_pred_binary = (test_pred > 0.5).astype(int)
        submission_model = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': test_pred_binary
        })
        model_path = f'submissions/baseline_simple_{model_name.lower()}.csv'
        submission_model.to_csv(model_path, index=False)
        print(f"  {model_name}: {model_path}")

    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print("\n次のステップ:")
    print("1. 提出ファイルをKaggleにアップロードしてLBスコアを確認")
    print("2. LBスコアを確認後、特徴量を追加・改善")
    print("3. LB 0.8を目指す")


if __name__ == '__main__':
    main()
