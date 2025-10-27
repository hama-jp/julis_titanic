"""
Seedアンサンブル版 - 複数のseedで学習して安定性を向上
最適化版(OOF 0.8451)をベースに、複数seedでアンサンブル
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


def create_optimized_features(df, train_df=None):
    """最適化された特徴量を作成"""
    data = df.copy()

    # 1. Sex
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # 2. Title
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
    age_by_title = all_data.groupby('Title')['Age'].median()

    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

    # 3. Age
    data['Age_fillna'] = data.apply(
        lambda row: age_by_title[row['Title']] if pd.isna(row['Age']) else row['Age'],
        axis=1
    )

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
    data['Embarked_S'] = (data['Embarked'] == 'S').astype(int)

    # 8. Title one-hot
    data['Title_Mr'] = (data['Title'] == 'Mr').astype(int)
    data['Title_Miss'] = (data['Title'] == 'Miss').astype(int)
    data['Title_Mrs'] = (data['Title'] == 'Mrs').astype(int)
    data['Title_Master'] = (data['Title'] == 'Master').astype(int)
    data['Title_Rare'] = (data['Title'] == 'Rare').astype(int)

    # 9. FarePerPerson
    data['FarePerPerson'] = data['Fare_fillna'] / data['FamilySize']

    # 10. TicketGroupSize
    if train_df is not None:
        all_tickets = pd.concat([train_df['Ticket'], df['Ticket']], ignore_index=True)
        ticket_counts = all_tickets.value_counts()
    else:
        ticket_counts = data['Ticket'].value_counts()
    data['TicketGroupSize'] = data['Ticket'].map(ticket_counts)

    # 11. Age_Pclass
    data['Age_Pclass'] = data['Age_fillna'] * data['Pclass']

    # 12. Fare_Pclass
    data['Fare_Pclass'] = data['Fare_fillna'] / data['Pclass']

    return data


def get_feature_columns():
    """使用する特徴量のリストを返す"""
    features = [
        'Pclass', 'Sex_male', 'Age_fillna', 'SibSp', 'Parch',
        'FamilySize', 'IsAlone', 'Fare_fillna',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Title_Mr', 'Title_Miss', 'Title_Mrs', 'Title_Master', 'Title_Rare',
        'FarePerPerson', 'TicketGroupSize', 'Age_Pclass', 'Fare_Pclass'
    ]
    return features


def train_and_predict_with_seed(X_train, y_train, X_test, model_type, n_folds=5, seed=42):
    """指定されたseedでモデルを学習"""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    oof_predictions = np.zeros(len(X_train))
    test_predictions = np.zeros(len(X_test))

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # モデル定義
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_split=10,
                min_samples_leaf=4, max_features='sqrt', random_state=seed
            )
        elif model_type == 'lr':
            model = LogisticRegression(
                C=0.3, max_iter=1000, random_state=seed
            )
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=4,
                subsample=0.8, random_state=seed
            )
        elif model_type == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.04,
                num_leaves=20, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8,
                random_state=seed, verbose=-1
            )

        model.fit(X_tr, y_tr)

        # OOF予測
        oof_predictions[valid_idx] = model.predict_proba(X_val)[:, 1]

        # テスト予測
        test_predictions += model.predict_proba(X_test)[:, 1] / n_folds

    return oof_predictions, test_predictions


def main():
    print("=" * 80)
    print("Seedアンサンブル版 - 複数seedで学習して安定性向上")
    print("=" * 80)

    # データ読み込み
    print("\n[1] データ読み込み")
    train, test = load_data()

    # 特徴量作成
    print("\n[2] 特徴量作成")
    train_features = create_optimized_features(train)
    test_features = create_optimized_features(test, train_df=train)

    feature_cols = get_feature_columns()
    print(f"使用する特徴量: {len(feature_cols)}個")

    X_train = train_features[feature_cols]
    y_train = train['Survived']
    X_test = test_features[feature_cols]

    # 複数seedでアンサンブル
    print("\n[3] 複数seedでアンサンブル")
    seeds = [42, 123, 456, 789, 2024]
    n_seeds = len(seeds)
    print(f"Seeds: {seeds}")

    model_types = {
        'RandomForest': 'rf',
        'LogisticRegression': 'lr',
        'GradientBoosting': 'gb',
        'LightGBM': 'lgb'
    }

    # 各モデルタイプで複数seedの予測を集約
    all_oof_predictions = {}
    all_test_predictions = {}

    for model_name, model_type in model_types.items():
        print(f"\n{model_name}:")
        model_oof_list = []
        model_test_list = []

        for i, seed in enumerate(seeds, 1):
            oof_pred, test_pred = train_and_predict_with_seed(
                X_train, y_train, X_test, model_type, seed=seed
            )
            model_oof_list.append(oof_pred)
            model_test_list.append(test_pred)

            # 各seedのOOFスコア
            oof_acc = accuracy_score(y_train, (oof_pred > 0.5).astype(int))
            print(f"  Seed {seed}: OOF {oof_acc:.4f}")

        # seedの平均を取る
        model_oof_avg = np.mean(model_oof_list, axis=0)
        model_test_avg = np.mean(model_test_list, axis=0)

        all_oof_predictions[model_name] = model_oof_avg
        all_test_predictions[model_name] = model_test_avg

        # 平均OOFスコア
        oof_acc_avg = accuracy_score(y_train, (model_oof_avg > 0.5).astype(int))
        oof_auc_avg = roc_auc_score(y_train, model_oof_avg)
        print(f"  {model_name} Avg OOF: {oof_acc_avg:.4f}, AUC: {oof_auc_avg:.4f}")

    # 全モデルのアンサンブル
    print("\n[4] 全モデルアンサンブル")
    oof_ensemble = np.mean(list(all_oof_predictions.values()), axis=0)
    test_ensemble = np.mean(list(all_test_predictions.values()), axis=0)

    oof_ensemble_acc = accuracy_score(y_train, (oof_ensemble > 0.5).astype(int))
    oof_ensemble_auc = roc_auc_score(y_train, oof_ensemble)

    print(f"Final Ensemble OOF: {oof_ensemble_acc:.4f}, AUC: {oof_ensemble_auc:.4f}")

    # 提出ファイル作成
    print("\n[5] 提出ファイル作成")

    # 全モデル平均
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': (test_ensemble > 0.5).astype(int)
    })
    submission_path = 'submissions/baseline_seed_ensemble_all.csv'
    submission.to_csv(submission_path, index=False)
    print(f"全モデル平均: {submission_path}")

    # 各モデルの個別提出
    for model_name, test_pred in all_test_predictions.items():
        submission_model = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': (test_pred > 0.5).astype(int)
        })
        model_path = f'submissions/baseline_seed_{model_name.lower()}.csv'
        submission_model.to_csv(model_path, index=False)
        print(f"  {model_name}: {model_path}")

    # Top3モデルのアンサンブル
    model_scores = [(name, accuracy_score(y_train, (oof > 0.5).astype(int)))
                    for name, oof in all_oof_predictions.items()]
    model_scores_sorted = sorted(model_scores, key=lambda x: x[1], reverse=True)
    top3_models = [name for name, _ in model_scores_sorted[:3]]

    print(f"\nTop 3 models: {top3_models}")

    test_top3 = np.mean([all_test_predictions[name] for name in top3_models], axis=0)
    submission_top3 = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': (test_top3 > 0.5).astype(int)
    })
    submission_top3_path = 'submissions/baseline_seed_top3.csv'
    submission_top3.to_csv(submission_top3_path, index=False)
    print(f"Top3平均: {submission_top3_path}")

    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print(f"\n比較:")
    print(f"  最適化版(single seed):  OOF 0.8451, AUC 0.8858")
    print(f"  Seedアンサンブル版:    OOF {oof_ensemble_acc:.4f}, AUC {oof_ensemble_auc:.4f}")


if __name__ == '__main__':
    main()
