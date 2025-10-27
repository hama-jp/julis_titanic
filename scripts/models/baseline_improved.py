"""
改善版ベースラインモデル - 特徴量を追加してLB 0.8を目指す
追加特徴量:
- Fare per person (Fare / FamilySize)
- Age * Pclass (交互作用)
- Cabin Deck (デッキ情報)
- Ticket Group Size (チケットグループ人数)
- Age bins (年齢のカテゴリ化)
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


def create_improved_features(df, train_df=None):
    """改善された特徴量を作成"""
    # 全データを結合してグループ情報を計算（trainとtestで一貫性を保つため）
    data = df.copy()

    # === 基本特徴量 ===
    # 1. Sex: male=1, female=0
    data['Sex_male'] = (data['Sex'] == 'male').astype(int)

    # 2. Pclass: そのまま使用

    # 3. Age: 欠損値をTitle別の中央値で補完
    if train_df is not None:
        # trainデータから学習
        combined = pd.concat([train_df, df], sort=False, ignore_index=True)
    else:
        combined = data.copy()

    # Titleを抽出
    combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Titleをグループ化
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    combined['Title'] = combined['Title'].map(title_mapping)
    combined['Title'] = combined['Title'].fillna('Rare')

    # Title別のAge中央値で補完
    age_by_title = combined.groupby('Title')['Age'].median()

    # data用にTitleを抽出
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna('Rare')

    data['Age_fillna'] = data.apply(
        lambda row: age_by_title[row['Title']] if pd.isna(row['Age']) else row['Age'],
        axis=1
    )

    # 4. FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 5. IsAlone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    # 6. Fare: 欠損値と0を中央値で補完
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

    # === 追加特徴量 ===
    # 9. Fare per person
    data['FarePerPerson'] = data['Fare_fillna'] / data['FamilySize']

    # 10. Age * Pclass（交互作用）
    data['Age_Pclass'] = data['Age_fillna'] * data['Pclass']

    # 11. Cabin Deck
    data['Cabin_Deck'] = data['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')
    # 出現頻度が低いデッキをOtherにまとめる
    deck_counts = data['Cabin_Deck'].value_counts()
    rare_decks = deck_counts[deck_counts < 10].index
    data['Cabin_Deck'] = data['Cabin_Deck'].apply(lambda x: 'Other' if x in rare_decks else x)

    # Cabin Deck one-hot
    for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Unknown']:
        data[f'Deck_{deck}'] = (data['Cabin_Deck'] == deck).astype(int)

    # 12. Ticket Group Size
    # trainとtestを結合してチケットグループを計算
    # combinedは既にignore_index=Trueで作成されているので、Ticketで集計
    if train_df is not None:
        # trainとtestを結合してチケット数を計算
        all_tickets = pd.concat([train_df['Ticket'], df['Ticket']], ignore_index=True)
        ticket_counts = all_tickets.value_counts()
    else:
        ticket_counts = data['Ticket'].value_counts()
    data['TicketGroupSize'] = data['Ticket'].map(ticket_counts)

    # 13. Ticket Group Survival Rate (trainデータのみから計算)
    if train_df is not None:
        # trainデータからチケットごとの生存率を計算
        ticket_survival = train_df.groupby('Ticket')['Survived'].mean()
        # グループサイズ1の場合は全体の生存率を使用
        overall_survival = train_df['Survived'].mean()
        data['TicketGroupSurvivalRate'] = data['Ticket'].map(ticket_survival).fillna(overall_survival)
    else:
        # trainデータ自身の場合は0で初期化（後でfoldごとに計算）
        data['TicketGroupSurvivalRate'] = 0.0

    # 14. Age bins
    data['AgeBin'] = pd.cut(data['Age_fillna'], bins=[0, 12, 18, 35, 60, 100],
                            labels=['Child', 'Teen', 'Adult', 'MiddleAge', 'Senior'])
    # Age bin one-hot
    data['Age_Child'] = (data['AgeBin'] == 'Child').astype(int)
    data['Age_Teen'] = (data['AgeBin'] == 'Teen').astype(int)
    data['Age_Adult'] = (data['AgeBin'] == 'Adult').astype(int)
    data['Age_MiddleAge'] = (data['AgeBin'] == 'MiddleAge').astype(int)
    data['Age_Senior'] = (data['AgeBin'] == 'Senior').astype(int)

    # 15. Sex * Pclass（交互作用）
    data['Sex_Pclass'] = data['Sex_male'] * data['Pclass']

    return data


def get_feature_columns():
    """使用する特徴量のリストを返す"""
    features = [
        # 基本特徴
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
        'Title_Rare',
        # 追加特徴
        'FarePerPerson',
        'Age_Pclass',
        'Deck_A',
        'Deck_B',
        'Deck_C',
        'Deck_D',
        'Deck_E',
        'Deck_F',
        'Deck_G',
        'Deck_Unknown',
        'TicketGroupSize',
        # 'TicketGroupSurvivalRate',  # リークの可能性があるため一旦除外
        'Age_Child',
        'Age_Teen',
        'Age_Adult',
        'Age_MiddleAge',
        'Age_Senior',
        'Sex_Pclass'
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
    print("改善版ベースラインモデル - 特徴量追加")
    print("=" * 80)

    # データ読み込み
    print("\n[1] データ読み込み")
    train, test = load_data()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # 特徴量作成
    print("\n[2] 特徴量作成")
    train_features = create_improved_features(train)
    test_features = create_improved_features(test, train_df=train)

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
            n_estimators=300,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            C=0.5,
            max_iter=1000,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=8,
            min_samples_leaf=3,
            subsample=0.8,
            random_state=42
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            num_leaves=25,
            min_child_samples=8,
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

    # アンサンブル（平均）
    print("\n[5] アンサンブル")
    oof_ensemble = np.mean(list(oof_predictions_dict.values()), axis=0)
    oof_ensemble_binary = (oof_ensemble > 0.5).astype(int)
    oof_ensemble_acc = accuracy_score(y_train, oof_ensemble_binary)
    oof_ensemble_auc = roc_auc_score(y_train, oof_ensemble)

    print(f"Ensemble OOF Accuracy: {oof_ensemble_acc:.4f}, AUC: {oof_ensemble_auc:.4f}")

    # 重み付きアンサンブル（OOFスコアベース）
    # 上位3モデルを選択
    model_scores = [(name, accuracy_score(y_train, (oof > 0.5).astype(int)))
                    for name, oof in oof_predictions_dict.items()]
    model_scores_sorted = sorted(model_scores, key=lambda x: x[1], reverse=True)
    top3_models = [name for name, _ in model_scores_sorted[:3]]

    print(f"\nTop 3 models: {top3_models}")

    # Top3の重み付き平均
    top3_weights = np.array([model_scores_sorted[i][1] for i in range(3)])
    top3_weights = top3_weights / top3_weights.sum()

    print(f"Weights: {dict(zip(top3_models, top3_weights))}")

    test_ensemble_top3 = sum(test_predictions_dict[name] * weight
                              for name, weight in zip(top3_models, top3_weights))
    test_ensemble_top3_binary = (test_ensemble_top3 > 0.5).astype(int)

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
    submission_path = 'submissions/baseline_improved_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"全モデル平均: {submission_path}")

    # Top3重み付き平均
    submission_top3 = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_ensemble_top3_binary
    })
    submission_top3_path = 'submissions/baseline_improved_top3_weighted.csv'
    submission_top3.to_csv(submission_top3_path, index=False)
    print(f"Top3重み付き: {submission_top3_path}")

    # 各モデルの個別提出ファイル
    for model_name, test_pred in test_predictions_dict.items():
        test_pred_binary = (test_pred > 0.5).astype(int)
        submission_model = pd.DataFrame({
            'PassengerId': test['PassengerId'],
            'Survived': test_pred_binary
        })
        model_path = f'submissions/baseline_improved_{model_name.lower()}.csv'
        submission_model.to_csv(model_path, index=False)
        print(f"  {model_name}: {model_path}")

    print("\n" + "=" * 80)
    print("完了！")
    print("=" * 80)
    print(f"\nベースライン比較:")
    print(f"  シンプル版: OOF 0.8451, AUC 0.8810")
    print(f"  改善版:     OOF {oof_ensemble_acc:.4f}, AUC {oof_ensemble_auc:.4f}")
    print(f"  改善幅:     {oof_ensemble_acc - 0.8451:+.4f}")


if __name__ == '__main__':
    main()
