"""
FareとFarePerPersonの関係を分析
FarePerPerson = Fare / FamilySizeなので冗長性がある可能性
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """データを読み込む"""
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    return train, test


def create_features(df, train_df=None, use_fare=True, use_fare_per_person=True):
    """特徴量作成（FareとFarePerPersonの使用を選択可能）"""
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

    age_pclass_title_median = all_data.groupby(['Pclass', 'Title'])['Age'].median()
    age_title_median = all_data.groupby('Title')['Age'].median()
    age_overall_median = all_data['Age'].median()

    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data['Title'] = data['Title'].map(title_mapping).fillna('Rare')

    # 3. Age補完
    def impute_age(row):
        if pd.notna(row['Age']):
            return row['Age']
        pclass_title_key = (row['Pclass'], row['Title'])
        if pclass_title_key in age_pclass_title_median.index:
            return age_pclass_title_median[pclass_title_key]
        if row['Title'] in age_title_median.index:
            return age_title_median[row['Title']]
        return age_overall_median

    data['Age_fillna'] = data.apply(impute_age, axis=1)
    data.drop('Title', axis=1, inplace=True)

    # 4. FamilySize
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    # 5. Fare（オプション）
    if use_fare:
        data['Fare_fillna'] = data['Fare'].fillna(data['Fare'].median())
        data.loc[data['Fare_fillna'] == 0, 'Fare_fillna'] = data['Fare'].median()

    # 6. FarePerPerson（オプション）
    if use_fare_per_person:
        if not use_fare:
            # Fareを一時的に作成
            fare_temp = data['Fare'].fillna(data['Fare'].median())
            fare_temp.loc[fare_temp == 0] = data['Fare'].median()
            data['FarePerPerson'] = fare_temp / data['FamilySize']
        else:
            data['FarePerPerson'] = data['Fare_fillna'] / data['FamilySize']

    # 7. Embarked
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked_C'] = (data['Embarked'] == 'C').astype(int)
    data['Embarked_Q'] = (data['Embarked'] == 'Q').astype(int)

    return data


def get_feature_columns(use_fare=True, use_fare_per_person=True):
    """使用する特徴量"""
    features = [
        'Pclass',
        'Sex_male',
        'Age_fillna',
        'FamilySize',
        'Embarked_C',
        'Embarked_Q'
    ]
    if use_fare:
        features.append('Fare_fillna')
    if use_fare_per_person:
        features.append('FarePerPerson')
    return features


def train_and_evaluate(X_train, y_train, n_folds=5):
    """GradientBoostingで評価"""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_predictions = np.zeros(len(X_train))

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            min_samples_split=10, min_samples_leaf=4, subsample=0.8, random_state=42
        )

        model.fit(X_tr, y_tr)
        oof_predictions[valid_idx] = model.predict_proba(X_val)[:, 1]

    oof_pred_binary = (oof_predictions > 0.5).astype(int)
    oof_accuracy = accuracy_score(y_train, oof_pred_binary)
    oof_auc = roc_auc_score(y_train, oof_predictions)

    return oof_accuracy, oof_auc


def main():
    print("=" * 80)
    print("FareとFarePerPersonの関係を分析")
    print("=" * 80)

    train, test = load_data()

    print("\n[1] FareとFarePerPersonの関係")
    print("FarePerPerson = Fare / FamilySize")
    print("→ Fareの情報は既にFarePerPersonに含まれている")
    print("→ 両方を使うと冗長性がある可能性")

    # 相関を確認
    train_temp = create_features(train, use_fare=True, use_fare_per_person=True)
    corr = train_temp[['Fare_fillna', 'FarePerPerson', 'FamilySize']].corr()
    print("\n相関行列:")
    print(corr)

    print("\n" + "=" * 80)
    print("[2] 3つのパターンを比較（GradientBoostingで評価）")
    print("=" * 80)

    y_train = train['Survived']

    # パターン1: Fareのみ
    print("\n【パターン1】Fareのみ")
    train_features_1 = create_features(train, use_fare=True, use_fare_per_person=False)
    feature_cols_1 = get_feature_columns(use_fare=True, use_fare_per_person=False)
    print(f"特徴量 ({len(feature_cols_1)}個): {', '.join(feature_cols_1)}")
    X_train_1 = train_features_1[feature_cols_1]
    oof_acc_1, oof_auc_1 = train_and_evaluate(X_train_1, y_train)
    print(f"OOF Accuracy: {oof_acc_1:.4f}, AUC: {oof_auc_1:.4f}")

    # パターン2: FarePerPersonのみ
    print("\n【パターン2】FarePerPersonのみ")
    train_features_2 = create_features(train, use_fare=False, use_fare_per_person=True)
    feature_cols_2 = get_feature_columns(use_fare=False, use_fare_per_person=True)
    print(f"特徴量 ({len(feature_cols_2)}個): {', '.join(feature_cols_2)}")
    X_train_2 = train_features_2[feature_cols_2]
    oof_acc_2, oof_auc_2 = train_and_evaluate(X_train_2, y_train)
    print(f"OOF Accuracy: {oof_acc_2:.4f}, AUC: {oof_auc_2:.4f}")

    # パターン3: 両方（現在の方法）
    print("\n【パターン3】Fare + FarePerPerson（現在の方法）")
    train_features_3 = create_features(train, use_fare=True, use_fare_per_person=True)
    feature_cols_3 = get_feature_columns(use_fare=True, use_fare_per_person=True)
    print(f"特徴量 ({len(feature_cols_3)}個): {', '.join(feature_cols_3)}")
    X_train_3 = train_features_3[feature_cols_3]
    oof_acc_3, oof_auc_3 = train_and_evaluate(X_train_3, y_train)
    print(f"OOF Accuracy: {oof_acc_3:.4f}, AUC: {oof_auc_3:.4f}")

    # パターン4: どちらも使わない
    print("\n【パターン4】Fareなし（参考）")
    train_features_4 = create_features(train, use_fare=False, use_fare_per_person=False)
    feature_cols_4 = get_feature_columns(use_fare=False, use_fare_per_person=False)
    print(f"特徴量 ({len(feature_cols_4)}個): {', '.join(feature_cols_4)}")
    X_train_4 = train_features_4[feature_cols_4]
    oof_acc_4, oof_auc_4 = train_and_evaluate(X_train_4, y_train)
    print(f"OOF Accuracy: {oof_acc_4:.4f}, AUC: {oof_auc_4:.4f}")

    # 比較表
    print("\n" + "=" * 80)
    print("[3] 比較結果")
    print("=" * 80)
    print(f"\n{'パターン':30s} | {'特徴量数':8s} | {'OOF Acc':10s} | {'AUC':10s}")
    print("-" * 80)
    print(f"{'1. Fareのみ':30s} | {len(feature_cols_1):8d} | {oof_acc_1:10.4f} | {oof_auc_1:10.4f}")
    print(f"{'2. FarePerPersonのみ':30s} | {len(feature_cols_2):8d} | {oof_acc_2:10.4f} | {oof_auc_2:10.4f}")
    print(f"{'3. Fare + FarePerPerson':30s} | {len(feature_cols_3):8d} | {oof_acc_3:10.4f} | {oof_auc_3:10.4f}")
    print(f"{'4. Fareなし（参考）':30s} | {len(feature_cols_4):8d} | {oof_acc_4:10.4f} | {oof_auc_4:10.4f}")

    print("\n" + "=" * 80)
    print("結論")
    print("=" * 80)

    results = [
        (1, "Fareのみ", oof_acc_1),
        (2, "FarePerPersonのみ", oof_acc_2),
        (3, "Fare + FarePerPerson", oof_acc_3),
        (4, "Fareなし", oof_acc_4)
    ]
    best = max(results, key=lambda x: x[2])
    print(f"\n最良のパターン: パターン{best[0]} - {best[1]}")
    print(f"OOF Accuracy: {best[2]:.4f}")

    if best[0] == 1:
        print("\n推奨: Fareのみを使用")
        print("理由: FarePerPersonは冗長、Fareだけで十分")
    elif best[0] == 2:
        print("\n推奨: FarePerPersonのみを使用")
        print("理由: 一人当たり運賃の方が生存率との関連が強い")
    elif best[0] == 3:
        print("\n推奨: 両方を使用")
        print("理由: それぞれが独立した情報を持つ")
    else:
        print("\n推奨: Fare関連の特徴量は不要")
        print("理由: 他の特徴量で十分")


if __name__ == '__main__':
    main()
