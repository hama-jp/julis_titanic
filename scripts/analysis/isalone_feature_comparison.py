"""
IsAlone特徴量の有無を比較
IsAloneはFamilySize==1から派生した冗長な特徴量の可能性がある
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


def create_features(df, train_df=None, include_isalone=True):
    """特徴量作成（IsAloneの有無を選択可能）"""
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

    # 5. IsAlone（オプション）
    if include_isalone:
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


def get_feature_columns(include_isalone=True):
    """使用する特徴量"""
    features = [
        'Pclass',
        'Sex_male',
        'Age_fillna',
        'FamilySize',
        'Fare_fillna',
        'Embarked_C',
        'Embarked_Q',
        'FarePerPerson',
        'TicketGroupSize'
    ]
    if include_isalone:
        features.insert(4, 'IsAlone')  # FamilySizeの後に挿入
    return features


def train_and_evaluate(X_train, y_train, model_name, n_folds=5, random_state=42):
    """モデルを学習して評価"""
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_predictions = np.zeros(len(X_train))

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        # モデル定義
        if model_name == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_split=10,
                min_samples_leaf=4, max_features='sqrt', random_state=42
            )
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(C=0.3, max_iter=1000, random_state=42)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                min_samples_split=10, min_samples_leaf=4, subsample=0.8, random_state=42
            )
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(
                n_estimators=250, max_depth=5, learning_rate=0.04,
                num_leaves=20, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            )

        model.fit(X_tr, y_tr)
        oof_predictions[valid_idx] = model.predict_proba(X_val)[:, 1]

    # OOF全体のスコア
    oof_pred_binary = (oof_predictions > 0.5).astype(int)
    oof_accuracy = accuracy_score(y_train, oof_pred_binary)
    oof_auc = roc_auc_score(y_train, oof_predictions)

    return oof_accuracy, oof_auc


def main():
    print("=" * 80)
    print("IsAlone特徴量の有無を比較")
    print("=" * 80)

    # データ読み込み
    train, test = load_data()

    print("\n[1] IsAloneの冗長性について")
    print("IsAloneはFamilySizeから派生:")
    print("  IsAlone = 1 if FamilySize == 1 else 0")
    print("\nFamilySizeだけで同じ情報を表現できるため、冗長な可能性がある")

    # IsAloneの分布を確認
    train_temp = create_features(train, include_isalone=True)
    print("\n[2] IsAloneとFamilySizeの関係（訓練データ）")
    print(pd.crosstab(train_temp['FamilySize'], train_temp['IsAlone'], margins=True))

    print("\n[3] 生存率との関係")
    print("\nFamilySize別の生存率:")
    print(train.groupby(train_temp['FamilySize'])['Survived'].agg(['count', 'mean']))

    print("\nIsAlone別の生存率:")
    print(train.groupby(train_temp['IsAlone'])['Survived'].agg(['count', 'mean']))

    # 2つのバージョンで比較
    print("\n" + "=" * 80)
    print("[4] モデル性能の比較")
    print("=" * 80)

    models = ['RandomForest', 'LogisticRegression', 'GradientBoosting', 'LightGBM']

    results_with_isalone = {}
    results_without_isalone = {}

    # IsAloneあり
    print("\n【IsAloneあり】10特徴量")
    train_features_with = create_features(train, include_isalone=True)
    feature_cols_with = get_feature_columns(include_isalone=True)
    print(f"特徴量: {', '.join(feature_cols_with)}")
    X_train_with = train_features_with[feature_cols_with]
    y_train = train['Survived']

    for model_name in models:
        oof_acc, oof_auc = train_and_evaluate(X_train_with, y_train, model_name)
        results_with_isalone[model_name] = (oof_acc, oof_auc)
        print(f"  {model_name:20s}: OOF {oof_acc:.4f}, AUC {oof_auc:.4f}")

    # アンサンブル計算（概算）
    avg_acc_with = np.mean([acc for acc, _ in results_with_isalone.values()])
    avg_auc_with = np.mean([auc for _, auc in results_with_isalone.values()])
    print(f"  {'Ensemble (avg)':20s}: OOF {avg_acc_with:.4f}, AUC {avg_auc_with:.4f}")

    # IsAloneなし
    print("\n【IsAloneなし】9特徴量")
    train_features_without = create_features(train, include_isalone=False)
    feature_cols_without = get_feature_columns(include_isalone=False)
    print(f"特徴量: {', '.join(feature_cols_without)}")
    X_train_without = train_features_without[feature_cols_without]

    for model_name in models:
        oof_acc, oof_auc = train_and_evaluate(X_train_without, y_train, model_name)
        results_without_isalone[model_name] = (oof_acc, oof_auc)
        print(f"  {model_name:20s}: OOF {oof_acc:.4f}, AUC {oof_auc:.4f}")

    # アンサンブル計算（概算）
    avg_acc_without = np.mean([acc for acc, _ in results_without_isalone.values()])
    avg_auc_without = np.mean([auc for _, auc in results_without_isalone.values()])
    print(f"  {'Ensemble (avg)':20s}: OOF {avg_acc_without:.4f}, AUC {avg_auc_without:.4f}")

    # 比較表
    print("\n" + "=" * 80)
    print("[5] 比較結果")
    print("=" * 80)
    print(f"\n{'モデル':20s} | {'IsAloneあり':15s} | {'IsAloneなし':15s} | {'差分':10s}")
    print("-" * 80)
    for model_name in models:
        acc_with, auc_with = results_with_isalone[model_name]
        acc_without, auc_without = results_without_isalone[model_name]
        diff = acc_without - acc_with
        print(f"{model_name:20s} | {acc_with:.4f} ({auc_with:.4f}) | {acc_without:.4f} ({auc_without:.4f}) | {diff:+.4f}")

    print("-" * 80)
    diff_ensemble = avg_acc_without - avg_acc_with
    print(f"{'Ensemble (avg)':20s} | {avg_acc_with:.4f} ({avg_auc_with:.4f}) | {avg_acc_without:.4f} ({avg_auc_without:.4f}) | {diff_ensemble:+.4f}")

    print("\n" + "=" * 80)
    print("結論")
    print("=" * 80)
    if avg_acc_without > avg_acc_with:
        print(f"✓ IsAloneなしの方が良い（差分: {diff_ensemble:+.4f}）")
        print("  → IsAloneは冗長な特徴量。FamilySizeだけで十分")
    elif avg_acc_without < avg_acc_with:
        print(f"✓ IsAloneありの方が良い（差分: {diff_ensemble:+.4f}）")
        print("  → IsAloneは有用な特徴量。モデルが一人旅の重要性を学習")
    else:
        print("✓ ほぼ同じパフォーマンス")
        print("  → IsAloneなしの方が特徴量が少なくシンプル")


if __name__ == '__main__':
    main()
