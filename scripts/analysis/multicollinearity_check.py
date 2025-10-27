"""
多重共線性の確認 - 特徴量間の相関を分析
"""

import pandas as pd
import numpy as np
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


def main():
    print("=" * 80)
    print("多重共線性の確認")
    print("=" * 80)

    # データ読み込み
    train, test = load_data()
    train_features = create_optimized_features(train)

    # 使用する特徴量
    feature_cols = [
        'Pclass', 'Sex_male', 'Age_fillna', 'SibSp', 'Parch',
        'FamilySize', 'IsAlone', 'Fare_fillna',
        'Embarked_C', 'Embarked_Q', 'Embarked_S',
        'Title_Mr', 'Title_Miss', 'Title_Mrs', 'Title_Master', 'Title_Rare',
        'FarePerPerson', 'TicketGroupSize', 'Age_Pclass', 'Fare_Pclass'
    ]

    X = train_features[feature_cols]

    # 相関行列を計算
    print("\n[1] Pearson相関係数の計算")
    corr_matrix = X.corr()

    # 高い相関（|r| > 0.7）を持つ特徴量ペアを抽出
    print("\n高い相関を持つ特徴量ペア（|r| > 0.7）:")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
                print(f"  {corr_matrix.columns[i]:20s} - {corr_matrix.columns[j]:20s}: {corr_matrix.iloc[i, j]:6.3f}")

    # VIF（分散拡大係数）を計算
    print("\n[2] VIF（分散拡大係数）の計算")
    print("VIF > 10: 多重共線性の問題あり")
    print("VIF > 5: 注意が必要")

    from sklearn.linear_model import LinearRegression

    vif_data = []
    for i, col in enumerate(feature_cols):
        X_temp = X.drop(columns=[col])
        y_temp = X[col]

        # 定数項を含まない線形回帰
        lr = LinearRegression()
        lr.fit(X_temp, y_temp)
        r_squared = lr.score(X_temp, y_temp)

        # VIF = 1 / (1 - R^2)
        if r_squared < 0.9999:  # R^2が1に近い場合の数値誤差対策
            vif = 1 / (1 - r_squared)
        else:
            vif = 9999.0

        vif_data.append((col, vif))

    vif_data_sorted = sorted(vif_data, key=lambda x: x[1], reverse=True)

    print("\nVIF値:")
    for col, vif in vif_data_sorted:
        status = "⚠️ 問題あり" if vif > 10 else ("⚠️ 注意" if vif > 5 else "")
        print(f"  {col:20s}: {vif:8.2f} {status}")

    # 問題のある特徴量を特定
    print("\n[3] 推奨される削除候補:")
    print("\n多重共線性が疑われる特徴量グループ:")

    print("\n1. FamilySize関連:")
    print("   - SibSp, Parch → FamilySize = SibSp + Parch + 1")
    print("   - IsAlone → FamilySize == 1から派生")
    print("   推奨: SibSp, Parchを削除（FamilySize, IsAloneを残す）")

    print("\n2. Embarked One-hot:")
    print("   - Embarked_C, Embarked_Q, Embarked_S（3つで冗長）")
    print("   推奨: Embarked_Sを削除（ベースラインとして）")

    print("\n3. Title One-hot:")
    print("   - Title_Mr, Title_Miss, Title_Mrs, Title_Master, Title_Rare（5つ）")
    print("   - Title_Mrは Sex_maleと高相関の可能性")
    print("   推奨: Title_Rareを削除（最も稀なカテゴリ）")

    print("\n4. 交互作用特徴:")
    print("   - Age_Pclass = Age * Pclass")
    print("   - Fare_Pclass = Fare / Pclass")
    print("   - これらは元の特徴量との相関が高い可能性")

    print("\n5. FarePerPerson vs TicketGroupSize:")
    print("   - FarePerPerson = Fare / FamilySize")
    print("   - TicketGroupSizeとFamilySizeは似た情報")

    # 推奨特徴量セット
    print("\n" + "=" * 80)
    print("推奨される特徴量セット（多重共線性を軽減）")
    print("=" * 80)

    recommended_features = [
        'Pclass',
        'Sex_male',
        'Age_fillna',
        # 'SibSp',  # 削除: FamilySizeに統合
        # 'Parch',  # 削除: FamilySizeに統合
        'FamilySize',
        'IsAlone',
        'Fare_fillna',
        'Embarked_C',
        'Embarked_Q',
        # 'Embarked_S',  # 削除: ベースライン
        'Title_Mr',
        'Title_Miss',
        'Title_Mrs',
        'Title_Master',
        # 'Title_Rare',  # 削除: 最も稀
        'FarePerPerson',
        'TicketGroupSize',
        # 'Age_Pclass',  # 削除: 相関が高い
        # 'Fare_Pclass',  # 削除: 相関が高い
    ]

    print(f"\n元の特徴量: {len(feature_cols)}個")
    print(f"推奨特徴量: {len(recommended_features)}個")
    print(f"削除: {len(feature_cols) - len(recommended_features)}個")

    print("\n推奨特徴量リスト:")
    for i, feat in enumerate(recommended_features, 1):
        print(f"  {i:2d}. {feat}")

    print("\n削除する特徴量:")
    removed = set(feature_cols) - set(recommended_features)
    for i, feat in enumerate(sorted(removed), 1):
        print(f"  {i:2d}. {feat}")


if __name__ == '__main__':
    main()
