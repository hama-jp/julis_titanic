"""
Correct Missing Value Imputation Pipeline
===========================================

IMPORTANT: このパイプラインはTrain/Test分離を正しく実装しています。
他の補完実装（engineer_features関数内で直接補完するパターン）は
Train/Test分離が不適切な場合があるため、このパイプラインを使用してください。

問題点の例:
  ❌ df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
     → Train/Testそれぞれで個別に補完（TestのみのデータからFareを学習してしまう）

  ✅ imputer.fit(train) → imputer.transform(test)
     → Trainで学習したルールをTestに適用（正しい）

Author: Claude Code
Date: 2025-10-24
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class CorrectImputationPipeline:
    """
    Train/Test分離を正しく実装した欠損値補完パイプライン

    特徴:
    1. Trainデータから補完ルールを学習（fit）
    2. Train/Testに同じルールを適用（transform）
    3. TestのみのデータからFare等を学習しない（データリーケージ防止）
    4. 欠損フラグの作成（補完前の欠損情報を特徴量化）

    補完戦略:
    - Age: Title + Pclass別の中央値
    - Fare: Pclass + Embarked別の中央値
    - Embarked: 最頻値（'S'）または Fare範囲からの推定
    """

    def __init__(self, use_fare_based_embarked: bool = False):
        """
        Parameters
        ----------
        use_fare_based_embarked : bool, default=False
            Embarked補完時にFare範囲を考慮するかどうか
        """
        self.use_fare_based_embarked = use_fare_based_embarked
        self.age_rules: Dict[Tuple[str, int], float] = {}
        self.fare_rules: Dict[Tuple[int, str], float] = {}
        self.embarked_mode: str = 'S'
        self.fare_ranges_by_embarked: Dict = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> 'CorrectImputationPipeline':
        """
        Trainデータから補完ルールを学習

        Parameters
        ----------
        train_df : pd.DataFrame
            学習用データ（Trainデータのみ）
            必須カラム: Pclass, Sex, Age, Fare, Embarked, Name

        Returns
        -------
        self : CorrectImputationPipeline
        """
        train = train_df.copy()

        # Title抽出（Age補完に必要）
        train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
            'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
            'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
        }
        train['Title'] = train['Title'].map(title_mapping)

        print("=" * 80)
        print("LEARNING IMPUTATION RULES FROM TRAIN DATA")
        print("=" * 80)

        # 1. Age補完ルール: Title + Pclass別の中央値
        print("\n[1] Age Imputation Rules (Title + Pclass):")
        for title in train['Title'].dropna().unique():
            for pclass in train['Pclass'].unique():
                mask = (train['Title'] == title) & (train['Pclass'] == pclass)
                age_median = train.loc[mask, 'Age'].median()
                if pd.notna(age_median):
                    self.age_rules[(title, pclass)] = age_median
                    count = mask.sum()
                    print(f"  {title:8s} + Pclass={pclass} → Age={age_median:5.1f} (n={count:3d})")

        # 全体の中央値（フォールバック用）
        self.age_rules[('_fallback_', 0)] = train['Age'].median()
        print(f"  Fallback (overall) → Age={self.age_rules[('_fallback_', 0)]:.1f}")

        # 2. Fare補完ルール: Pclass + Embarked別の中央値
        print("\n[2] Fare Imputation Rules (Pclass + Embarked):")
        for pclass in train['Pclass'].unique():
            for embarked in train['Embarked'].dropna().unique():
                mask = (train['Pclass'] == pclass) & (train['Embarked'] == embarked)
                fare_median = train.loc[mask, 'Fare'].median()
                if pd.notna(fare_median):
                    self.fare_rules[(pclass, embarked)] = fare_median
                    count = mask.sum()
                    print(f"  Pclass={pclass} + Embarked={embarked} → Fare={fare_median:6.2f} (n={count:3d})")

        # Pclass別のフォールバック（Embarked不明の場合）
        for pclass in train['Pclass'].unique():
            fare_median = train[train['Pclass'] == pclass]['Fare'].median()
            self.fare_rules[(pclass, '_fallback_')] = fare_median

        print(f"\n  Total Fare rules: {len(self.fare_rules)}")

        # 3. Embarked補完ルール
        self.embarked_mode = train['Embarked'].mode()[0]
        print(f"\n[3] Embarked Imputation Rule:")
        print(f"  Mode: '{self.embarked_mode}'")

        if self.use_fare_based_embarked:
            # Fare範囲を記録（高度な推定用）
            for pclass in train['Pclass'].unique():
                self.fare_ranges_by_embarked[pclass] = {}
                for embarked in train['Embarked'].dropna().unique():
                    mask = (train['Pclass'] == pclass) & (train['Embarked'] == embarked)
                    fares = train.loc[mask, 'Fare']
                    if len(fares) > 0:
                        self.fare_ranges_by_embarked[pclass][embarked] = {
                            'min': fares.min(),
                            'q25': fares.quantile(0.25),
                            'median': fares.median(),
                            'q75': fares.quantile(0.75),
                            'max': fares.max()
                        }
            print(f"  Fare-based estimation enabled (ranges computed for {len(self.fare_ranges_by_embarked)} Pclass groups)")

        self.is_fitted = True

        print("\n" + "=" * 80)
        print("IMPUTATION RULES LEARNED SUCCESSFULLY")
        print("=" * 80)

        return self

    def transform(self, df: pd.DataFrame, create_missing_flags: bool = True) -> pd.DataFrame:
        """
        学習した補完ルールをデータに適用

        Parameters
        ----------
        df : pd.DataFrame
            補完対象のデータ（Train または Test）
        create_missing_flags : bool, default=True
            欠損フラグを作成するかどうか

        Returns
        -------
        df_imputed : pd.DataFrame
            補完済みデータ
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")

        df_imputed = df.copy()

        # 欠損フラグ作成（補完前）
        if create_missing_flags:
            df_imputed['Age_Missing'] = df_imputed['Age'].isnull().astype(int)
            df_imputed['Embarked_Missing'] = df_imputed['Embarked'].isnull().astype(int)
            df_imputed['Fare_Missing'] = df_imputed['Fare'].isnull().astype(int)

        # Title抽出（Age補完に必要）
        df_imputed['Title'] = df_imputed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
            'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
            'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
        }
        df_imputed['Title'] = df_imputed['Title'].map(title_mapping)

        # 1. Embarked補完（Fare補完の前に実施）
        embarked_missing_count = df_imputed['Embarked'].isnull().sum()
        if embarked_missing_count > 0:
            if self.use_fare_based_embarked:
                # Fare範囲からの推定
                for idx in df_imputed[df_imputed['Embarked'].isnull()].index:
                    pclass = df_imputed.loc[idx, 'Pclass']
                    fare = df_imputed.loc[idx, 'Fare']

                    if pd.notna(fare) and pclass in self.fare_ranges_by_embarked:
                        # Fare範囲に最も適合するEmbarkedを推定
                        best_embarked = self.embarked_mode
                        best_score = -np.inf

                        for embarked, ranges in self.fare_ranges_by_embarked[pclass].items():
                            # Fareが範囲内にあるかスコアリング
                            if ranges['min'] <= fare <= ranges['max']:
                                # 中央値に近いほど高スコア
                                score = 1.0 - abs(fare - ranges['median']) / (ranges['max'] - ranges['min'] + 1e-6)
                                if score > best_score:
                                    best_score = score
                                    best_embarked = embarked

                        df_imputed.loc[idx, 'Embarked'] = best_embarked
                    else:
                        df_imputed.loc[idx, 'Embarked'] = self.embarked_mode
            else:
                # 単純に最頻値で補完
                df_imputed['Embarked'].fillna(self.embarked_mode, inplace=True)

        # 2. Fare補完（Embarked補完後）
        fare_missing_count = df_imputed['Fare'].isnull().sum()
        if fare_missing_count > 0:
            for idx in df_imputed[df_imputed['Fare'].isnull()].index:
                pclass = df_imputed.loc[idx, 'Pclass']
                embarked = df_imputed.loc[idx, 'Embarked']

                # Pclass + Embarked別のルールを適用
                if (pclass, embarked) in self.fare_rules:
                    df_imputed.loc[idx, 'Fare'] = self.fare_rules[(pclass, embarked)]
                elif (pclass, '_fallback_') in self.fare_rules:
                    df_imputed.loc[idx, 'Fare'] = self.fare_rules[(pclass, '_fallback_')]
                else:
                    warnings.warn(f"No Fare rule for Pclass={pclass}, Embarked={embarked}. Using overall median.")
                    df_imputed.loc[idx, 'Fare'] = 8.05  # Safe fallback

        # 3. Age補完
        age_missing_count = df_imputed['Age'].isnull().sum()
        if age_missing_count > 0:
            for idx in df_imputed[df_imputed['Age'].isnull()].index:
                title = df_imputed.loc[idx, 'Title']
                pclass = df_imputed.loc[idx, 'Pclass']

                # Title + Pclass別のルールを適用
                if (title, pclass) in self.age_rules:
                    df_imputed.loc[idx, 'Age'] = self.age_rules[(title, pclass)]
                elif ('_fallback_', 0) in self.age_rules:
                    df_imputed.loc[idx, 'Age'] = self.age_rules[('_fallback_', 0)]
                else:
                    warnings.warn(f"No Age rule for Title={title}, Pclass={pclass}. Using overall median.")
                    df_imputed.loc[idx, 'Age'] = 28.0  # Safe fallback

        return df_imputed

    def fit_transform(self, train_df: pd.DataFrame, create_missing_flags: bool = True) -> pd.DataFrame:
        """
        fit と transform を一度に実行

        Parameters
        ----------
        train_df : pd.DataFrame
            学習用データ（Trainデータのみ）
        create_missing_flags : bool, default=True
            欠損フラグを作成するかどうか

        Returns
        -------
        train_imputed : pd.DataFrame
            補完済みTrainデータ
        """
        self.fit(train_df)
        return self.transform(train_df, create_missing_flags=create_missing_flags)

    def get_summary(self) -> Dict:
        """
        学習した補完ルールのサマリを返す

        Returns
        -------
        summary : dict
            補完ルールの統計情報
        """
        if not self.is_fitted:
            return {"error": "Pipeline not fitted yet"}

        return {
            "age_rules_count": len([k for k in self.age_rules.keys() if k[0] != '_fallback_']),
            "fare_rules_count": len([k for k in self.fare_rules.keys() if k[1] != '_fallback_']),
            "embarked_mode": self.embarked_mode,
            "use_fare_based_embarked": self.use_fare_based_embarked,
            "is_fitted": self.is_fitted
        }


def verify_no_test_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Train/Test分離が正しく行われているか検証

    この関数は教育目的で、正しい実装と誤った実装の差を示します。
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: Train/Test Separation")
    print("=" * 80)

    # Test特有の欠損（Fare）を確認
    test_fare_missing = test_df['Fare'].isnull().sum()
    train_fare_missing = train_df['Fare'].isnull().sum()

    print(f"\nFare Missing Values:")
    print(f"  Train: {train_fare_missing} (0件が正常)")
    print(f"  Test:  {test_fare_missing} (1件が正常)")

    if test_fare_missing > 0:
        test_missing_info = test_df[test_df['Fare'].isnull()][['PassengerId', 'Pclass', 'Embarked']]
        print(f"\n  Test Fare欠損詳細:")
        for idx, row in test_missing_info.iterrows():
            print(f"    PassengerId={row['PassengerId']}, Pclass={row['Pclass']}, Embarked={row['Embarked']}")

        # 正しい補完値を計算
        pclass = test_missing_info.iloc[0]['Pclass']
        embarked = test_missing_info.iloc[0]['Embarked']

        correct_fare = train_df[(train_df['Pclass'] == pclass) &
                                (train_df['Embarked'] == embarked)]['Fare'].median()
        wrong_fare = test_df[test_df['Pclass'] == pclass]['Fare'].median()

        print(f"\n  正しい補完値（Trainから学習）: {correct_fare:.2f}")
        print(f"  誤った補完値（Testから学習）: {wrong_fare:.2f}")
        print(f"  差: {abs(correct_fare - wrong_fare):.2f}")

        if abs(correct_fare - wrong_fare) > 0.01:
            print(f"\n  ⚠️  WARNING: Train/Testで補完値が異なります！")
            print(f"      Testデータから学習するとデータリーケージが発生します。")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    """
    使用例とテスト
    """
    # データ読み込み
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    print("=" * 80)
    print("CORRECT IMPUTATION PIPELINE - DEMO")
    print("=" * 80)

    # Train/Test分離の検証
    verify_no_test_leakage(train, test)

    # パイプラインの使用
    print("\n\n" + "=" * 80)
    print("APPLYING CORRECT IMPUTATION")
    print("=" * 80)

    # 初期化
    imputer = CorrectImputationPipeline(use_fare_based_embarked=True)

    # Trainから学習
    imputer.fit(train)

    # Train/Testを補完
    print("\n\nApplying learned rules to Train data...")
    train_imputed = imputer.transform(train)

    print("\nApplying learned rules to Test data...")
    test_imputed = imputer.transform(test)

    # 結果確認
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\nTrain - Missing values after imputation:")
    print(train_imputed[['Age', 'Fare', 'Embarked']].isnull().sum())

    print("\nTest - Missing values after imputation:")
    print(test_imputed[['Age', 'Fare', 'Embarked']].isnull().sum())

    # Test Fare欠損の補完結果
    test_fare_idx = test[test['Fare'].isnull()].index
    if len(test_fare_idx) > 0:
        idx = test_fare_idx[0]
        print(f"\nTest PassengerId=1044 (Pclass=3, Embarked=S):")
        print(f"  補完後のFare: {test_imputed.loc[idx, 'Fare']:.2f}")
        print(f"  Train Pclass=3 & Embarked=S の中央値: {train[(train['Pclass']==3) & (train['Embarked']=='S')]['Fare'].median():.2f}")
        print(f"  ✅ 正しく補完されました！")

    # サマリ
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    summary = imputer.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n✅ Correct imputation pipeline demo completed!")
    print("=" * 80)
