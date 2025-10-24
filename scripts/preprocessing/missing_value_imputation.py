#!/usr/bin/env python3
"""
欠損値補完パイプライン
train/test共通で使用できる欠損値補完ロジックを提供
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any


class MissingValueImputer:
    """
    欠損値補完を行うクラス
    train/testで一貫した補完戦略を適用
    """

    def __init__(self):
        self.fare_imputation_dict = {}  # Pclass + Embarked別のFare中央値
        self.embarked_mode = None  # Embarkedの最頻値
        self.age_imputation_dict = {}  # Pclass + Sex別のAge中央値

    def fit(self, train_df: pd.DataFrame) -> 'MissingValueImputer':
        """
        訓練データから補完値を学習

        Parameters:
        -----------
        train_df : pd.DataFrame
            訓練データ

        Returns:
        --------
        self
        """

        # 1. Fare補完値の計算（Pclass + Embarked別の中央値）
        print("Learning Fare imputation values...")
        for pclass in train_df['Pclass'].unique():
            for embarked in train_df['Embarked'].dropna().unique():
                mask = (train_df['Pclass'] == pclass) & (train_df['Embarked'] == embarked)
                fare_median = train_df.loc[mask, 'Fare'].median()
                self.fare_imputation_dict[(pclass, embarked)] = fare_median
                print(f"  Pclass={pclass}, Embarked={embarked}: median_fare={fare_median:.2f}")

        # フォールバック: Pclass別の中央値も保存
        for pclass in train_df['Pclass'].unique():
            mask = train_df['Pclass'] == pclass
            fare_median = train_df.loc[mask, 'Fare'].median()
            self.fare_imputation_dict[(pclass, None)] = fare_median

        # 2. Embarked補完値の計算（最頻値）
        print("\nLearning Embarked imputation value...")
        self.embarked_mode = train_df['Embarked'].mode()[0]
        print(f"  Embarked mode: {self.embarked_mode}")

        # より高度な補完: FareとPclassから推定
        # Embarked欠損のある乗客のFareとPclassを見て、最も近いEmbarkedを推定
        self.embarked_fare_based = self._learn_embarked_from_fare(train_df)

        # 3. Age補完値の計算（Pclass + Sex別の中央値）
        print("\nLearning Age imputation values...")
        for pclass in train_df['Pclass'].unique():
            for sex in train_df['Sex'].unique():
                mask = (train_df['Pclass'] == pclass) & (train_df['Sex'] == sex)
                age_median = train_df.loc[mask, 'Age'].median()
                self.age_imputation_dict[(pclass, sex)] = age_median
                print(f"  Pclass={pclass}, Sex={sex}: median_age={age_median:.1f}")

        # フォールバック: 全体の中央値
        self.overall_age_median = train_df['Age'].median()

        return self

    def _learn_embarked_from_fare(self, train_df: pd.DataFrame) -> Dict[Tuple[int, float], str]:
        """
        FareとPclassからEmbarkedを推定するための辞書を作成

        Parameters:
        -----------
        train_df : pd.DataFrame
            訓練データ

        Returns:
        --------
        dict : (Pclass, Fare)に基づく推定Embarked
        """
        embarked_fare_dict = {}

        # Pclass別のEmbarked毎のFare範囲を記録
        for pclass in train_df['Pclass'].unique():
            pclass_data = train_df[train_df['Pclass'] == pclass]
            for embarked in pclass_data['Embarked'].dropna().unique():
                embarked_data = pclass_data[pclass_data['Embarked'] == embarked]
                fare_25 = embarked_data['Fare'].quantile(0.25)
                fare_75 = embarked_data['Fare'].quantile(0.75)
                embarked_fare_dict[(pclass, embarked)] = (fare_25, fare_75)

        return embarked_fare_dict

    def _impute_embarked_from_fare(self, pclass: int, fare: float) -> str:
        """
        FareとPclassからEmbarkedを推定

        Parameters:
        -----------
        pclass : int
            客室クラス
        fare : float
            運賃

        Returns:
        --------
        str : 推定されたEmbarked
        """
        if pd.isna(fare):
            return self.embarked_mode

        # Pclass別の各Embarkedの料金範囲と比較
        best_match = None
        best_score = float('inf')

        for (pc, emb), (fare_25, fare_75) in self.embarked_fare_based.items():
            if pc == pclass:
                # Fareが範囲内にある場合のスコアを計算
                if fare_25 <= fare <= fare_75:
                    score = 0  # 範囲内
                else:
                    # 範囲外の場合、最も近い境界までの距離
                    score = min(abs(fare - fare_25), abs(fare - fare_75))

                if score < best_score:
                    best_score = score
                    best_match = emb

        return best_match if best_match else self.embarked_mode

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        欠損値を補完

        Parameters:
        -----------
        df : pd.DataFrame
            補完対象のデータ
        is_train : bool
            訓練データかどうか（ログ出力用）

        Returns:
        --------
        pd.DataFrame : 補完後のデータ
        """
        df = df.copy()
        data_type = "TRAIN" if is_train else "TEST"

        print(f"\n{'='*60}")
        print(f"Imputing missing values for {data_type} data")
        print(f"{'='*60}")

        # 1. Embarked の補完（trainのみ）
        if 'Embarked' in df.columns:
            missing_embarked = df['Embarked'].isnull().sum()
            if missing_embarked > 0:
                print(f"\nImputing {missing_embarked} missing Embarked values...")

                # FareとPclassから推定
                for idx in df[df['Embarked'].isnull()].index:
                    pclass = df.loc[idx, 'Pclass']
                    fare = df.loc[idx, 'Fare']

                    # Fareから推定
                    imputed_embarked = self._impute_embarked_from_fare(pclass, fare)
                    df.loc[idx, 'Embarked'] = imputed_embarked

                    print(f"  Row {idx}: Pclass={pclass}, Fare={fare:.2f} -> Embarked={imputed_embarked}")

        # 2. Fare の補完（testのみ）
        if 'Fare' in df.columns:
            missing_fare = df['Fare'].isnull().sum()
            if missing_fare > 0:
                print(f"\nImputing {missing_fare} missing Fare values...")

                for idx in df[df['Fare'].isnull()].index:
                    pclass = df.loc[idx, 'Pclass']
                    embarked = df.loc[idx, 'Embarked']

                    # Pclass + Embarked の組み合わせで補完
                    if (pclass, embarked) in self.fare_imputation_dict:
                        imputed_fare = self.fare_imputation_dict[(pclass, embarked)]
                    else:
                        # フォールバック: Pclassのみで補完
                        imputed_fare = self.fare_imputation_dict[(pclass, None)]

                    df.loc[idx, 'Fare'] = imputed_fare

                    print(f"  Row {idx}: Pclass={pclass}, Embarked={embarked} -> Fare={imputed_fare:.2f}")

        # 3. Age の補完
        if 'Age' in df.columns:
            missing_age = df['Age'].isnull().sum()
            if missing_age > 0:
                print(f"\nImputing {missing_age} missing Age values...")
                print("  Using Pclass + Sex based median values...")

                for idx in df[df['Age'].isnull()].index:
                    pclass = df.loc[idx, 'Pclass']
                    sex = df.loc[idx, 'Sex']

                    # Pclass + Sex の組み合わせで補完
                    if (pclass, sex) in self.age_imputation_dict:
                        imputed_age = self.age_imputation_dict[(pclass, sex)]
                    else:
                        # フォールバック: 全体の中央値
                        imputed_age = self.overall_age_median

                    df.loc[idx, 'Age'] = imputed_age

                print(f"  Successfully imputed {missing_age} Age values")

        # 4. Cabin の処理（欠損フラグ化）
        if 'Cabin' in df.columns:
            print(f"\nProcessing Cabin column...")
            df['Has_Cabin'] = df['Cabin'].notna().astype(int)
            missing_cabin = df['Cabin'].isnull().sum()
            print(f"  Created Has_Cabin feature: {df['Has_Cabin'].sum()} passengers have cabin info, "
                  f"{missing_cabin} don't")

        # 最終確認
        print(f"\n{'-'*60}")
        print(f"Final missing values check for {data_type}:")
        remaining_missing = df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        if len(remaining_missing) > 0:
            print(remaining_missing)
        else:
            print("  No missing values remaining!")

        return df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        訓練データに対してfitとtransformを同時実行

        Parameters:
        -----------
        train_df : pd.DataFrame
            訓練データ

        Returns:
        --------
        pd.DataFrame : 補完後の訓練データ
        """
        self.fit(train_df)
        return self.transform(train_df, is_train=True)


def main():
    """
    メイン処理: train/testデータの欠損値補完をデモンストレーション
    """
    print("="*80)
    print("MISSING VALUE IMPUTATION PIPELINE")
    print("="*80)

    # データ読み込み
    print("\nLoading data...")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 補完前の欠損値確認
    print("\n" + "="*80)
    print("BEFORE IMPUTATION")
    print("="*80)
    print("\nTrain missing values:")
    print(train.isnull().sum()[train.isnull().sum() > 0])
    print("\nTest missing values:")
    print(test.isnull().sum()[test.isnull().sum() > 0])

    # 欠損値補完
    imputer = MissingValueImputer()
    train_imputed = imputer.fit_transform(train)
    test_imputed = imputer.transform(test, is_train=False)

    # 補完後の確認
    print("\n" + "="*80)
    print("AFTER IMPUTATION")
    print("="*80)
    print("\nTrain missing values:")
    train_missing_after = train_imputed.isnull().sum()[train_imputed.isnull().sum() > 0]
    if len(train_missing_after) > 0:
        print(train_missing_after)
    else:
        print("  None (except Cabin which is converted to Has_Cabin)")

    print("\nTest missing values:")
    test_missing_after = test_imputed.isnull().sum()[test_imputed.isnull().sum() > 0]
    if len(test_missing_after) > 0:
        print(test_missing_after)
    else:
        print("  None (except Cabin which is converted to Has_Cabin)")

    # 補完後のデータを保存（オプション）
    print("\n" + "="*80)
    print("Saving imputed data...")
    train_imputed.to_csv('data/processed/train_imputed.csv', index=False)
    test_imputed.to_csv('data/processed/test_imputed.csv', index=False)
    print("  Saved to data/processed/train_imputed.csv")
    print("  Saved to data/processed/test_imputed.csv")

    print("\n" + "="*80)
    print("IMPUTATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
