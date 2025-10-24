#!/usr/bin/env python3
"""
本番用の欠損値補完パイプライン（Production-Ready）

実験結果と実用性のバランスを考慮した最適な補完戦略：
- Age: Title + Pclass別の中央値（実験で最良）
- Fare: Pclass + Embarked別の中央値（testデータの欠損に対応）
- Embarked: Fare + Pclassから推定（論理的に妥当）
- Cabin: Has_Cabinフラグ化（欠損率78%を考慮）

Experimental Evidence:
- 全4組み合わせを比較評価
- CV Scores: 既存ベスト 0.81592, 新パイプライン 0.80807
- 実用性: testデータの欠損パターンに完全対応
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional


class ProductionImputer:
    """
    本番用欠損値補完クラス

    既存のベストプラクティスと新しい高度な補完方法を統合
    """

    def __init__(self, verbose: bool = True):
        """
        Parameters:
        -----------
        verbose : bool
            詳細なログを出力するかどうか
        """
        self.verbose = verbose

        # Age補完用（Title + Pclass別）
        self.age_imputation_dict = {}
        self.overall_age_median = None

        # Fare補完用（Pclass + Embarked別）
        self.fare_imputation_dict = {}

        # Embarked補完用（Fare範囲ベース）
        self.embarked_mode = None
        self.embarked_fare_ranges = {}

        # Title mapping
        self.title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Rev': 'Rare', 'Dr': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
            'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
            'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
        }

    def _extract_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """Titleを抽出してマッピング"""
        df = df.copy()
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].map(self.title_mapping)
        return df

    def fit(self, train_df: pd.DataFrame) -> 'ProductionImputer':
        """
        訓練データから補完パラメータを学習

        Parameters:
        -----------
        train_df : pd.DataFrame
            訓練データ

        Returns:
        --------
        self
        """
        if self.verbose:
            print("="*80)
            print("PRODUCTION IMPUTATION PIPELINE - LEARNING")
            print("="*80)

        train_df = self._extract_title(train_df)

        # 1. Age補完値の学習（Title + Pclass別の中央値）
        if self.verbose:
            print("\n1. Learning Age imputation values (Title + Pclass)...")

        for title in train_df['Title'].dropna().unique():
            for pclass in train_df['Pclass'].unique():
                mask = (train_df['Title'] == title) & (train_df['Pclass'] == pclass)
                age_median = train_df.loc[mask, 'Age'].median()
                if pd.notna(age_median):
                    self.age_imputation_dict[(title, pclass)] = age_median
                    if self.verbose:
                        print(f"   Title={title:8s}, Pclass={pclass}: median_age={age_median:.1f}")

        # 全体の中央値（フォールバック）
        self.overall_age_median = train_df['Age'].median()
        if self.verbose:
            print(f"   Overall median age (fallback): {self.overall_age_median:.1f}")

        # 2. Fare補完値の学習（Pclass + Embarked別の中央値）
        if self.verbose:
            print("\n2. Learning Fare imputation values (Pclass + Embarked)...")

        for pclass in train_df['Pclass'].unique():
            for embarked in train_df['Embarked'].dropna().unique():
                mask = (train_df['Pclass'] == pclass) & (train_df['Embarked'] == embarked)
                fare_median = train_df.loc[mask, 'Fare'].median()
                self.fare_imputation_dict[(pclass, embarked)] = fare_median
                if self.verbose:
                    print(f"   Pclass={pclass}, Embarked={embarked}: median_fare={fare_median:.2f}")

        # Pclass別（フォールバック）
        for pclass in train_df['Pclass'].unique():
            fare_median = train_df[train_df['Pclass'] == pclass]['Fare'].median()
            self.fare_imputation_dict[(pclass, None)] = fare_median

        # 3. Embarked補完値の学習（Fare範囲ベース）
        if self.verbose:
            print("\n3. Learning Embarked imputation (Fare-based estimation)...")

        self.embarked_mode = train_df['Embarked'].mode()[0]
        if self.verbose:
            print(f"   Embarked mode (fallback): {self.embarked_mode}")

        # Pclass + Embarked別のFare範囲
        for pclass in train_df['Pclass'].unique():
            pclass_data = train_df[train_df['Pclass'] == pclass]
            for embarked in pclass_data['Embarked'].dropna().unique():
                embarked_data = pclass_data[pclass_data['Embarked'] == embarked]
                fare_25 = embarked_data['Fare'].quantile(0.25)
                fare_75 = embarked_data['Fare'].quantile(0.75)
                fare_median = embarked_data['Fare'].median()
                self.embarked_fare_ranges[(pclass, embarked)] = {
                    'q25': fare_25,
                    'q75': fare_75,
                    'median': fare_median
                }
                if self.verbose:
                    print(f"   Pclass={pclass}, Embarked={embarked}: "
                          f"Fare range [{fare_25:.2f} - {fare_75:.2f}], median={fare_median:.2f}")

        if self.verbose:
            print("\n" + "="*80)
            print("LEARNING COMPLETE")
            print("="*80)

        return self

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

        best_match = None
        best_score = float('inf')

        for (pc, emb), ranges in self.embarked_fare_ranges.items():
            if pc == pclass:
                # Fareが範囲内にある場合のスコアを計算
                if ranges['q25'] <= fare <= ranges['q75']:
                    score = 0  # 範囲内
                else:
                    # 範囲外の場合、中央値との距離
                    score = abs(fare - ranges['median'])

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
        df = self._extract_title(df)

        data_type = "TRAIN" if is_train else "TEST"

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"IMPUTING MISSING VALUES - {data_type} DATA")
            print(f"{'='*80}")

        # 補完前の欠損数を記録
        age_missing_before = df['Age'].isnull().sum()
        fare_missing_before = df['Fare'].isnull().sum() if 'Fare' in df.columns else 0
        embarked_missing_before = df['Embarked'].isnull().sum() if 'Embarked' in df.columns else 0

        # 1. Embarked補完（Fareベース推定）
        if 'Embarked' in df.columns and embarked_missing_before > 0:
            if self.verbose:
                print(f"\n1. Embarked imputation ({embarked_missing_before} missing)...")

            for idx in df[df['Embarked'].isnull()].index:
                pclass = df.loc[idx, 'Pclass']
                fare = df.loc[idx, 'Fare'] if 'Fare' in df.columns else None
                imputed_embarked = self._impute_embarked_from_fare(pclass, fare)
                df.loc[idx, 'Embarked'] = imputed_embarked

                if self.verbose:
                    fare_str = f"{fare:.2f}" if pd.notna(fare) else "NaN"
                    print(f"   Row {idx}: Pclass={pclass}, Fare={fare_str} "
                          f"→ Embarked={imputed_embarked}")

        # 2. Fare補完（Pclass + Embarked別の中央値）
        if 'Fare' in df.columns and fare_missing_before > 0:
            if self.verbose:
                print(f"\n2. Fare imputation ({fare_missing_before} missing)...")

            for idx in df[df['Fare'].isnull()].index:
                pclass = df.loc[idx, 'Pclass']
                embarked = df.loc[idx, 'Embarked'] if 'Embarked' in df.columns else None

                # Pclass + Embarked の組み合わせで補完
                if pd.notna(embarked) and (pclass, embarked) in self.fare_imputation_dict:
                    imputed_fare = self.fare_imputation_dict[(pclass, embarked)]
                else:
                    # フォールバック: Pclassのみ
                    imputed_fare = self.fare_imputation_dict[(pclass, None)]

                df.loc[idx, 'Fare'] = imputed_fare

                if self.verbose:
                    print(f"   Row {idx}: Pclass={pclass}, Embarked={embarked} "
                          f"→ Fare={imputed_fare:.2f}")

        # 3. Age補完（Title + Pclass別の中央値）
        if age_missing_before > 0:
            if self.verbose:
                print(f"\n3. Age imputation ({age_missing_before} missing)...")
                print(f"   Using Title + Pclass based median values...")

            for idx in df[df['Age'].isnull()].index:
                title = df.loc[idx, 'Title']
                pclass = df.loc[idx, 'Pclass']

                # Title + Pclass の組み合わせで補完
                if (title, pclass) in self.age_imputation_dict:
                    imputed_age = self.age_imputation_dict[(title, pclass)]
                else:
                    # フォールバック: 全体の中央値
                    imputed_age = self.overall_age_median

                df.loc[idx, 'Age'] = imputed_age

            if self.verbose:
                print(f"   Successfully imputed {age_missing_before} Age values")

        # 4. Cabin処理（Has_Cabinフラグ化）
        if 'Cabin' in df.columns:
            if self.verbose:
                print(f"\n4. Cabin feature engineering...")

            df['Has_Cabin'] = df['Cabin'].notna().astype(int)
            cabin_count = df['Has_Cabin'].sum()
            cabin_missing = len(df) - cabin_count

            if self.verbose:
                print(f"   Created Has_Cabin: {cabin_count} have cabin, {cabin_missing} don't")

        # 最終確認
        if self.verbose:
            print(f"\n{'-'*80}")
            print(f"Final missing values check for {data_type}:")
            remaining_missing = df[['Age', 'Fare', 'Embarked']].isnull().sum()
            remaining_missing = remaining_missing[remaining_missing > 0]
            if len(remaining_missing) > 0:
                print(remaining_missing)
            else:
                print("   ✅ No missing values in Age, Fare, Embarked!")
            print(f"{'-'*80}")

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
    メイン処理: デモンストレーション
    """
    print("="*80)
    print("PRODUCTION IMPUTATION PIPELINE - DEMO")
    print("="*80)
    print("\nOptimal Strategy (based on experiments):")
    print("  - Age:      Title + Pclass median (best CV score)")
    print("  - Fare:     Pclass + Embarked median (handles test missing)")
    print("  - Embarked: Fare-based estimation (logically sound)")
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
    print(train[['Age', 'Fare', 'Embarked', 'Cabin']].isnull().sum())
    print("\nTest missing values:")
    print(test[['Age', 'Fare', 'Cabin']].isnull().sum())

    # 欠損値補完
    imputer = ProductionImputer(verbose=True)
    train_imputed = imputer.fit_transform(train)
    test_imputed = imputer.transform(test, is_train=False)

    # 補完後の確認
    print("\n" + "="*80)
    print("AFTER IMPUTATION")
    print("="*80)
    print("\nTrain missing values:")
    print(train_imputed[['Age', 'Fare', 'Embarked']].isnull().sum())

    print("\nTest missing values:")
    print(test_imputed[['Age', 'Fare']].isnull().sum())

    # 補完後のデータを保存
    print("\n" + "="*80)
    print("Saving imputed data...")
    train_imputed.to_csv('data/processed/train_production_imputed.csv', index=False)
    test_imputed.to_csv('data/processed/test_production_imputed.csv', index=False)
    print("  ✅ Saved to data/processed/train_production_imputed.csv")
    print("  ✅ Saved to data/processed/test_production_imputed.csv")

    print("\n" + "="*80)
    print("PRODUCTION PIPELINE COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
