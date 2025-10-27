"""
Age補完方法の分析 - PclassとTitleを組み合わせた補完
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


def main():
    print("=" * 80)
    print("Age補完方法の分析")
    print("=" * 80)

    train, test = load_data()

    # 全データを結合
    all_data = pd.concat([train, test], sort=False, ignore_index=True)

    # Titleを抽出
    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Mme': 'Mrs', 'Don': 'Rare', 'Dona': 'Rare',
        'Lady': 'Rare', 'Countess': 'Rare', 'Jonkheer': 'Rare',
        'Sir': 'Rare', 'Capt': 'Rare', 'Ms': 'Miss'
    }
    all_data['Title'] = all_data['Title'].map(title_mapping).fillna('Rare')

    # 欠損値の状況を確認
    print("\n[1] Age欠損値の状況")
    print(f"全体: {all_data['Age'].isna().sum()} / {len(all_data)} ({all_data['Age'].isna().sum()/len(all_data)*100:.1f}%)")
    print(f"Train: {train['Age'].isna().sum()} / {len(train)} ({train['Age'].isna().sum()/len(train)*100:.1f}%)")
    print(f"Test: {test['Age'].isna().sum()} / {len(test)} ({test['Age'].isna().sum()/len(test)*100:.1f}%)")

    # Title別のAge統計
    print("\n[2] Title別のAge統計（欠損値を除く）")
    age_by_title = all_data.groupby('Title')['Age'].agg(['count', 'mean', 'median', 'std'])
    print(age_by_title)

    # Pclass別のAge統計
    print("\n[3] Pclass別のAge統計（欠損値を除く）")
    age_by_pclass = all_data.groupby('Pclass')['Age'].agg(['count', 'mean', 'median', 'std'])
    print(age_by_pclass)

    # Pclass x Title別のAge統計
    print("\n[4] Pclass x Title別のAge統計（欠損値を除く）")
    age_by_pclass_title = all_data.groupby(['Pclass', 'Title'])['Age'].agg(['count', 'mean', 'median', 'std'])
    print(age_by_pclass_title)

    # 欠損値の分布
    print("\n[5] Age欠損値のPclass x Title分布")
    missing_age = all_data[all_data['Age'].isna()]
    missing_dist = missing_age.groupby(['Pclass', 'Title']).size().reset_index(name='count')
    print(missing_dist)

    # 補完戦略の比較
    print("\n" + "=" * 80)
    print("補完戦略の比較")
    print("=" * 80)

    # 戦略1: Title別の中央値のみ
    print("\n戦略1: Title別の中央値のみ（現在の方法）")
    age_title_median = all_data.groupby('Title')['Age'].median()
    print(age_title_median)

    # 戦略2: Pclass x Title別の中央値（フォールバック付き）
    print("\n戦略2: Pclass x Title別の中央値（推奨）")
    age_pclass_title_median = all_data.groupby(['Pclass', 'Title'])['Age'].median()
    print("\nPclass x Title別の中央値:")
    for (pclass, title), age in age_pclass_title_median.items():
        count = len(all_data[(all_data['Pclass'] == pclass) & (all_data['Title'] == title) & all_data['Age'].notna()])
        print(f"  Pclass {pclass}, {title:10s}: {age:5.1f} (n={count})")

    # フォールバックが必要なケースを確認
    print("\n補完値が存在しないケース（フォールバックが必要）:")
    for pclass in [1, 2, 3]:
        for title in ['Mr', 'Miss', 'Mrs', 'Master', 'Rare']:
            if (pclass, title) not in age_pclass_title_median.index:
                # Title別のフォールバック
                fallback = age_title_median.get(title, all_data['Age'].median())
                print(f"  Pclass {pclass}, {title:10s}: フォールバック → {fallback:.1f}")

    # 戦略3: Pclass x Title別の平均値
    print("\n戦略3: Pclass x Title別の平均値")
    age_pclass_title_mean = all_data.groupby(['Pclass', 'Title'])['Age'].mean()
    print("\nPclass x Title別の平均値:")
    for (pclass, title), age in age_pclass_title_mean.items():
        print(f"  Pclass {pclass}, {title:10s}: {age:5.1f}")

    print("\n" + "=" * 80)
    print("推奨補完戦略")
    print("=" * 80)
    print("\n1. Pclass x Title別の中央値を使用")
    print("2. 該当グループにデータがない場合 → Title別の中央値にフォールバック")
    print("3. Titleにもデータがない場合 → 全体の中央値にフォールバック")
    print("\nこれにより、より精密な年齢推定が可能になります。")


if __name__ == '__main__':
    main()
