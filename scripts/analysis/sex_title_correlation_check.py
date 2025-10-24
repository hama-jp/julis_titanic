"""
SexとTitleの相関を確認
なぜSexの特徴量重要度が低いのか？
"""

import pandas as pd
import numpy as np

train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')

print("=" * 80)
print("SexとTitleの関係分析")
print("=" * 80)

# Titleを抽出
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

print("\n1. TitleとSexのクロス集計:")
print("-" * 80)
crosstab = pd.crosstab(train['Title'], train['Sex'], margins=True)
print(crosstab)

print("\n2. 主要Titleの性別分布:")
print("-" * 80)
title_counts = train['Title'].value_counts()
for title in title_counts.head(10).index:
    sex_dist = train[train['Title'] == title]['Sex'].value_counts()
    print(f"\n{title} (n={title_counts[title]}):")
    print(f"  {sex_dist.to_dict()}")

print("\n3. SexとTitleの相関（数値化後）:")
print("-" * 80)

# エンコード
train_encoded = train.copy()
train_encoded['Sex_Numeric'] = train_encoded['Sex'].map({'male': 0, 'female': 1})

title_mapping = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
    'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Mme': 2,
    'Don': 4, 'Dona': 4, 'Lady': 4, 'Countess': 4, 'Jonkheer': 4, 'Sir': 4,
    'Capt': 4, 'Ms': 1
}
train_encoded['Title_Numeric'] = train_encoded['Title'].map(title_mapping).fillna(4)

correlation = train_encoded[['Sex_Numeric', 'Title_Numeric']].corr()
print(correlation)

print("\n4. TitleとSexの情報重複:")
print("-" * 80)
print("\nTitle → Sex の予測精度:")

# Titleから性別を完全に予測できるか
title_to_sex = {}
for title in train['Title'].unique():
    most_common_sex = train[train['Title'] == title]['Sex'].mode()[0]
    title_to_sex[title] = most_common_sex

train['Sex_Predicted'] = train['Title'].map(title_to_sex)
accuracy = (train['Sex'] == train['Sex_Predicted']).mean()
print(f"精度: {accuracy:.2%}")

# 予測が外れるケース
errors = train[train['Sex'] != train['Sex_Predicted']]
print(f"\n予測ミス: {len(errors)} / {len(train)}")
if len(errors) > 0:
    print("\n予測ミスの例:")
    print(errors[['Name', 'Title', 'Sex', 'Sex_Predicted']].head(10))

print("\n5. 生存率への影響:")
print("-" * 80)
print("\nSex別の生存率:")
sex_survival = train.groupby('Sex')['Survived'].agg(['mean', 'count'])
print(sex_survival)

print("\nTitle別の生存率（主要のみ）:")
title_survival = train.groupby('Title')['Survived'].agg(['mean', 'count'])
title_survival = title_survival[title_survival['count'] >= 10].sort_values('mean', ascending=False)
print(title_survival)

print("\n" + "=" * 80)
print("結論:")
print("=" * 80)
print("""
【なぜSexの重要度が低いのか】

1. **Titleが性別情報を含んでいる**:
   - Mr → 男性（100%）
   - Miss, Mrs, Ms → 女性（100%）
   - Master → 男児（100%）

2. **TitleはSex以上の情報を持つ**:
   - 性別だけでなく、既婚/未婚、年齢層も含む
   - Mrs（既婚女性）とMiss（未婚女性）の生存率は異なる可能性

3. **特徴量重要度の測定**:
   - ツリーモデルは最初にTitleで分割
   - Titleで分割後、Sexの情報は既に使われている
   - → Sexは追加の情報ゲインをもたらさない

4. **Sexは間接的に重要**:
   - Titleを通じて間接的に使われている
   - Title（重要度0.509）の大部分はSex情報
""")
