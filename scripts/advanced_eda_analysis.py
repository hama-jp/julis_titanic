"""
Advanced EDA Analysis based on ML_MODEL_IMPROVEMENT_REPORT.md Section 4.3
実施内容:
1. クラス不均衡の詳細分析 (Proposal 1)
2. 相互作用の探索 (Proposal 2)
3. 時系列・空間的パターンの分析 (Proposal 3)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# グラフ設定
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style('whitegrid')

# データ読み込み
print("="*80)
print("データ読み込み中...")
print("="*80)
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print(f"\nTrain shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ========================================================================
# 提案1: クラス不均衡の詳細分析
# ========================================================================
print("\n" + "="*80)
print("提案1: クラス不均衡の詳細分析")
print("="*80)

# 全体の生存率
overall_survival = train['Survived'].mean()
print(f"\n全体の生存率: {overall_survival:.2%} (不均衡度: {1 - overall_survival:.2%} vs {overall_survival:.2%})")

# サブグループ別の不均衡分析
print("\n【サブグループ別の生存率と不均衡】")
print("-" * 80)

subgroups = train.groupby(['Pclass', 'Sex'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count'),
    ('生存者数', 'sum')
]).round(4)

# 死亡者数を追加
subgroups['死亡者数'] = subgroups['サンプル数'] - subgroups['生存者数']

# 不均衡度を計算 (生存率と全体生存率の乖離)
subgroups['不均衡度'] = abs(subgroups['生存率'] - overall_survival)

# ソートして表示
subgroups_sorted = subgroups.sort_values('不均衡度', ascending=False)
print(subgroups_sorted)

# 最も不均衡なグループ
print(f"\n【最も不均衡なグループ】")
most_imbalanced = subgroups_sorted.index[0]
print(f"グループ: Pclass={most_imbalanced[0]}, Sex={most_imbalanced[1]}")
print(f"生存率: {subgroups_sorted.iloc[0]['生存率']:.2%}")
print(f"サンプル数: {int(subgroups_sorted.iloc[0]['サンプル数'])}")

# 年齢層別の分析も追加
train_age = train.copy()
train_age['AgeGroup'] = pd.cut(train_age['Age'], bins=[0, 12, 18, 40, 100],
                                labels=['Child', 'Teen', 'Adult', 'Senior'])

age_analysis = train_age.groupby(['AgeGroup', 'Sex'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)

print("\n【年齢層×性別の生存率】")
print("-" * 80)
print(age_analysis)

# ========================================================================
# 提案2: 相互作用の探索
# ========================================================================
print("\n" + "="*80)
print("提案2: 相互作用の探索")
print("="*80)

# 2.1 Sex × Pclass の相互作用
print("\n【Sex × Pclass の相互作用】")
print("-" * 80)

train['Sex_Pclass'] = train['Sex'].astype(str) + '_Class' + train['Pclass'].astype(str)
sex_pclass_interaction = train.groupby('Sex_Pclass')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count'),
    ('生存者数', 'sum')
]).round(4)

sex_pclass_interaction = sex_pclass_interaction.sort_values('生存率', ascending=False)
print(sex_pclass_interaction)

# 相互作用の強さを評価
# 単純な主効果との比較
sex_effect = train.groupby('Sex')['Survived'].mean()
pclass_effect = train.groupby('Pclass')['Survived'].mean()

print(f"\n【主効果】")
print(f"Sex効果:")
for sex, rate in sex_effect.items():
    print(f"  {sex}: {rate:.2%}")

print(f"\nPclass効果:")
for pclass, rate in pclass_effect.items():
    print(f"  Class {pclass}: {rate:.2%}")

# 2.2 Age × Sex の相互作用
print("\n【Age × Sex の相互作用（年齢層別）】")
print("-" * 80)

# 年齢層をより細かく分析
train_age['AgeGroup_Fine'] = pd.cut(train_age['Age'],
                                     bins=[0, 5, 12, 18, 30, 50, 100],
                                     labels=['Infant(0-5)', 'Child(6-12)', 'Teen(13-18)',
                                            'Young(19-30)', 'Middle(31-50)', 'Senior(51+)'])

age_sex_interaction = train_age.groupby(['AgeGroup_Fine', 'Sex'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)

print(age_sex_interaction)

# 子供優先ルールの検証
print("\n【子供優先ルールの検証】")
print("-" * 80)
train_age['IsChild'] = (train_age['Age'] < 16).astype(int)
child_analysis = train_age.groupby(['IsChild', 'Sex'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)
print(child_analysis)

# 2.3 Fare × Pclass の相互作用
print("\n【Fare × Pclass の相互作用】")
print("-" * 80)

train_fare = train.copy()
train_fare['FareLevel'] = pd.qcut(train_fare['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')

fare_pclass_interaction = train_fare.groupby(['Pclass', 'FareLevel'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)

print(fare_pclass_interaction)

# ========================================================================
# 提案3: 時系列・空間的パターンの分析
# ========================================================================
print("\n" + "="*80)
print("提案3: 時系列・空間的パターンの分析")
print("="*80)

# 3.1 Ticket番号からグループ旅行を検出
print("\n【Ticket番号からグループ旅行を検出】")
print("-" * 80)

train_ticket = train.copy()
train_ticket['TicketGroup'] = train_ticket['Ticket'].map(train_ticket['Ticket'].value_counts())

# グループサイズ別の生存率
ticket_group_analysis = train_ticket.groupby('TicketGroup')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count'),
    ('グループ数', lambda x: len(x))
]).round(4)

# TicketGroupが多い上位10件を表示
print("グループサイズ別の生存率:")
print(ticket_group_analysis.sort_index())

# 単独 vs グループの比較
train_ticket['IsTicketGroup'] = (train_ticket['TicketGroup'] > 1).astype(int)
solo_vs_group = train_ticket.groupby('IsTicketGroup')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)
solo_vs_group.index = ['単独チケット', 'グループチケット']
print("\n単独 vs グループチケット:")
print(solo_vs_group)

# 3.2 Cabin情報から甲板レベルを抽出
print("\n【Cabin情報から甲板レベル（Deck）を抽出】")
print("-" * 80)

train_cabin = train.copy()
test_cabin = test.copy()

# Cabinの最初の文字（甲板レベル）を抽出
train_cabin['Deck'] = train_cabin['Cabin'].str[0]
test_cabin['Deck'] = test_cabin['Cabin'].str[0]

# 甲板別の生存率
deck_analysis = train_cabin.groupby('Deck')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count'),
    ('生存者数', 'sum')
]).round(4)

deck_analysis = deck_analysis.sort_values('生存率', ascending=False)
print(deck_analysis)

# 欠損値の生存率
cabin_missing_rate = train_cabin.groupby(train_cabin['Cabin'].isnull())['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)
cabin_missing_rate.index = ['Cabin情報あり', 'Cabin情報なし']
print("\nCabin情報の有無:")
print(cabin_missing_rate)

# 3.3 同じCabinの人数（共同利用）
print("\n【同じCabinの共同利用分析】")
print("-" * 80)

train_cabin['CabinShared'] = train_cabin.groupby('Cabin')['PassengerId'].transform('count')

# Cabin情報があるデータのみで分析
train_cabin_valid = train_cabin[train_cabin['Cabin'].notna()].copy()

cabin_shared_analysis = train_cabin_valid.groupby('CabinShared')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)

print("Cabin共有人数別の生存率:")
print(cabin_shared_analysis.sort_index())

# 単独 vs 共有の比較
train_cabin_valid['IsCabinShared'] = (train_cabin_valid['CabinShared'] > 1).astype(int)
shared_comparison = train_cabin_valid.groupby('IsCabinShared')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)
shared_comparison.index = ['Cabin単独利用', 'Cabin共同利用']
print("\n単独 vs 共同利用:")
print(shared_comparison)

# ========================================================================
# 追加分析: Embarked（乗船港）の詳細分析
# ========================================================================
print("\n" + "="*80)
print("追加分析: Embarked（乗船港）の詳細分析")
print("="*80)

embarked_analysis = train.groupby('Embarked')['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count'),
    ('生存者数', 'sum')
]).round(4)

print(embarked_analysis)

# Embarked × Pclass の相互作用
embarked_pclass = train.groupby(['Embarked', 'Pclass'])['Survived'].agg([
    ('生存率', 'mean'),
    ('サンプル数', 'count')
]).round(4)

print("\nEmbarked × Pclass:")
print(embarked_pclass)

# ========================================================================
# サマリー: 重要な発見
# ========================================================================
print("\n" + "="*80)
print("サマリー: 重要な発見")
print("="*80)

# 最も生存率が高いグループ
best_group = sex_pclass_interaction.index[0]
best_rate = sex_pclass_interaction.iloc[0]['生存率']
print(f"\n1. 最も生存率が高いグループ: {best_group} ({best_rate:.2%})")

# 最も生存率が低いグループ
worst_group = sex_pclass_interaction.index[-1]
worst_rate = sex_pclass_interaction.iloc[-1]['生存率']
print(f"2. 最も生存率が低いグループ: {worst_group} ({worst_rate:.2%})")

# グループ旅行の効果
solo_rate = solo_vs_group.loc['単独チケット', '生存率']
group_rate = solo_vs_group.loc['グループチケット', '生存率']
print(f"\n3. チケットグループの効果:")
print(f"   単独チケット: {solo_rate:.2%}")
print(f"   グループチケット: {group_rate:.2%}")
print(f"   差異: {abs(group_rate - solo_rate):.2%}")

# Deckの効果（最も生存率が高い甲板）
if len(deck_analysis) > 0:
    best_deck = deck_analysis.index[0]
    best_deck_rate = deck_analysis.iloc[0]['生存率']
    print(f"\n4. 最も安全な甲板: Deck {best_deck} ({best_deck_rate:.2%})")

# Cabin情報の価値
cabin_info_diff = cabin_missing_rate.loc['Cabin情報あり', '生存率'] - cabin_missing_rate.loc['Cabin情報なし', '生存率']
print(f"\n5. Cabin情報の価値: {cabin_info_diff:.2%} (Cabin情報ありの方が生存率が高い)")

# 子供優先ルール
child_female_rate = child_analysis.loc[(1, 'female'), '生存率'] if (1, 'female') in child_analysis.index else 0
child_male_rate = child_analysis.loc[(1, 'male'), '生存率'] if (1, 'male') in child_analysis.index else 0
adult_female_rate = child_analysis.loc[(0, 'female'), '生存率'] if (0, 'female') in child_analysis.index else 0
adult_male_rate = child_analysis.loc[(0, 'male'), '生存率'] if (0, 'male') in child_analysis.index else 0

print(f"\n6. 子供優先ルールの検証:")
print(f"   女性・子供: {child_female_rate:.2%}")
print(f"   男性・子供: {child_male_rate:.2%}")
print(f"   女性・大人: {adult_female_rate:.2%}")
print(f"   男性・大人: {adult_male_rate:.2%}")

print("\n" + "="*80)
print("分析完了")
print("="*80)
