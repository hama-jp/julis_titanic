"""
Ticket情報のパターン分析

Ticket列から抽出可能な特徴量を検討する:
1. Ticketプレフィックス（文字部分）
2. Ticket番号（数値部分）
3. Ticketの長さ
4. 同じTicketを持つグループサイズ
5. プレフィックスのカテゴリ
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
train = pd.read_csv('/home/user/julis_titanic/data/raw/train.csv')
test = pd.read_csv('/home/user/julis_titanic/data/raw/test.csv')

print("=" * 80)
print("Ticket情報のパターン分析")
print("=" * 80)

# 全データ結合（パターン分析用）
train['Dataset'] = 'train'
test['Dataset'] = 'test'
test['Survived'] = -1  # ダミー
combined = pd.concat([train, test], axis=0, ignore_index=True)

print(f"\nTotal records: {len(combined)}")
print(f"Train: {len(train)}, Test: {len(test)}")

# ===== 1. Ticketの基本統計 =====
print("\n" + "=" * 80)
print("1. Ticketの基本統計")
print("=" * 80)

print(f"Unique tickets: {combined['Ticket'].nunique()}")
print(f"Total passengers: {len(combined)}")
print(f"Tickets shared by multiple passengers: {(combined.groupby('Ticket').size() > 1).sum()}")

print("\nTicket examples:")
print(combined['Ticket'].head(20).tolist())

# ===== 2. Ticketのパターン抽出 =====
print("\n" + "=" * 80)
print("2. Ticketのパターン抽出")
print("=" * 80)

def extract_ticket_prefix(ticket):
    """チケットのプレフィックス（文字部分）を抽出"""
    ticket = str(ticket).strip()
    # 数字のみの場合
    if ticket.replace('.', '').isdigit():
        return 'NUMERIC'
    # 文字部分を抽出
    match = re.match(r'^([A-Za-z./\s]+)', ticket)
    if match:
        prefix = match.group(1).strip()
        # スラッシュやドットを正規化
        prefix = re.sub(r'[./\s]+', '_', prefix)
        prefix = prefix.strip('_')
        return prefix.upper()
    return 'OTHER'

def extract_ticket_number(ticket):
    """チケットの数値部分を抽出"""
    ticket = str(ticket).strip()
    # 数値部分を探す
    match = re.search(r'(\d+)', ticket)
    if match:
        return int(match.group(1))
    return 0

combined['Ticket_Prefix'] = combined['Ticket'].apply(extract_ticket_prefix)
combined['Ticket_Number'] = combined['Ticket'].apply(extract_ticket_number)
combined['Ticket_Length'] = combined['Ticket'].astype(str).str.len()

print("\nTicket Prefix 分布（上位20）:")
prefix_counts = combined['Ticket_Prefix'].value_counts()
print(prefix_counts.head(20))

print("\nTicket Prefix 種類数:", combined['Ticket_Prefix'].nunique())

# ===== 3. Ticketグループサイズ =====
print("\n" + "=" * 80)
print("3. Ticketグループサイズ分析")
print("=" * 80)

ticket_counts = combined.groupby('Ticket').size()
combined['Ticket_GroupSize'] = combined['Ticket'].map(ticket_counts)

print("\nTicket Group Size 分布:")
print(combined['Ticket_GroupSize'].value_counts().sort_index())

print("\n最大グループサイズのチケット:")
max_group = combined[combined['Ticket_GroupSize'] == combined['Ticket_GroupSize'].max()]
print(max_group[['PassengerId', 'Name', 'Ticket', 'Ticket_GroupSize', 'Pclass', 'Fare']].head(10))

# ===== 4. Ticket特徴量と生存率の関係 =====
print("\n" + "=" * 80)
print("4. Ticket特徴量と生存率の関係（Train データ）")
print("=" * 80)

train_data = combined[combined['Dataset'] == 'train'].copy()

# Prefix別の生存率
print("\nTicket Prefix別の生存率（上位15）:")
prefix_survival = train_data.groupby('Ticket_Prefix').agg({
    'Survived': ['mean', 'count']
}).round(4)
prefix_survival.columns = ['Survival_Rate', 'Count']
prefix_survival = prefix_survival[prefix_survival['Count'] >= 5].sort_values('Survival_Rate', ascending=False)
print(prefix_survival.head(15))

# Group Size別の生存率
print("\nTicket Group Size別の生存率:")
groupsize_survival = train_data.groupby('Ticket_GroupSize').agg({
    'Survived': ['mean', 'count']
}).round(4)
groupsize_survival.columns = ['Survival_Rate', 'Count']
print(groupsize_survival)

# ===== 5. Ticket特徴量とPclass/Fareの関係 =====
print("\n" + "=" * 80)
print("5. Ticket特徴量とPclass/Fareの関係")
print("=" * 80)

print("\nTicket Prefix別の平均Fare（上位15）:")
prefix_fare = train_data.groupby('Ticket_Prefix').agg({
    'Fare': ['mean', 'median', 'count'],
    'Pclass': 'mean'
}).round(4)
prefix_fare.columns = ['Fare_Mean', 'Fare_Median', 'Count', 'Pclass_Mean']
prefix_fare = prefix_fare[prefix_fare['Count'] >= 5].sort_values('Fare_Mean', ascending=False)
print(prefix_fare.head(15))

# ===== 6. Prefixのカテゴリ化提案 =====
print("\n" + "=" * 80)
print("6. Prefix カテゴリ化提案")
print("=" * 80)

# 高生存率のPrefix
high_survival_prefixes = prefix_survival[prefix_survival['Survival_Rate'] > 0.6].index.tolist()
print(f"\n高生存率 Prefix (>60%): {high_survival_prefixes}")

# 低生存率のPrefix
low_survival_prefixes = prefix_survival[prefix_survival['Survival_Rate'] < 0.3].index.tolist()
print(f"低生存率 Prefix (<30%): {low_survival_prefixes}")

# 高運賃のPrefix
high_fare_prefixes = prefix_fare[prefix_fare['Fare_Mean'] > 50].index.tolist()
print(f"\n高運賃 Prefix (>50): {high_fare_prefixes}")

# ===== 7. 同じTicketの乗客の統計 =====
print("\n" + "=" * 80)
print("7. 同じTicketグループ内の統計")
print("=" * 80)

# 同じTicketグループ内の生存者数
ticket_group_stats = train_data.groupby('Ticket').agg({
    'Survived': ['sum', 'count', 'mean'],
    'Pclass': 'first',
    'Fare': 'first'
})
ticket_group_stats.columns = ['Survivors_In_Group', 'Group_Size', 'Group_Survival_Rate', 'Pclass', 'Fare']
ticket_group_stats['All_Survived'] = (ticket_group_stats['Survivors_In_Group'] == ticket_group_stats['Group_Size']).astype(int)
ticket_group_stats['None_Survived'] = (ticket_group_stats['Survivors_In_Group'] == 0).astype(int)

print("\nグループ全員生存:")
print(f"Count: {ticket_group_stats['All_Survived'].sum()}")
print(ticket_group_stats[ticket_group_stats['All_Survived'] == 1].head(10))

print("\nグループ全員死亡:")
print(f"Count: {ticket_group_stats['None_Survived'].sum()}")
print(ticket_group_stats[ticket_group_stats['None_Survived'] == 1].head(10))

# ===== 8. 具体的な特徴量候補のまとめ =====
print("\n" + "=" * 80)
print("8. 特徴量候補のまとめ")
print("=" * 80)

feature_proposals = """
【推奨する特徴量】

1. Ticket_GroupSize (チケット共有人数)
   - 生存率との相関あり
   - グループサイズ2-4で高い生存率

2. Ticket_Prefix (カテゴリカル)
   - 主要なPrefix（PC, C_A_SOTON, など）をOne-hot化
   - または高/中/低生存率グループに分類

3. Ticket_Prefix_Category (3カテゴリ)
   - HIGH: 高生存率Prefix
   - LOW: 低生存率Prefix
   - MEDIUM: その他

4. Ticket_Is_Numeric (0/1)
   - 数字のみのチケットかどうか
   - Pclass/Fareと相関の可能性

5. Ticket_Length
   - チケット文字列の長さ
   - 補助的な特徴量

6. Ticket_Fare_Per_Person
   - Fare / Ticket_GroupSize
   - 実質的な個人運賃

7. Ticket_Group_Survival_Rate (注意: リーク防止必要)
   - 同じチケットグループの生存率
   - CVでリークしないように実装要注意
"""

print(feature_proposals)

# ===== 9. 相関分析 =====
print("\n" + "=" * 80)
print("9. Ticket特徴量の相関分析")
print("=" * 80)

train_data['Ticket_Is_Numeric'] = (train_data['Ticket_Prefix'] == 'NUMERIC').astype(int)

correlation_features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                        'Ticket_GroupSize', 'Ticket_Length', 'Ticket_Is_Numeric']

# Sexをエンコード
train_corr = train_data.copy()
train_corr['Sex'] = train_corr['Sex'].map({'male': 0, 'female': 1})

corr_matrix = train_corr[correlation_features].corr()
print("\nSurvivedとの相関:")
print(corr_matrix['Survived'].sort_values(ascending=False))

# ===== 10. 保存 =====
print("\n" + "=" * 80)
print("10. 分析結果の保存")
print("=" * 80)

# 統計情報を保存
analysis_results = {
    'prefix_survival': prefix_survival.to_dict(),
    'groupsize_survival': groupsize_survival.to_dict(),
    'prefix_fare': prefix_fare.to_dict(),
    'high_survival_prefixes': high_survival_prefixes,
    'low_survival_prefixes': low_survival_prefixes,
    'high_fare_prefixes': high_fare_prefixes
}

import json
with open('/home/user/julis_titanic/experiments/results/ticket_analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2, default=str)

print("\n分析結果を保存しました: experiments/results/ticket_analysis_results.json")

# サンプルデータを確認
print("\n" + "=" * 80)
print("サンプルデータ（Ticket特徴量付き）")
print("=" * 80)
print(combined[['PassengerId', 'Ticket', 'Ticket_Prefix', 'Ticket_Number',
                'Ticket_GroupSize', 'Ticket_Length', 'Pclass', 'Fare', 'Survived']].head(20))

print("\n分析完了！")
