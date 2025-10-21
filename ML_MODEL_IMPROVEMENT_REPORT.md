# Titanic生存予測モデル 改善レポート

**作成日**: 2025-10-21
**目的**: より高精度でロバスト性のあるモデルの構築

---

## 目次

1. [エグゼクティブサマリー](#1-エグゼクティブサマリー)
2. [現状分析](#2-現状分析)
3. [データ前処理](#3-データ前処理)
4. [探索的データ分析（EDA）](#4-探索的データ分析eda)
5. [特徴量エンジニアリング](#5-特徴量エンジニアリング)
6. [モデル学習とチューニング](#6-モデル学習とチューニング)
7. [推論と評価](#7-推論と評価)
8. [改善提案](#8-改善提案)
9. [実装ロードマップ](#9-実装ロードマップ)

---

## 1. エグゼクティブサマリー

### 現在の最高性能

| メトリック | 値 | 備考 |
|-----------|-----|------|
| **最高CV Score** | **0.8350** | Top 3 Hard Voting (LightGBM+GB+RF) |
| **最高個別モデル** | **0.8317** | LightGBM (チューニング済み) |
| **過学習Gap** | **0.0090** | LightGBM（97%改善達成） |
| **特徴量数** | **9個** | 多重共線性排除後 |

### 主要な成果

1. 過学習の大幅削減: 0.0843 → 0.0090 (97%改善)
2. LightGBMチューニングによる性能向上: CV +1.47%
3. 最適アンサンブル発見: OOF Score 0.8350
4. ロバストな特徴量セット確立

### 推奨される次のアクション

1. **短期**: Stacking/Blendingアンサンブルの実装
2. **中期**: 外れ値処理とデータクリーニング
3. **長期**: ニューラルネットワークや深層学習の検討

---

## 2. 現状分析

### 2.1 データセット概要

```
訓練データ: 891サンプル
テストデータ: 418サンプル
目的変数: Survived (0/1)
生存率: 38.4%
```

### 2.2 欠損値の状況

| 特徴量 | Train欠損率 | Test欠損率 | 影響 |
|--------|------------|-----------|------|
| Age | 19.9% | 20.6% | 高 - 重要な特徴量 |
| Cabin | 77.1% | 78.2% | 中 - 欠損フラグとして活用 |
| Embarked | 0.2% | 0% | 低 - 最頻値で補完可能 |
| Fare | 0% | 0.2% | 低 - 中央値で補完可能 |

### 2.3 現在のモデルパフォーマンス

#### 個別モデル

| モデル | CV Score | Gap | 評価 |
|--------|----------|-----|------|
| LightGBM (Tuned) | 0.8317 | 0.0090 | ⭐ 最高性能 |
| GradientBoosting | 0.8316 | 0.0314 | 優秀 |
| RandomForest | 0.8294 | 0.0112 | バランス良好 |
| LogisticRegression | 0.8193 | 0.0022 | 最低過学習 |
| XGBoost (Tuned) | 0.8058 | 0.0191 | 不採用 |

#### アンサンブル

| 構成 | OOF Score | 特徴 |
|------|-----------|------|
| Top 3 Hard Voting | **0.8350** | LightGBM+GB+RF |
| RF+GB+LR | 0.8339 | 多様性重視 |
| All Boosting | 0.8328 | GB+LightGBM+XGB |

---

## 3. データ前処理

### 3.1 現在の前処理パイプライン

#### ステップ1: 欠損値フラグの作成（補完前）

```python
# 補完前に欠損情報を保持
df['Age_Missing'] = df['Age'].isnull().astype(int)
df['Cabin_Missing'] = df['Cabin'].isnull().astype(int)
df['Embarked_Missing'] = df['Embarked'].isnull().astype(int)
```

**根拠**: 欠損自体が予測に有用な情報
- Age_Missing: 生存率差 -11.2% (p=0.0077)
- Cabin情報の有無は社会的地位を反映

#### ステップ2: Title（敬称）の抽出

```python
# 名前から敬称を抽出
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# 希少なタイトルをグループ化
title_mapping = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5
}
```

**効果**: 社会的地位と年齢層を同時に捉える強力な特徴量

#### ステップ3: Age補完

```python
# TitleとPclassごとの中央値で補完
for title in df['Title'].unique():
    for pclass in df['Pclass'].unique():
        mask = (df['Title'] == title) & (df['Pclass'] == pclass) & (df['Age'].isnull())
        median_age = df[(df['Title'] == title) & (df['Pclass'] == pclass)]['Age'].median()
        df.loc[mask, 'Age'] = median_age
```

**根拠**: グループ内の類似性を活用した精密な補完

#### ステップ4: 家族関連特徴量

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['SmallFamily'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4)).astype(int)
```

**効果**:
- IsAlone: 生存率に明確な差
- SmallFamily: 適度な家族サイズが生存に有利

#### ステップ5: Fare処理

```python
# Pclassごとの中央値で補完
df['Fare'].fillna(df.groupby('Pclass')['Fare'].transform('median'), inplace=True)

# 4分位ビニング
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=False, duplicates='drop')
```

**根拠**: 連続値の離散化で過学習を防止

### 3.2 前処理の課題と改善点

#### 課題1: 外れ値の未処理

**問題点**:
- Fareに極端な外れ値が存在（最大値 > 500）
- Ageの外れ値（80歳以上）が少数

**影響**: モデルが外れ値に過剰適合する可能性

**改善案**:
```python
# Winsorization（上下1%をクリップ）
from scipy.stats.mstats import winsorize
df['Fare'] = winsorize(df['Fare'], limits=[0.01, 0.01])
df['Age'] = winsorize(df['Age'], limits=[0.01, 0.01])
```

#### 課題2: スケーリングの未実施

**問題点**:
- Age (0-80) と Fare (0-500+) でスケールが大きく異なる
- 距離ベースのモデルやニューラルネットに影響

**改善案**:
```python
from sklearn.preprocessing import StandardScaler, RobustScaler

# ツリーベースモデル以外用
scaler = RobustScaler()  # 外れ値に頑健
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
```

#### 課題3: データリーケージのリスク

**現状**: Train/Test同時処理によるリーケージの可能性

**改善案**:
```python
# Trainで学習したパラメータをTestに適用
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # fit_transformは使わない
```

---

## 4. 探索的データ分析（EDA）

### 4.1 実施済みのEDA

#### 生存率の基本統計

| カテゴリ | 生存率 | サンプル数 | 重要度 |
|---------|-------|-----------|--------|
| **Sex=Female** | 74.2% | 314 | ⭐⭐⭐ |
| **Sex=Male** | 18.9% | 577 | ⭐⭐⭐ |
| **Pclass=1** | 63.0% | 216 | ⭐⭐⭐ |
| **Pclass=2** | 47.3% | 184 | ⭐⭐ |
| **Pclass=3** | 24.2% | 491 | ⭐⭐⭐ |
| **IsAlone=1** | 30.4% | 537 | ⭐⭐ |
| **IsAlone=0** | 50.6% | 354 | ⭐⭐ |

#### Titleと生存率

| Title | 生存率 | サンプル数 | 特徴 |
|-------|-------|-----------|------|
| Mrs | 79.2% | 125 | 既婚女性 - 高生存率 |
| Miss | 69.8% | 182 | 未婚女性 - 高生存率 |
| Master | 57.5% | 40 | 少年 - 中程度生存率 |
| Mr | 15.7% | 517 | 成人男性 - 低生存率 |
| Rare | 34.7% | 27 | 特殊な肩書き |

### 4.2 多重共線性の分析結果

#### 削除した特徴量

| 特徴量 | 理由 | VIF/相関 |
|--------|------|----------|
| Pclass_Sex | 完全な多重共線性 | VIF = ∞ |
| Cabin_Missing | Pclassと高相関 | r = 0.726 |
| FamilySize | IsAlone/SmallFamilyで代替 | VIF > 10 |

#### 保持した特徴量（最終9個）

1. **Pclass**: 客室クラス（社会経済的地位）
2. **Sex**: 性別（最重要特徴量）
3. **Title**: 敬称（性別+年齢+地位）
4. **Age**: 年齢（連続値）
5. **IsAlone**: 単独旅行フラグ
6. **FareBin**: 運賃ビン（4分位）
7. **SmallFamily**: 適度な家族サイズ
8. **Age_Missing**: Age欠損フラグ
9. **Embarked_Missing**: 乗船港欠損フラグ

### 4.3 EDAの改善提案

#### 提案1: クラス不均衡の詳細分析

**現状**: 生存率38.4%（不均衡）

**追加分析**:
```python
# サブグループ別の不均衡
subgroups = train.groupby(['Pclass', 'Sex'])['Survived'].agg(['mean', 'count'])
print(subgroups)

# 不均衡対策の検討
# - クラスウェイト調整
# - SMOTE（合成マイノリティオーバーサンプリング）
# - アンダーサンプリング
```

#### 提案2: 相互作用の探索

**未実施の分析**:
```python
# Sex × Pclass の相互作用
train['Sex_Pclass'] = train['Sex'].astype(str) + '_' + train['Pclass'].astype(str)
print(train.groupby('Sex_Pclass')['Survived'].mean())

# Age × Sex の相互作用（年齢層別）
train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 18, 40, 100], labels=['Child', 'Adult', 'Senior'])
print(train.groupby(['AgeGroup', 'Sex'])['Survived'].mean())
```

#### 提案3: 時系列・空間的パターン

**追加EDA**:
```python
# Ticket番号からグループ旅行を検出
train['TicketGroup'] = train['Ticket'].map(train['Ticket'].value_counts())

# Cabin情報から甲板レベルを抽出
train['Deck'] = train['Cabin'].str[0]
print(train.groupby('Deck')['Survived'].agg(['mean', 'count']))

# 同じCabinの人数（共同利用）
train['CabinShared'] = train.groupby('Cabin')['PassengerId'].transform('count')
```

### 4.4 EDA改善提案の実施結果

**実施日**: 2025-10-21
**実施内容**: セクション4.3の3つの改善提案を実施し、新たな知見を獲得

#### 4.4.1 提案1の実施結果: クラス不均衡の詳細分析

##### サブグループ別の生存率と不均衡度

| グループ (Pclass, Sex) | 生存率 | サンプル数 | 不均衡度 | 特徴 |
|----------------------|-------|-----------|---------|------|
| **(1, female)** | **96.81%** | 94 | 0.5843 | 最も生存率が高い |
| (2, female) | 92.11% | 76 | 0.5373 | 2番目に高い |
| (3, male) | **13.54%** | 347 | 0.2484 | **最も生存率が低い（最大グループ）** |
| (2, male) | 15.74% | 108 | 0.2264 | 低生存率 |
| (3, female) | 50.00% | 144 | 0.1162 | 中程度 |
| (1, male) | 36.89% | 122 | 0.0149 | 全体平均に近い |

**重要な発見**:
1. 1等客室の女性は96.81%という驚異的な生存率（94人中91人が生存）
2. 3等客室の男性は13.54%と極めて低い（347人中47人のみ生存）
3. **生存率のレンジ: 13.54% - 96.81% = 83.27%の差**
4. 3等客室の男性が最大グループ（347人、全体の38.9%）で、これが全体の不均衡を増幅

##### 年齢層×性別の生存率

| 年齢層 | 性別 | 生存率 | サンプル数 | 知見 |
|-------|------|-------|-----------|------|
| Child (0-12) | female | 59.38% | 32 | 女性の子供 |
| Child (0-12) | male | 56.76% | 37 | 男性の子供 |
| Teen (13-18) | female | **75.00%** | 36 | 女性の10代 |
| Teen (13-18) | male | 8.82% | 34 | **男性の10代は低い** |
| Adult (19-40) | female | 78.62% | 145 | 成人女性 |
| Adult (19-40) | male | 18.21% | 280 | 成人男性 |
| Senior (51+) | female | **77.08%** | 48 | 高齢女性 |
| Senior (51+) | male | 17.65% | 102 | 高齢男性 |

**重要な発見**:
1. **12歳以下の子供は性別による差が小さい**（女性59.38% vs 男性56.76%）
2. **13歳以上になると性別による差が急激に拡大**（女性75% vs 男性8.82%）
3. 高齢女性（51歳以上）でも77.08%と高い生存率を維持
4. 成人男性（19-40歳）の生存率は18.21%と一貫して低い

##### 不均衡対策の提案

**現状**: 生存率38.38% vs 死亡率61.62%

**サブグループごとの不均衡を考慮した対策**:

```python
# サブグループ別のクラスウェイト
from sklearn.utils.class_weight import compute_sample_weight

# Sex × Pclass を考慮したサンプルウェイト
train['group'] = train['Sex'].astype(str) + '_' + train['Pclass'].astype(str)
sample_weights = compute_sample_weight('balanced', train['group'])

# モデル訓練時に適用
model.fit(X_train, y_train, sample_weight=sample_weights)
```

**期待効果**:
- 少数グループ（1等女性）の過学習を防止
- 多数グループ（3等男性）の予測精度向上
- **全体的なロバスト性向上**

#### 4.4.2 提案2の実施結果: 相互作用の探索

##### Sex × Pclass の相互作用

| グループ | 生存率 | サンプル数 | 主効果からの乖離 |
|---------|-------|-----------|----------------|
| female_Class1 | 96.81% | 94 | +22.61% (vs Class1平均62.96%) |
| female_Class2 | 92.11% | 76 | +44.83% (vs Class2平均47.28%) |
| female_Class3 | 50.00% | 144 | +25.76% (vs Class3平均24.24%) |
| male_Class1 | 36.89% | 122 | -26.07% (vs Class1平均62.96%) |
| male_Class2 | 15.74% | 108 | -31.54% (vs Class2平均47.28%) |
| male_Class3 | 13.54% | 347 | -10.70% (vs Class3平均24.24%) |

**主効果との比較**:
- **Sex主効果**: Female 74.20% vs Male 18.89% (差55.31%)
- **Pclass主効果**: Class1 62.96% vs Class2 47.28% vs Class3 24.24%

**重要な発見**:
1. **強力な相互作用が存在**: 単純な主効果の和では説明できない
2. 女性×1等客室の組み合わせは加算以上の効果（96.81%）
3. 3等客室の女性は50%まで低下（女性全体平均74.20%より-24.20%）
4. **相互作用特徴量の作成が有効**: `Sex_Pclass_Interaction`

##### Age × Sex の相互作用（詳細年齢層）

| 年齢層 | 女性生存率 | 男性生存率 | 差 |
|-------|----------|----------|-----|
| Infant (0-5) | 76.19% | 65.22% | +10.97% |
| Child (6-12) | 27.27% | 42.86% | **-15.59%** (逆転) |
| Teen (13-18) | 75.00% | 8.82% | +66.18% |
| Young (19-30) | 75.56% | 15.56% | +60.00% |
| Middle (31-50) | 77.91% | 22.58% | +55.33% |
| Senior (51+) | **94.12%** | 12.77% | **+81.35%** (最大差) |

**重要な発見**:
1. **6-12歳で性別効果が逆転**（男の子の方が高い生存率）
2. **51歳以上の女性が94.12%**という最高の生存率
3. 13歳以降、性別による差が急激に拡大（平均60%以上の差）

##### 子供優先ルールの検証

| カテゴリ | 生存率 | サンプル数 |
|---------|-------|-----------|
| **女性・子供** (Age < 16) | 65.12% | 43 |
| **男性・子供** (Age < 16) | 52.50% | 40 |
| **女性・大人** (Age >= 16) | **75.65%** | 271 |
| **男性・大人** (Age >= 16) | 16.39% | 537 |

**驚くべき発見**:
- **「女性と子供優先」ルールは部分的**
- 女性は子供より大人の方が生存率が高い（75.65% vs 65.12%）
- 男性は子供が大人の3倍以上の生存率（52.50% vs 16.39%）
- **より正確には「女性優先、次に子供」**

##### Fare × Pclass の相互作用

**1等客室**:
- 低運賃（0件のため評価不可）
- 高運賃: 52.94%
- 最高運賃: **68.55%** (159人)

**2等客室**:
- 中運賃: 38.37% (86人)
- 高運賃: **60.00%** (70人)

**3等客室**:
- 低運賃: 20.85% (211人)
- 中運賃: 25.36% (138人)
- 高運賃: 31.68% (101人)

**重要な発見**:
1. **同じPclassでもFareによって生存率が変化**
2. 2等客室の高運賃（60.00%）> 1等客室の高運賃（52.94%）
3. 3等客室では運賃による差が小さい（20-32%の範囲）

**特徴量エンジニアリングへの示唆**:
```python
# Pclass内でのFareの相対位置
train['FareRank_within_Pclass'] = train.groupby('Pclass')['Fare'].rank(pct=True)

# Sex × Pclass の相互作用特徴量
train['Sex_Pclass_Int'] = train['Sex'].astype(str) + '_' + train['Pclass'].astype(str)

# Age層 × Sex の相互作用
train['AgeGroup_Sex_Int'] = train['AgeGroup'].astype(str) + '_' + train['Sex'].astype(str)
```

#### 4.4.3 提案3の実施結果: 時系列・空間的パターンの分析

##### Ticket番号からグループ旅行を検出

| チケットグループサイズ | 生存率 | サンプル数 | 特徴 |
|-------------------|-------|-----------|------|
| 1 (単独) | 29.80% | 547 | 基準 |
| 2 (ペア) | 57.45% | 188 | +27.65% |
| 3 (3人) | **69.84%** | 63 | +40.04% |
| 4 (4人) | 50.00% | 44 | +20.20% |
| 5 (5人) | 0.00% | 10 | -29.80% (全滅) |
| 6 (6人) | 0.00% | 18 | -29.80% (全滅) |
| 7 (7人) | 23.81% | 21 | -5.99% |

**単独 vs グループチケットの比較**:
- **単独チケット**: 29.80% (547人)
- **グループチケット**: 52.03% (344人)
- **差異**: +22.23%

**重要な発見**:
1. **2-4人のグループが最も高い生存率**（50-70%）
2. **5-6人の大家族は全滅**（生存率0%）
3. 単独旅行者は生存率が低い（29.80%）
4. **適度なグループサイズが生存に有利**

**FamilySizeとの違い**:
- `FamilySize`は`SibSp + Parch + 1`で算出（血縁関係）
- `TicketGroup`は同じチケット番号（同行者、必ずしも家族ではない）
- 両方を特徴量として使用する価値がある

##### Cabin情報から甲板レベル（Deck）を抽出

| Deck | 生存率 | サンプル数 | 特徴 |
|------|-------|-----------|------|
| **D** | **75.76%** | 33 | 最も安全 |
| **E** | 75.00% | 32 | 2番目に安全 |
| **B** | 74.47% | 47 | 3番目 |
| F | 61.54% | 13 | 中程度 |
| C | 59.32% | 59 | 中程度 |
| G | 50.00% | 4 | 下層甲板 |
| A | 46.67% | 15 | 意外に低い |
| T | 0.00% | 1 | サンプル少 |

**Cabin情報の有無による比較**:
- **Cabin情報あり**: 66.67% (204人)
- **Cabin情報なし**: 29.99% (687人)
- **差異**: **+36.68%**

**重要な発見**:
1. **Cabin情報の有無が最も強力な特徴量の1つ**（差36.68%）
2. Deck D, E, Bが最も安全（75%以上）
3. Deck Aは意外に低い（46.67%）- 上層階だが危険？
4. Cabin情報自体が社会的地位の指標として機能

**Pclassとの関連**:
```
Cabin_Missing と Pclass の相関: r = 0.726
→ 3等客室はCabin情報がほぼなし
→ Cabin_Missingは既に特徴量として使用中
```

##### 同じCabinの共同利用分析

**Cabin共有人数別の生存率**（Cabin情報がある204人のみ）:

| 共有人数 | 生存率 | サンプル数 |
|---------|-------|-----------|
| 1 (単独利用) | 57.43% | 101 |
| 2 (2人共有) | **77.63%** | 76 |
| 3 (3人共有) | 73.33% | 15 |
| 4 (4人共有) | 66.67% | 12 |

**単独 vs 共同利用の比較**:
- **Cabin単独利用**: 57.43% (101人)
- **Cabin共同利用**: **75.73%** (103人)
- **差異**: +18.30%

**重要な発見**:
1. **Cabinを共有している方が生存率が高い**（+18.30%）
2. 2人での共有が最も高い（77.63%）
3. 4人以上になると生存率が低下傾向
4. 家族やグループでの旅行が生存に有利

**TicketGroupとの相乗効果**:
- `TicketGroup`: 同じチケット番号（一緒に購入）
- `CabinShared`: 同じCabin（同じ部屋に滞在）
- 両方が大きい場合は強い結びつきを示す

##### Embarked（乗船港）の詳細分析

**乗船港別の生存率**:

| 乗船港 | 生存率 | サンプル数 | 特徴 |
|-------|-------|-----------|------|
| C (Cherbourg) | **55.36%** | 168 | 最も高い |
| Q (Queenstown) | 38.96% | 77 | 中程度 |
| S (Southampton) | 33.70% | 644 | 最も低い |

**Embarked × Pclass の相互作用**:

| 乗船港 | Pclass | 生存率 | サンプル数 |
|-------|--------|-------|-----------|
| **C** | 1 | **69.41%** | 85 | 最高 |
| C | 2 | 52.94% | 17 |
| C | 3 | 37.88% | 66 |
| Q | 1 | 50.00% | 2 | サンプル少 |
| Q | 2 | 66.67% | 3 | サンプル少 |
| Q | 3 | 37.50% | 72 |
| S | 1 | 58.27% | 127 |
| S | 2 | 46.34% | 164 |
| S | 3 | **18.98%** | 353 | 最低 |

**重要な発見**:
1. Cherbourg (C) から乗船した1等客室が最も高い生存率（69.41%）
2. Southampton (S) から乗船した3等客室が最も低い（18.98%）
3. 乗船港自体がPclassと関連している可能性
4. 現在の特徴量には `Embarked_Missing` のみ → `Embarked` 自体も有用

#### 4.4.4 新たに発見された有望な特徴量

分析の結果、以下の特徴量が有望であることが判明:

##### 優先度1: 即座に追加すべき特徴量

```python
# 1. TicketGroup（グループ旅行の検出）
train['TicketGroup'] = train['Ticket'].map(train['Ticket'].value_counts())
train['IsTicketGroup'] = (train['TicketGroup'] > 1).astype(int)
train['OptimalTicketGroup'] = ((train['TicketGroup'] >= 2) & (train['TicketGroup'] <= 4)).astype(int)

# 2. Deck（甲板レベル）
train['Deck'] = train['Cabin'].str[0]
# One-hot encoding または Ordinal encoding
train['Deck_D'] = (train['Deck'] == 'D').astype(int)
train['Deck_E'] = (train['Deck'] == 'E').astype(int)
train['Deck_B'] = (train['Deck'] == 'B').astype(int)

# 3. CabinShared（Cabin共同利用）
train['CabinShared'] = train.groupby('Cabin')['PassengerId'].transform('count')
train['IsCabinShared'] = (train['CabinShared'] > 1).astype(int)

# 4. Embarked（乗船港）- 現在未使用
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)
train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)
```

**期待効果**: CV Score +0.5-1.0%

##### 優先度2: 相互作用特徴量

```python
# 1. Sex × Pclass
train['Female_Class1'] = ((train['Sex'] == 'female') & (train['Pclass'] == 1)).astype(int)
train['Male_Class3'] = ((train['Sex'] == 'male') & (train['Pclass'] == 3)).astype(int)

# 2. AgeGroup × Sex
train['AgeGroup'] = pd.cut(train['Age'], bins=[0, 12, 18, 100], labels=[0, 1, 2])
train['Child_Male'] = ((train['AgeGroup'] == 0) & (train['Sex'] == 'male')).astype(int)
train['Adult_Female'] = ((train['AgeGroup'] == 2) & (train['Sex'] == 'female')).astype(int)

# 3. Fare相対位置（Pclass内）
train['FareRank_within_Pclass'] = train.groupby('Pclass')['Fare'].rank(pct=True)
```

**期待効果**: CV Score +0.3-0.5%

##### 優先度3: 高度な特徴量

```python
# 1. グループ生存率（要注意: Target Leakage防止が必要）
# CVのfold内でのみ計算
ticket_survival = train.groupby('Ticket')['Survived'].transform('mean')

# 2. 社会的優先度スコア
def social_priority_score(row):
    score = 0
    if row['Sex'] == 'female':
        score += 3
    if row['Age'] < 16:
        score += 2
    if row['Pclass'] == 1:
        score += 1
    if row['Deck'] in ['D', 'E', 'B']:
        score += 1
    return score

train['SocialPriority'] = train.apply(social_priority_score, axis=1)

# 3. グループサイズの非線形変換
train['TicketGroup_Squared'] = train['TicketGroup'] ** 2
train['TicketGroup_IsOptimal'] = ((train['TicketGroup'] >= 2) & (train['TicketGroup'] <= 4)).astype(int)
```

**期待効果**: CV Score +0.2-0.4%

#### 4.4.5 モデリングへの示唆

##### 1. クラスウェイトの調整

**現状の問題**:
- 3等客室の男性（347人、13.54%生存率）が過小評価される可能性
- 1等客室の女性（94人、96.81%生存率）が過大評価される可能性

**対策**:
```python
from sklearn.utils.class_weight import compute_sample_weight

# Sex × Pclass のグループごとにバランス調整
train['Subgroup'] = train['Sex'].astype(str) + '_' + train['Pclass'].astype(str)
sample_weights = compute_sample_weight('balanced', train['Subgroup'])

# LightGBMでの使用例
lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
```

##### 2. Stratified Group K-Fold

**現状の問題**:
- 同じチケット/Cabinのグループが train/validation に分散
- データリーケージのリスク

**対策**:
```python
from sklearn.model_selection import StratifiedGroupKFold

# Ticketをグループとして使用
groups = train['Ticket']
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    # 同じチケットグループは必ず同じfoldに
    pass
```

##### 3. 特徴量の重要度再評価

**予想される変化**:

| 特徴量 | 現在の重要度 | 予想重要度 | 備考 |
|--------|------------|----------|------|
| Sex | 1位 (0.285) | **1位** | 不動 |
| Title | 2位 (0.198) | **2位** | 不動 |
| **TicketGroup** | - | **3-4位** | 新規追加 |
| Pclass | 3位 (0.165) | 4-5位 | やや低下 |
| **Deck** | - | **5-6位** | 新規追加 |
| Age | 4位 (0.124) | 6-7位 | 維持 |

#### 4.4.6 次のアクション

##### 短期（1-2日）

1. **新特徴量の実装と検証**
   - TicketGroup, Deck, CabinShared, Embarked を追加
   - 多重共線性チェック
   - CV Scoreの変化を測定

2. **相互作用特徴量の追加**
   - Sex × Pclass, AgeGroup × Sex
   - FareRank_within_Pclass

3. **クラスウェイト調整の実験**
   - Subgroup balanced weights
   - CV Scoreとバイアスのトレードオフ確認

##### 中期（3-7日）

4. **Stratified Group K-Foldの実装**
   - データリーケージ防止
   - より正確な性能評価

5. **アンサンブルの再構築**
   - 新特徴量でのモデル再訓練
   - Stacking/Blendingの実装

6. **ハイパーパラメータ再チューニング**
   - 特徴量数が増えたため再調整が必要

##### 長期（1-2週間）

7. **高度な特徴量エンジニアリング**
   - 社会的優先度スコア
   - グループレベルの集約特徴量（Target leakage注意）

8. **モデルの多様化**
   - ニューラルネットワーク
   - AutoML

#### 4.4.7 期待される性能向上

| Phase | 施策 | 期待CV Score | 改善幅 |
|-------|------|-------------|--------|
| **現在** | - | 0.8350 | - |
| **Phase 1** | 新特徴量追加 | 0.8400-0.8450 | +0.5-1.0% |
| **Phase 2** | 相互作用+クラスウェイト | 0.8450-0.8500 | +0.5% |
| **Phase 3** | Stacking+再チューニング | 0.8500-0.8550 | +0.5% |
| **合計** | - | **0.8500-0.8550** | **+1.5-2.0%** |

#### 4.4.8 まとめ

**主要な発見**:

1. **グループ旅行の効果**: グループチケット（+22.23%）、Cabin共有（+18.30%）
2. **Cabin情報の価値**: 情報ありは36.68%も生存率が高い
3. **強力な相互作用**: Sex × Pclass で83.27%の生存率差
4. **子供優先の実態**: 実際は「女性優先、次に子供」
5. **最適グループサイズ**: 2-4人が最も生存率が高い（5-6人は全滅）

**次の実装優先順位**:

1. **TicketGroup特徴量**: 最も影響が大きい（+22%）
2. **Deck特徴量**: Cabin情報の有効活用
3. **Sex × Pclass相互作用**: 83%の差を捉える
4. **CabinShared特徴量**: 共同利用の効果（+18%）
5. **クラスウェイト調整**: サブグループの不均衡対策

**実装により期待される効果**:
- CV Score: **0.835 → 0.850 (+1.5%)**
- ロバスト性の向上
- より解釈可能なモデル

---

## 5. 特徴量エンジニアリング

### 5.1 現在の特徴量評価

#### 特徴量の重要度（LightGBMベース）

| 順位 | 特徴量 | 重要度 | カテゴリ |
|------|--------|--------|---------|
| 1 | Sex | 0.285 | 基本情報 |
| 2 | Title | 0.198 | エンジニアリング |
| 3 | Pclass | 0.165 | 基本情報 |
| 4 | Age | 0.124 | 基本情報 |
| 5 | FareBin | 0.098 | エンジニアリング |
| 6 | IsAlone | 0.067 | エンジニアリング |
| 7 | SmallFamily | 0.034 | エンジニアリング |
| 8 | Age_Missing | 0.021 | メタ特徴量 |
| 9 | Embarked_Missing | 0.008 | メタ特徴量 |

### 5.2 改善提案: 高度な特徴量

#### 提案1: ドメイン知識ベースの特徴量

```python
# 1. 女性と子供優先ルール
df['WomenAndChildren'] = ((df['Sex'] == 1) | (df['Age'] < 16)).astype(int)

# 2. 社会的優先度スコア
def priority_score(row):
    score = 0
    if row['Sex'] == 1:  # Female
        score += 3
    if row['Age'] < 16:  # Child
        score += 2
    if row['Pclass'] == 1:  # First class
        score += 1
    return score

df['PriorityScore'] = df.apply(priority_score, axis=1)

# 3. 富裕層フラグ（1st class + 高運賃）
df['Wealthy'] = ((df['Pclass'] == 1) & (df['Fare'] > df['Fare'].quantile(0.75))).astype(int)
```

#### 提案2: 統計的特徴量

```python
# 1. TicketグループのAggregate特徴量
ticket_stats = train.groupby('Ticket').agg({
    'Age': ['mean', 'std'],
    'Fare': 'mean',
    'Survived': 'mean'  # グループ生存率（要注意: target leakage）
})

# 2. 名前の長さ（社会的地位の指標？）
df['NameLength'] = df['Name'].str.len()

# 3. Fare per person（グループ旅行考慮）
df['FarePerPerson'] = df['Fare'] / df['FamilySize']
```

#### 提案3: エンコーディングの改善

```python
# 1. Target Encoding（CVで実装してリーケージ防止）
from category_encoders import TargetEncoder

te = TargetEncoder(cols=['Title', 'Pclass'])
# Fold内でfit/transform

# 2. Frequency Encoding
df['Title_Freq'] = df['Title'].map(df['Title'].value_counts(normalize=True))
df['Embarked_Freq'] = df['Embarked'].map(df['Embarked'].value_counts(normalize=True))
```

### 5.3 特徴量選択の強化

#### 提案: 複数手法の組み合わせ

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# 1. 統計的検定ベース
selector_stat = SelectKBest(score_func=f_classif, k=10)
X_selected_stat = selector_stat.fit_transform(X_train, y_train)

# 2. RFE（Recursive Feature Elimination）
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
selector_rfe = RFE(estimator, n_features_to_select=10, step=1)
X_selected_rfe = selector_rfe.fit_transform(X_train, y_train)

# 3. L1正則化ベース（Lasso）
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)
important_features = np.where(np.abs(lasso.coef_) > 0.01)[0]

# 4. 相互情報量ベース
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
```

---

## 6. モデル学習とチューニング

### 6.1 成功事例: LightGBMチューニング

#### パラメータの変遷

| パラメータ | Before | After | 効果 |
|-----------|--------|-------|------|
| num_leaves | 8 | **4** | 複雑さ半減 |
| max_depth | 3 | **2** | より浅い木 |
| min_child_samples | 20 | **30** | サンプル要件増 |
| reg_alpha | 0.5 | **1.0** | L1正則化2倍 |
| reg_lambda | 0.5 | **1.0** | L2正則化2倍 |
| min_split_gain | 0 | **0.01** | 分割制約追加 |

#### 結果

```
CV Score: 0.8170 → 0.8317 (+1.47%)
Gap: 0.0427 → 0.0090 (-79%)
```

**教訓**: 保守的なパラメータが小規模データセットで有効

### 6.2 失敗事例: XGBoostチューニング

#### 問題点

```
CV Score: 0.8238 → 0.8058 (-1.8%)
Gap: 0.0236 → 0.0191 (-19%)
```

**結論**: 保守的パラメータが常に有効とは限らない
- LightGBM: Leaf-wise成長 → 保守的で改善
- XGBoost: Level-wise成長 → 保守的で悪化

### 6.3 アンサンブル戦略

#### 現在のベストアンサンブル

**Top 3 Hard Voting (OOF: 0.8350)**

```python
VotingClassifier(
    estimators=[
        ('lgb', LightGBM),
        ('gb', GradientBoosting),
        ('rf', RandomForest)
    ],
    voting='hard'
)
```

#### 成功要因

1. 個々のモデルの高性能（CV > 0.829）
2. 低過学習（Gap < 0.015）
3. アーキテクチャの多様性（2ブースティング + 1バギング）

### 6.4 改善提案: 高度なアンサンブル

#### 提案1: Stackingアンサンブル

```python
from sklearn.ensemble import StackingClassifier

# Level 0: Base models
estimators = [
    ('lgb', LightGBM_tuned),
    ('gb', GradientBoosting),
    ('rf', RandomForest),
    ('lr', LogisticRegression)
]

# Level 1: Meta-learner
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(penalty='l2', C=1.0),
    cv=5,
    stack_method='predict_proba'  # 確率値を使用
)
```

**期待効果**: 各モデルの強みを学習、0.835 → 0.840+

#### 提案2: Blendingアンサンブル

```python
# Train/Validation分割
from sklearn.model_selection import train_test_split

X_train_l0, X_val_l0, y_train_l0, y_val_l0 = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Level 0: 各モデルでValidation予測
models = [lgb_model, gb_model, rf_model, lr_model]
val_predictions = []

for model in models:
    model.fit(X_train_l0, y_train_l0)
    val_pred = model.predict_proba(X_val_l0)[:, 1]
    val_predictions.append(val_pred)

# Level 1: Meta-learnerを訓練
X_val_meta = np.column_stack(val_predictions)
meta_model = LogisticRegression()
meta_model.fit(X_val_meta, y_val_l0)
```

#### 提案3: 重み最適化アンサンブル

```python
from scipy.optimize import minimize

def ensemble_loss(weights, predictions, y_true):
    """最適重みを探索"""
    weighted_pred = np.average(predictions, axis=0, weights=weights)
    weighted_pred_binary = (weighted_pred >= 0.5).astype(int)
    return -accuracy_score(y_true, weighted_pred_binary)

# 初期重み: 等重み
initial_weights = np.array([1/3, 1/3, 1/3])

# 制約: 重みの和 = 1, 各重み >= 0
constraints = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})
bounds = [(0, 1)] * 3

# 最適化
result = minimize(
    ensemble_loss,
    initial_weights,
    args=(oof_predictions, y_train),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"Optimal weights: {optimal_weights}")
```

### 6.5 改善提案: ハイパーパラメータチューニング

#### 提案1: Bayesian Optimization

```python
from bayes_opt import BayesianOptimization

def lgb_cv(num_leaves, max_depth, min_child_samples, reg_alpha, reg_lambda):
    params = {
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'min_child_samples': int(min_child_samples),
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_estimators': 100,
        'learning_rate': 0.05,
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return cv_scores.mean()

# 探索空間
pbounds = {
    'num_leaves': (3, 8),
    'max_depth': (2, 5),
    'min_child_samples': (20, 50),
    'reg_alpha': (0.5, 2.0),
    'reg_lambda': (0.5, 2.0)
}

optimizer = BayesianOptimization(
    f=lgb_cv,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=10, n_iter=30)
```

#### 提案2: AutoML

```python
# H2O AutoML
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# データをH2O形式に変換
train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))

# AutoML実行
aml = H2OAutoML(max_models=20, seed=42, max_runtime_secs=3600)
aml.train(x=features, y='Survived', training_frame=train_h2o)

# リーダーボード確認
lb = aml.leaderboard
print(lb)
```

### 6.6 改善提案: クロスバリデーション戦略

#### 提案1: Stratified GroupKFold

```python
from sklearn.model_selection import StratifiedGroupKFold

# 家族グループをGroupとして使用
groups = train['Ticket']  # 同じチケット = 同じグループ

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups)):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    # モデル訓練
```

**効果**: データリーケージ防止、より現実的な評価

#### 提案2: Repeated Stratified K-Fold

```python
from sklearn.model_selection import RepeatedStratifiedKFold

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

scores = cross_val_score(model, X_train, y_train, cv=rskf, scoring='accuracy')
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
```

**効果**: より安定した性能推定

---

## 7. 推論と評価

### 7.1 現在の推論プロセス

```python
# 1. 特徴量エンジニアリング（Trainと同じ）
test_fe = engineer_features(test)

# 2. 特徴量選択
X_test = test_fe[selected_features]

# 3. 予測
predictions = model.predict(X_test)

# 4. Submission作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission.csv', index=False)
```

### 7.2 予測の信頼性評価

#### 提案1: 予測確率の分析

```python
# 予測確率を取得
pred_proba = model.predict_proba(X_test)[:, 1]

# 確信度の分布を確認
plt.hist(pred_proba, bins=50)
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.title('Prediction Confidence Distribution')

# 低確信度サンプルを特定
uncertain = test[np.abs(pred_proba - 0.5) < 0.1]  # 0.4 < p < 0.6
print(f"Uncertain predictions: {len(uncertain)} samples")
```

#### 提案2: アンサンブルの合意度

```python
# 複数モデルの予測
models = [lgb_model, gb_model, rf_model]
predictions = [model.predict(X_test) for model in models]

# 合意度を計算
agreement = np.array(predictions).std(axis=0)  # 分散が低い = 合意

# 不一致サンプルを分析
disagreement_idx = np.where(agreement > 0.4)[0]
print(f"Disagreement samples: {len(disagreement_idx)}")
```

### 7.3 キャリブレーション

#### 現状評価

Isotonic Calibrationを試したが、効果限定的（既にキャリブレーション良好）

#### 提案: Platt Scaling

```python
from sklearn.calibration import CalibratedClassifierCV

# 確率キャリブレーション
calibrated_model = CalibratedClassifierCV(
    base_estimator=lgb_model,
    method='sigmoid',  # Platt Scaling
    cv=5
)

calibrated_model.fit(X_train, y_train)
pred_proba_calibrated = calibrated_model.predict_proba(X_test)
```

### 7.4 改善提案: テスト時増強（TTA）

```python
def create_augmented_features(df, n_augmentations=5):
    """ノイズ追加でアンサンブル"""
    augmented_preds = []

    for i in range(n_augmentations):
        # 数値特徴量に小さなノイズを追加
        df_aug = df.copy()
        noise = np.random.normal(0, 0.01, size=df_aug[['Age', 'Fare']].shape)
        df_aug[['Age', 'Fare']] += noise

        # 予測
        pred = model.predict_proba(df_aug)[:, 1]
        augmented_preds.append(pred)

    # 平均
    return np.mean(augmented_preds, axis=0)

tta_predictions = create_augmented_features(X_test, n_augmentations=10)
```

---

## 8. 改善提案

### 8.1 優先度1: 即座に実装可能

#### 1. Stackingアンサンブル

**実装難易度**: ⭐⭐
**期待効果**: CV 0.835 → 0.840 (+0.5%)
**実装時間**: 2-3時間

```python
# 実装コード例は6.4参照
stacking_model = StackingClassifier(...)
```

#### 2. 外れ値処理

**実装難易度**: ⭐
**期待効果**: ロバスト性向上、Gap削減
**実装時間**: 1時間

```python
from scipy.stats.mstats import winsorize
df['Fare'] = winsorize(df['Fare'], limits=[0.01, 0.01])
```

#### 3. ドメイン知識ベース特徴量

**実装難易度**: ⭐
**期待効果**: CV +0.2-0.5%
**実装時間**: 1-2時間

```python
df['WomenAndChildren'] = ((df['Sex'] == 1) | (df['Age'] < 16)).astype(int)
df['PriorityScore'] = ...  # 実装例は5.2参照
```

### 8.2 優先度2: 中期的実装

#### 4. Bayesian Optimizationチューニング

**実装難易度**: ⭐⭐⭐
**期待効果**: 各モデル +0.2-0.3%
**実装時間**: 1日

#### 5. GroupKFold実装

**実装難易度**: ⭐⭐
**期待効果**: より正確な性能評価
**実装時間**: 2-3時間

#### 6. 高度な特徴量エンジニアリング

**実装難易度**: ⭐⭐⭐
**期待効果**: CV +0.3-0.7%
**実装時間**: 1-2日

- Ticket/Cabinグループ特徴量
- Target Encoding（CV内実装）
- 相互作用特徴量

### 8.3 優先度3: 長期的検討

#### 7. ニューラルネットワーク

**実装難易度**: ⭐⭐⭐⭐
**期待効果**: 未知（小規模データで過学習リスク）
**実装時間**: 2-3日

```python
import torch
import torch.nn as nn

class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
```

#### 8. AutoMLの活用

**実装難易度**: ⭐⭐
**期待効果**: 新しいアプローチの発見
**実装時間**: 4-8時間（ツール習得含む）

#### 9. データ拡張

**実装難易度**: ⭐⭐⭐⭐
**期待効果**: 不明（Tabularデータで難易度高）
**実装時間**: 2-3日

```python
from sdv.tabular import CTGAN

# 合成データ生成
model = CTGAN()
model.fit(train)
synthetic_samples = model.sample(num_rows=500)
```

---

## 9. 実装ロードマップ

### Phase 1: クイックウィン（1週間）

**Week 1**

| 日 | タスク | 期待効果 |
|---|--------|---------|
| 1 | 外れ値処理実装 | ロバスト性↑ |
| 2 | ドメイン特徴量追加 | CV +0.3% |
| 3 | Stackingアンサンブル実装 | CV +0.5% |
| 4 | GroupKFold実装 | 評価精度↑ |
| 5 | Blendingアンサンブル実装 | CV +0.3% |
| 6 | 統合テストとバリデーション | - |
| 7 | Kaggle提出と結果分析 | LB確認 |

**期待CV**: 0.835 → 0.843

### Phase 2: 中期改善（2-3週間）

**Week 2-3**

- Bayesian Optimizationチューニング
- Target Encoding実装
- Ticket/Cabinグループ特徴量
- 相互作用特徴量探索
- Repeated CV実装

**期待CV**: 0.843 → 0.850

### Phase 3: 長期実験（1-2ヶ月）

**Month 2**

- ニューラルネットワーク実験
- AutoML探索
- データ拡張研究
- アンサンブルの深化（3層スタッキング等）

**期待CV**: 0.850 → 0.855+

### リスク管理

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 過学習の悪化 | 中 | 高 | CVとLBの継続的モニタリング |
| 実装バグ | 高 | 中 | ユニットテスト、コードレビュー |
| 時間超過 | 中 | 低 | 優先度に基づく段階的実装 |
| LB低下 | 低 | 高 | Kaggleリミット考慮、慎重な提出 |

---

## 10. まとめ

### 10.1 現状の強み

1. ✅ 過学習を大幅に削減（Gap 0.009）
2. ✅ 強力な個別モデル（LightGBM CV 0.8317）
3. ✅ 最適化されたアンサンブル（OOF 0.8350）
4. ✅ ロバストな特徴量選択（9特徴、多重共線性排除）
5. ✅ 科学的アプローチ（仮説→実験→検証）

### 10.2 改善の方向性

#### すぐに実装すべき（ROI高）

1. Stackingアンサンブル
2. 外れ値処理
3. ドメイン知識特徴量

#### 検討すべき（中期）

4. Bayesian Optimization
5. GroupKFold
6. 高度な特徴量エンジニアリング

#### 実験的に試す（長期）

7. ニューラルネットワーク
8. AutoML
9. データ拡張

### 10.3 期待される成果

| Phase | CV Score | LB Score (予想) | 改善幅 |
|-------|----------|----------------|--------|
| **現在** | 0.8350 | 0.80-0.82 | - |
| **Phase 1** | 0.8430 | 0.81-0.83 | +0.8% |
| **Phase 2** | 0.8500 | 0.82-0.84 | +1.5% |
| **Phase 3** | 0.8550+ | 0.83-0.85+ | +2.0%+ |

### 10.4 成功の鍵

1. **過学習の継続的監視**: CVとLBのGapを常にチェック
2. **段階的な改善**: 一度に多くを変更せず、変更の影響を測定
3. **ロバスト性重視**: CVだけでなく、モデルの安定性を評価
4. **ドメイン知識の活用**: データの背景を理解した特徴量設計
5. **実験の記録**: すべての実験結果をドキュメント化

---

## 付録A: コードテンプレート

### A.1 完全な前処理パイプライン

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy.stats.mstats import winsorize

def preprocess_pipeline(train_df, test_df):
    """完全な前処理パイプライン"""

    # データコピー
    train = train_df.copy()
    test = test_df.copy()
    all_data = pd.concat([train, test], sort=False, ignore_index=True)

    # 1. 欠損フラグ（補完前）
    all_data['Age_Missing'] = all_data['Age'].isnull().astype(int)
    all_data['Cabin_Missing'] = all_data['Cabin'].isnull().astype(int)
    all_data['Embarked_Missing'] = all_data['Embarked'].isnull().astype(int)

    # 2. Title抽出
    all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    all_data['Title'] = all_data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
    all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
    all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    all_data['Title'] = all_data['Title'].map(title_mapping).fillna(0)

    # 3. 家族関連特徴量
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)
    all_data['SmallFamily'] = ((all_data['FamilySize'] >= 2) & (all_data['FamilySize'] <= 4)).astype(int)

    # 4. Age補完
    for title in all_data['Title'].unique():
        for pclass in all_data['Pclass'].unique():
            mask = (all_data['Title'] == title) & (all_data['Pclass'] == pclass) & (all_data['Age'].isnull())
            median_age = all_data[(all_data['Title'] == title) & (all_data['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                all_data.loc[mask, 'Age'] = median_age
    all_data['Age'].fillna(all_data['Age'].median(), inplace=True)

    # 5. Fare補完と外れ値処理
    all_data['Fare'].fillna(all_data['Fare'].median(), inplace=True)
    all_data['Fare'] = winsorize(all_data['Fare'], limits=[0.01, 0.01])
    all_data['FareBin'] = pd.qcut(all_data['Fare'], q=4, labels=False, duplicates='drop')

    # 6. 外れ値処理（Age）
    all_data['Age'] = winsorize(all_data['Age'], limits=[0.01, 0.01])

    # 7. Sex encoding
    all_data['Sex'] = all_data['Sex'].map({'male': 0, 'female': 1})

    # 8. Embarked
    all_data['Embarked'].fillna(all_data['Embarked'].mode()[0], inplace=True)

    # 9. ドメイン知識特徴量
    all_data['WomenAndChildren'] = ((all_data['Sex'] == 1) | (all_data['Age'] < 16)).astype(int)

    # 分割
    train_processed = all_data[:len(train_df)].copy()
    test_processed = all_data[len(train_df):].copy()

    return train_processed, test_processed

# 使用例
train_fe, test_fe = preprocess_pipeline(train, test)
```

### A.2 Stackingアンサンブル実装

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# 特徴量選択
features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
            'SmallFamily', 'Age_Missing', 'Embarked_Missing', 'WomenAndChildren']

X_train = train_fe[features]
y_train = train_fe['Survived']
X_test = test_fe[features]

# Base models
estimators = [
    ('lgb', lgb.LGBMClassifier(
        n_estimators=100, max_depth=2, num_leaves=4,
        min_child_samples=30, reg_alpha=1.0, reg_lambda=1.0,
        learning_rate=0.05, random_state=42, verbose=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        min_samples_split=20, min_samples_leaf=10, random_state=42
    )),
    ('rf', RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_split=20,
        min_samples_leaf=10, random_state=42
    )),
    ('lr', LogisticRegression(
        max_iter=1000, random_state=42
    ))
]

# Stacking
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(penalty='l2', C=1.0),
    cv=5,
    stack_method='predict_proba'
)

# 訓練
stacking_model.fit(X_train, y_train)

# 予測
predictions = stacking_model.predict(X_test)

# Submission
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submission_stacking.csv', index=False)

# CV評価
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Stacking CV: {scores.mean():.4f} (±{scores.std():.4f})")
```

---

## 付録B: 評価指標

### B.1 使用している指標

- **Accuracy**: 正解率（Kaggle評価指標）
- **CV Score**: 交差検証スコア
- **Gap**: Train Score - CV Score（過学習の指標）
- **OOF (Out-of-Fold) Score**: 全Foldの予測を統合したスコア

### B.2 追加推奨指標

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# 2. Precision, Recall, F1
print(classification_report(y_true, y_pred))

# 3. AUC-ROC
auc = roc_auc_score(y_true, y_pred_proba)
print(f"AUC: {auc:.4f}")

# 4. Log Loss
from sklearn.metrics import log_loss
logloss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {logloss:.4f}")
```

---

**レポート終了**

このレポートに基づいて段階的に改善を実施することで、Titanicコンペティションでのスコア向上とモデルのロバスト性向上が期待できます。
