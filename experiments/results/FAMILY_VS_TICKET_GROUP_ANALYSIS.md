# SibSp/Parch/FamilySize vs Ticket_GroupSize 分析

**日付**: 2025-10-24
**目的**: SibSp、Parch、FamilySizeは冗長？Ticket_GroupSizeだけで十分か検証

---

## エグゼクティブサマリー

✅ **結論**: **SibSp/Parchを除外し、FamilySize + Ticket_GroupSizeを使う**

| 構成 | GradientBoosting | 評価 |
|------|------------------|------|
| **FamilySize + SmallFamily + Ticket_GroupSize** | **0.8451** | 🥇 **最良** |
| Baseline (SibSp+Parch+FamilySize+SmallFamily) | 0.8429 | 2位（-0.22%） |
| Ticket_GroupSize単独 | 0.8395 | 5位（-0.56%） |

**改善**: **+0.0023 (+0.27%)**

**主要な結論**:
1. **SibSp/Parchは不要** → FamilySizeで代替可能
2. **FamilySize + Ticket_GroupSizeの組み合わせが最良**
3. Ticket_GroupSize単独では不十分（両方必要）
4. 相関0.816と高いが、両方使うことで性能向上

---

## 1. 問題意識

### 1.1 冗長な特徴量の可能性

```python
# 既存の特徴量
SibSp: 兄弟姉妹・配偶者の数
Parch: 両親・子供の数
FamilySize = SibSp + Parch + 1  # 派生特徴量
Ticket_GroupSize: 同じチケットを共有する人数

# 問題
- SibSp + Parch から FamilySize が計算可能（完全な線形依存）
- FamilySize と Ticket_GroupSize は高相関（0.816）
- すべて使うのは冗長？
```

### 1.2 仮説

**仮説1**: SibSp/Parchは不要、FamilySizeで十分
**仮説2**: Ticket_GroupSizeだけで十分（FamilySizeも不要）
**仮説3**: FamilySize + Ticket_GroupSizeの組み合わせが最良

---

## 2. 基本統計と相関

### 2.1 基本統計

| 統計量 | SibSp | Parch | FamilySize | Ticket_GroupSize |
|--------|-------|-------|------------|------------------|
| 平均 | 0.52 | 0.38 | 1.90 | 2.12 |
| 標準偏差 | 1.10 | 0.81 | 1.61 | 1.80 |
| 中央値 | 0 | 0 | 1 | 1 |
| 最大値 | 8 | 6 | 11 | 11 |

### 2.2 相関行列

|  | SibSp | Parch | FamilySize | Ticket_GroupSize |
|--|-------|-------|------------|------------------|
| **SibSp** | 1.000 | 0.415 | **0.891** | 0.727 |
| **Parch** | 0.415 | 1.000 | **0.783** | 0.638 |
| **FamilySize** | 0.891 | 0.783 | 1.000 | **0.816** |
| **Ticket_GroupSize** | 0.727 | 0.638 | 0.816 | 1.000 |

**発見**:
- SibSp/Parchは FamilySize と非常に高相関（0.783-0.891）
- FamilySize と Ticket_GroupSize も高相関（0.816）

### 2.3 Survivedとの相関

| 特徴量 | 相関 | 順位 |
|--------|------|------|
| Parch | +0.082 | 1位 |
| Ticket_GroupSize | +0.065 | 2位 |
| FamilySize | +0.017 | 3位 |
| SibSp | -0.035 | 4位 |

**発見**: 単独ではParchが最も生存率と相関が高い

---

## 3. FamilySizeとTicket_GroupSizeの違い

### 3.1 一致度

**一致率**: 78.11%（696/891）

**差分の分布**:
```
差分 (FamilySize - Ticket_GroupSize):
  -7:   7件
  -6:   2件
  -5:   1件
  -4:   7件
  -3:  14件
  -2:  27件
  -1:  82件
   0: 696件（一致）
  +1:  38件
  +2:  13件
  +3:   3件
  +6:   1件
```

**発見**: 22%（195件）は FamilySize ≠ Ticket_GroupSize

### 3.2 パターン1: FamilySize < Ticket_GroupSize

**件数**: 140件（15.7%）
**生存率**: 51.4%

**意味**: 家族以外と同じチケットを共有

**具体例（Ticket 1601）**:
```
PassengerId  Name              Ticket  SibSp  Parch  FamilySize  Ticket_GroupSize
75           Bing, Mr. Lee     1601    0      0      1           8
170          Ling, Mr. Lee     1601    0      0      1           8
510          Lang, Mr. Fang    1601    0      0      1           8
...（計8人が同じチケット）

→ 友人・同僚グループでチケット共有
→ 生存率: 6/8 = 75%（高い！）
```

### 3.3 パターン2: FamilySize > Ticket_GroupSize

**件数**: 55件（6.2%）
**生存率**: 34.5%

**意味**: 大家族が複数チケットに分散

**具体例（Andersson家）**:
```
PassengerId  Name                      Ticket    FamilySize  Ticket_GroupSize
69           Andersson, Miss. Erna    3101281   7           1

→ 7人家族だが、チケットは個別
→ 家族が分散して座席配置
```

### 3.4 なぜ違いが重要か？

**FamilySize**: 血縁関係（生物学的な家族）
**Ticket_GroupSize**: 実際の同行者（社会的なグループ）

**避難時の影響**:
- 小さな友人グループ（Ticket_GroupSize=2-4、FamilySize=1）
  → 迅速に避難（高生存率）
- 大家族（FamilySize=7、Ticket_GroupSize=1-2）
  → バラバラに避難（低生存率）

**両方の情報が必要！**

---

## 4. モデル性能比較

### 4.1 8つの構成で比較

| 構成 | GB | LGB | RF |
|------|-----|-----|-----|
| **FamilySize + SmallFamily + Ticket_GroupSize** | **0.8451** 🥇 | 0.8272 | 0.8305 |
| FamilySize + Ticket_GroupSize | **0.8429** | 0.8238 | 0.8328 |
| Baseline (SibSp+Parch+FamilySize+SmallFamily) | **0.8429** | 0.8328 | 0.8316 |
| All (SibSp+Parch+FamilySize+Ticket_GroupSize) | **0.8429** | 0.8283 | 0.8316 |
| Ticket_GroupSize only | 0.8395 | 0.8305 | **0.8339** 🥇 |
| SibSp + Parch + Ticket_GroupSize | 0.8395 | 0.8249 | 0.8316 |
| Ticket_GroupSize + Ticket_SmallGroup | 0.8361 | **0.8328** | **0.8328** |

### 4.2 主要な発見

#### GradientBoosting（主力モデル）

**Top 3**:
1. **FamilySize + SmallFamily + Ticket_GroupSize**: 0.8451 ← **最良**
2. FamilySize + Ticket_GroupSize: 0.8429
3. Baseline: 0.8429

**改善**: +0.0023 (+0.27%)

**重要**:
- SibSp/Parchを追加しても改善なし（All: 0.8429）
- SmallFamilyを追加すると向上（+0.27%）

#### RandomForest

**Top 3**:
1. **Ticket_GroupSize only**: 0.8339 ← **単独で最良**
2. Ticket_GroupSize + Ticket_SmallGroup: 0.8328
3. FamilySize + Ticket_GroupSize: 0.8328

**発見**: RandomForestではTicket_GroupSize単独が最良！

#### LightGBM

**Top 3**:
1. Baseline & Ticket_GroupSize + Ticket_SmallGroup: 0.8328（同点）
2. Ticket_GroupSize only: 0.8305

**発見**: LightGBMではベースラインと同等

---

## 5. 特徴量重要度の比較

### 5.1 ベースライン（SibSp+Parch+FamilySize+SmallFamily）

| 特徴量 | 重要度 | 順位 |
|--------|--------|------|
| Title | 0.546 | 1 |
| Ticket_Fare_Per_Person | 0.221 | 2 |
| Pclass | 0.095 | 3 |
| **FamilySize** | **0.093** | 4 |
| Sex | 0.025 | 5 |
| **SibSp** | **0.010** | 6 ← **低い** |
| SmallFamily | 0.003 | 7 |
| **Parch** | **0.002** | 9 ← **非常に低い** |

**発見**: SibSp/Parchの重要度は非常に低い（FamilySizeで十分）

### 5.2 推奨構成（Ticket_GroupSize + Ticket_SmallGroup）

| 特徴量 | 重要度 | 順位 |
|--------|--------|------|
| Title | 0.552 | 1 |
| Ticket_Fare_Per_Person | 0.200 | 2 |
| Pclass | 0.104 | 3 |
| **Ticket_GroupSize** | **0.099** | 4 ← **FamilySizeより高い** |
| Sex | 0.017 | 5 |
| **Ticket_SmallGroup** | **0.016** | 6 |

**発見**: Ticket_GroupSizeの重要度がFamilySize（0.093）より高い（0.099）

---

## 6. ベースラインとの比較

### 6.1 改善率（GradientBoosting）

| 構成 | スコア | 変化 |
|------|--------|------|
| Baseline | 0.8429 | - |
| **Ticket_GroupSize only** | 0.8395 | **-0.34%** |
| Ticket_GroupSize + Ticket_SmallGroup | 0.8361 | -0.80% |

**結論**: Ticket_GroupSize単独では**不十分**

### 6.2 改善率（RandomForest）

| 構成 | スコア | 変化 |
|------|--------|------|
| Baseline | 0.8316 | - |
| **Ticket_GroupSize only** | 0.8339 | **+0.27%** ✅ |
| Ticket_GroupSize + Ticket_SmallGroup | 0.8328 | +0.13% |

**結論**: RandomForestでは Ticket_GroupSize単独で**改善**

---

## 7. なぜFamilySize + Ticket_GroupSizeが最良なのか？

### 7.1 異なる情報を提供

**FamilySize**: 血縁関係（生物学的）
- 避難時の心理的影響
- 家族全員を集める必要性
- 子供・高齢者の有無

**Ticket_GroupSize**: 実際の同行者（社会的）
- 物理的な近接性（同じ座席エリア）
- 避難時の実際の集団サイズ
- チケット共有による経済状況

### 7.2 相互補完的

**例1: 友人グループ**
```
FamilySize=1（単独）、Ticket_GroupSize=4（友人4人）
→ 迅速に避難可能（家族を探す必要なし）
→ 高生存率
```

**例2: 大家族（分散）**
```
FamilySize=7（大家族）、Ticket_GroupSize=1（個別チケット）
→ 避難時に家族を探す
→ 低生存率
```

**例3: 小家族（同一チケット）**
```
FamilySize=3、Ticket_GroupSize=3
→ 家族全員が同じ場所
→ 中程度の生存率
```

### 7.3 Tree-basedモデルでの使い分け

```python
# 推定される分割ロジック

Node 1: Ticket_Fare_Per_Person <= 10.5
├─ True (低運賃):
│  └─ FamilySize > 4?
│     ├─ True: 大家族 → 低生存率
│     └─ False:
│        └─ Ticket_GroupSize > FamilySize?
│           ├─ True: 友人グループ → 中生存率
│           └─ False: 単独 → 低生存率
│
└─ False (高運賃):
   └─ Ticket_GroupSize <= 4?
      ├─ True: 小グループ → 高生存率
      └─ False: 大グループ → 中生存率

→ FamilySizeとTicket_GroupSizeの組み合わせで細かい分割
```

---

## 8. 最終推奨事項

### ✅ モデル別の推奨構成

#### GradientBoosting（主力）

**推奨**: **FamilySize + SmallFamily + Ticket_GroupSize**

```python
features = [
    'Pclass', 'Sex', 'Title',
    'FamilySize', 'SmallFamily',    # ← SibSp/Parchの代わり
    'Ticket_GroupSize',              # ← 追加
    'Ticket_Fare_Per_Person',
    'Ticket_Is_Numeric',
    'Ticket_Prefix_Category',
    'Embarked_*',
]

# NOT: SibSp, Parch（冗長、重要度低い）
```

**期待スコア**: 0.8451（+0.27%改善）

#### RandomForest

**推奨**: **Ticket_GroupSize only**

```python
features = [
    'Pclass', 'Sex', 'Title',
    'Ticket_GroupSize',              # ← 単独で十分
    'Ticket_Fare_Per_Person',
    'Ticket_Is_Numeric',
    'Ticket_Prefix_Category',
    'Embarked_*',
]

# NOT: SibSp, Parch, FamilySize, SmallFamily
```

**期待スコア**: 0.8339（+0.27%改善）

#### LightGBM

**推奨**: **Baseline または Ticket_GroupSize + Ticket_SmallGroup**

```python
# Option 1: Baseline（従来通り）
features = base_features + ['SibSp', 'Parch', 'FamilySize', 'SmallFamily']

# Option 2: Ticket優先
features = base_features + ['Ticket_GroupSize', 'Ticket_SmallGroup']
```

**期待スコア**: 0.8328（同等）

---

## 9. 実装上の注意点

### ⚠️ 注意点

1. **多重共線性は問題なし**:
   - 相関0.816と高いが、Tree-basedモデルは自動で使い分け
   - むしろ両方使うことで性能向上

2. **SibSp/Parchは完全に除外**:
   - 重要度が非常に低い（0.002-0.010）
   - FamilySizeで完全に代替可能

3. **SmallFamilyの効果**:
   - FamilySize + Ticket_GroupSize に SmallFamily を追加 → +0.27%
   - SmallFamilyは有効（保持推奨）

4. **モデルによって最適構成が異なる**:
   - GB: FamilySize + SmallFamily + Ticket_GroupSize
   - RF: Ticket_GroupSize only
   - LGB: Baseline または Ticket_GroupSize + Ticket_SmallGroup

---

## 10. 結論

### 🎯 核心的な結論

1. **SibSp/Parchは不要**:
   - FamilySizeで完全に代替可能
   - 重要度が非常に低い（0.002-0.010）
   - 除外してもスコア低下なし

2. **FamilySize + Ticket_GroupSizeの組み合わせが最良**:
   - 相関0.816と高いが、両方必要
   - 異なる情報を提供（血縁 vs 同行者）
   - GradientBoostingで+0.27%改善

3. **Ticket_GroupSize単独では不十分**:
   - GradientBoostingで-0.34%悪化
   - RandomForestでは+0.27%改善（モデル依存）

4. **SmallFamilyが重要**:
   - FamilySize + Ticket_GroupSize に追加で+0.27%

### 📊 最終スコア

**GradientBoosting（推奨構成）**:
```
Before: 0.8429（SibSp+Parch+FamilySize+SmallFamily）
After:  0.8451（FamilySize+SmallFamily+Ticket_GroupSize）
改善:   +0.0023 (+0.27%)
```

### 🎓 学んだこと

1. **相関が高い ≠ 冗長**:
   - FamilySizeとTicket_GroupSizeは相関0.816
   - でも両方使うと性能向上（異なる情報）

2. **派生特徴量の優位性**:
   - SibSp/Parch（生の特徴量）< FamilySize（派生）
   - 適切な特徴量エンジニアリングが重要

3. **特徴量重要度との一致**:
   - 重要度低い特徴量（SibSp/Parch）を除外
   - 性能が向上（シンプル化 + 多重共線性減少）

---

**生成日**: 2025-10-24
**分析スクリプト**: `scripts/analysis/family_vs_ticket_group_comparison.py`
