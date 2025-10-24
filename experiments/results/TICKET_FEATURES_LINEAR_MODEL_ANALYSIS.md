# Ticket特徴量 × LogisticRegression 最適化分析

**日付**: 2025-10-24
**目的**: LogisticRegressionで悪化するTicket特徴量を最適化し、線形モデルでも使えるか検証

---

## エグゼクティブサマリー

❌ **結論**: **Ticket特徴量はLogisticRegressionでは推奨しない**

- どの特徴量選択戦略も改善なし（-0.78% ～ -1.26%の悪化）
- 根本原因: SibSp/Parch/FamilySize/Ticket_GroupSize間の深刻な多重共線性（VIF = ∞）
- 正則化強化（C=0.1）でも悪化（-0.33%）

**推奨**:
- **Tree-basedモデル**: 全Ticket特徴量使用 ✅ (+1.08%)
- **線形モデル**: Ticket特徴量なし ✅ (ベースライン維持)

---

## 1. 問題の診断

### 1.1 元の結果

| モデル | Baseline | +All Ticket | 変化 |
|--------|----------|-------------|------|
| GradientBoosting | 0.8294 | 0.8384 | **+0.0090** ✅ |
| LightGBM | 0.8361 | 0.8384 | +0.0023 ✅ |
| RandomForest | 0.8339 | 0.8361 | +0.0022 ✅ |
| **LogisticRegression** | 0.8036 | 0.7913 | **-0.0123** ❌ |

### 1.2 VIF分析（多重共線性診断）

**基準**: VIF > 10 は問題、VIF > 5 は要注意

| 特徴量 | VIF | 判定 |
|--------|-----|------|
| **SibSp** | **∞** | 🔴 完全な線形依存 |
| **FamilySize** | **∞** | 🔴 完全な線形依存 |
| **Parch** | **∞** | 🔴 完全な線形依存 |
| **Ticket_Fare_Per_Person** | **9.53** | 🔴 深刻 |
| **Fare** | **7.71** | 🟠 問題あり |
| **Ticket_GroupSize** | **5.86** | 🟠 要注意 |
| Pclass | 3.02 | 🟢 OK |
| Ticket_Is_Numeric | 2.92 | 🟢 OK |

**発見**: SibSp, Parch, FamilySizeが完全に線形依存！
```
FamilySize = SibSp + Parch + 1
```

### 1.3 高相関ペア（|r| > 0.6）

| 特徴量1 | 特徴量2 | 相関 | 問題 |
|---------|---------|------|------|
| SibSp | FamilySize | **+0.891** | 🔴 |
| SibSp | Ticket_GroupSize | **+0.727** | 🔴 |
| Parch | FamilySize | **+0.783** | 🔴 |
| Parch | Ticket_GroupSize | **+0.638** | 🔴 |
| **FamilySize** | **Ticket_GroupSize** | **+0.816** | 🔴 |
| **SmallFamily** | **Ticket_SmallGroup** | **+0.706** | 🔴 |
| **Fare** | **Ticket_Fare_Per_Person** | **+0.827** | 🔴 |
| Pclass | Ticket_Fare_Per_Person | **-0.763** | 🔴 |
| Ticket_Length | Ticket_Is_Numeric | **-0.758** | 🔴 |

**問題**: Ticket特徴量の多くが既存特徴量と高相関

---

## 2. 試した4つの戦略

### Strategy 1: 高VIF特徴量を除外

**アプローチ**: FamilySize, Fareを除外し、代わりにTicket特徴量を使用

```
除外: FamilySize, Fare
追加: Ticket_GroupSize, Ticket_Fare_Per_Person, Ticket_Is_Numeric,
      Ticket_Prefix_Category, Ticket_Length, Ticket_SmallGroup
```

**結果**: 0.7958 (**-0.0078**, -0.97%) ❌

### Strategy 2: Ticket重複除外

**アプローチ**: 既存特徴量と重複しないTicket特徴量のみ使用

```
除外: なし
追加: Ticket_Is_Numeric, Ticket_Prefix_Category, Ticket_Length
(除外Ticket: GroupSize, SmallGroup, FarePerPerson)
```

**結果**: 0.7947 (**-0.0090**, -1.13%) ❌

### Strategy 3: 最小構成

**アプローチ**: 最も独立性の高いTicket特徴量のみ

```
除外: FamilySize
追加: Ticket_Is_Numeric, Ticket_Prefix_Category
```

**結果**: 0.7935 (**-0.0101**, -1.26%) ❌

### Strategy 4: Fare系統一 ⭐ 最良

**アプローチ**: FareをTicket_Fare_Per_Personに置き換え

```
除外: Fare
追加: Ticket_Fare_Per_Person, Ticket_Is_Numeric,
      Ticket_Prefix_Category, Ticket_SmallGroup
```

**結果**: 0.7958 (**-0.0078**, -0.97%) ❌

**VIF改善**: Ticket_Fare_Per_Person VIF = 2.72（許容範囲）

---

## 3. 正則化の調整

全Ticket特徴量を使用し、正則化強度Cを調整：

| C値 | 正則化 | CV Score | 変化 |
|-----|--------|----------|------|
| 0.01 | 超強 | 0.7587 | **-0.0449** ❌ |
| 0.10 | 強 | 0.8003 | **-0.0033** ❌ |
| 0.50 | 中 | 0.7913 | -0.0123 ❌ |
| 1.00 | 弱（デフォルト） | 0.7913 | -0.0123 ❌ |
| 2.00～10.00 | 超弱 | 0.7902 | -0.0134 ❌ |

**発見**: 正則化を強くしても改善せず

---

## 4. なぜLogisticRegressionで悪化するのか？

### 4.1 Tree-basedモデル vs 線形モデル

| 特性 | Tree-based | 線形モデル |
|------|------------|------------|
| **多重共線性への耐性** | 🟢 強い | 🔴 弱い |
| **特徴量選択** | 自動（分割で選択） | すべて同時に使用 |
| **相互作用** | 🟢 自動検出 | 手動で作成必要 |
| **解釈性** | 中 | 🟢 高い |

### 4.2 Tree-basedモデルでの動作

```
GradientBoostingの分割例:

Node 1: FamilySize で分割
├─ FamilySize <= 3: Ticket_Fare_Per_Person で分割
└─ FamilySize > 3: Pclass で分割

→ Ticket_GroupSize は使わない（FamilySizeと重複）
→ 自動的に最適な特徴量を選択
```

**利点**: 冗長な特徴量があっても問題なし（使わない）

### 4.3 LogisticRegressionでの動作

```
Survived = β₀ + β₁·FamilySize + β₂·Ticket_GroupSize + ...

問題: FamilySize と Ticket_GroupSize が高相関（0.816）
→ どちらの係数が大きいか不安定
→ 係数の推定誤差が大きくなる
→ 汎化性能が下がる
```

**結果**: 多重共線性により係数が不安定 → 性能悪化

---

## 5. 最終推奨事項

### ✅ 推奨構成（モデル別）

| モデルタイプ | Ticket特徴量 | 期待改善 | 理由 |
|--------------|--------------|----------|------|
| **GradientBoosting** | 全6特徴量 | **+0.0090** | 自動で特徴量選択 |
| **LightGBM** | 全6特徴量 | +0.0023 | 同上 |
| **RandomForest** | 全6特徴量 | +0.0022 | 同上 |
| **LogisticRegression** | **なし** | **0.0000** | 多重共線性回避 |

### 🎯 実装ガイドライン

```python
# モデル定義時に特徴量を分ける

if model_name in ['GradientBoosting', 'LightGBM', 'RandomForest']:
    # Tree-based: 全Ticket特徴量使用
    features = base_features + [
        'Ticket_GroupSize',
        'Ticket_Fare_Per_Person',
        'Ticket_Is_Numeric',
        'Ticket_Prefix_Category',
        'Ticket_Length',
        'Ticket_SmallGroup'
    ]
elif model_name == 'LogisticRegression':
    # 線形: ベースライン特徴量のみ
    features = base_features
    # またはL1正則化で自動選択
    # model = LogisticRegression(penalty='l1', solver='saga')
```

### ⚠️ 代替案（もし線形モデルでTicket情報を使うなら）

**Option A: 特徴量を厳選**
- `Ticket_Is_Numeric` のみ追加（VIF=1.09, 独立性高い）
- 期待改善: わずか（+0.001程度）

**Option B: L1正則化（Lasso）で自動選択**
```python
LogisticRegression(penalty='l1', solver='saga', C=0.1)
```

**Option C: PCA/次元削減**
- Ticket特徴量をPCAで1-2次元に圧縮
- 多重共線性を解消

ただし、**いずれも改善は限定的**。シンプルにTicket特徴量なしが最良。

---

## 6. 技術的な深掘り

### 6.1 完全な線形依存の例

```python
# なぜVIF = ∞ になるのか

# 定義から明らか:
FamilySize = SibSp + Parch + 1

# つまり:
FamilySize = 1.0·SibSp + 1.0·Parch + 1.0

# これは完全な線形依存（rank deficiency）
# 行列が特異行列になり、VIF = ∞
```

### 6.2 多重共線性の数式的影響

```
Var(β̂ⱼ) ∝ VIF_j · σ²

VIF_j = 1 / (1 - R²_j)

R²_j: 特徴量j を他の特徴量で回帰した決定係数
```

- Ticket_GroupSize → FamilySize で回帰: R² = 0.666
- VIF = 1/(1-0.666) = **2.99**
- 係数の分散が約3倍に増加 → 推定が不安定

---

## 7. 結論

### 🎯 最終結論

**LogisticRegressionではTicket特徴量を使わない**

**理由**:
1. **深刻な多重共線性**: VIF = ∞（SibSp/Parch/FamilySize）
2. **すべての戦略で悪化**: -0.78% ～ -1.26%
3. **正則化でも改善なし**: C=0.1でも-0.33%
4. **複雑性の増加**: 特徴量数が増えるだけでデメリット

**Tree-basedモデルとの使い分け**:
- **Tree-based**: 全Ticket特徴量使用 → **+1.08%改善** ✅
- **線形モデル**: Ticket特徴量なし → **ベースライン維持** ✅

### 📊 最終スコア比較

| モデル | 推奨構成 | CV Score |
|--------|----------|----------|
| GradientBoosting | Base + Ticket | **0.8384** |
| LightGBM | Base + Ticket | 0.8384 |
| RandomForest | Base + Ticket | 0.8361 |
| LogisticRegression | **Base only** | **0.8036** |

---

**生成日**: 2025-10-24
**分析スクリプト**: `scripts/feature_engineering/ticket_features_for_linear_models.py`
