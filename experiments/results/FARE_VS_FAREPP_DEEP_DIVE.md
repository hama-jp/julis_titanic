# Fare vs Ticket_Fare_Per_Person 深掘り分析

**日付**: 2025-10-24
**目的**: FareとTicket_Fare_Per_Personの多重共線性（相関0.827）がモデルに与える影響を詳細分析

---

## エグゼクティブサマリー

🎯 **驚きの発見**: **Ticket_Fare_Per_Person単独が最良！Fareは不要**

| 特徴量構成 | GradientBoosting | 改善 vs Fare単独 |
|------------|------------------|------------------|
| Baseline + Fare | 0.8294 | - |
| **Baseline + Ticket_Fare_Per_Person** | **0.8429** | **+0.0135** 🏆 |
| Baseline + Both | 0.8339 | +0.0045 |

**主要な結論**:
1. Ticket_Fare_Per_Person単独がFare単独より**+1.35%良い**
2. 両方使うより、Ticket_Fare_Per_Person単独の方が**+0.90%良い**
3. 多重共線性（相関0.827）はTree-basedモデルでは問題なし
4. **推奨**: FareではなくTicket_Fare_Per_Personを使う

---

## 1. 基本統計の比較

### 1.1 分布の違い

| 統計量 | Fare | Ticket_Fare_Per_Person | 差 |
|--------|------|------------------------|-----|
| 平均 | $32.20 | $14.55 | $17.65 |
| 標準偏差 | $49.69 | $13.58 | - |
| 中央値 | $14.45 | $8.05 | $6.40 |
| 最大値 | $512.33 | $128.08 | $384.25 |
| 相関 | - | 0.8271 | - |

**発見**:
- Fareの分散が大きい（std=$49.69）→ グループサイズの影響
- Ticket_Fare_Per_Personの分散が小さい（std=$13.58）→ より安定

### 1.2 差分の分析

```
Fare - Ticket_Fare_Per_Person:
  平均: $17.65
  中央値: $0.00（50%は単独旅行者）
  最大: $384.25（4人グループの高額チケット）
```

**比率（Ticket_Fare_Per_Person / Fare）**:
- 平均: 0.71（約3割がグループ割引効果）
- 中央値: 1.00（50%は単独旅行者）
- 最小: 0.09（11人グループ）

---

## 2. 最大差のケース分析

### Top 10 差が大きいケース

| 乗客 | Ticket | GroupSize | Fare | FarePP | 差 | 生存 | Pclass |
|------|--------|-----------|------|--------|-----|------|--------|
| Ward, Miss. Anna | PC 17755 | 4 | $512.33 | $128.08 | **$384.25** | ✓ | 1st |
| Cardeza, Mr. Thomas | PC 17755 | 4 | $512.33 | $128.08 | $384.25 | ✓ | 1st |
| Ryerson, Miss. Emily | PC 17608 | 7 | $262.38 | $37.48 | $224.89 | ✓ | 1st |
| Fortune, Mr. Charles | 19950 | 6 | $263.00 | $43.83 | $219.17 | ✗ | 1st |

**パターン**:
- すべて1等客室
- グループサイズ4-7人
- ほとんど生存（大家族の1等客は生存率高い）

---

## 3. グループサイズ別の分析

| GroupSize | 平均Fare | 平均FarePP | 比率 | 生存率 | 件数 |
|-----------|----------|------------|------|--------|------|
| 1（単独） | $11.85 | $11.85 | 1.00 | 27.0% | 481 |
| 2 | $36.88 | $18.44 | 0.50 | 51.4% | 181 |
| 3 | $50.51 | $16.84 | 0.33 | 65.4% | 101 |
| **4** | **$108.28** | **$27.07** | **0.25** | **72.7%** | 44 |
| 5 | $87.82 | $17.56 | 0.20 | 33.3% | 21 |
| 6 | $103.75 | $17.29 | 0.17 | 21.1% | 19 |
| 7 | $61.45 | $8.78 | 0.14 | 20.8% | 24 |
| 8 | $52.07 | $6.51 | 0.13 | 38.5% | 13 |
| 11 | $69.55 | $6.32 | 0.09 | 0.0% | 7 |

**重要な発見**:
- **GroupSize=4**: 最高生存率（72.7%）+ 高Fare_Per_Person（$27.07）
- **GroupSize=1**: 低生存率（27.0%）
- GroupSize ≥ 5: 生存率急低下（20-38%）

**解釈**:
- 小グループ（2-4人）の裕福な家族 → 高生存率
- 大グループ（5+人）の貧困層 → 低生存率
- **Ticket_Fare_Per_Personがこの違いを捉える！**

---

## 4. 生存率との関係

### 4.1 4分位数別の生存率

**Fare 4分位数別**:
| Quartile | 生存率 | 件数 |
|----------|--------|------|
| Q1 | 19.7% | 223 |
| Q2 | 30.4% | 224 |
| Q3 | 45.5% | 222 |
| Q4 | 58.1% | 222 |

**Ticket_Fare_Per_Person 4分位数別**:
| Quartile | 生存率 | 件数 |
|----------|--------|------|
| Q1 | 27.2% | 224 |
| Q2 | 24.8% | 234 |
| Q3 | 39.0% | 210 |
| Q4 | **63.2%** | 223 |

**発見**: Ticket_Fare_Per_Person Q4の生存率（63.2%）がFare Q4（58.1%）より高い！

### 4.2 クロス集計（生存率）

```
             FarePP_Q1  FarePP_Q2  FarePP_Q3  FarePP_Q4
Fare_Q1        13.8%     23.5%       -         -
Fare_Q2        45.2%     22.6%     29.7%     66.7%  ← 注目！
Fare_Q3        41.5%     33.3%     53.3%     49.2%
Fare_Q4        19.5%       -       54.5%     68.6%
                ↑                              ↑
           大家族                         裕福な小グループ
```

**非対角要素の解釈**:

1. **Fare Q2 × FarePP Q4（生存率66.7%）**:
   - 中程度の総運賃だが、一人あたりは高い
   - → 裕福な2-3人グループ（高生存率）

2. **Fare Q4 × FarePP Q1（生存率19.5%）**:
   - 高総運賃だが、一人あたりは低い
   - → 大家族（5+人）の3等客（低生存率）

3. **Fare Q2 × FarePP Q1（生存率45.2%）**:
   - 中運賃、低FarePP
   - → 小グループの3等客（中程度の生存率）

**結論**: FareとTicket_Fare_Per_Personは独立した情報を提供！

---

## 5. モデル性能比較

### 5.1 詳細比較（3モデル）

#### GradientBoosting

| 特徴量構成 | CV Score | ランク |
|------------|----------|--------|
| **Baseline + Ticket_Fare_Per_Person** | **0.8429** | 🥇 1位 |
| Baseline + Ticket_GroupSize + FarePP | 0.8417 | 🥈 2位 |
| Baseline + Both (Fare + FarePP) | 0.8339 | 3位 |
| Baseline + Ticket_GroupSize + Fare | 0.8328 | 4位 |
| Baseline + Fare | 0.8294 | 5位 |
| Baseline + Ticket_GroupSize | 0.8283 | 6位 |
| Baseline (no Fare) | 0.8215 | 7位 |

**発見**: Ticket_Fare_Per_Person単独が最良（+0.0135 vs Fare単独）

#### LightGBM

| 特徴量構成 | CV Score | ランク |
|------------|----------|--------|
| **Baseline + Fare** | **0.8361** | 🥇 1位 |
| Baseline + Ticket_Fare_Per_Person | 0.8328 | 2位 |
| Baseline + Ticket_GroupSize + Fare | 0.8316 | 3位 |

**発見**: LightGBMではFare単独が最良（GBとは逆）

#### RandomForest

| 特徴量構成 | CV Score | ランク |
|------------|----------|--------|
| **Baseline + Both** | **0.8373** | 🥇 1位 |
| Baseline + Ticket_Fare_Per_Person | 0.8339 | 2位 |
| Baseline + Ticket_GroupSize + Fare | 0.8316 | 3位 |

**発見**: RandomForestでは両方使うのが最良

### 5.2 モデル別の推奨

| モデル | 推奨構成 | CV Score | 理由 |
|--------|----------|----------|------|
| **GradientBoosting** | **Ticket_Fare_Per_Person単独** | **0.8429** | 最高性能、シンプル |
| LightGBM | Fare単独 | 0.8361 | 従来の特徴量で十分 |
| RandomForest | Fare + FarePP | 0.8373 | 両方の情報を活用 |

---

## 6. 特徴量重要度の分析

### 6.1 両方使用時（GradientBoosting）

| 特徴量 | 重要度 | 順位 |
|--------|--------|------|
| Title | 0.5262 | 1位 |
| **Ticket_Fare_Per_Person** | **0.1541** | 2位 |
| **Fare** | **0.1231** | 3位 |
| Pclass | 0.0839 | 4位 |
| FamilySize | 0.0673 | 5位 |

**比率**: Ticket_Fare_Per_Person / Fare = **1.25x**

### 6.2 解釈

```python
# Tree-basedモデルの分割戦略（推定）

Node 1: Ticket_Fare_Per_Person <= 10.5 (主分割)
├─ True (低FarePP):
│  ├─ 生存率: 25%
│  └─ さらに Fare で分割（大家族 vs 貧困単独）
│     ├─ Fare > 20 → 大家族（一部生存）
│     └─ Fare <= 20 → 貧困単独（低生存）
│
└─ False (高FarePP):
   ├─ 生存率: 65%
   └─ Pclass で分割
       ├─ Pclass=1,2 → 高生存（裕福）
       └─ Pclass=3 → 中程度

→ Ticket_Fare_Per_Person が主要な分割基準
→ Fare は補助的（大家族の識別用）
```

**結論**: Ticket_Fare_Per_Personの方が情報量が多い

---

## 7. なぜ多重共線性が問題にならないのか？

### 7.1 Tree-basedモデルの特性

**線形モデルの問題**:
```
Survived = β₀ + β₁·Fare + β₂·Ticket_Fare_Per_Person + ...

問題: Fare と Ticket_Fare_Per_Person が相関0.827
→ どちらの係数が大きいか不安定（多重共線性）
→ 両方の係数が減少（VIF=7-10）
```

**Tree-basedモデルの利点**:
```
if Ticket_Fare_Per_Person <= 10.5:
    # Ticket_Fare_Per_Person を使用
    低生存率グループ
else:
    if Fare > 50:
        # ここでは Fare を使用（追加情報）
        大家族を識別
    else:
        高生存率グループ

→ 自動的に最適な特徴量を選択
→ 相関があっても問題なし（使い分ける）
```

### 7.2 なぜTicket_Fare_Per_Person単独が最良なのか？

**仮説**: Fareは冗長な情報を持つ

```
Ticket_Fare_Per_Person = Fare / Ticket_GroupSize

すでにTicket_GroupSizeを特徴量として持っている場合:
→ Fare = Ticket_Fare_Per_Person × Ticket_GroupSize
→ Fareは他の2つの特徴量から計算可能（冗長）

Tree-basedモデルの動作:
→ Ticket_Fare_Per_Person で分割（情報ゲイン大）
→ Fare で分割を試みる（情報ゲイン小、既に使用済み）
→ Fareを使わない（効率的）
```

**GBでの結果**:
- Ticket_Fare_Per_Person単独: 0.8429
- Fare追加: 0.8339（**-0.0090悪化！**）

**理由**: Fareを追加することで、モデルが複雑化し、過学習の可能性

---

## 8. 実用的な意味

### 8.1 Ticket_Fare_Per_Personが捉えるもの

**経済状況の指標**:
- 一人あたりの実質的な支払い能力
- グループサイズを考慮した富裕度

**具体例**:

| ケース | Fare | GroupSize | FarePP | 解釈 | 生存率予測 |
|--------|------|-----------|--------|------|------------|
| A | $100 | 1 | $100 | 裕福な単独 | 高 |
| B | $100 | 10 | $10 | 大家族の貧困層 | 低 |
| C | $50 | 2 | $25 | 中流の小グループ | 中 |

**Fareだけでは**:
- ケースAとBが同じ$100 → 区別できない

**Ticket_Fare_Per_Personでは**:
- ケースA: $100（高FarePP → 高生存）
- ケースB: $10（低FarePP → 低生存）
- → 正しく区別できる！

### 8.2 歴史的背景

タイタニック号の避難方針:
1. **女性と子供優先（最優先）**
2. **客室クラス順（1等 > 2等 > 3等）**
3. **グループサイズの影響**:
   - 小グループ: 迅速に避難
   - 大グループ: 全員を集めるのに時間がかかる

**Ticket_Fare_Per_Personが反映するもの**:
- 個人の富裕度（客室クラス）
- 迅速な避難能力（小グループ vs 大グループ）

---

## 9. 最終推奨事項

### ✅ モデル別の推奨（更新版）

#### GradientBoosting（主力モデル）

**推奨**: **Ticket_Fare_Per_Person単独**

```python
features = base_features + ['Ticket_Fare_Per_Person']
# Fareは不要（むしろ追加すると悪化）
```

**理由**:
- 最高性能（0.8429）
- シンプル（Fareより）
- Fare追加で-0.0090悪化

**代替案**: Ticket_GroupSize + Ticket_Fare_Per_Person（0.8417、-0.0012）

#### LightGBM

**推奨**: **Fare単独** または **Ticket_Fare_Per_Person単独**

```python
features = base_features + ['Fare']
# または
features = base_features + ['Ticket_Fare_Per_Person']
```

**理由**:
- Fare単独が最良（0.8361）
- Ticket_Fare_Per_Personでもほぼ同じ（0.8328、-0.0033）

#### RandomForest

**推奨**: **Both（Fare + Ticket_Fare_Per_Person）**

```python
features = base_features + ['Fare', 'Ticket_Fare_Per_Person']
```

**理由**:
- 両方使うのが最良（0.8373）
- RandomForestは特徴量の冗長性に強い

### ⚠️ 注意点

1. **GradientBoostingでは両方使わない**:
   - 両方使うと-0.0090悪化
   - Ticket_Fare_Per_Person単独が最良

2. **モデルによって最適構成が異なる**:
   - GB: Ticket_Fare_Per_Person単独
   - LGB: Fare単独
   - RF: 両方

3. **多重共線性は心配不要（Tree-based）**:
   - 相関0.827でも問題なし
   - むしろ冗長性（過剰情報）が問題

---

## 10. 結論

### 🎯 核心的な発見

1. **Ticket_Fare_Per_Personは Fare よりも優れた特徴量**:
   - GradientBoostingで+1.35%改善
   - 特徴量重要度1.25倍

2. **多重共線性（相関0.827）は問題ではない**:
   - Tree-basedモデルは自動で使い分け
   - むしろ冗長性（両方使うこと）が問題

3. **Fareを除外しTicket_Fare_Per_Personを使う**:
   - より良い性能（+1.35%）
   - よりシンプル（特徴量1つ減）

4. **グループサイズの影響を正しく捉える**:
   - 裕福な小グループ（高FarePP）→ 高生存
   - 大家族の貧困層（低FarePP）→ 低生存

### 📊 実用的な推奨

**GradientBoostingの最終構成**:
```python
features = [
    'Pclass', 'Sex', 'SibSp', 'Parch', 'Title',
    'FamilySize', 'SmallFamily', 'Embarked_*',
    'Ticket_Fare_Per_Person'  # ← Fareの代わりに
]

# NOT: Fare（冗長、悪化）
```

**期待性能**: 0.8429（Fare使用時0.8294から+1.35%改善）

---

**生成日**: 2025-10-24
**分析スクリプト**: `scripts/analysis/fare_vs_ticket_fare_per_person.py`
