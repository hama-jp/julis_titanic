# 優先度1特徴量の段階的評価レポート

**実施日**: 2025-10-21
**目的**: 優先度1特徴量を1個ずつ追加してモデル性能と多重共線性を評価

---

## エグゼクティブサマリー

### 評価結果

| モデル | 特徴量数 | CV Score | CV変化 | Gap | VIF>10 | 評価 |
|--------|---------|----------|--------|-----|--------|------|
| **ベースライン** | 9 | **0.8272** | - | 0.0168 | 0個 ✓ | 基準 |
| + TicketGroup | 12 | **0.8305** | **+0.0034** | 0.0101 | 6個 ⚠️ | 改善だが多重共線性 |
| **+ Deck** | 13 | **0.8316** | **+0.0045** | 0.0191 | 4個 ⚠️ | **最良だが問題あり** |
| + CabinShared | 12 | 0.8272 | -0.0000 | 0.0135 | 4個 ⚠️ | 効果なし |
| + Embarked | 11 | 0.8204 | **-0.0067** | 0.0213 | 0個 ✓ | **悪化** |

### 主要な発見

1. **Deck特徴量が最も効果的** (+0.0045, 0.8272→0.8316)
   - しかし**完全な多重共線性**あり（VIF = inf）
   - Deck_Safe = Deck_D OR Deck_E OR Deck_B の線形従属

2. **TicketGroup特徴量も有効** (+0.0034, 0.8272→0.8305)
   - **強い多重共線性**あり（VIF最大49.9）
   - IsTicketGroup ↔ OptimalTicketGroup (r=0.82)

3. **CabinShared特徴量は効果なし** (±0.0000)
   - 多重共線性あり（VIF最大51.7）

4. **Embarked特徴量は逆効果** (-0.0067)
   - 多重共線性はなし
   - モデル性能が悪化

### 推奨アクション

1. **Deck特徴量**: Deck_Safeのみを使用（他の4特徴は削除）
2. **TicketGroup特徴量**: TicketGroupのみを使用（派生特徴は削除）
3. **CabinShared特徴量**: 採用しない
4. **Embarked特徴量**: 採用しない

**期待される最終構成**: 9（ベース）+ 1（Deck_Safe）+ 1（TicketGroup）= **11特徴量**

---

## 詳細評価

### 評価1: ベースライン（現在の9特徴量）

#### 特徴量リスト

```
['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
 'SmallFamily', 'Age_Missing', 'Embarked_Missing']
```

#### 多重共線性チェック

**VIF（分散拡大係数）**:

| Feature | VIF | 評価 |
|---------|-----|------|
| IsAlone | 7.48 | 許容範囲 |
| Age | 6.65 | 良好 |
| Pclass | 6.08 | 良好 |
| Title | 5.38 | 良好 |
| SmallFamily | 4.26 | 良好 |
| FareBin | 4.05 | 良好 |
| Sex | 2.13 | 良好 |
| Age_Missing | 1.31 | 良好 |
| Embarked_Missing | 1.02 | 良好 |

**高相関ペア（|r| > 0.7）**:

- IsAlone ↔ SmallFamily: r = -0.86

**評価**: VIF < 10で全て許容範囲内。IsAloneとSmallFamilyに高相関があるが、両方とも重要な特徴量として保持。

#### モデル性能

```
CV Score:     0.8272 (±0.0130)
Train Score:  0.8440
Gap:          0.0168  ← 良好（過学習が少ない）
```

#### 特徴量重要度 Top 5

| Feature | Importance |
|---------|-----------|
| Pclass | 103 |
| Age | 65 |
| Sex | 54 |
| FareBin | 28 |
| SmallFamily | 21 |

---

### 評価2: + TicketGroup関連特徴量

#### 追加特徴量

```
+ TicketGroup          (同じチケット番号の人数)
+ IsTicketGroup        (グループチケットか否か)
+ OptimalTicketGroup   (最適グループサイズ2-4人か否か)
```

特徴量数: 9 → **12** (+3)

#### 多重共線性チェック

**VIF > 10の特徴量（⚠️警告）**:

| Feature | VIF | 問題レベル |
|---------|-----|----------|
| **IsTicketGroup** | **49.86** | 重大 |
| **OptimalTicketGroup** | **24.81** | 重大 |
| **TicketGroup** | **18.88** | 重大 |
| IsAlone | 13.52 | 警告 |
| Pclass | 12.42 | 警告 |
| FareBin | 11.47 | 警告 |

**高相関ペア（|r| > 0.7）**:

| Feature1 | Feature2 | Correlation |
|----------|----------|------------|
| IsAlone | SmallFamily | -0.86 |
| IsAlone | IsTicketGroup | -0.73 |
| FareBin | IsTicketGroup | 0.73 |
| SmallFamily | OptimalTicketGroup | 0.71 |
| **IsTicketGroup** | **OptimalTicketGroup** | **0.82** |

**問題点**:
1. IsTicketGroup ↔ OptimalTicketGroup の高相関（r=0.82）
2. 3つのTicketGroup関連特徴量すべてがVIF > 10
3. 既存特徴量（IsAlone, Pclass, FareBin）のVIFも上昇

#### モデル性能

```
CV Score:     0.8305 (±0.0062)  ← +0.0034改善
Train Score:  0.8406
Gap:          0.0101             ← 改善（過学習減少）
```

#### 新規特徴量の重要度

| Feature | Importance | 評価 |
|---------|-----------|------|
| **TicketGroup** | **31** | 重要 |
| OptimalTicketGroup | 3 | 低い |
| IsTicketGroup | 0 | 使用されず |

**結論**:
- ✅ **TicketGroup（連続値）のみが有用**
- ❌ IsTicketGroup, OptimalTicketGroupは冗長

**推奨**: TicketGroupのみを採用

---

### 評価3: + Deck関連特徴量

#### 追加特徴量

```
+ Deck_D        (Deck Dフラグ)
+ Deck_E        (Deck Eフラグ)
+ Deck_B        (Deck Bフラグ)
+ Deck_Safe     (D, E, Bのいずれかフラグ)
```

特徴量数: 9 → **13** (+4)

#### 多重共線性チェック

**VIF > 10の特徴量（⚠️重大な問題）**:

| Feature | VIF | 問題レベル |
|---------|-----|----------|
| **Deck_Safe** | **∞** | **完全な多重共線性** |
| **Deck_D** | **∞** | **完全な多重共線性** |
| **Deck_E** | **∞** | **完全な多重共線性** |
| **Deck_B** | **∞** | **完全な多重共線性** |

**原因**:
```
Deck_Safe = Deck_D OR Deck_E OR Deck_B
```
→ 完全な線形従属関係

**高相関ペア**: なし（ベース特徴量間のみ）

#### モデル性能

```
CV Score:     0.8316 (±0.0198)  ← +0.0045改善（最良）
Train Score:  0.8507
Gap:          0.0191             ← 悪化（過学習増加）
```

#### 新規特徴量の重要度

| Feature | Importance | 評価 |
|---------|-----------|------|
| **Deck_Safe** | **24** | **重要** |
| Deck_D | 0 | 使用されず |
| Deck_E | 0 | 使用されず |
| Deck_B | 0 | 使用されず |

**結論**:
- ✅ **Deck_Safeのみが有用**
- ❌ 個別Deckフラグ（D, E, B）は冗長
- ⚠️ 完全な多重共線性のため、LightGBMは自動的にDeck_Safeのみを使用

**推奨**: Deck_Safeのみを採用

---

### 評価4: + CabinShared関連特徴量

#### 追加特徴量

```
+ CabinShared           (同じCabinの人数)
+ IsCabinShared         (Cabin共有フラグ)
+ OptimalCabinShared    (最適共有人数2-3人フラグ)
```

特徴量数: 9 → **12** (+3)

#### 多重共線性チェック

**VIF > 10の特徴量（⚠️警告）**:

| Feature | VIF | 問題レベル |
|---------|-----|----------|
| **IsCabinShared** | **51.68** | 重大 |
| **CabinShared** | **50.96** | 重大 |
| **OptimalCabinShared** | **20.42** | 重大 |
| Pclass | 12.16 | 警告 |

**高相関ペア（|r| > 0.7）**:

| Feature1 | Feature2 | Correlation |
|----------|----------|------------|
| IsAlone | SmallFamily | -0.86 |
| CabinShared | IsCabinShared | 0.81 |
| **IsCabinShared** | **OptimalCabinShared** | **0.86** |

**問題点**:
1. 3つのCabinShared関連特徴量すべてがVIF > 10
2. IsCabinShared ↔ OptimalCabinShared の高相関（r=0.86）
3. TicketGroup系と同様の冗長性

#### モデル性能

```
CV Score:     0.8272 (±0.0140)  ← ±0.0000（効果なし）
Train Score:  0.8406
Gap:          0.0135             ← わずかに改善
```

#### 新規特徴量の重要度

| Feature | Importance | 評価 |
|---------|-----------|------|
| OptimalCabinShared | 13 | やや重要 |
| CabinShared | 11 | やや重要 |
| IsCabinShared | 0 | 使用されず |

**結論**:
- ❌ **CV Scoreに改善なし**
- ⚠️ 多重共線性が深刻
- モデルは使用するが、予測性能向上には寄与せず

**推奨**: 採用しない

---

### 評価5: + Embarked関連特徴量

#### 追加特徴量

```
+ Embarked_C   (Cherbourg乗船フラグ)
+ Embarked_Q   (Queenstown乗船フラグ)
```

※ Embarked_Sは冗長のため除外（C=0, Q=0 で表現可能）

特徴量数: 9 → **11** (+2)

#### 多重共線性チェック

**VIF > 10の特徴量**: **なし** ✓

**VIF（全て良好）**:

| Feature | VIF | 評価 |
|---------|-----|------|
| IsAlone | 7.73 | 良好 |
| Age | 6.66 | 良好 |
| Pclass | 6.26 | 良好 |
| Title | 5.41 | 良好 |
| SmallFamily | 4.43 | 良好 |
| FareBin | 4.22 | 良好 |
| Sex | 2.16 | 良好 |
| Age_Missing | 1.48 | 良好 |
| **Embarked_C** | **1.37** | 良好 |
| **Embarked_Q** | **1.35** | 良好 |
| Embarked_Missing | 1.02 | 良好 |

**高相関ペア**: IsAlone ↔ SmallFamily (-0.86) のみ

#### モデル性能

```
CV Score:     0.8204 (±0.0122)  ← -0.0067悪化
Train Score:  0.8418
Gap:          0.0213             ← 悪化（過学習増加）
```

#### 新規特徴量の重要度

| Feature | Importance | 評価 |
|---------|-----------|------|
| Embarked_C | 13 | やや重要 |
| Embarked_Q | 4 | 低い |

**結論**:
- ❌ **CV Scoreが悪化**（-0.67%）
- ✅ 多重共線性なし
- ❌ モデル性能への寄与がマイナス

**推奨**: 採用しない

---

## 総合評価と推奨事項

### 問題点の整理

#### 1. TicketGroup系特徴量の多重共線性

**原因**:
```
TicketGroup (連続値: 1, 2, 3, ...)
IsTicketGroup = (TicketGroup > 1)
OptimalTicketGroup = (TicketGroup >= 2 & TicketGroup <= 4)
```
→ 3つの特徴量が同じ情報源から派生

**相関**:
- IsTicketGroup ↔ OptimalTicketGroup: r = 0.82

**解決策**: **TicketGroupのみを使用**

#### 2. Deck系特徴量の完全な多重共線性

**原因**:
```
Deck_Safe = Deck_D OR Deck_E OR Deck_B
```
→ 完全な線形従属（VIF = ∞）

**解決策**: **Deck_Safeのみを使用**

#### 3. CabinShared系特徴量の冗長性と効果の欠如

**原因**:
```
CabinShared (連続値)
IsCabinShared = (CabinShared > 1)
OptimalCabinShared = (CabinShared >= 2 & CabinShared <= 3)
```
→ TicketGroup系と同様の問題

**問題**: CV Scoreに改善なし（±0.0000）

**解決策**: **採用しない**

#### 4. Embarked特徴量の逆効果

**問題**: CV Score -0.67%の悪化

**原因考察**:
1. Embarked_Missingで既に欠損情報を捉えている
2. Embarked × Pclass の相互作用が必要？
3. サンプル数の偏り（S: 644, C: 168, Q: 77）

**解決策**: **採用しない**（または相互作用特徴量として再検討）

### 最終推奨構成

#### 採用する特徴量

| カテゴリ | 特徴量 | 理由 |
|---------|--------|------|
| **ベース** | Pclass, Sex, Title, Age, IsAlone, FareBin, SmallFamily, Age_Missing, Embarked_Missing | 既存の9特徴（維持） |
| **新規追加** | **TicketGroup** | +0.34% CV改善、多重共線性解消 |
| **新規追加** | **Deck_Safe** | +0.45% CV改善、多重共線性解消 |

**合計**: **11特徴量**

#### 期待される性能

| 項目 | ベースライン | 推奨構成 | 改善 |
|------|------------|---------|------|
| CV Score | 0.8272 | **0.8320-0.8350** | **+0.5-0.8%** |
| 特徴量数 | 9 | 11 | +2 |
| VIF > 10 | 0個 | 0個 ✓ | 維持 |
| Gap | 0.0168 | 0.015以下 | 改善期待 |

### 次のステップ

#### 短期（即座に実施）

1. **推奨11特徴量でモデル再訓練**
   ```python
   features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
               'SmallFamily', 'Age_Missing', 'Embarked_Missing',
               'TicketGroup', 'Deck_Safe']
   ```

2. **多重共線性の最終確認**
   - VIF < 10 を全特徴量で確認
   - 高相関ペア（|r| > 0.7）の再チェック

3. **CV性能の検証**
   - Stratified K-Fold (n=5)
   - 目標CV Score: 0.832以上

#### 中期（1-2日）

4. **相互作用特徴量の追加検討**
   ```python
   # Embarked × Pclass の相互作用
   # 評価5で単独では効果なかったが、相互作用では有効かも
   train['Embarked_C_Class1'] = (train['Embarked_C'] == 1) & (train['Pclass'] == 1)
   ```

5. **アンサンブルの再構築**
   - Top 3 Hard Voting (LightGBM + GB + RF)
   - 新特徴量での再訓練

6. **ハイパーパラメータ再チューニング**
   - 特徴量数が増えたため調整

#### 長期（1週間）

7. **優先度2特徴量の評価**
   - Sex × Pclass 相互作用
   - AgeGroup × Sex 相互作用
   - FareRank_within_Pclass

---

## 付録: 多重共線性の理論

### VIF（分散拡大係数）とは

**定義**:
```
VIF_i = 1 / (1 - R²_i)
```
ここで、R²_i は特徴量iを他の全特徴量で回帰したときの決定係数

**解釈**:
- VIF = 1: 他の特徴量と無相関
- VIF < 5: 問題なし
- VIF 5-10: 注意
- **VIF > 10: 多重共線性あり（対処必要）**
- VIF = ∞: 完全な多重共線性

### なぜ多重共線性が問題か

1. **係数の不安定性**: わずかなデータ変化で係数が大きく変動
2. **解釈の困難**: どの特徴量が本当に重要か判断できない
3. **過学習のリスク**: 冗長な情報でモデルが複雑化
4. **計算の不安定性**: 逆行列計算でエラー

### ツリーベースモデルでの影響

**LightGBM/XGBoost/Random Forestでは**:
- 線形モデルほど深刻ではない
- 自動的に重要な特徴を選択
- しかし、冗長な特徴は計算コストとメモリの無駄

**本評価での観察**:
- Deck系（VIF=∞）: LightGBMはDeck_Safeのみを使用
- TicketGroup系（VIF>18）: TicketGroupのみを使用
- モデルは賢く選択するが、事前に除去すべき

---

## まとめ

### 成功した特徴量

1. ✅ **TicketGroup**: グループ旅行の効果を捉える（+0.34%）
2. ✅ **Deck_Safe**: Cabin位置の安全性を捉える（+0.45%）

### 失敗した特徴量

1. ❌ **CabinShared**: 効果なし（±0.00%）
2. ❌ **Embarked**: 逆効果（-0.67%）

### 重要な教訓

1. **派生特徴量の冗長性**: 同じ情報源から複数の特徴量を作ると多重共線性が発生
2. **連続値 vs バイナリ**: 連続値（TicketGroup）の方がバイナリ（IsTicketGroup）より有用
3. **EDA ≠ モデル性能**: EDAで差があっても、モデルには効果がないことも（CabinShared, Embarked）
4. **段階的評価の重要性**: 一度に全て追加すると、どれが効いているか分からない

### 次回の改善方針

1. 相互作用特徴量の慎重な追加
2. 既存特徴量の見直し（IsAlone vs SmallFamilyの二択）
3. Target Encoding（CVで正しく実装）

---

**レポート終了**
