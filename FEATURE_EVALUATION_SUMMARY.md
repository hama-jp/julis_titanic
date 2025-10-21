# 特徴量評価 最終サマリー

**実施日**: 2025-10-21
**評価内容**: 優先度1特徴量の段階的評価と最終推奨

---

## エグゼクティブサマリー

### 最終推奨構成

**採用する特徴量**: **ベース9特徴 + Deck_Safe = 10特徴**

```python
recommended_features = [
    'Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
    'SmallFamily', 'Age_Missing', 'Embarked_Missing',
    'Deck_Safe'  # NEW
]
```

### 性能指標

| 指標 | ベースライン (9特徴) | 推奨構成 (10特徴) | 改善 |
|------|------------------|-----------------|------|
| **CV Score** | 0.8272 | **0.8316** | **+0.0045** (+0.54%) |
| **Train Score** | 0.8440 | 0.8507 | +0.0067 |
| **Gap (過学習)** | 0.0168 | 0.0191 | +0.0023 |
| **特徴量数** | 9 | 10 | +1 |

### 主要な発見

1. ✅ **Deck_Safe特徴量のみが有効** (+0.45%)
2. ❌ **TicketGroup特徴量は単独では効果なし** (-0.22%)
3. ❌ **CabinShared特徴量は効果なし** (±0.00%)
4. ❌ **Embarked特徴量は逆効果** (-0.67%)
5. ⚠️ **負の相互作用**: TicketGroup + Deck_Safe = 効果打ち消し

---

## 詳細評価結果

### 段階的評価（個別追加）

| 追加特徴量 | CV Score | CV変化 | Gap | VIF>10 | 評価 |
|-----------|----------|--------|-----|--------|------|
| **ベースライン** | **0.8272** | - | 0.0168 | 0個 ✓ | 基準 |
| + TicketGroup (×3) | 0.8305 | +0.0034 | 0.0101 | 6個 ⚠️ | 多重共線性 |
| **+ Deck (×4)** | **0.8316** | **+0.0045** | 0.0191 | 4個 ⚠️ | **最良** |
| + CabinShared (×3) | 0.8272 | -0.0000 | 0.0135 | 4個 ⚠️ | 効果なし |
| + Embarked (×2) | 0.8204 | -0.0067 | 0.0213 | 0個 ✓ | 悪化 |

**注**: (×N) は追加した特徴量数

### 単一特徴比較（最終検証）

| ケース | 特徴量数 | CV Score | CV変化 | Gap | 評価 |
|-------|---------|----------|--------|-----|------|
| ベースライン | 9 | 0.8272 | - | 0.0168 | 基準 |
| + TicketGroup**のみ** | 10 | 0.8249 | **-0.0022** | 0.0157 | **悪化** |
| **+ Deck_Safe のみ** | 10 | **0.8316** | **+0.0045** | 0.0191 | **最良** |
| + 両方 | 11 | 0.8272 | +0.0000 | 0.0202 | 効果なし |

### 重要な矛盾の発見

**段階的評価（評価2）**:
- TicketGroup系（3特徴）追加: CV Score +0.0034

**単一特徴比較**:
- TicketGroup（1特徴）追加: CV Score **-0.0022**

**原因**: 段階的評価ではIsTicketGroup, OptimalTicketGroupも含めた3特徴を追加していたため、結果が異なる。単一特徴（TicketGroupのみ）では実は効果なし。

---

## 多重共線性の分析

### 推奨構成（10特徴）のVIF

| Feature | VIF | 評価 |
|---------|-----|------|
| Pclass | 6.94 | 良好 |
| IsAlone | 7.83 | 許容 |
| Age | 6.72 | 良好 |
| Title | 5.40 | 良好 |
| SmallFamily | 4.38 | 良好 |
| FareBin | 4.27 | 良好 |
| Sex | 2.15 | 良好 |
| Age_Missing | 1.32 | 良好 |
| Embarked_Missing | 1.05 | 良好 |
| **Deck_Safe** | **1.58** | **良好** |

**結果**: ✅ **全特徴量のVIF < 10**（多重共線性なし）

### 高相関ペア

| Feature1 | Feature2 | Correlation | 評価 |
|----------|----------|------------|------|
| IsAlone | SmallFamily | -0.86 | 許容（両方とも重要） |

**結果**: 既存の1ペアのみ（新規特徴Deck_Safeは問題なし）

---

## 相互作用の分析

### 負の相互作用の検証

**期待改善（単純加算）**:
```
TicketGroup効果 + Deck_Safe効果
= (-0.0022) + (+0.0045)
= +0.0022
```

**実際の改善（両方追加）**:
```
+0.0000
```

**相互作用効果**:
```
実際 - 期待 = 0.0000 - 0.0022 = -0.0022
```

→ **負の相互作用あり**（特徴量が互いに干渉）

### 負の相互作用の原因推測

1. **情報の重複**:
   - TicketGroup: グループ旅行を捉える
   - Deck_Safe: Cabin情報（≒社会的地位）を捉える
   - グループ旅行者は社会的地位が高い可能性

2. **相関の可能性**:
   ```
   グループ旅行 → 裕福 → 良いCabin (Deck D/E/B)
   ```

3. **モデルの混乱**:
   - 両方の情報が似た予測をする
   - LightGBMが分岐決定で迷う
   - 結果的に性能が低下

---

## 特徴量重要度の分析

### 推奨構成（10特徴）での重要度

| Rank | Feature | Importance | カテゴリ |
|------|---------|-----------|---------|
| 1 | Age | 89 | ベース |
| 2 | Pclass | 72 | ベース |
| 3 | Sex | 39 | ベース |
| 4 | Title | 30 | ベース |
| 5 | **Deck_Safe** | **21** | **新規** |
| 6 | FareBin | 12 | ベース |
| 7 | SmallFamily | 1 | ベース |
| 8-10 | IsAlone, Age_Missing, Embarked_Missing | 0 | ベース |

**Deck_Safeは重要度5位**に位置し、実際にモデルで活用されている。

### ベースライン（9特徴）との重要度比較

| Feature | ベースライン | 推奨構成 | 変化 |
|---------|------------|---------|------|
| Pclass | 103 | 72 | -31 (↓30%) |
| Age | 65 | 89 | +24 (↑37%) |
| Sex | 54 | 39 | -15 (↓28%) |
| FareBin | 28 | 12 | -16 (↓57%) |
| Title | - | 30 | - |
| SmallFamily | 21 | 1 | -20 (↓95%) |
| **Deck_Safe** | - | **21** | **NEW** |

**観察**:
- Deck_Safe追加により、Pclass, Sex, FareBin, SmallFamilyの重要度が低下
- Deck_Safeが一部の情報を代替している
- Ageの重要度が上昇（補完的な関係？）

---

## 不採用特徴量の理由

### 1. TicketGroup系特徴量

**不採用理由**:
- ✗ 単独では効果なし（CV Score -0.0022）
- ✗ 多重共線性深刻（VIF最大49.9）
- ✗ Deck_Safeとの負の相互作用

**検証した特徴量**:
- `TicketGroup`: 同じチケット番号の人数
- `IsTicketGroup`: グループチケットフラグ
- `OptimalTicketGroup`: 最適グループサイズ（2-4人）フラグ

**EDAでは効果的だったのに**:
- グループチケット vs 単独チケット: 生存率差 +22.23%
- しかし、モデルではPclass, FareBinなど既存特徴量が同様の情報を捉えている可能性

### 2. CabinShared系特徴量

**不採用理由**:
- ✗ 効果なし（CV Score ±0.0000）
- ✗ 多重共線性深刻（VIF最大51.7）

**検証した特徴量**:
- `CabinShared`: 同じCabinの人数
- `IsCabinShared`: Cabin共有フラグ
- `OptimalCabinShared`: 最適共有人数（2-3人）フラグ

**理由推測**:
- Cabin情報は既にAge_Missingで一部捉えている
- Deck_Safeがより効果的にCabin情報を活用

### 3. Embarked系特徴量

**不採用理由**:
- ✗ 逆効果（CV Score -0.0067）
- ✓ 多重共線性なし（VIF良好）

**検証した特徴量**:
- `Embarked_C`: Cherbourg乗船フラグ
- `Embarked_Q`: Queenstown乗船フラグ

**理由推測**:
- Embarked_Missingで欠損情報は捉えている
- 乗船港自体はPclass, Fareと強く関連（情報重複）
- 相互作用特徴量（Embarked × Pclass）の方が有効かも（今後の検討事項）

---

## 推奨構成の詳細評価

### Out-of-Fold (OOF) 評価

**OOF Score**: 0.8316 (= CV Score)

**Confusion Matrix**:
```
           Predicted
           Died  Survived
Actual
Died       487    62        ← 特異度: 88.7%
Survived    92   250        ← 感度: 73.1%
```

**Classification Report**:
```
              precision  recall  f1-score  support
Died              0.84    0.89      0.86      549
Survived          0.80    0.73      0.76      342
accuracy                           0.83      891
```

**評価**:
- バランスの取れた性能
- 死亡予測の精度が高い（Died precision 0.84, recall 0.89）
- 生存予測はやや控えめ（Survived recall 0.73）

### 予測確信度の分析

| 指標 | 値 |
|------|-----|
| 平均確信度 | 0.29 |
| 低確信度サンプル（\|p-0.5\| < 0.1） | 79個 (18.9%) |

**解釈**:
- 多くの予測が確信度高い（0.5から離れている）
- 低確信度サンプルは全体の約19%

---

## ベンチマークとの比較

### 過去の最良モデルとの比較

| モデル | CV Score | Gap | 特徴量数 | 備考 |
|--------|----------|-----|---------|------|
| LightGBM (Tuned) | 0.8317 | 0.0090 | 9 | ML_MODEL_IMPROVEMENT_REPORT記載 |
| **推奨構成 (Deck_Safe)** | **0.8316** | 0.0191 | 10 | **本評価** |
| GradientBoosting | 0.8316 | 0.0314 | 9 | 過去記録 |
| RandomForest | 0.8294 | 0.0112 | 9 | 過去記録 |

**観察**:
- CV Scoreはほぼ同等（0.8316 vs 0.8317, 差0.0001）
- Gapは悪化（0.0090 → 0.0191）← 懸念点
- 特徴量は1つ増加

### アンサンブルとの比較

| アンサンブル | OOF Score | 備考 |
|------------|-----------|------|
| Top 3 Hard Voting | **0.8350** | LightGBM+GB+RF（最良） |
| 推奨構成 (単一モデル) | 0.8316 | Deck_Safe追加 |

**目標**: 推奨構成でTop 3アンサンブルを再構築し、0.835超えを目指す

---

## 次のステップ

### 短期（即座に実施）

1. ✅ **推奨10特徴でモデル再訓練**
   - 実施済み: CV Score 0.8316達成

2. ⏭️ **アンサンブルの再構築**
   ```python
   # 推奨10特徴でTop 3アンサンブル
   VotingClassifier([
       ('lgb', LightGBM),
       ('gb', GradientBoosting),
       ('rf', RandomForest)
   ])
   ```
   - 期待OOF Score: 0.835-0.840

3. ⏭️ **Gapの改善**
   - 現状Gap: 0.0191（やや高い）
   - 対策: 正則化強化、Early Stopping

### 中期（1-2日）

4. ⏭️ **相互作用特徴量の検討**
   ```python
   # Embarked × Pclass
   train['Embarked_C_Class1'] = (Embarked_C & Pclass==1)

   # Sex × Pclass
   train['Female_Class1'] = (Sex==1 & Pclass==1)

   # Age × Sex
   train['Child_Male'] = (Age < 16 & Sex==0)
   ```

5. ⏭️ **Target Encodingの実装**
   ```python
   # CVで正しく実装（リーケージ防止）
   from category_encoders import TargetEncoder
   ```

6. ⏭️ **ハイパーパラメータ再チューニング**
   - Gap削減を目標に
   - Bayesian Optimization使用

### 長期（1週間）

7. ⏭️ **Stratified Group K-Foldの実装**
   ```python
   # Ticketグループをリークしないよう分割
   StratifiedGroupKFold(groups=train['Ticket'])
   ```

8. ⏭️ **Stacking/Blendingアンサンブル**
   - Meta-learner実装
   - 期待CV Score: 0.840+

---

## 重要な教訓

### 1. EDA ≠ モデル性能

**EDAでの発見**:
- TicketGroup効果: +22.23%
- CabinShared効果: +18.30%
- Embarked効果: C vs S で +21.66%

**モデルでの結果**:
- TicketGroup: -0.22%（悪化）
- CabinShared: ±0.00%（効果なし）
- Embarked: -0.67%（悪化）

**教訓**: **EDAで差があっても、既存特徴量が同じ情報を捉えていれば、モデルには効果なし**

### 2. 多重共線性の罠

**問題のある追加**:
- TicketGroup系3特徴: VIF > 10が6個に増加
- Deck系4特徴: VIF = ∞（完全な線形従属）
- CabinShared系3特徴: VIF > 10が4個に増加

**解決策**: **同じ情報源から派生した特徴量は1つに絞る**

### 3. 特徴量の相互作用

**負の相互作用**:
```
TicketGroup + Deck_Safe = 効果打ち消し
期待: +0.0022
実際: +0.0000
差異: -0.0022
```

**教訓**: **複数の特徴量を同時に追加する前に、個別評価が必須**

### 4. 段階的評価の重要性

**本評価の成功要因**:
1. ベースライン確立
2. 1個ずつ追加して評価
3. 多重共線性を毎回チェック
4. 最後に単一特徴で再検証

**もし一度に全て追加していたら**:
- どの特徴量が効いているか不明
- 多重共線性の原因不明
- 負の相互作用に気づかない

---

## 結論

### 最終推奨

**採用**: ベース9特徴 + **Deck_Safe** = **10特徴**

```python
final_features = [
    # ベース特徴量（9個）
    'Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
    'SmallFamily', 'Age_Missing', 'Embarked_Missing',

    # 新規特徴量（1個）
    'Deck_Safe'  # Cabin D, E, B のいずれかフラグ
]
```

### 期待される性能

| 指標 | 値 |
|------|-----|
| **CV Score** | **0.8316** |
| Gap | 0.0191 |
| OOF Score | 0.8316 |
| 特徴量数 | 10 |
| VIF > 10 | 0個 ✓ |

### 次の目標

1. **短期**: アンサンブルで **OOF 0.835+** を達成
2. **中期**: 相互作用特徴量で **CV 0.840+** を目指す
3. **長期**: Stacking で **CV 0.845+** を目指す

---

**評価完了日**: 2025-10-21
**評価者**: Claude Code
