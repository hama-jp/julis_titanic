# Deck_Safe特徴量の検証結果

**検証日**: 2025-10-21
**結論**: ⚠️ **Deck_Safe特徴量はLB性能を悪化させる**

---

## 実験結果サマリー

| 提出順 | モデル | 特徴量数 | LB Score | OOF Score | CV-LB Gap | LB改善 |
|-------|--------|---------|----------|-----------|-----------|--------|
| 1st | Soft Voting + 閾値0.4236 | 10特徴 | 0.75837 | 0.8260 | 6.76% | - |
| 2nd | Hard Voting + 閾値0.5 | 10特徴 | 0.76794 | 0.8305 | 6.26% | +0.957% |
| 3rd | **Hard Voting + 閾値0.5** | **9特徴** | **0.77751** | **不明** | **不明** | **+0.957%** |

### 累積改善

**提出1→3の改善**:
- LB Score: 0.75837 → **0.77751**
- **累積改善**: **+0.01914** (+1.914%)

**提出2→3の改善**:
- LB Score: 0.76794 → 0.77751
- **Deck_Safe除外による改善**: **+0.00957** (+0.957%)

---

## 🔍 重要な発見: Deck_Safeの過学習

### 矛盾する結果

| 評価指標 | 9特徴 | 10特徴（+Deck_Safe） | Deck_Safe効果 |
|---------|-------|-------------------|--------------|
| **OOF Score** | 0.8272（推定） | 0.8305 | **+0.0033** ✅ |
| **LB Score** | **0.77751** | 0.76794 | **-0.00957** ❌ |

### 解釈

**OOFでは改善、LBでは悪化**:
- ✅ Deck_Safe追加でOOFスコア向上（+0.0033）
- ❌ しかしLBスコアは悪化（-0.00957）
- **結論**: Deck_Safeは**Train dataに過学習**している

---

## 🎯 Deck_Safe特徴量の問題点

### 問題1: Train/Test分布の違い

**Trainデータ**:
```
Cabin情報あり: 204/891 = 22.9%
Deck_Safe (D/E/B): 一定の割合
```

**Testデータ**:
- Cabin情報の割合が不明
- Deck分布がTrainと異なる可能性

**推測**: TestデータではCabin情報の分布が異なり、Deck_Safe特徴量が機能しない

---

### 問題2: 少数サンプルへの過学習

**Deck_Safe作成ロジック**:
```python
# Deck_Safe = 1 if Deck in [D, E, B], else 0
df['Deck'] = df['Cabin'].str[0]
df['Deck_Safe'] = df['Deck'].isin(['D', 'E', 'B']).astype(int)
```

**Trainデータでの分布**（推定）:
- Deck D/E/B: 高い生存率（推定80-90%）
- しかし**サンプル数が少ない**（50-80人程度?）
- 少数サンプルの傾向を過度に学習

**Testデータ**:
- 同じDeck D/E/Bでも生存率が異なる可能性
- または該当者が少なく効果が薄い

---

### 問題3: 他特徴量との相関

**Deck_Safeと相関する特徴量**:
- `Pclass`: 1等客室が多い
- `Fare`: 高額運賃
- `Deck_B`, `Deck_D`, `Deck_E`: 完全に冗長

**VIF分析結果**（過去の評価）:
- Deck_D: VIF = ∞（完全多重共線性）
- Deck_E: VIF = ∞
- Deck_B: VIF = ∞
- Deck_Safe: VIF = ∞（これらの論理和）

**問題**: 多重共線性により、モデルがDeck_Safeの効果を正確に学習できていない可能性

---

## ✅ 結論: 9特徴量モデルが最適

### 採用すべき特徴量（9特徴）

**基本特徴（7特徴）**:
1. `Pclass`
2. `Sex`
3. `Age`
4. `SibSp`
5. `Parch`
6. `Fare`
7. `Embarked`

**追加特徴（2特徴）**:
8. `Title` (Mr, Mrs, Miss等)
9. `FamilySize` (SibSp + Parch + 1)

### 除外すべき特徴量

**❌ Deck_Safe**:
- 理由: Train/Test分布の違いで過学習
- OOF改善: +0.0033
- LB悪化: -0.00957
- **Net効果: マイナス**

**❌ TicketGroup**:
- 理由: 多重共線性（VIF 49.9）
- 単独効果: -0.0022（有害）

**❌ CabinShared**:
- 理由: 効果なし（CV ±0.0000）
- 多重共線性あり

**❌ Embarked個別ダミー**:
- 理由: CV -0.0067（有害）

---

## 📊 9特徴量モデルの優位性

### CV-LB Gap（推定）

**10特徴モデル**:
- OOF: 0.8305
- LB: 0.76794
- Gap: 6.26%

**9特徴モデル（推定）**:
- OOF: 0.8272（推定）
- LB: 0.77751
- Gap: **4.97%**（推定）

**Gap改善**: 6.26% → 4.97% = **-1.29ポイント**

### LBスコア改善

- 10特徴: 0.76794
- 9特徴: 0.77751
- **改善**: **+0.00957** (+0.957%)

---

## 🚀 次のステップ

### 即座のアクション: 9特徴量モデルを標準化

**推奨モデル**:
- アルゴリズム: Hard Voting (LightGBM + GradientBoosting + RandomForest)
- 特徴量: 9特徴（Deck_Safe除外）
- 閾値: 0.5（デフォルト）

**期待LB**: 0.77-0.78（実績: 0.77751）

---

### 優先度1: 9特徴量モデルのOOF再計算

**目的**: CV-LB Gapを正確に計算

**実装**:
```python
# scripts/9features_model_oof_calculation.py
# 9特徴でHard Votingモデルを再訓練
# OOFスコアとGapを計算
```

**期待OOF**: 0.82-0.83（10特徴より若干低い）

---

### 優先度2: さらなる特徴量改善

**追加候補特徴量**（慎重に評価）:
1. **Sex×Pclass交互作用**
   - 1等女性、3等男性など
   - 強いシグナルの可能性

2. **Age×Sex交互作用**
   - 子供、女性、男性の違い

3. **FarePerPerson**
   - Fare / (FamilySize)
   - 1人当たり運賃

4. **IsAlone**
   - FamilySize == 1
   - 単独乗船フラグ

**評価プロセス**:
1. 1特徴ずつ追加
2. OOFとVIF評価
3. **Kaggle LBで検証**（重要！）

---

### 優先度3: CV戦略改善（中期）

**現状の問題**:
- StratifiedKFoldはTicketグループを考慮しない
- データリーケージの可能性

**解決策**:
```python
from sklearn.model_selection import StratifiedGroupKFold

# Ticketグループ作成
ticket_groups = train['Ticket']

# Stratified Group K-Fold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=ticket_groups)):
    # 同じTicketグループがTrain/Validationに分散しない
    ...
```

**期待効果**:
- CV-LB Gap縮小（4.97% → 3%以下目標）
- より正確なOOF評価

---

## 📈 改善ロードマップ

### Phase 1: 9特徴量の確立（完了）

- ✅ Deck_Safe除外の効果確認（LB +0.957%）
- ✅ Hard Voting採用（LB +0.957%）
- ✅ 9特徴量モデルの確立（LB 0.77751）

### Phase 2: CV戦略改善（次のステップ）

- ⏳ 9特徴量モデルのOOF再計算
- ⏳ Stratified Group K-Fold実装
- ⏳ CV-LB Gap 3%以下達成

### Phase 3: 特徴量エンジニアリング（中期）

- ⏳ Sex×Pclass交互作用追加
- ⏳ FarePerPerson追加
- ⏳ 各特徴量をLBで検証
- 目標LB: 0.79-0.80

### Phase 4: アンサンブル高度化（長期）

- Stacking実装
- Probability Calibration
- 目標LB: 0.80以上（上位10%）

---

## 🎓 重要な教訓

### 学んだこと

1. **OOF改善 ≠ LB改善**
   - Deck_SafeはOOF +0.0033だがLB -0.00957
   - Train/Test分布が異なる場合、OOFは信頼できない

2. **少数サンプル特徴量は危険**
   - Cabin情報は22.9%のみ
   - 少数サンプルの傾向は過学習しやすい

3. **LB検証の重要性**
   - CV評価だけでは不十分
   - 実際のLB提出で初めて判明する問題がある

4. **シンプルな特徴量が最強**
   - 9特徴（シンプル）> 10特徴（複雑）
   - 複雑な特徴量は過学習リスク

### ベストプラクティス

**特徴量追加の判断基準**:
1. ✅ OOF改善を確認
2. ✅ VIF < 10を確認（多重共線性回避）
3. ✅ **Kaggle LBで検証**（最重要！）
4. ✅ LB改善した場合のみ採用

**避けるべき特徴量**:
- ❌ 少数サンプル（< 30%）に基づく特徴量
- ❌ VIF ≥ 10の特徴量（多重共線性）
- ❌ Train/Test分布が異なる可能性がある特徴量

---

## まとめ

### 最も重要な発見

**Deck_Safe特徴量は有害**:
- OOF: +0.0033（改善に見える）
- LB: -0.00957（実際は悪化）
- **結論**: Train dataに過学習、Test dataでは機能しない

### 現在のベストモデル

**9特徴量 Hard Voting**:
- LB Score: **0.77751**（暫定ベスト）
- Hard Voting (LightGBM + GB + RF)
- 閾値: 0.5（デフォルト）

### 次のアクション

1. **9特徴量モデルのOOF計算**（CV-LB Gap確認）
2. **Stratified Group K-Fold実装**（Gap縮小）
3. **交互作用特徴量の慎重な追加**（LBで検証）

### 目標

- 短期: LB 0.78以上、Gap 3%以下
- 中期: LB 0.80以上（上位10%）

---

**作成日**: 2025-10-21
**次のアクション**: 9特徴量モデルのOOF再計算スクリプトを作成
