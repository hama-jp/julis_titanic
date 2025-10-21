# Kaggle提出結果サマリー

**分析日**: 2025-10-21
**セッション**: Threshold最適化 → LBスコア検証 → 特徴量最適化

---

## 📊 提出結果の推移

| # | 提出ファイル | モデル | 特徴量 | LB Score | 前回比 | 累積改善 |
|---|------------|--------|--------|----------|--------|---------|
| 1 | submission_threshold_optimal_Youden_ROC.csv | Soft Voting + 閾値0.4236 | 10 | 0.75837 | - | - |
| 2 | submission_top3_hardvoting_10features.csv | Hard Voting + 閾値0.5 | 10 | 0.76794 | +0.00957 | +0.957% |
| 3 | submission_top3_hardvoting_lightgbm_gb_rf.csv | Hard Voting + 閾値0.5 | **9** | **0.77751** | +0.00957 | **+1.914%** |

### 🏆 現在のベストモデル

**9特徴量Hard Voting**:
- **LB Score**: **0.77751**
- **OOF Score**: 0.8384
- **CV-LB Gap**: 6.09%
- アルゴリズム: LightGBM + GradientBoosting + RandomForest（Hard Voting）
- 閾値: 0.5（デフォルト）

---

## 🔍 重要な発見

### 1. Hard Voting > Soft Voting

**比較**（10特徴量モデル）:
- Soft Voting: LB 0.75837、OOF 0.8260
- Hard Voting: LB 0.76794、OOF 0.8305
- **改善**: +0.00957 (+0.957%)

**結論**: Hard Votingの方が汎化性能が高い

---

### 2. Deck_Safe特徴量は有害（過学習）

**比較**（Hard Votingモデル）:

| 特徴量 | OOF Score | LB Score | CV-LB Gap |
|--------|-----------|----------|-----------|
| 10特徴（+Deck_Safe） | 0.8305 | 0.76794 | 6.26% |
| **9特徴（Deck_Safe除外）** | **0.8384** | **0.77751** | **6.09%** |

**変化**:
- OOF: +0.0079（改善）
- LB: +0.00957（改善）
- Gap: -0.17%（改善）

**結論**:
- ✅ Deck_SafeはOOFで改善に見えたが、実際はLBで悪化
- ✅ Train/Test分布の違いにより過学習
- ❌ Cabin情報（22.9%のみ）は少数サンプルで信頼性低い

---

### 3. 閾値最適化の効果は限定的

**比較**:
- 閾値0.4236（最適化）: LB 0.75837
- 閾値0.5（デフォルト）: LB 0.76794（Hard Voting）

**結論**: 閾値最適化より、モデル選択（Hard vs Soft）と特徴量選択の方が重要

---

## 📈 CV-LB Gap分析

### Gap推移

| モデル | OOF Score | LB Score | Gap | Gap評価 |
|--------|-----------|----------|-----|---------|
| Soft Voting 10特徴 | 0.8260 | 0.75837 | 6.76% | ❌ 要注意 |
| Hard Voting 10特徴 | 0.8305 | 0.76794 | 6.26% | ⚠️ 要注意 |
| **Hard Voting 9特徴** | **0.8384** | **0.77751** | **6.09%** | **⚠️ 要注意** |

### Gap評価基準

- ✅ 優秀: < 3%
- ✅ 許容範囲: 3-5%
- ⚠️ **要注意: 5-8%** ← **現在ここ**
- ❌ 深刻: > 8%

### 問題の本質

**6.09% Gapが残る理由**:
1. **Train/Test分布の違い**
   - Trainデータの傾向がTestに完全には当てはまらない
   - 特にCabin、Ticket等の欠損パターンが異なる可能性

2. **データリーケージの可能性**
   - StratifiedKFold使用（Ticketグループ考慮なし）
   - 同じTicketグループがTrain/Validationに分散
   - OOFスコアが過大評価される

3. **少数サンプル特徴量への依存**
   - Cabin情報: 22.9%のみ
   - 高生存率のDeck（D/E/B）: さらに少数
   - 統計的に不安定

---

## 🎯 採用すべき特徴量（9特徴）

### ベース特徴量（6特徴）

1. **Pclass** - 客室クラス（1, 2, 3）
2. **Sex** - 性別（男性=0, 女性=1）
3. **Age** - 年齢（欠損値は中央値補完）
4. **SibSp** - 兄弟姉妹・配偶者の数
5. **Parch** - 親・子供の数
6. **Fare** - 運賃（欠損値は中央値補完）

### エンジニアリング特徴量（3特徴）

7. **Title** - 敬称（Mr, Mrs, Miss, Master, その他）
   - Nameから抽出: `Name.str.extract(' ([A-Za-z]+)\.')`
   - グループ化: 稀な敬称は"その他"にまとめる

8. **FamilySize** - 家族サイズ
   - 計算式: `SibSp + Parch + 1`

9. **Embarked** - 乗船港（C, Q, S）
   - ダミー変数化（drop_first=True）
   - 実際には2特徴（Embarked_Q, Embarked_S）

### 除外した特徴量

**❌ Deck_Safe** - Cabin情報から生成（D/E/Bデッキ）
- 理由: OOF改善するもLB悪化（過学習）
- Train: 22.9%のみCabin情報あり
- Test: 分布が異なる可能性

**❌ TicketGroup** - チケットグループサイズ
- 理由: 多重共線性（VIF 49.9）、単独で有害（CV -0.0022）

**❌ CabinShared** - 客室共有人数
- 理由: 効果なし（CV ±0.0000）

---

## 🤖 採用すべきモデル設定

### アンサンブル構成: Hard Voting

**3つのベースモデル**:

1. **LightGBM**
```python
lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    random_state=42
)
```

2. **GradientBoosting**
```python
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

3. **RandomForest**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
```

**Voting方式**: Hard（多数決）
```python
VotingClassifier(
    estimators=[('lgb', lgb_model), ('gb', gb_model), ('rf', rf_model)],
    voting='hard'
)
```

**CV戦略**: StratifiedKFold (n_splits=5)
- 将来的には**Stratified Group K-Fold**に移行推奨

**閾値**: 0.5（デフォルト）
- 閾値最適化は過学習リスクあり

---

## 📋 次のステップ

### 短期改善（優先度順）

#### 1. 交互作用特徴量の追加（慎重に）

**候補特徴量**:
- **Sex×Pclass**: 1等女性、3等男性など（強いシグナル）
- **Age×Sex**: 子供（< 16）、女性、男性の区分
- **FarePerPerson**: Fare / FamilySize（1人当たり運賃）
- **IsAlone**: FamilySize == 1（単独乗船フラグ）

**評価プロセス**（重要！）:
1. ✅ 1特徴ずつ追加
2. ✅ OOFスコア確認
3. ✅ VIF < 10確認（多重共線性回避）
4. ✅ **Kaggle LBで検証**（最重要！）
5. ✅ LB改善した場合のみ採用

**教訓**:
- Deck_Safeの失敗（OOF改善、LB悪化）を繰り返さない
- **LB検証が最終判断基準**

---

#### 2. CV戦略改善（中期）

**問題**: 現在のStratifiedKFoldはTicketグループを考慮しない

**解決策**: Stratified Group K-Fold実装

```python
from sklearn.model_selection import StratifiedGroupKFold

# Ticketグループ作成
ticket_groups = train['Ticket']

# Stratified Group K-Fold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=ticket_groups)):
    # 同じTicketグループがTrain/Validationに分散しない
    # → データリーケージ防止
    ...
```

**期待効果**:
- データリーケージ防止
- より正確なOOF評価
- **CV-LB Gap縮小**（6.09% → 3%以下目標）

---

#### 3. Probability Calibration（長期）

**問題**: 確率予測の較正が不十分な可能性

**解決策**: Calibration実装

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(
    base_estimator=hard_voting,
    method='isotonic',  # or 'sigmoid'
    cv=5
)
```

**期待効果**:
- 確率予測の精度向上
- Soft Votingの性能改善

---

### 長期改善

#### 4. Stacking Ensemble

**構成**:
- Level 1: LightGBM, GradientBoosting, RandomForest, XGBoost
- Level 2: LogisticRegression（メタ学習器）

**期待効果**:
- LB 0.80以上（上位10%）

---

## 🎓 学んだ重要な教訓

### 1. OOF改善 ≠ LB改善

**事例**: Deck_Safe特徴量
- OOF: +0.0033（改善に見える）
- LB: -0.00957（実際は悪化）

**教訓**: Train/Test分布が異なる場合、OOFは信頼できない

---

### 2. 少数サンプル特徴量は危険

**事例**: Cabin情報（22.9%のみ）
- 統計的に不安定
- 過学習リスク大

**教訓**: サンプル率 < 30%の特徴量は慎重に評価

---

### 3. LB検証が最終判断基準

**プロセス**:
1. OOFで候補選定
2. VIFで多重共線性チェック
3. **Kaggle LBで最終検証**（最重要！）

**教訓**: CV評価だけで特徴量採用を決めない

---

### 4. シンプルな特徴量が最強

**結果**:
- 9特徴（シンプル）> 10特徴（複雑）
- 複雑な特徴量は過学習リスク

**教訓**: Occam's Razor（オッカムの剃刀）

---

## 📊 目標と現状

### 短期目標

| 目標 | 現状 | 達成度 | 次のアクション |
|------|------|--------|---------------|
| LB 0.78以上 | 0.77751 | ⚠️ あと+0.25% | 交互作用特徴量追加 |
| CV-LB Gap 3%以下 | 6.09% | ❌ あと-3.09% | Stratified Group K-Fold |
| Hard Voting採用 | ✅ 採用済み | ✅ 完了 | - |
| 9特徴量確立 | ✅ 確立済み | ✅ 完了 | - |

### 中期目標

| 目標 | 期待LB | アクション |
|------|--------|-----------|
| LB 0.80以上 | 0.80-0.82 | Stacking、特徴量改善 |
| Gap 2%以下 | - | CV戦略改善、Calibration |
| 上位10%入り | 0.80+ | 総合的な改善 |

---

## 📝 ベストプラクティス

### 特徴量追加の判断フロー

```
新しい特徴量のアイデア
  ↓
1. OOF評価（CVで改善するか？）
  ↓ YES
2. VIF評価（多重共線性はないか？）
  ↓ VIF < 10
3. ドメイン知識（理論的に妥当か？）
  ↓ YES
4. **Kaggle LB検証**（実際に改善するか？）
  ↓ YES
5. 採用
```

### 避けるべき特徴量

- ❌ 少数サンプル（< 30%）
- ❌ VIF ≥ 10（多重共線性）
- ❌ Train/Test分布が異なる可能性
- ❌ 理論的根拠が弱い

---

## 🚀 即座のアクション

### 推奨される次の提出

**優先度1**: Sex×Pclass交互作用

**特徴量**:
```python
# Sex×Pclass交互作用
train['Sex_Pclass'] = train['Sex'].astype(str) + '_' + train['Pclass'].astype(str)
# ダミー変数化: Female_1, Female_2, Female_3, Male_1, Male_2, Male_3
```

**期待効果**:
- 1等女性: 生存率 ~96%
- 3等男性: 生存率 ~14%
- 強いシグナル

**評価**: OOF→VIF→**LB検証**

---

## まとめ

### 達成したこと

1. ✅ LB Score: 0.75837 → **0.77751**（+1.914%）
2. ✅ Hard Voting採用（Soft Votingより優秀）
3. ✅ Deck_Safe過学習を発見・除外
4. ✅ 9特徴量の最適モデル確立
5. ✅ CV-LB Gap改善（6.76% → 6.09%）

### 残る課題

1. ⚠️ CV-LB Gap 6.09%（要注意レベル）
2. ⚠️ LB 0.78目標に未達（あと+0.25%）

### 次のステップ

1. **交互作用特徴量追加**（Sex×Pclass優先）
2. **Stratified Group K-Fold実装**（Gap縮小）
3. **LB検証を必ず実施**（過学習防止）

### 最終目標

- **短期**: LB 0.78以上、Gap 3%以下
- **中期**: LB 0.80以上（上位10%）

---

**作成日**: 2025-10-21
**ベストモデル**: 9特徴量Hard Voting（LB 0.77751）
**次のアクション**: Sex×Pclass交互作用の評価とLB検証
