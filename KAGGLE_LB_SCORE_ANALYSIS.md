# Kaggle LB Score分析レポート

**提出日**: 2025-10-21
**Submission**: `submission_threshold_optimal_Youden_ROC.csv`
**LB Score**: **0.75837**

---

## スコア分析

### CV-LB Gapの評価

| 指標 | 値 |
|------|-----|
| **OOF Score (CV)** | 0.8260 |
| **LB Score (Kaggle)** | 0.75837 |
| **Gap (CV - LB)** | **0.06763** (6.76%) |

### Gap評価

**Gap 6.76%は懸念レベル**

| Gap範囲 | 評価 | 本結果 |
|---------|------|--------|
| < 2% | 優秀（良好な汎化） | |
| 2-5% | 許容範囲 | |
| 5-8% | **要注意** | **✗ 該当** |
| > 8% | 深刻な過学習 | |

---

## 過去のSubmissionとの比較

### 過去の主要Submission推定

過去のレポートから、以下のSubmissionが存在:

| Submission | 推定CV/OOF | 推定LB | 備考 |
|-----------|----------|--------|------|
| Top 3 Hard Voting (9特徴) | 0.8350 | 0.80-0.82? | 過去最良（OOF） |
| LightGBM Tuned (9特徴) | 0.8317 | 0.79-0.81? | Gap 0.0090 |
| Top 3 Hard Voting (10特徴) | 0.8305 | ? | 本シリーズ |
| **Threshold Optimal** | **0.8260** | **0.75837** | **本提出** |

### 懸念点

1. **OOF 0.8260は他のモデルより低い**
   - Top 3 Hard Voting (9特徴): 0.8350
   - LightGBM Tuned: 0.8317
   - Top 3 Hard Voting (10特徴): 0.8305

2. **Soft VotingではなくHard Votingの方が良かった可能性**
   - Soft Voting OOF: 0.8249（最適化前）→ 0.8260（最適化後）
   - Hard Voting OOF: 0.8305（最適化なし）

---

## 問題の推定原因

### 1. Soft Voting vs Hard Voting

**使用したモデル**: Soft Voting
- OOF: 0.8249 → 0.8260（閾値最適化後）
- LB: 0.75837

**使用すべきだったモデル**: Hard Voting？
- OOF: 0.8305（閾値最適化なし）
- LB: 未検証

**問題**:
- Soft Votingを閾値最適化に使用
- Hard VotingはOOFが高いが確率値取得が難しい
- Soft VotingのOOF 0.8260 < Hard VotingのOOF 0.8305

### 2. 閾値最適化のリスク

**訓練データへの過剰適合**:
- OOFデータで閾値を最適化
- テストデータでは異なる分布の可能性
- 閾値0.4236が汎化しなかった可能性

**検証**:
- デフォルト閾値0.5でも提出すべきだった
- 比較でどちらが良いか確認

### 3. 特徴量の問題

**Deck_Safe特徴量**:
- OOFでは効果あり（+0.0045）
- テストデータでは効果が限定的？
- Cabin情報の分布がTrain/Testで異なる可能性

### 4. CV戦略の問題

**Stratified K-Fold使用**:
- Ticketグループを考慮せず
- 同じグループがTrain/Validationに分散
- データリーケージの可能性

---

## 提案される次のアクション

### 優先度1: 即座に検証（同日中）

#### 1. Hard Voting Submissionの提出

**ファイル**: `submission_top3_hardvoting_10features.csv`
- OOF: 0.8305（最も高い）
- 閾値最適化なし（デフォルト0.5）

**期待**:
- LB Score 0.76-0.78?
- Hard VotingがSoft Votingより良い可能性

#### 2. デフォルト閾値0.5のSubmission提出

**ファイル**: `submission_threshold_default_0.5.csv`
- 閾値最適化なし
- Soft Voting使用

**目的**:
- 閾値最適化の効果を検証
- 0.4236が過学習だったか確認

#### 3. 9特徴量の過去最良Submissionを特定

**確認すべき**:
- `submission_top3_hardvoting_lightgbm_gb_rf.csv`（過去のTop 3）
- `submission_lightgbm_tuned_9features.csv`

**理由**:
- 10特徴量（Deck_Safe追加）が悪影響だった可能性
- 9特徴量の方が汎化性能が高い？

### 優先度2: 短期改善（1-2日）

#### 4. Stratified Group K-Foldの実装

```python
from sklearn.model_selection import StratifiedGroupKFold

# Ticketグループでデータリーケージ防止
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in sgkf.split(X, y, groups=train['Ticket']):
    # より正確なCV評価
    pass
```

**期待効果**:
- CV-LB Gapの削減
- より信頼性の高いOOF Score

#### 5. Calibrationの実装

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(
    base_estimator=model,
    method='isotonic',
    cv=5
)
```

**期待効果**:
- 確率値のキャリブレーション
- より正確な閾値最適化

#### 6. Testデータの確率値分布確認

```python
# TestデータとTrainデータの確率値分布比較
import matplotlib.pyplot as plt

plt.hist(train_proba, bins=50, alpha=0.5, label='Train')
plt.hist(test_proba, bins=50, alpha=0.5, label='Test')
plt.legend()
```

**目的**:
- Train/Testの分布差を確認
- 閾値最適化の妥当性検証

### 優先度3: 中期改善（1週間）

#### 7. 相互作用特徴量の追加

```python
# EDAで効果的だった相互作用
train['Female_Class1'] = ((train['Sex'] == 1) & (train['Pclass'] == 1)).astype(int)
train['Male_Class3'] = ((train['Sex'] == 0) & (train['Pclass'] == 3)).astype(int)
```

**期待効果**: CV +0.3-0.7%

#### 8. Stackingアンサンブル

```python
from sklearn.ensemble import StackingClassifier

stacking = StackingClassifier(
    estimators=[('lgb', lgb), ('gb', gb), ('rf', rf)],
    final_estimator=LogisticRegression(),
    cv=StratifiedGroupKFold(...)  # グループ考慮
)
```

**期待効果**: CV 0.835-0.840

---

## デフォルト閾値vs最適閾値の比較計画

### 仮説

**仮説1**: 閾値最適化が過学習
- OOFで最適化した0.4236がテストで機能せず
- デフォルト0.5の方が汎化性能高い
- **検証**: `submission_threshold_default_0.5.csv` を提出

**仮説2**: Soft VotingがHard Votingより劣る
- Soft Voting OOF 0.8260 < Hard Voting OOF 0.8305
- Hard Votingの方が汎化性能高い
- **検証**: `submission_top3_hardvoting_10features.csv` を提出

**仮説3**: Deck_Safe特徴量がテストで効果なし
- Train/TestでCabin分布が異なる
- 10特徴より9特徴の方が良い
- **検証**: 9特徴量の過去Submissionと比較

### 検証手順

1. **即座に提出（優先）**:
   - `submission_threshold_default_0.5.csv`
   - `submission_top3_hardvoting_10features.csv`

2. **結果比較**:
   | Submission | OOF | LB | Gap | 評価 |
   |-----------|-----|-----|-----|------|
   | Threshold Optimal (0.4236) | 0.8260 | 0.75837 | 0.0676 | 既提出 |
   | Default (0.5) | 0.8249 | ? | ? | 要提出 |
   | Hard Voting | 0.8305 | ? | ? | 要提出 |

3. **分析**:
   - 最もGapが小さいモデルを特定
   - 最もLB Scoreが高いモデルを特定
   - 閾値最適化の効果を評価

---

## CV-LB Gapの原因仮説

### 可能性の高い原因

#### 1. データリーケージ（Ticket/Cabin grouping）

**問題**:
```python
# 現在のCV
cv = StratifiedKFold(n_splits=5)
# → 同じTicketグループがTrain/Validationに分散
```

**影響**:
- OOFスコアが過大評価
- 実際のテストデータではリーケージなし
- CV-LB Gapが拡大

**対策**: Stratified Group K-Fold

#### 2. Train/Test分布の差（Deck_Safe特徴量）

**仮説**:
```
Train: Cabin情報あり 22.9% (204/891)
Test:  Cabin情報あり ??% (不明)
```

**もしTestのCabin欠損率が異なると**:
- Deck_Safe特徴量の効果が異なる
- モデルが混乱

**検証**: Testデータの特徴量分布確認

#### 3. 閾値最適化の過学習

**問題**:
- OOF確率値分布で最適化
- テスト確率値分布が異なる
- 最適閾値0.4236が機能せず

**検証**: デフォルト0.5との比較

#### 4. Soft Votingの不安定性

**問題**:
- Soft Voting OOF 0.8260
- Hard Voting OOF 0.8305（より高い）
- Soft Votingの確率値平均が不安定？

**検証**: Hard Votingとの比較

### 可能性の低い原因

- 過学習（Gap 0.0112は低い）
- 特徴量選択ミス（VIF < 10で問題なし）
- モデル選択ミス（Top 3は実績あり）

---

## 推奨される即時アクション

### アクション1: Hard Voting提出（最優先）

```bash
# 既存ファイル
submission_top3_hardvoting_10features.csv
```

**理由**:
- OOF 0.8305（最も高い）
- Gap 0.0112（良好）
- 閾値最適化なし（安全）

**期待LB**: 0.76-0.78

### アクション2: デフォルト閾値提出

```bash
# 既存ファイル
submission_threshold_default_0.5.csv
```

**理由**:
- 閾値最適化の効果検証
- 過学習の有無確認

**期待LB**: 0.754-0.760（最適より低い可能性）

### アクション3: 9特徴量の最良Submission特定

**確認**:
```bash
ls -la submission_top3_*.csv
ls -la submission_lightgbm_*.csv
```

**目的**:
- 過去最良を再確認
- 10特徴 vs 9特徴の比較

---

## まとめ

### 現状評価

**LB Score 0.75837の評価**:
- ❌ 期待（0.79-0.81）より低い
- ❌ CV-LB Gap 6.76%は要注意レベル
- ⚠️ 閾値最適化が裏目に出た可能性

### 問題点

1. **Soft Voting使用**: Hard Votingの方が良かった可能性
2. **閾値最適化の過学習**: OOFに過剰適合
3. **データリーケージ**: Ticket groupingを考慮せず
4. **Deck_Safe効果**: Train/Testで分布が異なる可能性

### 次の優先アクション（順番に実施）

1. ✅ **Hard Voting提出** - `submission_top3_hardvoting_10features.csv`
2. ✅ **デフォルト閾値提出** - `submission_threshold_default_0.5.csv`
3. 📊 **結果比較と分析**
4. 🔧 **Stratified Group K-Fold実装**
5. 🚀 **改善版の開発と提出**

### 期待される改善

**Hard Voting提出での期待**:
- LB Score: 0.76-0.78
- Gap削減: 5%以下

**最終目標**:
- LB Score: **0.80+**
- CV-LB Gap: **< 3%**

---

**分析日**: 2025-10-21
**次のアクション**: Hard Voting と デフォルト閾値のSubmission提出
