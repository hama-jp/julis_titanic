# Hard Voting LB結果分析

**提出日**: 2025-10-21
**提出ファイル**: `submission_top3_hardvoting_10features.csv`
**LB Score**: **0.76794**

---

## 結果サマリー

| 提出 | モデル | OOF Score | LB Score | CV-LB Gap | Gap評価 |
|------|--------|-----------|----------|-----------|---------|
| 1st | Soft Voting + 閾値0.4236 | 0.8260 | 0.75837 | 0.06763 (6.76%) | ❌ 要注意 |
| 2nd | **Hard Voting + 閾値0.5** | **0.8305** | **0.76794** | **0.06256 (6.26%)** | ⚠️ 要注意 |

### 改善度

- **LB Score改善**: 0.75837 → 0.76794 = **+0.00957** (+0.957%)
- **Gap改善**: 6.76% → 6.26% = **-0.50ポイント**

---

## ✅ 検証結果: Hard Voting > Soft Voting

### 仮説1の検証結果: **部分的に正しい**

**仮説**: Soft VotingよりHard Votingの方が汎化性能が高い

**検証結果**:
- ✅ Hard Voting LB (0.76794) > Soft Voting LB (0.75837)
- ✅ LB Scoreで **+0.957%の改善**
- ✅ CV-LB Gapも改善（6.76% → 6.26%）

**結論**: Hard Votingは確かにSoft Votingより優れている

---

## ⚠️ 残る問題: 依然として大きなCV-LB Gap

### Gap分析

| 指標 | 値 | 評価 |
|------|-----|------|
| OOF Score | 0.8305 | 優秀 |
| LB Score | 0.76794 | 普通 |
| **CV-LB Gap** | **0.06256** | **6.26%（要注意レベル）** |

**Gap評価基準**:
- 優秀: < 3%
- 許容範囲: 3-5%
- 要注意: 5-8% ← **現在ここ**
- 深刻: > 8%

### 問題の本質

Hard Votingに変更してもGapが6.26%と依然大きい理由:

1. **閾値最適化だけが原因ではない**
   - Soft→Hardで改善したが、Gapは残存
   - 他の構造的問題が存在

2. **考えられる主要因**:
   - ❌ ~~Soft Voting vs Hard Voting~~ → ✅ **解決済み**
   - ⚠️ **特徴量の汎化性能不足**（Deck_Safe等）
   - ⚠️ **データリーケージ**（StratifiedKFoldの問題）
   - ⚠️ **Train/Test分布の違い**

---

## 🔍 詳細分析: 各モデルのGap比較

### OOF vs LB Gap（推定）

Hard Votingのアンサンブル構成モデルのOOFスコア:

| モデル | OOF Score | OOF Gap | 推定LB | 推定LB Gap |
|--------|-----------|---------|--------|-----------|
| LightGBM（強正則化） | 0.8248 | 0.0191 | 0.75-0.77? | 5-7%? |
| GradientBoosting（超強正則化） | 0.8146 | 0.0135 | 0.74-0.76? | 5-7%? |
| RandomForest（ベースライン） | 0.8180 | 0.0022 | 0.77-0.79? | 3-5%? |
| **Hard Voting** | **0.8305** | **0.0112** | **0.76794** | **6.26%** |

**観察**:
- Hard VotingのOOF Gap: 0.0112（優秀）
- しかしCV-LB Gap: 6.26%（要注意）
- **OOF Gapが小さくてもLB Gapは大きい** = Train/Testの分布が異なる

---

## 🎯 次のアクション: 根本原因の特定

### 優先度1: Deck_Safe特徴量の検証

**仮説**: Deck_Safe特徴量がTrain/Testで効果が異なる

**検証方法**: 9特徴量モデル（Deck_Safe除外）を提出

**提出ファイル**: `submission_top3_hardvoting_lightgbm_gb_rf.csv`

**期待結果**:
- **シナリオA**: 9特徴LB ≥ 0.77 → Deck_Safeが悪影響
- **シナリオB**: 9特徴LB < 0.76 → Deck_Safeは有効、他に原因

---

### 優先度2: 閾値最適化の再検証（念のため）

**仮説**: 閾値0.4236が依然として問題

**検証方法**: Soft Voting + デフォルト閾値0.5を提出

**提出ファイル**: `submission_threshold_default_0.5.csv`

**期待結果**:
- デフォルト0.5のLB > 閾値最適化0.4236のLB → 閾値最適化が過学習
- デフォルト0.5のLB ≈ 閾値最適化0.4236のLB → 閾値最適化は問題ない

---

### 優先度3（長期改善）: CV戦略の見直し

**問題**: StratifiedKFoldがTicketグループを考慮していない

**解決策**: Stratified Group K-Fold実装

**実装内容**:
```python
from sklearn.model_selection import StratifiedGroupKFold

# Ticketグループを考慮したCV分割
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=ticket_groups)):
    # 同じTicketグループがTrain/Validationに分散しない
    ...
```

**期待効果**:
- データリーケージ防止
- より正確なCV評価
- CV-LB Gapの縮小

---

## 📊 現時点でのベストモデル

### LBスコアランキング（判明分）

| 順位 | 提出ファイル | LB Score | CV Score | Gap |
|------|------------|----------|----------|-----|
| 🥇 | submission_top3_hardvoting_10features.csv | **0.76794** | 0.8305 | 6.26% |
| 🥈 | submission_threshold_optimal_Youden_ROC.csv | 0.75837 | 0.8260 | 6.76% |
| ❓ | submission_top3_hardvoting_lightgbm_gb_rf.csv (9特徴) | ??? | 不明 | ??? |
| ❓ | submission_threshold_default_0.5.csv | ??? | 0.8249 | ??? |

**暫定ベスト**: Hard Voting 10特徴（LB 0.76794）

---

## 🚀 推奨される次の提出

### 提出1: 9特徴量Hard Voting（最優先）

**ファイル**: `submission_top3_hardvoting_lightgbm_gb_rf.csv`

**目的**: Deck_Safe特徴量の効果検証

**期待LB**:
- 良好シナリオ: **0.77-0.79**（Deck_Safeが逆効果だった場合）
- 中立シナリオ: **0.76-0.77**（Deck_Safeは効果なし）
- 悪化シナリオ: **< 0.76**（Deck_Safeは有効だった）

**判断基準**:
- 9特徴LB ≥ 0.77 → Deck_Safe除外を継続、特徴量改善へ
- 9特徴LB < 0.76 → 10特徴維持、CV戦略改善へ

---

### 提出2: Soft Voting + デフォルト閾値0.5（参考）

**ファイル**: `submission_threshold_default_0.5.csv`

**目的**: 閾値最適化の効果再検証

**期待LB**: 0.754-0.76

**判断基準**:
- デフォルト0.5 > 閾値0.4236 → 閾値最適化が過学習
- デフォルト0.5 ≈ 閾値0.4236 → Soft Voting自体が問題

---

## 📈 目標と現状

### 短期目標（現在の改善サイクル）

| 目標 | 現状 | 達成度 |
|------|------|--------|
| LB Score 0.78以上 | 0.76794 | ❌ 未達成（-1.2%） |
| CV-LB Gap 3%以下 | 6.26% | ❌ 未達成（+3.26%） |
| Hard Voting採用 | ✅ 採用済み | ✅ 達成 |

### 中期目標

| 目標 | アクション |
|------|-----------|
| LB 0.80以上 | 特徴量エンジニアリング改善 |
| Gap 2%以下 | Stratified Group K-Fold実装 |
| 上位10%入り | Stacking/Calibration導入 |

---

## 🎓 学んだこと

### ✅ 検証できたこと

1. **Hard Voting > Soft Voting**
   - LB Score: +0.957%改善
   - Gap: -0.50ポイント改善
   - 今後はHard Votingを標準採用

2. **閾値最適化は副次的問題**
   - Hard Voting（閾値0.5）でも6.26% Gap残存
   - 主原因は他にある

### ⚠️ 未解決の問題

1. **6.26%のCV-LB Gap**
   - 依然として要注意レベル
   - 特徴量またはCV戦略に問題

2. **Train/Test分布の違い**
   - OOF Gapは小さい（1.12%）のにLB Gapは大きい（6.26%）
   - Train内のCVは正確だがTestとは異なる

### 🔜 次の重点課題

1. **Deck_Safe特徴量の検証**（優先度最高）
2. **CV戦略の改善**（中期改善）
3. **新しい特徴量の追加**（長期改善）

---

## まとめ

### 現状評価

**良かった点**:
- ✅ Hard VotingがSoft Votingより優れていることを確認
- ✅ LB Scoreを0.76794まで改善（前回+0.957%）
- ✅ CV-LB Gapも6.76%→6.26%に改善

**課題**:
- ❌ CV-LB Gap 6.26%は依然として大きい
- ❌ LB 0.78目標に未達（-1.2%）
- ❌ 根本原因（特徴量 or CV戦略）未特定

### 次のステップ

**即座に実施**:
1. 9特徴量Hard Voting提出 → Deck_Safe効果検証
2. 結果に応じて方針決定:
   - 9特徴LB > 10特徴LB → 特徴量改善路線
   - 9特徴LB < 10特徴LB → CV戦略改善路線

**中期実施**:
- Stratified Group K-Fold実装
- 交互作用特徴量追加（Sex×Pclass等）
- Probability Calibration導入

---

**作成日**: 2025-10-21
**次のアクション**: `submission_top3_hardvoting_lightgbm_gb_rf.csv`（9特徴量）を提出してDeck_Safe効果を検証
