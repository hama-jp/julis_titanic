# 次のKaggle提出計画

**作成日**: 2025-10-21
**目的**: LB Score 0.75837の原因を特定し、改善策を検証

---

## 現状の問題

| 提出ファイル | OOF Score | LB Score | CV-LB Gap | 評価 |
|------------|----------|----------|-----------|------|
| submission_threshold_optimal_Youden_ROC.csv | 0.8260 | **0.75837** | **0.06763** (6.76%) | ❌ 要注意レベル |

**問題点**:
- CV-LB Gap 6.76%は過学習の兆候
- 期待LB 0.79-0.81に対して大幅に低い
- Soft Voting + 閾値最適化の組み合わせが裏目に出た可能性

---

## 提出優先順位

### 🔴 **優先度1: Hard Voting（最重要）**

**ファイル**: `submission_top3_hardvoting_10features.csv`

**理由**:
- OOF 0.8305（全モデル中最高）
- Gap 0.0112（優秀レベル）
- 閾値最適化なし（デフォルト0.5で安全）
- Hard VotingはSoft Votingより汎化性能が高い可能性

**期待結果**:
- LB Score: **0.76-0.78**（保守的）～ **0.79-0.82**（楽観的）
- Gap: 0.02-0.05（許容範囲）

**検証内容**: Soft Voting vs Hard Voting の性能差

---

### 🟡 **優先度2: デフォルト閾値0.5**

**ファイル**: `submission_threshold_default_0.5.csv`

**理由**:
- 閾値最適化の効果を検証
- 閾値0.4236が過学習だったかを確認
- Soft Voting使用（確率値から生成）

**期待結果**:
- LB Score: **0.754-0.760**（閾値最適化より悪い可能性）
- または **0.77-0.78**（閾値最適化が過学習だった場合）

**検証内容**: 閾値最適化(0.4236) vs デフォルト(0.5)

---

### 🟢 **優先度3: 過去の9特徴量モデル（参考）**

**ファイル**: `submission_top3_hardvoting_lightgbm_gb_rf.csv`

**理由**:
- 9特徴量（Deck_Safe追加前）
- Deck_Safe特徴量の効果を検証
- Train/Testで分布が異なる可能性

**期待結果**:
- LB Score: **0.77-0.80?**（過去レポートから推定）

**検証内容**: 10特徴 vs 9特徴（Deck_Safe効果）

---

## 提出順序と分析手順

### ステップ1: Hard Voting提出（最優先）

```bash
# Kaggleに提出
submission_top3_hardvoting_10features.csv
```

**期待**: LB 0.76以上なら成功、0.78以上なら大成功

---

### ステップ2: 結果分析（Hard Voting提出後）

**シナリオA: Hard Voting LB ≥ 0.77**
- ✅ **結論**: Soft VotingよりHard Votingが優れている
- ✅ **対策**: 今後はHard Votingを使用
- 次のアクション: Hard Votingベースで特徴量改善

**シナリオB: Hard Voting LB 0.75-0.77**
- ⚠️ **結論**: Hard Votingも改善は限定的
- ⚠️ **原因**: 閾値最適化以外に問題がある可能性
- 次のアクション: デフォルト閾値0.5を提出して検証

**シナリオC: Hard Voting LB < 0.75**
- ❌ **結論**: モデル全体に構造的問題
- ❌ **原因**: データリーケージ、特徴量選択ミス等
- 次のアクション: CV戦略見直し（Stratified Group K-Fold）

---

### ステップ3: デフォルト閾値提出（Hard Voting結果次第）

**条件**: Hard Voting LB < 0.77の場合に提出

```bash
# Kaggleに提出
submission_threshold_default_0.5.csv
```

**分析**:
| 閾値 | LB Score | 評価 |
|------|----------|------|
| 0.4236（最適化） | 0.75837 | 既知 |
| 0.5（デフォルト） | ? | 要検証 |

**期待**: もしデフォルト0.5の方が良ければ、閾値最適化が過学習だった証拠

---

### ステップ4: 9特徴量提出（参考検証）

**条件**: 時間がある場合、またはHard Voting/デフォルト閾値の両方が改善しなかった場合

```bash
# Kaggleに提出
submission_top3_hardvoting_lightgbm_gb_rf.csv
```

**分析**: 9特徴と10特徴の比較でDeck_Safe効果を検証

---

## 根本原因の仮説と検証計画

### 仮説1: Soft VotingがHard Votingより劣る

| 指標 | Soft Voting | Hard Voting |
|------|------------|-------------|
| OOF Score | 0.8260 | 0.8305 |
| Gap | 0.0157 | 0.0112 |

**検証**: Hard Voting提出（優先度1）
**期待**: Hard VotingのLB > Soft VotingのLB

---

### 仮説2: 閾値最適化が過学習

**問題**:
- OOFデータで閾値0.4236を最適化
- Testデータの確率分布が異なる可能性
- 最適閾値が汎化しない

**検証**: デフォルト閾値0.5提出（優先度2）
**期待**: デフォルト0.5のLB ≥ 最適化0.4236のLB

---

### 仮説3: Deck_Safe特徴量がTestで効果なし

**問題**:
- Train: Cabin情報あり 22.9% (204/891)
- Test: Cabin情報あり ??%（分布不明）
- Train/Testで分布が異なればDeck_Safeが機能しない

**検証**: 9特徴量提出（優先度3）
**期待**: 9特徴のLB ≥ 10特徴のLB

---

### 仮説4: データリーケージ（Ticketグループ）

**問題**:
- StratifiedKFold使用
- 同じTicketグループがTrain/Validationに分散
- OOFスコアが過大評価

**検証**: Stratified Group K-Fold実装（長期改善）

---

## 最終目標

### 短期目標（今回の提出）

- LB Score: **0.78以上**
- CV-LB Gap: **3%以下**

### 中期目標（次の改善サイクル）

- LB Score: **0.80以上**
- CV-LB Gap: **2%以下**
- Stratified Group K-Fold実装
- Stacking/Calibration導入

---

## まとめ

### 即座に実施すべきアクション

1. ✅ **Hard Voting提出** - `submission_top3_hardvoting_10features.csv`
   - 最も期待値が高い
   - OOF 0.8305、Gap 0.0112

2. ⏳ **結果待ち** - Hard VotingのLBスコア確認

3. 📊 **分析と次の提出決定**
   - LB ≥ 0.77: 成功、Hard Votingベースで改善継続
   - LB < 0.77: デフォルト閾値0.5を追加検証

### 期待される改善

**最良シナリオ**:
- Hard Voting LB: **0.79-0.82**
- Gap: 2-3%
- 問題解決: Soft Voting → Hard Votingへの切り替え

**最悪シナリオ**:
- Hard Voting LB: **0.75-0.76**
- 根本的見直し必要: CV戦略、特徴量選択、アンサンブル手法

---

**作成日**: 2025-10-21
**次のアクション**: `submission_top3_hardvoting_10features.csv` をKaggleに提出
