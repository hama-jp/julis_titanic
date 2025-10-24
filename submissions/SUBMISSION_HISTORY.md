# Kaggle提出履歴

このドキュメントは、Kaggle Titanicコンペティションへの提出履歴を記録します。

---

## 📊 提出履歴

### 1st提出
- **ファイル**: `submission_1st_hard_voting_LB_0.77511.csv`
- **日付**: 2025-10-24
- **モデル**: Hard Voting (LightGBM + GradientBoosting + RandomForest)
- **CV Score**: 0.8294
- **LB Score**: **0.77511**
- **CV-LB Gap**: 5.4%
- **備考**: Train/Test分離が不適切（Fare補完誤り）

### 2nd提出
- **ファイル**: `submission_2nd_groupkfold_gb_LB_0.76555.csv`
- **日付**: 2025-10-24
- **モデル**: GradientBoosting (StratifiedGroupKFold使用)
- **CV Score**: 0.8232
- **LB Score**: **0.76555**
- **CV-LB Gap**: 5.8%
- **備考**:
  - GroupKFoldで正確なCV評価を試みたが、LBは悪化
  - Train/Test分離が不適切（Fare補完誤り）

### 3rd提出
- **ファイル**: `submission_3rd_correct_hard_voting_LB_0.77751.csv`
- **日付**: 2025-10-24
- **モデル**: Hard Voting (LightGBM + GradientBoosting + RandomForest)
  - LightGBM/GB: AgeGroup（カテゴリ）使用
  - RandomForest: Age（連続値）使用
- **CV Score**: 0.8294
- **LB Score**: **0.77751**
- **CV-LB Gap**: 5.2%
- **改善点**:
  - ✅ Train/Test分離を正しく実装
  - ✅ Fare補完: 7.90 → 8.05
  - ✅ モデル固有の最適特徴量
  - ✅ Pclass + Embarked別のFare補完
- **結果分析**:
  - 1st提出（0.77511）から **+0.0024** 改善
  - 正しいFare補完の効果が実証された

### 4th提出 🎉🎉🎉 **大幅改善達成！新記録！**
- **ファイル**: `submission_4th_soft_voting_LB_0.78468.csv`
- **日付**: 2025-10-24
- **モデル**: Soft Voting (LightGBM + GradientBoosting + RandomForest)
  - 確率の平均による柔軟な予測
  - LightGBM/GB: AgeGroup（カテゴリ）使用
  - RandomForest: Age（連続値）使用
- **CV Score**: 0.8249
- **LB Score**: **0.78468** 🎉🎉🎉
- **CV-LB Gap**: 3.9% ← **大幅に改善！**
- **改善幅**:
  - 3rd提出（0.77751）から **+0.00717** ⭐
  - 1st提出（0.77511）から **+0.00957**
  - これまでで最大の改善幅
- **重要な発見**:
  - ✅ **Soft Voting > Hard Voting** が実証された！
  - ✅ 確率ベースの柔軟な判定が効果的
  - ✅ CV-LB Gap が 5.2% → 3.9% に縮小
  - ✅ 目標（0.80）まで残り **+0.01532**

---

## 🎯 残り提出回数

- **総提出回数**: 5回
- **使用済み**: 4回
- **残り**: **1回** ⚠️⚠️⚠️ **（最後のチャンス！）**

---

## 📁 ファイル構成

### 現在の提出候補（submissions/）
- `submission_correct_hard_voting.csv` ⭐ 3rd提出推奨
- `submission_correct_randomforest.csv`
- `submission_correct_lightgbm.csv`
- `submission_correct_gradientboosting.csv`
- `submission_correct_soft_voting.csv`

### モデル固有最適化版（submissions/）
- `20251024_0215_ensemble_hard_voting_modelspecific.csv`
- `20251024_0215_ensemble_soft_voting_modelspecific.csv`
- `20251024_0215_randomforest_modelspecific.csv`
- `20251024_0215_lightgbm_modelspecific.csv`
- `20251024_0215_gradientboosting_modelspecific.csv`

### 過去の提出（submissions/）
- `submission_1st_hard_voting_LB_0.77511.csv` - 1st提出
- `submission_2nd_groupkfold_gb_LB_0.76555.csv` - 2nd提出

### アーカイブ（submissions/archive/）
- 41ファイル（過去の実験結果）

---

## 📈 スコア推移

| 提出 | LB Score | CV Score | Gap | 差分 | モデル |
|------|----------|----------|-----|------|--------|
| 1st | 0.77511 | 0.8294 | 5.4% | - | Hard Voting（Fare補完誤り） |
| 2nd | 0.76555 | 0.8232 | 5.8% | -0.0096 | GB (GroupKFold, Fare補完誤り) |
| 3rd | 0.77751 | 0.8294 | 5.2% | +0.0024 | Hard Voting (Corrected) |
| **4th** | **0.78468** 🎉🎉🎉 | 0.8249 | **3.9%** | **+0.00717** ⭐ | **Soft Voting** |

**現在のベスト**: 0.78468（4th提出）
**目標**: LB 0.80以上（残り **+0.01532**）
**残り提出**: **1回のみ**

---

## 🔧 既知の問題と改善

### 1st & 2nd提出の問題点
- ❌ Train/Test分離が不適切
  - Test Fare欠損をTestデータのみから補完
  - Fare=7.90（誤）← 正しくは8.05
- ❌ CV-LB Gapが大きい（5.4-5.8%）
  - 過学習の可能性
  - Train/Test分布の違い

### 3rd提出での改善
- ✅ 正しいTrain/Test分離
  - `scripts/preprocessing/correct_imputation_pipeline.py`使用
  - Trainから学習 → Train/Testに適用
- ✅ モデル固有の最適特徴量
  - LightGBM/GB: AgeGroup（カテゴリ）
  - RandomForest: Age（連続値）
- ✅ Fare補完の改善
  - Pclass + Embarked別の中央値

---

## 📝 次のアクション

### 現状分析（4th提出結果を受けて）⚠️ **最後の1回！**

**大成功した点**:
- ✅ **Soft Voting が大成功！** +0.00717（最大の改善幅）
- ✅ CV-LB Gap が 5.2% → **3.9%** に大幅縮小
- ✅ 確率ベースのアンサンブルの有効性を実証
- ✅ 現在のベスト: **0.78468**

**分析**:
- Soft Voting（確率平均）> Hard Voting（多数決）
- CV Score 0.8249 でも LB 0.78468 を達成
- Gap縮小により、CVスコアとLBスコアの相関が改善

**残された課題**:
- 目標（0.80）まで **+0.01532** 必要
- ⚠️⚠️⚠️ **残り提出回数: 1回のみ（最後のチャンス）**

### 5th提出（最終）の戦略分析

#### 戦略A: 特徴量追加で攻める（推奨） ⭐

**アプローチ**:
1. Cabin Deck情報を活用（22.9%のデータあり）
   - DeckをA-G, Unknown にカテゴリ化
   - Deck × Pclass 交互作用
2. 家族サイズの詳細化
   - Alone (1人) / Small (2-4人) / Large (5人以上)
3. Age × Pclass 交互作用
   - 子供の1等客 vs 3等客など

**期待効果**:
- CV Score: 0.825 - 0.835
- 期待LB: **0.79 - 0.805** ← 目標達成の可能性
- リスク: 中〜高（過学習の可能性あり）

**判断理由**:
- ✅ Soft Votingの成功で基盤が確立
- ✅ CV-LB Gapが縮小（3.9%）→ 特徴量の効果が出やすい
- ✅ 最後の1回なので攻める価値あり

#### 戦略B: 保守的に既存ファイル（非推奨）

**候補1**: RandomForest単体
- ファイル: `submission_correct_randomforest.csv`
- OOF Score: 0.8294
- 期待LB: 0.78 - 0.785（4thと同等か微増）
- リスク: 低

**候補2**: モデル固有最適化版のSoft Voting
- ファイル: `20251024_0215_ensemble_soft_voting_modelspecific.csv`
- 期待LB: 0.78 - 0.785
- リスク: 低

**判断理由**:
- ❌ 4th提出（0.78468）を大きく超える可能性は低い
- ❌ 目標（0.80）到達は困難
- ❌ 最後の1回を消極的に使うのはもったいない

### 🎯 推奨戦略（最終決定）

**5th提出（最終）**: **戦略A - 特徴量追加で攻める**

**実装方針**:
1. Soft Voting（確率ベース）をベースに
2. 以下の特徴量を追加:
   - Cabin Deck（A-G, Unknown）
   - Deck × Pclass 交互作用
   - Family Size詳細化（Alone/Small/Large）
   - Age × Pclass 交互作用
3. 正しい補完パイプライン使用
4. モデル固有最適化（Age vs AgeGroup）維持

**目標**:
- **LB 0.79 - 0.805** を目指す
- 0.80到達で目標達成！

**リスク対策**:
- CVでの過学習チェック（Gap < 5%）
- 特徴量は1つずつ追加して効果を検証
- CVスコアが0.83以上を維持

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
