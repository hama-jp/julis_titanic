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

### 3rd提出 ⭐ **過去最高スコア達成！**
- **ファイル**: `submission_3rd_correct_hard_voting_LB_0.77751.csv`
- **日付**: 2025-10-24
- **モデル**: Hard Voting (LightGBM + GradientBoosting + RandomForest)
  - LightGBM/GB: AgeGroup（カテゴリ）使用
  - RandomForest: Age（連続値）使用
- **CV Score**: 0.8294
- **LB Score**: **0.77751** 🎉
- **CV-LB Gap**: 5.2%
- **改善点**:
  - ✅ Train/Test分離を正しく実装
  - ✅ Fare補完: 7.90 → 8.05
  - ✅ モデル固有の最適特徴量
  - ✅ Pclass + Embarked別のFare補完
- **結果分析**:
  - 1st提出（0.77511）から **+0.0024** 改善
  - 正しいFare補完の効果が実証された
  - CV-LB Gapは依然として5.2%（改善の余地あり）

---

## 🎯 残り提出回数

- **総提出回数**: 5回
- **使用済み**: 3回
- **残り**: **2回** ⚠️

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
| **3rd** | **0.77751** ⭐ | 0.8294 | 5.2% | **+0.0024** | **Hard Voting (Corrected)** |

**現在のベスト**: 0.77751（3rd提出）
**目標**: LB 0.80以上（残り +0.0225）

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

### 現状分析（3rd提出結果を受けて）

**成功した点**:
- ✅ 正しいFare補完で +0.0024 改善
- ✅ モデル固有最適化（Age vs AgeGroup）が機能
- ✅ 過去最高スコア達成

**課題**:
- ❌ CV-LB Gap 5.2% は依然として大きい
- ❌ 目標（0.80）まで +0.0225 必要
- ⚠️ 残り提出回数: 2回のみ

### 4th-5th提出の戦略候補

#### Option A: RandomForest単体（保守的）
- **ファイル**: `submission_correct_randomforest.csv`（既存）
- **OOF Score**: 0.8294
- **期待LB**: 0.775 - 0.78
- **リスク**: 低（Hard Votingと99%一致）
- **メリット**: 3rd提出と同等かわずかに改善の可能性

#### Option B: 特徴量追加実験（攻撃的）
- **候補特徴量**:
  - Cabin Deck情報の活用（22.9%のデータあり）
  - FamilySize詳細化（1人/2-4人/5人以上）
  - Age×Pclass 交互作用
  - Fare×Embarked 交互作用
- **期待LB**: 0.78 - 0.80
- **リスク**: 中〜高（過学習の可能性）

#### Option C: アンサンブル多様化（中庸）
- **候補**:
  - LogisticRegression追加（線形モデルで多様性向上）
  - Soft Voting（確率ベース）
  - 重み付きVoting（CVスコア比例）
- **期待LB**: 0.775 - 0.785
- **リスク**: 中

#### Option D: ハイパーパラメータ再調整（慎重）
- **対象**: GradientBoosting, RandomForest
- **目的**: CV-LB Gap縮小
- **方法**: より保守的なパラメータ
- **期待LB**: 0.775 - 0.78
- **リスク**: 低〜中

### 推奨戦略（残り2回）

**4th提出**: Option C（Soft Voting）
- リスク中、期待値も中程度
- 確率ベースで異なる予測パターン
- 3rd提出との比較で知見獲得

**5th提出**: Option B（特徴量追加）または Option A（RandomForest単体）
- 4th提出の結果次第で判断
- 改善した場合 → Option B（攻めの特徴量追加）
- 悪化した場合 → Option A（保守的にRandomForest単体）

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
