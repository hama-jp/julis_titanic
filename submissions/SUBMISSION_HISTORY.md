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

### 3rd提出（予定）
- **ファイル**: `submission_correct_hard_voting.csv`
- **日付**: 2025-10-24（未提出）
- **モデル**: Hard Voting (LightGBM + GradientBoosting + RandomForest)
  - LightGBM/GB: AgeGroup（カテゴリ）使用
  - RandomForest: Age（連続値）使用
- **CV Score**: 0.8294
- **期待LB Score**: 0.775 - 0.78
- **改善点**:
  - ✅ Train/Test分離を正しく実装
  - ✅ Fare補完: 7.90 → 8.05
  - ✅ モデル固有の最適特徴量
  - ✅ Pclass + Embarked別のFare補完

---

## 🎯 残り提出回数

- **総提出回数**: 5回
- **使用済み**: 2回
- **残り**: **3回**

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

| 提出 | LB Score | CV Score | Gap | モデル |
|------|----------|----------|-----|--------|
| 1st | 0.77511 | 0.8294 | 5.4% | Hard Voting |
| 2nd | 0.76555 | 0.8232 | 5.8% | GB (GroupKFold) |
| 3rd | ? | 0.8294 | ? | Hard Voting (Corrected) |

**目標**: LB 0.80以上

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

1. **3rd提出を実行**
   - `submission_correct_hard_voting.csv`をKaggleに提出
   - LBスコアを確認
   - 0.775-0.78を期待

2. **4th-5th提出の戦略検討**（3rd提出結果次第）
   - Option A: RandomForest単体（OOF 0.8294）
   - Option B: Soft Voting
   - Option C: 特徴量追加実験

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
