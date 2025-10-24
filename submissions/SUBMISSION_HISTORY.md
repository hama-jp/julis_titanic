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

### 5th提出（最終）の戦略分析 ⚠️ 過去の実験結果を反映

#### ❌ 過去にテストして失敗した特徴量（使用禁止）

**以下の特徴量は過去の実験で効果がないことが実証済み**:

1. **Cabin Deck (Deck_Safe)** ← **最も危険**
   - 実験結果: OOF +0.0033, **LB -0.00957**
   - 理由: Train/Test分布の不一致（Trainの22.9%のみCabin情報あり）
   - 結論: Trainに過学習、Testで性能悪化
   - 参照: `docs/analysis/DECK_SAFE_FEATURE_ANALYSIS.md`

2. **Sex × Pclass 交互作用**
   - 実験結果: OOF -0.0023, 期待LB -0.00227
   - VIF: 999999（極度の多重共線性）
   - 結論: ベースライン（9特徴）より悪化
   - 参照: `experiments/results/sex_pclass_interaction_evaluation.txt`

3. **FarePerPerson** (Fare / FamilySize)
   - 実験結果: OOF -0.0034
   - 結論: ベースラインより悪化

**重要**: これらの特徴量は絶対に使用しないこと！

#### ✅ 過去に効果が認められた特徴量

1. **IsAlone** (FamilySize == 1)
   - 実験結果: OOF +0.0011
   - 現状: **未使用**（検討価値あり）
   - 参照: `experiments/results/simple_features_evaluation.txt`

2. **モデル固有の最適化**
   - LightGBM/GB: AgeGroup（カテゴリ）
   - RandomForest: Age（連続値）
   - 実験結果: 各モデルで最適なAge表現が異なる
   - 現状: 4th提出で使用済み

3. **Soft Voting**
   - Hard Votingより +0.00717 改善
   - CV-LB Gap: 5.2% → 3.9% に縮小
   - 現状: 4th提出で使用済み

---

#### 戦略A: 保守的アプローチ（推奨） ⭐⭐⭐

**現実的な判断**:
- 目標（LB 0.80）まで **+0.01532** 必要
- 過去の特徴量追加実験はほぼ全て失敗
- 残り1回のみ → 安全策を取るべき

**推奨候補1**: `submission_correct_randomforest.csv`
- OOF Score: 0.8294
- 期待LB: **0.78 - 0.785**
- 特徴:
  - RandomForest単体（深い木でAge連続値を活用）
  - 正しい補完パイプライン使用
  - 過学習リスク低
- リスク: **低**
- 改善期待: 4thと同等～微増

**推奨候補2**: `20251024_0215_ensemble_soft_voting_modelspecific.csv`
- モデル固有最適化版のSoft Voting
- 期待LB: **0.78 - 0.785**
- リスク: **低**

**判断理由**:
- ✅ 既存の検証済みファイル
- ✅ 失敗特徴量を含まない
- ✅ 正しい補完パイプライン使用
- ✅ 4thの0.78468を下回るリスクが低い
- ❌ 0.80到達は難しいが、現実的な範囲内

---

#### 戦略B: 慎重に特徴量追加（リスク中）

**最後のチャンスで0.80を狙うなら**:

**追加候補1**: IsAlone特徴量（OOF +0.0011実証済み）
- 現在の9特徴 + IsAlone
- 期待OOF: 0.8395
- 期待LB: 0.785 - 0.790（Gap 5%想定）
- リスク: 中

**追加候補2**: Age × Pclass 交互作用（未テスト）
- 理論的根拠: 子供の1等客と3等客で生存率が大きく異なる
- **注意**: 多重共線性の可能性あり（Sex×Pclass失敗の前例）
- リスク: 中〜高

**実装方針**（戦略Bを選ぶ場合）:
1. Soft Voting（確率ベース）をベースに
2. IsAlone特徴量のみ追加（最小限のリスク）
3. 正しい補完パイプライン使用
4. モデル固有最適化（Age vs AgeGroup）維持
5. CVで厳密に検証（Gap < 5%、OOF > 0.835）

**リスク**:
- 新特徴量が4thより悪化させる可能性
- 目標0.80到達の可能性は依然として低い（+0.005程度の改善が限界）

---

#### 戦略C: 超保守的（非推奨）

**候補**: 4th提出と同じファイルを再提出
- `submission_4th_soft_voting_LB_0.78468.csv`
- 確実に0.78468を再現
- リスク: 最低
- **問題**: 最後の1回を無駄にする

---

### 🎯 推奨戦略（最終決定）⭐

**推奨**: **戦略A - 保守的アプローチ（RandomForest単体）**

**理由**:
1. **現実的な期待値**: LB 0.78 - 0.785（4thと同等～微増）
2. **低リスク**: 失敗した特徴量を含まない
3. **未使用のポテンシャル**: RandomForest単体は未提出
4. **過去の教訓**: 特徴量追加はほぼ全て失敗している
5. **最後の1回**: 悪化させるリスクを最小化

**次善策**: 戦略B（IsAlone追加）
- 0.80を狙うならIsAlone追加を慎重に実装
- ただし、目標到達の可能性は低い（+0.005程度の改善が限界）

**目標の現実**:
- LB 0.80到達は、現在の特徴量セットでは困難
- 4th提出の0.78468は既に優秀な結果
- 最後の1回で改悪するより、同等～微増を狙うべき

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
