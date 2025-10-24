# モデル毎に最適な特徴量を使用するパイプライン

**日付**: 2025-10-24
**目的**: 年齢特徴量の比較実験の結果を活かし、モデル毎に最適な特徴量セットを使用するパイプラインを構築

## 背景

年齢特徴量の比較実験（20251024_age_feature_comparison）から以下の知見を得た：
- **LightGBM**: 年齢層ラベル（AgeGroup）が最適
  - 理由: 浅い木構造、安定性向上、過学習抑制
- **Random Forest**: 年齢の生値（Age）が最適
  - 理由: 深い木構造で連続値の細かい境界を学習可能

この知見を活かし、各モデルに最適な特徴量を割り当てるパイプラインを実装した。

## 仮説

モデル毎に最適な特徴量セットを使用することで：
1. 各モデルの性能を最大化できる
2. アンサンブル時の多様性が向上する
3. より安定した予測が可能になる

## 実装内容

### 特徴量エンジニアリング

全ての特徴量を作成し、モデル毎に選択する方式を採用：

**基本特徴量（全モデル共通）:**
- Pclass, Sex, Title, IsAlone, FareBin, SmallFamily
- Age_Missing, Embarked_Missing

**年齢関連特徴量（モデル毎に選択）:**
- Age: 年齢の生値
- AgeGroup: 年齢層ラベル（0=Child (0-18), 1=Adult (18-60), 2=Elderly (60-100)）

### モデル別特徴量セット

| モデル              | 年齢特徴量 | 理由                                    |
|---------------------|------------|-----------------------------------------|
| LightGBM            | AgeGroup   | 浅い木（max_depth=2）でカテゴリが有効  |
| Random Forest       | Age        | 深い木（max_depth=5）で連続値が有効    |
| Gradient Boosting   | AgeGroup   | 浅い木（max_depth=3）でカテゴリが有効  |
| Logistic Regression | AgeGroup   | 線形モデルでカテゴリが解釈しやすい      |

### モデルパラメータ

**LightGBM:**
```python
LGBMClassifier(
    n_estimators=100, max_depth=2, num_leaves=4,
    min_child_samples=30, reg_alpha=1.0, reg_lambda=1.0,
    min_split_gain=0.01, learning_rate=0.05
)
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=100, max_depth=5,
    min_samples_split=10, min_samples_leaf=5
)
```

**Gradient Boosting:**
```python
GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_samples_split=20, min_samples_leaf=10, subsample=0.8
)
```

**Logistic Regression:**
```python
LogisticRegression(C=0.5, penalty='l2', max_iter=1000)
```

## 結果

### 個別モデルの性能

| モデル              | 年齢特徴量 | CV Score | CV Std   | Train Acc | Gap      |
|---------------------|-----------|----------|----------|-----------|----------|
| RandomForest        | Age       | 0.83277  | 0.01088  | 0.84961   | 0.01683  |
| LightGBM            | AgeGroup  | 0.82827  | 0.00931  | 0.83165   | 0.00338  |
| GradientBoosting    | AgeGroup  | 0.82715  | 0.01107  | 0.84961   | 0.02246  |
| LogisticRegression  | AgeGroup  | 0.81033  | 0.01384  | 0.81369   | 0.00336  |

**最良モデル**: RandomForest with Age（CV: 0.83277）

### アンサンブル手法

3つのアンサンブル手法を実装：

1. **Hard Voting (Majority Vote)**
   - 全4モデルの予測を多数決
   - Survival Rate: 0.349

2. **Soft Voting (Average Probabilities)**
   - 予測確率の平均を使用
   - Survival Rate: 0.380

3. **Weighted Voting (CV Score Based)**
   - CVスコアに基づいた重み付け
   - Weights: LGB=0.252, RF=0.251, GB=0.251, LR=0.246
   - Survival Rate: 0.354

### 生成ファイル

**個別モデル:**
- submissions/20251024_0215_lightgbm_modelspecific.csv
- submissions/20251024_0215_randomforest_modelspecific.csv
- submissions/20251024_0215_gradientboosting_modelspecific.csv
- submissions/20251024_0215_logisticregression_modelspecific.csv

**アンサンブル:**
- submissions/20251024_0215_ensemble_hard_voting_modelspecific.csv
- submissions/20251024_0215_ensemble_soft_voting_modelspecific.csv
- submissions/20251024_0215_ensemble_weighted_modelspecific.csv

## 考察

### うまくいった点

1. **モデル特性に応じた特徴量選択**:
   - Random Forestが年齢の生値で最高精度（0.83277）を達成
   - LightGBMが年齢層ラベルで過学習を抑制（Gap: 0.00338）
   - 年齢特徴量実験の結果が実践で再現された

2. **過学習の制御**:
   - LightGBMとLogistic Regressionで非常に低い過学習（Gap < 0.004）
   - 年齢層ラベルによる正則化効果が確認された

3. **モデル多様性の向上**:
   - 異なる特徴量を使用することで、各モデルが異なる視点から学習
   - アンサンブルの効果が期待できる構成

### 課題と改善点

1. **Gradient Boostingの過学習**:
   - Gap: 0.02246と他モデルより高い
   - パラメータ調整の余地あり（learning_rateをさらに下げる、subsampleを調整）

2. **Logistic Regressionの精度**:
   - CV: 0.81033と他モデルより低い
   - 非線形パターンを捉えられない限界
   - アンサンブルの多様性には貢献

3. **アンサンブルの最適化**:
   - 現在は単純な重み付け
   - スタッキングやブレンディングでさらなる改善の可能性

### 学んだこと

1. **モデル固有の最適化の重要性**:
   - 全モデルに同じ特徴量を使うのではなく、各モデルの特性に合わせる
   - 木の深さに応じて連続値/カテゴリを使い分ける

2. **過学習と精度のトレードオフ**:
   - RandomForest: 高精度だが過学習あり（Gap: 0.01683）
   - LightGBM: やや精度低いが安定（Gap: 0.00338）
   - アンサンブルで両方の長所を活かせる

3. **実験結果の実践応用**:
   - 小規模な実験（年齢特徴量比較）の知見が大規模パイプラインに活かせた
   - 仮説検証→実装のサイクルが効果的

## 次のアクション

- [x] モデル毎に最適な特徴量を使用するパイプラインの実装
- [ ] アンサンブルサブミッションをKaggleに提出し、LBスコアを確認
- [ ] 他の特徴量（Fare, Title等）でもモデル固有の最適化を検証
- [ ] スタッキング/ブレンディングによる高度なアンサンブル手法を試す
- [ ] GradientBoostingのパラメータチューニングで過学習を抑制

## 関連ファイル

- スクリプト: `scripts/models/model_specific_features_pipeline.py`
- 実験ログ: `experiments/logs/20251024_model_specific_features_pipeline.md`
- 関連実験: `experiments/logs/20251024_age_feature_comparison.md`
- サブミッションファイル: `submissions/20251024_0215_*.csv` (7ファイル)
