# 最終アンサンブルモデル完成レポート

**日付**: 2025-10-24
**実行内容**: Ticket特徴量抽出 → モデル別最適化 → Optuna → 10シードアンサンブル

---

## エグゼクティブサマリー

✅ **最終OOFスコア**: **0.8467** (LightGBM, 10-seed average)
✅ **前回ベストからの改善**: 0.8361 → 0.8467 (+0.0106, +1.27%)
✅ **提出ファイル**: 7種類生成（推奨: ensemble_weighted_4models.csv）
✅ **安定性**: 非常に高い（σ = 0.0021）

---

## 1. 開発プロセスの総括

### Phase 1: Ticket特徴量の発見と実装
**実施内容**:
- Ticketカラムのパターン分析（39種類のプレフィックス発見）
- 6つの有効な特徴量を実装:
  1. `Ticket_GroupSize`: チケット共有人数
  2. `Ticket_Fare_Per_Person`: 一人あたり運賃 (Fare / GroupSize)
  3. `Ticket_Is_Numeric`: 数字のみのチケットか
  4. `Ticket_Prefix_Category`: 高/中/低生存率カテゴリ
  5. `Ticket_Length`: チケット文字列の長さ
  6. `Ticket_SmallGroup`: グループサイズ2-4のフラグ

**重要な発見**:
- グループサイズ2-4で生存率が高い（51-73%）
- `Ticket_Fare_Per_Person`が既存のFareより優れている（+1.35%）
- Tree-basedモデルで+1.08%の改善（LogisticRegressionでは-1.53%）

### Phase 2: 特徴量冗長性の排除
**ユーザーからの洞察**:
- 「SibSp, Parch, Ticket_GroupSizeは冗長では？」
- 「FareとTicket_Fare_Per_Personは多重共線性では？」

**分析結果**:
- SibSp/Parchの重要度は極めて低い（0.002-0.010）
- FamilySize + Ticket_GroupSizeの組み合わせが最良（0.8451）
- Ticket_Fare_Per_Person単独 > Fare単独（0.8429 vs 0.8294）
- **推奨**: Fareを削除、Ticket_Fare_Per_Personのみ使用

### Phase 3: モデル別最適特徴量セットの決定
**各モデルの最適構成**:

| モデル | 特徴量数 | 主要な特徴量 | 理由 |
|--------|----------|--------------|------|
| **GradientBoosting** | 12 | FamilySize + SmallFamily + Ticket_GroupSize | 最良スコア 0.8451 |
| **RandomForest** | 10 | Ticket_GroupSize中心、家族特徴量なし | Ticket_GroupSizeで十分 |
| **LightGBM** | 11 | Ticket_GroupSize + Ticket_SmallGroup | グループフラグが有効 |
| **LogisticRegression** | 11 | SibSp + Parch + Fare（Ticket特徴量なし） | 多重共線性回避 |

---

## 2. Optuna ハイパーパラメータ最適化結果

### 最適化設定
- **GradientBoosting**: 100 trials, TPE sampler
- **RandomForest**: 100 trials, TPE sampler
- **LightGBM**: 100 trials, TPE sampler
- **LogisticRegression**: 50 trials, TPE sampler

### 最良パラメータ

**GradientBoosting** (OOF: 0.8496):
```python
{
  'n_estimators': 215,
  'max_depth': 5,
  'learning_rate': 0.0380,
  'subsample': 0.9525,
  'min_samples_split': 12,
  'min_samples_leaf': 2
}
```

**RandomForest** (OOF: 0.8485):
```python
{
  'n_estimators': 239,
  'max_depth': 6,
  'min_samples_split': 9,
  'min_samples_leaf': 1,
  'max_features': None  # すべての特徴量を使用
}
```

**LightGBM** (OOF: 0.8507):
```python
{
  'n_estimators': 194,
  'max_depth': 6,
  'learning_rate': 0.0831,
  'num_leaves': 56,
  'subsample': 0.9021,
  'colsample_bytree': 0.8550,
  'min_child_samples': 25
}
```

**LogisticRegression** (OOF: 0.7721):
```python
{
  'C': 0.9847,
  'penalty': 'l1',
  'solver': 'saga',  # l1正則化サポートのため
  'max_iter': 1000
}
```

---

## 3. 10シードアンサンブル結果

### Seeds使用
`[42, 123, 456, 789, 1234, 5678, 9012, 3456, 7890, 2468]`

### 最終スコア（10シード平均）

| 順位 | モデル | OOF Score | 標準偏差 | 安定性 |
|------|--------|-----------|----------|--------|
| **1** | **LightGBM** | **0.8467** | **±0.0021** | **★★★★★** |
| 2 | RandomForest | 0.8448 | ±0.0020 | ★★★★★ |
| 3 | GradientBoosting | 0.8443 | ±0.0028 | ★★★★☆ |
| 4 | LogisticRegression | 0.7721 | ±0.0000 | ★★★☆☆ |

**観察**:
- LightGBMが最高スコアかつ最も安定
- RandomForestも非常に安定（σ=0.0020）
- GradientBoostingはやや分散が大きい（σ=0.0028）
- LogisticRegressionは全seedで同一スコア（決定論的）

---

## 4. 生成された提出ファイル

### 7つの提出戦略

| # | ファイル名 | 説明 | 推奨度 |
|---|-----------|------|--------|
| **1** | **ensemble_weighted_4models.csv** | **OOFスコアで重み付けアンサンブル** | **★★★★★** |
| 2 | lig_10seeds.csv | LightGBM単独（最高OOF） | ★★★★☆ |
| 3 | ensemble_top3_models.csv | Top3モデル平均（LGB+RF+GB） | ★★★★☆ |
| 4 | ensemble_equal_4models.csv | 4モデル等重みアンサンブル | ★★★☆☆ |
| 5 | gra_10seeds.csv | GradientBoosting単独 | ★★☆☆☆ |
| 6 | ran_10seeds.csv | RandomForest単独 | ★★☆☆☆ |
| 7 | log_10seeds.csv | LogisticRegression単独 | ★☆☆☆☆ |

### 重み付けアンサンブルの重み
```python
{
  'LightGBM': 0.2560,          # 25.60%
  'RandomForest': 0.2554,       # 25.54%
  'GradientBoosting': 0.2552,   # 25.52%
  'LogisticRegression': 0.2334  # 23.34%
}
```

---

## 5. 前回提出との比較

### スコア推移

| 提出 | 日付 | CV Score | LB Score | 改善 | 手法 |
|------|------|----------|----------|------|------|
| 1st | - | - | 0.76555 | - | Basic |
| 2nd | - | - | 0.77033 | +0.00478 | Feature Eng |
| 3rd | - | - | 0.76794 | -0.00239 | Worse |
| 4th | - | - | 0.77990 | +0.01196 | Better |
| **5th** | **2025-10-24** | **0.8361** | **0.77511** | **-0.00479** | **Optuna 4-model ensemble** |
| **6th (今回)** | **2025-10-24** | **0.8467** | **TBD** | **TBD** | **Optuna + 10-seed + Ticket features** |

**期待**:
- CV改善: 0.8361 → 0.8467 (+0.0106)
- LB期待: 0.77511 → 0.780-0.785（推定）

---

## 6. 技術的改善点

### 解決した問題
1. **LogisticRegression solver bug**: penalty='l1'でsolver='lbfgs'を使用していた問題を修正（solver='saga'に変更）
2. **多重共線性**: Ticket特徴量をLogisticRegressionから除外
3. **特徴量冗長性**: SibSp/Parchを削除、FamilySize + Ticket_GroupSizeに集約
4. **Fareの代替**: Ticket_Fare_Per_Personの方が優れていることを発見

### コードの改善
- モデル別特徴量セットの実装
- Optunaパラメータの保存/ロード機能
- 10シードアンサンブルの自動化
- 7種類の提出ファイル自動生成

---

## 7. 重要な洞察

### Ticket特徴量の価値
- **Ticket_Fare_Per_Person**: 2番目に重要な特徴量（GradientBoostingで重要度0.139）
- **理由**: 同じ運賃でもグループサイズで意味が変わる
  - $100, 1人 → $100/人 → 富裕層単独旅行 → 高生存率
  - $100, 10人 → $10/人 → 貧困層家族 → 低生存率
- Fareだけではこの区別が不可能

### Sexの重要度が低い理由
- Titleが性別情報を包含（相関0.9989）
- Title = Sex + 婚姻状態 + 年齢グループ
- Titleの方が情報量が多い（Mr/Miss/Mrs/Masterの区別）

### 家族特徴量の最適化
- SibSp/Parchは重要度が極めて低い
- FamilySize（= SibSp + Parch + 1）で十分
- Ticket_GroupSizeと組み合わせることで補完的な情報を提供
  - 78%は一致（家族でチケット共有）
  - 22%は異なる（友人とチケット共有、または家族が複数チケット）

---

## 8. 次のステップ（推奨）

### 短期（Kaggle提出）
1. **優先提出**: `ensemble_weighted_4models.csv`
2. **代替案1**: `lig_10seeds.csv`（LightGBM単独、最高OOF）
3. **代替案2**: `ensemble_top3_models.csv`（LogisticRegression除外）

### 中期（さらなる改善）
1. **特徴量の交互作用項**:
   - Ticket_Fare_Per_Person × Pclass
   - Ticket_GroupSize × Sex
   - Title × FamilySize

2. **アンサンブルの高度化**:
   - Stacking（メタ学習器）
   - Blending（holdout set）
   - Optuna for ensemble weights

3. **外部データ**:
   - 歴史的な乗客リスト
   - 客室配置図

### 長期（モデルの深化）
1. **ニューラルネットワーク**:
   - TabNet
   - Entity Embedding
   - AutoML (H2O, AutoGluon)

2. **特徴量選択の最適化**:
   - Boruta
   - SHAP feature importance
   - Permutation importance

---

## 9. ファイル構成

```
/home/user/julis_titanic/
├── scripts/
│   ├── feature_engineering/
│   │   ├── ticket_pattern_analysis.py
│   │   ├── ticket_feature_evaluation.py
│   │   └── ticket_feature_importance_analysis.py
│   ├── analysis/
│   │   ├── fare_vs_ticket_fare_per_person.py
│   │   ├── family_vs_ticket_group_comparison.py
│   │   └── sex_title_correlation_check.py
│   └── models/
│       └── final_optimized_ensemble.py ✓ (最終版)
├── experiments/results/
│   ├── TICKET_FEATURES_ANALYSIS.md
│   ├── FARE_VS_FAREPP_DEEP_DIVE.md
│   ├── FAMILY_VS_TICKET_GROUP_ANALYSIS.md
│   ├── final_optuna_params.json ✓ (最適パラメータ)
│   └── FINAL_ENSEMBLE_SUMMARY.md ✓ (本ドキュメント)
└── submissions/
    ├── 20251024_0716_ensemble_weighted_4models.csv ✓ 推奨
    ├── 20251024_0716_lig_10seeds.csv ✓
    ├── 20251024_0716_ensemble_top3_models.csv ✓
    ├── 20251024_0716_ensemble_equal_4models.csv
    ├── 20251024_0716_gra_10seeds.csv
    ├── 20251024_0716_ran_10seeds.csv
    └── 20251024_0716_log_10seeds.csv
```

---

## 10. 結論

### 達成したこと
✅ Ticket特徴量の発見と実装（6種類）
✅ 特徴量の冗長性排除（SibSp/Parch → FamilySize, Fare → Ticket_Fare_Per_Person）
✅ モデル別最適特徴量セットの決定
✅ Optunaハイパーパラメータ最適化（100 trials）
✅ 10シードアンサンブルで分散低減
✅ LogisticRegressionのsolverバグ修正
✅ 7種類の提出ファイル生成
✅ **CV Score 0.8467達成**（前回0.8361から+1.27%改善）

### 今回のキー成果
- **LightGBM**: 0.8467 ± 0.0021（非常に安定）
- **Ticket_Fare_Per_Person**: ゲームチェンジャー特徴量
- **モデル別最適化**: 各モデルに最適な特徴量セット
- **バグ修正**: LogisticRegressionのsolver問題を解決

### 期待されるKaggle順位への影響
- 現在のLB: 0.77511（前回提出）
- 期待LB: 0.780-0.785（推定+0.5-1.0%改善）
- Top 10%圏内への可能性

---

**生成日**: 2025-10-24
**実行時間**: 約20分（Optuna 10分 + 10-seed ensemble 10分）
**総特徴量数**: 10-12（モデル別）
**訓練データサイズ**: 891行

🎉 **最終提出ファイル準備完了！Kaggleへの提出をお待ちしています。**
