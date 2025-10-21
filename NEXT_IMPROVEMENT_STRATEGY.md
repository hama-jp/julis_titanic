# 次の改善戦略

**作成日**: 2025-10-21
**現状**: LB 0.77751（9特徴量Hard Voting）
**目標**: LB 0.78以上

---

## 📊 現状分析

### 試したアプローチと結果

| アプローチ | 試行回数 | LB改善 | 結論 |
|-----------|---------|--------|------|
| **Voting方式変更**（Soft→Hard） | 1回 | +0.957% | ✅ 成功 |
| **特徴量削除**（Deck_Safe除外） | 1回 | +0.957% | ✅ 成功 |
| **特徴量追加** | 5回 | すべて失敗 | ❌ 限界到達 |
| **閾値最適化** | 1回 | -0.957% | ❌ 失敗 |

**結論**:
- ✅ モデル選択（Hard Voting）は成功
- ✅ 特徴量削減は成功
- ❌ 特徴量追加は限界
- ❌ 閾値最適化は過学習

---

## 🎯 残された改善アプローチ

### アプローチ1: XGBoost追加アンサンブル

**現在の構成**:
- LightGBM + GradientBoosting + RandomForest（3モデル）

**提案**:
- **XGBoostを追加** → 4モデルのアンサンブル

**理論的根拠**:
- XGBoostは異なるアルゴリズム（Column Sampling, Regularization等）
- モデルの多様性向上 → アンサンブル精度向上
- Kaggleで高い実績

**期待効果**:
- LB +0.003-0.005（+0.3-0.5%）
- LB 0.7805-0.7825

**実装難易度**: ⭐（簡単）

**リスク**:
- 低（9特徴量は変更しない）
- XGBoostが他モデルと相関高ければ効果なし

---

### アプローチ2: Stacking Ensemble

**構成**:
- **Level 1**: LightGBM, GradientBoosting, RandomForest, XGBoost
- **Level 2**: LogisticRegression（メタ学習器）

**仕組み**:
1. Level 1モデルで予測確率を生成
2. Level 2モデルがこれらを組み合わせる最適重みを学習
3. より洗練されたアンサンブル

**期待効果**:
- LB +0.005-0.010（+0.5-1.0%）
- LB 0.7825-0.7875

**実装難易度**: ⭐⭐⭐（やや複雑）

**リスク**:
- 中（過学習の可能性）
- CV戦略が重要（Stratified K-Fold）

---

### アプローチ3: ハイパーパラメータ再最適化

**現状**:
- LightGBM, GB, RFは固定パラメータ使用
- 9特徴量に最適化されていない

**提案**:
- **Optuna**で9特徴量に特化したパラメータ最適化

**最適化対象**:
- LightGBM: max_depth, num_leaves, learning_rate, reg_alpha, reg_lambda
- GradientBoosting: max_depth, learning_rate, min_samples_split
- RandomForest: max_depth, min_samples_split, min_samples_leaf

**期待効果**:
- LB +0.002-0.005（+0.2-0.5%）
- LB 0.7795-0.7825

**実装難易度**: ⭐⭐（中）

**リスク**:
- 中（過学習の可能性）
- 計算時間がかかる

---

### アプローチ4: Stratified Group K-Fold

**目的**:
- CV-LB Gap縮小（6.09% → 3%以下）
- より正確なOOF評価

**実装**:
```python
from sklearn.model_selection import StratifiedGroupKFold

sgkf = StratifiedGroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=ticket_groups)):
    # 同じTicketグループがTrain/Validationに分散しない
    ...
```

**期待効果**:
- **LB改善は不確実**（CVの改善が主目的）
- Gap縮小で特徴量評価が正確に
- 将来的な改善の基盤

**実装難易度**: ⭐⭐（中）

**リスク**:
- 低（LBが悪化する可能性は低い）
- ただし、LB改善も保証されない

---

### アプローチ5: Probability Calibration

**問題**:
- Hard Votingは確率を使わない
- Soft Votingは較正されていない

**提案**:
- 各モデルの確率を較正
- Calibrated Soft Voting

**実装**:
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_lgb = CalibratedClassifierCV(lgb_model, method='isotonic', cv=5)
calibrated_gb = CalibratedClassifierCV(gb_model, method='isotonic', cv=5)
calibrated_rf = CalibratedClassifierCV(rf_model, method='isotonic', cv=5)

soft_voting = VotingClassifier(
    estimators=[('lgb', calibrated_lgb), ('gb', calibrated_gb), ('rf', calibrated_rf)],
    voting='soft'
)
```

**期待効果**:
- LB +0.002-0.005（+0.2-0.5%）
- LB 0.7795-0.7825

**実装難易度**: ⭐⭐（中）

**リスク**:
- 中（Soft Votingが過去に失敗）
- Calibrationで改善するかは不明

---

## 🏆 推奨される優先順位

### Priority 1: XGBoost追加アンサンブル（最優先）

**理由**:
- ✅ 実装が簡単（既存コードに数行追加）
- ✅ リスクが低い（9特徴量は変更しない）
- ✅ 効果が期待できる（モデル多様性向上）
- ✅ 即座にLB検証可能

**実装時間**: 30分

**期待LB**: 0.7805-0.7825

---

### Priority 2: Stacking Ensemble（中期）

**理由**:
- ✅ 高い効果が期待できる
- ⚠️ 実装がやや複雑
- ✅ Kaggleで実績あり

**実装時間**: 1-2時間

**期待LB**: 0.7825-0.7875

---

### Priority 3: ハイパーパラメータ再最適化（並行可能）

**理由**:
- ✅ 9特徴量に特化した最適化
- ⚠️ 計算時間がかかる
- ⚠️ 効果は不確実

**実装時間**: 2-3時間（計算時間含む）

**期待LB**: 0.7795-0.7825

---

### Priority 4: Stratified Group K-Fold（長期基盤）

**理由**:
- ✅ CV精度向上
- ✅ 将来的な改善の基盤
- ❌ LB改善は不確実

**実装時間**: 1時間

**期待効果**: Gap縮小（LB改善は副次的）

---

## 📋 具体的なアクションプラン

### Step 1: XGBoost追加（今すぐ実施）

**実装内容**:
1. XGBoostをインストール
2. 9特徴量でXGBoostモデルを訓練
3. 4モデルのHard Votingを構築
4. OOF評価
5. Kaggle LB提出

**判断基準**:
- OOF > 0.8384 かつ LB > 0.77751 → 採用
- それ以外 → 不採用

---

### Step 2: 結果に応じた次のステップ

**XGBoost成功時**:
- Stacking Ensemble実装
- さらなる精度向上

**XGBoost失敗時**:
- ハイパーパラメータ再最適化
- または Stratified Group K-Fold

---

## 🎯 目標設定

### 短期目標（今回のセッション）

| 目標 | 現状 | Target | アプローチ |
|------|------|--------|-----------|
| LB Score | 0.77751 | **0.78以上** | XGBoost追加 |
| OOF Score | 0.8384 | 0.84以上 | モデル改善 |
| CV-LB Gap | 6.09% | 5%以下 | - |

### 中期目標

| 目標 | Target | アプローチ |
|------|--------|-----------|
| LB Score | **0.80以上** | Stacking |
| CV-LB Gap | **3%以下** | Stratified Group K-Fold |
| 順位 | **上位10%** | 総合的改善 |

---

## 🚀 即座のアクション

### 推奨: XGBoost追加アンサンブル

**メリット**:
- 最も手軽
- リスク低
- 効果期待大

**実装ステップ**:
1. scripts/xgboost_ensemble_evaluation.py作成
2. XGBoost + LightGBM + GB + RF の4モデルアンサンブル
3. OOF評価
4. submission_xgboost_4model_ensemble.csv 作成
5. Kaggle LB提出

**期待時間**: 30分

---

## まとめ

### 現状

- ✅ 9特徴量Hard Voting確立（LB 0.77751）
- ❌ 特徴量追加アプローチは限界
- ⚠️ CV-LB Gap 6.09%が課題

### 次の一手

**XGBoost追加アンサンブル**を最優先で実施

**理由**:
- 実装簡単
- 効果期待
- リスク低

**期待結果**:
- LB 0.7805-0.7825
- 0.78目標達成の可能性

---

**作成日**: 2025-10-21
**推奨アクション**: XGBoost追加アンサンブルの実装・評価
