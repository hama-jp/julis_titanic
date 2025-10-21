## Summary

安定して0.8以上のスコアを達成するため、包括的なEDAと戦略的なモデル改善を実施しました。

### 実施内容

#### 1. 初回モデル作成とEDA
- **結果**: CV 0.8373, LB 0.7703 (gap: 0.067)
- 詳細なEDAを実施（生存率分析、欠損値処理、特徴量相関）
- 有効な特徴量を設計: Title, FamilySize, IsAlone, AgeBin, FareBin
- 過学習を抑えた保守的なRandomForest

#### 2. 戦略的モデル改善
- CV/LBギャップの問題を特定（過学習の兆候）
- 10個以上のモデルバリエーションを生成
- 3つのアルゴリズム（LR, RF, GB）× 8つの特徴量セット
- アンサンブル手法（Voting）を実装

#### 3. 深い分析（無駄打ち回避）
- **Train/Test分布比較**: Fare, Age, Embarkedに有意な差異を発見
- **CV安定性分析**: Fold間のばらつきを評価
- **Out-of-Fold予測**: 実際の汎化性能を確認
- **過学習検証**: 全モデルで許容範囲内を確認

### 主要な発見

🔍 **問題の根本原因**:
- Train/Testデータセットの分布差（特にFare: 3.36の差）
- これがCV/LBギャップ（0.067）を説明

📊 **最適モデルの特定**:
1. **RF_Conservative**: OOF 0.8283, 過学習 0.0067（最小）
2. **LR_V5**: OOF 0.7946, CV安定性 0.0057（最高）

### 推奨サブミット

#### 🥇 第1推奨: `submission_V3_WithFare_RF.csv`
- アルゴリズム: Random Forest（保守的パラメータ）
- 特徴量: Pclass, Sex, Title, Fare（4個）
- 期待LB: 0.78-0.81
- 理由: 過学習が最小（0.0067）で汎化性能が最も高い

#### 🥈 第2推奨: `submission_V5_WithEmbarked_LR.csv`
- アルゴリズム: Logistic Regression
- 特徴量: Pclass, Sex, Title, Age, Embarked（5個）
- 期待LB: 0.77-0.79
- 理由: CV安定性が最高で、最も予測可能

### ファイル構成

- `eda_and_model.ipynb`: 詳細なEDAと初回モデリング
- `improved_models.py`: 複数モデルの生成と評価
- `deep_analysis.py`: 深い分析と最適モデル選択
- `SUBMISSION_STRATEGY.md`: 戦略ドキュメント
- `submission_*.csv`: 各種サブミット候補（10個）

### 次のステップ

1. RF_ConservativeまたはLR_V5をサブミット
2. LBスコアとOOFスコアのギャップを確認
3. 結果に基づいて残り8回のサブミットを最適化

🤖 Generated with [Claude Code](https://claude.com/claude-code)
