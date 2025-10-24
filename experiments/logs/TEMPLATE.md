# [実験名]

**日付**: YYYY-MM-DD
**担当**: [名前]
**目的**: [この実験で何を明らかにしたいか]

## 仮説

[検証したい仮説を明確に記述]

## 実験内容

### データ
- 使用データ: [train.csv, test.csv等]
- データサイズ: [行数、列数]
- データ分割: [train/validation split方法]

### 特徴量
- 使用特徴量リスト:
  - Feature1: [説明]
  - Feature2: [説明]
  - Feature3: [説明]
  - ...
- 特徴量数: [N個]

### モデル
- アルゴリズム: [RandomForest, LightGBM, LogisticRegression等]
- ハイパーパラメータ:
  ```python
  {
      'param1': value1,
      'param2': value2,
      ...
  }
  ```

### 評価方法
- クロスバリデーション: [5-fold stratified等]
- 評価指標: [Accuracy, F1-score, ROC-AUC等]

## 結果

### スコア
- CV Score: [数値] ± [標準偏差]
- OOF Score: [数値]
- Public LB Score: [数値]
- Private LB Score: [数値]（コンペ終了後）

### 詳細結果

| Metric | Value |
|--------|-------|
| Accuracy | X.XXXX |
| Precision | X.XXXX |
| Recall | X.XXXX |
| F1-Score | X.XXXX |

### 特徴量重要度（上位10個）

| Feature | Importance |
|---------|------------|
| Feature1 | X.XXX |
| Feature2 | X.XXX |
| ... | ... |

### その他の観察結果

- [観察1]
- [観察2]
- [観察3]

## 考察

### うまくいった点
- [ポイント1]
- [ポイント2]

### うまくいかなかった点
- [ポイント1]
- [ポイント2]

### 学んだこと
- [学び1]
- [学び2]

### 仮説の検証結果
- [仮説に対する結論]

## 次のアクション

- [ ] [TODO1: 次に試すべきこと]
- [ ] [TODO2: 改善案]
- [ ] [TODO3: 追加分析]

## 関連ファイル

- スクリプト: `scripts/xxx/yyy.py`
- 提出ファイル: `submissions/YYYYMMDD_HHMM_description.csv`
- ノートブック: `notebooks/experiments/zzz.ipynb`
- 結果ファイル: `experiments/results/YYYYMMDD_experiment_name.txt`
- 設定ファイル: `experiments/configs/YYYYMMDD_experiment_name.yaml`

## 参考資料

- [関連する論文やブログ記事のリンク]
- [参考にしたKaggleノートブック]
