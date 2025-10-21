# julis_titanic

Kaggle Titanic コンペティション用のプロジェクト

## ディレクトリ構造

```
/
├── data/
│   └── raw/              # 元データ (train.csv, test.csv, gender_submission.csv)
├── notebooks/            # Jupyter notebooks (EDA, 分析)
├── scripts/              # Pythonスクリプト (モデル学習、特徴量エンジニアリング)
├── submissions/          # Kaggleサブミッション用CSVファイル
├── docs/                 # ドキュメント、戦略、結果ログ
└── README.md
```

## 最新スコア

- **Submission Enhanced V4 Depth5**: 0.78229

## 使い方

### データの確認
```bash
ls data/raw/
```

### スクリプトの実行
```bash
python scripts/submission_5_enhanced_features.py
```

### ノートブックの起動
```bash
jupyter notebook notebooks/eda_and_model.ipynb
```

## サブミッション履歴

詳細は `docs/RESULTS_LOG.md` を参照してください。
