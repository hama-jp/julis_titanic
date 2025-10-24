# julis_titanic

Kaggle Titanic コンペティション用のプロジェクト

## クイックスタート

### データの確認
```bash
ls data/raw/
```

### スクリプトの実行
```bash
python scripts/models/your_model.py
```

### ノートブックの起動
```bash
jupyter notebook notebooks/
```

## 最新スコア

- **Submission Enhanced V4 Depth5**: 0.78229
- **9-Feature Hard Voting**: 0.78229

## プロジェクト構造

このプロジェクトは整理されたディレクトリ構造とファイル命名規則に従っています。

詳細なプロジェクト構造、ファイル命名規則、実験メモの残し方については、以下のドキュメントを参照してください：

**[PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md)** - プロジェクト構造とルールの詳細

### 主要ディレクトリ

```
julis_titanic/
├── data/                 # データファイル
├── notebooks/            # Jupyter Notebook
├── scripts/              # Pythonスクリプト
├── submissions/          # Kaggle提出ファイル
├── experiments/          # 実験記録とログ
├── docs/                 # プロジェクトドキュメント
└── models/               # 保存したモデル
```

## ドキュメント

- [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) - プロジェクト構造とルール
- [docs/RESULTS_LOG.md](./docs/RESULTS_LOG.md) - サブミッション履歴と結果
- [docs/strategies/](./docs/strategies/) - 戦略ドキュメント

## 実験の進め方

1. **仮説を立てる**: 何を検証したいか明確にする
2. **実験ログを作成**: `experiments/logs/YYYYMMDD_experiment_name.md` を作成
3. **コードを書く**: `scripts/` または `notebooks/` で実装
4. **結果を記録**: 実験ログに結果と考察を記録
5. **提出**: `submissions/YYYYMMDD_HHMM_description.csv` で提出

詳細は [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) の「実験メモの管理方法」を参照。
