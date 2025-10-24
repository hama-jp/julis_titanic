# プロジェクト構造とルール

このドキュメントでは、Titanic Kaggleプロジェクトのディレクトリ構造、ファイル管理、実験メモの残し方についてのルールを定義します。

## ディレクトリ構造

```
julis_titanic/
├── data/                          # データファイル
│   ├── raw/                       # 元データ（変更不可）
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── gender_submission.csv
│   ├── processed/                 # 前処理済みデータ（オプション）
│   └── features/                  # 特徴量エンジニアリング後のデータ（オプション）
│
├── notebooks/                     # Jupyter Notebook
│   ├── eda/                       # 探索的データ分析用
│   ├── experiments/               # 実験用ノートブック
│   └── archive/                   # 古いノートブック
│
├── scripts/                       # Pythonスクリプト
│   ├── preprocessing/             # データ前処理スクリプト
│   ├── feature_engineering/       # 特徴量エンジニアリング
│   ├── models/                    # モデル学習スクリプト
│   ├── evaluation/                # 評価・分析スクリプト
│   └── utils/                     # ユーティリティ関数
│
├── submissions/                   # Kaggle提出ファイル
│   ├── YYYYMMDD_HHMM_description.csv  # 提出ファイル（タイムスタンプ付き）
│   └── archive/                   # 古い提出ファイル
│
├── experiments/                   # 実験記録
│   ├── logs/                      # 実験ログ
│   │   └── YYYYMMDD_experiment_name.md
│   ├── results/                   # 実験結果（数値、テキスト）
│   │   └── YYYYMMDD_experiment_name.txt
│   └── configs/                   # 実験設定ファイル（JSON, YAML等）
│
├── models/                        # 保存したモデル（オプション）
│   └── YYYYMMDD_model_name.pkl
│
├── docs/                          # プロジェクトドキュメント
│   ├── analysis/                  # 分析レポート
│   ├── strategies/                # 戦略ドキュメント
│   ├── summaries/                 # サマリー、まとめ
│   └── archive/                   # 古いドキュメント
│
├── .gitignore                     # Git除外設定
├── README.md                      # プロジェクト概要
├── PROJECT_STRUCTURE.md           # このファイル
└── requirements.txt               # Python依存パッケージ
```

## ファイル命名規則

### 1. 提出ファイル（submissions/）

**フォーマット**: `YYYYMMDD_HHMM_description.csv`

**例**:
- `20241024_1430_baseline_rf.csv`
- `20241024_1520_ensemble_lgb_gb_lr.csv`
- `20241024_1600_9features_hardvoting.csv`

**ルール**:
- 日付時刻を先頭に付ける（ソートしやすい）
- 簡潔な説明を含める（モデル名、特徴量セット等）
- アンダースコア区切り
- 小文字を推奨

### 2. 実験ログ（experiments/logs/）

**フォーマット**: `YYYYMMDD_experiment_name.md`

**例**:
- `20241024_baseline_models.md`
- `20241024_feature_engineering_round2.md`
- `20241024_ensemble_comparison.md`

### 3. スクリプト（scripts/）

**フォーマット**: `purpose_description.py`

**例**:
- `train_baseline_models.py`
- `evaluate_feature_importance.py`
- `create_ensemble_predictions.py`

**ルール**:
- 機能がわかる名前をつける
- 動詞で始めることを推奨（train, evaluate, create等）
- サブディレクトリで整理

### 4. ノートブック（notebooks/）

**フォーマット**: `YYYYMMDD_description.ipynb` または `description.ipynb`

**例**:
- `20241024_initial_eda.ipynb`
- `feature_engineering_experiment.ipynb`

## 実験メモの管理方法

### 実験ログのテンプレート

各実験は `experiments/logs/YYYYMMDD_experiment_name.md` に記録します。

```markdown
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

### 特徴量
- 使用特徴量リスト:
  - Feature1: [説明]
  - Feature2: [説明]
  - ...

### モデル
- アルゴリズム: [RandomForest, LightGBM等]
- ハイパーパラメータ:
  ```python
  {
      'param1': value1,
      'param2': value2,
      ...
  }
  ```

### 評価方法
- クロスバリデーション: [k-fold, stratified等]
- 評価指標: [Accuracy, F1-score, ROC-AUC等]

## 結果

### スコア
- CV Score: [数値] ± [標準偏差]
- Public LB Score: [数値]
- Private LB Score: [数値]（コンペ終了後）

### 詳細結果
[テーブル、グラフ、重要な観察結果]

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

## 次のアクション

- [ ] [TODO1]
- [ ] [TODO2]
- [ ] [TODO3]

## 関連ファイル

- スクリプト: `scripts/xxx/yyy.py`
- 提出ファイル: `submissions/YYYYMMDD_HHMM_description.csv`
- ノートブック: `notebooks/experiments/zzz.ipynb`
```

### 簡易メモ（results/）

実験の数値結果や簡単なメモは `experiments/results/` にテキストファイルとして保存できます。

**フォーマット**: シンプルなテキスト出力で十分

## コード管理のルール

### 1. スクリプトの整理

- **使用中のスクリプト**: 適切なサブディレクトリに配置
- **古いスクリプト**: `scripts/archive/YYYYMMDD_original_name.py` に移動
- **共通機能**: `scripts/utils/` に抽出

### 2. ノートブックの整理

- **アクティブなノートブック**: 目的別にサブディレクトリに配置
- **完了したノートブック**: `notebooks/archive/` に移動
- **セル出力はクリア**: Gitコミット前にノートブックの出力をクリア推奨

### 3. 提出ファイルの整理

- **最新の提出**: `submissions/` 直下に配置
- **古い提出**: 定期的に `submissions/archive/` に移動（月1回程度）

## ドキュメント管理のルール

### ルートディレクトリのMarkdownファイル

ルートディレクトリには最小限のMarkdownファイルのみを配置します：

- `README.md`: プロジェクト概要
- `PROJECT_STRUCTURE.md`: このファイル

### docs/ ディレクトリのMarkdownファイル

その他のドキュメントは `docs/` 以下に整理します：

- **分析レポート**: `docs/analysis/YYYYMMDD_topic.md`
- **戦略ドキュメント**: `docs/strategies/topic.md`
- **サマリー**: `docs/summaries/topic.md`

### 既存のMarkdownファイルの整理

現在ルートディレクトリにある以下のようなファイルは適切なディレクトリに移動することを推奨：

- 分析レポート系 → `docs/analysis/`
- 戦略・計画系 → `docs/strategies/`
- サマリー系 → `docs/summaries/`

## Git運用ルール

### コミットメッセージ

```
[種別] 簡潔な説明

詳細な説明（必要に応じて）
```

**種別の例**:
- `[feat]`: 新機能追加
- `[fix]`: バグ修正
- `[exp]`: 実験の追加
- `[docs]`: ドキュメント更新
- `[refactor]`: リファクタリング
- `[clean]`: コード整理、不要ファイル削除

### .gitignore

以下をGit管理から除外：
- `__pycache__/`, `*.pyc`
- `.ipynb_checkpoints/`
- `models/*.pkl`（大きなモデルファイル）
- `data/processed/`, `data/features/`（再現可能な中間データ）
- `.env`, `*.log`

## ベストプラクティス

1. **実験は必ずログを残す**: 後から振り返れるように
2. **コードは再現可能に**: 乱数シードの固定、バージョン管理
3. **定期的に整理**: 月1回程度、古いファイルをarchiveに移動
4. **命名は一貫性を保つ**: このドキュメントのルールに従う
5. **ドキュメントは更新する**: 大きな変更があれば README.md や実験ログを更新

## 参考リンク

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Kaggleノートブックのベストプラクティス](https://www.kaggle.com/code)
