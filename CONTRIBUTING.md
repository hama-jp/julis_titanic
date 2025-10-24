# コントリビューションガイド

このドキュメントは、プロジェクトへの貢献方法とコーディング規約をまとめたものです。

---

## 📁 ファイル配置ルール

### ⚠️ 重要：提出ファイル（CSV）の保存場所

**❌ ルートフォルダにCSVを保存しないでください**

```python
# ❌ 間違い：ルートフォルダに保存
submission.to_csv('my_predictions.csv', index=False)
```

**✅ submissions/ ディレクトリに保存してください**

```python
# ✅ 正しい方法1：ヘルパー関数を使用（推奨）
from scripts.utils.submission_helper import save_submission

save_submission(submission, 'my_predictions.csv')

# ✅ 正しい方法2：直接指定
submission.to_csv('submissions/my_predictions.csv', index=False)
```

### ヘルパー関数の使い方

`scripts/utils/submission_helper.py` を使用すると：
- 自動的に submissions/ に保存
- 必須カラム（PassengerId, Survived）の検証
- 行数の検証（418行）
- タイムスタンプ付きファイル名
- ルートフォルダへの誤保存を防止

```python
from scripts.utils.submission_helper import save_submission

# 基本的な使い方
save_submission(submission, 'my_model.csv')

# タイムスタンプ付き（20251024_1234_my_model.csv）
save_submission(submission, 'my_model.csv', add_timestamp=True)

# 上書き許可
save_submission(submission, 'existing_file.csv', overwrite=True)
```

---

## 🚫 .gitignore の設定

ルートフォルダの `*.csv` は `.gitignore` で無視されます：

```gitignore
# Submission files in root (should be in submissions/ directory)
/*.csv
```

これにより、誤ってルートに保存されたCSVファイルは自動的にGit管理外となります。

---

## 📂 ディレクトリ構造

```
julis_titanic/
├── data/
│   ├── raw/              # 元データ（train.csv, test.csv）
│   ├── processed/        # 前処理済みデータ
│   └── interim/          # 中間データ
├── scripts/
│   ├── preprocessing/    # 前処理スクリプト
│   ├── models/           # モデル学習スクリプト
│   ├── feature_engineering/  # 特徴量エンジニアリング
│   └── utils/            # ユーティリティ関数
├── submissions/          # ✅ 提出ファイルはここ
│   ├── archive/          # 古い提出ファイル
│   └── *.csv             # 最新の提出ファイル
├── experiments/          # 実験ログ
│   ├── logs/             # Markdownログ
│   └── results/          # 実験結果
└── docs/                 # ドキュメント
```

---

## 🔧 欠損値補完のベストプラクティス

### ⚠️ Train/Test分離を正しく実装する

**❌ 間違い：Train/Testそれぞれで個別に補完**

```python
def engineer_features(df):
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    return df

train_fe = engineer_features(train)  # Trainのみから学習
test_fe = engineer_features(test)    # Testのみから学習 ← 問題！
```

**✅ 正しい：Trainから学習したルールをTestに適用**

```python
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

imputer = CorrectImputationPipeline()
imputer.fit(train)  # Trainのみから学習

train_imputed = imputer.transform(train)
test_imputed = imputer.transform(test)  # 同じルールを適用
```

詳細は `scripts/preprocessing/IMPUTATION_WARNING.md` を参照してください。

---

## 📝 コミットメッセージ

### フォーマット

```
[type] Short description

Detailed explanation (optional)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Type

- `feat`: 新機能
- `fix`: バグ修正
- `refactor`: リファクタリング
- `docs`: ドキュメント
- `test`: テスト追加
- `chore`: 雑務（ビルド、設定など）

### 例

```
[feat] Add correct imputation pipeline

- Implement proper train/test separation
- Fix Fare imputation (7.90 → 8.05)
- Add CorrectImputationPipeline class

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## 🧪 テストとバリデーション

### 提出ファイルの検証

提出前に以下を確認：

```python
# 1. 行数の確認（418行必須）
assert len(submission) == 418, "Must have 418 rows"

# 2. カラムの確認
assert list(submission.columns) == ['PassengerId', 'Survived']

# 3. PassengerIdの範囲確認
assert submission['PassengerId'].min() == 892
assert submission['PassengerId'].max() == 1309

# 4. Survivedの値確認（0 or 1）
assert submission['Survived'].isin([0, 1]).all()

# 5. 欠損値の確認
assert submission.isnull().sum().sum() == 0
```

### モデルの検証

```python
# CV Score
cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Score: {cv_score.mean():.4f} (±{cv_score.std():.4f})")

# Train Score（過学習チェック）
train_score = model.score(X_train, y_train)
gap = train_score - cv_score.mean()
print(f"Overfitting Gap: {gap:.4f}")

if gap > 0.05:
    warnings.warn("High overfitting detected (gap > 0.05)")
```

---

## 📊 実験ログの記録

実験結果は `experiments/logs/` にMarkdown形式で記録：

```markdown
# 実験名

**日付**: 2025-10-24
**目的**: 〇〇を改善

## 仮説
- ...

## 実験内容
- ...

## 結果
- CV Score: 0.8294
- LB Score: 0.77511

## 考察
- ...

## 次のアクション
- [ ] ...
```

---

## 🔄 ブランチ戦略

- `main`: 本番ブランチ（安定版）
- `claude/*`: Claude Codeの作業ブランチ
- `feature/*`: 機能追加ブランチ
- `fix/*`: バグ修正ブランチ

---

## 📚 参考ドキュメント

- [欠損値補完の警告](scripts/preprocessing/IMPUTATION_WARNING.md)
- [正しい補完パイプライン](scripts/preprocessing/correct_imputation_pipeline.py)
- [提出履歴](submissions/SUBMISSION_HISTORY.md)
- [プロジェクトまとめ](CORRECT_IMPUTATION_UPDATE.md)

---

## ❓ FAQ

### Q: なぜルートフォルダにCSVを保存してはいけないのか？

A:
1. **整理整頓**: 提出ファイルが散乱すると管理が困難
2. **誤コミット防止**: .gitignoreで自動的に除外されるが、意図しないファイルが混入する可能性
3. **チーム作業**: 他の開発者が混乱しない
4. **自動化**: スクリプトで submissions/ 内のファイルを処理しやすい

### Q: 既存のスクリプトがルートに保存している場合は？

A: `scripts/utils/submission_helper.py` を使うように修正してください。
   移行ガイドは `CONTRIBUTING.md` を参照。

### Q: Gitでルートの*.csvを追跡したい場合は？

A: `.gitignore` から `/*.csv` の行を削除してください。
   ただし、推奨されません。

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
