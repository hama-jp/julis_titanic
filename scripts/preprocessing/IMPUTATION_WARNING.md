# ⚠️ 重要：欠損値補完の正しい実装について

## 問題の概要

プロジェクト内の多くのスクリプトで、**Train/Test分離が不適切な欠損値補完**が行われています。

### ❌ 誤った実装パターン

```python
# ❌ 間違い：Train/Testそれぞれで個別に補完
def engineer_features(df):
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(
        lambda x: x.fillna(x.median())
    )
    return df

train_fe = engineer_features(train)  # Trainのみから学習
test_fe = engineer_features(test)    # Testのみから学習 ← ここが問題！
```

**問題点**:
- Test PassengerId=1044 の Fare欠損を、**Testデータのみ**から計算した中央値で補完
- Trainには Fare欠損がないため、Testから学習してしまう
- 結果: Fare=7.90 (誤) ← 本来は Fare=8.05 (正)
- 差は0.15と小さいが、**原則として誤り**

### ✅ 正しい実装パターン

```python
# ✅ 正しい：Trainから学習したルールをTestに適用
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

imputer = CorrectImputationPipeline()
imputer.fit(train)  # Trainのみから学習

train_imputed = imputer.transform(train)  # Trainに適用
test_imputed = imputer.transform(test)    # 同じルールをTestに適用
```

**利点**:
- Trainデータから学習したルールをTestに適用
- Test PassengerId=1044 → Fare=8.05 (正確)
- データリーケージなし
- 再現性が高い

---

## 影響を受けるファイル（部分リスト）

以下のファイルは誤ったパターンを含んでいます：

### 提出に使用されたファイル（優先度：高）
- `scripts/models/create_best_submissions.py` ⚠️ 1st提出で使用
- `scripts/models/stratified_groupkfold_evaluation.py` ⚠️ 2nd提出で使用
- `scripts/models/stacking_ensemble.py`

### その他のモデルスクリプト
- `scripts/models/gap_reduction_tuning.py`
- `scripts/models/advanced_improvements.py`
- `scripts/models/optuna_hyperparameter_tuning.py`
- `scripts/models/final_model_9features_safe.py`
- `scripts/models/reduced_features_conservative.py`
- その他多数（約15ファイル）

---

## 正しいパイプラインの使い方

### 1. 基本的な使用方法

```python
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# パイプライン初期化
imputer = CorrectImputationPipeline()

# Trainから補完ルールを学習
imputer.fit(train)

# Train/Testに適用
train_imputed = imputer.transform(train, create_missing_flags=True)
test_imputed = imputer.transform(test, create_missing_flags=True)

# 特徴量エンジニアリング（補完済みデータで）
train_fe = engineer_features(train_imputed)
test_fe = engineer_features(test_imputed)
```

### 2. 欠損フラグの使用

```python
# 補完前の欠損情報を特徴量として使用
# create_missing_flags=True で以下が自動作成されます：
# - Age_Missing: Age が欠損していた場合 1
# - Fare_Missing: Fare が欠損していた場合 1
# - Embarked_Missing: Embarked が欠損していた場合 1

features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
            'SmallFamily', 'Age_Missing', 'Embarked_Missing', 'Fare_Missing']
```

### 3. 高度な Embarked 推定（オプション）

```python
# Fare範囲からEmbarkedを推定する高度な方法
imputer = CorrectImputationPipeline(use_fare_based_embarked=True)
imputer.fit(train)
```

---

## 補完戦略の詳細

### Age補完
- **方法**: Title + Pclass別の中央値
- **例**: Mr + Pclass=3 → Age=26.0
- **フォールバック**: 全体の中央値（28.0）

### Fare補完
- **方法**: Pclass + Embarked別の中央値
- **例**: Pclass=3 & Embarked=S → Fare=8.05
- **フォールバック**: Pclass別の中央値

### Embarked補完
- **基本**: 最頻値（'S'）
- **高度**: Fare範囲から推定（use_fare_based_embarked=True の場合）

---

## 実験結果（比較）

| 補完戦略 | Logistic Regression | Random Forest |
|---------|---------------------|---------------|
| Strategy 1: Group-based Median | **0.8002** | 0.8182 |
| Strategy 2: KNN Imputation | 0.7991 | 0.8193 |
| Strategy 3: Iterative Imputation | 0.7991 | 0.8148 |

**結論**: Group-based Median（このパイプラインで実装）がシンプルかつ効果的

出典: `experiments/logs/20241024_missing_value_imputation_comparison.md`

---

## 移行ガイド

### ステップ1: 古いコードを特定

```bash
# 誤ったパターンを含むファイルを検索
grep -r "df.groupby.*Fare.*transform" scripts/models/
```

### ステップ2: 新しいパイプラインに置き換え

**Before:**
```python
def engineer_features(df):
    df = df.copy()
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna('S')
    return df
```

**After:**
```python
from scripts.preprocessing.correct_imputation_pipeline import CorrectImputationPipeline

# 補完パイプライン（補完のみ）
imputer = CorrectImputationPipeline()
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# 特徴量エンジニアリング（補完以外の処理）
def engineer_features(df):
    df = df.copy()
    # Age, Fare, Embarkedは既に補完済み
    # 他の特徴量エンジニアリングのみ実装
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    # ...
    return df
```

### ステップ3: テスト

```python
# 補完が正しく行われているか確認
assert train_imputed['Age'].isnull().sum() == 0
assert train_imputed['Fare'].isnull().sum() == 0
assert test_imputed['Age'].isnull().sum() == 0
assert test_imputed['Fare'].isnull().sum() == 0

# Test Fare欠損の補完値を確認
test_fare_idx = test[test['Fare'].isnull()].index[0]
assert abs(test_imputed.loc[test_fare_idx, 'Fare'] - 8.05) < 0.01
print("✅ Imputation verified!")
```

---

## 今後のベストプラクティス

1. **新しいモデル開発**: 必ず `CorrectImputationPipeline` を使用
2. **既存コードの修正**: 提出に使うファイルから優先的に修正
3. **コードレビュー**: Train/Test分離が適切か確認
4. **ドキュメント**: このガイドを参照

---

## 関連ファイル

- **正しいパイプライン**: `scripts/preprocessing/correct_imputation_pipeline.py`
- **実験レポート**: `experiments/logs/20241024_missing_value_imputation_comparison.md`
- **ガイド**: `docs/strategies/missing_value_imputation_guide.md`
- **比較**: `docs/strategies/missing_value_comparison.md`

---

## FAQ

### Q: 既存の提出ファイル（LB 0.77511）はどうなる？
A: 影響は限定的（Fare差0.15のみ）ですが、正しい方法でLBスコアが改善する可能性があります。

### Q: 全ファイルを修正する必要がある？
A: 優先度順に修正を推奨：
   1. 提出用ファイル（高優先度）
   2. アンサンブルモデル（中優先度）
   3. 実験スクリプト（低優先度）

### Q: Age補完は Title+Pclass と Pclass+Sex どちらが良い？
A: Title+Pclass の方が精緻ですが、Pclass+Sex の方がシンプルです。
   実験レポートによると、どちらも同等の精度です。
   `CorrectImputationPipeline` では Title+Pclass を採用しています。

---

**最終更新**: 2025-10-24
**作成者**: Claude Code
**重要度**: ⚠️ 高
