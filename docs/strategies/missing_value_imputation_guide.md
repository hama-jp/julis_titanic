# 欠損値補完ガイド

## 概要

このドキュメントでは、Titanicプロジェクトにおける欠損値補完の標準パイプラインについて説明します。

## 欠損値の状況

### Train データ
- **Age**: 177件 (19.87%)
- **Cabin**: 687件 (77.10%)
- **Embarked**: 2件 (0.22%)

### Test データ
- **Age**: 86件 (20.57%)
- **Cabin**: 327件 (78.23%)
- **Fare**: 1件 (0.24%)

## 採用した補完戦略

**Strategy 1: Group-based Median Imputation**

実験の結果、この戦略が最も優れたバランスを示しました：
- Logistic Regressionで最高精度 (0.8002)
- Random Forestでも高精度 (0.8182)
- 解釈性が高く、実装がシンプル
- 計算が高速

### 補完方法の詳細

#### 1. Fare（運賃）
**方法**: Pclass + Embarked別の中央値

```python
# 例: Pclass=3, Embarked=S の場合
median_fare = 8.05
```

**適用対象**:
- Testデータの1件（PassengerId=1044）

#### 2. Embarked（乗船港）
**方法**: FareとPclassから推定

- Pclass別、Embarked別のFare範囲を学習
- 欠損値のFareとPclassから最も適合するEmbarkedを推定

```python
# 例: Pclass=1, Fare=80.00 の場合
# → Embarked=C (Pclass=1,Embarked=Cの料金範囲に適合)
```

**適用対象**:
- Trainデータの2件（Row 61, 829）

#### 3. Age（年齢）
**方法**: Pclass + Sex別の中央値

```python
# 学習された中央値:
# Pclass=1, Sex=male   → 40.0歳
# Pclass=1, Sex=female → 35.0歳
# Pclass=2, Sex=male   → 30.0歳
# Pclass=2, Sex=female → 28.0歳
# Pclass=3, Sex=male   → 25.0歳
# Pclass=3, Sex=female → 21.5歳
```

**適用対象**:
- Trainデータの177件
- Testデータの86件

#### 4. Cabin（客室番号）
**方法**: 有無フラグ化

```python
# Cabin の値が存在するかどうかを0/1で表現
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
```

**理由**:
- 欠損率が約78%と非常に高い
- 客室番号そのものより、有無が重要な特徴量

## 使い方

### 基本的な使用方法

```python
from scripts.preprocessing.missing_value_imputation import MissingValueImputer

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 補完パイプライン初期化
imputer = MissingValueImputer()

# Trainデータで学習 & 補完
train_imputed = imputer.fit_transform(train)

# Testデータを補完
test_imputed = imputer.transform(test)

# 補完済みデータを保存
train_imputed.to_csv('data/processed/train_imputed.csv', index=False)
test_imputed.to_csv('data/processed/test_imputed.csv', index=False)
```

### スタンドアロン実行

```bash
# 補完パイプラインを実行
python scripts/preprocessing/missing_value_imputation.py

# 欠損値の詳細チェック
python scripts/preprocessing/check_missing_values.py
```

## 実験結果

3つの補完戦略を比較した結果：

| Strategy | Model | Accuracy |
|----------|-------|----------|
| **Strategy 1: Group-based Median** | Logistic Regression | **0.8002** |
| Strategy 1: Group-based Median | Random Forest | 0.8182 |
| Strategy 2: KNN Imputation | Logistic Regression | 0.7991 |
| Strategy 2: KNN Imputation | Random Forest | 0.8193 |
| Strategy 3: Iterative Imputation | Logistic Regression | 0.7991 |
| Strategy 3: Iterative Imputation | Random Forest | 0.8148 |

詳細は `experiments/logs/20241024_missing_value_imputation_comparison.md` を参照。

## 設計思想

### なぜ Group-based Median を選んだか

1. **解釈性**: ビジネスロジックが明確
   - 「同じ客室クラス・性別の乗客の中央値年齢」は直感的

2. **安定性**: 外れ値の影響を受けにくい
   - 中央値は平均値より頑健

3. **シンプルさ**: 実装・保守が容易
   - KNNやIterativeより実装がシンプル
   - デバッグしやすい

4. **計算効率**: 高速
   - グループ統計の計算のみ
   - 大規模データでもスケール可能

5. **精度**: 十分な予測性能
   - 複雑な方法と比較して遜色ない精度

### 代替案との比較

**KNN Imputation**:
- メリット: Random Forestでわずかに高精度 (+0.0011)
- デメリット: 計算コスト高、解釈性低、実装複雑

**Iterative Imputation (MICE)**:
- メリット: 理論的に高度
- デメリット: 今回の実験では最低精度、計算コスト高

## ベストプラクティス

### 1. 常にtrainデータで学習
```python
# Good
imputer.fit(train)
train_imputed = imputer.transform(train)
test_imputed = imputer.transform(test)

# Bad - リーケージの原因
imputer.fit(pd.concat([train, test]))  # NG!
```

### 2. 補完前にデータを確認
```python
# 欠損値の確認
print(train.isnull().sum())
print(test.isnull().sum())

# 欠損値のパターン分析
python scripts/preprocessing/check_missing_values.py
```

### 3. 補完後の検証
```python
# 補完後も欠損がないか確認
assert train_imputed[['Age', 'Fare', 'Embarked']].isnull().sum().sum() == 0
assert test_imputed[['Age', 'Fare', 'Embarked']].isnull().sum().sum() == 0
```

## トラブルシューティング

### Q: Embarkedの補完値が期待と違う
A: FareとPclassの組み合わせから推定しています。`_impute_embarked_from_fare()` メソッドのロジックを確認してください。

### Q: 新しいデータで使えるか？
A: はい。`MissingValueImputer` は再利用可能です。ただし、データの分布が大きく異なる場合は再学習を推奨します。

### Q: 他の補完方法を試したい
A: `experiments/missing_value_imputation_experiment.py` を参考に、新しい戦略を追加してください。

## 関連ファイル

- **補完パイプライン**: `scripts/preprocessing/missing_value_imputation.py`
- **チェックツール**: `scripts/preprocessing/check_missing_values.py`
- **比較実験**: `experiments/missing_value_imputation_experiment.py`
- **実験ログ**: `experiments/logs/20241024_missing_value_imputation_comparison.md`
- **実験結果**: `experiments/results/20241024_imputation_comparison.csv`

## 参考資料

- [Handling Missing Values - Kaggle](https://www.kaggle.com/code)
- [Scikit-learn Imputation Documentation](https://scikit-learn.org/stable/modules/impute.html)
- [MICE Imputation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)
