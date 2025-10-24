# 欠損値補完コードの関係性と進化

## 概要

このドキュメントでは、プロジェクト内の既存の欠損値補完コードと、新しく実装した統一パイプラインとの関係を説明します。

## 既存の欠損値補完パターン

プロジェクト内には3つの主要な補完パターンが混在していました：

### パターン1: シンプルな補完（最も一般的）

**使用ファイル**:
- `scripts/models/lightgbm_tuning.py`
- `scripts/models/top3_ensemble.py`
- `scripts/models/xgboost_tuning_and_diversity.py`
- その他多数（約20ファイル）

**補完方法**:
```python
# Age: 全体の中央値
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fare: 全体の中央値
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Embarked: 最頻値（'S'）
df['Embarked'].fillna('S', inplace=True)
# または
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

**特徴**:
- ✅ 実装が最もシンプル
- ✅ 高速
- ❌ グループ特性を考慮していない
- ❌ Fareの欠損（testのみ）に対応していない明示的なロジックなし

---

### パターン2: Pclass別の補完（中程度の高度化）

**使用ファイル**:
- `scripts/feature_engineering/incremental_feature_evaluation.py`
- `scripts/feature_engineering/recommended_11features_model.py`
- `scripts/evaluation/threshold_optimization.py`
- `scripts/feature_engineering/age_feature_comparison.py`
- その他（約7ファイル）

**補完方法**:
```python
# Age: 全体の中央値（変わらず）
df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

# Fare: Pclass別の中央値
df_fe['Fare'].fillna(
    df_fe.groupby('Pclass')['Fare'].transform('median'),
    inplace=True
)

# Embarked: 最頻値
df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0], inplace=True)
```

**特徴**:
- ✅ Fareについて客室クラスを考慮
- ✅ より精緻な補完
- ❌ Ageはまだ全体の中央値のみ
- ❌ Embarkedの高度な推定なし

---

### パターン3: Title + Pclass別の補完（最も高度）

**使用ファイル**:
- `scripts/models/model_specific_features_pipeline.py` ⭐
- `scripts/evaluation/missing_value_analysis.py`

**補完方法**:
```python
# Age: Title + Pclass別の中央値
for title in df_fe['Title'].unique():
    for pclass in df_fe['Pclass'].unique():
        mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
        median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
        if pd.notna(median_age):
            df_fe.loc[mask, 'Age'] = median_age
df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

# Fare: Pclass別の中央値
df_fe['Fare'] = df_fe.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: 最頻値
df_fe['Embarked'] = df_fe['Embarked'].fillna('S')

# 重要: 欠損フラグの作成（補完前）
df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)
```

**特徴**:
- ✅ Ageについて敬称（Title）とPclassを考慮
- ✅ 欠損フラグを特徴量として活用
- ✅ 最も精緻な補完
- ❌ Fareは依然としてPclass別の中央値のみ
- ❌ Embarkedの高度な推定なし

---

## 新しい統一パイプライン（MissingValueImputer）

**実装ファイル**: `scripts/preprocessing/missing_value_imputation.py`

### 既存コードとの主な違い

#### 1. Fare補完の高度化 ⭐ NEW

```python
# Pclass + Embarked別の中央値（既存にはない）
fare_median = train[(train['Pclass'] == pclass) &
                   (train['Embarked'] == embarked)]['Fare'].median()
```

**改善点**:
- testデータの1件のFare欠損（PassengerId=1044）を明示的に処理
- Pclass=3, Embarked=S の組み合わせで8.05を補完
- 既存のPclassのみの補完より精緻

#### 2. Embarked補完の高度化 ⭐ NEW

```python
# FareとPclassの範囲から推定（既存にはない）
def _impute_embarked_from_fare(self, pclass: int, fare: float) -> str:
    # Pclass別の各Embarkedの料金範囲と比較
    # Fareが範囲内にある、または最も近いEmbarkedを選択
```

**改善点**:
- trainデータの2件のEmbarked欠損を高度に推定
- 単純な最頻値（'S'）ではなく、Fare=80.00とPclass=1から'C'を推定
- より妥当な補完

#### 3. Age補完: Pclass + Sex別 ⭐ IMPROVED

```python
# Pclass + Sex別の中央値
age_median = train[(train['Pclass'] == pclass) &
                   (train['Sex'] == sex)]['Age'].median()
```

**改善点**:
- Title + Pclassより実装がシンプル
- Sexは直接的な特徴量で解釈しやすい
- 既存のTitle抽出ロジックへの依存がない

#### 4. train/test分離の明確化 ⭐ NEW

```python
imputer = MissingValueImputer()
imputer.fit(train)  # trainで学習
train_imputed = imputer.transform(train)
test_imputed = imputer.transform(test)  # 同じロジックで補完
```

**改善点**:
- リーケージを防止
- train/testで一貫した補完
- 既存コードは各スクリプト内で個別に実装

#### 5. Cabin処理の標準化

```python
# Has_Cabinフラグ化（既存にもあるが標準化）
df['Has_Cabin'] = df['Cabin'].notna().astype(int)
```

**改善点**:
- 既存コードと同様だが、パイプライン化
- 78%の欠損率を考慮した適切な処理

---

## 比較表

| 項目 | パターン1（シンプル） | パターン2（中程度） | パターン3（高度） | **新パイプライン** |
|------|---------------------|-------------------|-----------------|-------------------|
| **Age補完** | 全体中央値 | 全体中央値 | Title+Pclass中央値 | **Pclass+Sex中央値** |
| **Fare補完** | 全体中央値 | Pclass中央値 | Pclass中央値 | **Pclass+Embarked中央値** |
| **Embarked補完** | 最頻値'S' | 最頻値'S' | 最頻値'S' | **Fare+Pclassから推定** |
| **欠損フラグ** | なし | なし | Age, Embarked | **オプション** |
| **train/test分離** | ❌ | ❌ | ❌ | **✅** |
| **再利用性** | 各スクリプトに埋込 | 各スクリプトに埋込 | 各スクリプトに埋込 | **✅ クラス化** |
| **実験的検証** | なし | なし | なし | **✅ 3戦略比較** |

---

## 精度比較（新パイプラインの実験結果）

| 補完戦略 | Logistic Regression | Random Forest |
|---------|---------------------|---------------|
| **新パイプライン (Group-based Median)** | **0.8002** | 0.8182 |
| KNN Imputation | 0.7991 | **0.8193** |
| Iterative Imputation | 0.7991 | 0.8148 |

**結論**: 新パイプラインは高精度かつシンプル

---

## 移行ガイド

### 既存コードから新パイプラインへの移行

#### Before（既存コード）

```python
# 各スクリプトで個別に実装
def engineer_features(df):
    df = df.copy()

    # Age補完
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fare補完
    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Embarked補完
    df['Embarked'].fillna('S', inplace=True)

    return df

train_fe = engineer_features(train)
test_fe = engineer_features(test)
```

#### After（新パイプライン）

```python
from scripts.preprocessing.missing_value_imputation import MissingValueImputer

# 補完パイプライン
imputer = MissingValueImputer()
train_imputed = imputer.fit_transform(train)
test_imputed = imputer.transform(test)

# 特徴量エンジニアリング（補完済みデータで）
def engineer_features(df):
    df = df.copy()
    # Age, Fare, Embarkedは既に補完済み
    # Has_Cabinも既に作成済み

    # Title抽出などの他の処理のみ
    ...

    return df

train_fe = engineer_features(train_imputed)
test_fe = engineer_features(test_imputed)
```

### 移行のメリット

1. **コードの重複削減**: 各スクリプトで補完ロジックを書く必要がない
2. **一貫性の向上**: 全モデルで同じ補完方法を使用
3. **保守性の向上**: 補完ロジックの変更が1箇所で済む
4. **リーケージ防止**: train/test分離が明確
5. **精度の向上**: より高度な補完戦略

---

## 既存コードとの共存

新パイプラインは既存コードと**完全に独立**しています：

### 現状維持の選択肢

既存スクリプトはそのまま動作します。新パイプラインは：
- ✅ 既存コードを壊さない
- ✅ 新しいモデル開発時に選択的に使用可能
- ✅ 段階的な移行が可能

### 推奨される使用シナリオ

**新パイプラインを推奨**:
- 新しいモデルの開発
- アンサンブルモデルの構築
- 提出用の最終モデル

**既存コードを維持**:
- 既に良好な精度を出しているモデル
- 比較実験のベースライン
- 再現性が重要な既存の実験

---

## 次のステップ

### 短期的

1. **新規モデル開発**: 新パイプラインを使用
2. **ドキュメント整備**: このドキュメントを参照
3. **実験的移行**: 1-2個の既存モデルで試す

### 長期的

1. **段階的移行**: 高性能モデルから順次移行
2. **統一化**: 最終的に全モデルで統一パイプライン使用
3. **継続改善**: 新しい補完戦略の実験と統合

---

## 関連ファイル

### 新パイプライン
- `scripts/preprocessing/missing_value_imputation.py` - メインパイプライン
- `scripts/preprocessing/check_missing_values.py` - 分析ツール
- `experiments/missing_value_imputation_experiment.py` - 比較実験
- `docs/strategies/missing_value_imputation_guide.md` - 使い方ガイド

### 既存の主要ファイル
- `scripts/models/model_specific_features_pipeline.py` - 最も高度（パターン3）
- `scripts/evaluation/missing_value_analysis.py` - 欠損フラグ分析
- その他多数のモデルスクリプト

---

## まとめ

**新パイプライン（MissingValueImputer）の位置づけ**:

1. 既存の補完方法を**統合・発展**させたもの
2. **Fare**と**Embarked**の補完を大幅に高度化
3. **Age**補完はTitle+Pclassよりシンプルで効果的
4. train/test分離とクラス化で**再利用性**を向上
5. 実験的検証により**精度を確認済み**

**既存コードとの関係**:
- 完全に**独立**して動作
- **共存可能**
- **段階的移行**を推奨
- 既存コードの**ベストプラクティス**を継承・改善
