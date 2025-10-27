# 過学習問題の分析

## 問題の概要

OOFスコアとLBスコアの大きなギャップが継続している。

## スコア推移

| バージョン | 特徴量数 | OOF | LB | ギャップ | 主な変更 |
|-----------|---------|-----|-----|---------|---------|
| 初期版 | 20個 | 0.8451 | 0.75837 | 0.0867 | 多重共線性あり |
| 多重共線性解消 | 14個 | 0.8507 | 0.75119 | 0.0995 | SibSp, Parch等削除 |
| Title削除 | 10個 | 0.8373 | ??? | ??? | Title特徴量削除 |
| IsAlone削除 | 9個 | 0.8384 | 0.75598 | **0.0824** | IsAlone削除（少し改善） |

## 問題の原因候補

### 1. データリークの可能性

以下の特徴量がデータリークの可能性：
- **TicketGroupSize**: train+testで計算しているため、テストデータの情報が混入
- **FarePerPerson**: FamilySizeを使用

### 2. 特徴量エンジニアリングの過剰適合

- Age補完方法（Pclass x Title）が複雑すぎる可能性
- FarePerPerson, TicketGroupSize等の派生特徴が過学習を引き起こしている可能性

### 3. モデルの複雑さ

- アンサンブル（4モデル平均）が過剰かもしれない
- 個々のモデルのハイパーパラメータが訓練データに適合しすぎている

## 次のアプローチ

### アプローチ1: データリークの可能性がある特徴量を削除

削除候補：
- TicketGroupSize
- FarePerPerson

### アプローチ2: 最もシンプルなモデルに戻る

- 基本特徴量のみ（Pclass, Sex, Age, FamilySize, Fare, Embarked）
- 単一モデル（GradientBoostingまたはRandomForest）
- 強い正則化

### アプローチ3: Age補完を単純化

- Pclass x Titleではなく、Title別のみに戻す
- または全体の中央値のみ

## 推奨：最もシンプルなベースラインに戻る

1. 特徴量: 最小限の6個
   - Pclass
   - Sex_male
   - Age_fillna（全体の中央値）
   - FamilySize
   - Fare_fillna（全体の中央値）
   - Embarked（3カテゴリ）

2. モデル: 単一モデル
   - GradientBoosting（強い正則化）

3. 検証: OOFとLBのギャップを確認
