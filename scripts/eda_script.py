#!/usr/bin/env python
# coding: utf-8

# # Titanic EDA and Feature Engineering
# 安定して0.8以上のスコアを目指すためのEDAと特徴量エンジニアリング

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# データ読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print('Train shape:', train.shape)
print('Test shape:', test.shape)


# ## 1. 基本的なデータ確認

# In[ ]:


# データ概要
print('\n=== Train Data Info ===')
print(train.info())
print('\n=== Basic Statistics ===')
print(train.describe())
print('\n=== Survival Rate ===')
print(train['Survived'].value_counts(normalize=True))


# In[ ]:


# 欠損値の確認
print('\n=== Missing Values ===')
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
missing_df = pd.DataFrame({
    'Train': missing_train[missing_train > 0],
    'Test': missing_test[missing_test > 0]
})
print(missing_df)


# ## 2. 生存率との相関分析

# In[ ]:


# 性別と生存率
print('\n=== Survival by Sex ===')
print(train.groupby('Sex')['Survived'].agg(['mean', 'count']))

# クラスと生存率
print('\n=== Survival by Pclass ===')
print(train.groupby('Pclass')['Survived'].agg(['mean', 'count']))

# 乗船港と生存率
print('\n=== Survival by Embarked ===')
print(train.groupby('Embarked')['Survived'].agg(['mean', 'count']))


# In[ ]:


# 年齢と生存率
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 生存者と非生存者の年齢分布
train[train['Survived']==1]['Age'].dropna().hist(bins=30, alpha=0.5, label='Survived', ax=axes[0])
train[train['Survived']==0]['Age'].dropna().hist(bins=30, alpha=0.5, label='Not Survived', ax=axes[0])
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')
axes[0].legend()
axes[0].set_title('Age Distribution by Survival')

# 運賃と生存率
train[train['Survived']==1]['Fare'].hist(bins=30, alpha=0.5, label='Survived', ax=axes[1])
train[train['Survived']==0]['Fare'].hist(bins=30, alpha=0.5, label='Not Survived', ax=axes[1])
axes[1].set_xlabel('Fare')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].set_title('Fare Distribution by Survival')

plt.tight_layout()
plt.show()


# ## 3. 特徴量エンジニアリング（慎重に）
# 
# 過学習と多重共線性を避けるため、以下の点に注意：
# - 相関が高い特徴量は避ける
# - 実際に予測力がある特徴量のみを使用
# - シンプルな特徴量を優先

# In[ ]:


# データのコピーを作成
train_fe = train.copy()
test_fe = test.copy()

# 結合してまとめて処理
all_data = pd.concat([train_fe, test_fe], sort=False).reset_index(drop=True)


# In[ ]:


# 1. Title（敬称）の抽出
# 名前から敬称を抽出（社会的地位を示す重要な情報）
all_data['Title'] = all_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print('\n=== Title Distribution ===')
print(all_data['Title'].value_counts())

# 生存率を確認
print('\n=== Survival by Title ===')
print(train_fe.assign(Title=all_data['Title'][:len(train_fe)]).groupby('Title')['Survived'].agg(['mean', 'count']))


# In[ ]:


# Titleをグループ化（希少なものをまとめる）
title_mapping = {
    'Mr': 'Mr',
    'Miss': 'Miss',
    'Mrs': 'Mrs',
    'Master': 'Master',
    'Rev': 'Rare',
    'Dr': 'Rare',
    'Col': 'Rare',
    'Major': 'Rare',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Lady': 'Rare',
    'Countess': 'Rare',
    'Jonkheer': 'Rare',
    'Sir': 'Rare',
    'Capt': 'Rare',
    'Ms': 'Miss'
}
all_data['Title'] = all_data['Title'].map(title_mapping)
print('\n=== Title After Mapping ===')
print(all_data['Title'].value_counts())


# In[ ]:


# 2. Family Size（家族サイズ）
# SibSpとParchを統合
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1

print('\n=== Survival by Family Size ===')
temp_train = train_fe.copy()
temp_train['FamilySize'] = all_data['FamilySize'][:len(train_fe)]
print(temp_train.groupby('FamilySize')['Survived'].agg(['mean', 'count']))


# In[ ]:


# 3. IsAlone（一人旅かどうか）
all_data['IsAlone'] = (all_data['FamilySize'] == 1).astype(int)

print('\n=== Survival by IsAlone ===')
temp_train['IsAlone'] = all_data['IsAlone'][:len(train_fe)]
print(temp_train.groupby('IsAlone')['Survived'].agg(['mean', 'count']))


# In[ ]:


# 4. Age の欠損値補完
# TitleとPclassごとの中央値で補完
all_data['Age'] = all_data.groupby(['Title', 'Pclass'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

print('\n=== Age Missing After Imputation ===')
print(all_data['Age'].isnull().sum())


# In[ ]:


# 5. Age Binning（年齢のビニング）
# 細かすぎると過学習するので、シンプルに
all_data['AgeBin'] = pd.cut(all_data['Age'], bins=[0, 12, 18, 35, 60, 100], 
                             labels=['Child', 'Teenager', 'Adult', 'MiddleAge', 'Senior'])

print('\n=== Survival by Age Bin ===')
temp_train['AgeBin'] = all_data['AgeBin'][:len(train_fe)]
print(temp_train.groupby('AgeBin')['Survived'].agg(['mean', 'count']))


# In[ ]:


# 6. Fare の欠損値補完と変換
# Pclassごとの中央値で補完
all_data['Fare'] = all_data.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)

# Fareのビニング（過学習を避けるため、粗めに）
all_data['FareBin'] = pd.qcut(all_data['Fare'], q=4, labels=['Low', 'Medium', 'High', 'VeryHigh'], duplicates='drop')

print('\n=== Survival by Fare Bin ===')
temp_train['FareBin'] = all_data['FareBin'][:len(train_fe)]
print(temp_train.groupby('FareBin')['Survived'].agg(['mean', 'count']))


# In[ ]:


# 7. Embarked の欠損値補完
all_data['Embarked'] = all_data['Embarked'].fillna('S')  # 最頻値で補完

print('\n=== Embarked Missing After Imputation ===')
print(all_data['Embarked'].isnull().sum())


# ## 4. エンコーディングと特徴量選択

# In[ ]:


# カテゴリカル変数をエンコーディング
from sklearn.preprocessing import LabelEncoder

le_sex = LabelEncoder()
all_data['Sex'] = le_sex.fit_transform(all_data['Sex'])

le_embarked = LabelEncoder()
all_data['Embarked'] = le_embarked.fit_transform(all_data['Embarked'])

le_title = LabelEncoder()
all_data['Title'] = le_title.fit_transform(all_data['Title'])

le_agebin = LabelEncoder()
all_data['AgeBin'] = le_agebin.fit_transform(all_data['AgeBin'])

le_farebin = LabelEncoder()
all_data['FareBin'] = le_farebin.fit_transform(all_data['FareBin'])


# In[ ]:


# Train/Testに分割
train_processed = all_data[:len(train_fe)].copy()
test_processed = all_data[len(train_fe):].copy()

# 目的変数を追加
train_processed['Survived'] = train['Survived']


# In[ ]:


# 特徴量の相関を確認（多重共線性チェック）
feature_candidates = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
                      'Title', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin']

correlation_matrix = train_processed[feature_candidates].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

print('\n=== High Correlation Pairs (|corr| > 0.7) ===')
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append((correlation_matrix.columns[i], 
                            correlation_matrix.columns[j], 
                            correlation_matrix.iloc[i, j]))
for pair in high_corr:
    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")


# ## 5. モデリングと特徴量選択
# 
# 複数の特徴量セットを試して、最も安定したものを選択

# In[ ]:


# 特徴量セットを定義
feature_sets = {
    'Baseline': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
    'WithTitle': ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title'],
    'WithFamily': ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone'],
    'WithBins': ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin', 'IsAlone'],
    'Minimal': ['Pclass', 'Sex', 'Title', 'AgeBin', 'FareBin'],
    'Best': ['Pclass', 'Sex', 'Title', 'Age', 'Fare', 'IsAlone']
}

# 交差検証の設定
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, features in feature_sets.items():
    X = train_processed[features]
    y = train_processed['Survived']

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, 
                                 random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

    # LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')

    results[name] = {
        'RF_mean': rf_scores.mean(),
        'RF_std': rf_scores.std(),
        'LR_mean': lr_scores.mean(),
        'LR_std': lr_scores.std(),
        'features': features
    }

# 結果を表示
print('\n=== Cross-Validation Results ===')
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('RF_mean', ascending=False)
print(results_df[['RF_mean', 'RF_std', 'LR_mean', 'LR_std']])

# 最良のモデルを選択
best_feature_set = results_df.index[0]
print(f'\n=== Best Feature Set: {best_feature_set} ===')
print(f"Features: {results[best_feature_set]['features']}")
print(f"RF Score: {results[best_feature_set]['RF_mean']:.4f} (+/- {results[best_feature_set]['RF_std']:.4f})")
print(f"LR Score: {results[best_feature_set]['LR_mean']:.4f} (+/- {results[best_feature_set]['LR_std']:.4f})")


# ## 6. 最終モデルの訓練と予測

# In[ ]:


# 最良の特徴量セットで訓練
best_features = results[best_feature_set]['features']
X_train = train_processed[best_features]
y_train = train_processed['Survived']
X_test = test_processed[best_features]

# RandomForestモデル
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# 特徴量の重要度
feature_importance = pd.DataFrame({
    'feature': best_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print('\n=== Feature Importance ===')
print(feature_importance)

# 可視化
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()


# In[ ]:


# 予測
predictions = final_model.predict(X_test)

# 提出ファイルの作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
print('\n=== Submission file created: submission.csv ===')
print(f"Predicted survival rate: {predictions.mean():.3f}")
print(f"Predicted survivors: {predictions.sum()}/{len(predictions)}")


# ## 7. モデルの最終評価
# 
# 交差検証での詳細評価

# In[ ]:


# 各foldでのスコアを確認
cv_scores = cross_val_score(final_model, X_train, y_train, cv=cv, scoring='accuracy')

print('\n=== Final Model CV Scores ===')
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")
print(f"\nMean: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")
print(f"95% CI: [{cv_scores.mean() - 2*cv_scores.std():.4f}, {cv_scores.mean() + 2*cv_scores.std():.4f}]")

if cv_scores.mean() >= 0.80:
    print('\n✓ Target score of 0.80+ achieved!')
else:
    print(f'\n✗ Target score not achieved. Current: {cv_scores.mean():.4f}, Target: 0.80')

