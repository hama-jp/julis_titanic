"""
推奨11特徴量でのモデル訓練と評価

特徴量構成:
- ベース9特徴: Pclass, Sex, Title, Age, IsAlone, FareBin, SmallFamily, Age_Missing, Embarked_Missing
- 新規2特徴: TicketGroup, Deck_Safe

目標:
- CV Score 0.832以上
- VIF < 10 (全特徴量)
- Gap < 0.015
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("推奨11特徴量でのモデル訓練と評価")
print("="*80)

# ========================================================================
# データ読み込み
# ========================================================================
print("\n【データ読み込み】")
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# ========================================================================
# 特徴量エンジニアリング
# ========================================================================
print("\n【特徴量エンジニアリング】")

def engineer_features(df):
    """推奨11特徴量のエンジニアリング"""
    df_fe = df.copy()

    # 1. 欠損フラグ（補完前）
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)

    # 2. Title抽出
    df_fe['Title'] = df_fe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_fe['Title'] = df_fe['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_fe['Title'] = df_fe['Title'].replace('Mlle', 'Miss')
    df_fe['Title'] = df_fe['Title'].replace('Ms', 'Miss')
    df_fe['Title'] = df_fe['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df_fe['Title'] = df_fe['Title'].map(title_mapping).fillna(0)

    # 3. 家族関連
    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    df_fe['SmallFamily'] = ((df_fe['FamilySize'] >= 2) & (df_fe['FamilySize'] <= 4)).astype(int)

    # 4. Age補完
    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    # 5. Fare補完とビニング
    df_fe['Fare'].fillna(df_fe.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    df_fe['FareBin'] = pd.qcut(df_fe['Fare'], q=4, labels=False, duplicates='drop')

    # 6. Sex encoding
    df_fe['Sex'] = df_fe['Sex'].map({'male': 0, 'female': 1})

    # 7. Embarked補完
    df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0], inplace=True)

    # 8. TicketGroup（新規）
    df_fe['TicketGroup'] = df_fe['Ticket'].map(df_fe['Ticket'].value_counts())

    # 9. Deck_Safe（新規）
    df_fe['Deck'] = df_fe['Cabin'].str[0]
    df_fe['Deck_Safe'] = ((df_fe['Deck'] == 'D') | (df_fe['Deck'] == 'E') | (df_fe['Deck'] == 'B')).astype(int)

    return df_fe

# データ結合（特徴量エンジニアリング用）
all_data = pd.concat([train, test], sort=False, ignore_index=True)
all_data_fe = engineer_features(all_data)

# Train/Test分割
train_fe = all_data_fe[:len(train)].copy()
test_fe = all_data_fe[len(train):].copy()

# 推奨11特徴量
recommended_features = [
    'Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
    'SmallFamily', 'Age_Missing', 'Embarked_Missing',
    'TicketGroup', 'Deck_Safe'
]

X_train = train_fe[recommended_features]
y_train = train['Survived']
X_test = test_fe[recommended_features]

print(f"\n推奨特徴量: {recommended_features}")
print(f"特徴量数: {len(recommended_features)}")

# ========================================================================
# 多重共線性の最終確認
# ========================================================================
print("\n" + "="*80)
print("多重共線性の最終確認")
print("="*80)

# VIF計算
vif_data = pd.DataFrame()
vif_data["Feature"] = recommended_features
vif_values = []
for i in range(len(recommended_features)):
    try:
        vif = variance_inflation_factor(X_train.values, i)
        if np.isinf(vif) or np.isnan(vif):
            vif = 999999
        vif_values.append(vif)
    except:
        vif_values.append(999999)

vif_data["VIF"] = vif_values
vif_data = vif_data.sort_values('VIF', ascending=False)

print("\n【VIF（分散拡大係数）】")
print(vif_data)

# VIF > 10のチェック
high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ VIF > 10の特徴量が{len(high_vif)}個検出されました:")
    print(high_vif)
else:
    print("\n✓ 全特徴量のVIF < 10（多重共線性なし）")

# 相関係数チェック
print("\n【高相関ペア（|r| > 0.7）】")
corr_matrix = X_train.corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature1': corr_matrix.columns[i],
                'Feature2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if len(high_corr_pairs) > 0:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df)
else:
    print("高相関ペアなし（良好）")

# ========================================================================
# モデル訓練と評価
# ========================================================================
print("\n" + "="*80)
print("モデル訓練と評価")
print("="*80)

# LightGBMモデル（チューニング済みパラメータ）
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=2,
    num_leaves=4,
    min_child_samples=30,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_split_gain=0.01,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

# Stratified K-Fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV評価
print("\n【Cross Validation評価】")
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"Fold scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"\nCV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train score
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
print(f"Train Score: {train_score:.4f}")

# Gap（過学習の指標）
gap = train_score - cv_scores.mean()
print(f"Gap (過学習): {gap:.4f}")

# 目標達成チェック
print("\n【目標達成チェック】")
target_cv = 0.832
target_gap = 0.015

cv_check = "✓" if cv_scores.mean() >= target_cv else "✗"
gap_check = "✓" if gap <= target_gap else "✗"

print(f"{cv_check} CV Score >= {target_cv}: {cv_scores.mean():.4f}")
print(f"{gap_check} Gap <= {target_gap}: {gap:.4f}")

# ========================================================================
# 特徴量重要度
# ========================================================================
print("\n" + "="*80)
print("特徴量重要度")
print("="*80)

feature_importance = pd.DataFrame({
    'Feature': recommended_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n【全特徴量の重要度】")
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] / 5)
    print(f"{row['Feature']:20s} {row['Importance']:3.0f} {bar}")

print("\n【新規特徴量の重要度】")
new_features = feature_importance[feature_importance['Feature'].isin(['TicketGroup', 'Deck_Safe'])]
print(new_features)

# ========================================================================
# OOF (Out-of-Fold) 予測と詳細評価
# ========================================================================
print("\n" + "="*80)
print("Out-of-Fold予測と詳細評価")
print("="*80)

# OOF予測
oof_predictions = cross_val_predict(model, X_train, y_train, cv=cv, method='predict')
oof_score = accuracy_score(y_train, oof_predictions)

print(f"\nOOF Score: {oof_score:.4f}")

# Confusion Matrix
print("\n【Confusion Matrix】")
cm = confusion_matrix(y_train, oof_predictions)
print(cm)
print(f"\nTN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# Classification Report
print("\n【Classification Report】")
print(classification_report(y_train, oof_predictions, target_names=['Died', 'Survived']))

# ========================================================================
# ベースライン（9特徴）との比較
# ========================================================================
print("\n" + "="*80)
print("ベースライン（9特徴）との比較")
print("="*80)

# ベースライン特徴量
base_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
                 'SmallFamily', 'Age_Missing', 'Embarked_Missing']

X_base = train_fe[base_features]

# ベースラインモデル訓練
model_base = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=2,
    num_leaves=4,
    min_child_samples=30,
    reg_alpha=1.0,
    reg_lambda=1.0,
    min_split_gain=0.01,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

cv_scores_base = cross_val_score(model_base, X_base, y_train, cv=cv, scoring='accuracy')
model_base.fit(X_base, y_train)
train_score_base = model_base.score(X_base, y_train)
gap_base = train_score_base - cv_scores_base.mean()

print("\n【比較結果】")
comparison = pd.DataFrame({
    'モデル': ['ベースライン (9特徴)', '推奨構成 (11特徴)'],
    '特徴量数': [len(base_features), len(recommended_features)],
    'CV Score': [cv_scores_base.mean(), cv_scores.mean()],
    'CV Std': [cv_scores_base.std(), cv_scores.std()],
    'Train Score': [train_score_base, train_score],
    'Gap': [gap_base, gap],
    'OOF Score': [
        accuracy_score(y_train, cross_val_predict(model_base, X_base, y_train, cv=cv)),
        oof_score
    ]
})

print(comparison.to_string(index=False))

# 改善率
cv_improvement = cv_scores.mean() - cv_scores_base.mean()
gap_improvement = gap_base - gap
print(f"\n【改善】")
print(f"CV Score: +{cv_improvement:.4f} ({cv_improvement/cv_scores_base.mean()*100:+.2f}%)")
print(f"Gap改善: {gap_improvement:+.4f}")

# ========================================================================
# テストデータ予測
# ========================================================================
print("\n" + "="*80)
print("テストデータ予測")
print("="*80)

# 予測
test_predictions = model.predict(X_test)
test_pred_proba = model.predict_proba(X_test)[:, 1]

# 確信度の分布
print("\n【予測確信度の分布】")
print(f"平均確信度: {np.abs(test_pred_proba - 0.5).mean():.4f}")
print(f"低確信度サンプル（|p-0.5| < 0.1）: {np.sum(np.abs(test_pred_proba - 0.5) < 0.1)}個")

# Submission作成
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})

output_file = 'submission_recommended_11features.csv'
submission.to_csv(output_file, index=False)
print(f"\n✓ Submission保存: {output_file}")

# ========================================================================
# まとめ
# ========================================================================
print("\n" + "="*80)
print("まとめ")
print("="*80)

print(f"""
【最終結果】
- 特徴量数: {len(recommended_features)}
- CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
- OOF Score: {oof_score:.4f}
- Train Score: {train_score:.4f}
- Gap: {gap:.4f}

【ベースラインからの改善】
- CV Score: {cv_scores_base.mean():.4f} → {cv_scores.mean():.4f} (+{cv_improvement:.4f})
- Gap: {gap_base:.4f} → {gap:.4f} ({gap_improvement:+.4f})

【多重共線性】
- VIF > 10の特徴量: {len(high_vif)}個
- 高相関ペア: {len(high_corr_pairs)}個

【目標達成状況】
- CV Score >= 0.832: {cv_check}
- Gap <= 0.015: {gap_check}
""")

print("="*80)
print("評価完了")
print("="*80)
