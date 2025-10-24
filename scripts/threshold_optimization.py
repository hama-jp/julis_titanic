"""
Threshold最適化

目的:
1. デフォルト閾値0.5を最適化
2. 生存者のRecall向上
3. 各種評価指標での最適閾値探索
4. 最適閾値でのSubmission作成
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve, confusion_matrix,
                             classification_report)
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("Threshold最適化")
print("="*80)

# ========================================================================
# データ読み込みと特徴量エンジニアリング
# ========================================================================
print("\n【データ準備】")
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

def engineer_features(df):
    """10特徴量のエンジニアリング"""
    df_fe = df.copy()
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Embarked_Missing'] = df_fe['Embarked'].isnull().astype(int)

    df_fe['Title'] = df_fe['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df_fe['Title'] = df_fe['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_fe['Title'] = df_fe['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_fe['Title'] = df_fe['Title'].replace('Mme', 'Mrs')
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df_fe['Title'] = df_fe['Title'].map(title_mapping).fillna(0)

    df_fe['FamilySize'] = df_fe['SibSp'] + df_fe['Parch'] + 1
    df_fe['IsAlone'] = (df_fe['FamilySize'] == 1).astype(int)
    df_fe['SmallFamily'] = ((df_fe['FamilySize'] >= 2) & (df_fe['FamilySize'] <= 4)).astype(int)

    for title in df_fe['Title'].unique():
        for pclass in df_fe['Pclass'].unique():
            mask = (df_fe['Title'] == title) & (df_fe['Pclass'] == pclass) & (df_fe['Age'].isnull())
            median_age = df_fe[(df_fe['Title'] == title) & (df_fe['Pclass'] == pclass)]['Age'].median()
            if pd.notna(median_age):
                df_fe.loc[mask, 'Age'] = median_age
    df_fe['Age'].fillna(df_fe['Age'].median(), inplace=True)

    df_fe['Fare'].fillna(df_fe.groupby('Pclass')['Fare'].transform('median'), inplace=True)
    df_fe['FareBin'] = pd.qcut(df_fe['Fare'], q=4, labels=False, duplicates='drop')
    df_fe['Sex'] = df_fe['Sex'].map({'male': 0, 'female': 1})
    df_fe['Embarked'].fillna(df_fe['Embarked'].mode()[0], inplace=True)

    df_fe['Deck'] = df_fe['Cabin'].str[0]
    df_fe['Deck_Safe'] = ((df_fe['Deck'] == 'D') | (df_fe['Deck'] == 'E') | (df_fe['Deck'] == 'B')).astype(int)

    return df_fe

all_data = pd.concat([train, test], sort=False, ignore_index=True)
all_data_fe = engineer_features(all_data)

train_fe = all_data_fe[:len(train)].copy()
test_fe = all_data_fe[len(train):].copy()

features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
            'SmallFamily', 'Age_Missing', 'Embarked_Missing', 'Deck_Safe']

X_train = train_fe[features]
y_train = train['Survived']
X_test = test_fe[features]

print(f"特徴量数: {len(features)}")
print(f"Train shape: {X_train.shape}")

# ========================================================================
# 最適モデルの構築
# ========================================================================
print("\n【最適モデルの構築】")

# 前回の最適設定を使用
lgb_model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=40,
    reg_alpha=1.5, reg_lambda=1.5, min_split_gain=0.02,
    learning_rate=0.05, random_state=42, verbose=-1
)

gb_model = GradientBoostingClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    min_samples_split=40, min_samples_leaf=20, max_features='sqrt',
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_split=20,
    min_samples_leaf=10, random_state=42
)

# Soft Voting（確率値取得のため）
soft_voting = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    voting='soft'
)

print("モデル: Soft Voting Ensemble (LightGBM + GB + RF)")

# ========================================================================
# OOF確率値の取得
# ========================================================================
print("\n【OOF確率値の取得】")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = cross_val_predict(soft_voting, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

print(f"OOF確率値の形状: {oof_proba.shape}")
print(f"確率値の範囲: [{oof_proba.min():.4f}, {oof_proba.max():.4f}]")

# ========================================================================
# 確率値分布の分析
# ========================================================================
print("\n" + "="*80)
print("確率値分布の分析")
print("="*80)

# 基本統計
print(f"\n【確率値の基本統計】")
print(f"平均: {oof_proba.mean():.4f}")
print(f"中央値: {np.median(oof_proba):.4f}")
print(f"標準偏差: {oof_proba.std():.4f}")

# クラス別の確率値分布
died_proba = oof_proba[y_train == 0]
survived_proba = oof_proba[y_train == 1]

print(f"\n【クラス別の確率値】")
print(f"死亡者の平均確率: {died_proba.mean():.4f} (低いほど正確)")
print(f"生存者の平均確率: {survived_proba.mean():.4f} (高いほど正確)")
print(f"分離度: {survived_proba.mean() - died_proba.mean():.4f}")

# ========================================================================
# ROC曲線とAUC
# ========================================================================
print("\n" + "="*80)
print("ROC曲線分析")
print("="*80)

fpr, tpr, roc_thresholds = roc_curve(y_train, oof_proba)
roc_auc = auc(fpr, tpr)

print(f"\n【ROC AUC】: {roc_auc:.4f}")

# Youden's Index（最大化）
youden_index = tpr - fpr
optimal_idx_youden = np.argmax(youden_index)
optimal_threshold_youden = roc_thresholds[optimal_idx_youden]

print(f"\n【Youden's Index最適閾値】")
print(f"閾値: {optimal_threshold_youden:.4f}")
print(f"TPR: {tpr[optimal_idx_youden]:.4f}")
print(f"FPR: {fpr[optimal_idx_youden]:.4f}")
print(f"Youden's Index: {youden_index[optimal_idx_youden]:.4f}")

# ========================================================================
# Precision-Recall曲線
# ========================================================================
print("\n" + "="*80)
print("Precision-Recall曲線分析")
print("="*80)

precision, recall, pr_thresholds = precision_recall_curve(y_train, oof_proba)

# F1-score最大化
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx_f1 = np.argmax(f1_scores)
optimal_threshold_f1 = pr_thresholds[optimal_idx_f1]

print(f"\n【F1-Score最適閾値】")
print(f"閾値: {optimal_threshold_f1:.4f}")
print(f"Precision: {precision[optimal_idx_f1]:.4f}")
print(f"Recall: {recall[optimal_idx_f1]:.4f}")
print(f"F1-Score: {f1_scores[optimal_idx_f1]:.4f}")

# ========================================================================
# 複数の閾値候補での評価
# ========================================================================
print("\n" + "="*80)
print("複数の閾値候補での詳細評価")
print("="*80)

threshold_candidates = {
    'Default (0.5)': 0.5,
    'Youden (ROC)': optimal_threshold_youden,
    'F1-Score': optimal_threshold_f1,
    'High Recall (0.4)': 0.4,
    'High Precision (0.6)': 0.6,
    'Balanced (0.45)': 0.45,
}

results = []

for name, threshold in threshold_candidates.items():
    pred = (oof_proba >= threshold).astype(int)

    acc = accuracy_score(y_train, pred)
    prec = precision_score(y_train, pred)
    rec = recall_score(y_train, pred)
    f1 = f1_score(y_train, pred)

    # クラス別のRecall
    cm = confusion_matrix(y_train, pred)
    recall_died = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall_survived = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    results.append({
        '閾値名': name,
        '閾値': threshold,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Recall_Died': recall_died,
        'Recall_Survived': recall_survived
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n【閾値候補の性能比較】")
print(results_df.to_string(index=False))

# 最高Accuracyの閾値
best_acc_idx = results_df['Accuracy'].idxmax()
best_threshold_name = results_df.loc[best_acc_idx, '閾値名']
best_threshold_value = results_df.loc[best_acc_idx, '閾値']

print(f"\n【最高Accuracy閾値】: {best_threshold_name} ({best_threshold_value:.4f})")
print(f"Accuracy: {results_df.loc[best_acc_idx, 'Accuracy']:.4f}")
print(f"Recall (Survived): {results_df.loc[best_acc_idx, 'Recall_Survived']:.4f}")

# 生存者Recall最高の閾値
best_recall_idx = results_df['Recall_Survived'].idxmax()
best_recall_name = results_df.loc[best_recall_idx, '閾値名']
best_recall_value = results_df.loc[best_recall_idx, '閾値']

print(f"\n【最高Recall(Survived)閾値】: {best_recall_name} ({best_recall_value:.4f})")
print(f"Accuracy: {results_df.loc[best_recall_idx, 'Accuracy']:.4f}")
print(f"Recall (Survived): {results_df.loc[best_recall_idx, 'Recall_Survived']:.4f}")

# ========================================================================
# デフォルト閾値との詳細比較
# ========================================================================
print("\n" + "="*80)
print("デフォルト閾値(0.5)との詳細比較")
print("="*80)

default_pred = (oof_proba >= 0.5).astype(int)
optimal_pred = (oof_proba >= best_threshold_value).astype(int)

print("\n【デフォルト閾値 (0.5)】")
print(classification_report(y_train, default_pred, target_names=['Died', 'Survived']))

print("\n【最適閾値】", f"({best_threshold_name}: {best_threshold_value:.4f})")
print(classification_report(y_train, optimal_pred, target_names=['Died', 'Survived']))

# 改善度
default_acc = accuracy_score(y_train, default_pred)
optimal_acc = accuracy_score(y_train, optimal_pred)
default_recall_survived = recall_score(y_train, default_pred)
optimal_recall_survived = recall_score(y_train, optimal_pred)

print("\n【改善度】")
print(f"Accuracy: {default_acc:.4f} → {optimal_acc:.4f} ({optimal_acc - default_acc:+.4f})")
print(f"Recall (Survived): {default_recall_survived:.4f} → {optimal_recall_survived:.4f} ({optimal_recall_survived - default_recall_survived:+.4f})")

# ========================================================================
# テストデータ予測とSubmission作成
# ========================================================================
print("\n" + "="*80)
print("テストデータ予測とSubmission作成")
print("="*80)

# モデル再訓練
soft_voting.fit(X_train, y_train)
test_proba = soft_voting.predict_proba(X_test)[:, 1]

print(f"\nテスト確率値の範囲: [{test_proba.min():.4f}, {test_proba.max():.4f}]")

# デフォルト閾値(0.5)
default_test_pred = (test_proba >= 0.5).astype(int)
default_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': default_test_pred
})
default_file = 'submission_threshold_default_0.5.csv'
default_submission.to_csv(default_file, index=False)
print(f"\n✓ デフォルト閾値: {default_file}")
print(f"  生存予測数: {default_test_pred.sum()}/{len(default_test_pred)} ({default_test_pred.sum()/len(default_test_pred)*100:.1f}%)")

# 最適閾値（Accuracy最大）
optimal_test_pred = (test_proba >= best_threshold_value).astype(int)
optimal_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': optimal_test_pred
})
optimal_file = f'submission_threshold_optimal_{best_threshold_name.replace(" ", "_").replace("(", "").replace(")", "")}.csv'
optimal_submission.to_csv(optimal_file, index=False)
print(f"\n✓ 最適閾値 ({best_threshold_name}): {optimal_file}")
print(f"  閾値: {best_threshold_value:.4f}")
print(f"  生存予測数: {optimal_test_pred.sum()}/{len(optimal_test_pred)} ({optimal_test_pred.sum()/len(optimal_test_pred)*100:.1f}%)")

# 生存者Recall最適閾値
recall_test_pred = (test_proba >= best_recall_value).astype(int)
recall_submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': recall_test_pred
})
recall_file = f'submission_threshold_high_recall_{best_recall_value:.2f}.csv'
recall_submission.to_csv(recall_file, index=False)
print(f"\n✓ 高Recall閾値 ({best_recall_name}): {recall_file}")
print(f"  閾値: {best_recall_value:.4f}")
print(f"  生存予測数: {recall_test_pred.sum()}/{len(recall_test_pred)} ({recall_test_pred.sum()/len(recall_test_pred)*100:.1f}%)")

# ========================================================================
# まとめ
# ========================================================================
print("\n" + "="*80)
print("まとめ")
print("="*80)

print(f"""
【使用モデル】
- Soft Voting Ensemble (LightGBM + GradientBoosting + RandomForest)

【閾値最適化結果】
1. デフォルト閾値 (0.5):
   - OOF Accuracy: {default_acc:.4f}
   - Recall (Survived): {default_recall_survived:.4f}

2. 最適閾値 ({best_threshold_name}: {best_threshold_value:.4f}):
   - OOF Accuracy: {optimal_acc:.4f} ({optimal_acc - default_acc:+.4f})
   - Recall (Survived): {optimal_recall_survived:.4f} ({optimal_recall_survived - default_recall_survived:+.4f})

3. 高Recall閾値 ({best_recall_name}: {best_recall_value:.4f}):
   - OOF Accuracy: {results_df.loc[best_recall_idx, 'Accuracy']:.4f}
   - Recall (Survived): {results_df.loc[best_recall_idx, 'Recall_Survived']:.4f}

【ROC分析】
- AUC: {roc_auc:.4f}
- Youden's Index最適閾値: {optimal_threshold_youden:.4f}

【Precision-Recall分析】
- F1-Score最適閾値: {optimal_threshold_f1:.4f}

【作成されたSubmission】
- {default_file} (デフォルト)
- {optimal_file} (Accuracy最大)
- {recall_file} (Recall最大)

【推奨Submission】
{'- ' + optimal_file if optimal_acc > default_acc else '- ' + default_file}
""")

# 改善があった場合のメッセージ
if optimal_acc > default_acc:
    print(f"✓ 閾値最適化により Accuracy {optimal_acc - default_acc:+.4f} の改善")
else:
    print(f"⚠️ デフォルト閾値(0.5)が既に最適に近い")

print("\n" + "="*80)
print("完了")
print("="*80)
