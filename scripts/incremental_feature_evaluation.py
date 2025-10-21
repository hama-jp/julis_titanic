"""
優先度1特徴量の段階的評価
各特徴量を1つずつ追加してモデル性能と多重共線性を評価

実施内容:
1. ベースライン（現在の9特徴量）
2. + TicketGroup関連特徴量
3. + Deck関連特徴量
4. + CabinShared関連特徴量
5. + Embarked関連特徴量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 設定
np.random.seed(42)
plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style('whitegrid')

print("="*80)
print("優先度1特徴量の段階的評価")
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
# 特徴量エンジニアリング関数
# ========================================================================

def engineer_base_features(df):
    """ベース特徴量エンジニアリング（既存の9特徴量）"""
    df_fe = df.copy()

    # 1. 欠損フラグ（補完前）
    df_fe['Age_Missing'] = df_fe['Age'].isnull().astype(int)
    df_fe['Cabin_Missing'] = df_fe['Cabin'].isnull().astype(int)
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

    # 4. Age補完（Title×Pclassごとの中央値）
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

    return df_fe

def add_ticketgroup_features(df):
    """TicketGroup関連特徴量を追加"""
    df_new = df.copy()

    # Ticket出現回数
    df_new['TicketGroup'] = df_new['Ticket'].map(df_new['Ticket'].value_counts())
    df_new['IsTicketGroup'] = (df_new['TicketGroup'] > 1).astype(int)
    df_new['OptimalTicketGroup'] = ((df_new['TicketGroup'] >= 2) & (df_new['TicketGroup'] <= 4)).astype(int)

    return df_new

def add_deck_features(df):
    """Deck関連特徴量を追加"""
    df_new = df.copy()

    # Cabinの最初の文字を抽出
    df_new['Deck'] = df_new['Cabin'].str[0]

    # 上位3つのDeck（D, E, B）をバイナリ特徴量に
    df_new['Deck_D'] = (df_new['Deck'] == 'D').astype(int)
    df_new['Deck_E'] = (df_new['Deck'] == 'E').astype(int)
    df_new['Deck_B'] = (df_new['Deck'] == 'B').astype(int)

    # 安全な甲板フラグ（D, E, Bのいずれか）
    df_new['Deck_Safe'] = ((df_new['Deck'] == 'D') | (df_new['Deck'] == 'E') | (df_new['Deck'] == 'B')).astype(int)

    return df_new

def add_cabinshared_features(df):
    """CabinShared関連特徴量を追加"""
    df_new = df.copy()

    # 同じCabinの人数（NaNの場合は1とする）
    df_new['CabinShared'] = df_new.groupby('Cabin')['PassengerId'].transform('count')
    # NaN（Cabin情報なし）の場合は1（単独）とする
    df_new['CabinShared'] = df_new['CabinShared'].fillna(1)

    df_new['IsCabinShared'] = (df_new['CabinShared'] > 1).astype(int)

    # 最適なCabin共有人数（2-3人）
    df_new['OptimalCabinShared'] = ((df_new['CabinShared'] >= 2) & (df_new['CabinShared'] <= 3)).astype(int)

    return df_new

def add_embarked_features(df):
    """Embarked関連特徴量を追加"""
    df_new = df.copy()

    # One-hot encoding
    df_new['Embarked_C'] = (df_new['Embarked'] == 'C').astype(int)
    df_new['Embarked_Q'] = (df_new['Embarked'] == 'Q').astype(int)
    df_new['Embarked_S'] = (df_new['Embarked'] == 'S').astype(int)

    return df_new

# ========================================================================
# 多重共線性評価関数
# ========================================================================

def calculate_vif(X, feature_names):
    """VIF（分散拡大係数）を計算"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names

    # VIFを計算（エラー処理付き）
    vif_values = []
    for i in range(len(feature_names)):
        try:
            vif = variance_inflation_factor(X.values, i)
            # inf/nanの場合は大きな値に置き換え
            if np.isinf(vif) or np.isnan(vif):
                vif = 999999
            vif_values.append(vif)
        except:
            vif_values.append(999999)

    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values('VIF', ascending=False)
    return vif_data

def check_correlation(X, feature_names, threshold=0.7):
    """相関係数を計算して高相関ペアを検出"""
    corr_matrix = pd.DataFrame(X, columns=feature_names).corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'Feature1': corr_matrix.columns[i],
                    'Feature2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })

    return pd.DataFrame(high_corr_pairs)

# ========================================================================
# モデル評価関数
# ========================================================================

def evaluate_model(X, y, feature_names, model_name="LightGBM"):
    """モデルを評価（CV Score、特徴量重要度）"""

    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

    # CV評価
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # 全データで訓練して特徴量重要度を取得
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Train score
    train_score = model.score(X, y)

    results = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_score': train_score,
        'gap': train_score - cv_scores.mean(),
        'feature_importance': feature_importance
    }

    return results

# ========================================================================
# メイン評価プロセス
# ========================================================================

# データ結合（特徴量エンジニアリングのため）
all_data = pd.concat([train, test], sort=False, ignore_index=True)

# ベース特徴量エンジニアリング
all_data_fe = engineer_base_features(all_data)

# Train/Test分割
train_fe = all_data_fe[:len(train)].copy()
test_fe = all_data_fe[len(train):].copy()

y_train = train['Survived']

# ========================================================================
# 評価1: ベースライン（現在の9特徴量）
# ========================================================================
print("\n" + "="*80)
print("評価1: ベースライン（現在の9特徴量）")
print("="*80)

base_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
                 'SmallFamily', 'Age_Missing', 'Embarked_Missing']

X_base = train_fe[base_features]

print(f"\n特徴量: {base_features}")
print(f"特徴量数: {len(base_features)}")

# 多重共線性チェック
print("\n【多重共線性チェック - VIF】")
vif_base = calculate_vif(X_base, base_features)
print(vif_base)

print("\n【高相関ペア（|r| > 0.7）】")
high_corr_base = check_correlation(X_base, base_features)
if len(high_corr_base) > 0:
    print(high_corr_base)
else:
    print("高相関ペアなし（良好）")

# モデル評価
print("\n【モデル評価】")
results_base = evaluate_model(X_base, y_train, base_features)
print(f"CV Score: {results_base['cv_mean']:.4f} (±{results_base['cv_std']:.4f})")
print(f"Train Score: {results_base['train_score']:.4f}")
print(f"Gap (過学習): {results_base['gap']:.4f}")

print("\n【特徴量重要度 Top 5】")
print(results_base['feature_importance'].head())

# ========================================================================
# 評価2: + TicketGroup関連特徴量
# ========================================================================
print("\n" + "="*80)
print("評価2: + TicketGroup関連特徴量")
print("="*80)

# 特徴量追加（Train/Test両方に）
all_data_ticket = add_ticketgroup_features(all_data_fe)
train_ticket = all_data_ticket[:len(train)].copy()
test_ticket = all_data_ticket[len(train):].copy()

ticket_features = base_features + ['TicketGroup', 'IsTicketGroup', 'OptimalTicketGroup']
X_ticket = train_ticket[ticket_features]

print(f"\n追加特徴量: TicketGroup, IsTicketGroup, OptimalTicketGroup")
print(f"特徴量数: {len(base_features)} → {len(ticket_features)} (+{len(ticket_features) - len(base_features)})")

# 多重共線性チェック
print("\n【多重共線性チェック - VIF】")
vif_ticket = calculate_vif(X_ticket, ticket_features)
print(vif_ticket)

# VIF > 10の特徴量を警告
high_vif = vif_ticket[vif_ticket['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ VIF > 10の特徴量が{len(high_vif)}個検出されました:")
    print(high_vif)

print("\n【高相関ペア（|r| > 0.7）】")
high_corr_ticket = check_correlation(X_ticket, ticket_features)
if len(high_corr_ticket) > 0:
    print(high_corr_ticket)
else:
    print("高相関ペアなし（良好）")

# モデル評価
print("\n【モデル評価】")
results_ticket = evaluate_model(X_ticket, y_train, ticket_features)
print(f"CV Score: {results_ticket['cv_mean']:.4f} (±{results_ticket['cv_std']:.4f})")
print(f"Train Score: {results_ticket['train_score']:.4f}")
print(f"Gap (過学習): {results_ticket['gap']:.4f}")

print(f"\n【ベースラインとの比較】")
print(f"CV Score変化: {results_base['cv_mean']:.4f} → {results_ticket['cv_mean']:.4f} ({results_ticket['cv_mean'] - results_base['cv_mean']:+.4f})")
print(f"Gap変化: {results_base['gap']:.4f} → {results_ticket['gap']:.4f} ({results_ticket['gap'] - results_base['gap']:+.4f})")

print("\n【新規特徴量の重要度】")
new_features_importance = results_ticket['feature_importance'][results_ticket['feature_importance']['Feature'].isin(['TicketGroup', 'IsTicketGroup', 'OptimalTicketGroup'])]
print(new_features_importance)

# ========================================================================
# 評価3: + Deck関連特徴量
# ========================================================================
print("\n" + "="*80)
print("評価3: + Deck関連特徴量")
print("="*80)

# 特徴量追加
all_data_deck = add_deck_features(all_data_fe)
train_deck = all_data_deck[:len(train)].copy()
test_deck = all_data_deck[len(train):].copy()

deck_features = base_features + ['Deck_D', 'Deck_E', 'Deck_B', 'Deck_Safe']
X_deck = train_deck[deck_features]

print(f"\n追加特徴量: Deck_D, Deck_E, Deck_B, Deck_Safe")
print(f"特徴量数: {len(base_features)} → {len(deck_features)} (+{len(deck_features) - len(base_features)})")

# 多重共線性チェック
print("\n【多重共線性チェック - VIF】")
vif_deck = calculate_vif(X_deck, deck_features)
print(vif_deck)

high_vif = vif_deck[vif_deck['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ VIF > 10の特徴量が{len(high_vif)}個検出されました:")
    print(high_vif)

print("\n【高相関ペア（|r| > 0.7）】")
high_corr_deck = check_correlation(X_deck, deck_features)
if len(high_corr_deck) > 0:
    print(high_corr_deck)
else:
    print("高相関ペアなし（良好）")

# モデル評価
print("\n【モデル評価】")
results_deck = evaluate_model(X_deck, y_train, deck_features)
print(f"CV Score: {results_deck['cv_mean']:.4f} (±{results_deck['cv_std']:.4f})")
print(f"Train Score: {results_deck['train_score']:.4f}")
print(f"Gap (過学習): {results_deck['gap']:.4f}")

print(f"\n【ベースラインとの比較】")
print(f"CV Score変化: {results_base['cv_mean']:.4f} → {results_deck['cv_mean']:.4f} ({results_deck['cv_mean'] - results_base['cv_mean']:+.4f})")
print(f"Gap変化: {results_base['gap']:.4f} → {results_deck['gap']:.4f} ({results_deck['gap'] - results_base['gap']:+.4f})")

print("\n【新規特徴量の重要度】")
new_features_importance = results_deck['feature_importance'][results_deck['feature_importance']['Feature'].isin(['Deck_D', 'Deck_E', 'Deck_B', 'Deck_Safe'])]
print(new_features_importance)

# ========================================================================
# 評価4: + CabinShared関連特徴量
# ========================================================================
print("\n" + "="*80)
print("評価4: + CabinShared関連特徴量")
print("="*80)

# 特徴量追加
all_data_cabin = add_cabinshared_features(all_data_fe)
train_cabin = all_data_cabin[:len(train)].copy()
test_cabin = all_data_cabin[len(train):].copy()

cabin_features = base_features + ['CabinShared', 'IsCabinShared', 'OptimalCabinShared']
X_cabin = train_cabin[cabin_features]

print(f"\n追加特徴量: CabinShared, IsCabinShared, OptimalCabinShared")
print(f"特徴量数: {len(base_features)} → {len(cabin_features)} (+{len(cabin_features) - len(base_features)})")

# 多重共線性チェック
print("\n【多重共線性チェック - VIF】")
vif_cabin = calculate_vif(X_cabin, cabin_features)
print(vif_cabin)

high_vif = vif_cabin[vif_cabin['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ VIF > 10の特徴量が{len(high_vif)}個検出されました:")
    print(high_vif)

print("\n【高相関ペア（|r| > 0.7）】")
high_corr_cabin = check_correlation(X_cabin, cabin_features)
if len(high_corr_cabin) > 0:
    print(high_corr_cabin)
else:
    print("高相関ペアなし（良好）")

# モデル評価
print("\n【モデル評価】")
results_cabin = evaluate_model(X_cabin, y_train, cabin_features)
print(f"CV Score: {results_cabin['cv_mean']:.4f} (±{results_cabin['cv_std']:.4f})")
print(f"Train Score: {results_cabin['train_score']:.4f}")
print(f"Gap (過学習): {results_cabin['gap']:.4f}")

print(f"\n【ベースラインとの比較】")
print(f"CV Score変化: {results_base['cv_mean']:.4f} → {results_cabin['cv_mean']:.4f} ({results_cabin['cv_mean'] - results_base['cv_mean']:+.4f})")
print(f"Gap変化: {results_base['gap']:.4f} → {results_cabin['gap']:.4f} ({results_cabin['gap'] - results_base['gap']:+.4f})")

print("\n【新規特徴量の重要度】")
new_features_importance = results_cabin['feature_importance'][results_cabin['feature_importance']['Feature'].isin(['CabinShared', 'IsCabinShared', 'OptimalCabinShared'])]
print(new_features_importance)

# ========================================================================
# 評価5: + Embarked関連特徴量
# ========================================================================
print("\n" + "="*80)
print("評価5: + Embarked関連特徴量")
print("="*80)

# 特徴量追加
all_data_embarked = add_embarked_features(all_data_fe)
train_embarked = all_data_embarked[:len(train)].copy()
test_embarked = all_data_embarked[len(train):].copy()

# Embarked_Sは冗長なので除外（C, Qの2つで表現可能）
embarked_features = base_features + ['Embarked_C', 'Embarked_Q']
X_embarked = train_embarked[embarked_features]

print(f"\n追加特徴量: Embarked_C, Embarked_Q")
print(f"特徴量数: {len(base_features)} → {len(embarked_features)} (+{len(embarked_features) - len(base_features)})")

# 多重共線性チェック
print("\n【多重共線性チェック - VIF】")
vif_embarked = calculate_vif(X_embarked, embarked_features)
print(vif_embarked)

high_vif = vif_embarked[vif_embarked['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n⚠️ VIF > 10の特徴量が{len(high_vif)}個検出されました:")
    print(high_vif)

print("\n【高相関ペア（|r| > 0.7）】")
high_corr_embarked = check_correlation(X_embarked, embarked_features)
if len(high_corr_embarked) > 0:
    print(high_corr_embarked)
else:
    print("高相関ペアなし（良好）")

# モデル評価
print("\n【モデル評価】")
results_embarked = evaluate_model(X_embarked, y_train, embarked_features)
print(f"CV Score: {results_embarked['cv_mean']:.4f} (±{results_embarked['cv_std']:.4f})")
print(f"Train Score: {results_embarked['train_score']:.4f}")
print(f"Gap (過学習): {results_embarked['gap']:.4f}")

print(f"\n【ベースラインとの比較】")
print(f"CV Score変化: {results_base['cv_mean']:.4f} → {results_embarked['cv_mean']:.4f} ({results_embarked['cv_mean'] - results_base['cv_mean']:+.4f})")
print(f"Gap変化: {results_base['gap']:.4f} → {results_embarked['gap']:.4f} ({results_embarked['gap'] - results_base['gap']:+.4f})")

print("\n【新規特徴量の重要度】")
new_features_importance = results_embarked['feature_importance'][results_embarked['feature_importance']['Feature'].isin(['Embarked_C', 'Embarked_Q'])]
print(new_features_importance)

# ========================================================================
# 総合比較
# ========================================================================
print("\n" + "="*80)
print("総合比較サマリー")
print("="*80)

summary = pd.DataFrame({
    'モデル': ['ベースライン (9特徴)', '+ TicketGroup (+3)', '+ Deck (+4)', '+ CabinShared (+3)', '+ Embarked (+2)'],
    '特徴量数': [len(base_features), len(ticket_features), len(deck_features), len(cabin_features), len(embarked_features)],
    'CV Score': [results_base['cv_mean'], results_ticket['cv_mean'], results_deck['cv_mean'],
                 results_cabin['cv_mean'], results_embarked['cv_mean']],
    'CV Std': [results_base['cv_std'], results_ticket['cv_std'], results_deck['cv_std'],
               results_cabin['cv_std'], results_embarked['cv_std']],
    'Train Score': [results_base['train_score'], results_ticket['train_score'], results_deck['train_score'],
                    results_cabin['train_score'], results_embarked['train_score']],
    'Gap': [results_base['gap'], results_ticket['gap'], results_deck['gap'],
            results_cabin['gap'], results_embarked['gap']],
    'CV変化': [0, results_ticket['cv_mean'] - results_base['cv_mean'],
               results_deck['cv_mean'] - results_base['cv_mean'],
               results_cabin['cv_mean'] - results_base['cv_mean'],
               results_embarked['cv_mean'] - results_base['cv_mean']]
})

print("\n")
print(summary.to_string(index=False))

# 最良のモデルを特定
best_idx = summary['CV Score'].idxmax()
print(f"\n【最良のモデル】")
print(f"モデル: {summary.iloc[best_idx]['モデル']}")
print(f"CV Score: {summary.iloc[best_idx]['CV Score']:.4f}")
print(f"ベースラインからの改善: +{summary.iloc[best_idx]['CV変化']:.4f}")

# VIF警告のサマリー
print("\n【多重共線性の評価】")
print("VIF > 10の特徴量:")
for model_name, vif_data in [('ベースライン', vif_base), ('TicketGroup', vif_ticket),
                              ('Deck', vif_deck), ('CabinShared', vif_cabin), ('Embarked', vif_embarked)]:
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"  {model_name}: {len(high_vif)}個 - {', '.join(high_vif['Feature'].tolist())}")
    else:
        print(f"  {model_name}: なし ✓")

print("\n" + "="*80)
print("評価完了")
print("="*80)
