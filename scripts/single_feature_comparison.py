"""
TicketGroupのみ vs Deck_Safeのみ の比較

目的: なぜ組み合わせると効果がないのかを確認
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("単一特徴追加の比較: TicketGroup vs Deck_Safe")
print("="*80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

# 特徴量エンジニアリング
def engineer_features(df, add_ticketgroup=False, add_deck=False):
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

    if add_ticketgroup:
        df_fe['TicketGroup'] = df_fe['Ticket'].map(df_fe['Ticket'].value_counts())

    if add_deck:
        df_fe['Deck'] = df_fe['Cabin'].str[0]
        df_fe['Deck_Safe'] = ((df_fe['Deck'] == 'D') | (df_fe['Deck'] == 'E') | (df_fe['Deck'] == 'B')).astype(int)

    return df_fe

# データ結合
all_data = pd.concat([train, test], sort=False, ignore_index=True)

# モデル設定
model = lgb.LGBMClassifier(
    n_estimators=100, max_depth=2, num_leaves=4, min_child_samples=30,
    reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.01,
    learning_rate=0.05, random_state=42, verbose=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ========================================================================
# ケース1: ベースライン（9特徴）
# ========================================================================
print("\n【ケース1: ベースライン（9特徴）】")

all_data_base = engineer_features(all_data, add_ticketgroup=False, add_deck=False)
train_base = all_data_base[:len(train)]
y_train = train['Survived']

base_features = ['Pclass', 'Sex', 'Title', 'Age', 'IsAlone', 'FareBin',
                 'SmallFamily', 'Age_Missing', 'Embarked_Missing']
X_base = train_base[base_features]

cv_scores_base = cross_val_score(model, X_base, y_train, cv=cv, scoring='accuracy')
model.fit(X_base, y_train)
train_score_base = model.score(X_base, y_train)

print(f"特徴量数: {len(base_features)}")
print(f"CV Score: {cv_scores_base.mean():.4f} (±{cv_scores_base.std():.4f})")
print(f"Train Score: {train_score_base:.4f}")
print(f"Gap: {train_score_base - cv_scores_base.mean():.4f}")

# ========================================================================
# ケース2: + TicketGroupのみ（10特徴）
# ========================================================================
print("\n【ケース2: + TicketGroupのみ（10特徴）】")

all_data_ticket = engineer_features(all_data, add_ticketgroup=True, add_deck=False)
train_ticket = all_data_ticket[:len(train)]

ticket_features = base_features + ['TicketGroup']
X_ticket = train_ticket[ticket_features]

cv_scores_ticket = cross_val_score(model, X_ticket, y_train, cv=cv, scoring='accuracy')
model.fit(X_ticket, y_train)
train_score_ticket = model.score(X_ticket, y_train)

print(f"特徴量数: {len(ticket_features)}")
print(f"CV Score: {cv_scores_ticket.mean():.4f} (±{cv_scores_ticket.std():.4f})")
print(f"Train Score: {train_score_ticket:.4f}")
print(f"Gap: {train_score_ticket - cv_scores_ticket.mean():.4f}")
print(f"\nベースラインからの変化:")
print(f"  CV Score: {cv_scores_ticket.mean() - cv_scores_base.mean():+.4f}")
print(f"  Gap: {(train_score_ticket - cv_scores_ticket.mean()) - (train_score_base - cv_scores_base.mean()):+.4f}")

# ========================================================================
# ケース3: + Deck_Safeのみ（10特徴）
# ========================================================================
print("\n【ケース3: + Deck_Safeのみ（10特徴）】")

all_data_deck = engineer_features(all_data, add_ticketgroup=False, add_deck=True)
train_deck = all_data_deck[:len(train)]

deck_features = base_features + ['Deck_Safe']
X_deck = train_deck[deck_features]

cv_scores_deck = cross_val_score(model, X_deck, y_train, cv=cv, scoring='accuracy')
model.fit(X_deck, y_train)
train_score_deck = model.score(X_deck, y_train)

print(f"特徴量数: {len(deck_features)}")
print(f"CV Score: {cv_scores_deck.mean():.4f} (±{cv_scores_deck.std():.4f})")
print(f"Train Score: {train_score_deck:.4f}")
print(f"Gap: {train_score_deck - cv_scores_deck.mean():.4f}")
print(f"\nベースラインからの変化:")
print(f"  CV Score: {cv_scores_deck.mean() - cv_scores_base.mean():+.4f}")
print(f"  Gap: {(train_score_deck - cv_scores_deck.mean()) - (train_score_base - cv_scores_base.mean()):+.4f}")

# ========================================================================
# ケース4: + TicketGroup + Deck_Safe（11特徴）
# ========================================================================
print("\n【ケース4: + TicketGroup + Deck_Safe（11特徴）】")

all_data_both = engineer_features(all_data, add_ticketgroup=True, add_deck=True)
train_both = all_data_both[:len(train)]

both_features = base_features + ['TicketGroup', 'Deck_Safe']
X_both = train_both[both_features]

cv_scores_both = cross_val_score(model, X_both, y_train, cv=cv, scoring='accuracy')
model.fit(X_both, y_train)
train_score_both = model.score(X_both, y_train)

print(f"特徴量数: {len(both_features)}")
print(f"CV Score: {cv_scores_both.mean():.4f} (±{cv_scores_both.std():.4f})")
print(f"Train Score: {train_score_both:.4f}")
print(f"Gap: {train_score_both - cv_scores_both.mean():.4f}")
print(f"\nベースラインからの変化:")
print(f"  CV Score: {cv_scores_both.mean() - cv_scores_base.mean():+.4f}")
print(f"  Gap: {(train_score_both - cv_scores_both.mean()) - (train_score_base - cv_scores_base.mean()):+.4f}")

# ========================================================================
# 総合比較
# ========================================================================
print("\n" + "="*80)
print("総合比較")
print("="*80)

comparison = pd.DataFrame({
    'ケース': ['ベースライン', '+ TicketGroup', '+ Deck_Safe', '+ Both'],
    '特徴量数': [len(base_features), len(ticket_features), len(deck_features), len(both_features)],
    'CV Score': [cv_scores_base.mean(), cv_scores_ticket.mean(),
                 cv_scores_deck.mean(), cv_scores_both.mean()],
    'Train Score': [train_score_base, train_score_ticket,
                    train_score_deck, train_score_both],
    'Gap': [train_score_base - cv_scores_base.mean(),
            train_score_ticket - cv_scores_ticket.mean(),
            train_score_deck - cv_scores_deck.mean(),
            train_score_both - cv_scores_both.mean()],
    'CV変化': [0,
               cv_scores_ticket.mean() - cv_scores_base.mean(),
               cv_scores_deck.mean() - cv_scores_base.mean(),
               cv_scores_both.mean() - cv_scores_base.mean()]
})

print("\n")
print(comparison.to_string(index=False))

# 最良のケース
best_idx = comparison['CV Score'].idxmax()
print(f"\n【最良のケース】")
print(f"{comparison.iloc[best_idx]['ケース']}")
print(f"CV Score: {comparison.iloc[best_idx]['CV Score']:.4f}")

# 相互作用の評価
expected_improvement = (cv_scores_ticket.mean() - cv_scores_base.mean()) + \
                      (cv_scores_deck.mean() - cv_scores_base.mean())
actual_improvement = cv_scores_both.mean() - cv_scores_base.mean()
interaction_effect = actual_improvement - expected_improvement

print(f"\n【相互作用の評価】")
print(f"期待改善（単純加算）: {expected_improvement:+.4f}")
print(f"実際の改善: {actual_improvement:+.4f}")
print(f"相互作用効果: {interaction_effect:+.4f}")

if interaction_effect < -0.001:
    print("→ 負の相互作用あり（特徴量が互いに干渉している）")
elif interaction_effect > 0.001:
    print("→ 正の相互作用あり（特徴量が相乗効果を発揮）")
else:
    print("→ 相互作用なし（独立）")

print("\n" + "="*80)
print("分析完了")
print("="*80)
