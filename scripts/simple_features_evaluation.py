"""
シンプルな新特徴量の評価

Sex×Pclass交互作用が失敗したため、よりシンプルで独立した特徴量を評価:

1. IsAlone: 単独乗船フラグ (FamilySize == 1)
2. FarePerPerson: 1人当たり運賃 (Fare / FamilySize)
3. AgeGroup: 年齢グループ (Child, YoungAdult, Adult, Senior)

評価方法: 1特徴ずつ追加して効果を検証
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("シンプルな新特徴量の段階的評価")
print("=" * 80)

# データ読み込み
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')

def add_title_feature(df):
    """Titleカラム追加"""
    df_new = df.copy()
    df_new['Title'] = df_new['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
        'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1, 'Mme': 2,
        'Don': 4, 'Dona': 4, 'Lady': 4, 'Countess': 4, 'Jonkheer': 4, 'Sir': 4,
        'Capt': 4, 'Ms': 1
    }
    df_new['Title'] = df_new['Title'].map(title_mapping).fillna(4)
    return df_new

def add_familysize_feature(df):
    """FamilySizeカラム追加"""
    df_new = df.copy()
    df_new['FamilySize'] = df_new['SibSp'] + df_new['Parch'] + 1
    return df_new

def add_isalone_feature(df):
    """IsAlone: 単独乗船フラグ"""
    df_new = df.copy()
    df_new['IsAlone'] = (df_new['FamilySize'] == 1).astype(int)
    return df_new

def add_fareperperson_feature(df):
    """FarePerPerson: 1人当たり運賃"""
    df_new = df.copy()
    df_new['FarePerPerson'] = df_new['Fare'] / df_new['FamilySize']
    # inf/nan対策
    df_new['FarePerPerson'] = df_new['FarePerPerson'].replace([np.inf, -np.inf], np.nan)
    df_new['FarePerPerson'] = df_new['FarePerPerson'].fillna(df_new['FarePerPerson'].median())
    return df_new

def add_agegroup_feature(df):
    """AgeGroup: 年齢グループ"""
    df_new = df.copy()

    # 年齢グループ定義
    def categorize_age(age):
        if pd.isna(age):
            return 3  # Unknown -> Adult扱い
        elif age < 16:
            return 0  # Child
        elif age < 30:
            return 1  # YoungAdult
        elif age < 60:
            return 2  # Adult
        else:
            return 3  # Senior

    df_new['AgeGroup'] = df_new['Age'].apply(categorize_age)
    return df_new

def preprocess_data(df):
    """前処理"""
    df_new = df.copy()
    df_new['Age'] = df_new['Age'].fillna(df_new['Age'].median())
    df_new['Fare'] = df_new['Fare'].fillna(df_new['Fare'].median())
    df_new['Embarked'] = df_new['Embarked'].fillna('S')
    df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})
    return df_new

def calculate_vif(X, feature_names):
    """VIF計算（簡易版）"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_values = []

    for i in range(len(feature_names)):
        try:
            vif = variance_inflation_factor(X.values, i)
            if np.isinf(vif) or np.isnan(vif) or vif > 1000:
                vif = 999
            vif_values.append(vif)
        except:
            vif_values.append(999)

    vif_data["VIF"] = vif_values
    return vif_data.sort_values('VIF', ascending=False)

def evaluate_model(X_train, y_train, X_test, feature_names):
    """モデル評価（OOF計算）"""

    # モデル定義
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, num_leaves=8,
        learning_rate=0.05, random_state=42, verbose=-1
    )
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.05, random_state=42
    )
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None,
        min_samples_split=10, min_samples_leaf=4, random_state=42
    )

    hard_voting = VotingClassifier(
        estimators=[('lgb', lgb_model), ('gb', gb_model), ('rf', rf_model)],
        voting='hard'
    )

    # CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X_train))
    test_predictions_list = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        hard_voting.fit(X_fold_train, y_fold_train)
        val_pred = hard_voting.predict(X_fold_val)
        oof_predictions[val_idx] = val_pred

        test_pred = hard_voting.predict(X_test)
        test_predictions_list.append(test_pred)

    oof_score = accuracy_score(y_train, oof_predictions)
    test_predictions = np.round(np.array(test_predictions_list).mean(axis=0)).astype(int)

    # VIF計算
    vif_df = calculate_vif(X_train, feature_names)
    max_vif = vif_df['VIF'].max()

    return oof_score, test_predictions, max_vif, vif_df

# ========================================
# ベースライン（9特徴量）
# ========================================
print("\n" + "=" * 80)
print("ベースライン: 9特徴量")
print("=" * 80)

train_baseline = preprocess_data(train.copy())
train_baseline = add_title_feature(train_baseline)
train_baseline = add_familysize_feature(train_baseline)
train_baseline = pd.get_dummies(train_baseline, columns=['Embarked'], drop_first=True)

test_baseline = preprocess_data(test.copy())
test_baseline = add_title_feature(test_baseline)
test_baseline = add_familysize_feature(test_baseline)
test_baseline = pd.get_dummies(test_baseline, columns=['Embarked'], drop_first=True)

base_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']
embarked_cols = [col for col in train_baseline.columns if col.startswith('Embarked_')]
base_features.extend(embarked_cols)

X_train_baseline = train_baseline[base_features]
y_train = train_baseline['Survived']
X_test_baseline = test_baseline[base_features]

print(f"特徴量数: {len(base_features)}")
baseline_oof, _, baseline_vif, _ = evaluate_model(X_train_baseline, y_train, X_test_baseline, base_features)
print(f"OOF Score: {baseline_oof:.4f}")
print(f"最大VIF: {baseline_vif:.2f}")

# ========================================
# 評価1: IsAlone追加
# ========================================
print("\n" + "=" * 80)
print("評価1: IsAlone（単独乗船フラグ）追加")
print("=" * 80)

train_isalone = train_baseline.copy()
train_isalone = add_isalone_feature(train_isalone)

test_isalone = test_baseline.copy()
test_isalone = add_isalone_feature(test_isalone)

features_isalone = base_features + ['IsAlone']
X_train_isalone = train_isalone[features_isalone]
X_test_isalone = test_isalone[features_isalone]

print(f"特徴量数: {len(features_isalone)}")
oof_isalone, test_pred_isalone, vif_isalone, vif_df_isalone = evaluate_model(
    X_train_isalone, y_train, X_test_isalone, features_isalone
)

print(f"OOF Score: {oof_isalone:.4f}")
print(f"ベースライン比: {oof_isalone - baseline_oof:+.4f} ({(oof_isalone - baseline_oof)*100:+.2f}%)")
print(f"最大VIF: {vif_isalone:.2f}")
print(f"IsAlone VIF: {vif_df_isalone[vif_df_isalone['Feature'] == 'IsAlone']['VIF'].values[0]:.2f}")

# ========================================
# 評価2: FarePerPerson追加
# ========================================
print("\n" + "=" * 80)
print("評価2: FarePerPerson（1人当たり運賃）追加")
print("=" * 80)

train_fpp = train_baseline.copy()
train_fpp = add_fareperperson_feature(train_fpp)

test_fpp = test_baseline.copy()
test_fpp = add_fareperperson_feature(test_fpp)

features_fpp = base_features + ['FarePerPerson']
X_train_fpp = train_fpp[features_fpp]
X_test_fpp = test_fpp[features_fpp]

print(f"特徴量数: {len(features_fpp)}")
oof_fpp, test_pred_fpp, vif_fpp, vif_df_fpp = evaluate_model(
    X_train_fpp, y_train, X_test_fpp, features_fpp
)

print(f"OOF Score: {oof_fpp:.4f}")
print(f"ベースライン比: {oof_fpp - baseline_oof:+.4f} ({(oof_fpp - baseline_oof)*100:+.2f}%)")
print(f"最大VIF: {vif_fpp:.2f}")
print(f"FarePerPerson VIF: {vif_df_fpp[vif_df_fpp['Feature'] == 'FarePerPerson']['VIF'].values[0]:.2f}")

# ========================================
# 評価3: AgeGroup追加
# ========================================
print("\n" + "=" * 80)
print("評価3: AgeGroup（年齢グループ）追加")
print("=" * 80)

train_agegroup = train_baseline.copy()
train_agegroup = add_agegroup_feature(train_agegroup)

test_agegroup = test_baseline.copy()
test_agegroup = add_agegroup_feature(test_agegroup)

features_agegroup = base_features + ['AgeGroup']
X_train_agegroup = train_agegroup[features_agegroup]
X_test_agegroup = test_agegroup[features_agegroup]

print(f"特徴量数: {len(features_agegroup)}")
oof_agegroup, test_pred_agegroup, vif_agegroup, vif_df_agegroup = evaluate_model(
    X_train_agegroup, y_train, X_test_agegroup, features_agegroup
)

print(f"OOF Score: {oof_agegroup:.4f}")
print(f"ベースライン比: {oof_agegroup - baseline_oof:+.4f} ({(oof_agegroup - baseline_oof)*100:+.2f}%)")
print(f"最大VIF: {vif_agegroup:.2f}")
print(f"AgeGroup VIF: {vif_df_agegroup[vif_df_agegroup['Feature'] == 'AgeGroup']['VIF'].values[0]:.2f}")

# ========================================
# 結果サマリー
# ========================================
print("\n" + "=" * 80)
print("評価結果サマリー")
print("=" * 80)

results = pd.DataFrame({
    '特徴量': ['ベースライン（9特徴）', '+IsAlone', '+FarePerPerson', '+AgeGroup'],
    '特徴量数': [len(base_features), len(features_isalone), len(features_fpp), len(features_agegroup)],
    'OOF Score': [baseline_oof, oof_isalone, oof_fpp, oof_agegroup],
    'OOF改善': [0, oof_isalone - baseline_oof, oof_fpp - baseline_oof, oof_agegroup - baseline_oof],
    '最大VIF': [baseline_vif, vif_isalone, vif_fpp, vif_agegroup]
})

print("\n" + results.to_string(index=False))

# 最良の特徴量を特定
best_idx = results['OOF Score'].idxmax()
best_feature = results.iloc[best_idx]['特徴量']
best_oof = results.iloc[best_idx]['OOF Score']
best_improvement = results.iloc[best_idx]['OOF改善']

print(f"\n最良の特徴量: {best_feature}")
print(f"OOF Score: {best_oof:.4f} ({best_improvement:+.4f})")

# 提出ファイル生成
if best_feature == '+IsAlone' and oof_isalone > baseline_oof:
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred_isalone
    })
    submission.to_csv('submission_isalone_added.csv', index=False)
    print(f"\n✅ 提出ファイル作成: submission_isalone_added.csv")
elif best_feature == '+FarePerPerson' and oof_fpp > baseline_oof:
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred_fpp
    })
    submission.to_csv('submission_fareperperson_added.csv', index=False)
    print(f"\n✅ 提出ファイル作成: submission_fareperperson_added.csv")
elif best_feature == '+AgeGroup' and oof_agegroup > baseline_oof:
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_pred_agegroup
    })
    submission.to_csv('submission_agegroup_added.csv', index=False)
    print(f"\n✅ 提出ファイル作成: submission_agegroup_added.csv")
else:
    print(f"\n⚠️ ベースラインを超える特徴量なし → 提出ファイル未作成")

print("\n推奨アクション:")
if best_improvement > 0:
    print(f"✅ {best_feature}をKaggle LBで検証")
else:
    print(f"❌ いずれの特徴量もOOF改善なし")
    print(f"   → 別のアプローチ（CV戦略改善等）を検討")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)

# 結果をファイルに保存
summary_file = 'results/simple_features_evaluation.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("シンプルな新特徴量の評価結果\n")
    f.write("=" * 80 + "\n\n")
    f.write(results.to_string(index=False))
    f.write(f"\n\n最良の特徴量: {best_feature}\n")
    f.write(f"OOF Score: {best_oof:.4f} ({best_improvement:+.4f})\n")

print(f"\n詳細結果を保存: {summary_file}")
