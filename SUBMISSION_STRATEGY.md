# Titanic Submission Strategy

## Current Status
- **Submission #1**: CV=0.837, LB=0.770 (gap: 0.067)
- **Remaining Submissions**: 9
- **Issue**: Overfitting detected (CV/LB gap too large)

## Analysis

### Key Findings
1. **Overfitting Problem**: CV score (0.837) is much higher than LB score (0.770)
2. **Strategy**: Need models with lower CV variance and better generalization
3. **Hypothesis**: Simpler models with CV ~0.78-0.80 may perform better on LB

### Models Generated

| Rank | Model Name | Algorithm | Features | CV Score | Predicted |
|------|------------|-----------|----------|----------|-----------|
| 1 | V3_WithFare_GB | GradientBoost | Pclass, Sex, Title, Fare | 0.8372 | 0.347 |
| 2 | Ensemble_VotingHard | Voting (LR+RF+GB) | Pclass, Sex, Title, Fare | 0.8327 | 0.371 |
| 3 | V6_Original_RF | RandomForest | Pclass, Sex, Age, Fare, Embarked, Title | 0.8283 | 0.378 |
| 4 | V3_WithFare_RF | RandomForest | Pclass, Sex, Title, Fare | 0.8283 | 0.383 |
| 5 | V6_Original_GB | GradientBoost | Pclass, Sex, Age, Fare, Embarked, Title | 0.8249 | 0.328 |
| 6 | Ensemble_VotingSoft | Voting (LR+RF+GB) | Pclass, Sex, Title, Fare | 0.8215 | 0.373 |
| 7 | V5_WithEmbarked_GB | GradientBoost | Pclass, Sex, Title, Age, Embarked | 0.8204 | 0.289 |
| 8 | V5_WithEmbarked_RF | RandomForest | Pclass, Sex, Title, Age, Embarked | 0.8170 | 0.309 |
| 9 | V5_WithEmbarked_LR | LogisticReg | Pclass, Sex, Title, Age, Embarked | 0.7946 | 0.349 |
| 10 | V6_Original_LR | LogisticReg | Pclass, Sex, Age, Fare, Embarked, Title | 0.7901 | 0.352 |

## Recommended Submission Order

### Phase 1: Test Generalization Hypothesis (Submissions #2-4)

**Goal**: Find models with better CV/LB alignment by testing simpler models

**Recommended submissions**:

1. **Submission #2: V5_WithEmbarked_LR** (CV: 0.7946)
   - **Why**: Logistic Regression is simple, less prone to overfitting
   - **CV closest to current LB**: May have better alignment
   - **Features**: Pclass, Sex, Title, Age, Embarked (5 features)
   - **File**: `submission_V5_WithEmbarked_LR.csv`

2. **Submission #3: Ensemble_VotingHard** (CV: 0.8327)
   - **Why**: Ensemble reduces variance, more stable predictions
   - **Voting mechanism**: Combines LR, RF, GB predictions
   - **Features**: Pclass, Sex, Title, Fare (4 features)
   - **File**: `submission_Ensemble_VotingHard.csv`

3. **Submission #4: V6_Original_LR** (CV: 0.7901)
   - **Why**: Simplest model with original feature set
   - **Alternative hypothesis**: If LR performs well, use it as baseline
   - **Features**: Pclass, Sex, Age, Fare, Embarked, Title (6 features)
   - **File**: `submission_V6_Original_LR.csv`

**Expected Outcomes**:
- If LR models (#2, #4) score 0.78-0.79 on LB → Confirms our hypothesis, focus on simpler models
- If Ensemble (#3) scores 0.79+ on LB → Ensembles work better, explore more
- If all score <0.77 → Need different approach, try higher CV models

### Phase 2: Optimize Based on Phase 1 (Submissions #5-7)

**Wait for Phase 1 LB results, then:**

**Scenario A**: If simple models work better (LR scores 0.78+)
- Try more LR variations with different regularization
- Try SVM or other simple classifiers
- Focus on feature selection

**Scenario B**: If ensembles work better (Ensemble scores 0.79+)
- Try different voting weights
- Try stacking models
- Try bagging approaches

**Scenario C**: If higher CV models needed (all Phase 1 <0.77)
- Submit V3_WithFare_GB (CV 0.8372)
- Submit V6_Original_RF (CV 0.8283)
- Optimize hyperparameters

### Phase 3: Final Push (Submissions #8-10)

Based on best performers from Phases 1-2:
- Fine-tune hyperparameters of best model
- Try small variations (feature combinations, preprocessing)
- Reserve #10 for absolute best model

## Feature Sets Overview

### V3_WithFare (4 features)
- Pclass, Sex, Title, Fare
- **Pros**: Minimal features, avoids multicollinearity
- **Cons**: Missing Age info

### V5_WithEmbarked (5 features)
- Pclass, Sex, Title, Age, Embarked
- **Pros**: Balanced feature set, includes Age
- **Cons**: Embarked may be weak signal

### V6_Original (6 features)
- Pclass, Sex, Age, Fare, Embarked, Title
- **Pros**: Most comprehensive
- **Cons**: More features = more overfitting risk

## Model Parameters Used

### Logistic Regression
```python
LogisticRegression(C=0.1, max_iter=1000, random_state=42)
```
- Low C (0.1) = strong regularization → prevents overfitting

### Random Forest
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=4,           # Conservative depth
    min_samples_split=15,  # Prevent overfitting
    min_samples_leaf=5,    # Prevent overfitting
    random_state=42
)
```

### Gradient Boosting
```python
GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,           # Very conservative
    learning_rate=0.05,    # Slow learning
    min_samples_split=15,
    random_state=42
)
```

### Voting Ensemble
```python
VotingClassifier(
    estimators=[('lr', ...), ('rf', ...), ('gb', ...)],
    voting='hard'  # or 'soft'
)
```

## Tracking Template

| Sub# | Model | Features | Algorithm | CV | LB | Gap | Notes |
|------|-------|----------|-----------|----|----|-----|-------|
| 1 | Original | 6 | RF | 0.8373 | 0.7703 | 0.067 | Overfitting |
| 2 | V5_LR | 5 | LR | 0.7946 | ? | ? | Test simple model |
| 3 | Ensemble_Hard | 4 | Voting | 0.8327 | ? | ? | Test ensemble |
| 4 | V6_LR | 6 | LR | 0.7901 | ? | ? | Test original+LR |
| 5-10 | TBD | - | - | - | - | - | Based on 2-4 results |

## Next Immediate Actions

1. ✅ Generate multiple model candidates (DONE)
2. ⏭️ **Submit #2**: `submission_V5_WithEmbarked_LR.csv`
3. ⏭️ **Submit #3**: `submission_Ensemble_VotingHard.csv`
4. ⏭️ **Submit #4**: `submission_V6_Original_LR.csv`
5. ⏸️ Wait for LB scores
6. ⏸️ Analyze results and plan Phase 2
