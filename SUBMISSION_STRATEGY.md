# Titanic Submission Strategy

## Current Status (Updated: 2025-10-21)
- **Submission #1**: CV=0.8373, LB=0.7703 (gap: 0.0670)
- **Submission #2**: OOF=0.8283, LB=0.7799 (gap: 0.0484) ✓ **Current Best**
- **Remaining Submissions**: 8
- **Progress**: Gap reduced by 27.8%, LB improved by +0.0096

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

---

## UPDATE: Submission #2 Results (2025-10-21)

### ✅ Submission #2: submission_V3_WithFare_RF.csv

**Results**:
- **LB Score**: 0.7799
- **OOF Score**: 0.8283
- **Gap**: 0.0484
- **Improvement from #1**: +0.0096 (1.25%)
- **Gap Reduction**: 0.0670 → 0.0484 (-27.8%)

**Model Details**:
- Algorithm: Random Forest (Conservative)
- Features: Pclass, Sex, Title, Fare (4 features)
- Parameters: max_depth=4, min_samples_split=15, min_samples_leaf=5

### 📊 Key Learnings

**What Worked** ✅:
1. **Feature Reduction**: 6 → 4 features reduced overfitting
2. **Conservative Parameters**: Prevented overfitting, improved generalization
3. **Gap Reduction**: 27.8% reduction confirms simpler models work better

**What Still Needs Work** ⚠️:
1. **Remaining Gap**: 0.0484 still indicates some overfitting
2. **Distribution Differences**: Fare has 3.36 mean difference (Train vs Test)
3. **Score Below Expectation**: Expected 0.78-0.81, got 0.7799

### 🎯 Next Generation Strategy

Based on Submission #2 results, new approach focuses on:

1. **Remove Fare** (largest Train/Test distribution gap)
2. **Even Simpler Models** (target OOF ~0.79-0.80 for better alignment)
3. **Focus on Stable Features** (Age, IsAlone instead of Fare)

### 📁 New Models Generated (Next Generation)

| Model | Algorithm | Features | OOF | Expected LB | Description |
|-------|-----------|----------|-----|-------------|-------------|
| **NextGen_Simple_Minimal_5** | LR | Pclass, Sex, Title, Age, IsAlone (5) | 0.7946 | 0.77-0.80 | Closest to current LB ⭐ |
| NextGen_Ensemble_LR_RF | Voting | Pclass, Sex, Title, Age, IsAlone (5) | 0.7980 | 0.78-0.81 | Weighted ensemble |
| NextGen_LR_NoFare_V2 | LR | Pclass, Sex, Title, Age, IsAlone (5) | 0.7991 | 0.78-0.81 | No Fare, stable |
| NextGen_RF_NoFare_V2 | RF | Pclass, Sex, Title, Age, IsAlone (5) | 0.8058 | 0.79-0.82 | No Fare, RF |

### 🥇 Recommendation for Submission #3

**Primary**: `submission_NextGen_Simple_Minimal_5.csv`

**Rationale**:
- OOF 0.7946 is **closest to current LB** (0.7799)
- Expected gap: ~0.015 (vs current 0.048)
- Uses stable features (removed Fare)
- Simple Logistic Regression (highly generalizable)
- Strong regularization (C=0.05)

**Expected Outcome**:
- **Optimistic**: LB 0.79-0.80 (small gap, improvement)
- **Realistic**: LB 0.77-0.79 (small gap, stable)
- **Pessimistic**: LB 0.76-0.77 (need different approach)

**Key Hypothesis**:
- Removing Fare will improve Train/Test alignment
- Lower OOF (0.7946 vs 0.8283) will reduce gap
- Target gap <0.03 (vs current 0.048)

### 📋 Updated Submission Plan (8 Remaining)

#### Phase 1: Test "Remove Fare" Hypothesis (Submissions #3-4)
- **#3**: NextGen_Simple_Minimal_5 (OOF 0.7946) - Test minimal gap approach
- **#4**: NextGen_LR_NoFare_V2 (OOF 0.7991) OR NextGen_Ensemble (OOF 0.7980)

**Decision Point after #3**:
- If gap < 0.03: Continue with No-Fare approach, optimize further
- If gap >= 0.03: Try ensemble or different features

#### Phase 2: Optimization (Submissions #5-7)
Based on Phase 1 results:
- **Scenario A** (gap < 0.03): Fine-tune regularization, try feature variants
- **Scenario B** (gap >= 0.03): Try ensemble, stacking, or different algorithms

#### Phase 3: Final Push (Submissions #8-10)
- Best model variations
- Ensemble of best performers
- Reserve #10 for absolute best

### 🔬 Technical Insights

**Why Remove Fare?**
1. **Train/Test Distribution Gap**: 
   - Train mean: 32.2
   - Test mean: 35.6
   - Difference: 3.36 (largest among all features)
2. **High Variance**: Fare has high std deviation in both sets
3. **Replacement**: Age + IsAlone provide similar information with better stability

**Why Lower OOF Target?**
1. Previous pattern: High OOF (0.82-0.84) → Large gap (0.04-0.07)
2. New strategy: OOF ~0.79-0.80 → Target gap <0.03
3. Better CV/LB alignment = more predictable results

### 📈 Progress Tracking

| Metric | Initial (#1) | Current (#2) | Target (#3) | Goal |
|--------|--------------|--------------|-------------|------|
| LB Score | 0.7703 | 0.7799 | 0.77-0.79 | 0.80+ |
| CV/OOF | 0.8373 | 0.8283 | 0.7946 | ~0.80 |
| Gap | 0.0670 | 0.0484 | <0.03 | <0.02 |
| Gap % | - | -27.8% | -40%+ | -70%+ |

### ✅ Action Items

1. **Immediate**: Submit `submission_NextGen_Simple_Minimal_5.csv`
2. **Record**: Update RESULTS_LOG.md with LB score
3. **Analyze**: Calculate gap, compare to expectation
4. **Decide**: Choose #4 based on #3 results
5. **Iterate**: Continue gap reduction strategy

---

**Last Updated**: 2025-10-21  
**Next Review**: After Submission #3 results
