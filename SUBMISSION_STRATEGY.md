# Titanic Submission Strategy

## Current Status (Updated: 2025-10-21)
- **Submission #1**: CV=0.8373, LB=0.7703 (gap: 0.0670)
- **Submission #2**: OOF=0.8283, LB=0.7799 (gap: 0.0484) ✓ **Best LB Score**
- **Submission #3**: OOF=0.7946, LB=0.77272 (gap: 0.0219) ✓ **Lowest Gap**
- **Remaining Submissions**: 7
- **Progress**: Gap reduced by 67%, LB best at 0.7799

## Analysis

### Key Findings (Updated After Submission #3)
1. ✅ **Fare is Problematic**: Removing Fare reduced gap by 55% (0.048 → 0.022)
2. ✅ **No-Fare Models Stable**: Gap consistently ~0.022 (very predictable)
3. ⚠️ **Trade-off Exists**: Lower gap but lower LB score
4. 🎯 **Sweet Spot Identified**: Need OOF ~0.80-0.82 with gap <0.03
5. 📊 **Pattern Found**: For no-Fare models, LB ≈ OOF - 0.025

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

## Recommended Submission Order (Updated)

### Phase 1: Find the Sweet Spot ✓ IN PROGRESS (Submissions #4-5)

**Goal**: Achieve OOF ~0.80-0.82 with gap <0.03 for LB ~0.78+

**Submission #4 (NEXT)**: 🥇 **submission_NextGen_RF_NoFare_V2.csv**
- **Why**: Higher capacity than #3 (LR), same stable features
- **Algorithm**: Random Forest (conservative parameters)
- **Features**: Pclass, Sex, Title, Age, IsAlone (5 features - no Fare)
- **Expected OOF**: ~0.8058
- **Expected Gap**: ~0.025-0.030
- **Expected LB**: ~0.78-0.79

**Rationale**:
1. Uses proven stable features from #3 (no raw Fare)
2. Higher model capacity than LR → better performance
3. Should maintain low gap while improving LB score
4. Expected to beat both #2 (0.7799) and #3 (0.77272)

**Submission #5**: Based on #4 results
- **If #4 LB ≥ 0.78**: Try parameter variations or enhanced features
- **If #4 LB = 0.77-0.78**: Try GB or ensemble
- **If #4 gap >0.03**: Try pure LR or ensemble

**Expected Outcomes**:
- **Success** (LB ≥ 0.78, gap <0.03): Found sweet spot, optimize further
- **Partial Success** (LB 0.77-0.78, gap <0.03): Try different algorithms
- **Need Adjustment** (gap >0.03): May need to bring back processed Fare

### Phase 2: Optimize Best Approach (Submissions #6-8)

**Goal**: Push LB score toward 0.79+ while maintaining low gap

**Based on #4-5 results:**

**Scenario A**: If no-Fare RF works (#4 LB ≥ 0.78)
- Try different RF depths (max_depth=5, 6)
- Test small feature additions (IsChild, SmallFamily, Pclass_Sex)
- Try GB with same features
- Target: LB 0.79+

**Scenario B**: If need ensemble approach
- Weighted voting (LR, RF, GB)
- Stacking classifier
- Different ensemble weights
- Target: LB 0.78-0.79

**Scenario C**: If need to bring back Fare
- Use FareBin (quartile binning)
- Try Fare rank transformation
- Test Fare * Pclass interaction
- Balance predictive power and stability

### Phase 3: Final Push (Submissions #9-10)

**Goal**: Reach LB ≥ 0.80

Based on best performers from Phases 1-2:
- Submission #9: Best model + final fine-tuning
- Submission #10: **ABSOLUTE BEST** (reserved for final optimized model)
- Focus on incremental improvements
- Target: LB 0.80+, Gap <0.025

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

| Sub# | Model | Features | Algorithm | OOF | LB | Gap | Notes |
|------|-------|----------|-----------|-----|----|----|-------|
| 1 | Original | 6 | RF | 0.8373 | 0.7703 | 0.0670 | High overfitting |
| 2 | RF_Conservative | 4 | RF | 0.8283 | 0.7799 | 0.0484 | **Best LB** |
| 3 | NextGen_Simple (No Fare) | 5 | LR | 0.7946 | 0.77272 | 0.0219 | **Lowest gap** |
| 4 | NextGen_RF (No Fare) | 5 | RF | ~0.806 | ? | ? | **NEXT** - Expected LB ~0.78 |
| 5 | TBD | - | - | - | ? | ? | Based on #4 |
| 6-10 | TBD | - | - | - | - | - | Optimization phase |

## Next Immediate Actions

1. ✅ Submissions #1-3 completed
2. ✅ Analyzed Submission #3 results
3. ✅ Updated all documentation (RESULTS_LOG.md, SUBMISSION_STRATEGY.md)
4. ⏭️ **Submit #4**: `submission_NextGen_RF_NoFare_V2.csv` ← **RECOMMENDED NEXT STEP**
5. ⏸️ Wait for #4 LB score
6. ⏸️ Based on #4 results, plan submissions #5-7
7. ⏸️ Reserve submissions #9-10 for final optimization

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
**Next Review**: After Submission #4 results

---

## Summary and Key Takeaways

### Breakthrough Discovery
**Fare is the main source of overfitting**:
- With Fare (Sub #2): Gap = 0.048
- Without Fare (Sub #3): Gap = 0.022
- **Impact**: 55% gap reduction

### The Formula for Success
For no-Fare models: **LB ≈ OOF - 0.025**

This means:
- Target OOF 0.805 → Expected LB 0.78
- Target OOF 0.820 → Expected LB 0.795
- Target OOF 0.825 → Expected LB 0.80

### Next Critical Test
**Submission #4** will test if we can achieve high OOF (~0.806) while maintaining low gap (<0.03)

**Success = Finding the sweet spot between predictive power and generalization**
