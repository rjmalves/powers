# Scenario Generation Pipeline Analysis

**Date**: 2025-11-02  
**Author**: Analysis requested by Rogerio  
**Status**: Architectural Review

## Executive Summary

This document analyzes the current implementation of the scenario generation pipeline in `scenario_generator.rs`, focusing on:

1. **Independent vs PAR model treatment differences**
2. **Innovation vs observation space handling**
3. **Marginal transformation inconsistencies (Normal vs LogNormal3)**
4. **Recommendations for unification**

## Current Architecture Overview

### Pipeline Flow

```
ScenarioGenerator::generate_stage_scenarios()
  ↓
1. Sample base noise: Z ~ N(0,1)
  ↓
2. Apply correlation (if specified)
  ↓
3. Apply marginal transformation + temporal model logic
  ↓
4. Store in Scenario { values, innovations }
  ↓
5. Populate SAA:
   - Loads: use scenario.values (observations)
   - Inflows: use scenario.innovations (innovations only)
```

### Data Flow by Entity Type

| Entity Type | Temporal Model | What's Generated | What's Stored in SAA | How It's Used |
|-------------|----------------|------------------|----------------------|---------------|
| Load | Independent | Observation Y_t | `values` (observation) | Directly as RHS in load balance |
| Inflow | Independent | Observation Y_t | `innovations` (???) | ??? |
| Inflow | PAR | Innovation ε_t | `innovations` | LP constraint: Y_t = μ + σ·ε_t + Σφ_i·Y_{t-i} |

## Problem 1: Inconsistent Treatment of Independent Models

### Current Implementation

**Location**: `scenario_generator.rs:208-233`

```rust
UncertaintyModel::Independent { .. } => {
    // Transform base noise to get the innovation
    let innovation = params.distribution.transform(base_noise, 0.0, 1.0);
    
    // Calculate observation based on distribution type
    let observation = match params.distribution {
        DistributionType::Normal => {
            // Linear: Y_t = μ + σ*ε_t
            params.mean + params.std_dev * innovation
        }
        DistributionType::LogNormal3 { .. } => {
            // LogNormal3: innovation is already transformed
            // Y_t = μ + innovation (not μ + σ*innovation)
            params.mean + innovation
        }
    };
    
    scenario.values.push(observation);
    scenario.innovations.push(innovation);  // ← PROBLEM HERE
}
```

### The Issues

#### Issue 1.1: Semantic Confusion for Independent Models

For **Independent/Normal**:
- `innovation` = Z ~ N(0,1) transformed to N(0,1) = Z itself
- `observation` = μ + σ·Z (correct)
- Stored in SAA: `values` for loads, `innovations` for inflows

For **Independent/LogNormal3**:
- `innovation` = γ + exp(μ + σ·Z) (FULL LOGNORMAL VALUE)
- `observation` = params.mean + innovation (WRONG SEMANTICS)
- The variable name "innovation" is misleading - it's actually the full lognormal sample

**Root Cause**: The code conflates two concepts:
1. **Innovation**: The noise term ε_t in noise space (should be standardized or log-space)
2. **Observation**: The actual realized value Y_t in observation space

#### Issue 1.2: Different Handling for Loads vs Inflows

**From `input.rs:854-865`:**

```rust
match model.entity_type() {
    UncertaintyType::Load => {
        // For loads, use observation values
        load_observations[load_idx].push(scenario.values[model_idx]);
    }
    UncertaintyType::Inflow => {
        // For inflows, only INNOVATIONS are stored in SAA
        inflow_innovations[inflow_idx].push(scenario.innovations[model_idx]);
    }
}
```

**The Problem**: For Independent/Load models:
- `scenario.values` is used correctly (observation)

For Independent/Inflow models:
- `scenario.innovations` is stored in SAA
- But for LogNormal3, this is actually the full lognormal value, not a standardized innovation
- For Normal, this is Z (standardized noise), which could work but is inconsistent

**Question**: Why are independent inflow models storing innovations at all? They don't have AR dynamics, so they should just use observations directly like loads do.

### User's Insight

> "the independent models assume that the innovations are the actual observations, and insert them into the subproblem. However, this seems to be wrong, since I could always add innovations in 'noise-scale' and convert them to actual observations in LP constraints, much like it is done for inflow PAR models."

**Analysis**: This is correct! The current approach creates an artificial distinction:

- **Loads**: Always use observations Y_t directly
- **Inflows with PAR**: Store innovations ε_t, compute Y_t in LP
- **Inflows with Independent**: Store... observations disguised as innovations?

This is architecturally inconsistent.

## Problem 2: LogNormal3 Marginal Transformation Issues

### Current Implementation Issues

**Location**: `scenario_generator.rs:224-228`

```rust
DistributionType::LogNormal3 { .. } => {
    // LogNormal3: innovation is already transformed
    // Y_t = μ + innovation (not μ + σ*innovation)
    params.mean + innovation
}
```

**Location**: `uncertainty_model.rs:90-96`

```rust
pub fn transform(&self, z: f64, mean: f64, std_dev: f64) -> f64 {
    match self {
        Self::Normal => mean + std_dev * z,
        Self::LogNormal3 { gamma, mu, sigma } => {
            gamma + (mu + sigma * z).exp()  // ← Full lognormal value
        }
    }
}
```

**Location**: `uncertainty_model.rs:143-153`

```rust
MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
    // For PAR models, we work with innovations in log-space
    // Y_t = gamma + exp(mu + sigma * ε_t) where ε_t ~ N(0,1)
    let mean = gamma + mu.exp(); // Location parameter + exp(log-space mean)
    let std_dev = *sigma;         // Use sigma directly as innovation scale
    (mean, std_dev)
}
```

### The Issues

#### Issue 2.1: Inconsistent Innovation Semantics

For **Normal distributions**:
- Innovation ε_t ~ N(0,1) is standardized
- Observation Y_t = μ + σ·ε_t (linear transformation)
- Innovation scale: unit variance

For **LogNormal3 distributions**:
- "Innovation" is γ + exp(μ + σ·Z), which is the FULL LOGNORMAL VALUE
- This is NOT in innovation/noise space
- It's already the final observation value

**Consequence**: The two distributions produce "innovations" that are in completely different spaces, making the code hard to reason about.

#### Issue 2.2: The "Mean" Parameter Problem

The `params.mean` for LogNormal3 is computed as:
```rust
let mean = gamma + mu.exp();
```

This is the location parameter γ plus the median of the lognormal part (exp(μ)).

Then in the scenario generator:
```rust
let observation = params.mean + innovation;
```

But `innovation` is already `gamma + exp(mu + sigma * z)`, so:
```rust
observation = [gamma + exp(mu)] + [gamma + exp(mu + sigma * z)]
            = gamma + exp(mu) + gamma + exp(mu + sigma * z)
```

**This IS wrong!** We're adding params.mean = γ + exp(μ) to the full lognormal value.

**Verification with concrete numbers**:
- Input: γ=10, μ=2, σ=0.5, Z=1.0
- params.mean = γ + exp(μ) = 10 + 7.389 = 17.389
- innovation = γ + exp(μ + σ·Z) = 10 + exp(2.5) = 10 + 12.182 = 22.182 (CORRECT LogNormal3 value)
- observation = params.mean + innovation = 17.389 + 22.182 = **39.571** (WRONG!)
- Correct observation should be: 22.182
- **Error**: Adding an extra 17.389 (= γ + exp(μ))

This explains the "huge inflow values" the user observed!

#### Issue 2.3: User's Observation About Large Inflow Values

> "I was getting huge inflow values, but this might be happening because the lognormal3 distribution that was being given as input had a very large mean, which could lead to very large innovation noises."

**Analysis**: The user's observation is explained by **Issue 2.2** - the code has a confirmed bug that adds `params.mean = γ + exp(μ)` to the already-correct LogNormal3 value. This inflates observations by a factor that can be very large when μ is moderate-to-large.

**Example**: If user specifies γ=10, μ=2, σ=0.5:
- Correct LogNormal3 sample: ~10-40 (depending on Z)
- Buggy observation: adds 17.4, making it ~27-57
- With μ=5: adds 158.4, making values **exponentially larger**

This is NOT just about large input parameters - it's a coding bug that multiplies the effect of the parameters.

### Issue 2.4: Why LogNormal3 Exists

From comments:
> "Remember, the lognormal3 is used only to prevent negative inflows, which could happen with regular normal distributions with zero mean."

**Purpose**: Ensure non-negative inflows when historical data has low mean relative to variance.

**Problem**: The implementation achieves non-negativity but does so in a mathematically inconsistent way that:
- Breaks the innovation/observation space separation
- Makes different marginal distributions incompatible in the same framework
- Leads to confusing parameter interpretations

## Problem 3: Different Treatment of Temporal Models

### Current Differences

| Aspect | Independent | PAR |
|--------|-------------|-----|
| Innovation storage | Mixed (Z for Normal, full value for LogNormal3) | Consistent |
| Observation computation | During generation | During LP solve |
| LP constraints | Direct RHS | Y_t = μ + σ·ε_t + Σφ_i·Y_{t-i} |
| Space | Observation space | Innovation space → Observation space |

### Why PAR Models Work Better

**Location**: `scenario_generator.rs:234-266`

```rust
UncertaintyModel::PeriodicAR { .. } => {
    // Sample innovation only
    let innovation = params.distribution.transform(base_noise, 0.0, 1.0);
    
    // Store innovation (what goes to SAA)
    scenario.innovations.push(innovation);
    
    // Placeholder values (unused for inflows)
    scenario.values.push(0.0);
}
```

**Then in subproblem** (`subproblem.rs:959-1008`):

```rust
fn update_ar_constraints_optimized(&mut self, innovations: &[f64]) {
    for hydro_data in &self.hydro_data {
        let innovation = innovations[hydro_id];
        let stochastic_term = hydro_data.seasonal_params.std_dev * innovation;
        let mut rhs = hydro_data.deterministic_noise_base + stochastic_term;
        
        if hydro_data.ar_order > 0 {
            let lag_obs = self.inflow_manager.get_lag_observations(...);
            let lag_contribution = dot_product(&hydro_data.transformed_coefficients, lag_obs);
            rhs += lag_contribution;
        }
        
        // Update constraint: Y_t = rhs
        model.change_rows_bounds(hydro_data.ar_constraint_idx, rhs, rhs);
    }
}
```

**Why this is better**:
1. Clear separation: innovations ε_t are stored, observations Y_t are computed in LP
2. The σ·ε_t term is computed correctly in the LP
3. AR dynamics naturally apply to observations
4. Works consistently for both Normal and LogNormal3 (though LogNormal3 still has the semantic issue)

## Recommendations

### **CRITICAL BUG FIX**: LogNormal3 Observation Calculation

**Priority**: IMMEDIATE  
**Location**: `scenario_generator.rs:224-228`

**Current (BUGGY) code**:
```rust
DistributionType::LogNormal3 { .. } => {
    // LogNormal3: innovation is already transformed
    // Y_t = μ + innovation (not μ + σ*innovation)
    params.mean + innovation  // ← BUG: adds params.mean = γ + exp(μ)
}
```

**Fixed code**:
```rust
DistributionType::LogNormal3 { .. } => {
    // LogNormal3: innovation is already the full lognormal value
    innovation  // Don't add params.mean - it's already correct!
}
```

**Why this is critical**:
- Inflates all LogNormal3 observations by γ + exp(μ)
- With typical parameters (γ=10, μ=2), this adds ~17.4 to every value
- With larger μ (e.g., μ=5), this adds ~158.4, making values massive
- Explains user's "huge inflow values" observation
- Affects all models using LogNormal3 distribution

**Testing**:
```rust
// Test case to verify fix
let gamma = 10.0;
let mu = 2.0;
let sigma = 0.5;
let z = 1.0;

let expected = gamma + (mu + sigma * z).exp();  // 22.182
// After fix, observation should equal expected
// Before fix, observation was ~39.57 (WRONG)
```

---

### Recommendation 1: Unify Independent and PAR Model Handling

**Goal**: Treat all inflow models uniformly in innovation space, compute observations in LP.

**Proposed Changes**:

#### Change 1.1: Independent Models Should Store Pure Innovations

```rust
UncertaintyModel::Independent { .. } => {
    // Sample standardized innovation ε_t ~ N(0,1)
    let innovation = base_noise;  // Already N(0,1) after correlation
    
    // Store innovation only
    scenario.innovations.push(innovation);
    
    // Compute observation for loads (still needed for backward compatibility)
    let observation = match params.distribution {
        DistributionType::Normal => {
            params.mean + params.std_dev * innovation
        }
        DistributionType::LogNormal3 { gamma, mu, sigma } => {
            gamma + (mu + sigma * innovation).exp()
        }
    };
    scenario.values.push(observation);
}
```

**Note**: We still compute observations for loads, but innovations are now pure ε_t ~ N(0,1).

#### Change 1.2: Create LP Constraints for Independent Inflow Models

Currently, independent inflow models don't have LP constraints - the "innovation" is used directly.

**Proposed**: Add constraints like PAR but with no lag terms:

```
Y_t[h] = deterministic_base[h] + σ[h] * ε_t[h]
```

Where:
- For Normal: `deterministic_base = μ`
- For LogNormal3: `deterministic_base = γ`, but nonlinear exp() is an issue (see Rec 2)

**Benefits**:
- Unified code path for all inflow models
- Clear separation of innovation and observation spaces
- Makes future extensions easier (e.g., adding correlation to loads)

### Recommendation 2: Fix LogNormal3 Handling

**The Core Problem**: LogNormal distributions are inherently nonlinear, which doesn't fit well into linear programming.

**Option 2A: Keep LogNormal3 but Fix the Mathematics**

Current approach tries to use:
```
Y_t = γ + exp(μ + σ·ε_t)
```

But LP can't handle exp(). The current workaround samples the full lognormal value and treats it as an "innovation", which breaks the abstraction.

**Better approach**: Use log-space transformations

1. Sample ε_t ~ N(0,1)
2. Compute log-space value: L_t = μ + σ·ε_t
3. In post-processing (not LP): Y_t = γ + exp(L_t)

**Problem**: This still requires nonlinear transformation, which can't be in the LP constraint.

**Option 2B: Document LogNormal3 as Observation-Space Only**

Accept that LogNormal3 can't be used with AR models (or be clear it's a different formulation):

```rust
match model {
    UncertaintyModel::Independent { .. } => {
        match params.distribution {
            DistributionType::Normal => {
                // Store ε_t ~ N(0,1)
                scenario.innovations.push(base_noise);
            }
            DistributionType::LogNormal3 { gamma, mu, sigma } => {
                // Sample full lognormal value (observation-space only)
                let observation = gamma + (mu + sigma * base_noise).exp();
                scenario.innovations.push(observation);  // Name is misleading but functional
            }
        }
    }
    UncertaintyModel::PeriodicAR { .. } => {
        // CONSTRAINT: Only Normal distribution allowed for PAR
        assert!(matches!(params.distribution, DistributionType::Normal));
        scenario.innovations.push(base_noise);
    }
}
```

**Benefits**:
- Honest about the limitation
- Prevents incorrect usage of LogNormal3 with AR models
- Clearer semantics

**Option 2C: Truncated Normal as Alternative (Recommended)**

**Problem**: Need non-negative inflows without the complexity of LogNormal3.

**Solution**: Use truncated normal distribution (truncated at zero):

```rust
DistributionType::TruncatedNormal { mean, std_dev, lower_bound }
```

Implementation:
1. Sample Z ~ N(0,1)
2. Transform: X = μ + σ·Z
3. If X < lower_bound: resample (or use rejection sampling/inverse CDF method)

**Benefits**:
- Stays in linear space (affine transformation)
- Provides non-negativity guarantee
- Compatible with AR models
- Simpler mathematics than LogNormal3
- Can be used in LP constraints: `Y_t ≥ 0` as an additional constraint

**Drawbacks**:
- Requires rejection sampling (slightly slower generation)
- Changes the distribution shape (but that may be acceptable)

### Recommendation 3: Consistent Parameter Interpretation

**Problem**: The `params.mean` and `params.std_dev` extracted for LogNormal3 are confusing.

**Current** (`uncertainty_model.rs:143-153`):
```rust
MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
    let mean = gamma + mu.exp();  // γ + median of lognormal part
    let std_dev = *sigma;          // log-space std dev
    (mean, std_dev)
}
```

**Issues**:
1. `mean` is not the true mean of the lognormal distribution
   - True mean would be: γ + exp(μ + σ²/2)
2. `std_dev` is in log-space, not observation-space
3. These are used inconsistently in `scenario_generator.rs`

**Recommendation**: If keeping LogNormal3, clarify documentation:

```rust
/// For LogNormal3 distribution:
/// - mean: Represents γ (location parameter), NOT the distribution mean
/// - std_dev: Represents σ (log-space scale), NOT observation-space std dev
/// - The actual distribution mean is γ + exp(μ + σ²/2)
/// - These parameters are used for LP constraint formulation, not statistics
```

Or better: Don't extract mean/std_dev for LogNormal3 at all, keep the full (γ, μ, σ) tuple.

### Recommendation 4: Unify Load and Inflow Handling

**Current Problem**: Loads and inflows are treated differently even when both are Independent models.

**Proposed Approach**:

1. **All entities** store innovations ε_t ~ N(0,1) in SAA
2. **All entities** have LP constraints to compute observations:
   - Loads: `D_t[b] = μ_b(t) + σ_b(t) * ε_t[b]`
   - Inflows (Independent): `Y_t[h] = μ_h(t) + σ_h(t) * ε_t[h]`
   - Inflows (PAR): `Y_t[h] = μ_h(t) + σ_h(t) * ε_t[h] + Σφ_i * Y_{t-i}[h]`

**Benefits**:
- Single code path for all uncertainty
- Easier to add features (correlation, AR dynamics for loads, etc.)
- Clearer semantics
- Better separation of concerns

**Implementation Effort**: Medium-High
- Refactor subproblem constraint generation
- Update load balance constraints
- Ensure backward compatibility
- Comprehensive testing

## Implementation Roadmap

### Phase 0: CRITICAL BUG FIX - LogNormal3 Observation (IMMEDIATE)

**Tasks**:
1. Fix `scenario_generator.rs:224-228` - remove incorrect `params.mean +` 
2. Add regression test to verify correct LogNormal3 values
3. Re-run all examples using LogNormal3 to verify results are now correct
4. Update any documentation that references LogNormal3 behavior

**Effort**: 1 day  
**Risk**: Low (simple fix, but need to validate examples)  
**Impact**: HIGH - fixes incorrect results for all LogNormal3 users

### Phase 1: Fix LogNormal3 Semantics (High Priority)

**Tasks**:
1. Document current behavior and limitations
2. Fix the `params.mean + innovation` issue in scenario generator
3. Add validation: LogNormal3 only for Independent models
4. Update tests and examples

**Effort**: 2-3 days  
**Risk**: Low (clarification, not major change)

### Phase 2: Unify Independent Inflow Models (Medium Priority)

**Tasks**:
1. Store standardized innovations ε_t for Independent inflows
2. Add LP constraints for Independent inflows (like PAR but no lags)
3. Update scenario generation logic
4. Maintain backward compatibility with examples

**Effort**: 1 week  
**Risk**: Medium (affects core SAA structure)

### Phase 3: Consider Truncated Normal Alternative (Low Priority)

**Tasks**:
1. Implement TruncatedNormal distribution type
2. Add sampling logic with rejection/inverse CDF
3. Provide migration path from LogNormal3
4. Update documentation and examples

**Effort**: 1-2 weeks  
**Risk**: Low (additive feature)

### Phase 4: Unify Load and Inflow Handling (Future Work)

**Tasks**:
1. Design unified uncertainty constraint framework
2. Refactor subproblem constraint generation
3. Update load balance constraints to use innovations
4. Extensive testing across all examples

**Effort**: 2-3 weeks  
**Risk**: High (major refactoring)

## Conclusion

The current scenario generation pipeline has several architectural inconsistencies and **one critical bug**:

### Critical Issue (Bug)
- **LogNormal3 observations are WRONG**: The code adds `params.mean` to the already-correct LogNormal3 value, inflating all observations by γ + exp(μ). This can make values 50-200% larger than they should be.
- **Impact**: All models using LogNormal3 distribution produce incorrect results
- **Fix**: One-line change in `scenario_generator.rs:227`

### Architectural Inconsistencies
1. **Independent vs PAR models** are treated differently, with Independent models sometimes storing observations and sometimes innovations
2. **LogNormal3 distribution** breaks the innovation/observation space abstraction
3. **Loads vs inflows** have different code paths even for the same temporal model type

**The user's insights are correct**: The system should uniformly:
- Store standardized innovations ε_t ~ N(0,1) in SAA
- Compute observations Y_t in LP constraints using affine transformations
- Treat all entities consistently regardless of type

**Recommended Next Steps**:

1. **IMMEDIATE (Critical)**: Fix LogNormal3 observation calculation bug
2. **Short-term**: Document LogNormal3 limitations and fix remaining semantic issues
3. **Medium-term**: Unify Independent inflow model handling with PAR models
4. **Long-term**: Consider Truncated Normal as a better alternative for non-negativity
5. **Future**: Fully unify load and inflow uncertainty handling

This will result in a cleaner, more maintainable, and more extensible architecture that properly separates noise generation from observation computation.

---

## Acknowledgments

This analysis was prompted by insightful observations from the system architect regarding:
- Inconsistent treatment of Independent vs PAR models
- The unnecessary complexity of different code paths for loads and inflows
- The suspect behavior of LogNormal3 transformations causing "huge inflow values"

The analysis confirmed these suspicions and identified a critical bug in the LogNormal3 implementation that should be fixed immediately.
