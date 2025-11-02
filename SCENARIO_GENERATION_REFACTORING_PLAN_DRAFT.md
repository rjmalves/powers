# Scenario Generation Refactoring Plan

**Date**: 2025-11-02  
**Status**: Architectural Redesign Proposal  
**Objective**: Unify uncertainty handling across all entity types and temporal models

---

## Executive Summary

This document proposes a comprehensive refactoring of the scenario generation pipeline to:

1. **Eliminate the Independent/PAR dichotomy** - Use a single unified temporal model representation
2. **Unify Load and Inflow handling** - All uncertainties follow the same code path
3. **Fix marginal distribution semantics** - Use proper inverse CDF transformation for all marginals
4. **Simplify the architecture** - Reduce complexity, improve maintainability, enable future extensions

**Core Insight**: The current distinction between "Independent" and "PAR" models is artificial. Independent models are just PAR(0) models (zero autoregressive order). By unifying them, we get a single, cleaner architecture.

---

## Current Problems

### Problem 1: Artificial Independent/PAR Dichotomy

**Current Design**:
```rust
enum UncertaintyModel {
    Independent { seasonal_params: Vec<SeasonalParams> },
    PeriodicAR { par_params: PARParams },
}
```

**Issues**:
- Two completely different code paths for scenario generation
- Independent models are unclear about innovation vs observation
- Load vs Inflow handled differently even with same temporal model
- Different LP constraint generation logic
- Cannot easily add AR dynamics to loads

**Reality**: Independent is just PAR with `ar_order = 0` for all seasons.

### Problem 2: Marginal Distribution Confusion

**Current Approach**: Direct transformation in scenario generator
```rust
// scenario_generator.rs:216
let innovation = params.distribution.transform(base_noise, 0.0, 1.0);

// uncertainty_model.rs:90-96
pub fn transform(&self, z: f64, mean: f64, std_dev: f64) -> f64 {
    match self {
        Self::Normal => mean + std_dev * z,
        Self::LogNormal3 { gamma, mu, sigma } => {
            gamma + (mu + sigma * z).exp()  // ← NOT inverse CDF!
        }
    }
}
```

**Issues**:
- LogNormal3 transform returns observation-space values, not innovations
- Not using proper inverse CDF (quantile function) approach
- The `mean` and `std_dev` parameters are ignored for LogNormal3, passed as 0.0, 1.0
- Semantics are inconsistent between Normal and LogNormal3
- Cannot easily add new distributions (e.g., Beta, Gamma, Truncated Normal)

**Correct Approach**: 
1. Start with correlated standard normal Z ~ N(0,1)
2. Transform to uniform via CDF: U = Φ(Z) where Φ is standard normal CDF
3. Apply inverse CDF of target distribution: X = F⁻¹(U) where F is target CDF

This is the **probability integral transform** and works for ANY distribution.

### Problem 3: Load vs Inflow Divergence

**Current**:
- Loads: Store observations in SAA, use directly in LP RHS
- Inflows (PAR): Store innovations in SAA, compute observations in LP constraints
- Inflows (Independent): Store "innovations" (actually observations for LogNormal3) in SAA

**Issues**:
- Inconsistent handling creates complexity
- Cannot add AR dynamics to loads without major refactoring
- Different debugging/validation paths
- Code duplication

---

## Proposed Architecture

### Core Principle: Everything is PAR

**Unified Model**:
```rust
pub struct UnifiedTemporalModel {
    entity_type: UncertaintyType,  // Load or Inflow
    entity_id: usize,
    num_seasons: usize,
    
    // Seasonal parameters (all seasons)
    seasonal_means: Vec<f64>,           // μ_s for each season s
    seasonal_stds: Vec<f64>,            // σ_s for each season s
    seasonal_distributions: Vec<MarginalDistribution>,  // F_s for each season s
    
    // AR structure (all seasons)
    ar_orders: Vec<usize>,              // p_s for each season s (can be 0!)
    ar_coefficients: Vec<Vec<f64>>,     // φ coefficients for each season
    
    // Precomputed for efficiency
    max_ar_order: usize,
    psi_coefficients: Vec<Vec<f64>>,    // ψ = transformed AR coefficients
    deterministic_bases: Vec<f64>,      // Precomputed μ_s - Σ(φ_i·μ_{s-i})
}
```

**Key Features**:
- **Independent models**: Just set `ar_orders[s] = 0` for all seasons
- **PAR models**: Set `ar_orders[s] = p_s` as needed
- **Same code path** for all models - no branching on Independent vs PAR
- **Loads and Inflows**: Both use the same structure

### Scenario Generation Pipeline

**Unified Pipeline** (4 stages):

```
Stage 1: Sample Base Noise
  Z ~ N(0,1)  [independent samples]
  
Stage 2: Apply Correlation
  W = L·Z where R = L·Lᵀ  [correlated normal]
  
Stage 3: Marginal Transformation (PROPER INVERSE CDF)
  U = Φ(W)  [uniform via CDF]
  ε = F⁻¹(U)  [target distribution via inverse CDF]
  
Stage 4: Store Innovations
  SAA stores: ε_t  [innovations only, for ALL entities]
```

**Key Change**: Stage 3 uses **proper inverse CDF transformation**:

```rust
// Proper probability integral transform
pub fn to_marginal_innovation(z: f64, distribution: &MarginalDistribution) -> f64 {
    use statrs::distribution::{Normal, ContinuousCDF};
    
    // Step 1: Z ~ N(0,1) → U ~ Uniform(0,1) via CDF
    let standard_normal = Normal::new(0.0, 1.0).unwrap();
    let u = standard_normal.cdf(z);
    
    // Step 2: U → ε via inverse CDF of target distribution
    match distribution {
        MarginalDistribution::Normal { mean, std_dev } => {
            let target_normal = Normal::new(*mean, *std_dev).unwrap();
            target_normal.inverse_cdf(u)
        }
        MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
            use statrs::distribution::LogNormal;
            let log_normal = LogNormal::new(*mu, *sigma).unwrap();
            gamma + log_normal.inverse_cdf(u)
        }
        // Easy to add new distributions!
        MarginalDistribution::TruncatedNormal { mean, std_dev, lower, upper } => {
            // Use statrs or custom implementation
            truncated_normal_inverse_cdf(u, *mean, *std_dev, *lower, *upper)
        }
    }
}
```

**Benefits**:
1. **Mathematically correct**: Preserves correlation structure through copula
2. **Distribution-agnostic**: Works for ANY distribution with inverse CDF
3. **Clear semantics**: ε is always an innovation in the target distribution space
4. **Extensible**: Adding new distributions is trivial

### LP Constraint Generation

**Unified for All Entities**:

```rust
// For ALL entities (loads and inflows, independent or PAR):
//
// Innovation constraint:
//   η_t[i] = ε_t[i]  [from SAA]
//
// Observation constraint:
//   Y_t[i] = μ_s + σ_s·η_t[i] + Σ_{k=1}^{p_s} ψ_k · Y_{t-k}[i]
//
// Where:
//   - μ_s, σ_s: seasonal mean/std
//   - ψ_k: transformed AR coefficients (precomputed)
//   - p_s: AR order for season s (0 for independent models)
//   - η_t[i]: innovation variable (from SAA ε_t[i])
//   - Y_t[i]: observation variable (inflow or load)
```

**For Independent Models** (ar_order = 0):
```
Y_t[i] = μ_s + σ_s·η_t[i]
```

**For PAR Models** (ar_order > 0):
```
Y_t[i] = deterministic_base + σ_s·η_t[i] + Σ_{k=1}^{p_s} ψ_k · Y_{t-k}[i]
```

**Key Insight**: Same constraint structure, just `Σ` term is empty for independent models!

---

## Implementation Plan

### Phase 1: Add Proper Inverse CDF Transformation

**Goal**: Fix marginal transformation to use probability integral transform

**Changes**:

1. **Add inverse CDF function** (`src/uncertainty_model.rs`):
```rust
impl MarginalDistribution {
    /// Transform standard normal to target distribution via inverse CDF
    pub fn inverse_cdf(&self, z: f64) -> f64 {
        use statrs::distribution::{Normal, LogNormal, ContinuousCDF};
        
        // Z ~ N(0,1) → U ~ Uniform(0,1)
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let u = standard_normal.cdf(z);
        
        // U → target distribution
        match self {
            Self::Normal { mean, std_dev } => {
                let target = Normal::new(*mean, *std_dev).unwrap();
                target.inverse_cdf(u)
            }
            Self::LogNormal3 { gamma, mu, sigma } => {
                let log_normal = LogNormal::new(*mu, *sigma).unwrap();
                gamma + log_normal.inverse_cdf(u)
            }
        }
    }
}
```

2. **Update scenario generator** (`src/scenario_generator.rs`):
```rust
// Replace:
let innovation = params.distribution.transform(base_noise, 0.0, 1.0);

// With:
let innovation = params.distribution.inverse_cdf(base_noise);
```

3. **Remove old transform() method** - no longer needed

**Testing**:
- Verify LogNormal3 values are correct (no more huge values)
- Check correlation preservation
- Compare with previous results for Normal distribution (should be identical)

**Effort**: 2-3 days  
**Risk**: Low - isolated change, backward compatible for Normal distribution

---

### Phase 2: Unify Independent and PAR Models

**Goal**: Merge `Independent` and `PeriodicAR` into single `UnifiedTemporalModel`

**Changes**:

1. **Create unified model struct** (`src/uncertainty_model.rs`):
```rust
pub struct UnifiedTemporalModel {
    pub entity_type: UncertaintyType,
    pub entity_id: usize,
    pub num_seasons: usize,
    pub seasonal_means: Vec<f64>,
    pub seasonal_stds: Vec<f64>,
    pub seasonal_distributions: Vec<MarginalDistribution>,
    pub ar_orders: Vec<usize>,
    pub ar_coefficients: Vec<Vec<f64>>,
    pub max_ar_order: usize,
    pub psi_coefficients: Vec<Vec<f64>>,      // Precomputed
    pub deterministic_bases: Vec<f64>,         // Precomputed
}
```

2. **Add constructor from old models**:
```rust
impl UnifiedTemporalModel {
    pub fn from_independent(
        entity_type: UncertaintyType,
        entity_id: usize,
        seasonal_params: Vec<SeasonalParams>,
    ) -> Self {
        let num_seasons = seasonal_params.len();
        
        // Extract seasonal data
        let seasonal_means = seasonal_params.iter().map(|p| p.mean).collect();
        let seasonal_stds = seasonal_params.iter().map(|p| p.std_dev).collect();
        let seasonal_distributions = seasonal_params.iter()
            .map(|p| p.distribution.to_marginal())
            .collect();
        
        // Independent = PAR with ar_order = 0
        let ar_orders = vec![0; num_seasons];
        let ar_coefficients = vec![vec![]; num_seasons];
        let psi_coefficients = vec![vec![]; num_seasons];
        let deterministic_bases = seasonal_means.clone();  // No AR adjustment
        
        Self {
            entity_type,
            entity_id,
            num_seasons,
            seasonal_means,
            seasonal_stds,
            seasonal_distributions,
            ar_orders,
            ar_coefficients,
            max_ar_order: 0,
            psi_coefficients,
            deterministic_bases,
        }
    }
    
    pub fn from_par(
        entity_type: UncertaintyType,
        entity_id: usize,
        par_params: PARParams,
    ) -> Self {
        // Precompute psi coefficients and deterministic bases
        let psi_coefficients = compute_psi_transform(&par_params);
        let deterministic_bases = compute_deterministic_bases(&par_params);
        
        Self {
            entity_type,
            entity_id,
            num_seasons: par_params.num_seasons,
            seasonal_means: par_params.seasonal_means,
            seasonal_stds: par_params.seasonal_stds,
            seasonal_distributions: par_params.seasonal_distributions,
            ar_orders: par_params.ar_orders,
            ar_coefficients: par_params.ar_coefficients,
            max_ar_order: par_params.max_ar_order,
            psi_coefficients,
            deterministic_bases,
        }
    }
}
```

3. **Update scenario generator** to use unified model:
```rust
// Single code path for all models
for (entity_idx, model) in self.models.iter().enumerate() {
    let base_noise = transformed_noise[entity_idx];
    let params = model.seasonal_params(season_id);
    
    // Transform to marginal innovation via inverse CDF
    let innovation = params.distribution.inverse_cdf(base_noise);
    
    // Store innovation (for ALL entities)
    scenario.innovations.push(innovation);
    
    // No need for observation computation here - done in LP
}
```

4. **Update subproblem constraint generation**:
```rust
// Unified constraint update for all entities
fn update_uncertainty_constraints(&mut self, innovations: &[f64]) {
    for entity_data in &self.entity_data {
        let innovation = innovations[entity_data.entity_idx];
        let stochastic_term = entity_data.seasonal_std * innovation;
        let mut rhs = entity_data.deterministic_base + stochastic_term;
        
        // Add AR lag contribution (if ar_order > 0)
        if entity_data.ar_order > 0 {
            let lag_obs = self.lag_buffer.get_lags(entity_data.entity_idx);
            let lag_contribution = dot_product(
                &entity_data.psi_coefficients,
                lag_obs
            );
            rhs += lag_contribution;
        }
        
        // Update constraint: Y_t = rhs
        model.change_rows_bounds(entity_data.constraint_idx, rhs, rhs);
    }
}
```

**Testing**:
- Convert all examples to use unified model
- Verify identical results for existing test cases
- Test with Independent, PAR, and mixed scenarios

**Effort**: 1-2 weeks  
**Risk**: Medium - core refactoring but well-contained

---

### Phase 3: Update JSON Schema and Examples

**Goal**: Simplify input specification, migrate all examples

**New JSON Schema**:

```json
{
  "uncertainty_specifications": [
    {
      "uncertainty_type": "inflow" | "load",
      "entity_id": 0,
      "temporal_model": {
        "num_seasons": 12,
        "seasonal_means": [70.0, 65.0, ...],
        "seasonal_stds": [20.0, 20.0, ...],
        "ar_orders": [1, 1, ...],           // ← Can be [0, 0, ...] for independent
        "ar_coefficients": [[0.7], [0.7], ...]  // ← Can be [[], [], ...] for independent
      },
      "seasonal_distributions": [
        {
          "season_id": 0,
          "type": "normal",
          "mean": 70.0,      // ← Now for marginal distribution, not temporal
          "std_dev": 20.0
        },
        ...
      ]
    }
  ]
}
```

**Key Changes**:
1. **Remove `"type": "independent"` vs `"type": "periodic_ar"`** - no longer needed
2. **All models have seasonal_means, seasonal_stds** - even independent
3. **All models have ar_orders, ar_coefficients** - just use `[0]` and `[[]]` for independent
4. **Marginal distributions** are separate from temporal parameters

**Migration Strategy**:

1. **Backward compatibility parser**:
```rust
// During transition, support both old and new formats
impl TemporalModelInput {
    pub fn to_unified(&self) -> UnifiedTemporalModelSpec {
        match self {
            Self::Independent => {
                // Old format: no means/stds in temporal model
                // Extract from seasonal_distributions
                UnifiedTemporalModelSpec {
                    num_seasons: /* infer from seasonal_distributions */,
                    ar_orders: vec![0; num_seasons],
                    ar_coefficients: vec![vec![]; num_seasons],
                    // means/stds extracted from seasonal_distributions
                }
            }
            Self::PeriodicAr { num_seasons, ar_orders, ar_coefficients, 
                               seasonal_means, seasonal_stds } => {
                UnifiedTemporalModelSpec {
                    num_seasons: *num_seasons,
                    ar_orders: ar_orders.clone(),
                    ar_coefficients: ar_coefficients.clone(),
                    // Use provided means/stds
                }
            }
        }
    }
}
```

2. **Migrate examples one by one**:
   - Start with simplest (03-multistage)
   - Verify output matches old version
   - Document changes in example README

3. **Deprecation warning** for old format (keep for 1-2 releases)

**Example Migration**:

**Before** (Independent model):
```json
{
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "lognormal3",
      "gamma": 0.0,
      "mu": 3.689,
      "sigma": 0.7
    }
  ]
}
```

**After** (Unified format):
```json
{
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [40.0, 40.0, ...],   // From exp(mu + sigma²/2) + gamma
    "seasonal_stds": [50.0, 50.0, ...],     // From observation space std dev
    "ar_orders": [0, 0, 0, ...],            // Independent = AR(0)
    "ar_coefficients": [[], [], [], ...]    // No coefficients for AR(0)
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "lognormal3",
      "gamma": 0.0,
      "mu": 3.689,
      "sigma": 0.7
    }
  ]
}
```

**Effort**: 1 week  
**Risk**: Low with backward compatibility, medium if breaking change

---

### Phase 4: Unify Load and Inflow Constraints

**Goal**: Both loads and inflows use same uncertainty constraint structure

**Current**:
- Loads: Direct RHS in load balance constraint
- Inflows: Observation-space AR constraint

**Proposed**:
- **Both**: Innovation variable η_t, observation variable Y_t, same constraint form

**Changes**:

1. **Add load observation variables** to LP model
2. **Replace direct load RHS** with:
```
Load balance: Σ generation - Σ Y_load[bus] = 0

Load observation: Y_load[bus,t] = μ + σ·η_load[bus,t] + Σ_k ψ_k·Y_load[bus,t-k]
```

3. **Unified lag buffer** for both loads and inflows

**Benefits**:
- Can add AR dynamics to loads (e.g., load forecasting with autocorrelation)
- Single code path for all uncertainties
- Easier debugging and validation
- Clearer separation: η (innovation) vs Y (observation)

**Effort**: 1-2 weeks  
**Risk**: Medium - changes LP structure, need careful testing

---

## Benefits of Unified Architecture

### 1. Simplified Codebase

**Before**:
- 2 enum variants (Independent, PeriodicAR)
- 2 scenario generation paths
- 2 constraint update paths  
- 2 SAA storage formats (values vs innovations)
- Loads and inflows handled differently

**After**:
- 1 struct (UnifiedTemporalModel)
- 1 scenario generation path
- 1 constraint update path
- 1 SAA storage format (innovations always)
- Loads and inflows handled identically

**Result**: ~40% less code, easier to understand and maintain

### 2. Mathematically Correct

**Current Issues**:
- LogNormal3 transform doesn't use inverse CDF
- Correlation may not be preserved through marginal transformation
- Unclear innovation semantics

**After**:
- Proper probability integral transform (copula approach)
- Correlation preserved in Gaussian copula
- Clear innovation semantics (always in target distribution space)

### 3. Extensibility

**Easy to Add**:
- New distributions (just implement inverse_cdf)
- AR dynamics for loads
- Multivariate distributions
- Time-varying parameters
- Heteroskedastic models

**Example**: Adding Beta distribution
```rust
MarginalDistribution::Beta { alpha, beta } => {
    use statrs::distribution::Beta;
    let beta_dist = Beta::new(*alpha, *beta).unwrap();
    beta_dist.inverse_cdf(u)
}
```

### 4. Performance

- Precomputed ψ coefficients and deterministic bases
- Single branch instead of match on Independent vs PAR
- Unified lag buffer management
- No Vec allocations in hot path

**Expected**: 5-10% faster scenario generation, identical LP solve time

---

## Migration Timeline

### Week 1-2: Phase 1 - Inverse CDF Transformation
- Implement proper inverse CDF
- Update scenario generator
- Test and validate

### Week 3-4: Phase 2 - Unify Models
- Create UnifiedTemporalModel
- Update scenario generator
- Update subproblem constraints
- Test with existing examples

### Week 5: Phase 3 - Update Schema
- Design new JSON schema
- Implement backward compatibility
- Migrate examples
- Update documentation

### Week 6-7: Phase 4 - Unify Load/Inflow (Optional)
- Add load observation variables
- Update LP constraint generation
- Unified lag buffer
- Test and validate

### Week 8: Final Testing and Documentation
- Comprehensive testing across all examples
- Performance benchmarking
- Update all documentation
- Write migration guide

**Total Effort**: 6-8 weeks for complete refactoring

---

## Risks and Mitigation

### Risk 1: Breaking Changes

**Mitigation**:
- Maintain backward compatibility for JSON for 1-2 releases
- Comprehensive test suite
- Side-by-side comparison with old implementation
- Clear migration documentation

### Risk 2: Numerical Differences

**Mitigation**:
- Verify Normal distribution gives identical results
- Compare correlation preservation
- Use same random seeds for comparison
- Extensive validation on real-world examples

### Risk 3: Performance Regression

**Mitigation**:
- Benchmark before and after
- Profile hot paths
- Optimize precomputation
- Keep old implementation available for comparison

### Risk 4: User Confusion

**Mitigation**:
- Clear migration guide
- Example conversions
- Deprecation warnings (not errors)
- Active user support during transition

---

## Conclusion

The proposed refactoring eliminates artificial complexity in the scenario generation pipeline by:

1. **Unifying temporal models**: Independent is just PAR(0)
2. **Fixing marginal transformations**: Use proper inverse CDF
3. **Unifying entity handling**: Loads and inflows follow same code path

**Benefits**:
- Simpler, cleaner codebase (~40% code reduction)
- Mathematically correct marginal transformation
- Easy to extend with new distributions
- Enables AR dynamics for loads
- Better performance through precomputation

**Recommended Approach**: Phased implementation with backward compatibility, allowing gradual migration and thorough testing at each step.

The current Independent/PAR dichotomy was a reasonable initial design, but we now have enough experience to see that unification is the right path forward. This refactoring will pay dividends in maintainability and extensibility for years to come.

---

## Appendix A: Probability Integral Transform

The **probability integral transform** is a fundamental statistical technique for transforming between distributions:

**Theorem**: If X ~ F (arbitrary distribution), then U = F(X) ~ Uniform(0,1)

**Corollary**: If U ~ Uniform(0,1), then X = F⁻¹(U) ~ F

**Application to Scenario Generation**:

1. Start with correlated Gaussian: W ~ N(0, R)
2. Transform to uniform: U = Φ(W) where Φ is standard normal CDF
3. Transform to target: X = F⁻¹(U) where F is target CDF

**Key Property**: Correlation structure is preserved through the Gaussian copula.

**Why This Matters**:
- Works for ANY distribution with CDF
- Mathematically rigorous
- Preserves correlation (approximately, via Gaussian copula)
- Standard approach in finance, insurance, hydrology

**Example** (LogNormal3):

```rust
// Z ~ N(0,1)
let z = 1.5;

// U ~ Uniform(0,1)
let standard_normal = Normal::new(0.0, 1.0).unwrap();
let u = standard_normal.cdf(z);  // u ≈ 0.933

// X ~ LogNormal3(γ=10, μ=2, σ=0.5)
let log_normal = LogNormal::new(2.0, 0.5).unwrap();
let x_shifted = log_normal.inverse_cdf(u);  // x ≈ 10.8
let x = 10.0 + x_shifted;  // x ≈ 20.8

// Correct value, properly preserves correlation!
```

---

## Appendix B: Example JSON Conversions

### Example 1: Independent Inflow (Current)

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "type": "independent"
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "normal",
      "mean": 40.0,
      "std_dev": 10.0
    }
  ]
}
```

### Example 1: Independent Inflow (New)

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "num_seasons": 12,
    "seasonal_means": [40.0, 40.0, 40.0, ...],
    "seasonal_stds": [10.0, 10.0, 10.0, ...],
    "ar_orders": [0, 0, 0, ...],
    "ar_coefficients": [[], [], [], ...]
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "normal",
      "mean": 40.0,
      "std_dev": 10.0
    }
  ]
}
```

**Note**: seasonal_distributions now specify marginal only, temporal model has means/stds for LP constraints.

### Example 2: PAR Inflow (Current - Unchanged)

```json
{
  "uncertainty_type": "inflow",
  "entity_id": 0,
  "temporal_model": {
    "type": "periodic_ar",
    "num_seasons": 12,
    "ar_orders": [1, 1, 1, ...],
    "ar_coefficients": [[0.7], [0.7], [0.7], ...],
    "seasonal_means": [70.0, 65.0, 55.0, ...],
    "seasonal_stds": [20.0, 20.0, 20.0, ...]
  },
  "seasonal_distributions": [
    {
      "season_id": 0,
      "type": "normal",
      "mean": 0.0,
      "std_dev": 1.0
    }
  ]
}
```

**Note**: Already in the right format! Just remove `"type": "periodic_ar"` field.

---

## Appendix C: Implementation Checklist

### Phase 1: Inverse CDF
- [ ] Add `MarginalDistribution::inverse_cdf()` method
- [ ] Update `scenario_generator.rs` to use inverse CDF
- [ ] Remove old `DistributionType::transform()` method
- [ ] Add unit tests for inverse CDF
- [ ] Verify Normal distribution unchanged
- [ ] Verify LogNormal3 values corrected
- [ ] Test correlation preservation

### Phase 2: Unified Model
- [ ] Create `UnifiedTemporalModel` struct
- [ ] Add `from_independent()` constructor
- [ ] Add `from_par()` constructor
- [ ] Update `ScenarioGenerator` to use unified model
- [ ] Update `Subproblem` constraint generation
- [ ] Unified lag buffer management
- [ ] Remove old `UncertaintyModel` enum
- [ ] Update all tests
- [ ] Verify identical results for all examples

### Phase 3: Schema Update
- [ ] Design new JSON schema
- [ ] Implement backward compatibility parser
- [ ] Add deprecation warnings
- [ ] Convert example: 03-multistage
- [ ] Convert example: 05-large-scale-brazilian
- [ ] Convert example: 07-par-model-with-inflow-state
- [ ] Convert all remaining examples
- [ ] Update JSON schema documentation
- [ ] Update user guide

### Phase 4: Unify Load/Inflow
- [ ] Add load observation variables to LP
- [ ] Update load balance constraints
- [ ] Unified constraint update function
- [ ] Unified lag buffer for loads and inflows
- [ ] Test AR dynamics for loads
- [ ] Performance testing
- [ ] Update documentation

### Final
- [ ] Comprehensive regression testing
- [ ] Performance benchmarking
- [ ] Documentation review
- [ ] Migration guide
- [ ] CHANGELOG update
- [ ] Release notes
