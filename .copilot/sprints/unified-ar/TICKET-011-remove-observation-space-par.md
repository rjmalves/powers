# TICKET-011: Remove Observation Space PAR Generator Code

**Sprint:** 3 - Cleanup  
**Phase:** 3 - Clean Up Dependencies  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** Medium  
**Status:** Not Started

## Context

The PAR (Periodic Autoregressive) model generation currently has code paths for both observation and residual space. With the unified model, **only residual space makes sense** for AR coefficients and modeling. This ticket removes observation space generation and estimation code, simplifying the PAR generator to work exclusively in residual space.

This cleanup eliminates confusion about which space to use and ensures consistency across the codebase.

## Acceptance Criteria

- [ ] Given PAR generator, when creating models, then it only works in residual space
- [ ] Given PAR estimation, when fitting models, then it only uses residual space transformations
- [ ] Given PAR coefficients, when stored, then they are clearly documented as residual space
- [ ] Given PAR generator code, when reviewed, then no observation space paths remain
- [ ] Given example inputs, when they use PAR, then they work correctly with residual-only generation
- [ ] Performance: PAR generation should be slightly faster (removed unnecessary code paths)

## Tasks

### Implementation

- [ ] Review `src/par_generator.rs` for observation vs residual space code paths
- [ ] Remove observation space generation methods:
  - Any functions with "observation" or "obs" in name related to PAR
  - Any code paths that transform to/from observation space during generation
- [ ] Keep only residual space methods:
  - Functions that work with normalized, zero-mean data
  - Functions that output φ coefficients for residual dynamics
- [ ] Update PAR estimation to work exclusively with residuals:
  - Remove observation space data preparation
  - Ensure input data is transformed to residuals before estimation
- [ ] Update seasonal parameter handling:
  - Keep μ, σ parameters (needed for observation transformation in LP)
  - Ensure φ coefficients are for residual space dynamics
- [ ] Remove conditional logic based on space choice
- [ ] Update function signatures to remove space parameters/flags

### Testing

- [ ] Unit test: PAR generation with simple synthetic data (residual space)
- [ ] Unit test: PAR estimation produces valid φ coefficients (|φ| < 1 for stationarity)
- [ ] Unit test: Generated PAR model has correct seasonal parameters
- [ ] Unit test: Verify no observation space transformation during generation
- [ ] Integration test: Generate PAR model and use in UnifiedInflowModel
- [ ] Regression test: Example 07 (PAR model) produces same results
- [ ] Numerical test: Verify estimated coefficients match expected values

### Documentation

- [ ] Update module-level docs for `par_generator.rs`:
  - Clarify exclusive use of residual space
  - Explain why observation space is not used
- [ ] Add doc comments explaining residual space representation:
  - Z'\_t = (Y_t - μ_s) / σ_s (transformation)
  - Z'_t = Σφ_k Z'_{t-k} + ε_t (dynamics)
- [ ] Update PAR generation examples to show residual space workflow
- [ ] Remove any documentation mentioning observation space PAR
- [ ] Update CHANGELOG.md with "Changed: PAR generator works exclusively in residual space"

## Technical Notes

### Residual Space PAR Model

**Complete Model:**

```
Observation: Y_t (physical inflow units)
Residual: Z'_t = (Y_t - μ_s) / σ_s
Dynamics: Z'_t = φ₁_s Z'_{t-1} + φ₂_s Z'_{t-2} + ... + ε_t
where: ε_t ~ N(0, σ²_ε,s)
```

**What PAR Generator Produces:**

- Seasonal means: μ_s for each season s
- Seasonal std devs: σ_s for each season s
- AR coefficients: φ₁_s, φ₂_s, ... for each season s
- Innovation variance: σ²_ε,s for each season s

**All coefficients in residual space!**

### Why Observation Space PAR is Wrong

**Problem:** AR dynamics are not stationary in observation space.

```
Y_t = φ₁ Y_{t-1} + ε_t  // WRONG! Non-stationary due to seasonal mean shifts
```

If μ_winter = 100 and μ_summer = 200, the process has non-constant mean, violating stationarity.

**Solution:** Transform to residuals first.

```
Z'_t = (Y_t - μ_s) / σ_s  // Remove seasonal effects
Z'_t = φ₁ Z'_{t-1} + ε_t  // NOW stationary!
```

### Code to Remove

**Pattern to search for:**

```rust
// Remove functions like:
fn estimate_par_observation_space(...) { }
fn transform_to_observation(...) { }
fn generate_observation_space_model(...) { }

// Remove conditionals like:
if space == Space::Observation {
    // ... remove this branch ...
} else {
    // ... keep residual space logic ...
}

// Remove enum like:
enum Space {
    Observation,  // DELETE
    Residual,     // KEEP (or remove enum entirely)
}
```

### Updated PAR Generator Interface

```rust
// SIMPLIFIED: Only residual space
pub struct ParGenerator {
    seasonal_means: Vec<Vec<f64>>,     // μ_s per hydro per season
    seasonal_std_devs: Vec<Vec<f64>>,  // σ_s per hydro per season
    ar_coefficients: Vec<Vec<Vec<f64>>>, // φ_s per hydro per season
}

impl ParGenerator {
    /// Estimate PAR model from historical data
    /// Input data is transformed to residuals internally
    pub fn estimate_from_observations(
        historical_data: &[Vec<f64>],  // [time][hydro]
        season_ids: &[usize],          // [time]
        max_lag: usize,
    ) -> Self {
        // Transform to residuals
        let residuals = Self::transform_to_residuals(historical_data, season_ids);

        // Estimate AR coefficients in residual space
        let ar_coefficients = Self::estimate_ar_coefficients(&residuals, season_ids, max_lag);

        // ...
    }

    /// Generate UnifiedNoiseSpec for use in UnifiedInflowModel
    pub fn to_unified_noise_spec(&self, hydro: usize) -> UnifiedNoiseSpec {
        // Produces spec with φ coefficients in residual space
    }
}
```

### Migration Notes

**If users have PAR models in observation space:**

1. Re-estimate using updated generator (automatic residual transform)
2. Old observation space coefficients are invalid
3. Provide script to re-estimate if needed

**Backward Compatibility:**

- PAR model file format unchanged (coefficients are just different values)
- Example 07 needs regeneration with new PAR estimation

### Edge Cases

- **Legacy PAR models**: If users have pre-generated PAR models in observation space, they need re-estimation
- **Mixed data**: If some data is already in residuals, don't transform again
- **Zero variance seasons**: Handle σ_s = 0 case (log warning, use σ_s = 1)
- **Non-stationary estimates**: Check |φ| < 1, warn if violated

### Files to Update

Expected files based on project structure:

- `src/par_generator.rs` - Main cleanup
- `src/estimation/` - PAR estimation modules
- `tests/test_par_*.rs` - Update PAR tests
- `examples/07-par-model-with-inflow-state/` - Regenerate PAR models

## Dependencies

- **Blocked by**: TICKET-001 (needs UnifiedInflowModel to target)
- **Blocks**: None (independent cleanup)
- **Related**: TICKET-010 (refactors State trait interface, PAR works with both state types)

## References

- `src/par_generator.rs` - PAR generation implementation
- `src/estimation/` - Statistical estimation methods
- UNIFIED_AR_ROADMAP.md - Section 2.2 (Update PAR generator)
- PAR theory references (Box & Jenkins, seasonal ARMA models)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All PAR tests pass with residual space only
- [ ] Example 07 runs successfully (may need regeneration)
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Grep for "observation.*space.*par" returns minimal results
- [ ] Estimated coefficients are stationary (|φ| < 1)
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
