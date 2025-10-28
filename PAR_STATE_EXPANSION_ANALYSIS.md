# PAR State Expansion Analysis & Recommendations

**Date**: October 28, 2025  
**Status**: 🔴 IMPLEMENTATION REQUIRED  
**Priority**: P1 - Affects PAR model cut generation correctness

---

## Executive Summary

After investigating the PAR (Periodic Auto-Regressive) model implementation and consulting the SDDP.jl documentation on the **state expansion trick** ([link](https://sddp.dev/stable/tutorial/arma/)), we've identified that the current implementation does not correctly handle AR constraints for cut generation.

**Key Finding**: The current code passes full observations (Y_t) to AR constraint RHS, but should pass innovations (ε_t) for proper dual variable extraction.

---

## Background: The State Expansion Trick

### What It Is

The state expansion trick is a technique in SDDP to handle autoregressive (AR) processes by:

1. **Adding AR state variables to the Bellman function**
2. **Modeling the AR relationship as LP constraints**
3. **Extracting dual variables from AR constraints for cuts**

### Why It Matters

The dual variables (shadow prices) of AR constraints represent the marginal value of the AR relationship. These duals must be included in Bellman cuts for correct policy approximation.

### SDDP.jl Example

```julia
# inflow is a STATE VARIABLE (part of Bellman state)
@variable(sp, inflow, SDDP.State, initial_value = 50.0)

# ε is the INNOVATION (random variable)
@variable(sp, ε)

# AR constraint: inflow_out = inflow_in + ε
@constraint(sp, inflow.out == inflow.in + ε)

# Parameterize ONLY the innovation, not full inflow!
SDDP.parameterize(sp, Ω) do ω
    return JuMP.fix(ε, ω)  # ω is innovation ε, not full value!
end
```

**Critical Point**: The scenario parameterization sets **ε (innovation)**, NOT the full inflow value.

---

## Current Implementation Analysis

### What Happens Now

1. **PAR Generator** (`src/par_generator.rs`):
   - Takes innovation `a_t` as input
   - Computes AR terms from internal buffer: `Σ(φ_l · Z'_{t-l})`
   - Computes residual: `Z'_t = AR_terms + a_t`
   - Returns observation: `Y_t = μ_m + σ_m · Z'_t`

2. **Scenario Realization** (`src/subproblem.rs::realize_uncertainties`):
   ```rust
   // Gets Y_t (full observation) from PAR generator
   let inflow_noises = first_process.realize(noises.get_inflow_noises());
   
   // Passes Y_t directly to set_uncertainties
   self.set_uncertainties(load, inflow_noises);
   ```

3. **AR Constraint Setup** (`src/state.rs::set_inflows_in_subproblem`):
   ```rust
   // Sets AR constraint RHS to inflow_noises[hydro], which is Y_t
   // Constraint: inflow_noise - Σ(φ_l · lag[l]) = Y_t  ❌ WRONG!
   model.change_rows_bounds(
       inflow_rhs_constraint,
       inflows[hydro],  // This is Y_t, should be ε_t!
       inflows[hydro],
   );
   ```

### The Problem

**The AR constraint RHS is set to Y_t (full observation), but should be ε_t (innovation).**

This means:
- The LP constraint is: `inflow_noise - Σ(φ_l · lag[l]) = Y_t` (incorrect)
- Should be: `inflow_noise - Σ(φ_l · lag[l]) = ε_t` (correct)

### Why This Breaks Cut Generation

The dual variable of the AR constraint represents ∂V/∂(AR_relationship). If the RHS is wrong, the dual is wrong, and cuts will have incorrect coefficients for lagged inflow states.

**Result**: Suboptimal policies, incorrect water values for lagged states.

---

## Mathematical Framework

### PAR Model Equations

```
Observation space:  Y_t = μ_m + σ_m · Z'_t
Residual space:     Z'_t = φ₁·Z'_{t-1} + φ₂·Z'_{t-2} + ... + φₚ·Z'_{t-p} + ε_t
Innovation:         ε_t = Z'_t - Σ(φ_l · Z'_{t-l})
```

### Correct LP Formulation

```
Variables:
  - inflow_noise_t: Current residual Z'_t (decision variable)
  - lag[k]: Past residual Z'_{t-k-1} (state, fixed by past decisions)
  
Constraint (state expansion):
  inflow_noise_t - φ₁·lag[0] - φ₂·lag[1] - ... - φₚ·lag[p-1] = ε_t
  
Where ε_t is the innovation from the scenario.
```

### Cut Generation

```
Bellman function state:
  state_t = [storage_t, Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-p}]

Cut form:
  α ≥ intercept + Σ(λ_storage · storage) + Σ(λ_lag[k] · Z'_{t-k})

Where:
  - λ_storage: Dual of hydro balance constraint
  - λ_lag[k]: Dual of AR constraint (critical!)
```

---

## Why Tests Currently Pass

Several factors mask this issue:

1. **Moderate AR coefficients**: Example 06 uses φ = 0.7, errors are partially dampened
2. **Short horizons**: 5 iterations don't allow errors to compound significantly  
3. **Approximate method**: SDDP produces approximate solutions anyway
4. **Feasibility focus**: Tests check convergence and feasibility, not optimality
5. **No ground truth**: No comparison against analytical optimal solution
6. **Specific scenarios**: Test scenarios might not expose the sensitivity

### Tests That Would Expose the Bug

- **Long horizons**: 50+ stages where errors compound
- **High AR coefficients**: φ > 0.9 for strong temporal coupling
- **Out-of-sample evaluation**: Different seed, measure policy quality
- **Known optimal solution**: Compare against analytical benchmark
- **Sensitivity analysis**: Vary initial lags, check cut coefficient stability

---

## Proposed Solutions

### Option 1: Modify PAR Generator to Return Innovations ⭐ (Recommended)

**Change**: Make PAR generator return both observation and innovation

```rust
pub struct PARGenerationResult {
    pub observation: f64,  // Y_t = μ + σ·Z'_t (for output/reporting)
    pub innovation: f64,   // ε_t (for AR constraint RHS)
}

impl PeriodicARGenerator {
    pub fn generate_next_for_season_detailed(
        &mut self,
        season_id: usize,
        a_t: f64,
    ) -> PARGenerationResult {
        // Compute AR term from lags
        let ar_term = self.compute_ar_term(season_id);
        
        // Innovation is just the input (already transformed)
        let innovation = a_t;
        
        // Full residual includes AR dynamics
        let z_prime = ar_term + a_t;
        
        // Observation space
        let observation = self.params.get_mean(season_id) 
                        + self.params.get_std(season_id) * z_prime;
        
        // Update internal buffer with z_prime
        self.residual_buffer.push_front(z_prime);
        
        PARGenerationResult { observation, innovation }
    }
}
```

**Pros**:
- Clean API, separates concerns
- Mathematically correct
- Minimal changes to calling code

**Cons**:
- Requires updating PAR generator API
- Need to update all callers (NoiseModelCache, etc.)

**Effort**: ~3 hours

---

### Option 2: Extract Innovation from Observation in Subproblem

**Change**: Compute ε_t in `realize_uncertainties()` using state's lagged values

```rust
pub fn realize_uncertainties(
    &mut self,
    noises: &scenario::SampledBranchingNoises,
    state: &dyn State,  // NEW: Need access to lags
    ...
) {
    // Get Y_t from PAR generator
    let observations = first_process.realize(noises.get_inflow_noises());
    
    // Transform to residuals: Z'_t = (Y_t - μ) / σ
    let residuals = transform_to_residuals(observations, self.season_id);
    
    // Extract innovations: ε_t = Z'_t - Σ(φ_l · lag[l])
    let innovations = compute_innovations(
        residuals,
        state.get_lagged_residuals(),  // Need this accessor
        self.ar_coefficients,
    );
    
    // Pass innovations to AR constraint RHS
    self.set_uncertainties(load, &innovations);
}
```

**Pros**:
- No changes to PAR generator
- Localizes fix to subproblem

**Cons**:
- Circular dependency (subproblem needs state, state needs subproblem)
- Requires new State trait methods
- More complex control flow

**Effort**: ~4 hours

---

### Option 3: Refactor Scenario Generation to Produce Innovations ⭐⭐ (Most Correct)

**Change**: Have NoiseModelCache generate innovations directly, enforce AR in LP only

```rust
// NoiseModelCache::generate_stage_scenarios()
// Instead of:
let value = gen.generate_next_for_season(season_id, base_noise);

// Do:
let innovation = marginal_transform(base_noise);  // ε_t
scenario_inflows[hydro] = innovation;  // Store innovation, not observation
```

**Then in Subproblem**:
```rust
// AR constraint in LP enforces: Z'_t = Σ(φ_l · lag[l]) + ε_t
// LP solves for optimal Z'_t given innovation ε_t

// For output/reporting:
let observations = transform_residuals_to_observations(residuals);
```

**Pros**:
- Aligns perfectly with state expansion trick theory
- AR dynamics enforced ONLY in LP (proper SDDP formulation)
- Clean separation: scenarios = innovations, AR = constraints
- Matches SDDP.jl approach exactly

**Cons**:
- Major refactoring of NoiseModelCache and PAR generator
- Changes scenario generation pipeline
- Need to update all scenario-related code
- Requires careful testing

**Effort**: ~8-10 hours

---

### Option 4: Two-Phase Approach (Minimal Change)

**Change**: Split scenario realization into two phases

```rust
// Phase 1: Generate and store observations
let observations = first_process.realize(noises.get_inflow_noises());
realization_container.inflow.clone_from_slice(observations);

// Phase 2: Before LP solve, compute innovations from observations + lags
fn prepare_lp_rhs(&mut self, state: &dyn State) {
    let residuals = transform_to_residuals(
        &self.realization_container.inflow,
        self.season_id
    );
    
    let innovations = extract_innovations(
        &residuals,
        state.get_lagged_residuals(),
        &self.ar_coefficients
    );
    
    self.set_uncertainties_rhs(innovations);
}
```

**Pros**:
- Minimal API changes
- Separates scenario generation from LP preparation
- Preserves observations for output

**Cons**:
- More complex flow with two phases
- Still need State accessor for lags
- Potential for state inconsistency

**Effort**: ~4-5 hours

---

## Recommendations

### Immediate Next Steps

1. **Create failing test** to expose the issue:
   ```rust
   #[test]
   fn test_par_cut_coefficients_with_high_ar() {
       // Use φ = 0.95, 50 stages, verify cut lag coefficients
       // Compare against analytical gradient or finite differences
   }
   ```

2. **Choose implementation approach**:
   - **Short term**: Option 1 (modify PAR generator) - fastest, correct
   - **Long term**: Option 3 (refactor to innovations) - most theoretically sound

3. **Implementation roadmap**:
   - Phase A: Implement Option 1 (3 hours)
   - Phase B: Add comprehensive tests (2 hours)  
   - Phase C: Validate against known solutions (2 hours)
   - Phase D (future): Migrate to Option 3 if needed (10 hours)

### Testing Strategy

1. **Unit tests**: PAR generator returns correct innovation
2. **Integration tests**: AR constraint RHS equals innovation
3. **Regression tests**: Existing examples still work
4. **Validation tests**: 
   - Compare cuts with/without fix
   - Out-of-sample policy evaluation
   - Water value sensitivity to lagged inflows

### Success Criteria

- ✅ AR constraint RHS equals innovation ε_t
- ✅ Dual variables extracted from AR constraints
- ✅ Cuts include lag state coefficients
- ✅ All existing tests pass
- ✅ New validation tests pass
- ✅ Policy quality improves (or at least doesn't degrade)

---

## Technical Debt Assessment

### Current State
- ❌ AR constraint RHS incorrect (Y_t instead of ε_t)
- ❌ Cut coefficients for lagged states potentially wrong
- ❌ No validation tests for cut correctness
- ⚠️ Tests pass but don't validate optimality

### After Fix
- ✅ Mathematically correct AR formulation
- ✅ Proper state expansion implementation
- ✅ Validated cut generation
- ✅ Comprehensive test coverage

---

## References

1. **SDDP.jl AR Tutorial**: https://sddp.dev/stable/tutorial/arma/
   - State expansion trick explanation
   - Innovation vs full value distinction
   - Proper constraint formulation

2. **Pereira & Pinto (1991)**: "Multi-stage stochastic optimization applied to energy planning"
   - Original SDDP paper
   - State space formulation

3. **Shapiro et al. (2011)**: "Analysis of stochastic dual dynamic programming method"
   - Theoretical foundations
   - Cut validity conditions

---

## Appendix: Code Locations

### Key Files
- `src/par_generator.rs`: PAR generator implementation
- `src/noise_model_cache.rs`: Scenario generation
- `src/subproblem.rs`: LP setup and scenario realization
- `src/state.rs`: StorageAndInflowState with AR constraints
- `src/space_transform.rs`: Transformation utilities (Phase 1)

### Critical Functions
- `PeriodicARGenerator::generate_next_for_season()`: Returns Y_t
- `Subproblem::realize_uncertainties()`: Sets scenario values
- `StorageAndInflowState::set_inflows_in_subproblem()`: Sets AR constraint RHS
- `StorageAndInflowState::add_constraints_to_subproblem()`: Creates AR constraints

---

## Conclusion

The current PAR implementation does not correctly implement the **state expansion trick** for SDDP. The AR constraint RHS receives full observations (Y_t) instead of innovations (ε_t), leading to incorrect dual variables and suboptimal cuts.

**Recommended path forward**: Implement **Option 1** (modify PAR generator to return innovations) as a first step, then consider migrating to **Option 3** (full refactor) for long-term correctness.

**Priority**: High - affects policy quality for all PAR models with StorageAndInflowState.

**Risk**: Medium - existing tests pass, but policies may be suboptimal. Real-world impact depends on AR coefficient strength and horizon length.
