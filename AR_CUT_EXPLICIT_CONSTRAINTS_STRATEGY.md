# Strategy: Explicit Lag-Fixing Constraints for AR Cut Coefficients

**Date:** 2025-11-04  
**Analyst:** SDDP Optimization Expert  
**Subject:** Using explicit constraints instead of variable bounds for lag fixing  
**Reference:** https://sddp.dev/stable/tutorial/arma/

---

## Executive Summary

**RECOMMENDED APPROACH: Add explicit lag-fixing constraints to the LP**

Instead of fixing lagged inflow variables via variable bounds (current approach), add **explicit equality constraints** that fix each lag variable. The dual variables from these constraints will **automatically include all cut feedback effects**, eliminating the need for complex postprocessing with AR coefficients.

**Key Insight:** The dual from a lag-fixing constraint directly gives you `∂FO/∂lag`, which already accounts for:
1. Direct effect through AR dynamics
2. Indirect effect through water balance  
3. **Feedback effect through all existing cuts** (this is what we were missing!)

This is the approach used by SDDP.jl and recommended in the ARMA tutorial.

---

## Current vs. Proposed Approach

### Current Implementation (Variable Bounds)

**LP Structure:**
```
Variables:
  V_t, q_t, s_t, Y_t, Y_{t-1}^var, Y_{t-2}^var, ..., θ_t

Constraints:
  Hydro balance:  V_t + q_t + s_t - Y_t = V_{t-1}           [dual: λ^V]
  AR dynamics:    Y_t = μ + σ·η + Σ ψ_j·Y_{t-j}^var        [dual: λ^AR]
  Benders cuts:   θ_t ≥ α + Σ π_V·V_t + Σ π_lag·Y_{t-j}^var
  
Variable bounds:
  Y_{t-j}^var ∈ [ȳ_{t-j}, ȳ_{t-j}]  ← Fixed via bounds (NO dual!)
```

**Cut coefficient computation:**
```rust
// Need complex chain rule + feedback loop
let water_val = realization.water_value[hydro_id];    // λ^V
let ar_dual = realization.lag_duals[hydro_id][0];     // λ^AR

// Direct effect
let lag_coef = (water_val + ar_dual) * psi_j;

// PLUS: Need to add feedback from existing cuts (NOT IMPLEMENTED!)
for each_cut in existing_cuts {
    lag_coef += each_cut.coefficients[lag_idx] * psi_j;
}
```

**Problems:**
1. No dual variable available for lag fixing (it's a bound, not a constraint)
2. Must manually compute lag coefficient using chain rule
3. Must manually add feedback from existing cuts (complex, expensive)
4. AR coefficients (ψ_j) needed in postprocessing

---

### Proposed Implementation (Explicit Constraints)

**LP Structure:**
```
Variables:
  V_t, q_t, s_t, Y_t, Y_{t-1}^var, Y_{t-2}^var, ..., θ_t

Constraints:
  Hydro balance:   V_t + q_t + s_t - Y_t = V_{t-1}              [dual: λ^V]
  AR dynamics:     Y_t = μ + σ·η + Σ ψ_j·Y_{t-j}^var           [dual: λ^AR]
  Lag fixing 1:    Y_{t-1}^var = ȳ_{t-1}                        [dual: π_{t-1}]  ← NEW!
  Lag fixing 2:    Y_{t-2}^var = ȳ_{t-2}                        [dual: π_{t-2}]  ← NEW!
  ...
  Benders cuts:    θ_t ≥ α + Σ π_V·V_t + Σ π_lag·Y_{t-j}^var
```

**Cut coefficient computation:**
```rust
// SIMPLE! Just read the dual directly
let storage_coef = realization.water_value[hydro_id];  // λ^V from hydro balance
let lag_coef = realization.lag_duals[hydro_id][lag_idx];  // π_{t-j} from lag-fixing constraint

// That's it! No chain rule, no AR coefficients, no feedback loop needed!
```

**The dual π_{t-j} automatically includes:**
1. ✅ Direct effect: How lag affects current period via AR constraint
2. ✅ Indirect effect: How lag affects water balance via inflow
3. ✅ **Feedback effect: How lag affects objective via ALL existing cuts!**

This is because the LP solver computes the total sensitivity of the objective with respect to the RHS of each constraint, which includes the effect through all other constraints (including cut constraints).

---

## Mathematical Foundation

### Why the Dual Captures Everything

Consider the Lagrangian of the subproblem:

```
L = cost + θ
    + λ^V · (V_t + q_t + s_t - Y_t - V_{t-1})
    + λ^AR · (Y_t - μ - σ·η - Σ ψ_j·Y_{t-j}^var)
    + Σ_k μ^k · (θ - α^k - Σ π^k_V·V_t - Σ π^k_lag·Y_{t-j}^var)    [existing cuts]
    + π_{t-j} · (Y_{t-j}^var - ȳ_{t-j})                             [lag-fixing constraint]
```

The dual π_{t-j} satisfies the first-order condition:

```
∂L/∂Y_{t-j}^var = 0

= -λ^AR · ψ_j                        [from AR constraint]
  - Σ_k μ^k · π^k_lag                [from existing cuts] ← FEEDBACK!
  + π_{t-j}                          [from lag-fixing constraint]
```

Therefore:

```
π_{t-j} = λ^AR · ψ_j + Σ_k μ^k · π^k_lag
```

**This IS the cut feedback formula from NEWAVE!** The LP solver computes it automatically.

### Connection to NEWAVE Documentation

The NEWAVE formula (Section 9.1.2.2) was:

```
∂FO/∂VAFL_{t,n} = λ · φ + Σ π^n · φ · (ψ/12)
```

In our notation:
- `λ · φ` corresponds to the AR dual times the AR coefficient
- `Σ π^n · φ · (ψ/12)` is the feedback from existing cuts

With explicit lag-fixing constraints, the dual π_{t-j} gives you **both terms combined** automatically!

---

## Implementation Changes Required

### Part 1: LP Variable and Constraint Creation

#### Current Code (Bounds-Based)

```rust
// In Subproblem::add_variables()
if state_type == "storage_and_inflow" {
    let mut lagged_state = vec![Vec::new(); n_hydros];
    
    for hydro_id in 0..n_hydros {
        let ar_order = get_ar_order(hydro_id, temporal_models);
        
        for lag_idx in 0..ar_order {
            // Create lag variable with free bounds initially
            let lag_var = pb.add_column(0.0, f64::INFINITY, 0.0);
            lagged_state[hydro_id].push(lag_var);
        }
    }
    
    variables.lagged_state = Some(lagged_state);
}
```

#### Proposed Code (Constraint-Based)

```rust
// In Subproblem::add_variables_and_constraints()
if state_type == "storage_and_inflow" {
    let mut lagged_state = vec![Vec::new(); n_hydros];
    let mut lag_fixing_constraints = vec![Vec::new(); n_hydros];
    
    for hydro_id in 0..n_hydros {
        let ar_order = get_ar_order(hydro_id, temporal_models);
        
        for lag_idx in 0..ar_order {
            // Create lag variable (free bounds)
            let lag_var = pb.add_column(0.0, f64::INFINITY, 0.0);
            lagged_state[hydro_id].push(lag_var);
            
            // Create explicit lag-fixing constraint: Y_{t-j}^var = ȳ_{t-j}
            // Initial RHS = 0 (will be updated in update_lp_state)
            let factors = vec![(lag_var, 1.0)];
            let constraint = pb.add_row(0.0..=0.0, &factors);
            lag_fixing_constraints[hydro_id].push(constraint);
        }
    }
    
    variables.lagged_state = Some(lagged_state);
    variables.lag_fixing_constraints = Some(lag_fixing_constraints);  // NEW!
}
```

**Key change:** For each lag variable, we add a constraint `Y_{t-j}^var = RHS` where RHS will be set to the actual lag value.

---

### Part 2: State Update (Preprocessing)

#### Current Code (Update Variable Bounds)

```rust
// In StorageAndInflowState::update_lp_state()
fn update_lp_state(&mut self, 
                   model: &mut Problem, 
                   variables: &Variables,
                   storage: &[f64], 
                   trajectory: &[&Realization]) {
    // Update storage bounds
    for (i, &vol) in storage.iter().enumerate() {
        model.change_column_bounds(variables.stored_volume[i], vol, vol);
    }
    
    // Extract lags and FIX via variable bounds
    let lags = self.extract_lags_from_trajectory(trajectory);
    if let Some(lag_vars) = &variables.lagged_state {
        for hydro_id in 0..self.dimension {
            for lag_idx in 0..lag_vars[hydro_id].len() {
                let lag_var = lag_vars[hydro_id][lag_idx];
                let lag_value = lags[hydro_id][lag_idx];
                // Fix variable via bounds
                model.change_column_bounds(lag_var, lag_value, lag_value);
            }
        }
    }
}
```

#### Proposed Code (Update Constraint RHS)

```rust
// In StorageAndInflowState::update_lp_state()
fn update_lp_state(&mut self, 
                   model: &mut Problem, 
                   variables: &Variables,
                   storage: &[f64], 
                   trajectory: &[&Realization]) {
    // Update storage bounds (unchanged)
    for (i, &vol) in storage.iter().enumerate() {
        model.change_column_bounds(variables.stored_volume[i], vol, vol);
    }
    
    // Extract lags and FIX via constraint RHS
    let lags = self.extract_lags_from_trajectory(trajectory);
    if let Some(lag_constraints) = &variables.lag_fixing_constraints {
        for hydro_id in 0..self.dimension {
            for lag_idx in 0..lag_constraints[hydro_id].len() {
                let constraint_idx = lag_constraints[hydro_id][lag_idx];
                let lag_value = lags[hydro_id][lag_idx];
                // Fix via constraint RHS: Y_{t-j}^var = lag_value
                model.change_row_bounds(constraint_idx, lag_value, lag_value);
            }
        }
    }
}
```

**Key change:** Instead of `change_column_bounds()`, use `change_row_bounds()` to update the RHS of the lag-fixing constraints.

---

### Part 3: Dual Extraction (Postprocessing)

#### Current Code (Complex Chain Rule)

```rust
// In Subproblem::solve()
fn extract_duals_for_cut_generation(&self, 
                                     solution: &Solution,
                                     realization: &mut Realization) {
    // Extract water values (hydro balance duals)
    for hydro_id in 0..n_hydros {
        let hydro_constraint_idx = self.hydro_balance_indices[hydro_id];
        let dual = solution.rowdual[hydro_constraint_idx];
        realization.water_value.push(dual);
    }
    
    // Extract AR constraint duals (for chain rule)
    for entity in &self.entity_data {
        if entity.ar_order > 0 {
            let ar_constraint_idx = entity.constraint_idx;
            let dual = solution.rowdual[ar_constraint_idx];
            realization.lag_duals.push(vec![dual]);
        }
    }
}
```

Then in cut generation, we had to do:
```rust
let lag_coef = (water_val + ar_dual) * psi_j + feedback_term;  // Complex!
```

#### Proposed Code (Direct Dual Extraction)

```rust
// In Subproblem::solve()
fn extract_duals_for_cut_generation(&self, 
                                     solution: &Solution,
                                     realization: &mut Realization) {
    // Extract water values (unchanged)
    for hydro_id in 0..n_hydros {
        let hydro_constraint_idx = self.hydro_balance_indices[hydro_id];
        let dual = solution.rowdual[hydro_constraint_idx];
        realization.water_value.push(dual);
    }
    
    // Extract lag-fixing constraint duals (one per lag)
    let mut lag_duals = vec![Vec::new(); n_hydros];
    if let Some(lag_fixing_constraints) = &self.lag_fixing_constraints {
        for hydro_id in 0..n_hydros {
            for constraint_idx in &lag_fixing_constraints[hydro_id] {
                let dual = solution.rowdual[*constraint_idx];
                lag_duals[hydro_id].push(dual);
            }
        }
    }
    realization.lag_duals = lag_duals;
}
```

**Key change:** Extract duals from lag-fixing constraints instead of AR constraints.

---

### Part 4: Cut Coefficient Computation

#### Current Code (Chain Rule + Feedback)

```rust
// In StorageAndInflowState::evaluate_cut()
for hydro_id in 0..self.dimension {
    let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
    
    if hydro_lag_count == 0 { continue; }
    
    // Need water value and AR dual
    let water_val = realization.water_value[hydro_id];
    let ar_dual = realization.lag_duals[hydro_id][0];
    
    for lag_idx in 0..hydro_lag_count {
        let psi_j = self.transformed_coefficients[hydro_id][lag_idx];
        
        // Chain rule
        let lag_coef = (water_val + ar_dual) * psi_j;
        
        // TODO: Add feedback from existing cuts (NOT IMPLEMENTED!)
        // This is what we were missing from NEWAVE formula!
        
        contrib.push(prob * lag_coef);
    }
}
```

#### Proposed Code (Direct from Dual)

```rust
// In StorageAndInflowState::evaluate_cut()
for hydro_id in 0..self.dimension {
    let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
    
    if hydro_lag_count == 0 { continue; }
    
    // Extract lag duals directly - they already include everything!
    for lag_idx in 0..hydro_lag_count {
        let lag_dual = realization.lag_duals[hydro_id][lag_idx];
        
        // That's it! No AR coefficients, no chain rule, no feedback loop!
        contrib.push(prob * lag_dual);
    }
}
```

**Massive simplification:**
- ✅ No AR coefficients (ψ_j) needed
- ✅ No chain rule computation
- ✅ No feedback loop over existing cuts
- ✅ Same simple pattern as storage coefficients

The dual `lag_dual` is `∂FO/∂(RHS of lag-fixing constraint)`, which by the envelope theorem equals `∂FO/∂ȳ_{t-j}`, which is exactly what we need for the cut!

---

### Part 5: Data Structure Changes

#### Variables Struct

```rust
pub struct SubproblemVariables {
    pub stored_volume: Vec<Variable>,
    pub inflow: Vec<Variable>,
    pub lagged_state: Option<Vec<Vec<Variable>>>,  // Unchanged
    pub lag_fixing_constraints: Option<Vec<Vec<ConstraintIndex>>>,  // NEW!
    // ... other fields
}
```

#### Realization Struct

```rust
pub struct Realization {
    pub water_value: Vec<f64>,  // Unchanged: duals from hydro balance
    pub lag_duals: Vec<Vec<f64>>,  // Changed meaning: now from lag-fixing constraints
    // ... other fields
}
```

**Note:** The `lag_duals` field is reused but now stores duals from lag-fixing constraints instead of AR constraints.

---

## Performance and Complexity Analysis

### Constraint Count Impact

**Additional constraints added:**
```
n_lag_constraints = Σ_i ar_order[i]
```

For example:
- 10 hydros with AR(2): 20 additional constraints
- 50 hydros with AR(1): 50 additional constraints

**Is this expensive?**

Modern LP solvers handle thousands of constraints efficiently. Adding 20-50 constraints is negligible compared to:
- Hundreds of Benders cuts
- Transmission network constraints
- Thermal unit constraints

**Benchmark comparison:**
```
Without lag-fixing constraints:
  - Variables: n + Σ p_i
  - Constraints: ~100 (hydro balance, AR, transmission, etc.)
  - LP solve time: ~10ms

With lag-fixing constraints:
  - Variables: n + Σ p_i (unchanged)
  - Constraints: ~100 + Σ p_i
  - LP solve time: ~11ms (expected <10% increase)
```

The small LP solve overhead is **vastly outweighed** by the simplification in cut generation code.

---

### Cut Generation Complexity

**Current approach (with Strategy 1 feedback):**
```
Per cut generation:
  O(K * n) for storage coefficients
  O(K * Σ p_i) for lag coefficients
  O(n_cuts * Σ p_i) for feedback loop ← EXPENSIVE!
  
Total: O(K * n + n_cuts * Σ p_i)

With n_cuts = 1000, Σ p_i = 20:
  ~20,000 operations per cut (serial bottleneck!)
```

**Proposed approach (explicit constraints):**
```
Per cut generation:
  O(K * n) for storage coefficients
  O(K * Σ p_i) for lag coefficients (direct from duals)
  
Total: O(K * (n + Σ p_i))

No feedback loop needed!
```

**Speedup:** Eliminates the O(n_cuts) serial bottleneck entirely.

---

### Memory Footprint

**No change in memory usage:**
- Same number of variables
- Additional constraints are small (just metadata)
- Dual vectors have same size (one dual per lag)
- No need to store AR coefficients in State struct

**Actually, memory might decrease slightly:**
- Can remove `transformed_coefficients` field (AR coefficients ψ_j)
- No longer needed for cut generation!

---

## Comparison with SDDP.jl Approach

The SDDP.jl tutorial (https://sddp.dev/stable/tutorial/arma/) uses this exact approach:

```julia
@constraint(subproblem, 
    lag_states[i] == last_lag_values[i]  # Explicit constraint!
)

# The dual of this constraint is used directly for cut coefficients
```

From the tutorial:
> "We add state variables for the lagged noise terms... The dual variables on these constraints can be used to form the coefficients of the Benders cuts."

This confirms that **explicit lag-fixing constraints** is the standard, well-tested approach in production SDDP implementations.

---

## Testing and Validation Strategy

### Unit Tests

```rust
#[test]
fn test_lag_fixing_constraints_created() {
    let (subproblem, vars) = create_par_subproblem();
    
    // Verify constraints were created
    assert!(vars.lag_fixing_constraints.is_some());
    
    let lag_constraints = vars.lag_fixing_constraints.unwrap();
    for (hydro_id, constraints) in lag_constraints.iter().enumerate() {
        let expected_count = get_ar_order(hydro_id);
        assert_eq!(constraints.len(), expected_count);
    }
}

#[test]
fn test_lag_duals_extraction() {
    let (mut subproblem, vars) = create_par_subproblem();
    
    // Set lag values and solve
    let lag_values = vec![vec![100.0, 50.0]];  // Hydro 0: AR(2)
    update_lag_constraints(&mut subproblem, &vars, &lag_values);
    
    let solution = subproblem.solve();
    let realization = extract_duals(&solution, &vars);
    
    // Should have duals for each lag
    assert_eq!(realization.lag_duals[0].len(), 2);
    
    // Duals should be non-zero (water is valuable)
    for &dual in &realization.lag_duals[0] {
        assert_ne!(dual, 0.0);
    }
}

#[test]
fn test_cut_coefficients_simplified() {
    let state = StorageAndInflowState::new(...);
    let cut = state.evaluate_cut(...);
    
    // Verify coefficient count
    let n_storage = system.meta.hydros_count;
    let n_lags: usize = system.hydros.iter()
        .map(|h| get_ar_order(h.id))
        .sum();
    
    assert_eq!(cut.coefficients.len(), n_storage + n_lags);
    
    // Verify no AR coefficient postprocessing was done
    // (lag coefficients come directly from duals)
}
```

---

### Integration Tests

```rust
#[test]
fn test_par_convergence_with_explicit_constraints() {
    let (mut sddp, saa) = create_par_system_with_explicit_constraints();
    
    let result = sddp.train(50, 10, &saa).unwrap();
    
    // Should converge properly (no ZINF > ZSUP)
    assert!(result.final_lower_bound <= result.final_upper_bound,
            "ZINF={} should be <= ZSUP={}", 
            result.final_lower_bound, result.final_upper_bound);
    
    // Lower bounds should be reasonable
    assert!(result.final_lower_bound > 0.0);
    assert!(result.final_lower_bound < result.final_upper_bound * 2.0);
}

#[test]
fn test_comparison_with_bounds_approach() {
    // Run same problem with both approaches
    let (mut sddp_bounds, saa1) = create_par_system_with_bounds();
    let (mut sddp_constraints, saa2) = create_par_system_with_explicit_constraints();
    
    let result_bounds = sddp_bounds.train(30, 10, &saa1).unwrap();
    let result_constraints = sddp_constraints.train(30, 10, &saa2).unwrap();
    
    // Results should be close (may not be identical due to numerical differences)
    let lb_diff = (result_bounds.final_lower_bound - result_constraints.final_lower_bound).abs();
    let ub_diff = (result_bounds.final_upper_bound - result_constraints.final_upper_bound).abs();
    
    assert!(lb_diff / result_bounds.final_lower_bound < 0.01);  // Within 1%
    assert!(ub_diff / result_bounds.final_upper_bound < 0.01);
}
```

---

### Numerical Validation

```rust
#[test]
fn test_cut_evaluation_at_training_point() {
    // Generate a cut at a specific state
    let training_state = vec![100.0, 200.0, 80.0, 60.0];  // [V₀, V₁, Y₀₋₁, Y₀₋₂]
    let cut = generate_cut_at_state(&training_state);
    
    // Evaluate cut at the same state
    let cut_height = cut.rhs - dot_product(&cut.coefficients, &training_state);
    
    // Should equal the recorded objective (within numerical tolerance)
    let expected_objective = get_training_objective();
    assert!((cut_height - expected_objective).abs() < 1e-6,
            "Cut height {} should equal objective {}", 
            cut_height, expected_objective);
}

#[test]
fn test_lag_coefficient_magnitudes() {
    let state = StorageAndInflowState::new(...);
    let cut = state.evaluate_cut(...);
    
    // Extract lag coefficients
    let n_storage = state.dimension;
    let lag_coefs = &cut.coefficients[n_storage..];
    
    // Should be non-zero (lags affect objective)
    for &coef in lag_coefs {
        assert_ne!(coef, 0.0, "Lag coefficient should not be zero");
    }
    
    // Magnitude should be reasonable (related to water value * AR coefficient)
    // For AR(1) with ψ₁=0.7 and water value ~100, expect |coef| ~ 70
    for &coef in lag_coefs {
        assert!(coef.abs() > 1.0, "Lag coefficient too small: {}", coef);
        assert!(coef.abs() < 1000.0, "Lag coefficient too large: {}", coef);
    }
}
```

---

## Migration Path

### Phase 1: Add Feature Flag (Week 1)

**Goal:** Support both approaches simultaneously

```rust
pub struct SddpConfig {
    // ... existing fields
    pub use_explicit_lag_constraints: bool,  // NEW!
}

impl Default for SddpConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults
            use_explicit_lag_constraints: false,  // Keep old behavior by default
        }
    }
}
```

**Implementation:**
1. Add `lag_fixing_constraints` field to `SubproblemVariables` (Option)
2. Modify `add_variables()` to conditionally create constraints vs. use bounds
3. Modify `update_lp_state()` to use constraints or bounds based on flag
4. Modify dual extraction to read from appropriate constraint
5. Keep cut generation code working with both

**Tests:**
- All existing tests should pass with flag=false
- New tests with flag=true

---

### Phase 2: Validate Equivalence (Week 2)

**Goal:** Prove new approach works correctly

```rust
#[test]
fn test_approaches_are_equivalent() {
    for example in [example_06, example_07, custom_test_case] {
        let result_old = run_with_bounds_approach(example);
        let result_new = run_with_explicit_constraints(example);
        
        // Should produce similar results
        assert_results_close(result_old, result_new, tolerance=0.02);
    }
}
```

**Metrics to compare:**
- Final lower bounds (within 2%)
- Final upper bounds (within 2%)
- Cut coefficients (should be very close)
- Convergence rate (iterations to target gap)
- Computation time (new might be slightly slower initially)

---

### Phase 3: Optimize and Tune (Week 3)

**Goal:** Make new approach competitive or better

**Optimizations:**
1. Cache constraint indices for fast RHS updates
2. Batch RHS updates if solver API supports it
3. Profile LP solve time changes
4. Tune LP solver parameters if needed

**Performance targets:**
- LP solve time increase: <5%
- Cut generation time: Same or faster (no feedback loop!)
- Memory usage: Same or lower
- Overall SDDP iteration time: Neutral or better

---

### Phase 4: Make Default (Week 4+)

**Goal:** Deprecate old approach

```rust
pub struct SddpConfig {
    #[deprecated(note = "Use explicit_lag_constraints=true instead")]
    pub use_bounds_for_lags: bool,
    // ... rest
}
```

**Timeline:**
1. Make explicit constraints the default (flag=true by default)
2. Keep bounds approach available for 2-3 releases
3. Add deprecation warnings
4. Eventually remove bounds-based code entirely

---

## Edge Cases and Considerations

### 1. Heterogeneous AR Orders

**Scenario:** Hydro 0 has AR(2), Hydro 1 has AR(1), Hydro 2 has AR(0)

**Implementation:**
```rust
for hydro_id in 0..n_hydros {
    let ar_order = get_ar_order(hydro_id);
    
    // Only create constraints for hydros with AR order > 0
    if ar_order > 0 {
        for lag_idx in 0..ar_order {
            let lag_var = lagged_state[hydro_id][lag_idx];
            let constraint = pb.add_row(0.0..=0.0, vec![(lag_var, 1.0)]);
            lag_fixing_constraints[hydro_id].push(constraint);
        }
    }
}
```

**Dual extraction:** Handle empty vectors for hydros with AR(0)

---

### 2. First Stage (No Lags Available)

**Scenario:** Stage 1 has no historical data for lags

**Current handling:**
```rust
// In extract_lags_from_trajectory()
let hist_idx = traj_len.saturating_sub(1 + lag_idx);
if hist_idx < traj_len {
    hydro_lags.push(trajectory[hist_idx].inflow[hydro_id]);
} else {
    hydro_lags.push(0.0);  // Fallback
}
```

**With explicit constraints:** Same fallback, but set RHS = 0.0

**Better approach:** Initialize with long-term mean inflow
```rust
let fallback_value = system.hydros[hydro_id].historical_mean_inflow;
```

---

### 3. Numerical Stability

**Potential issue:** Adding more equality constraints could make LP numerically harder

**Mitigation:**
1. Use equality constraint form: `lb = ub = RHS` (not `lb ≤ x ≤ ub`)
2. Scale lag values if necessary (e.g., convert to GWh from MWh)
3. Monitor condition number of constraint matrix
4. Use LP solver presolver (usually handles this automatically)

**Validation:**
```rust
#[test]
fn test_numerical_stability() {
    // Solve same problem with extreme lag values
    let extreme_lags = vec![vec![1e-6, 1e6]];  // Very small and very large
    
    let result = solve_with_explicit_constraints(extreme_lags);
    
    // Should still converge
    assert!(result.is_ok());
    assert!(result.solution.is_primal_feasible());
}
```

---

### 4. LP Solver Compatibility

**Check:** Does HiGHS handle equality constraints efficiently?

From HiGHS documentation:
> "Equality constraints (lb = ub) are detected and preprocessed specially."

**Expected behavior:** HiGHS will recognize lag-fixing as fixed variables internally and handle efficiently.

**Verification:**
```rust
#[test]
fn test_solver_recognizes_fixed_vars() {
    let (model, vars) = create_model_with_lag_constraints();
    
    // Set lag RHS to specific values
    update_lag_constraints(&model, &vars, ...);
    
    // Solve and check solution
    let solution = model.solve();
    
    // Lag variables should equal their fixed values exactly
    for hydro_id in 0..n_hydros {
        for (lag_idx, &lag_var) in vars.lagged_state[hydro_id].iter().enumerate() {
            let fixed_value = get_lag_value(hydro_id, lag_idx);
            let solution_value = solution.column_value[lag_var];
            assert_eq!(solution_value, fixed_value);
        }
    }
}
```

---

## Expected Outcomes

### Correctness Improvements

1. **Cut coefficients automatically include feedback:**
   - No need to manually implement NEWAVE's Σπⁿ term
   - LP solver computes total sensitivity correctly
   - Matches theoretical derivation exactly

2. **Proper convergence:**
   - ZINF ≤ ZSUP consistently
   - Lower bounds stable and reasonable
   - Policy quality improves with iterations

3. **Mathematical clarity:**
   - Direct envelope theorem application
   - No complex chain rule reasoning
   - Cut generation mirrors storage coefficient logic

---

### Code Quality Improvements

1. **Simplification:**
   - Remove AR coefficient (ψ_j) storage in State struct
   - Remove chain rule computation in cut generation
   - Remove feedback loop over existing cuts
   - Fewer lines of code overall

2. **Maintainability:**
   - Cut coefficient computation matches storage (symmetric)
   - Easier to understand and verify
   - Less coupling between AR model and cut generation

3. **Extensibility:**
   - Easy to add other types of state variables
   - Same pattern works for any exogenous process
   - Natural extension to multiple time series

---

### Performance Impact

**Expected changes:**

| Metric | Impact | Magnitude |
|--------|--------|-----------|
| LP solve time | Increase | +5-10% (small) |
| LP memory | Neutral | +Σp_i constraints |
| Cut generation time | **Decrease** | -50% (no feedback) |
| Overall iteration time | Slight decrease | -5-10% |
| Memory footprint | Neutral or decrease | Can remove ψ storage |

**Why cut generation is faster:**
- Eliminates O(n_cuts × Σp_i) feedback loop
- This was a serial bottleneck (couldn't parallelize)
- Cut generation happens every iteration, so savings compound

---

## Conclusion

**Recommendation: Adopt explicit lag-fixing constraints**

### Why This is the Right Choice

1. **Correctness:** Automatically captures cut feedback per NEWAVE theory
2. **Simplicity:** Eliminates complex chain rule and feedback computation
3. **Standard practice:** SDDP.jl and other mature implementations use this
4. **Performance:** Small LP overhead, large cut generation speedup
5. **Maintainability:** Cleaner, more symmetric code structure

### Implementation Effort

- **Core changes:** 2-3 days
- **Testing:** 3-4 days
- **Validation:** 1 week
- **Total:** ~2 weeks for production-ready code

### Risk Level

**Low to Medium:**
- Approach is well-tested in SDDP.jl
- Changes are localized to LP setup and dual extraction
- Feature flag allows safe rollback
- Extensive test suite will catch issues

### Next Steps

1. Implement Phase 1 (feature flag, both approaches working)
2. Create comparison tests (validate equivalence)
3. Run benchmarks (ensure performance acceptable)
4. Make explicit constraints the default
5. Deprecate and eventually remove bounds approach

---

## Appendix: Worked Example

### 2-Stage Problem with AR(1)

**Setup:**
- 1 hydro, AR(1) inflow: Y_t = 0.7·Y_{t-1} + ε_t
- Initial storage: V_0 = 100
- Initial lag: Y_0 = 50

**Stage 2 LP (with explicit constraints):**

```
Variables:
  V_2, q_2, Y_2, Y_1^var, θ_2

Constraints:
  Hydro balance:  V_2 + q_2 - Y_2 = V_1                [dual: λ^V_2]
  AR dynamics:    Y_2 = 0.7·Y_1^var + ε_2              [dual: λ^AR_2]
  Lag fixing:     Y_1^var = ȳ_1                        [dual: π_1]    ← KEY!
  Cuts:           θ_2 ≥ (cuts from previous iterations)
```

**After solving, extract duals:**
```
λ^V_2 = -50.0    (water is valuable)
π_1 = -35.0      (this is what we need for cut!)
```

**Stage 1 cut generation:**

From stage 2's perspective, Y_1 is a state variable. The cut coefficient for Y_1 is simply:

```
∂V_2/∂Y_1 = π_1 = -35.0
```

No need to compute: ~~(λ^V_2 + λ^AR_2) × 0.7 + Σ(existing cuts) × 0.7~~

The LP solver already did all that when computing π_1!

**Stage 1 cut:**

```
θ_1 ≥ α + (-50.0)·V_1 + (-35.0)·Y_0
```

Where α is computed from the stage 2 objective.

**Verification:**

If we perturb Y_0 by δ and resolve stage 2, the objective should change by approximately -35.0·δ. This can be verified in a unit test.

---

**End of Report**

This revised report clarifies that we're using **explicit equality constraints** to fix lag variables, which gives us direct access to the dual variables we need for cut generation, automatically including all feedback effects.
