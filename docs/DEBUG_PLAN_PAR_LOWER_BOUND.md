# SDDP Cut Generation Debug Plan: PAR(1) Lower Bound Issue

## Problem Statement

**Symptom:** Lower bound (~20,000) consistently exceeds simulation cost (~2,700) by an order of magnitude in Example 07 (PAR(1) model with inflow state).

**Expected Behavior:** Lower bound should approach simulation cost from below (or be close in expectation), providing a valid lower bound on the optimal policy value.

**Current Observation:**
```
Iteration | Upper Bound | Lower Bound | Gap Ratio
----------|-------------|-------------|----------
    1     |   40,003    |   21,563    |   1.85x
    2     |   39,565    |   21,563    |   1.84x
   20     |   39,717    |   19,844    |   2.00x

Final Simulation: 20,082 ± 117
```

The lower bound is **systematically overestimating** the true value function, which violates the fundamental property of Benders cuts.

---

## Mathematical Foundation Review

### 1. PAR(1) Model in Observation Space

From `par_derivation.pdf`, the PAR(1) model in observation space is:

```
Y_t = ψ_1 · Y_{t-1} + η_t
```

Where:
- `Y_t`: Inflow observation at time t (actual MW)
- `ψ_1 = φ_1 · (σ_t / σ_{t-1})`: Transformed AR coefficient (0.7 in Example 07)
- `η_t = μ_t - ψ_1·μ_{t-1} + σ_t·ε_t`: Transformed innovation

### 2. LP Subproblem Formulation

The LP contains three constraint types relevant to cut generation:

#### a. Hydro Balance Constraint
```
storage_t + turbined_t + spillage_t - inflow_t - upstream_flows = initial_storage
```
- **Dual variable:** `λ^hydro` (water value)
- **Physical meaning:** Marginal value of water in reservoir
- **Expected sign:** Negative (in minimization, relaxing storage reduces cost)

#### b. Inflow Observation Constraint
```
Y_t - ψ_1·Y_{t-1} - σ_t·η_t = μ_t - ψ_1·μ_{t-1}
```
- **Dual variable:** `λ^obs` (not directly used in cuts)
- **Purpose:** Enforces AR dynamics in observation space
- **RHS:** Deterministic seasonal adjustment

#### c. Lag-Fixing Constraint
```
Y_{t-1} = value_from_state
```
- **Dual variable:** `λ^lag` (lag dual)
- **Purpose:** Fixes lagged inflow state variable
- **Physical meaning:** Sensitivity of future cost to lag value

### 3. Benders Cut Theory

The cut approximates the value function:

```
θ_t ≥ E[V_{t+1}(x_t, ξ_{t+1})] - π_t^T · (x_t - x̄_t)
```

Rearranging for implementation:
```
θ_t ≥ [E[V_{t+1}] - π_t^T · x̄_t] + π_t^T · x_t
       └──────── RHS ────────┘     └─ coefficients ─┘
```

Where:
- `x_t = [storage_t, Y_t]`: State vector
- `π_t = [∂V/∂storage, ∂V/∂Y_t]`: Cut coefficients (subgradient)
- `x̄_t`: State where cut was generated

### 4. Cut Coefficient Derivation via Chain Rule

For PAR models, the lag Y_{t-1} affects the future in two ways:

1. **Direct effect:** Through the lag-fixing constraint
2. **Indirect effect:** Through AR dynamics → inflow → hydro balance

The **correct** cut coefficient for the lag is:

```
∂V_{t+1}/∂Y_t = λ^hydro · (∂inflow_{t+1}/∂Y_t) + λ^lag

Where:
∂inflow_{t+1}/∂Y_t = ψ_1  (from AR constraint: Y_{t+1} = ψ_1·Y_t + η_{t+1})

Therefore:
π_lag = λ^hydro · ψ_1 + λ^lag
```

**Critical Insight:** The current code may be missing the `λ^hydro · ψ_1` term, only using `λ^lag` directly!

---

## Implementation Review

### Current Code Structure

#### File: `src/state.rs` (lines 985-1060)

**Cut Coefficient Computation:**
```rust
// Line 1012-1013: Storage coefficients
contrib.extend(realization.water_value.iter().map(|&val| prob * val));

// Line 1018-1030: Lag coefficients
for hydro_id in 0..self.dimension {
    let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
    if hydro_lag_count == 0 {
        continue;
    }
    
    let lag_duals = &realization.inflow_lag_duals[hydro_id];
    
    // ⚠️ POTENTIAL ISSUE: Using lag_dual directly
    for &lag_dual in lag_duals {
        contrib.push(prob * lag_dual);  // ← Missing chain rule?
    }
}
```

**Cut RHS Computation:**
```rust
// Line 1050-1051
let cut_rhs = objective
    - utils::dot_product(&cut_coefficients, state_coefficients);
```

### Hypothesis: Incomplete Chain Rule

**What the code does:**
```rust
π_lag = λ^lag  // From lag-fixing constraint dual
```

**What it should do (hypothesis):**
```rust
π_lag = λ^hydro · ψ_1 + λ^lag
```

**Why this matters:**

The lag Y_t directly impacts future inflow Y_{t+1} through the AR relationship:
```
Y_{t+1} = ψ_1 · Y_t + η_{t+1}
```

This means increasing Y_t by 1 unit increases inflow_{t+1} by ψ_1 units (0.7 in Example 07).

That extra inflow affects the hydro balance, which has marginal value λ^hydro.

Therefore, the total sensitivity is:
```
dV_{t+1}/dY_t = (effect through hydro balance) + (direct effect)
              = λ^hydro · ψ_1 + λ^lag
```

---

## Debugging Strategy

### Phase 1: Add Comprehensive Logging

Add detailed debug output to trace all components of cut generation.

#### Location 1: `src/state.rs`, function `evaluate_cut` (around line 1007)

```rust
fn evaluate_cut(
    &mut self,
    risk_measure: &dyn risk_measure::RiskMeasure,
    _forward_trajectory: &[&subproblem::Realization],
    branching_realizations: &[subproblem::Realization],
) -> cut::BendersCut {
    // ... existing code ...
    
    eprintln!("\n=== CUT GENERATION DEBUG ===");
    eprintln!("Iteration: {}, Forward Pass: {}", 
              self.get_iteration(), self.get_forward_pass_idx());
    eprintln!("Number of branching scenarios: {}", branching_realizations.len());
    
    for (index, realization) in branching_realizations.iter().enumerate() {
        let prob = adjusted_probabilities[index];
        eprintln!("\n--- Scenario {} (prob={:.4}) ---", index, prob);
        eprintln!("  Objective: {:.2f}", realization.total_stage_objective);
        
        // Storage coefficients
        for (h, &wv) in realization.water_value.iter().enumerate() {
            eprintln!("  Hydro {} water_value: {:.6f}", h, wv);
        }
        
        // Lag coefficients
        for hydro_id in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
            if hydro_lag_count == 0 {
                continue;
            }
            
            let lag_duals = &realization.inflow_lag_duals[hydro_id];
            let water_val = realization.water_value[hydro_id];
            
            eprintln!("  Hydro {} lag analysis:", hydro_id);
            for (lag_idx, &lag_dual) in lag_duals.iter().enumerate() {
                // Get ψ coefficient (need to add accessor)
                eprintln!("    Lag {}: lag_dual={:.6f}, water_val={:.6f}", 
                          lag_idx, lag_dual, water_val);
                eprintln!("    Current contrib: {:.6f} (= {:.4f} * {:.6f})", 
                          prob * lag_dual, prob, lag_dual);
                // TODO: Add ψ_1 and show alternative calculation
            }
        }
    }
    
    // ... continue with existing aggregation ...
    
    eprintln!("\n--- Aggregated Cut ---");
    eprintln!("Cut coefficients: {:?}", cut_coefficients);
    eprintln!("State at cut: {:?}", state_coefficients);
    eprintln!("Cut RHS: {:.6f}", cut_rhs);
    eprintln!("Expected objective: {:.6f}", objective);
    
    // CRITICAL VALIDATION: Cut should equal objective at training point
    let cut_at_state = cut_rhs - utils::dot_product(&cut_coefficients, state_coefficients);
    let error = (cut_at_state - objective).abs();
    eprintln!("Cut value at training state: {:.6f}", cut_at_state);
    eprintln!("Error (should be ~0): {:.6e}", error);
    
    if error > 1e-4 {
        eprintln!("⚠️  WARNING: Cut doesn't match objective at training point!");
        eprintln!("   This indicates incorrect cut generation!");
    }
    
    eprintln!("=== END CUT DEBUG ===\n");
    
    // ... return cut as before ...
}
```

#### Location 2: `src/subproblem.rs`, function `get_lag_duals_from_solution` (after line 1227)

```rust
// After extracting lag duals
eprintln!("\n=== LAG DUAL EXTRACTION ===");
for (entity_idx, entity_constraints) in lag_constraints.iter().enumerate() {
    if entity_constraints.is_empty() {
        continue;
    }
    
    let entity_data = &self.entity_data[entity_idx];
    eprintln!("Entity {:?}:{} (AR order {})", 
              entity_data.entity_type, 
              entity_data.entity_id,
              entity_data.ar_order);
    
    for (lag_idx, &constraint_idx) in entity_constraints.iter().enumerate() {
        let dual = solution.rowdual[constraint_idx];
        eprintln!("  Lag {} (constraint {}): dual = {:.6f}", 
                  lag_idx, constraint_idx, dual);
    }
}
eprintln!("=== END LAG DUAL EXTRACTION ===\n");
```

#### Location 3: `src/subproblem.rs`, function `realize_uncertainties` (track LP solution)

Add after solving the LP (around where water values are extracted):

```rust
eprintln!("\n=== LP SOLUTION ===");
eprintln!("Objective: {:.6f}", solution.obj);
eprintln!("Storage levels: {:?}", &solution.colsol[self.variables.stored_volume[0]..]);
eprintln!("Inflows: {:?}", &solution.colsol[self.variables.inflow[0]..]);
if let Some(lag_vars) = &self.variables.lagged_state {
    eprintln!("Lag variables:");
    for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
        for (lag_idx, &var_idx) in entity_lags.iter().enumerate() {
            eprintln!("  Entity {} Lag {}: {:.6f}", 
                      entity_idx, lag_idx, solution.colsol[var_idx]);
        }
    }
}
eprintln!("=== END LP SOLUTION ===\n");
```

### Phase 2: Run Example 07 with Debug Output

```bash
cd /home/rogerio/git/powers
cargo build --release
./target/release/powers examples/07-par-model-with-inflow-state/ 2>&1 | tee debug_output.txt
```

### Phase 3: Analyze Debug Output

Look for:

1. **Cut Validation Failure:**
   - Check if "Cut doesn't match objective at training point" warning appears
   - If yes, this confirms incorrect cut generation

2. **Water Value Signs:**
   - Water values should be **negative** (relaxing storage reduces cost)
   - If positive, there's a sign convention issue

3. **Lag Dual Magnitudes:**
   - Compare `lag_dual` vs `water_value`
   - Check if `lag_dual + water_value·ψ_1` makes more sense

4. **Objective Components:**
   - Track how objective evolves across iterations
   - Verify scenario probabilities sum to 1.0

### Phase 4: Mathematical Validation Test

Create a minimal test case to validate chain rule:

**Test Setup:**
- 2 stages
- 1 hydro
- PAR(1) with ψ_1 = 0.7
- Deterministic scenario

**Manual Calculation:**
1. Solve stage 2 LP → get λ^hydro_2
2. Fix lag Y_1 in stage 2 → get λ^lag_2
3. Compute analytical derivative: dV_2/dY_1 = λ^hydro_2 · 0.7 + λ^lag_2
4. Compare with code output

Add to `src/state.rs` tests:

```rust
#[test]
fn test_par_chain_rule_cut_coefficient() {
    // TODO: Implement minimal 2-stage PAR(1) test
    // Verify: π_lag = λ^hydro · ψ_1 + λ^lag
}
```

---

## Hypothesis Testing Plan

### Hypothesis 1: Missing Chain Rule Term

**Test:** Modify cut coefficient calculation to include chain rule.

**Location:** `src/state.rs`, line ~1029

**Current Code:**
```rust
for &lag_dual in lag_duals {
    contrib.push(prob * lag_dual);
}
```

**Modified Code (Test):**
```rust
let water_val = realization.water_value[hydro_id];

// Get ψ_1 coefficient from temporal model
// TODO: Need to add accessor to get psi from state or pass it in
let psi_1 = 0.7; // Hardcode for testing with Example 07

for (lag_idx, &lag_dual) in lag_duals.iter().enumerate() {
    // Include chain rule: π_lag = λ^hydro · ψ + λ^lag
    let chain_rule_coef = water_val * psi_1 + lag_dual;
    contrib.push(prob * chain_rule_coef);
    
    eprintln!("  Lag {} contribution:", lag_idx);
    eprintln!("    Original (lag_dual only): {:.6f}", prob * lag_dual);
    eprintln!("    Chain rule (w·ψ + λ): {:.6f}", prob * chain_rule_coef);
    eprintln!("    Difference: {:.6f}", prob * (chain_rule_coef - lag_dual));
}
```

**Expected Result if Correct:**
- Lower bound should decrease (cuts become less restrictive)
- Gap between lower bound and simulation should narrow
- Cut validation error should approach zero

### Hypothesis 2: Sign Convention Error

**Test:** Check if negating lag dual improves results.

**Rationale:** The lag-fixing constraint is `Y_{t-1} = value`. The dual λ^lag gives dL/d(value). But we want dV/dY_{t-1}, which might require a sign flip depending on constraint formulation.

**Modified Code (Alternative Test):**
```rust
for &lag_dual in lag_duals {
    contrib.push(-prob * lag_dual);  // Try negating
}
```

**Expected Result if Correct:**
- Lower bound should become more reasonable
- Should not violate lower bound property (LB ≤ simulation)

### Hypothesis 3: Water Value Sign Error

**Test:** Check extracted water values and their usage.

**Validation Points:**
1. LP dual signs from HiGHS
2. Hydro balance constraint formulation
3. Water value interpretation in cuts

---

## Implementation Roadmap

### Step 1: Add ψ Coefficient Accessor

To implement the chain rule fix, we need access to ψ coefficients in the state.

**Add to `StorageAndInflowState`:**

```rust
impl StorageAndInflowState {
    /// Get transformed AR coefficient ψ for a hydro's lag
    pub fn get_psi_coefficient(
        &self, 
        hydro_id: usize, 
        lag_index: usize
    ) -> f64 {
        // TODO: Store psi coefficients during initialization
        // For now, extract from temporal_model
        unimplemented!("Need to store psi coefficients")
    }
}
```

**Better Approach:** Store ψ coefficients during state initialization:

```rust
pub struct StorageAndInflowState {
    // ... existing fields ...
    
    /// Transformed AR coefficients ψ for each hydro
    /// psi_coefficients[hydro_id][lag_idx] = ψ_{lag_idx+1}
    psi_coefficients: Vec<Vec<f64>>,
}
```

Initialize in constructor using temporal models.

### Step 2: Implement Chain Rule Fix

**Modify `evaluate_cut` in `src/state.rs`:**

```rust
// Storage coefficients
contrib.extend(realization.water_value.iter().map(|&val| prob * val));

// Lag coefficients with chain rule
for hydro_id in 0..self.dimension {
    let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
    if hydro_lag_count == 0 {
        continue;
    }
    
    let lag_duals = &realization.inflow_lag_duals[hydro_id];
    let water_val = realization.water_value[hydro_id];
    
    for (lag_idx, &lag_dual) in lag_duals.iter().enumerate() {
        // Chain rule: ∂V/∂Y_{t-k} = λ^hydro · ψ_k + λ^lag_k
        let psi_k = self.psi_coefficients[hydro_id][lag_idx];
        let full_coefficient = water_val * psi_k + lag_dual;
        
        contrib.push(prob * full_coefficient);
    }
}
```

### Step 3: Add Validation Tests

```rust
#[cfg(test)]
mod cut_validation_tests {
    use super::*;
    
    #[test]
    fn test_cut_matches_objective_at_training_point() {
        // Verify: cut(x̄) ≈ E[V_{t+1}]
        // This should ALWAYS hold for valid cuts
    }
    
    #[test]
    fn test_chain_rule_vs_direct_lag_dual() {
        // Compare results with/without chain rule
        // Show that chain rule gives tighter bounds
    }
    
    #[test]
    fn test_lower_bound_property() {
        // Verify: LB ≤ simulation cost
        // Critical correctness property
    }
}
```

### Step 4: Refactor for Clarity

Once fix is validated, clean up the code:

1. Extract coefficient computation to helper function
2. Add comprehensive comments explaining chain rule
3. Reference `par_derivation.pdf` equations
4. Update documentation

---

## Expected Outcomes

### If Chain Rule Fix is Correct:

**Before Fix:**
```
Iteration | Upper Bound | Lower Bound | Simulation
----------|-------------|-------------|------------
   20     |   39,717    |   19,844    |  20,082
```
- LB > Simulation (invalid!)

**After Fix:**
```
Iteration | Upper Bound | Lower Bound | Simulation
----------|-------------|-------------|------------
   20     |   39,717    |   18,500    |  20,082
```
- LB ≤ Simulation (valid)
- Gap more realistic

### Performance Impact:

**Negligible:** Adding one multiply-add per lag coefficient.
- Complexity stays O(p) per hydro
- No memory overhead
- Fully inlined in hot path

### Convergence Improvement:

**Expected:**
- Fewer iterations to reach tolerance
- More accurate value function approximation
- Better policy decisions

---

## Fallback Plans

### If Chain Rule Fix Doesn't Resolve Issue:

1. **Check Constraint Formulation:**
   - Review LP constraint matrix
   - Verify RHS values
   - Check variable bounds

2. **Validate Dual Extraction:**
   - Inspect HiGHS dual signs
   - Compare with manual sensitivity analysis
   - Test on simple 2-stage deterministic case

3. **Numerical Issues:**
   - Check for floating-point accumulation errors
   - Verify Kahan summation usage
   - Test with higher precision

4. **Algorithmic Issues:**
   - Review risk measure adjustment
   - Check scenario probability normalization
   - Validate state update logic

---

## Documentation Updates

After fixing, update:

1. **`par_derivation.pdf`:**
   - Add section on chain rule in cut generation
   - Show explicit derivation for ∂V/∂Y_{t-k}

2. **Code Comments:**
   - Explain why chain rule is necessary
   - Reference mathematical derivation
   - Show formula clearly

3. **Example 07 README:**
   - Mention chain rule in cut generation
   - Explain why PAR requires special handling

4. **Developer Guide:**
   - Section on debugging SDDP cuts
   - Validation checklist
   - Common pitfalls

---

## Success Criteria

The fix is successful when:

1. ✅ Cut validation error < 1e-6 at training point
2. ✅ Lower bound ≤ simulation cost (always)
3. ✅ Lower bound converges monotonically upward
4. ✅ Gap between LB and UB closes over iterations
5. ✅ Final LB within ~5% of simulation cost

---

## Timeline

- **Phase 1 (Logging):** 30 minutes
- **Phase 2 (Run & Analyze):** 15 minutes
- **Phase 3 (Implement Fix):** 1 hour
- **Phase 4 (Testing & Validation):** 1 hour
- **Phase 5 (Documentation):** 30 minutes

**Total:** ~3.5 hours

---

## Next Steps

1. Add debug logging as specified in Phase 1
2. Run Example 07 and capture output
3. Analyze cut generation details
4. Implement chain rule fix
5. Validate with comprehensive tests
6. Document findings and update codebase

This systematic approach will definitively identify whether the missing chain rule term is the root cause and provide a validated fix.
