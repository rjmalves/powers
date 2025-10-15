# PAR-V2-007: Add AR Dynamics as LP Constraints

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 2 (LP Integration - CRITICAL WEEK)  
**Story Points**: 5  
**Priority**: 🔥🔥🔥 MOST CRITICAL TICKET IN ENTIRE EPIC  
**Status**: 🔵 Not Started

---

## ⚠️ CRITICAL NOTICE

**This is the single most important ticket in the entire PAR completion epic.**

This is where autoregressive dynamics actually enter the mathematical formulation of the SDDP subproblems. Everything before this ticket builds toward this moment. Everything after depends on getting this right.

**DO NOT PROCEED** without:
1. ✅ Thorough code review by architect
2. ✅ Hand-calculated validation of constraint structure
3. ✅ Complete understanding of the AR equation formulation

**If this ticket is implemented incorrectly**, the entire PAR implementation will produce wrong results even if everything else is perfect.

---

## Context

Currently, subproblems contain:
- Power system constraints (load balance, hydro balance)
- Storage state variables and constraints
- Inflow variables set by scenario realizations

For PAR models, we need to add:
- **AR dynamics constraints** that link inflow to lagged states
- AR coefficients (φ_k) as **constraint coefficients** in the LP matrix
- Innovation (ε_t) as **RHS values** that vary by scenario

**The Critical Insight**: AR dynamics are not external scenario generation - they are **constraints in the optimization problem**. This is what makes the state-space formulation correct for SDDP.

**Mathematical Foundation**:

For a PAR(p) model with seasonal parameters, the AR dynamics constraint is:

```
inflow_t - σ_m · (φ_1m · lag_{t-1} + φ_2m · lag_{t-2} + ... + φ_pm · lag_{t-p}) = μ_m + ε_t

where:
  inflow_t = inflow variable (existing)
  lag_{t-k} = lag state variables (from PAR-V2-006)
  φ_km = AR coefficient k for season m (from ar_params)
  σ_m = seasonal standard deviation (from ar_params)
  μ_m = seasonal mean (from ar_params)
  ε_t = innovation (scenario-specific RHS, set at runtime)
```

**Why This Structure**:
- Coefficients (φ_k) in constraint matrix → LP solver generates correct dual variables
- Innovation (ε_t) in RHS → varies by scenario during forward/backward passes
- Dual variables from this constraint → enter Benders cuts with proper coefficients

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.3
- Theory: Plan → "The State-Space PAR Formulation" section
- Existing pattern: `src/state.rs` → `StorageState::add_constraints_to_subproblem`

---

## Acceptance Criteria

### Functional Requirements

- [ ] AR dynamics constraints added for all PAR hydros
- [ ] AR coefficients (φ_k) appear in constraint matrix as coefficients
- [ ] Constraint structure: `inflow - σ(φ_1·lag[0] + φ_2·lag[1] + ... + φ_p·lag[p-1]) = μ + ε`
- [ ] One AR constraint per PAR hydro per subproblem
- [ ] Constraints added via `State::add_constraints_to_subproblem` method
- [ ] Non-PAR hydros unaffected (existing behavior preserved)

### Technical Requirements

- [ ] Constraint indices stored in `Constraints` struct for RHS updates
- [ ] Season-specific parameters (φ, μ, σ) looked up correctly
- [ ] RHS initialized to μ (innovations added later in PAR-V2-020)
- [ ] Constraint bounds set correctly (equality constraint)
- [ ] No memory leaks in constraint generation

### Validation Requirements

- [ ] Hand-calculated test case validates constraint structure
- [ ] Constraint matrix inspected and verified correct
- [ ] LP solves successfully with AR constraints
- [ ] Dual variables extracted successfully from AR constraints
- [ ] No regressions in existing (non-PAR) functionality

---

## Tasks

### Implementation

- [ ] **Step 1**: Extend `Constraints` struct in `src/subproblem.rs`
  ```rust
  pub struct Constraints {
      pub load_balance: Vec<usize>,
      pub hydro_balance: Vec<usize>,
      pub inflow_process: Vec<Vec<usize>>,
      // NEW: AR dynamics constraints
      pub ar_dynamics: HashMap<usize, usize>,  // hydro_id → constraint_idx
  }
  ```

- [ ] **Step 2**: Implement constraint generation in `StorageAndInflowState::add_constraints_to_subproblem`
  ```rust
  // For each PAR hydro:
  // 1. Get seasonal parameters (φ, μ, σ)
  // 2. Build constraint: inflow - σ·(Σ φ_k·lag[k]) = μ
  // 3. Store constraint index for later RHS updates
  ```

- [ ] **Step 3**: Handle season_id lookup
  - Extract from node context or pass as parameter
  - Map to correct seasonal parameters

- [ ] **Step 4**: Build constraint coefficients
  - inflow variable: coefficient = 1.0
  - lag[k] variable: coefficient = -σ * φ_k (for k = 0 to p-1)

- [ ] **Step 5**: Set RHS bounds
  - Lower bound = μ_m (deterministic part)
  - Upper bound = μ_m (equality constraint)
  - Innovation ε_t added later (PAR-V2-020)

- [ ] **Step 6**: Store constraint indices
  - Map hydro_id → constraint_idx in `ar_dynamics` field
  - Used later for RHS updates with scenario realizations

### Testing

- [ ] **Unit Test**: Constraint structure validation
  ```rust
  #[test]
  fn test_ar_constraint_structure() {
      // Create PAR(2) hydro with known parameters
      // φ_1 = 0.6, φ_2 = 0.3, μ = 100, σ = 20
      // Build subproblem, extract constraint matrix
      // Verify coefficients match theory
  }
  ```

- [ ] **Unit Test**: Constraint count validation
  ```rust
  #[test]
  fn test_ar_constraint_count() {
      // System with 3 hydros: 2 PAR, 1 naive
      // Verify exactly 2 AR constraints added
  }
  ```

- [ ] **Integration Test**: LP solve with AR constraints
  ```rust
  #[test]
  fn test_lp_solve_with_ar_constraints() {
      // Build subproblem with AR constraints
      // Set feasible RHS values
      // Verify LP solves successfully
      // Check solution is feasible
  }
  ```

- [ ] **Hand-Calculation Test**: Verify constraint coefficients
  ```rust
  #[test]
  fn test_ar_coefficients_match_theory() {
      // PAR(1) with φ=0.7, σ=10
      // Inflow coef should be 1.0
      // Lag coef should be -7.0 (=-σ*φ)
      // Verify by extracting from LP matrix
  }
  ```

- [ ] **Regression Test**: Non-PAR hydros unaffected
  ```rust
  #[test]
  fn test_naive_hydros_unchanged() {
      // Create system with only naive hydros
      // Verify no AR constraints added
      // Verify behavior identical to before
  }
  ```

### Documentation

- [ ] Add doc comments to `add_constraints_to_subproblem` explaining AR constraint structure
- [ ] Document constraint coefficient formulation (why -σ*φ_k)
- [ ] Document RHS initialization (why μ, where ε_t comes later)
- [ ] Add inline comments for each step in constraint generation
- [ ] Update module-level docs in `state.rs` to explain PAR constraint approach

---

## Technical Notes

### Constraint Formulation Deep Dive

**Standard Form**: `a^T x = b`

**Our AR Constraint**:
```
inflow - σ·(φ_1·lag[0] + φ_2·lag[1] + ... + φ_p·lag[p-1]) = μ + ε
```

**In Matrix Form**:
```
[ 1.0, -σ·φ_1, -σ·φ_2, ..., -σ·φ_p ] · [ inflow, lag[0], lag[1], ..., lag[p-1] ]^T = μ + ε
```

**Why the negative signs**: We're subtracting the AR term from inflow, so lag coefficients are negative.

**Why scale by σ**: The AR coefficients φ operate on standardized residuals. Scaling brings them to the scale of actual inflows.

### Season-to-Parameter Mapping

Challenge: Need to know which season we're in to pick parameters.

**Solution**: Get season_id from node context:
```rust
// In add_constraints_to_subproblem, we'll need to pass season_id
// This comes from the graph node's NodeData
// For now, we can use a placeholder or require it as a parameter
```

**TODO**: Decide on season_id passing mechanism:
- Option A: Add season_id to State trait methods
- Option B: Store season_id in StorageAndInflowState
- Option C: Pass as context parameter

### RHS Update Strategy

**Initial RHS**: Set to μ_m (deterministic part)

**At Runtime** (in PAR-V2-020):
```rust
// For each scenario realization:
// 1. Sample innovation ε_t
// 2. Update RHS to μ_m + ε_t
// 3. Solve subproblem
```

**Why Split**: Constraint structure (coefficients) is fixed, RHS varies by scenario.

### Dual Variable Preview

When this constraint is added, the LP solver will produce a dual variable (π_AR) that represents the "shadow price" of the AR relationship. This dual will:
1. Appear in the cut generation (PAR-V2-010)
2. Multiply the lag state values in the cut
3. Affect the optimal policy

**This is why correctness is critical**: Wrong coefficients → wrong duals → wrong cuts → wrong policy.

### Performance Considerations

**Constraint Count**: +1 per PAR hydro (minimal overhead)

**Coefficient Count**: +p per PAR(p) hydro (linear in AR order)

**Matrix Sparsity**: Highly sparse (only p+1 non-zeros per row)

**Expected Impact**: <5% increase in LP size for typical PAR(1-3) models

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: StorageAndInflowState struct exists
- ✅ PAR-V2-005: AR parameters extractable from config
- ✅ PAR-V2-006: Lag state variables added to subproblem

### Blocks

- PAR-V2-009: Validate AR constraint structure (depends on constraints existing)
- PAR-V2-010: Extract duals from lag states (depends on constraints existing)
- PAR-V2-020: Innovation-based RHS updates (depends on constraint indices)

### Related

- PAR-V2-008: Update Variables/Constraints structs (parallel work)

---

## Implementation Hints

### Step-by-Step Implementation

**1. Extend Constraints Struct First**:
```rust
// In src/subproblem.rs
pub struct Constraints {
    pub load_balance: Vec<usize>,
    pub hydro_balance: Vec<usize>,
    pub inflow_process: Vec<Vec<usize>>,
    pub ar_dynamics: HashMap<usize, usize>,  // NEW
}
```

**2. Implement in StorageAndInflowState**:
```rust
impl State for StorageAndInflowState {
    fn add_constraints_to_subproblem(
        &self,
        pb: &mut solver::Problem,
        variables: &subproblem::Variables,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_process: &dyn stochastic_process::StochasticProcess,
    ) -> Vec<Vec<usize>> {
        // 1. Delegate storage constraints to base
        let mut storage_constraints = self.storage.add_constraints_to_subproblem(
            pb, variables, _load_stochastic_process, _inflow_stochastic_process
        );
        
        // 2. Add AR dynamics constraints for PAR hydros
        let mut ar_constraint_indices = HashMap::new();
        
        for (hydro_id, p) in self.ar_orders.iter() {
            // Get seasonal parameters
            let season_id = self.get_current_season(); // TODO: Implement
            let params = self.ar_params.get(&(*hydro_id, season_id))
                .expect("AR params must exist for all PAR hydros");
            
            let phi = &params[0..*p];
            let mu = params[*p];
            let sigma = params[*p + 1];
            
            // Build constraint factors
            let inflow_var = variables.inflow[*hydro_id];
            let lag_vars = &variables.lag_inflow[*hydro_id];
            
            let mut factors = vec![(inflow_var, 1.0)];
            for (k, &lag_var) in lag_vars.iter().enumerate() {
                factors.push((lag_var, -sigma * phi[k]));
            }
            
            // Add constraint: inflow - σ·(Σ φ_k·lag[k]) = μ
            let constraint_idx = pb.add_row(mu..mu, factors);
            ar_constraint_indices.insert(*hydro_id, constraint_idx);
        }
        
        // 3. Store constraint indices for RHS updates
        // TODO: Need to return this somehow - might need to modify return type
        
        storage_constraints
    }
}
```

**3. Test Incrementally**:
```bash
# After implementing, test compile
cargo build

# Run specific test
cargo test test_ar_constraint_structure

# Check LP matrix structure (add debug printing)
```

### Debugging Aids

**Inspect Constraint Matrix**:
```rust
// Add to test
fn print_constraint_matrix(model: &solver::Model) {
    for row in 0..model.num_rows() {
        println!("Row {}: {:?}", row, model.get_row(row));
    }
}
```

**Validate Coefficients**:
```rust
#[test]
fn test_ar_coefficients_precise() {
    // Known PAR(2): φ_1=0.6, φ_2=0.3, σ=10
    // Build subproblem
    // Extract constraint
    // Verify: coef(lag[0]) = -6.0, coef(lag[1]) = -3.0
    assert!((coef_lag0 - (-6.0)).abs() < 1e-10);
    assert!((coef_lag1 - (-3.0)).abs() < 1e-10);
}
```

---

## Estimated Effort

**5 story points** (2-3 days)

**Confidence**: Medium (complexity + criticality)

**Breakdown**:
- Design & planning: 4 hours
- Implementation: 8 hours
- Testing: 6 hours
- Hand validation: 4 hours
- Code review: 4 hours
- Rework: 4 hours (buffer for issues)

---

## Definition of Done

- [x] AR dynamics constraints implemented in `add_constraints_to_subproblem`
- [x] Constraint coefficients match mathematical formulation
- [x] All unit tests pass
- [x] Hand-calculated validation test passes
- [x] LP solves successfully with AR constraints
- [x] Code reviewed by architect and approved
- [x] No regressions in existing tests
- [x] Documentation complete
- [x] Merged to branch

---

## Risk Assessment

**Risk Level**: 🔴 **EXTREME**

**Impact of Failure**: Complete PAR implementation failure

**Mitigation Strategy**:
1. Hand-calculated test cases with simple parameters
2. Multiple levels of validation (unit, integration, numerical)
3. Architect review before merge
4. Incremental testing at each step
5. Comparison with theoretical AR equation

**Signs of Problems**:
- LP becomes infeasible
- Dual variables are zero or nonsensical
- Solution values don't satisfy AR equation
- Numerical instability in solver

**If Problems Occur**:
1. ⛔ STOP - do not proceed to next tickets
2. 🔍 Debug constraint structure with print statements
3. 📊 Validate coefficients against hand calculations
4. 🧪 Test with simplest possible case (PAR(1), integer parameters)
5. 👥 Escalate to architect immediately

---

## Notes

- Take your time with this ticket - rushing leads to subtle bugs
- Validate each step before proceeding to the next
- Use simple test cases first (PAR(1), round numbers)
- Print constraint matrix to verify structure
- Cross-check with the mathematical formulation repeatedly

**Remember**: Everything depends on this being correct. Better to take an extra day to validate thoroughly than to discover issues weeks later when debugging cuts.

**Next Critical Ticket**: PAR-V2-010 (Extract duals from lag states) - depends on these constraints existing and being correct.
