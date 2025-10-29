# TICKET-008: Simplify realize_uncertainties() with Unified Model

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 3 days (8 story points)  
**Confidence:** Medium  
**Status:** Not Started

## Context

This is the **critical ticket** that eliminates all the conditional logic that motivated this refactoring. The `realize_uncertainties()` method currently has complex branching based on state type and constraint structure. With the unified model, this becomes a clean, straight-line function with no conditionals.

This ticket rewrites `realize_uncertainties()` to use the UnifiedInflowModel for all inflow operations, removing the magic number checks and state-dependent transformations that plague the current implementation.

## Acceptance Criteria

- [ ] Given any inflow model (independent or AR), when realize_uncertainties() is called, then it follows a single code path with no conditionals on state type
- [ ] Given sampled noises, when realize_uncertainties() is called, then innovations are set correctly in AR constraint RHS
- [ ] Given lag buffer from previous realizations, when subproblem is solved, then lag values are correctly used in AR dynamics
- [ ] Given solved subproblem, when solution is extracted, then both observation and residual space values are captured
- [ ] Given timing breakdown, when realize_uncertainties() completes, then timing is accurate
- [ ] Performance: realize_uncertainties() should be 15-25% faster due to eliminated conditionals

## Tasks

### Implementation

- [ ] Rewrite `realize_uncertainties()` in `src/subproblem.rs`:
  - Remove all conditional logic on `constraints.inflow_process` structure
  - Remove residual-to-observation transformation logic (now in LP)
  - Use clean, sequential steps (see Technical Notes below)
- [ ] Implement step 1: Extract innovations from noises
  - Get inflow innovations from `noises.get_inflow_innovations()`
  - Always in residual space (no transformation needed)
- [ ] Implement step 2: Update AR constraint RHS
  - Call `self.update_ar_constraint_rhs(innovations)`
  - Sets RHS of ar_dynamics constraints to innovation values
- [ ] Implement step 3: Update load constraints
  - Extract load noises and realize them
  - Update load_balance constraint RHS (existing logic, no change)
- [ ] Implement step 4: Solve subproblem
  - Call `self.retry_solve()` (existing method, no change)
  - Track solver time in timing breakdown
- [ ] Implement step 5: Extract solution
  - Call `self.extract_solution(realization)`
  - Must extract both observation and residual space values
  - Track extraction time in timing breakdown
- [ ] Add helper method `update_ar_constraint_rhs()`:
  - Takes innovations slice
  - Updates model RHS for each ar_dynamics constraint
  - Sets RHS to innovation value for each hydro
- [ ] Update `extract_solution()` to populate residual fields:
  - Extract `inflow_residual` from `vars.inflow_residual`
  - Extract `lag_duals` from ar_dynamics constraint duals (if lags present)

### Testing

- [ ] Unit test: realize_uncertainties with independent noise (AR(0))
- [ ] Unit test: realize_uncertainties with AR(1) noise
- [ ] Unit test: realize_uncertainties with AR(2) noise
- [ ] Unit test: realize_uncertainties with mixed AR orders
- [ ] Unit test: Verify innovations correctly set in AR constraint RHS
- [ ] Unit test: Verify solution extraction populates observation space (inflows)
- [ ] Unit test: Verify solution extraction populates residual space (inflow_residual)
- [ ] Unit test: Verify lag_duals extracted for AR hydros, empty for independent
- [ ] Integration test: Full forward pass with lag buffer updates
- [ ] Regression test: Compare output with old implementation on simple example
- [ ] Performance test: Benchmark realize_uncertainties (expect 15-25% speedup)

### Documentation

- [ ] Add comprehensive doc comment to realize_uncertainties() explaining:
  - Clean sequential flow
  - No conditional logic
  - Timing breakdown
- [ ] Add doc comment to update_ar_constraint_rhs() explaining RHS update
- [ ] Update module-level docs with simplified flow diagram
- [ ] Add inline comments for each major step
- [ ] Update CHANGELOG.md with "Changed: realize_uncertainties simplified with unified AR model"

## Technical Notes

### New Implementation Flow

```rust
impl Subproblem {
    pub fn realize_uncertainties(
        &mut self,
        noises: &OptimizedSampledBranchingNoises,
        realization: &mut Realization,
    ) -> Result<RealizeUncertaintiesTiming, String> {
        let solver_start = Instant::now();

        // STEP 1: Get innovations (always in residual space)
        let innovations = noises.get_inflow_innovations();

        // STEP 2: Update AR constraint RHS with innovations
        self.update_ar_constraint_rhs(innovations)?;

        // STEP 3: Update load uncertainties
        let loads = self.load_process.realize(noises.get_load_innovations());
        self.set_load_balance_rhs(loads)?;

        // STEP 4: Solve subproblem
        self.retry_solve()?;

        let solver_time = solver_start.elapsed();
        let extraction_start = Instant::now();

        // STEP 5: Extract solution (observation + residual spaces)
        self.extract_solution(realization)?;

        let state_extraction_time = extraction_start.elapsed();

        Ok(RealizeUncertaintiesTiming {
            solver_time,
            state_extraction_time,
        })
    }

    fn update_ar_constraint_rhs(
        &mut self,
        innovations: &[f64],
    ) -> Result<(), String> {
        let model = self.model.as_mut()
            .ok_or("Model not initialized")?;

        for hydro in 0..self.inflow_model.dimension() {
            let constraint_idx = self.constraints.ar_dynamics[hydro];
            let innovation = innovations[hydro];

            // Set RHS: Z'_t - ΣφZ'_{t-k} = ε_t
            model.change_rhs(constraint_idx, innovation, innovation);
        }

        Ok(())
    }
}
```

### Key Improvements Over Current Implementation

**Before (Complex):**

```rust
let inflow_observations = if !noises.get_inflow_residuals().is_empty()
    && !self.constraints.inflow_process.is_empty()
    && self.constraints.inflow_process[0].len() <= 2  // MAGIC NUMBER!
{
    // StorageState path: transform residuals to observations
    let mut observations = vec![0.0; noises.get_inflow_residuals().len()];
    for hydro in 0..observations.len() {
        let residual = noises.get_inflow_residuals()[hydro];
        let (mu, sigma) = /* lookup seasonal params */;
        observations[hydro] = mu + sigma * residual;
    }
    observations
} else {
    // StorageAndInflowState path: use innovations directly
    inflow_noises.to_vec()
};
// Then set constraint RHS with observations...
```

**After (Simple):**

```rust
// Innovations are always in residual space
let innovations = noises.get_inflow_innovations();

// Set AR constraint RHS directly (transformation in LP)
self.update_ar_constraint_rhs(innovations)?;
```

**Benefits:**

- ✅ Zero conditionals on state type
- ✅ No magic number checks
- ✅ No manual space transformations
- ✅ LP formulation handles transformation
- ✅ Single code path for all cases

### Solution Extraction Changes

**Old extraction** (observation space only):

```rust
for h in 0..n_hydros {
    realization.inflows[h] = solution.colvalue[vars.inflow[h]];
}
```

**New extraction** (dual space):

```rust
for h in 0..n_hydros {
    realization.inflows[h] = solution.colvalue[vars.inflow[h]];
    realization.inflow_residual[h] = solution.colvalue[vars.inflow_residual[h]];

    // Extract lag duals if present
    let lag_order = self.inflow_model.lag_order(h);
    if lag_order > 0 {
        let constraint_idx = self.constraints.ar_dynamics[h];
        // Extract dual value from AR constraint
        realization.lag_duals[h] = vec![solution.rowdual[constraint_idx]];
        // Note: For future lag transfer constraints, extract multiple duals
    }
}
```

### Edge Cases

- **First stage**: Lag buffer may not be initialized yet (use initial conditions)
- **Independent hydros**: innovations applied to AR(0) constraint (Z' = ε)
- **Solver failure**: retry_solve() handles this, propagate error
- **Missing variables**: Defensive checks for vars indices
- **NaN/Inf in solution**: Validate extracted values

### Performance Analysis

**Eliminated Operations:**

- Conditional branch on state type: ~5-10ns per call
- Conditional branch on constraint count: ~5-10ns per call
- Manual space transformation loop: O(n) with HashMap lookups
- Total per realize: ~50-100ns + O(n) work

**Expected Speedup:**

- Micro-benchmark: 20-30% faster
- Full algorithm: 15-25% faster (amortized over all operations)

### Backward Compatibility

This change **breaks backward compatibility** with:

- Code that directly inspects `constraints.inflow_process` structure
- State implementations that override inflow-related methods

**Mitigation:**

- This is internal refactoring, no public API change
- Examples will be updated in same PR
- Tests will be updated to match new behavior

## Dependencies

- **Blocked by**:
  - TICKET-001 (needs UnifiedInflowModel)
  - TICKET-003 (needs lag buffer methods)
  - TICKET-004 (needs updated Variables)
  - TICKET-005 (needs updated Constraints)
  - TICKET-006 (needs updated Realization)
  - TICKET-007 (needs Subproblem integration)
- **Blocks**: TICKET-010 (update_from_trajectory depends on this)
- **Related**: TICKET-011 (State trait cleanup)

## References

- Current `src/subproblem.rs` lines 600-750 - realize_uncertainties implementation
- UNIFIED_AR_ROADMAP.md - Section 1.2 (Simplified Subproblem Interface)
- Architecture review showing 15-25% speedup potential

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Example 06 runs and produces correct output
- [ ] Example 07 runs and produces correct output
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Performance benchmark shows 15-25% speedup
- [ ] No cyclomatic complexity warnings
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
