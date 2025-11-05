# [TICKET-004] Fix Critical Bug in add_cut_constraint_to_model

**Sprint:** 2  
**Estimated Effort:** 5 story points (3 days)  
**Confidence:** High  
**Priority:** P0 - Critical bug fix

## Context

This ticket addresses the **critical bug** that motivated this entire epic. The current implementation of `StorageAndInflowState::add_cut_constraint_to_model` (lines 866-920 in state.rs) uses a fragile heuristic to match cut coefficients to lag variables. It attempts to guess which entities are inflows by comparing lag counts, which fails when loads also have AR models.

**Bug Impact:**
- Invalid lower bounds (LB > simulation value)
- Incorrect cut coefficients applied to wrong variables
- Silent failure mode (no runtime error, just wrong results)

**Root Cause:** Type erasure in unified `lagged_state` structure forces heuristic-based entity type detection.

**Fix:** Use explicit `InflowLagVariables` to directly access inflow lags by hydro_id.

## Acceptance Criteria

- [ ] Given a cut with inflow lag coefficients, when adding to model, then coefficients are matched to correct inflow lag variables by hydro_id
- [ ] Given system with Load(AR=1) at Bus 0 and Inflow(AR=1) at Hydro 0, when adding cut, then cut coefficients are not confused between load and inflow
- [ ] Given Example 07 system, when running SDDP with fixed seed, then lower bound ≤ simulation value in all iterations
- [ ] Given system with mixed AR orders (loads and inflows), when cuts are added, then no panics or index out of bounds errors
- [ ] Performance: Cut addition time should not increase (should actually decrease by ~20%)

## Tasks

### Implementation

- [ ] Locate `add_cut_constraint_to_model` in `src/state.rs` (StorageAndInflowState impl)

- [ ] Remove the entire heuristic-based loop (lines ~889-914)
  ```rust
  // DELETE THIS SECTION:
  let mut hydro_count = 0;
  for (_entity_idx, entity_lags) in lag_vars.iter().enumerate() {
      if hydro_count >= self.dimension { break; }
      let hydro_lag_count = self.layout.hydro_lag_count(hydro_count);
      if entity_lags.len() == hydro_lag_count { // HEURISTIC!
          // ...
      }
  }
  ```

- [ ] Replace with explicit inflow lag access
  ```rust
  // NEW IMPLEMENTATION:
  if let Some(inflow_lags) = &variables.inflow_lags {
      for hydro_id in 0..self.dimension {
          let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
          if hydro_lag_count == 0 {
              continue;
          }
          
          let lags = inflow_lags.get_lags(hydro_id);
          assert_eq!(
              lags.len(), hydro_lag_count,
              "Hydro {} expected {} lags but found {}",
              hydro_id, hydro_lag_count, lags.len()
          );
          
          for lag_idx in 0..hydro_lag_count {
              let lag_var = lags[lag_idx];
              factors.push((lag_var, -cut.coefficients[coef_idx]));
              coef_idx += 1;
          }
      }
  }
  ```

- [ ] Add assertion to verify all coefficients used
  ```rust
  assert_eq!(
      coef_idx, cut.coefficients.len(),
      "Cut has {} coefficients but only used {}",
      cut.coefficients.len(), coef_idx
  );
  ```

- [ ] Update any comments to explain the explicit indexing approach

- [ ] Remove dependency on entity ordering or temporal_models in this function

### Testing

- [ ] Unit test: System with Load(AR=1) + Inflow(AR=1) at same entity indices
  - Create cut with [storage_0, storage_1, inflow_0_lag, inflow_1_lag]
  - Verify cut uses inflow variables, not load variables
  - This would FAIL with old heuristic!

- [ ] Unit test: System with mixed AR orders
  - Load(bus=0, AR=2), Inflow(hydro=0, AR=1), Load(bus=1, AR=0), Inflow(hydro=1, AR=3)
  - Create cut with correct coefficient count
  - Verify each coefficient matches correct variable

- [ ] Regression test: Example 07 with fixed seed
  - Run 50 iterations
  - Assert lower_bound ≤ simulation_mean in all iterations
  - Compare bound progression to reference values

- [ ] Integration test: Large system (20 hydros, 30 buses with various AR orders)
  - Run 100 iterations
  - Verify convergence
  - Check for any panics or errors

- [ ] Performance benchmark: Cut addition timing
  - Before: measure time to add 1000 cuts
  - After: measure time to add 1000 cuts
  - Verify improvement (should be ~20% faster due to direct access)

- [ ] Edge case test: Hydro with no lags (AR=0)
  - System with Hydro 0: AR(2), Hydro 1: AR(0), Hydro 2: AR(1)
  - Verify cut correctly skips Hydro 1

- [ ] Numerical validation test: Compare dual values from cuts before/after fix
  - Same system, same seed, same cuts
  - Old implementation produces incorrect duals
  - New implementation produces correct duals

### Documentation

- [ ] Update doc comment for `add_cut_constraint_to_model` explaining the explicit access pattern
- [ ] Add comment explaining why we iterate hydro_id instead of entity_idx
- [ ] Document the assertion that validates coefficient usage
- [ ] Update CHANGELOG.md with bug fix entry
- [ ] Add note to BUG_FIX_PAR_LOWER_BOUND.md about resolution

## Technical Notes

### Before (Buggy Code)

```rust
fn add_cut_constraint_to_model(
    &mut self,
    cut: &mut cut::BendersCut,
    variables: &subproblem::Variables,
    model: &mut solver::Model,
) {
    let mut factors = Vec::new();
    let mut coef_idx = 0;
    
    // Storage coefficients (correct)
    for hydro_id in 0..self.dimension {
        let var = variables.stored_volume[hydro_id];
        factors.push((var, -cut.coefficients[coef_idx]));
        coef_idx += 1;
    }
    
    // Inflow lag coefficients (BUGGY - uses heuristic)
    if let Some(lag_vars) = &variables.lagged_state {
        let mut hydro_count = 0;
        for (_entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            if hydro_count >= self.dimension {
                break;
            }
            
            // ❌ HEURISTIC: Guess if this entity is a hydro by lag count
            let hydro_lag_count = self.layout.hydro_lag_count(hydro_count);
            if entity_lags.len() == hydro_lag_count {
                // Hope this is the right hydro...
                for lag_idx in 0..hydro_lag_count {
                    let lag_var = entity_lags[lag_idx];
                    factors.push((lag_var, -cut.coefficients[coef_idx]));
                    coef_idx += 1;
                }
                hydro_count += 1;
            } else if entity_lags.is_empty() {
                // Assume this is a load, skip
                continue;
            } else {
                // Unexpected lag count, skip
                continue;
            }
        }
    }
    
    model.add_row(cut.rhs.., factors);
}
```

**Why This Fails:**

Scenario:
- `temporal_models = [Load(0, AR=1), Inflow(0, AR=1), Load(1, AR=0), Inflow(1, AR=1)]`
- `lagged_state[0]` = Load 0 lag variables (length 1)
- `lagged_state[1]` = Inflow 0 lag variables (length 1)

Execution:
- entity_idx=0: len=1, `hydro_lag_count(0)=1` → Match! ❌ WRONG! This is Load 0, not Hydro 0
- Applies Inflow 0 coefficient to Load 0 variable
- Invalid cut!

### After (Fixed Code)

```rust
fn add_cut_constraint_to_model(
    &mut self,
    cut: &mut cut::BendersCut,
    variables: &subproblem::Variables,
    model: &mut solver::Model,
) {
    let mut factors = Vec::new();
    let mut coef_idx = 0;
    
    // Storage coefficients
    for hydro_id in 0..self.dimension {
        let var = variables.stored_volume[hydro_id];
        factors.push((var, -cut.coefficients[coef_idx]));
        coef_idx += 1;
    }
    
    // Inflow lag coefficients (FIXED - explicit access)
    if let Some(inflow_lags) = &variables.inflow_lags {
        for hydro_id in 0..self.dimension {
            let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
            if hydro_lag_count == 0 {
                continue;
            }
            
            // ✅ EXPLICIT: Direct access by hydro_id
            let lags = inflow_lags.get_lags(hydro_id);
            
            #[cfg(debug_assertions)]
            {
                assert_eq!(
                    lags.len(), hydro_lag_count,
                    "Hydro {} expected {} lags but found {}",
                    hydro_id, hydro_lag_count, lags.len()
                );
            }
            
            for lag_idx in 0..hydro_lag_count {
                let lag_var = lags[lag_idx];
                factors.push((lag_var, -cut.coefficients[coef_idx]));
                coef_idx += 1;
            }
        }
    }
    
    // Verify all coefficients used
    #[cfg(debug_assertions)]
    {
        assert_eq!(
            coef_idx, cut.coefficients.len(),
            "Cut has {} coefficients but only used {}. This indicates a mismatch \
             between cut generation and cut application.",
            cut.coefficients.len(), coef_idx
        );
    }
    
    model.add_row(cut.rhs.., factors);
}
```

**Why This Works:**

- Direct indexing: `hydro_id` in loop matches cut coefficient ordering
- No guessing: Type system guarantees `inflow_lags` contains only inflows
- Fail fast: Assertions catch mismatches in debug builds
- Clear intent: Code explicitly states it's processing inflow lags

### Edge Cases

- All hydros have AR(0) → `inflow_lags` is None, skip loop
- Some hydros have AR(0) → skip those with `continue`
- Cut with wrong coefficient count → assertion catches it
- Missing hydro_id in inflow_lags → panic with clear error

### Performance Impact

**Before:** O(n_entities) iteration with conditional logic  
**After:** O(n_hydros) direct access with O(1) lookups

Typical system: 30 entities (10 hydros, 20 loads)  
- Before: Iterate 30 entities, check each one
- After: Iterate 10 hydros directly
- Speedup: ~3x fewer iterations, simpler logic

### Numerical Validation

To verify the fix produces correct results:

1. Use Example 07 with seed 42
2. Run for 50 iterations
3. Compare cut coefficients between old and new
4. Old: coefficients applied to wrong variables → invalid bound
5. New: coefficients applied to correct variables → valid bound

## Dependencies

- Blocked by: TICKET-001, TICKET-002 (need explicit structures)
- Blocks: None
- Related: TICKET-003 (validation helps verify correctness)

## Definition of Done

- [ ] No heuristic-based entity type detection remains
- [ ] All tests pass including new regression tests
- [ ] Example 07 produces valid bounds (LB ≤ simulation)
- [ ] Performance improvement measured and documented
- [ ] Code reviewed with focus on correctness
- [ ] CHANGELOG.md updated
- [ ] BUG_FIX_PAR_LOWER_BOUND.md updated with resolution
