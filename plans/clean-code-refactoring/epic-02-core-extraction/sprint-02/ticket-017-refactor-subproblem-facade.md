# [T-017] Refactor Subproblem.rs to Use New Modules (Facade Pattern)

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-013](./ticket-013-hydro-balance-constraints.md), [T-014](./ticket-014-bus-balance-constraints.md), [T-015](./ticket-015-ar-dynamics-constraints.md), [T-016](./ticket-016-bound-constraints.md)
> **Blocks**: Epic 3

---

## ⚠️ CRITICAL: Integration Point

This ticket integrates all extracted modules back into `subproblem.rs` using the **facade pattern**. The goal is to replace inline logic with calls to the new modules while maintaining **identical behavior**.

This is the highest-risk ticket in the epic. Run golden tests **after every change**.

---

## Files to Read Before Starting

- `src/subproblem.rs` - The file being refactored
- `src/model/solution_extract.rs` - SolutionExtractor from Sprint 1
- `src/model/constraints/` - All constraint builders from Sprint 2
- Golden test results to ensure baseline is passing

---

## Context

### Background

The facade pattern keeps `subproblem.rs` as the entry point while delegating to extracted modules. This:
- Minimizes changes to callers of `Subproblem`
- Makes the refactoring reversible if issues arise
- Enables gradual migration

### Current State

`subproblem.rs` contains:
- Solution extraction functions (can now delegate to `SolutionExtractor`)
- Constraint building (can now delegate to constraint builders)

### Target State

```rust
impl Subproblem {
    // Solution extraction becomes delegation
    fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
        self.solution_extractor.extract_deficit(solution, realization);
    }
    
    // Or even simpler - replace multiple calls with one
    fn extract_all_from_solution(&self, solution: &Solution, realization: &mut Realization) {
        self.solution_extractor.extract_all_primals(solution, realization);
        self.solution_extractor.extract_all_duals(solution, realization);
    }
}
```

---

## Specification

### Phase 1: Add SolutionExtractor to Subproblem

1. **Add field to Subproblem struct**:
   ```rust
   pub struct Subproblem {
       // ... existing fields ...
       solution_extractor: crate::model::SolutionExtractor,
   }
   ```

2. **Initialize in constructor**:
   ```rust
   // In Subproblem::new or similar
   let solution_extractor = SolutionExtractor::from_subproblem_types(
       &variables,
       &constraints,
   );
   ```

3. **Update Clone impl** (if manual):
   - Ensure `solution_extractor` is cloned

### Phase 2: Replace Solution Extraction Calls

For each `get_*_from_solution` method:

**Option A**: Delegate to extractor (preserves method signature)
```rust
fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    self.solution_extractor.extract_deficit(solution, realization);
}
```

**Option B**: Deprecate and use extractor directly at call sites

Recommendation: Use **Option A** first for safety, refactor call sites in Epic 3.

### Phase 3: Replace Constraint Building (If Ready)

If constraint builders are complete and tested:

```rust
fn add_constraints(
    pb: &mut solver::Problem,
    variables: &Variables,
    system: &system::System,
    _state: &dyn state::State,
    temporal_models: &[temporal_model::TemporalModel],
    season_id: usize,
) -> Constraints {
    let mut ctx = ConstraintContext {
        problem: pb,
        variables,
        system,
        temporal_models,
        season_id,
    };
    
    let load_balance = BusBalanceBuilder::build(&mut ctx);
    let hydro_balance = HydroBalanceBuilder::build(&mut ctx);
    let uncertainty_observation = ArDynamicsBuilder::build_observation(&mut ctx);
    let (load_lag_constraints, inflow_lag_constraints) = ArDynamicsBuilder::build_lag_fixing(&mut ctx);
    
    Constraints {
        load_balance,
        hydro_balance,
        uncertainty_observation,
        load_lag_constraints,
        inflow_lag_constraints,
    }
}
```

### Phase 4: Clean Up (Optional)

- Remove inline logic that's now in modules
- Keep method signatures for backward compatibility
- Add deprecation notices if methods will be removed later

---

## Acceptance Criteria

### Minimum (Phase 1-2):
- [ ] `SolutionExtractor` field added to `Subproblem`
- [ ] All `get_*_from_solution` methods delegate to extractor
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Full (Phase 1-4):
- [ ] Constraint building uses new builders
- [ ] `add_constraints` is simplified
- [ ] Inline logic removed from subproblem.rs
- [ ] Code reduction measured

### Correctness Verification

- [ ] Golden tests pass after EVERY phase
- [ ] No numerical differences
- [ ] No performance regression

---

## Implementation Guide

### Suggested Approach

1. **Ensure baseline passes**:
   ```bash
   cargo test
   ./scripts/golden-tests.sh verify
   ```

2. **Phase 1: Add SolutionExtractor field**:
   ```rust
   // In Subproblem struct definition
   solution_extractor: crate::model::SolutionExtractor,
   
   // In constructor
   let solution_extractor = crate::model::SolutionExtractor::from_subproblem_types(
       &variables,
       &constraints,
   );
   ```

3. **Update struct initialization** wherever Subproblem is created

4. **Test after Phase 1**:
   ```bash
   cargo build
   cargo test
   ./scripts/golden-tests.sh verify
   ```

5. **Phase 2: Replace ONE extraction method at a time**:
   ```rust
   // Before
   fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
       let first = *self.variables.deficit.first().unwrap();
       let last = *self.variables.deficit.last().unwrap() + 1;
       realization.deficit.clone_from_slice(&solution.colvalue[first..last]);
   }
   
   // After
   fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
       self.solution_extractor.extract_deficit(solution, realization);
   }
   ```

6. **Test after EACH method replacement**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

7. **Phase 3: Replace add_constraints** (if builders ready)

8. **Final verification**:
   ```bash
   cargo test
   cargo bench  # Check for regression
   ./scripts/golden-tests.sh verify
   ```

### Method Replacement Order

Replace in this order (simplest first):
1. `get_deficit_from_solution`
2. `get_spillage_from_solution`
3. `get_turbined_flow_from_solution`
4. `get_final_storage_from_solution`
5. `get_water_values_from_solution`
6. `get_marginal_cost_from_solution`
7. `get_thermal_gen_from_solution` (optional handling)
8. `get_net_exchange_from_solution` (optional handling)
9. `get_load_from_solution` (non-contiguous)
10. `get_inflow_from_solution` (non-contiguous)
11. `get_lag_duals_from_solution` (complex)

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/subproblem.rs` | Add field, delegate methods |

### Patterns to Follow

- Delegate to extractor, don't duplicate logic
- Keep method signatures unchanged
- Test after every change

### Pitfalls to Avoid

- ⚠️ Don't change method signatures—callers depend on them
- ⚠️ Don't remove old methods yet—deprecate first
- ⚠️ Test after EVERY single method replacement
- ⚠️ If golden tests fail, revert immediately and investigate
- ⚠️ Watch for `&mut self` vs `&self` differences

---

## Testing Requirements

### After Each Phase

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `./scripts/golden-tests.sh verify` passes

### After Complete

- [ ] All tests pass
- [ ] Benchmark shows no regression (within 5%)
- [ ] Code is cleaner

### Rollback Plan

If golden tests fail:
1. `git diff` to see changes
2. `git checkout src/subproblem.rs` to revert
3. Investigate which change caused failure
4. Fix and retry one change at a time

---

## Documentation Requirements

- [ ] Update module docs in `subproblem.rs`
- [ ] Document the facade pattern usage
- [ ] Note which methods are deprecated (if any)

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: High integration risk, requires careful testing after each change

---

## Definition of Done

- [ ] SolutionExtractor integrated
- [ ] Solution extraction methods delegate
- [ ] Constraint builders integrated (if ready)
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Benchmark shows no regression
- [ ] Code is cleaner and more modular
- [ ] Ready for Epic 3
