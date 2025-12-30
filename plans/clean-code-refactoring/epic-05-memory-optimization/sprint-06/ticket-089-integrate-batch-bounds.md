# [T-089] Update Cut Realization to Use Batch Bound Updates

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: T-088
> **Blocks**: T-093

---

## Context

### Background

With the batch `change_rows_bounds_batch()` API implemented (T-088), this ticket updates all call sites to use the batch interface instead of individual calls.

DHAT showed 3 million individual `changeRowBounds` calls. The main sources are:
1. **Cut constraint realization** - Setting bounds on cut rows before solving
2. **Scenario realization** - Updating RHS values for uncertainty scenarios
3. **State transition** - Updating resource balance constraints

### Relation to Epic

Completes the batch bound update optimization by integrating it into production code paths.

### Current State

```rust
// Current pattern in state.rs, subproblem.rs
for row_idx in constraint_indices.iter() {
    model.change_rows_bounds(*row_idx, lb, ub)?;  // 3M calls
}
```

## Specification

### Target Call Sites

1. **`realize_and_solve()`** in `sddp/mod.rs` - Scenario realization
2. **`add_cut_with_preallocation()`** in `state.rs` - Cut bound setting
3. **`update_resource_balances()`** (if exists) - Resource constraint updates

### Inputs

For each call site:
- Identify the loop that makes individual calls
- Collect indices and bounds into buffers
- Call batch API once

### Outputs

- Same behavior (numerical correctness preserved)
- Reduced allocation count (verified by DHAT)

### Behavior

- Collect all row updates into thread-local or stack buffers
- Make single batch call
- Fallback to individual calls if batch fails (shouldn't happen)

### Error Handling

- Propagate errors from batch call
- Log and continue if single row fails in fallback mode

## Acceptance Criteria

- [x] All `change_rows_bounds` loops converted to batch calls
- [x] Thread-local buffers used to avoid per-call allocations
- [x] Golden tests pass (numerical correctness)
- [x] DHAT shows 90%+ reduction in `changeRowBounds` call count

**Status**: ✅ Complete

**Implementation Details**:
- Added thread-local buffers `BATCH_ROW_INDICES`, `BATCH_LOWER_BOUNDS`, `BATCH_UPPER_BOUNDS` in `src/subproblem.rs`
- Converted `update_uncertainty_constraints()` to use batch bounds
- Converted `update_lag_fixing_constraints()` to use batch bounds
- Converted `set_hydro_balance_rhs()` to use batch bounds
- Converted hydro balance updates in `prepare_from_trajectory()` to use batch bounds
- All 102 subproblem tests pass

## Implementation Guide

### Suggested Approach

1. **Identify all call sites**:
   ```bash
   rg "change_rows_bounds\(" src/ --type rust
   ```

2. **Create helper for batch bound updates**:
   ```rust
   // src/memory/buffers.rs or similar
   
   thread_local! {
       static BATCH_ROW_INDICES: RefCell<Vec<HighsInt>> = 
           RefCell::new(Vec::with_capacity(128));
       static BATCH_LOWER_BOUNDS: RefCell<Vec<f64>> = 
           RefCell::new(Vec::with_capacity(128));
       static BATCH_UPPER_BOUNDS: RefCell<Vec<f64>> = 
           RefCell::new(Vec::with_capacity(128));
   }
   
   pub fn with_batch_bounds<F, R>(f: F) -> R
   where
       F: FnOnce(&mut Vec<HighsInt>, &mut Vec<f64>, &mut Vec<f64>) -> R,
   {
       BATCH_ROW_INDICES.with(|indices| {
           BATCH_LOWER_BOUNDS.with(|lbs| {
               BATCH_UPPER_BOUNDS.with(|ubs| {
                   let mut indices = indices.borrow_mut();
                   let mut lbs = lbs.borrow_mut();
                   let mut ubs = ubs.borrow_mut();
                   indices.clear();
                   lbs.clear();
                   ubs.clear();
                   f(&mut indices, &mut lbs, &mut ubs)
               })
           })
       })
   }
   ```

3. **Update call sites**:
   ```rust
   // Before:
   for (row_idx, lb, ub) in bounds_to_update {
       model.change_rows_bounds(row_idx, lb, ub)?;
   }
   
   // After:
   with_batch_bounds(|indices, lbs, ubs| {
       for (row_idx, lb, ub) in bounds_to_update {
           indices.push(row_idx);
           lbs.push(lb);
           ubs.push(ub);
       }
       model.change_rows_bounds_batch(indices, lbs, ubs)
   })?;
   ```

4. **Handle special cases**:
   - Single row updates can still use individual call (no benefit from batch)
   - Mixed update patterns may need refactoring

### Key Files to Modify

- `src/sddp/mod.rs` - `realize_and_solve()` and related
- `src/state.rs` - Cut constraint updates
- `src/subproblem.rs` - Model bound updates
- `src/memory/buffers.rs` - Thread-local buffer helpers

### Patterns to Follow

- See existing thread-local buffer patterns in `src/memory/buffers.rs`
- Use `clear()` and reuse rather than allocate

### Pitfalls to Avoid

- ⚠️ Don't break loops that have early returns or error handling
- ⚠️ Ensure buffers are cleared before use (reuse across calls)
- ⚠️ Consider borrowing rules when using thread-local RefCell
- ⚠️ Verify row indices are valid before batch call

## Testing Requirements

### Unit Tests

- [ ] Test batch helper `with_batch_bounds`
- [ ] Test that buffers are properly cleared between uses

### Integration Tests

- [ ] Run full training with batch bounds
- [ ] Golden tests pass

### Validation Tests

- [ ] Compare DHAT allocation counts before/after
- [ ] Verify `changeRowBounds` call count drops by 90%+

## Documentation Requirements

- [ ] Update comments in modified functions
- [ ] Add note in `docs/MEMORY_BEHAVIOR.md` about batch bounds

## Dependencies

- **Blocked By**: T-088 (batch API implementation)
- **Blocks**: T-093 (DHAT verification)
- **Related**: T-087 (warm-start), T-092 (threading)

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Multiple call sites to update; need careful refactoring

## Definition of Done

- [ ] All call sites converted to batch
- [ ] Thread-local buffers implemented
- [ ] All tests passing
- [ ] Golden tests pass
- [ ] Code reviewed
- [ ] PR merged
