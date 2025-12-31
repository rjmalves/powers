# [T-098] Batch Cut Constraint Bound Updates

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 2 (HiGHS Optimization)
> **Status**: ✅ Complete

## Files to Read Before Starting

- `docs/BATCH_CUT_BOUNDS_ANALYSIS.md` - Full analysis of this optimization
- `src/subproblem.rs` - `apply_aggregated_cut_selection_result()`, `add_cut_with_preallocation()`, `deactivate_cut_by_coords()`
- `src/solver.rs` - `change_rows_bounds_batch()` API

---

## Context

### Background

Sprint 6 implemented `change_rows_bounds_batch()` for general constraint updates, achieving **99.6% block reduction** in changeRowBounds allocations. However, cut constraint bound updates still use individual calls:

| Category | Bytes | Blocks | Source |
|----------|-------|--------|--------|
| Single changeRowBounds (cuts) | 0.17 GB | 370K | Cut add/deactivate |

### Current Code Flow

```rust
// In apply_aggregated_cut_selection_result():
for (_cut_id, cut) in cuts_to_process {
    self.add_cut_to_model(cut, cut.iteration, cut.forward_pass_idx);
    // ^ Calls change_rows_bounds(row, cut.rhs, f64::INFINITY) per cut
}

for &cut_id in &aggregated_result.removing_cut_ids {
    self.deactivate_cut_by_coords(cut.iteration, cut.forward_pass_idx);
    // ^ Calls change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY) per cut
}
```

### Target

Batch all cut bound updates into two `change_rows_bounds_batch()` calls per stage.

---

## Specification

### Inputs

- `aggregated_result`: Cut selection results (new, returning, removing cut IDs)
- `cut_ids`: List of cut IDs to process
- `cut_pool`: Reference to cut pool for cut data access

### Outputs

- Same behavior as current implementation
- Reduced allocation count

### Behavior

1. **Collect cut additions** into batch buffers:
   - Row indices, lower bounds (cut.rhs), upper bounds (INFINITY)
   
2. **Collect cut removals** into batch buffers:
   - Row indices, lower bounds (NEG_INFINITY), upper bounds (INFINITY)

3. **Apply batch updates**:
   - Single `change_rows_bounds_batch()` for additions
   - Single `change_rows_bounds_batch()` for removals

4. **Preserve determinism**:
   - Process in same order as current implementation
   - Coefficient updates remain individual (HiGHS API limitation)

---

## Acceptance Criteria

- [ ] `apply_aggregated_cut_selection_result()` refactored to use batch bounds
- [ ] Thread-local buffers added for batch accumulation
- [ ] All tests pass (especially golden tests)
- [ ] DHAT shows reduction in single changeRowBounds allocations
- [ ] No performance regression

---

## Implementation Guide

### Suggested Approach

1. **Add thread-local batch buffers** in `src/subproblem.rs`:
   ```rust
   thread_local! {
       static CUT_BOUND_ROWS: RefCell<Vec<HighsInt>> = RefCell::new(Vec::with_capacity(64));
       static CUT_BOUND_LOWERS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
       static CUT_BOUND_UPPERS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
   }
   ```

2. **Add coefficient-only update helper**:
   ```rust
   /// Update cut coefficients without changing bounds.
   /// Returns the row index for later batch bound update.
   fn update_cut_coefficients_only(
       &mut self,
       cut: &cut::BendersCut,
       iteration: usize,
       forward_pass_idx: usize,
   ) -> usize {
       let slot = self.compute_cut_slot(iteration, forward_pass_idx);
       let row = self.slot_to_row(slot);
       
       if let Some(model) = self.model.as_mut() {
           for (i, &var_idx) in self.cut_var_indices.iter().enumerate() {
               let coef = if i == 0 { 1.0 } else { -cut.coefficients[i - 1] };
               model.change_coefficient(row, var_idx, coef).expect("Failed to set coefficient");
           }
       }
       
       cut.set_slot_index(slot);
       row
   }
   ```

3. **Refactor `apply_aggregated_cut_selection_result()`**:
   ```rust
   pub fn apply_aggregated_cut_selection_result(
       &mut self,
       aggregated_result: &fcf::AggregatedCutSelectionResult,
       cut_ids: &[usize],
       cut_pool: &[cut::BendersCut],
   ) -> Result<(), String> {
       CUT_BOUND_ROWS.with(|rows| {
       CUT_BOUND_LOWERS.with(|lowers| {
       CUT_BOUND_UPPERS.with(|uppers| {
           let mut rows = rows.borrow_mut();
           let mut lowers = lowers.borrow_mut();
           let mut uppers = uppers.borrow_mut();
           
           // Clear buffers
           rows.clear();
           lowers.clear();
           uppers.clear();
           
           // Collect additions
           for (_cut_id, cut) in cuts_to_process {
               let row = self.update_cut_coefficients_only(cut, cut.iteration, cut.forward_pass_idx);
               rows.push(row as HighsInt);
               lowers.push(cut.rhs);
               uppers.push(f64::INFINITY);
           }
           
           // Apply batch addition
           if let Some(model) = self.model.as_mut() {
               if !rows.is_empty() {
                   let _ = model.change_rows_bounds_batch(&rows, &lowers, &uppers);
               }
           }
           
           // Clear for removals
           rows.clear();
           lowers.clear();
           uppers.clear();
           
           // Collect removals
           for &cut_id in &aggregated_result.removing_cut_ids {
               if cut_ids.contains(&cut_id) {
                   if let Some(cut) = cut_pool.get(cut_id) {
                       let slot = self.compute_cut_slot(cut.iteration, cut.forward_pass_idx);
                       let row = self.slot_to_row(slot);
                       rows.push(row as HighsInt);
                       lowers.push(f64::NEG_INFINITY);
                       uppers.push(f64::INFINITY);
                   }
               }
           }
           
           // Apply batch removal
           if let Some(model) = self.model.as_mut() {
               if !rows.is_empty() {
                   let _ = model.change_rows_bounds_batch(&rows, &lowers, &uppers);
               }
           }
           
           Ok(())
       })})})
   }
   ```

### Key Files to Modify

- `src/subproblem.rs`:
  - Add thread-local buffers
  - Add `update_cut_coefficients_only()` helper
  - Refactor `apply_aggregated_cut_selection_result()`

### Patterns to Follow

- See Sprint 6's batch bounds integration in `update_uncertainty_constraints()`
- Follow existing thread-local buffer patterns

### Pitfalls to Avoid

- ⚠️ Maintain deterministic processing order
- ⚠️ Don't forget to update coefficients (only bounds are batched)
- ⚠️ Handle empty cut lists correctly
- ⚠️ The `with()` nesting for thread-locals can be verbose - consider helper

---

## Testing Requirements

### Unit Tests

- [ ] Test batch addition with multiple cuts
- [ ] Test batch removal with multiple cuts
- [ ] Test mixed addition and removal
- [ ] Test empty cut lists (no-op)

### Integration Tests

- [ ] Golden tests pass (numerical determinism preserved)
- [ ] Full training run completes successfully

### Performance Tests

- [ ] DHAT shows reduced single changeRowBounds allocations
- [ ] No solve time regression

---

## Documentation Requirements

- [ ] Update doc comments for modified functions
- [ ] Note in `BATCH_CUT_BOUNDS_ANALYSIS.md` that implementation is complete

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None
- **Related**: Sprint 6 T-088, T-089 (batch bounds implementation)

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Pattern established in Sprint 6; localized refactoring

---

## Definition of Done

- [ ] Batch bounds implemented for cut constraints
- [ ] All tests passing
- [ ] Golden tests confirm determinism
- [ ] DHAT shows allocation reduction
- [ ] Documentation updated
- [ ] PR merged
