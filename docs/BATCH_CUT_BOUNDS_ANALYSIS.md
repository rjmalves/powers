# Batch Cut Bounds Optimization Analysis

> **Epic**: Epic 5 - Memory Optimization
> **Sprint**: Sprint 7 consideration
> **Date**: 2025-12-30
> **Related**: [DHAT_SPRINT6_ANALYSIS.md](./DHAT_SPRINT6_ANALYSIS.md)

---

## Executive Summary

Analysis of applying batch bounds API (`change_rows_bounds_batch`) to cut constraint updates reveals a **modest but achievable optimization opportunity**. The current DHAT shows ~0.17 GB and ~370K blocks from single `changeRowBounds` calls in cut operations, plus ~1.09 MB from cut-specific paths.

### Recommendation

**Include in Sprint 7** as a medium-priority ticket. The refactoring is localized to `apply_aggregated_cut_selection_result` and related functions, requiring buffer accumulation before batch submission.

---

## Current State Analysis

### DHAT Findings (Post-Sprint 6)

| Category | Bytes | Blocks | Source |
|----------|-------|--------|--------|
| Single changeRowBounds (all) | 0.17 GB | 370K | General bound changes |
| Cut-specific bound changes | 1.09 MB | 108K | Cut add/deactivate |
| **Total cut-related** | ~0.18 GB | ~480K | **Target** |

### Current Code Flow

```rust
// In apply_aggregated_cut_selection_result:
for (_cut_id, cut) in cuts_to_process {
    self.add_cut_to_model(cut, cut.iteration, cut.forward_pass_idx);
    // ^ Calls change_rows_bounds(row, cut.rhs, f64::INFINITY) per cut
}

for &cut_id in &aggregated_result.removing_cut_ids {
    self.deactivate_cut_by_coords(cut.iteration, cut.forward_pass_idx);
    // ^ Calls change_rows_bounds(row, f64::NEG_INFINITY, f64::INFINITY) per cut
}
```

**Problem**: Each cut addition and removal calls `change_rows_bounds` individually, triggering HiGHS internal allocations per call.

---

## Proposed Refactoring

### Target Architecture

```rust
pub fn apply_aggregated_cut_selection_result_batch(
    &mut self,
    aggregated_result: &AggregatedCutSelectionResult,
    cut_ids: &[usize],
    cut_pool: &[BendersCut],
) -> Result<(), String> {
    // 1. Collect all cut additions into batch buffers
    let mut add_rows = Vec::with_capacity(cut_ids.len());
    let mut add_lowers = Vec::with_capacity(cut_ids.len());
    let mut add_uppers = Vec::with_capacity(cut_ids.len());
    
    for (_cut_id, cut) in cuts_to_process {
        // Update coefficients (still individual - HiGHS API limitation)
        self.update_cut_coefficients(cut, iteration, forward_pass_idx);
        
        // Accumulate bound changes
        let row = self.slot_to_row(slot);
        add_rows.push(row as HighsInt);
        add_lowers.push(cut.rhs);
        add_uppers.push(f64::INFINITY);
    }
    
    // 2. Collect all cut removals into batch buffers
    let mut remove_rows = Vec::with_capacity(aggregated_result.removing_cut_ids.len());
    let mut remove_lowers = Vec::with_capacity(aggregated_result.removing_cut_ids.len());
    let mut remove_uppers = Vec::with_capacity(aggregated_result.removing_cut_ids.len());
    
    for &cut_id in &aggregated_result.removing_cut_ids {
        if cut_ids.contains(&cut_id) {
            if let Some(cut) = cut_pool.get(cut_id) {
                let slot = self.compute_cut_slot(cut.iteration, cut.forward_pass_idx);
                let row = self.slot_to_row(slot);
                remove_rows.push(row as HighsInt);
                remove_lowers.push(f64::NEG_INFINITY);
                remove_uppers.push(f64::INFINITY);
            }
        }
    }
    
    // 3. Apply batched bound changes
    if let Some(model) = self.model.as_mut() {
        if !add_rows.is_empty() {
            model.change_rows_bounds_batch(&add_rows, &add_lowers, &add_uppers)?;
        }
        if !remove_rows.is_empty() {
            model.change_rows_bounds_batch(&remove_rows, &remove_lowers, &remove_uppers)?;
        }
    }
    
    Ok(())
}
```

### Complexity Assessment

| Aspect | Difficulty | Notes |
|--------|------------|-------|
| `apply_aggregated_cut_selection_result` | **Easy** | Self-contained function |
| `add_cut_to_model` refactoring | **Medium** | Split coefficient update from bound update |
| `deactivate_cut_constraint` refactoring | **Easy** | Return row instead of modifying |
| Thread-local buffers | **Easy** | Reuse pattern from Sprint 6 |
| Testing | **Medium** | Need to verify determinism preserved |

### Refactoring Steps

1. **Add batch-compatible helpers**:
   - `update_cut_coefficients()` - Update only coefficients, not bounds
   - `get_cut_row()` - Return row index without modifying model

2. **Add thread-local batch buffers**:
   ```rust
   thread_local! {
       static CUT_BOUND_ROWS: RefCell<Vec<HighsInt>> = RefCell::new(Vec::with_capacity(64));
       static CUT_BOUND_LOWERS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
       static CUT_BOUND_UPPERS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
   }
   ```

3. **Refactor `apply_aggregated_cut_selection_result`** to use batch API

4. **Update tests** to verify identical behavior

---

## Impact Estimate

### Expected Improvement

| Metric | Current | Expected | Reduction |
|--------|---------|----------|-----------|
| Cut bound blocks | ~480K | ~2K | ~99% |
| Cut bound bytes | ~0.18 GB | ~0.001 GB | ~99% |
| **Overall impact** | 45.43 GB total | 45.25 GB | ~0.4% |

### Comparison to Sprint 6 Gains

| Sprint 6 optimization | Reduction |
|-----------------------|-----------|
| `reuse_forward_basis` disabled | 37.58 GB (95%) |
| Batch changeRowBounds (general) | 0.81 GB (82%) |
| **This proposal** | ~0.18 GB |

While smaller than Sprint 6 gains, this optimization:
1. Eliminates remaining single-row bound change overhead
2. Completes the batch bounds API adoption
3. Reduces allocation block count significantly

---

## Alternative: Combine with Coefficient Updates

A more aggressive optimization could batch coefficient changes too, but HiGHS `changeRowCoefficients` would need investigation.

**Not recommended for Sprint 7** - coefficient batching is less proven and higher risk.

---

## Recommendation

### For Sprint 7

Add as **T-097a: Batch cut constraint bound updates** with:
- **Points**: 3
- **Priority**: Medium
- **Confidence**: High (pattern established in Sprint 6)

### Implementation Order

1. T-097a: Batch cut constraint bound updates
2. T-094: Preallocated probability buffers (higher impact)
3. T-095: Thread-local scenario buffers
4. T-099: Remove `reuse_forward_basis` code

---

## Appendix: Affected Files

| File | Changes |
|------|---------|
| `src/subproblem.rs` | Refactor `apply_aggregated_cut_selection_result` |
| `src/subproblem.rs` | Add `update_cut_coefficients` helper |
| `src/subproblem.rs` | Add thread-local batch buffers |
| `tests/subproblem_tests.rs` | Verify identical behavior |

---

## Appendix: Related Tickets

- T-088: Implement batch changeRowBounds (✅ Complete)
- T-089: Integrate batch bounds (✅ Complete - general constraints)
- **T-097a**: Batch cut constraint bounds (NEW - Sprint 7)
