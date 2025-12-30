# [T-088] Implement Batch changeRowBounds Interface

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 6: HiGHS Solver Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-089

---

## Context

### Background

DHAT profiling revealed that `changeRowBounds` is called **3 million times** during SDDP training, with each call allocating:
- Vector for row indices
- Two strings for debug messages
- Small internal buffers

This accounts for ~188 MB of allocation churn with 3M × 62 bytes average per call.

HiGHS provides a bulk API `Highs_changeRowsBounds` (plural) that accepts arrays of indices and bounds, but our current wrapper `Model::change_rows_bounds()` only updates one row at a time.

### Relation to Epic

Reduces allocation frequency in hot path by batching individual calls into bulk operations.

### Current State

```rust
// src/solver.rs - Current implementation
pub fn change_rows_bounds(&mut self, row_idx: i32, lb: f64, ub: f64) -> Result<(), HighsError> {
    let status = unsafe {
        Highs_changeRowBounds(self.ptr, row_idx, lb, ub)  // Single row
    };
    // ...
}
```

Called from `state.rs` and `subproblem.rs` in loops for multiple rows.

## Specification

### Inputs

- `row_indices: &[HighsInt]` - Array of row indices to update
- `lower_bounds: &[f64]` - Array of new lower bounds
- `upper_bounds: &[f64]` - Array of new upper bounds

### Outputs

- `Result<(), HighsError>` - Success or error

### Behavior

- All three arrays must have the same length
- Row indices must be valid (within model row count)
- Bounds are updated atomically (all or none)
- Empty arrays are no-op

### Error Handling

- `HighsError::InvalidInput` if array lengths don't match
- `HighsError::InvalidRowIndex` if any index out of range
- Propagate HiGHS status errors

## Acceptance Criteria

- [x] `Model::change_rows_bounds_batch()` method implemented
- [x] Input validation for array length matching
- [x] Returns appropriate error for invalid indices
- [x] Unit tests cover happy path and error cases
- [x] DHAT shows reduced allocation count when used

**Status**: ✅ Complete

**Implementation Details**:
- Added `change_rows_bounds_batch()` in `src/solver.rs`
- Uses `Highs_changeRowsBoundsBySet` for single FFI call
- 5 unit tests covering: empty arrays, single row, multiple rows, mismatched lengths, and solve preservation

## Implementation Guide

### Suggested Approach

1. **Add batch method to Model**:
   ```rust
   // src/solver.rs
   
   /// Change bounds for multiple rows in a single call.
   /// All arrays must have the same length.
   pub fn change_rows_bounds_batch(
       &mut self,
       row_indices: &[HighsInt],
       lower_bounds: &[f64],
       upper_bounds: &[f64],
   ) -> Result<(), HighsError> {
       if row_indices.len() != lower_bounds.len() 
           || row_indices.len() != upper_bounds.len() 
       {
           return Err(HighsError::InvalidInput(
               "Array lengths must match".to_string()
           ));
       }
       
       if row_indices.is_empty() {
           return Ok(());
       }
       
       let status = unsafe {
           Highs_changeRowsBounds(
               self.ptr,
               row_indices.len() as HighsInt,
               row_indices.as_ptr(),
               lower_bounds.as_ptr(),
               upper_bounds.as_ptr(),
           )
       };
       
       if status == STATUS_OK {
           Ok(())
       } else {
           Err(HighsError::from_status(status))
       }
   }
   ```

2. **Add thread-local buffers for accumulating changes** (optional optimization):
   ```rust
   thread_local! {
       static ROW_INDICES: RefCell<Vec<HighsInt>> = RefCell::new(Vec::with_capacity(64));
       static LOWER_BOUNDS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
       static UPPER_BOUNDS: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(64));
   }
   ```

3. **Verify HiGHS binding exists**:
   - Check `highs-sys` crate for `Highs_changeRowsBounds`
   - If not present, may need to use `Highs_changeRowBounds` in loop (still reduces Rust overhead)

### Key Files to Modify

- `src/solver.rs` - Add `change_rows_bounds_batch()` method
- `tests/solver_tests.rs` - Add unit tests for batch method

### Patterns to Follow

- See existing `change_rows_bounds()` for error handling pattern
- Follow naming convention `_batch` suffix for bulk operations

### Pitfalls to Avoid

- ⚠️ Don't assume HiGHS checks index validity - validate first
- ⚠️ Ensure pointer lifetimes are correct in FFI call
- ⚠️ Empty slice handling must not crash

## Testing Requirements

### Unit Tests

- [ ] Test batch update with multiple rows
- [ ] Test empty arrays (no-op)
- [ ] Test mismatched array lengths (error)
- [ ] Test with maximum valid row indices
- [ ] Test with invalid row index (error)

### Integration Tests

- [ ] Verify bounds actually change in model
- [ ] Verify solve produces correct result after batch update

### Performance Tests

- [ ] Micro-benchmark: 1000 single calls vs 1 batch call
- [ ] DHAT comparison before/after

## Documentation Requirements

- [ ] Doc comments for public method
- [ ] Update `docs/MEMORY_BEHAVIOR.md` with batch API usage

## Dependencies

- **Blocked By**: None
- **Blocks**: T-089 (update call sites to use batch)
- **Related**: T-087 (warm-start investigation)

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward FFI wrapper with clear API

## Definition of Done

- [ ] Implementation complete
- [ ] Unit tests passing
- [ ] Doc comments added
- [ ] Code reviewed
- [ ] PR merged
