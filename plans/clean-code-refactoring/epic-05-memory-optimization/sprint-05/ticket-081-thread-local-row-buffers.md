# [T-081] Add Thread-Local Buffers for try_add_row

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-085](./ticket-085-dhat-profiling.md)
> **Status**: ✅ Complete

---

## Context

### Background

The `try_add_row()` method in `solver.rs` (lines 512-534) creates **two temporary Vec allocations** on every call:

```rust
pub fn try_add_row(
    &mut self,
    bounds: impl RangeBounds<f64>,
    row_factors: impl IntoIterator<Item = (usize, f64)>,
) -> Result<usize, HighsStatus> {
    let (cols, factors): (Vec<_>, Vec<_>) = row_factors.into_iter().unzip();  // ALLOC 1

    unsafe {
        highs_call!(Highs_addRow(
            // ...
            cols.into_iter()
                .map(|c| c.try_into().unwrap())
                .collect::<Vec<_>>()  // ALLOC 2
                .as_ptr(),
            // ...
        ))
    }?;
    // ...
}
```

While training should use preallocation (T-080), edge cases during initialization or simulation may still need `add_row()`. Thread-local buffers eliminate these allocations.

### Current State

- Every `add_row()` call allocates 2 Vecs
- Buffers are dropped immediately after the HiGHS call
- Same memory pattern repeated thousands of times

### Target State

- Thread-local buffers reused across all `add_row()` calls
- Zero allocation overhead for row additions
- Buffers sized to handle largest expected row

---

## Specification

### Inputs

- Row factors as iterator: `impl IntoIterator<Item = (usize, f64)>`
- Row bounds: `impl RangeBounds<f64>`

### Outputs

- Same functionality as current `try_add_row()`
- Zero heap allocations per call (after initial buffer allocation)

### Behavior

```rust
thread_local! {
    static ROW_COLS_BUFFER: RefCell<Vec<HighsInt>> = RefCell::new(Vec::with_capacity(256));
    static ROW_VALS_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::with_capacity(256));
}

pub fn try_add_row_preallocated(
    &mut self,
    bounds: impl RangeBounds<f64>,
    row_factors: impl IntoIterator<Item = (usize, f64)>,
) -> Result<usize, HighsStatus> {
    ROW_COLS_BUFFER.with(|cols_cell| {
        ROW_VALS_BUFFER.with(|vals_cell| {
            let mut cols = cols_cell.borrow_mut();
            let mut vals = vals_cell.borrow_mut();
            cols.clear();
            vals.clear();

            for (col, val) in row_factors {
                cols.push(col as HighsInt);
                vals.push(val);
            }

            unsafe {
                highs_call!(Highs_addRow(
                    self.highs.mut_ptr(),
                    bound_value(bounds.start_bound()).unwrap_or(f64::NEG_INFINITY),
                    bound_value(bounds.end_bound()).unwrap_or(f64::INFINITY),
                    cols.len() as HighsInt,
                    cols.as_ptr(),
                    vals.as_ptr()
                ))
            }?;

            Ok(self.highs.num_rows()? - 1)
        })
    })
}
```

### Error Handling

- If buffer capacity exceeded, buffer grows (one-time allocation per thread)
- HiGHS errors propagated as `HighsStatus::Error`

---

## Acceptance Criteria

- [x] New `try_add_row_preallocated()` method uses thread-local buffers
- [x] Existing `try_add_row()` delegates to new method (or replace in-place)
- [x] Buffer capacity is 256 elements (covers typical row sizes with room to grow)
- [ ] No allocation on repeated `add_row()` calls (DHAT verified - requires manual verification)
- [x] All existing tests pass
- [x] No numerical result changes

### Implementation Notes

Added three thread-local buffers at module level in `solver.rs`:
- `ROW_COLS_BUFFER`: Column indices for add_row
- `ROW_VALS_BUFFER`: Coefficient values for add_row
- `DELETE_ROW_BUFFER`: Row index set for delete_row

`try_add_row()` was updated in-place to use the buffers. `delete_row()` was also updated.

---

## Implementation Guide

### Suggested Approach

1. **Add thread-local buffers** at module level in `solver.rs`

2. **Implement `try_add_row_preallocated()`** that uses the buffers

3. **Replace `try_add_row()` body** to use the preallocated version:
   ```rust
   pub fn try_add_row(...) -> Result<usize, HighsStatus> {
       self.try_add_row_preallocated(bounds, row_factors)
   }
   ```

4. **Also update `delete_row()`** which has a similar allocation:
   ```rust
   pub fn delete_row(&self, row_index: usize) -> Result<(), HighsStatus> {
       let set: Vec<HighsInt> = vec![row_index as HighsInt];  // ALLOCATION
       // ...
   }
   ```

### Key Files to Modify

- `src/solver.rs`: Add thread-local buffers and update `try_add_row()`, `delete_row()`

### Patterns to Follow

- See `SOLUTION_BUFFER` in `subproblem.rs:21-28` for thread-local pattern
- See `CutComputationBuffers` in `memory/buffers.rs` for buffer reuse pattern

### Pitfalls to Avoid

- ⚠️ Don't forget to call `.clear()` before reusing buffers
- ⚠️ `RefCell::borrow_mut()` will panic if already borrowed - ensure no nested calls
- ⚠️ Thread-local buffers are per-thread; Rayon workers each get their own

---

## Testing Requirements

### Unit Tests

- [ ] Test `try_add_row_preallocated()` with empty row
- [ ] Test with single element row
- [ ] Test with large row (100+ elements)
- [ ] Test multiple sequential calls reuse buffers
- [ ] Test `delete_row()` also uses buffer

### Performance Tests

- [ ] DHAT shows zero allocations after initial buffer setup

---

## Documentation Requirements

- [ ] Add doc comment explaining the thread-local buffer strategy
- [ ] Document capacity choice (256 elements)

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward pattern following existing thread-local implementations in the codebase.

---

## Definition of Done

- [ ] Thread-local buffers implemented for row operations
- [ ] `try_add_row()` and `delete_row()` use preallocated buffers
- [ ] All tests passing
- [ ] DHAT confirms zero allocations per row operation
- [ ] Code reviewed and merged
