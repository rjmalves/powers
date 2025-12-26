# Epic 1: Solver Buffer Reuse

## Status

**Status**: ✅ Complete (2025-12-26)
**Priority**: HIGH (eliminates ~4.9 GB allocation churn)

## Summary

Eliminate the ~4.9 GB of transient allocations from `get_solution()` and `get_basis()` by adding buffer-into variants that reuse preallocated storage in `Realization`.

## Problem Statement

Every LP solve (153,600 solves in training) allocates new vectors:

```rust
pub fn get_solution(&self) -> Solution {
    let mut colvalue: Vec<f64> = vec![0.; cols];   // NEW ALLOCATION
    let mut coldual: Vec<f64> = vec![0.; cols];    // NEW ALLOCATION
    let mut rowvalue: Vec<f64> = vec![0.; rows];   // NEW ALLOCATION
    let mut rowdual: Vec<f64> = vec![0.; rows];    // NEW ALLOCATION
    // ...
}
```

**Critical Bug**: `Realization::with_capacity()` preallocates basis storage, but `realize_and_solve()` replaces it with newly allocated one:

```rust
realization_container.basis = basis;  // Preallocated buffer is discarded!
```

**Impact**: 153,600 solves × 32 KB = ~4.9 GB allocated (freed but causes RSS growth)

## Scope

### Included

- Add `Solution::with_capacity(cols, rows)` constructor
- Add `Basis::with_capacity(cols, rows)` constructor
- Add `Model::get_solution_into(&self, buf: &mut Solution)` method
- Add `Model::get_basis_into(&self, buf: &mut Basis)` method
- Update `Realization` to store and reuse buffers
- Update `realize_and_solve()` to use buffer-into pattern

### Excluded

- Changing `Solution` or `Basis` struct layouts
- Deprecating existing `get_solution()` / `get_basis()` methods (keep for compatibility)
- HiGHS internal buffer optimization

## Dependencies

- **Requires**: None
- **Enables**: Full memory determinism during training

## Acceptance Criteria

- [x] `get_solution_into()` writes to provided buffer without allocation
- [x] `get_basis_into()` writes to provided buffer without allocation
- [x] `realize_and_solve()` reuses thread-local solution buffer and realization basis buffer
- [ ] No new allocations per solve (verified via profiling)
- [x] Examples 01 and 07 produce identical results (tests pass)
- [x] No performance regression (expect improvement)

## Technical Approach

### Phase 1: Add Buffer-Into Methods to solver.rs

```rust
impl Solution {
    pub fn with_capacity(cols: usize, rows: usize) -> Self {
        Self {
            colvalue: vec![0.0; cols],
            coldual: vec![0.0; cols],
            rowvalue: vec![0.0; rows],
            rowdual: vec![0.0; rows],
        }
    }

    pub fn ensure_capacity(&mut self, cols: usize, rows: usize) {
        if self.colvalue.len() < cols {
            self.colvalue.resize(cols, 0.0);
            self.coldual.resize(cols, 0.0);
        }
        if self.rowvalue.len() < rows {
            self.rowvalue.resize(rows, 0.0);
            self.rowdual.resize(rows, 0.0);
        }
    }
}

impl Model {
    pub fn get_solution_into(&self, solution: &mut Solution) {
        let cols = self.num_cols();
        let rows = self.num_rows();
        solution.ensure_capacity(cols, rows);
        
        unsafe {
            Highs_getSolution(
                self.highs.unsafe_mut_ptr(),
                solution.colvalue.as_mut_ptr(),
                solution.coldual.as_mut_ptr(),
                solution.rowvalue.as_mut_ptr(),
                solution.rowdual.as_mut_ptr(),
            );
        }
    }
}
```

### Phase 2: Add Basis Buffer-Into Methods

Similar pattern for `Basis`:

```rust
impl Basis {
    pub fn with_capacity(cols: usize, rows: usize) -> Self {
        Self {
            colstatus: vec![0; cols],
            rowstatus: vec![0; rows],
        }
    }
}

impl Model {
    pub fn get_basis_into(&self, basis: &mut Basis) {
        // Use internal raw buffer, copy converted values
    }
}
```

**Note**: Basis requires i32→usize conversion. Options:
- Store raw c_int internally, convert on access
- Keep usize storage, use temporary buffer for FFI call

### Phase 3: Update Realization to Use Buffers

Find where `Realization` is created and ensure it stores preallocated buffers that are reused.

## Estimated Effort

- **Sprint 1**: 5 story points (1 week)
  - Ticket 1: Solution buffer-into (2 points)
  - Ticket 2: Basis buffer-into (2 points)
  - Ticket 3: Realization integration (1 point)

## Key Files

| File | Purpose |
|------|---------|
| `src/solver.rs` | Solution, Basis, Model structs |
| `src/subproblem.rs` | Realization struct, realize_and_solve() |
| `src/sddp/mod.rs` | Training loop calling solve |

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Buffer size mismatch | Low | ensure_capacity() handles resize |
| Basis conversion overhead | Medium | Profile; consider raw storage |
| API compatibility | Low | Add new methods, keep old ones |
