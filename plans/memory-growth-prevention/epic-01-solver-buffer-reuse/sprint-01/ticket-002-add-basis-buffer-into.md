# [TICKET-002] Add Basis buffer-into methods and fix with_capacity

> **Epic**: [Epic 1: Solver Buffer Reuse](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-003](./ticket-003-update-realize-and-solve.md)
> **Status**: ✅ Complete (2025-12-26)

## Context

### Background

Every LP solve calls `model.get_basis()` which allocates 4 vectors (2 raw c_int, 2 converted usize). Additionally, `Basis::with_capacity()` only reserves capacity but doesn't initialize, so when `get_basis()` returns a new Basis and it's assigned to `realization_container.basis`, the preallocated storage is discarded.

### Current State

```rust
// src/solver.rs:650-672
pub fn get_basis(&self) -> Basis {
    let cols = self.num_cols();
    let rows = self.num_rows();
    let mut raw_colstatus: Vec<c_int> = vec![0; cols];  // NEW ALLOCATION
    let mut raw_rowstatus: Vec<c_int> = vec![0; rows];  // NEW ALLOCATION
    
    unsafe {
        Highs_getBasis(
            self.highs.unsafe_mut_ptr(),
            raw_colstatus.as_mut_ptr(),
            raw_rowstatus.as_mut_ptr(),
        );
    }
    
    let colstatus = raw_colstatus.iter().map(|x| *x as usize).collect();  // NEW ALLOCATION
    let rowstatus = raw_rowstatus.iter().map(|x| *x as usize).collect();  // NEW ALLOCATION
    
    Basis { colstatus, rowstatus }
}

// src/solver.rs:898-903
pub fn with_capacity(num_cols: usize, num_rows: usize) -> Self {
    Self {
        colstatus: Vec::<usize>::with_capacity(num_cols),  // Only reserves, doesn't init!
        rowstatus: Vec::<usize>::with_capacity(num_rows),
    }
}
```

### Files to Read Before Starting

- `src/solver.rs` - Basis struct and Model impl

## Specification

### Inputs

- `Basis::with_size(cols: usize, rows: usize)`: Create initialized (not just reserved) basis
- `Model::get_basis_into(&self, basis: &mut Basis)`: Write basis to existing buffer

### Outputs

- Basis struct populated with solver values, no new allocations

### Behavior

- `with_size` creates zero-initialized vectors (not just reserved capacity)
- `get_basis_into` uses internal raw buffer for FFI, converts in-place to target buffer
- Existing `get_basis()` method remains unchanged for backward compatibility

### Design Decision: Raw Buffer Strategy

Two options for the FFI call which requires `c_int` buffers:

**Option A: Store raw internally, convert on access** (cleaner but changes struct)
```rust
pub struct Basis {
    raw_colstatus: Vec<c_int>,  // FFI-compatible
    raw_rowstatus: Vec<c_int>,
}
impl Basis {
    pub fn columns(&self) -> impl Iterator<Item = usize> { ... }
}
```

**Option B: Thread-local scratch buffer** (preserves API, adds complexity)
```rust
thread_local! {
    static RAW_COL_BUFFER: RefCell<Vec<c_int>> = RefCell::new(Vec::new());
    static RAW_ROW_BUFFER: RefCell<Vec<c_int>> = RefCell::new(Vec::new());
}
```

**Recommended: Option B** - Preserves existing API, minimal changes.

## Acceptance Criteria

- [x] `Basis::with_size(cols, rows)` creates zero-initialized Basis
- [x] `Model::get_basis_into(&self, buf: &mut Basis)` writes to buffer
- [x] No new heap allocations per call (uses thread-local scratch)
- [x] Existing `get_basis()` still works
- [x] Unit tests pass for both methods

## Implementation Guide

### Suggested Approach

1. Add `with_size` constructor that initializes vectors (not just reserves)
2. Add thread-local raw buffers for FFI calls
3. Add `ensure_capacity` method for buffer resizing
4. Add `get_basis_into` method using thread-local scratch
5. Add unit tests

### Key Files to Modify

- `src/solver.rs`: Add methods to `Basis` impl and `Model` impl

### Code Changes

```rust
use std::cell::RefCell;

// Thread-local scratch buffers for basis FFI calls
thread_local! {
    static RAW_COL_BUFFER: RefCell<Vec<c_int>> = RefCell::new(Vec::new());
    static RAW_ROW_BUFFER: RefCell<Vec<c_int>> = RefCell::new(Vec::new());
}

impl Basis {
    /// Creates a Basis with initialized (not just reserved) buffers.
    pub fn with_size(cols: usize, rows: usize) -> Self {
        Self {
            colstatus: vec![0; cols],
            rowstatus: vec![0; rows],
        }
    }
    
    /// Ensures buffers have sufficient size, resizing if needed.
    #[inline]
    pub fn ensure_size(&mut self, cols: usize, rows: usize) {
        if self.colstatus.len() < cols {
            self.colstatus.resize(cols, 0);
        }
        if self.rowstatus.len() < rows {
            self.rowstatus.resize(rows, 0);
        }
    }
}

impl Model {
    /// Gets the basis, writing into an existing buffer.
    /// 
    /// Uses thread-local scratch buffers for FFI call to avoid allocation.
    pub fn get_basis_into(&self, basis: &mut Basis) {
        let cols = self.num_cols();
        let rows = self.num_rows();
        basis.ensure_size(cols, rows);
        
        RAW_COL_BUFFER.with(|raw_col| {
            RAW_ROW_BUFFER.with(|raw_row| {
                let mut raw_col = raw_col.borrow_mut();
                let mut raw_row = raw_row.borrow_mut();
                
                // Resize scratch buffers if needed
                if raw_col.len() < cols {
                    raw_col.resize(cols, 0);
                }
                if raw_row.len() < rows {
                    raw_row.resize(rows, 0);
                }
                
                unsafe {
                    Highs_getBasis(
                        self.highs.unsafe_mut_ptr(),
                        raw_col.as_mut_ptr(),
                        raw_row.as_mut_ptr(),
                    );
                }
                
                // Convert c_int -> usize in-place
                for (i, &raw) in raw_col.iter().take(cols).enumerate() {
                    basis.colstatus[i] = raw as usize;
                }
                for (i, &raw) in raw_row.iter().take(rows).enumerate() {
                    basis.rowstatus[i] = raw as usize;
                }
            });
        });
    }
}
```

### Pitfalls to Avoid

- ⚠️ Don't confuse `with_capacity` (reserves) with `with_size` (initializes)
- ⚠️ Thread-local buffers must be sized correctly before use
- ⚠️ The `colstatus`/`rowstatus` fields are private - may need internal access

## Testing Requirements

### Unit Tests

- [x] Test `Basis::with_size(100, 50)` creates correct sizes (not just capacity)
- [x] Test `get_basis_into` produces same values as `get_basis`
- [x] Test `ensure_size` grows buffers when needed

### Integration Tests

- [x] Run example 01-deterministic, verify identical objective value (tests pass)

## Documentation Requirements

- [x] Add doc comments to new public methods
- [x] Document thread-local scratch buffer strategy

## Effort Estimate

**Points**: 2  
**Confidence**: Medium  
**Rationale**: Thread-local pattern adds complexity, but logic is straightforward

## Definition of Done

- [x] Implementation complete
- [x] Tests passing
- [ ] Code reviewed
- [ ] PR merged
