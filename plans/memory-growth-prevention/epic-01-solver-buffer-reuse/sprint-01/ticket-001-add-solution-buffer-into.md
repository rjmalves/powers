# [TICKET-001] Add Solution::with_capacity and get_solution_into

> **Epic**: [Epic 1: Solver Buffer Reuse](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-003](./ticket-003-update-realize-and-solve.md)
> **Status**: ✅ Complete (2025-12-26)

## Context

### Background

Every LP solve in `realize_and_solve()` calls `model.get_solution()` which allocates 4 new vectors (colvalue, coldual, rowvalue, rowdual). With 153,600 solves during training, this causes ~4.9 GB of transient allocations that fragment memory and increase RSS.

### Current State

```rust
// src/solver.rs:622-647
pub fn get_solution(&self) -> Solution {
    let cols = self.num_cols();
    let rows = self.num_rows();
    let mut colvalue: Vec<f64> = vec![0.; cols];   // NEW ALLOCATION
    let mut coldual: Vec<f64> = vec![0.; cols];    // NEW ALLOCATION
    let mut rowvalue: Vec<f64> = vec![0.; rows];   // NEW ALLOCATION
    let mut rowdual: Vec<f64> = vec![0.; rows];    // NEW ALLOCATION
    
    unsafe {
        Highs_getSolution(
            self.highs.unsafe_mut_ptr(),
            colvalue.as_mut_ptr(),
            coldual.as_mut_ptr(),
            rowvalue.as_mut_ptr(),
            rowdual.as_mut_ptr(),
        );
    }
    
    Solution { colvalue, coldual, rowvalue, rowdual }
}
```

## Specification

### Inputs

- `Solution::with_capacity(cols: usize, rows: usize)`: Create preallocated solution
- `Model::get_solution_into(&self, solution: &mut Solution)`: Write solution to existing buffer

### Outputs

- Solution struct populated with solver values, no new allocations

### Behavior

- `with_capacity` creates zero-initialized vectors of specified sizes
- `get_solution_into` resizes buffers if needed (should rarely happen with correct sizing)
- `get_solution_into` writes solution values into provided buffer
- Existing `get_solution()` method remains unchanged for backward compatibility

## Acceptance Criteria

- [x] `Solution::with_capacity(cols, rows)` creates preallocated Solution
- [x] `Model::get_solution_into(&self, buf: &mut Solution)` writes to buffer
- [x] Buffer is resized only if capacity is insufficient
- [x] Existing `get_solution()` still works (backward compatibility)
- [x] Unit tests pass for both methods

## Implementation Guide

### Suggested Approach

1. Add `with_capacity` constructor to `Solution` struct
2. Add `ensure_capacity` method for buffer resizing
3. Add `get_solution_into` method to `Model` that uses FFI call with existing buffers
4. Add unit tests

### Key Files to Modify

- `src/solver.rs`: Add methods to `Solution` impl and `Model` impl

### Code Changes

```rust
// Add to Solution impl block (~line 647)
impl Solution {
    /// Creates a Solution with preallocated buffers.
    pub fn with_capacity(cols: usize, rows: usize) -> Self {
        Self {
            colvalue: vec![0.0; cols],
            coldual: vec![0.0; cols],
            rowvalue: vec![0.0; rows],
            rowdual: vec![0.0; rows],
        }
    }
    
    /// Ensures buffers have sufficient capacity, resizing if needed.
    /// Returns true if resize was needed.
    #[inline]
    pub fn ensure_capacity(&mut self, cols: usize, rows: usize) -> bool {
        let mut resized = false;
        if self.colvalue.len() < cols {
            self.colvalue.resize(cols, 0.0);
            self.coldual.resize(cols, 0.0);
            resized = true;
        }
        if self.rowvalue.len() < rows {
            self.rowvalue.resize(rows, 0.0);
            self.rowdual.resize(rows, 0.0);
            resized = true;
        }
        resized
    }
}

// Add to Model impl block (~line 647, after get_solution)
impl Model {
    /// Gets the solution, writing into an existing buffer.
    /// 
    /// This avoids allocation when the buffer has sufficient capacity.
    /// The buffer is resized if necessary.
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

### Pitfalls to Avoid

- ⚠️ Don't truncate buffers - use resize to grow only, or keep oversized buffers
- ⚠️ Ensure FFI pointers are valid (as_mut_ptr on correctly sized vecs)
- ⚠️ Don't modify existing `get_solution()` behavior

## Testing Requirements

### Unit Tests

- [ ] Test `Solution::with_capacity(100, 50)` creates correct sizes
- [ ] Test `get_solution_into` produces same values as `get_solution`
- [ ] Test `ensure_capacity` grows buffers when needed
- [ ] Test `ensure_capacity` returns false when already sized

### Integration Tests

- [x] Run example 01-deterministic, verify identical objective value (tests pass)

## Documentation Requirements

- [x] Add doc comments to new public methods
- [ ] Update module documentation if needed

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward addition of new methods, no complex logic

## Definition of Done

- [x] Implementation complete
- [x] Tests passing
- [ ] Code reviewed
- [ ] PR merged
