# TICKET-014: Optimize Constraint RHS Updates with Batching

**Sprint:** 4 - Optimization  
**Phase:** 4 - Optimize Performance  
**Estimated Effort:** 2 days (5 story points)  
**Confidence:** Medium  
**Status:** Not Started

## Context

Currently, constraint RHS values are updated individually via repeated solver API calls. Each call has overhead from FFI boundaries (Rust → C), function call overhead, and potential cache misses. This ticket implements batched RHS updates where possible, reducing the number of API calls and improving cache locality.

This is a low-risk optimization that can yield 5-10% speedup in subproblem solving.

## Acceptance Criteria

- [ ] Given multiple constraint RHS updates, when performed in sequence, then they are batched into single API call
- [ ] Given solver API, when batch update method is available, then it is used instead of individual updates
- [ ] Given fallback scenario, when batch API is not available, then individual updates are used (backward compatibility)
- [ ] Given batched updates, when tested, then correctness is identical to individual updates
- [ ] Performance: realize_uncertainties should be 5-10% faster due to reduced API overhead
- [ ] Performance: update_from_trajectory should show measurable speedup

## Tasks

### Implementation

- [ ] Audit solver API for batch RHS update methods:
  - Check highs-sys bindings for batch operations
  - Document available batch methods (e.g., `changeRowBounds`, `changeColBounds` with arrays)
- [ ] Add batch update method to solver::Model wrapper:
  - `batch_change_rhs(constraints: &[usize], values: &[(f64, f64)]) -> Result<(), String>`
  - Handles both lower and upper bounds
- [ ] Update `update_ar_constraint_rhs()` to use batching:
  - Collect all constraint indices and RHS values
  - Single batch update call instead of loop of individual calls
- [ ] Update `update_from_trajectory()` lag RHS updates to use batching:
  - Collect all lag constraint indices and values
  - Batch update in single call
- [ ] Update load balance RHS updates to use batching:
  - Collect all load balance constraint indices
  - Batch update realized loads
- [ ] Add fallback path for solvers without batch API:
  - Detect batch capability at runtime or compile-time
  - Fall back to individual updates if needed
- [ ] Optimize data layout for batch operations:
  - Pre-allocate buffers for constraint indices and values
  - Reuse buffers across multiple solves

### Testing

- [ ] Unit test: Batch RHS update produces same result as individual updates
- [ ] Unit test: Batch update with empty list (edge case)
- [ ] Unit test: Batch update with single constraint (should work)
- [ ] Unit test: Batch update with all constraints (stress test)
- [ ] Unit test: Fallback path works when batch API unavailable
- [ ] Integration test: Full algorithm run with batched updates
- [ ] Performance test: Benchmark realize_uncertainties before/after batching
- [ ] Performance test: Benchmark update_from_trajectory before/after batching
- [ ] Correctness test: Verify subproblem solutions identical with batching

### Documentation

- [ ] Add doc comment to batch_change_rhs explaining usage and benefits
- [ ] Document batch vs individual trade-offs (when to use each)
- [ ] Add inline comments explaining batch assembly logic
- [ ] Add performance notes to module docs explaining batch benefits
- [ ] Update CHANGELOG.md with "Performance: Batched constraint RHS updates"

## Technical Notes

### Solver API Investigation

**HiGHS C API for batch operations:**

```c
// Batch row bound changes
HighsInt Highs_changeRowsBounds(
    void* highs,
    const HighsInt num_set_entries,
    const HighsInt* set,
    const double* lower,
    const double* upper
);
```

**Rust wrapper:**

```rust
impl Model {
    pub fn batch_change_rhs(
        &mut self,
        constraints: &[usize],
        bounds: &[(f64, f64)],  // (lower, upper) for each constraint
    ) -> Result<(), String> {
        assert_eq!(constraints.len(), bounds.len());

        let n = constraints.len();
        let set: Vec<i32> = constraints.iter().map(|&c| c as i32).collect();
        let lower: Vec<f64> = bounds.iter().map(|&(l, _)| l).collect();
        let upper: Vec<f64> = bounds.iter().map(|&(_, u)| u).collect();

        unsafe {
            let status = highs_sys::Highs_changeRowsBounds(
                self.highs_ptr,
                n as i32,
                set.as_ptr(),
                lower.as_ptr(),
                upper.as_ptr(),
            );

            if status != 0 {
                return Err("Batch RHS update failed".to_string());
            }
        }

        Ok(())
    }
}
```

### Batched Update Pattern

**Before (Individual Updates):**

```rust
// O(n) API calls, O(n) FFI crossings
for hydro in 0..n_hydros {
    let constraint_idx = self.constraints.ar_dynamics[hydro];
    let innovation = innovations[hydro];
    model.change_rhs(constraint_idx, innovation, innovation)?;
}
```

**After (Batched):**

```rust
// O(1) API call, O(1) FFI crossing, O(n) work in C
let indices: Vec<usize> = self.constraints.ar_dynamics.clone();
let bounds: Vec<(f64, f64)> = innovations.iter()
    .map(|&inn| (inn, inn))
    .collect();
model.batch_change_rhs(&indices, &bounds)?;
```

### Performance Analysis

**FFI Overhead:**

- Individual update: ~100-200ns per call (FFI crossing + function dispatch)
- Batch update: ~100-200ns total + ~10ns per constraint
- For n=30 hydros: 3000-6000ns → 400-500ns
- **Speedup: 6-15x on RHS updates**

**Total Impact:**

- RHS updates are ~5-10% of realize_uncertainties time
- 6-15x speedup on 5-10% → 2-7% overall speedup
- Combined with other optimizations: **cumulative 25-35% total speedup**

### Buffer Reuse Pattern

```rust
pub struct Subproblem {
    // ... existing fields ...

    // Reusable buffers for batch operations (avoid repeated allocation)
    rhs_update_buffer_indices: Vec<usize>,
    rhs_update_buffer_bounds: Vec<(f64, f64)>,
}

impl Subproblem {
    fn update_ar_constraint_rhs_batched(
        &mut self,
        innovations: &[f64],
    ) -> Result<(), String> {
        // Reuse pre-allocated buffers
        self.rhs_update_buffer_indices.clear();
        self.rhs_update_buffer_bounds.clear();

        for hydro in 0..self.inflow_model.dimension() {
            self.rhs_update_buffer_indices.push(self.constraints.ar_dynamics[hydro]);
            self.rhs_update_buffer_bounds.push((innovations[hydro], innovations[hydro]));
        }

        let model = self.model.as_mut().unwrap();
        model.batch_change_rhs(
            &self.rhs_update_buffer_indices,
            &self.rhs_update_buffer_bounds,
        )
    }
}
```

### Batch Size Considerations

**Small batches (< 10 constraints):**

- FFI overhead dominates
- Batching still beneficial but less pronounced

**Medium batches (10-100 constraints):**

- Sweet spot for batching
- 5-15x speedup typical

**Large batches (> 100 constraints):**

- Cache effects may reduce benefit
- Still faster than individual updates

**Recommendation:** Always use batching when updating > 2 constraints.

### Fallback Strategy

**When batch API is unavailable:**

```rust
impl Model {
    pub fn batch_change_rhs(
        &mut self,
        constraints: &[usize],
        bounds: &[(f64, f64)],
    ) -> Result<(), String> {
        #[cfg(feature = "highs_batch")]
        {
            // Use native batch API
            self.batch_change_rhs_native(constraints, bounds)
        }

        #[cfg(not(feature = "highs_batch"))]
        {
            // Fallback to individual updates
            for (i, &constraint_idx) in constraints.iter().enumerate() {
                let (lower, upper) = bounds[i];
                self.change_rhs(constraint_idx, lower, upper)?;
            }
            Ok(())
        }
    }
}
```

### Edge Cases

- **Empty batch**: No-op, return Ok
- **Single constraint**: Should work, though individual update equally fast
- **Duplicate constraints**: Undefined behavior, should validate or document
- **Out-of-bounds indices**: Solver should error, propagate error
- **NaN/Inf bounds**: Validate before passing to solver

### Cache Locality Benefits

**Individual updates:**

- Random access pattern (jump between constraint indices)
- Poor cache locality

**Batched updates:**

- Sequential processing in C code
- Better cache prefetching
- Reduced cache misses

**Additional benefit:** ~10-20% from improved cache behavior (beyond FFI savings).

## Dependencies

- **Blocked by**: None (independent optimization)
- **Blocks**: None (last optimization ticket)
- **Related**: TICKET-013 (seasonal cache optimization)

## References

- `src/solver.rs` - Current solver interface
- `highs-sys` documentation - HiGHS C API bindings
- UNIFIED_AR_ROADMAP.md - Section 3.2 (Optimize constraint updates)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Performance benchmark shows 5-10% realize_uncertainties speedup
- [ ] Performance benchmark shows measurable update_from_trajectory speedup
- [ ] Correctness verified: batched = individual updates
- [ ] Fallback path tested (if applicable)
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
