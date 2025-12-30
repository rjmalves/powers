# [T-082] Implement HiGHS Solver Warmup After Preallocation

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: [T-080](./ticket-080-audit-add-row.md)
> **Blocks**: [T-085](./ticket-085-dhat-profiling.md)
> **Status**: ✅ Complete

---

## Context

### Background

HiGHS allocates internal work vectors lazily during the first `Highs_run()` call:

- LU factorization storage: O(nnz × fill_factor)
- Work vectors for pricing: O(num_cols + num_rows)
- Steepest-edge weights: O(num_cols)
- Row/column pricing vectors: O(n)

These allocations happen on the **first solve** after model creation, causing memory growth during the initial training iterations.

### HiGHS Internal Allocation Points

From HiGHS source (`highs/presolve/HPresolve.cpp` and simplex implementation):

```cpp
// In HPresolve::okFromCSR
Avalue = ARval;
if (!okReserve(Acol, nnz)) return false;
if (!okReserve(Arow, nnz)) return false;
if (!okResize(Anext, nnz)) return false;
if (!okResize(Aprev, nnz)) return false;
```

### Current State

- Model created with `preallocate_cut_constraints()`
- First training iteration triggers HiGHS internal allocations
- Memory grows during early iterations, stabilizes later

### Target State

- After `preallocate_cut_constraints()`, call `warmup_solver()`
- Warmup solve allocates all HiGHS internal structures
- Training iterations see stable memory from first iteration

---

## Specification

### Inputs

- Subproblem with preallocated cut constraints
- Model ready for solving

### Outputs

- HiGHS internal work vectors pre-allocated
- Basis factorization initialized
- Solver state cleared (ready for fresh solve)

### Behavior

```rust
impl Subproblem {
    /// Warm up the HiGHS solver to pre-allocate internal work vectors.
    ///
    /// This should be called after `preallocate_cut_constraints()` but before
    /// training iterations. It performs a single solve with trivial bounds to
    /// force HiGHS to allocate all internal data structures.
    ///
    /// # Memory Effect
    ///
    /// After warmup, subsequent `Highs_run()` calls should not allocate.
    ///
    /// # Solver State
    ///
    /// Calls `clear_solver()` after warmup to reset solution state while
    /// preserving the allocated internal structures.
    pub fn warmup_solver(&mut self) -> Result<(), String> {
        let model = self.model.as_mut().ok_or("Model not initialized")?;
        
        // Solve once to allocate internal structures
        model.try_solve().map_err(|e| format!("Warmup solve failed: {:?}", e))?;
        
        // Clear solution state but preserve internal allocations
        model.clear_solver();
        
        Ok(())
    }
}
```

### Integration Point

Call warmup after preallocation in the training setup:

```rust
// In SDDP training initialization
for subproblem in subproblems.iter_mut() {
    subproblem.preallocate_cut_constraints(max_cuts, num_forward_passes)?;
    subproblem.warmup_solver()?;
}
```

### Error Handling

- If warmup solve fails (infeasible/unbounded), log warning but continue
- Warmup failure is non-fatal (just means first training solve will allocate)

---

## Acceptance Criteria

- [x] `warmup_solver()` method implemented on `Subproblem`
- [x] Warmup called after preallocation in SDDP training setup
- [x] `clear_solver()` called after warmup to reset state
- [ ] Memory profile shows allocations during warmup, not during training (requires manual verification)
- [x] All tests pass
- [x] No numerical result changes

### Implementation Notes

Added `warmup_solver()` to `Subproblem` which calls `try_solve()` then `clear_solver()`.
Added `warmup_solvers()` to `SddpTrainHandler` which warms up all subproblems.
Integrated warmup into training loop after `preallocate_cut_constraints()` loop.

---

## Implementation Guide

### Suggested Approach

1. **Add `warmup_solver()` method** to `Subproblem`:
   ```rust
   pub fn warmup_solver(&mut self) -> Result<(), String> {
       if let Some(model) = self.model.as_mut() {
           // Solve may fail for various reasons; that's OK
           let _ = model.try_solve();
           model.clear_solver();
       }
       Ok(())
   }
   ```

2. **Find the preallocation call site** in SDDP builder/training setup

3. **Add warmup call** immediately after preallocation:
   ```rust
   subproblem.preallocate_cut_constraints(max_cuts, num_fp)?;
   subproblem.warmup_solver()?;
   ```

4. **Consider parallel warmup** if many subproblems:
   ```rust
   subproblems.par_iter_mut().for_each(|sp| {
       let _ = sp.warmup_solver();
   });
   ```

### Key Files to Modify

- `src/subproblem.rs`: Add `warmup_solver()` method
- `src/sddp/mod.rs` or `src/sddp/builder.rs`: Call warmup after preallocation

### Patterns to Follow

- See `clear_solver()` in `solver.rs:814-816` for state clearing
- Follow error handling pattern from `preallocate_cut_constraints()`

### Pitfalls to Avoid

- ⚠️ Don't panic on warmup solve failure (model may be trivially infeasible with placeholder bounds)
- ⚠️ Must call `clear_solver()` to reset solution state
- ⚠️ Warmup adds startup time but eliminates runtime allocation

---

## Testing Requirements

### Unit Tests

- [ ] Test `warmup_solver()` on fresh subproblem
- [ ] Test warmup doesn't affect subsequent solve correctness
- [ ] Test warmup on subproblem without preallocation (should still work)

### Integration Tests

- [ ] Run example 05 with warmup enabled
- [ ] Verify memory profile: allocations during init/warmup, none during training

### Performance Tests

- [ ] Benchmark with/without warmup to quantify startup cost
- [ ] DHAT profile shows HiGHS allocations during warmup phase only

---

## Documentation Requirements

- [ ] Document `warmup_solver()` with rationale for why it exists
- [ ] Add performance note about startup cost vs. runtime benefit

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Simple method addition with clear integration point.

---

## Definition of Done

- [ ] `warmup_solver()` implemented and documented
- [ ] Warmup integrated into training initialization
- [ ] Memory profile confirms allocations shifted to warmup phase
- [ ] All tests passing
- [ ] Code reviewed and merged
