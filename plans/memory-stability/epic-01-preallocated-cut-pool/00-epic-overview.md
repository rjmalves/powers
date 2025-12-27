# Epic 1: Preallocated Cut Pool

## Status: ✅ COMPLETE

## Summary

Implement full preallocation of `BendersCut` instances at training initialization. Instead of allocating new cuts during training with `fcf.add_cut(cut)`, we preallocate all cuts with their coefficient vectors upfront and update them in place using slot-based access via `(iteration, forward_pass_idx)`.

## Scope

### Included

- Slot computation utility function
- `BendersCut::update()` method for in-place coefficient updates
- `BendersCutPool::preallocate()` method for full preallocation
- Modified FCF initialization to use preallocation
- Modified `add_cuts_batch()` to use slot-based access
- Updated cut domination evaluation for preallocated pool

### Excluded

- State pool preallocation (Epic 2)
- Arc-based cut sharing (Epic 3)
- HiGHS internal memory optimization

## Dependencies

- **Requires**: None (first epic)
- **Enables**: Epic 2 (similar pattern), Epic 3 (Arc wrapping)

## Acceptance Criteria

- [x] All cuts preallocated at training start
- [x] Zero `BendersCut` allocations during `add_cuts_batch()`
- [x] Slot computation is deterministic: `slot = (iteration-1) * num_fp + fp_idx`
- [x] Cut domination evaluation works with preallocated pool
- [x] All existing lib tests pass
- [ ] Lower bounds identical to baseline on examples 01, 05 (needs validation)

## Technical Approach

### Slot Computation

The key insight is that `(iteration, forward_pass_idx)` uniquely identifies each cut:

```rust
/// Compute slot index for (iteration, forward_pass_idx) pair.
/// slot = (iteration - 1) * num_forward_passes + forward_pass_idx
#[inline]
pub fn compute_slot(iteration: usize, forward_pass_idx: usize, num_forward_passes: usize) -> usize {
    debug_assert!(iteration >= 1, "iteration must be 1-based");
    debug_assert!(forward_pass_idx < num_forward_passes);
    (iteration - 1) * num_forward_passes + forward_pass_idx
}
```

### BendersCut Updates

Add method to update cut in place without allocation:

```rust
impl BendersCut {
    /// Update cut coefficients and RHS in place (no allocation).
    pub fn update(&mut self, coefficients: &[f64], rhs: f64, iteration: usize, forward_pass_idx: usize) {
        debug_assert_eq!(self.coefficients.len(), coefficients.len());
        self.coefficients.copy_from_slice(coefficients);
        self.rhs = rhs;
        self.iteration = iteration;
        self.forward_pass_idx = forward_pass_idx;
        self.active = true;
        self.non_dominated_state_count = 1;
    }
}
```

### Pool Preallocation

```rust
impl BendersCutPool {
    pub fn preallocate(num_iterations: usize, num_forward_passes: usize, state_dimension: usize) -> Self {
        let total_cuts = num_iterations * num_forward_passes;
        let pool: Vec<BendersCut> = (0..total_cuts)
            .map(|id| BendersCut {
                id,
                coefficients: vec![0.0; state_dimension],
                rhs: 0.0,
                active: false,
                non_dominated_state_count: 0,
                iteration: 0,
                forward_pass_idx: 0,
                slot_index: None,
            })
            .collect();
        
        Self {
            pool,
            active_cut_indices: HashMap::with_capacity(total_cuts),
            total_cut_count: 0,
        }
    }
}
```

## Estimated Effort

**1 Sprint (2 weeks)** / **21 story points**

## Files to Modify

| File | Changes |
|------|---------|
| `src/cut.rs` | Add `update()`, `preallocate()` |
| `src/fcf.rs` | Update `add_cuts_batch()`, add slot computation |
| `src/sddp/mod.rs` | Modify FCF initialization |
