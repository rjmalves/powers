# [T-063] Add update_from_staging() to Pools

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Handler Staging Buffers](./00-sprint-overview.md)
> **Dependencies**: [T-060](./ticket-060-create-staging-buffer.md)
> **Blocks**: [T-064](./ticket-064-parallel-then-sequential.md)

## Files to Read Before Starting

- `src/cut.rs:473-508` - `update_cut_and_state_slots()` (existing direct update)
- `src/state.rs:579-591` - `VisitedStatePool::update_state()`
- `src/memory/buffers.rs` - `CutStagingBuffer` struct

---

## Context

### Background

After parallel cut computation, each handler's staging buffer contains a computed cut and state. The pools need a method to copy from staging buffers efficiently.

### Current State

`update_cut_and_state_slots()` takes individual slices:
```rust
pub fn update_cut_and_state_slots(
    &mut self,
    iteration: usize,
    forward_pass_idx: usize,
    cut_coefficients: &[f64],
    cut_rhs: f64,
    state_coefficients: &[f64],
    state_pool: &mut VisitedStatePool,
) -> usize
```

### Target State

Add convenience method that takes a staging buffer:
```rust
pub fn update_from_staging(
    &mut self,
    staging: &CutStagingBuffer,
    state_pool: &mut VisitedStatePool,
) -> usize
```

---

## Specification

### New Method on BendersCutPool

```rust
/// Update cut and state slots from a staging buffer.
///
/// Convenience method for the parallel-then-sequential pattern.
/// Delegates to `update_cut_and_state_slots()`.
///
/// # Arguments
///
/// * `staging` - Staging buffer with computed cut and state
/// * `state_pool` - Mutable reference to state pool
///
/// # Returns
///
/// The slot index that was updated.
///
/// # Panics
///
/// Panics if staging buffer is not populated.
#[inline]
pub fn update_from_staging(
    &mut self,
    staging: &CutStagingBuffer,
    state_pool: &mut VisitedStatePool,
) -> usize {
    debug_assert!(staging.populated, "Cannot update from unpopulated staging buffer");
    
    self.update_cut_and_state_slots(
        staging.iteration,
        staging.forward_pass_idx,
        &staging.cut_coefficients,
        staging.cut_rhs,
        &staging.state_coefficients,
        state_pool,
    )
}
```

---

## Acceptance Criteria

- [ ] Method `update_from_staging()` added to `BendersCutPool`
- [ ] Method delegates to existing `update_cut_and_state_slots()`
- [ ] Panics if staging buffer not populated (debug mode)
- [ ] All existing tests pass

---

## Implementation Guide

### Step 1: Add import

In `src/cut.rs`:
```rust
use crate::memory::CutStagingBuffer;
```

### Step 2: Add method

Add to `impl BendersCutPool` (after `update_cut_and_state_slots`):

```rust
/// Update cut and state slots from a staging buffer.
///
/// Convenience method for the parallel-then-sequential pattern.
/// Delegates to `update_cut_and_state_slots()`.
///
/// # Arguments
///
/// * `staging` - Staging buffer with computed cut and state
/// * `state_pool` - Mutable reference to state pool
///
/// # Returns
///
/// The slot index that was updated.
///
/// # Panics
///
/// Panics in debug mode if staging buffer is not populated.
#[inline]
pub fn update_from_staging(
    &mut self,
    staging: &CutStagingBuffer,
    state_pool: &mut crate::state::VisitedStatePool,
) -> usize {
    debug_assert!(
        staging.populated,
        "Cannot update from unpopulated staging buffer"
    );
    
    self.update_cut_and_state_slots(
        staging.iteration,
        staging.forward_pass_idx,
        &staging.cut_coefficients,
        staging.cut_rhs,
        &staging.state_coefficients,
        state_pool,
    )
}
```

---

## Testing Requirements

### Unit Tests

Add to `src/cut.rs` tests module:

```rust
#[test]
fn test_update_from_staging() {
    use crate::memory::CutStagingBuffer;
    use crate::state::VisitedStatePool;
    
    let mut cut_pool = BendersCutPool::preallocate(2, 4, 3); // 2 iters, 4 fps, 3-dim
    
    // Create template state for state pool
    let template = create_test_state(3);
    let mut state_pool = VisitedStatePool::preallocate(2, 4, &*template);
    
    // Create and populate staging buffer
    let mut staging = CutStagingBuffer::new(3);
    staging.cut_coefficients[..3].copy_from_slice(&[1.0, 2.0, 3.0]);
    staging.state_coefficients[..3].copy_from_slice(&[4.0, 5.0, 6.0]);
    staging.cut_rhs = 42.0;
    staging.iteration = 1;
    staging.forward_pass_idx = 2;
    staging.populated = true;
    
    // Update from staging
    let slot = cut_pool.update_from_staging(&staging, &mut state_pool);
    
    // Verify slot
    assert_eq!(slot, compute_slot(1, 2, 4)); // = 2
    
    // Verify cut updated
    let cut = &cut_pool.pool[slot];
    assert_eq!(cut.coefficients, vec![1.0, 2.0, 3.0]);
    assert_eq!(cut.rhs, 42.0);
    
    // Verify state updated
    let state = &state_pool.pool[slot];
    assert_eq!(state.coefficients(), &[4.0, 5.0, 6.0]);
}

#[test]
#[should_panic(expected = "unpopulated")]
fn test_update_from_staging_panics_on_unpopulated() {
    let mut cut_pool = BendersCutPool::preallocate(1, 1, 3);
    let template = create_test_state(3);
    let mut state_pool = VisitedStatePool::preallocate(1, 1, &*template);
    
    let staging = CutStagingBuffer::new(3);  // Not populated!
    
    cut_pool.update_from_staging(&staging, &mut state_pool);  // Should panic
}
```

---

## Pitfalls to Avoid

- ⚠️ **Dimension mismatch**: Staging buffer may have larger capacity than actual data. The underlying `update_cut_and_state_slots` uses the full slice. May need to track actual lengths.
- ⚠️ **Circular import**: `CutStagingBuffer` is in `memory` module. Ensure no circular dependency with `cut` module.

### Handling Variable-Length Data

If staging buffer capacity > actual data length, add length tracking:

```rust
pub struct CutStagingBuffer {
    // ... existing fields ...
    pub actual_cut_len: usize,
    pub actual_state_len: usize,
}

// In update_from_staging:
self.update_cut_and_state_slots(
    staging.iteration,
    staging.forward_pass_idx,
    &staging.cut_coefficients[..staging.actual_cut_len],
    staging.cut_rhs,
    &staging.state_coefficients[..staging.actual_state_len],
    state_pool,
)
```

This may require updating T-060 to add these fields.

---

## Documentation Requirements

- [ ] Doc comment on method
- [ ] Reference parallel-then-sequential pattern

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Simple wrapper method. Main complexity is ensuring dimension handling is correct.

---

## Definition of Done

- [ ] Method implemented
- [ ] Unit tests pass
- [ ] Debug panic on unpopulated buffer
- [ ] All existing tests pass (549+)
- [ ] Doc comments complete
