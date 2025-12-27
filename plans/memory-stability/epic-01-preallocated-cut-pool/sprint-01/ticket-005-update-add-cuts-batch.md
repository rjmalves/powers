# [TICKET-005] Update add_cuts_batch() to use slot-based access

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-004](./ticket-004-modify-fcf-initialization.md)
> **Blocks**: [TICKET-006](./ticket-006-update-cut-domination.md)

## Context

### Background

The `add_cuts_batch()` method currently allocates new cuts and pushes them to the pool. With preallocation, we need to update existing cuts in place using slot-based access.

### Relation to Epic

This is the critical change that eliminates allocations during the training loop.

### Current State

```rust
// src/fcf.rs:278-301
for pair in cut_state_pairs.into_iter() {
    let mut cut = pair.cut;
    let mut state = pair.state;

    // Assign ID and add to pool
    cut.id = self.cut_pool.total_cut_count;
    new_cut_ids.insert(cut.id);
    self.update_cut_pool_on_add(cut.id);

    // ...
    self.add_cut(cut);  // ALLOCATION HERE
    self.add_state(state);  // ALLOCATION HERE (Epic 2)
}
```

## Files to Read Before Starting

- `src/fcf.rs` - `add_cuts_batch()` implementation
- `src/cut.rs` - BendersCutPool with new `update_cut()` method
- `MEMORY_STABILITY_ANALYSIS.md` - Integration Points section

## Specification

### Inputs

- `cut_state_pairs: Vec<CutStatePair>` - Pairs containing cut data and states
- `enable_cut_selection: bool` - Whether to enable cut selection

Each `CutStatePair` contains:
- `cut.iteration: usize` - Iteration number (1-based)
- `cut.forward_pass_idx: usize` - Forward pass index (0-based)
- `cut.coefficients: Vec<f64>` - Cut coefficients
- `cut.rhs: f64` - Cut RHS value

### Behavior Changes

**Before**: Create new cut, push to pool
**After**: Compute slot from (iteration, forward_pass_idx), update preallocated cut in place

```rust
for pair in cut_state_pairs.into_iter() {
    let iteration = pair.cut.iteration;
    let forward_pass_idx = pair.cut.forward_pass_idx;
    
    // Update preallocated cut in place (no allocation!)
    let cut_slot = self.cut_pool.update_cut(
        iteration,
        forward_pass_idx,
        &pair.cut.coefficients,
        pair.cut.rhs,
    );
    
    new_cut_ids.insert(cut_slot);
    self.update_cut_pool_on_add(cut_slot);
    
    // ... rest of domination evaluation
}
```

### Error Handling

- Slot must be within preallocated bounds
- debug_assert for out-of-bounds access

## Acceptance Criteria

- [ ] No `add_cut()` calls in `add_cuts_batch()`
- [ ] Cuts updated via slot-based access
- [ ] Cut IDs now equal slot indices
- [ ] All existing tests pass
- [ ] Lower bounds identical to baseline

## Implementation Guide

### Suggested Approach

1. Modify the loop in `add_cuts_batch()`:
   - Extract iteration and forward_pass_idx from pair.cut
   - Call `self.cut_pool.update_cut()` instead of `self.add_cut()`
   - Use returned slot as cut_id

2. Update `update_cut_pool_on_add()` to work with slot-based IDs

3. Ensure cut domination evaluation gets correct cut reference

### Key Files to Modify

- `src/fcf.rs`: Modify `add_cuts_batch()` method

### Patterns to Follow

- See `update_cut()` in `BendersCutPool` for slot computation

### Pitfalls to Avoid

- ⚠️ Don't forget that cut.id is now the slot index
- ⚠️ Ensure pair.cut has correct iteration/forward_pass_idx set before call
- ⚠️ State addition still uses push() (addressed in Epic 2)

## Testing Requirements

### Unit Tests

- [ ] Test slot-based cut update in add_cuts_batch
- [ ] Test cut IDs match slot indices
- [ ] Test multiple batches don't reallocate

### Integration Tests

- [ ] Run example-01 and verify identical results
- [ ] Run example-05 and verify identical results

### Performance Tests

- [ ] Measure memory usage during training (should be flat)

## Documentation Requirements

- [ ] Update `add_cuts_batch()` doc comment
- [ ] Note that cut.iteration and cut.forward_pass_idx must be set before call

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Core logic change affecting cut management, needs careful testing
