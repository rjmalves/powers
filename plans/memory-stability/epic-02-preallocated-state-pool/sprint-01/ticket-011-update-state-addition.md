# [TICKET-011] Update state addition to use slot-based access

> **Epic**: [Epic 2: Preallocated State Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-010](./ticket-010-implement-statepool-preallocate.md)
> **Blocks**: Epic 3 tickets

## Context

### Background

The `add_cuts_batch()` method currently allocates new states with `add_state()`. With preallocation, we need to update existing states in place using the same slot-based access pattern as cuts.

### Relation to Epic

Final integration that enables zero state allocation during training.

### Current State

```rust
// src/fcf.rs:301
self.add_state(state);  // Calls pool.push() - allocates!
```

## Files to Read Before Starting

- `src/fcf.rs` - `add_cuts_batch()` and `add_state()` methods
- `src/state.rs` - VisitedStatePool with new `update_state()` method
- [TICKET-005](../epic-01-preallocated-cut-pool/sprint-01/ticket-005-update-add-cuts-batch.md) - Cut slot-based access (similar pattern)

## Specification

### Changes to add_cuts_batch()

```rust
for pair in cut_state_pairs.into_iter() {
    let iteration = pair.cut.iteration;
    let forward_pass_idx = pair.cut.forward_pass_idx;
    
    // Compute slot (same for cut and state)
    let slot = compute_slot(iteration, forward_pass_idx, self.num_forward_passes);
    
    // Update preallocated cut (existing from TICKET-005)
    self.cut_pool.pool[slot].update(&pair.cut.coefficients, pair.cut.rhs, iteration, forward_pass_idx);
    
    // Update preallocated state IN PLACE (no allocation!)
    self.state_pool.update_state(
        slot,
        pair.state.coefficients(),
        iteration,
        forward_pass_idx,
    );
    
    // Domination evaluation uses the preallocated state reference
    let state = &mut self.state_pool.pool[slot];
    
    // ... rest of domination evaluation
}
```

### Behavior Changes

**Before**: Clone state into new Box, push to pool
**After**: Copy coefficients into preallocated slot

### State Metadata Update

The preallocated state needs its domination info updated:

```rust
// After updating state coefficients
let state = &mut self.state_pool.pool[slot];

// Update domination from source cut
let cut = &self.cut_pool.pool[slot];
let cut_height = cut.eval_height_at_state(state.coefficients());
state.set_dominating_cut_id(slot); // Cut ID == slot
state.set_dominating_objective(cut_height);
```

## Acceptance Criteria

- [ ] No `add_state()` calls in `add_cuts_batch()`
- [ ] States updated via slot-based access
- [ ] State domination info correctly updated
- [ ] All existing tests pass
- [ ] Lower bounds identical to baseline

## Implementation Guide

### Suggested Approach

1. Remove `self.add_state(state)` call
2. Add slot-based state update using `state_pool.update_state()`
3. Update state domination info from source cut
4. Ensure domination evaluation uses preallocated state

### Key Files to Modify

- `src/fcf.rs`: Modify `add_cuts_batch()` method

### Patterns to Follow

- See TICKET-005 for cut slot-based access pattern

### Pitfalls to Avoid

- ⚠️ State and cut share the same slot (same (iteration, fp_idx))
- ⚠️ Update domination info after coefficient update
- ⚠️ Use `state_pool.pool[slot]` reference for domination, not pair.state

## Testing Requirements

### Unit Tests

- [ ] Test state is updated at correct slot
- [ ] Test state coefficients match input
- [ ] Test state domination info set correctly
- [ ] Test no allocation during batch processing

### Integration Tests

- [ ] Run example-01 and verify identical results
- [ ] Run example-05 and verify identical results

### Performance Tests

- [ ] Memory profiling shows flat state pool memory

## Documentation Requirements

- [ ] Update `add_cuts_batch()` doc comment
- [ ] Note slot sharing between cut and state

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Integration with existing domination logic needs care
