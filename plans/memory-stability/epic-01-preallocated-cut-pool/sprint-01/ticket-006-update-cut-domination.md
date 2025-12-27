# [TICKET-006] Update cut domination for preallocated pool

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-005](./ticket-005-update-add-cuts-batch.md)
> **Blocks**: Epic 2 tickets

## Context

### Background

Cut domination evaluation currently works with dynamically growing pools. With preallocation, cuts may exist in the pool but be inactive (not yet populated). The domination logic needs to handle this.

### Relation to Epic

Ensures cut selection algorithm works correctly with preallocated pools.

### Current State

`eval_new_cut_domination()` and `update_old_cuts_domination()` iterate over all cuts in the pool, checking `active` status:

```rust
pub fn eval_new_cut_domination(&mut self, new_cut: &mut cut::BendersCut) {
    for state in self.state_pool.pool.iter_mut() {
        // ... evaluate domination
    }
}

pub fn update_old_cuts_domination(&mut self, new_state: &mut Box<dyn state::State>) -> Vec<usize> {
    for old_cut in self.cut_pool.pool.iter_mut() {
        match old_cut.active {
            true => continue,  // Skip active cuts
            false => { /* check if should return to model */ }
        }
    }
}
```

## Files to Read Before Starting

- `src/fcf.rs` - `eval_new_cut_domination()` and `update_old_cuts_domination()`
- `src/cut.rs` - BendersCut structure with `active` field

## Specification

### Behavior Changes

1. **Preallocated cuts start inactive**: All cuts have `active = false` and `non_dominated_state_count = 0` initially

2. **After update, cuts become active**: `update_cut()` sets `active = true`

3. **Domination evaluation must skip unpopulated cuts**: 
   - Unpopulated cuts have `iteration = 0` and `forward_pass_idx = 0`
   - Or check `non_dominated_state_count = 0` and `rhs = 0.0`
   - Safest: track `populated: bool` field

### Option A: Add `populated` field (Recommended)

```rust
pub struct BendersCut {
    // ... existing fields
    pub populated: bool,  // NEW: true after first update
}
```

### Option B: Check iteration > 0

```rust
fn is_populated(&self) -> bool {
    self.iteration > 0
}
```

### Domination Logic Changes

```rust
pub fn update_old_cuts_domination(&mut self, ...) {
    for old_cut in self.cut_pool.pool.iter_mut() {
        // Skip unpopulated preallocated cuts
        if !old_cut.is_populated() {
            continue;
        }
        
        match old_cut.active {
            true => continue,
            false => { /* existing logic */ }
        }
    }
}
```

## Acceptance Criteria

- [ ] Unpopulated preallocated cuts are skipped in domination evaluation
- [ ] Populated cuts work identically to before
- [ ] No performance regression from iterating over preallocated cuts
- [ ] All existing cut selection tests pass
- [ ] Lower bounds identical to baseline

## Implementation Guide

### Suggested Approach

1. Add `populated: bool` field to `BendersCut` (default `false`)
2. Set `populated = true` in `BendersCut::update()`
3. Add `is_populated()` helper method
4. Update domination methods to check `is_populated()` first

### Key Files to Modify

- `src/cut.rs`: Add `populated` field and `is_populated()`
- `src/fcf.rs`: Update domination methods to skip unpopulated cuts

### Pitfalls to Avoid

- ⚠️ Don't break existing tests that use `BendersCut::new()` directly
- ⚠️ Ensure `populated = true` after `new()` for backward compatibility
- ⚠️ Consider iteration performance with many preallocated cuts

## Testing Requirements

### Unit Tests

- [ ] Test preallocated cuts have `populated = false`
- [ ] Test `update()` sets `populated = true`
- [ ] Test `new()` sets `populated = true` (backward compat)
- [ ] Test domination skips unpopulated cuts
- [ ] Test domination works correctly for populated cuts

### Integration Tests

- [ ] Run example-01 with cut selection enabled
- [ ] Run example-05 with cut selection enabled
- [ ] Verify cut counts match baseline

## Documentation Requirements

- [ ] Document `populated` field purpose
- [ ] Update domination method doc comments

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Well-defined change, clear testing strategy
