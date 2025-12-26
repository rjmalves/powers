# Epic 3: HashMap Cloning Elimination

## Status

**Status**: ✅ Complete (2025-12-26)
**Priority**: LOW (eliminates ~120 MB transient allocations)

## Summary

~~Eliminate the cloning of `active_cut_indices` HashMap during cut selection by using a more efficient snapshot mechanism.~~

**Resolution**: The `active_cut_indices_before` HashMap was discovered to be **completely unused** (the parameter was prefixed with `_`). The clone has been removed entirely.

## Problem Statement

During backward pass, the active cut indices HashMap is cloned to track which cuts were active before batch processing:

```rust
// src/sddp/mod.rs:2077-2091
let active_cut_indices_before: HashMap<usize, usize> = {
    let fcf_locked = parent_fcf_node.data.lock().unwrap();
    fcf_locked.cut_pool.active_cut_indices.clone()  // HashMap CLONE
};
```

**Impact**: HashMap with ~5000 entries × ~50 bytes = ~250 KB per clone × 59 stages × 8 iterations = ~118 MB

## Scope

### Included

- Analyze what information from `active_cut_indices_before` is actually used
- Replace HashMap clone with minimal snapshot (likely just keys)
- Optimize comparison logic

### Excluded

- Changing HashMap data structure itself
- Removing active_cut_indices tracking

## Dependencies

- **Requires**: None (independent of Epics 1-2)
- **Enables**: Further memory optimization

## Acceptance Criteria

- [ ] No full HashMap clone per stage
- [ ] Equivalent functionality preserved
- [ ] Reduced memory churn
- [ ] Examples produce identical results

## Technical Approach

### Investigation: What's actually used?

The cloned HashMap is used to detect which cuts are new vs returning:

```rust
// Likely usage pattern:
if !active_cut_indices_before.contains_key(&cut_id) {
    // This is a new cut
}
```

If only keys are checked, we can use `Vec<usize>` instead of HashMap clone.

### Solution: Snapshot Keys Only

```rust
// Before:
let active_cut_indices_before: HashMap<usize, usize> = 
    fcf_locked.cut_pool.active_cut_indices.clone();

// After:
let active_cut_ids_before: Vec<usize> = 
    fcf_locked.cut_pool.active_cut_indices.keys().copied().collect();

// Usage:
if !active_cut_ids_before.contains(&cut_id) {
    // New cut
}
```

Or use `HashSet<usize>` for O(1) lookup if needed.

## Estimated Effort

- **Sprint 1**: 2-3 days
  - Ticket 1: Analyze HashMap usage and implement snapshot (2 points)

## Key Files

| File | Impact |
|------|--------|
| `src/sddp/mod.rs` | Clone site and usage |
