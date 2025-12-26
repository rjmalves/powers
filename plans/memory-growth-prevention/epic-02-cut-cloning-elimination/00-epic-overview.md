# Epic 2: Cut Cloning Elimination

## Status

**Status**: ⬜ Not Started  
**Priority**: MEDIUM (eliminates ~80 MB transient allocations)

## Summary

Eliminate the cloning of `BendersCut` structs during batch processing by using `Arc<BendersCut>` for shared ownership. Currently cuts are cloned for lock-free handler application, but this creates ~10 MB of transient allocations per iteration.

## Problem Statement

During backward pass batch processing, cuts are cloned to allow lock-free application to handlers:

```rust
// src/sddp/mod.rs:2184-2198
let cuts: Vec<(usize, crate::cut::BendersCut)> = aggregated_result
    .new_cut_ids
    .iter()
    .chain(aggregated_result.returning_cut_ids.iter())
    .filter_map(|&cut_id| {
        fcf_locked.cut_pool.pool.get(cut_id)
            .map(|cut| (cut_id, cut.clone()))  // CLONE per cut
    })
    .collect();
```

**Impact**: ~128 cuts × 59 stages × ~1.3 KB = ~9.7 MB cloned per iteration × 8 iterations = ~78 MB

## Scope

### Included

- Change `BendersCutPool` to store `Arc<BendersCut>`
- Update cut creation to wrap in Arc
- Update cloning sites to use `Arc::clone()` (cheap pointer copy)
- Update all consumers of cuts to work with Arc

### Excluded

- Changing cut structure itself
- Removing cuts from pool (dominated cuts become inactive, not removed)
- State pool changes (separate concern)

## Dependencies

- **Requires**: None (independent of Epic 1)
- **Enables**: Further memory optimization

## Acceptance Criteria

- [ ] `BendersCutPool` stores `Vec<Arc<BendersCut>>`
- [ ] Cut cloning sites use `Arc::clone()` instead of `.clone()`
- [ ] No change to cut behavior or values
- [ ] Examples produce identical results
- [ ] Reduced memory churn per iteration

## Technical Approach

### Change BendersCutPool Storage

```rust
// Before:
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    // ...
}

// After:
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    // ...
}
```

### Update Cut Creation

```rust
// Before:
fcf.add_cut(BendersCut::new(...));

// After:
fcf.add_cut(Arc::new(BendersCut::new(...)));
```

### Update Cloning Sites

```rust
// Before:
.map(|cut| (cut_id, cut.clone()))  // Clones entire struct

// After:
.map(|cut| (cut_id, Arc::clone(cut)))  // Only clones Arc pointer
```

### Consumer Updates

Places that receive cuts need to handle `Arc<BendersCut>`:
- Read-only access: `&*cut` or `cut.as_ref()`
- Most uses are read-only (eval_height, get coefficients)

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Arc overhead affects performance | Low | Arc::clone is 2 atomic ops, trivial vs cloning 1KB |
| Interior mutability needed | Medium | Check for mutable cut access patterns |
| API churn | Medium | Many files may need updates |

## Estimated Effort

- **Sprint 1**: 3-5 days
  - Ticket 1: Update BendersCutPool to use Arc (2 points)
  - Ticket 2: Update all cut consumers (3 points)

## Key Files

| File | Impact |
|------|--------|
| `src/cut.rs` | BendersCutPool struct |
| `src/fcf.rs` | add_cut method |
| `src/sddp/mod.rs` | Cut cloning and usage |
| `src/subproblem.rs` | Cut coefficient access |

## Investigation Needed

Before implementation, audit:
1. Places where `BendersCut` is mutated after creation
2. Places where cuts are cloned
3. Impact on `eval_height_at_state` and coefficient access patterns
