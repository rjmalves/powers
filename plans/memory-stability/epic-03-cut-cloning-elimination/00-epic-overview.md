# Epic 3: Cut Cloning Elimination

## Summary

Eliminate transient memory allocations from cut cloning when passing cuts to parallel handlers. Replace `Vec<BendersCut>` with `Vec<Arc<BendersCut>>` in the cut pool to enable cheap cloning via `Arc::clone()` instead of full data cloning.

## Scope

### Included

- Change `BendersCutPool::pool` from `Vec<BendersCut>` to `Vec<Arc<BendersCut>>`
- Update all cut pool access patterns for Arc
- Modify handler cut application to clone Arc instead of data
- Update cut domination for Arc-based access

### Excluded

- Cut pool preallocation (Epic 1 - already complete)
- State pool preallocation (Epic 2 - already complete)
- Handler architecture changes

## Dependencies

- **Requires**: Epic 1 (preallocated cut pool), Epic 2 (preallocated state pool)
- **Enables**: Epic 4 (final validation)

## Acceptance Criteria

- [ ] Cut pool uses `Vec<Arc<BendersCut>>`
- [ ] Handler cut application uses `Arc::clone()` (~16 bytes vs ~1 KB)
- [ ] Cut mutation works correctly with Arc (interior mutability or copy-on-write)
- [ ] ~80 MB transient allocation eliminated
- [ ] All existing tests pass
- [ ] Lower bounds identical to baseline

## Technical Approach

### Challenge: Arc and Mutability

Cuts need to be mutated for:
- `non_dominated_state_count` updates during domination
- `active` flag changes during cut selection

Options:
1. **Arc<RwLock<BendersCut>>**: Thread-safe interior mutability
2. **Arc with atomic counters**: Use `AtomicUsize` for counter, `AtomicBool` for active
3. **Clone-on-write**: Clone Arc content when mutation needed (defeats purpose)

**Recommended: Option 2 (Atomic fields)**

```rust
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: AtomicBool,
    pub non_dominated_state_count: AtomicUsize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub slot_index: Option<usize>,
    pub populated: bool,
}
```

### Handler Application Changes

```rust
// Current (clones data)
let cuts: Vec<(usize, BendersCut)> = cut_ids
    .iter()
    .filter_map(|&id| pool.get(id).map(|cut| (id, cut.clone())))
    .collect();

// New (clones Arc)
let cuts: Vec<(usize, Arc<BendersCut>)> = cut_ids
    .iter()
    .filter_map(|&id| pool.get(id).map(|cut| (id, Arc::clone(cut))))
    .collect();
```

## Estimated Effort

**1 Sprint (3 days)** / **8 story points**

## Files to Modify

| File | Changes |
|------|---------|
| `src/cut.rs` | Add atomic fields, update BendersCutPool to use Arc |
| `src/fcf.rs` | Update cut access patterns for Arc |
| `src/sddp/mod.rs` | Update handler cut application |
