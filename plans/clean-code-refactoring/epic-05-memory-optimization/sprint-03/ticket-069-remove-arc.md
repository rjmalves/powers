# [T-069] Remove Arc Wrapper from BendersCutPool

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Pool Memory Model Optimization](./00-sprint-overview.md)
> **Dependencies**: Sprint 2 complete
> **Blocks**: [T-070](./ticket-070-remove-hashmap.md)

## Files to Read Before Starting

- `src/cut.rs:276-510` - `BendersCutPool` implementation
- `src/fcf.rs:180-310` - FCF cut pool usage
- `src/algorithm/coordinator.rs` - Cut sharing during Phase 3

---

## Context

### Background

`BendersCutPool` currently stores cuts as `Vec<Arc<BendersCut>>`. The Arc provides shared ownership for concurrent read access. However, with preallocated pools:
- Cuts are never deallocated during training
- Mutable access uses `Arc::get_mut()` which requires exclusive ownership
- The Arc adds overhead (reference counting)

### Current State

```rust
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    ...
}
```

Usage patterns:
- `Arc::get_mut(&mut self.pool[slot])` - Mutable update (requires single owner)
- `Arc::clone(&self.pool[id])` - Share for Phase 3 cut application

### Target State

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,  // Direct storage
    ...
}
```

---

## Specification

### Changes Required

1. Change `pool: Vec<Arc<BendersCut>>` to `pool: Vec<BendersCut>`
2. Remove `Arc::get_mut()` calls - direct mutable access
3. Update `Arc::clone()` callsites to use references
4. Phase 3 cut application: pass references instead of Arc

### Arc Usage Audit

Before implementation, audit all `Arc<BendersCut>` usages:

```bash
grep -rn "Arc<BendersCut>\|Arc::clone.*cut\|Arc::get_mut.*pool" src/
```

Expected locations:
- `src/cut.rs` - Pool storage and update
- `src/fcf.rs` - Cut access
- `src/algorithm/coordinator.rs` - Phase 3 cut sharing
- `src/subproblem.rs` - Cut application

---

## Acceptance Criteria

- [ ] `pool` field is `Vec<BendersCut>`
- [ ] No `Arc<BendersCut>` in codebase (except deprecated paths)
- [ ] All mutable access is direct (`&mut pool[slot]`)
- [ ] Phase 3 uses references instead of Arc
- [ ] All tests pass (549+)
- [ ] Golden tests pass

---

## Implementation Guide

### Step 1: Update struct definition

In `src/cut.rs`:

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,  // Changed from Vec<Arc<BendersCut>>
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
    num_forward_passes: usize,
}
```

### Step 2: Update preallocate()

```rust
pub fn preallocate(...) -> Self {
    let pool: Vec<BendersCut> = (0..total_cuts)
        .map(|id| {
            BendersCut {  // No Arc::new()
                id,
                coefficients: vec![0.0; state_dimension],
                ...
            }
        })
        .collect();
    ...
}
```

### Step 3: Update update_cut()

```rust
pub fn update_cut(...) -> usize {
    let cut = &mut self.pool[slot];  // Direct mutable access
    cut.update(coefficients, rhs, iteration, forward_pass_idx);
    ...
}
```

### Step 4: Update update_cut_and_state_slots()

```rust
pub fn update_cut_and_state_slots(...) -> usize {
    let cut = &mut self.pool[slot];  // Direct access, no Arc::get_mut
    cut.update(cut_coefficients, cut_rhs, iteration, forward_pass_idx);
    ...
}
```

### Step 5: Update FCF access

In `src/fcf.rs`, change access patterns:

```rust
// Before
let cut = Arc::clone(&self.cut_pool.pool[id]);

// After
let cut = &self.cut_pool.pool[id];  // Borrow instead of clone
```

### Step 6: Update Phase 3 cut application

The `Phase2Result` likely has:
```rust
pub cuts_to_apply: Vec<Arc<BendersCut>>,
```

Change to:
```rust
pub cut_ids_to_apply: Vec<usize>,  // Just pass IDs
```

And in `apply_cuts_parallel`, access cuts via pool reference.

### Step 7: Handle atomic fields

`BendersCut` has atomic fields (`active`, `non_dominated_state_count`, `slot_index`). These work fine without Arc - they're used for concurrent reads, not ownership.

---

## Testing Requirements

### Compile Test

```bash
cargo build -j1
```

Fix all compilation errors.

### Unit Tests

```bash
RUST_TEST_THREADS=1 cargo test -j1
```

### Golden Tests

```bash
./scripts/golden-tests.sh verify
```

---

## Pitfalls to Avoid

- ⚠️ **Borrow checker**: Removing Arc may introduce borrow conflicts. May need restructuring.
- ⚠️ **Phase 3 sharing**: Cuts must be accessible while handlers are borrowed. Use indices + pool reference.
- ⚠️ **Trait object compatibility**: Some code may expect `&dyn` patterns.

### Potential Borrow Conflicts

```rust
// This may fail:
let cut = &fcf.cut_pool.pool[id];
self.handlers.par_iter_mut().for_each(|h| {
    h.apply_cut(cut);  // Can't share &cut across threads
});

// Solution: Clone cut data or pass pool reference
let coefficients = fcf.cut_pool.pool[id].coefficients.clone();
```

Or use index-based access inside the parallel loop.

---

## Documentation Requirements

- [ ] Update doc comments on BendersCutPool
- [ ] Update Phase2Result if changed
- [ ] Add migration note to CHANGELOG

---

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Straightforward change but requires careful borrow checker management.

---

## Definition of Done

- [ ] No Arc<BendersCut> in pool
- [ ] All access is direct reference
- [ ] Tests pass
- [ ] Golden tests pass
- [ ] No performance regression
