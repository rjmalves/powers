# [T-070] Remove HashMap from BendersCutPool

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Pool Memory Model Optimization](./00-sprint-overview.md)
> **Dependencies**: [T-069](./ticket-069-remove-arc.md)
> **Blocks**: [T-073](./ticket-073-cleanup-deprecated.md)

## Files to Read Before Starting

- `src/cut.rs:276-320` - `BendersCutPool` with HashMap
- `src/fcf.rs:268-302` - HashMap usage (add/remove/lookup)
- `src/subproblem.rs:1120-1180` - `slot_to_row()` conversion

---

## Context

### Background

`BendersCutPool.active_cut_indices` is a `HashMap<usize, usize>` mapping `cut_id → model_constraint_index`. This was needed before preallocation to track where cuts were in the solver model.

With preallocation:
- Model rows are preallocated at indices `first_preallocated_cut_row + slot`
- `slot` is stored in `cut.slot_index`
- The mapping is deterministic: no HashMap needed

### Current State

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_indices: HashMap<usize, usize>,  // cut_id → constraint_index
    ...
}
```

Usage:
- `update_cut_pool_on_add()`: Inserts into HashMap
- `update_cut_pool_on_remove()`: Removes + shifts indices
- `get_active_cut_index_by_id()`: O(1) lookup

### Target State

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    // No HashMap - use cut.slot_index directly
    ...
}
```

---

## Specification

### Why HashMap is Unnecessary

With preallocation:
1. Each cut has a deterministic `slot_index` from `(iteration, forward_pass_idx)`
2. Model row = `first_preallocated_cut_row + slot_index`
3. `slot_index` is stored in the cut itself

Therefore:
```rust
fn get_model_row_for_cut(cut: &BendersCut, first_row: usize) -> usize {
    first_row + cut.get_slot_index().unwrap()
}
```

### What About Non-Preallocated Path?

The non-preallocated path (dynamic cut addition) still needs some tracking. Options:
1. **Remove entirely**: Force preallocation (recommended)
2. **Keep separate**: Maintain HashMap only for non-preallocated mode
3. **Deprecate**: Mark non-preallocated path as deprecated

Recommendation: Option 1 - Force preallocation. The infrastructure is mature.

---

## Acceptance Criteria

- [ ] `active_cut_indices` HashMap removed from `BendersCutPool`
- [ ] All lookups use `cut.slot_index` directly
- [ ] `update_cut_pool_on_add/remove` simplified or removed
- [ ] No index shifting logic (cuts never actually removed from model)
- [ ] All tests pass
- [ ] Golden tests pass

---

## Implementation Guide

### Step 1: Remove HashMap field

In `src/cut.rs`:

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub total_cut_count: usize,
    num_forward_passes: usize,
    // REMOVED: active_cut_indices: HashMap<usize, usize>,
}
```

### Step 2: Update constructors

```rust
pub fn preallocate(...) -> Self {
    Self {
        pool,
        total_cut_count: 0,
        num_forward_passes,
        // No HashMap
    }
}
```

### Step 3: Remove/simplify FCF methods

In `src/fcf.rs`:

```rust
// REMOVE OR SIMPLIFY:
pub fn update_cut_pool_on_add(&mut self, cut_id: usize) {
    // With preallocation, cut is already in place
    // Just update total_cut_count if needed
    self.cut_pool.total_cut_count = self.cut_pool.total_cut_count.max(cut_id + 1);
}

// REMOVE:
pub fn get_active_cut_index_by_id(&self, cut_id: usize) -> usize {
    // No longer needed - use cut.slot_index directly
}

pub fn update_cut_pool_on_remove(&mut self, cut_id: usize) {
    // With preallocation, we don't remove cuts from model
    // Just deactivate via bound relaxation
    self.cut_pool.pool[cut_id].set_active(false);
    // No index shifting needed
}
```

### Step 4: Update subproblem access

In `src/subproblem.rs`, the `slot_to_row()` conversion already exists:

```rust
#[inline]
fn slot_to_row(&self, slot: usize) -> usize {
    self.first_preallocated_cut_row + slot
}
```

This is the correct pattern. Ensure all callsites use this.

### Step 5: Update cut application

Anywhere that looked up `active_cut_indices`:

```rust
// Before
let row = fcf.get_active_cut_index_by_id(cut_id);

// After
let slot = fcf.cut_pool.pool[cut_id].get_slot_index().unwrap();
let row = subproblem.slot_to_row(slot);
```

### Step 6: Handle cut activation tracking

If we need to know which cuts are active:
- Use `cut.is_active()` (already exists via atomic bool)
- Iterate pool and filter: `pool.iter().filter(|c| c.is_active())`

---

## Testing Requirements

### Compile

```bash
cargo build -j1
```

### Tests

```bash
RUST_TEST_THREADS=1 cargo test -j1
```

### Golden Tests

```bash
./scripts/golden-tests.sh verify
```

---

## Pitfalls to Avoid

- ⚠️ **Index shifting removal**: The old code shifted indices when cuts were removed. With preallocation, rows are never removed - they're deactivated via bound relaxation.
- ⚠️ **Non-preallocated path**: If still supported, needs separate handling.
- ⚠️ **cut_id vs slot_index**: These are the same in preallocated mode, but be explicit.

---

## Documentation Requirements

- [ ] Update BendersCutPool documentation
- [ ] Remove references to HashMap
- [ ] Document that preallocation is now required

---

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: HashMap removal is straightforward, but need to verify all usages.

---

## Definition of Done

- [ ] HashMap removed
- [ ] All lookups use slot_index
- [ ] No index shifting logic
- [ ] Tests pass
- [ ] Golden tests pass
