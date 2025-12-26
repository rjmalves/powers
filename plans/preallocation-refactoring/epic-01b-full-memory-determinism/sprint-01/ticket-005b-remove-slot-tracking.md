# [TICKET-005b] Remove slot tracking data structures

> **Epic**: [Epic 1b: Full Memory Determinism](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TICKET-004b](./ticket-004b-update-deactivation.md)  
> **Blocks**: None

## Context

### Background

After switching to deterministic slot calculation and storing slot_index in cuts, the following data structures are no longer needed:

- `free_cut_slots: Vec<usize>` - LIFO stack for slot reuse
- `cut_slot_to_id: Vec<Option<usize>>` - Mapping from slot to cut ID
- `next_available_cut_slot: usize` - Sequential counter

These were ~50 lines of complexity that can be removed.

### Memory Savings

For 128 preallocated cuts:
- `cut_slot_to_id`: 128 × 16 bytes (Option<usize>) = 2 KB
- `free_cut_slots`: Vec overhead + variable capacity = ~1 KB typical
- Total: ~3 KB per subproblem

With many subproblems, this adds up.

## Files to Read Before Starting

- `src/subproblem.rs` - Slot tracking fields (lines 767-783)
- `src/subproblem.rs` - `allocate_cut_slot()` method (lines 1084-1099)

## Specification

### Remove from Subproblem struct

Delete these fields:

```rust
// DELETE THESE:
/// Next available slot for sequential allocation.
/// Incremented when no free slots available.
next_available_cut_slot: usize,

/// Mapping from slot index to cut ID.
/// `None` indicates slot is free.
cut_slot_to_id: Vec<Option<usize>>,

/// Stack of freed slots for reuse (LIFO order).
/// When a cut is removed, its slot is pushed here.
free_cut_slots: Vec<usize>,
```

### Remove from constructor

Delete initialization of these fields:

```rust
// DELETE THESE:
next_available_cut_slot: 0,
cut_slot_to_id: Vec::new(),
free_cut_slots: Vec::new(),
```

### Remove from preallocate_cut_constraints()

Delete these lines:

```rust
// DELETE THESE:
self.next_available_cut_slot = 0;
self.cut_slot_to_id = vec![None; max_cuts];
self.free_cut_slots = Vec::with_capacity(max_cuts / 4);
```

### Remove allocate_cut_slot() method

Delete the entire method (~20 lines):

```rust
// DELETE THIS ENTIRE METHOD:
fn allocate_cut_slot(&mut self) -> Option<usize> {
    // Prefer reusing freed slots (LIFO for cache locality)
    if let Some(slot) = self.free_cut_slots.pop() {
        return Some(slot);
    }
    // ...
}
```

### Remove available_cut_slots() method

Delete if it exists:

```rust
// DELETE IF EXISTS:
pub fn available_cut_slots(&self) -> usize {
    self.free_cut_slots.len()
        + (self.num_preallocated_cuts - self.next_available_cut_slot)
}
```

### Update any remaining references

Search for any remaining references to deleted fields and update/remove them.

## Acceptance Criteria

- [x] `next_available_cut_slot` field removed
- [x] `cut_slot_to_id` field removed
- [x] `free_cut_slots` field removed
- [x] `allocate_cut_slot()` method removed
- [x] Constructor no longer initializes these fields
- [x] `preallocate_cut_constraints()` no longer sets these fields
- [x] No remaining references to deleted items
- [x] Code compiles
- [x] Examples pass

## Implementation Guide

### Step 1: Find all references

```bash
grep -n "next_available_cut_slot\|cut_slot_to_id\|free_cut_slots\|allocate_cut_slot" src/subproblem.rs
```

### Step 2: Remove struct fields

Edit the struct definition to remove the three fields.

### Step 3: Remove constructor initialization

Find the constructor and remove initialization lines.

### Step 4: Remove preallocate_cut_constraints() lines

Remove the lines that set these fields.

### Step 5: Remove allocate_cut_slot() method

Delete the entire method.

### Step 6: Remove available_cut_slots() method

Delete if it exists.

### Step 7: Verify compilation

```bash
cargo build --release
```

### Step 8: Run examples

```bash
cargo run --release -- run examples/01-deterministic
cargo run --release -- run examples/07-par-model-with-inflow-state
```

### Pitfalls to Avoid

- ⚠️ Make sure TICKET-003b and TICKET-004b are complete first
- ⚠️ Search thoroughly for any remaining references
- ⚠️ The `available_cut_slots()` method may be called from outside - check first

## Testing Requirements

### Unit Tests

- [ ] Ensure no tests reference deleted fields/methods

### Integration Tests

- [ ] Examples 01 and 07 pass
- [ ] Results identical to before

## Documentation Requirements

- [ ] Remove any doc comments referencing deleted items
- [ ] Update module-level documentation if needed

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Straightforward deletion once dependencies are done

## Definition of Done

- [x] All slot tracking code removed
- [x] Code compiles
- [x] Examples pass
- [x] Code is simpler (~50 fewer lines)
