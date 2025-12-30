# [T-099] Replace HashSet with BitVec in Cut Selection

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

Cut selection uses `HashSet<usize>` to track cut IDs:

```rust
// src/fcf.rs:321-322
let mut new_cut_ids = HashSet::new();
let mut returning_cut_ids = HashSet::new();
```

HashSet allocates hash table buckets, which creates per-batch allocation overhead. Since cut IDs are small integers with a known maximum, a `BitVec` or `Vec<bool>` is more efficient.

### Relation to Epic

Eliminates HashSet allocation overhead in cut selection.

### Current State

- 3 HashSets created per stage: `new_cut_ids`, `returning_cut_ids`, `removing_cut_ids`
- Each HashSet allocates internal buckets
- ~500 bytes per HashSet × 3 × 60 stages = ~90 KB per iteration

## Specification

### Changes Required

1. **Replace HashSet with BitVec or Vec<bool>** for cut ID tracking
2. **Preallocate based on max cut count**
3. **Update iteration to use bit operations**

### Approach Options

**Option A: BitVec (bitvec crate)**
```rust
use bitvec::prelude::*;
let mut new_cut_flags: BitVec = bitvec![0; max_cuts];
new_cut_flags.set(cut_id, true);
```

**Option B: Vec<bool> (no external dependency)**
```rust
let mut new_cut_flags: Vec<bool> = vec![false; max_cuts];
new_cut_flags[cut_id] = true;
```

**Option C: Preallocated HashSet (keep HashSet, preallocate)**
```rust
let mut new_cut_ids: HashSet<usize> = HashSet::with_capacity(expected_size);
```

### Behavior

- Same cut selection logic
- `contains()` → `flags[id]`
- `insert()` → `flags[id] = true`
- `iter()` → `flags.iter().enumerate().filter(|(_, &f)| f).map(|(i, _)| i)`

## Acceptance Criteria

- [ ] HashSet replaced with more efficient structure
- [ ] No per-batch allocation for cut ID tracking
- [ ] All tests pass
- [ ] Cut selection behavior unchanged

## Implementation Guide

### Suggested Approach

1. **Add flag buffers to coordinator or FCF**:
   ```rust
   struct CutSelectionBuffers {
       new_cut_flags: Vec<bool>,
       returning_cut_flags: Vec<bool>,
       removing_cut_flags: Vec<bool>,
   }
   
   impl CutSelectionBuffers {
       fn new(max_cuts: usize) -> Self {
           Self {
               new_cut_flags: vec![false; max_cuts],
               returning_cut_flags: vec![false; max_cuts],
               removing_cut_flags: vec![false; max_cuts],
           }
       }
       
       fn clear(&mut self) {
           self.new_cut_flags.fill(false);
           self.returning_cut_flags.fill(false);
           self.removing_cut_flags.fill(false);
       }
   }
   ```

2. **Update FCF cut selection**:
   ```rust
   // Before:
   let mut new_cut_ids = HashSet::new();
   new_cut_ids.insert(cut.id);
   if new_cut_ids.contains(&other_id) { ... }
   
   // After:
   buffers.clear();
   buffers.new_cut_flags[cut.id] = true;
   if buffers.new_cut_flags[other_id] { ... }
   ```

3. **Handle result collection**:
   ```rust
   // Collect IDs from flags
   fn collect_ids(flags: &[bool]) -> impl Iterator<Item = usize> + '_ {
       flags.iter()
           .enumerate()
           .filter_map(|(i, &f)| if f { Some(i) } else { None })
   }
   ```

### Key Files to Modify

- `src/fcf.rs` - Cut selection logic
- `src/algorithm/coordinator.rs` - May hold buffers

### Patterns to Follow

- See existing preallocated buffer patterns
- Keep HashSet as fallback if index exceeds buffer size

### Pitfalls to Avoid

- ⚠️ Cut IDs must be within buffer bounds
- ⚠️ Buffer must be cleared between batches
- ⚠️ May need to resize if max_cuts grows

## Testing Requirements

### Unit Tests

- [ ] Flag-based selection matches HashSet behavior
- [ ] Buffer clearing works correctly

### Integration Tests

- [ ] Cut selection produces same results
- [ ] Golden tests pass

## Documentation Requirements

- [ ] Document buffer sizing requirements

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101 (DHAT verification)
- **Related**: T-094, T-095 (other buffer optimizations)

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Need to ensure correct behavior with flag-based approach

## Definition of Done

- [ ] HashSet replaced
- [ ] Preallocated buffers used
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
