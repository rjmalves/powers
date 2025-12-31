# [T-102] Replace HashSet with BitVec in Cut Selection

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 7: Comprehensive Memory Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: None
> **Priority**: 3 (Rust Allocation Optimization)
> **Status**: ⏸️ Deferred

## Files to Read Before Starting

- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original audit identifying this allocation
- `src/fcf.rs` - HashSet usage in cut selection
- `src/algorithm/coordinator.rs` - AggregatedCutSelectionResult usage

---

## Context

### Background

Cut selection uses `HashSet<usize>` to track cut IDs:

```rust
// src/fcf.rs:321-322
let mut new_cut_ids = HashSet::new();
let mut returning_cut_ids = HashSet::new();

// src/fcf.rs:394
HashSet::new()  // removing_cut_ids

// src/fcf.rs:588-589, 627, 771-773
// Similar patterns
```

HashSet has allocation overhead:
- Initial bucket allocation
- Rehashing as elements are added
- Hash computation overhead

### Target

Replace with `BitVec` or `Vec<bool>` for known maximum cut count:
- O(1) insert/lookup (bit operations)
- No hashing overhead
- Preallocated to max capacity

### Consideration

The maximum cut count is known: `num_iterations * num_forward_passes`.

---

## Specification

### New Type

```rust
/// Efficient set for cut IDs with known maximum.
/// Uses bit vector for O(1) insert/lookup without hashing.
pub struct CutIdSet {
    bits: Vec<u64>,  // Or use bitvec crate
    max_id: usize,
}

impl CutIdSet {
    pub fn with_capacity(max_id: usize) -> Self {
        let num_words = (max_id + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            max_id,
        }
    }
    
    pub fn insert(&mut self, id: usize) {
        debug_assert!(id < self.max_id);
        let word = id / 64;
        let bit = id % 64;
        self.bits[word] |= 1 << bit;
    }
    
    pub fn contains(&self, id: usize) -> bool {
        if id >= self.max_id { return false; }
        let word = id / 64;
        let bit = id % 64;
        (self.bits[word] & (1 << bit)) != 0
    }
    
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
    
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                (0..64).filter_map(move |bit| {
                    if (word & (1 << bit)) != 0 {
                        Some(word_idx * 64 + bit)
                    } else {
                        None
                    }
                })
            })
    }
}
```

### Alternative: Use `bitvec` Crate

```rust
use bitvec::prelude::*;

type CutIdSet = BitVec<u64, Lsb0>;
```

### Behavior

- Same semantics as HashSet for insert, contains, iter
- Preallocated to maximum cut count
- Thread-local instances can be reused

---

## Acceptance Criteria

- [ ] `CutIdSet` type implemented (or bitvec crate added)
- [ ] FCF cut selection refactored to use new type
- [ ] `AggregatedCutSelectionResult` updated to use new type
- [ ] Thread-local instances preallocated
- [ ] All tests pass
- [ ] DHAT shows reduced HashSet allocations

---

## Implementation Guide

### Suggested Approach

#### Option A: Custom Implementation (No New Dependencies)

1. **Add `CutIdSet`** to `src/cut.rs` or new `src/memory/cut_id_set.rs`:
   - Implement as shown in specification
   - Add unit tests

2. **Update `AggregatedCutSelectionResult`** in `src/fcf.rs`:
   ```rust
   pub struct AggregatedCutSelectionResult {
       pub new_cut_ids: CutIdSet,
       pub returning_cut_ids: CutIdSet,
       pub removing_cut_ids: CutIdSet,
   }
   ```

3. **Refactor FCF functions** to use new type

#### Option B: Use `bitvec` Crate

1. **Add dependency** to `Cargo.toml`:
   ```toml
   [dependencies]
   bitvec = "1.0"
   ```

2. **Use directly** as `BitVec<u64, Lsb0>`

3. **Wrap in newtype** if needed for cleaner API

### Thread-Local Preallocated Sets

```rust
thread_local! {
    static NEW_CUT_IDS: RefCell<CutIdSet> = RefCell::new(CutIdSet::with_capacity(1024));
    static RETURNING_CUT_IDS: RefCell<CutIdSet> = RefCell::new(CutIdSet::with_capacity(1024));
    static REMOVING_CUT_IDS: RefCell<CutIdSet> = RefCell::new(CutIdSet::with_capacity(1024));
}
```

### Key Files to Modify

- `src/cut.rs` or new file: Add `CutIdSet` type
- `src/fcf.rs`: Refactor `add_cuts_batch()`, `finalize_cuts_batch()`, etc.
- `src/algorithm/coordinator.rs`: Update `AggregatedCutSelectionResult` handling

### Patterns to Follow

- Keep API similar to HashSet for minimal code changes
- Use `clear()` and reuse instead of creating new instances

### Pitfalls to Avoid

- ⚠️ Ensure max capacity is correctly computed
- ⚠️ BitVec iteration order may differ from HashSet (but should be deterministic)
- ⚠️ Consider if HashSet clone semantics are used anywhere
- ⚠️ The `iter()` implementation must be correct

---

## Testing Requirements

### Unit Tests

- [ ] `CutIdSet::insert` and `contains` work correctly
- [ ] `CutIdSet::iter` returns all inserted IDs
- [ ] `CutIdSet::clear` resets all bits
- [ ] Edge cases: empty set, single element, full capacity

### Integration Tests

- [ ] Golden tests pass (cut selection produces same results)
- [ ] Cut selection determinism preserved

### Performance Tests

- [ ] DHAT shows reduced HashSet allocations
- [ ] Benchmark insert/lookup performance vs HashSet

---

## Documentation Requirements

- [ ] Doc comments for `CutIdSet` type
- [ ] Note in `HOT_PATH_ALLOCATION_AUDIT.md` that fix is complete

---

## Dependencies

- **Blocked By**: None
- **Blocks**: None
- **Related**: T-099-T-101 (other Rust allocation optimizations)

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Custom data structure with multiple refactoring sites

---

## Definition of Done

- [ ] `CutIdSet` implemented and tested
- [ ] FCF refactored to use new type
- [ ] All tests passing
- [ ] DHAT shows improvement
- [ ] Documentation updated
- [ ] PR merged
