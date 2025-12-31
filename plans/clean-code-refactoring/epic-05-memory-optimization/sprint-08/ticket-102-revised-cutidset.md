# [T-102-r] CutIdSet Type Implementation (Phase 1)

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Original**: [T-102](../sprint-07/ticket-102-hashset-to-bitvec.md)
> **Dependencies**: None
> **Blocks**: T-106
> **Priority**: 2 (Deferred Optimization)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `src/fcf.rs` - HashSet usage in `BatchCutSelectionResult` (lines 321-322, 672-674)
- `src/algorithm/coordinator.rs` - `AggregatedCutSelectionResult` handling
- `docs/HOT_PATH_ALLOCATION_AUDIT.md` - Original allocation analysis

---

## Context

### Why Original T-102 Was Deferred

The original ticket required refactoring `BatchCutSelectionResult`, `AggregatedCutSelectionResult`, and all their call sites. This was too broad for Sprint 7.

### Phased Approach

**Phase 1 (This Ticket)**: Implement `CutIdSet` type with complete API and tests.

**Phase 2 (Future)**: Migrate existing HashSet usages to CutIdSet.

This phase creates the foundation without touching existing code.

### Why CutIdSet vs HashSet

| Aspect | HashSet | CutIdSet |
|--------|---------|----------|
| Insert | O(1) amortized | O(1) true |
| Contains | O(1) amortized | O(1) true |
| Memory | Hash table overhead | 1 bit per ID |
| Allocation | On insert/resize | Preallocated |
| Hash computation | Required | None |

For cut IDs (contiguous integers 0..N), bit-vector is optimal.

---

## Specification

### New Type

Create `src/memory/cut_id_set.rs`:

```rust
//! Efficient bit-vector based set for cut IDs.
//!
//! Provides O(1) operations without hashing overhead.

use std::collections::HashSet;

/// Efficient bit-vector based set for cut IDs.
///
/// Designed for sets of contiguous integer IDs [0, max_id).
/// Provides O(1) insert, contains, and clear operations.
///
/// # Memory Layout
///
/// Uses one bit per possible ID, packed into u64 words.
/// For max_id = 1000, uses ~128 bytes (16 words).
///
/// # Example
///
/// ```
/// use powers_rs::memory::CutIdSet;
///
/// let mut set = CutIdSet::with_capacity(1000);
/// set.insert(42);
/// set.insert(100);
/// assert!(set.contains(42));
/// assert!(!set.contains(999));
/// assert_eq!(set.len(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct CutIdSet {
    bits: Vec<u64>,
    max_id: usize,
    len: usize,
}

impl CutIdSet {
    /// Create a new CutIdSet with capacity for IDs [0, max_id).
    ///
    /// # Panics
    ///
    /// Panics if max_id is 0.
    pub fn with_capacity(max_id: usize) -> Self {
        assert!(max_id > 0, "CutIdSet max_id must be > 0");
        let num_words = (max_id + 63) / 64;
        Self {
            bits: vec![0u64; num_words],
            max_id,
            len: 0,
        }
    }
    
    /// Create an empty set with no capacity.
    ///
    /// This is useful as a placeholder. Call `resize()` before use.
    pub fn new() -> Self {
        Self {
            bits: Vec::new(),
            max_id: 0,
            len: 0,
        }
    }
    
    /// Resize the set to accommodate IDs [0, new_max_id).
    ///
    /// Preserves existing bits if growing. Truncates if shrinking.
    pub fn resize(&mut self, new_max_id: usize) {
        let new_num_words = (new_max_id + 63) / 64;
        self.bits.resize(new_num_words, 0);
        self.max_id = new_max_id;
        
        // Recount if shrinking (some bits may be truncated)
        if new_max_id < self.max_id {
            self.len = self.iter().count();
        }
    }
    
    /// Insert an ID into the set.
    ///
    /// Returns `true` if the ID was newly inserted, `false` if already present.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `id >= max_id`.
    #[inline]
    pub fn insert(&mut self, id: usize) -> bool {
        debug_assert!(id < self.max_id, "CutIdSet: id {} >= max_id {}", id, self.max_id);
        let word = id / 64;
        let bit = id % 64;
        let mask = 1u64 << bit;
        let was_set = (self.bits[word] & mask) != 0;
        if !was_set {
            self.bits[word] |= mask;
            self.len += 1;
        }
        !was_set
    }
    
    /// Check if an ID is in the set.
    #[inline]
    pub fn contains(&self, id: usize) -> bool {
        if id >= self.max_id {
            return false;
        }
        let word = id / 64;
        let bit = id % 64;
        (self.bits[word] & (1 << bit)) != 0
    }
    
    /// Remove an ID from the set.
    ///
    /// Returns `true` if the ID was present, `false` otherwise.
    #[inline]
    pub fn remove(&mut self, id: usize) -> bool {
        if id >= self.max_id {
            return false;
        }
        let word = id / 64;
        let bit = id % 64;
        let mask = 1u64 << bit;
        let was_set = (self.bits[word] & mask) != 0;
        if was_set {
            self.bits[word] &= !mask;
            self.len -= 1;
        }
        was_set
    }
    
    /// Clear all IDs from the set.
    #[inline]
    pub fn clear(&mut self) {
        self.bits.fill(0);
        self.len = 0;
    }
    
    /// Get the number of IDs in the set.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Check if the set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Get the maximum capacity (max_id).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.max_id
    }
    
    /// Iterate over IDs in the set in ascending order.
    ///
    /// # Determinism
    ///
    /// Unlike HashSet, iteration order is guaranteed to be ascending.
    /// This ensures deterministic behavior across runs.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter()
            .enumerate()
            .flat_map(|(word_idx, &word)| {
                (0..64).filter_map(move |bit| {
                    if (word & (1 << bit)) != 0 {
                        let id = word_idx * 64 + bit;
                        if id < self.max_id {
                            Some(id)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
            })
    }
    
    /// Convert to HashSet for compatibility with existing code.
    ///
    /// This allocates a new HashSet. Use during migration only.
    pub fn to_hashset(&self) -> HashSet<usize> {
        self.iter().collect()
    }
    
    /// Create from a HashSet.
    ///
    /// Use during migration from HashSet-based code.
    pub fn from_hashset(set: &HashSet<usize>) -> Self {
        let max_id = set.iter().max().map(|&x| x + 1).unwrap_or(64);
        let mut result = Self::with_capacity(max_id);
        for &id in set {
            result.insert(id);
        }
        result
    }
    
    /// Extend from an iterator of IDs.
    pub fn extend<I: IntoIterator<Item = usize>>(&mut self, iter: I) {
        for id in iter {
            self.insert(id);
        }
    }
    
    /// Union with another set (in-place).
    pub fn union_with(&mut self, other: &CutIdSet) {
        assert_eq!(self.max_id, other.max_id, "CutIdSet union requires same max_id");
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        self.len = self.iter().count();  // Recount
    }
}

impl Default for CutIdSet {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<usize> for CutIdSet {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let items: Vec<_> = iter.into_iter().collect();
        let max_id = items.iter().max().map(|&x| x + 1).unwrap_or(64);
        let mut set = Self::with_capacity(max_id);
        set.extend(items);
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_insert_contains() {
        let mut set = CutIdSet::with_capacity(100);
        assert!(!set.contains(42));
        assert!(set.insert(42));
        assert!(set.contains(42));
        assert!(!set.insert(42));  // Already present
        assert_eq!(set.len(), 1);
    }
    
    #[test]
    fn test_remove() {
        let mut set = CutIdSet::with_capacity(100);
        set.insert(42);
        assert!(set.remove(42));
        assert!(!set.contains(42));
        assert!(!set.remove(42));  // Already removed
        assert_eq!(set.len(), 0);
    }
    
    #[test]
    fn test_clear() {
        let mut set = CutIdSet::with_capacity(100);
        set.insert(1);
        set.insert(50);
        set.insert(99);
        assert_eq!(set.len(), 3);
        set.clear();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }
    
    #[test]
    fn test_iter_order() {
        let mut set = CutIdSet::with_capacity(100);
        set.insert(50);
        set.insert(10);
        set.insert(90);
        set.insert(30);
        
        let ids: Vec<_> = set.iter().collect();
        assert_eq!(ids, vec![10, 30, 50, 90]);  // Ascending order
    }
    
    #[test]
    fn test_to_hashset() {
        let mut set = CutIdSet::with_capacity(100);
        set.insert(1);
        set.insert(5);
        set.insert(10);
        
        let hashset = set.to_hashset();
        assert_eq!(hashset.len(), 3);
        assert!(hashset.contains(&1));
        assert!(hashset.contains(&5));
        assert!(hashset.contains(&10));
    }
    
    #[test]
    fn test_from_hashset() {
        let mut hashset = HashSet::new();
        hashset.insert(1);
        hashset.insert(5);
        hashset.insert(10);
        
        let set = CutIdSet::from_hashset(&hashset);
        assert_eq!(set.len(), 3);
        assert!(set.contains(1));
        assert!(set.contains(5));
        assert!(set.contains(10));
    }
    
    #[test]
    fn test_from_iterator() {
        let set: CutIdSet = vec![5, 10, 15].into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(5));
        assert!(set.contains(10));
        assert!(set.contains(15));
    }
    
    #[test]
    fn test_contains_out_of_range() {
        let set = CutIdSet::with_capacity(100);
        assert!(!set.contains(100));  // >= max_id
        assert!(!set.contains(1000));
    }
    
    #[test]
    fn test_word_boundary() {
        let mut set = CutIdSet::with_capacity(200);
        // Test around word boundary (64 bits)
        set.insert(63);
        set.insert(64);
        set.insert(127);
        set.insert(128);
        
        assert!(set.contains(63));
        assert!(set.contains(64));
        assert!(set.contains(127));
        assert!(set.contains(128));
        assert!(!set.contains(62));
        assert!(!set.contains(65));
    }
    
    #[test]
    fn test_union() {
        let mut set1 = CutIdSet::with_capacity(100);
        set1.insert(1);
        set1.insert(2);
        
        let mut set2 = CutIdSet::with_capacity(100);
        set2.insert(2);
        set2.insert(3);
        
        set1.union_with(&set2);
        
        assert_eq!(set1.len(), 3);
        assert!(set1.contains(1));
        assert!(set1.contains(2));
        assert!(set1.contains(3));
    }
}
```

### Module Export

Update `src/memory/mod.rs`:

```rust
mod cut_id_set;
pub use cut_id_set::CutIdSet;
```

---

## Acceptance Criteria

- [ ] `CutIdSet` type implemented with all methods
- [ ] Comprehensive unit tests passing
- [ ] Exported from `memory` module
- [ ] Documentation complete
- [ ] No changes to existing FCF code (Phase 1 only)

---

## Implementation Guide

### Suggested Approach

1. **Create `src/memory/cut_id_set.rs`** with implementation above
2. **Update `src/memory/mod.rs`** to export
3. **Run tests** to verify correctness

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/memory/cut_id_set.rs` | New file |
| `src/memory/mod.rs` | Add export |

### Pitfalls to Avoid

- ⚠️ Handle word boundaries correctly (id 63 vs 64)
- ⚠️ Ensure `iter()` returns ascending order (determinism)
- ⚠️ Don't panic on out-of-range `contains()` (return false)

---

## Testing Requirements

### Unit Tests

All tests in the specification above, plus:

- [ ] Empty set behavior
- [ ] Single element
- [ ] Full capacity (all bits set)
- [ ] Large capacity (10,000+ IDs)

---

## Documentation Requirements

- [ ] Module-level documentation
- [ ] Doc comments for all public methods
- [ ] Examples in doc comments
- [ ] Performance notes

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Self-contained type with clear specification

---

## Definition of Done

- [ ] `CutIdSet` implemented
- [ ] All unit tests passing
- [ ] Exported from memory module
- [ ] Documentation complete
- [ ] PR merged
