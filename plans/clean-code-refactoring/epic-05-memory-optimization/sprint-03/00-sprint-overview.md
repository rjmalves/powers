# Sprint 3: Pool Memory Model Optimization

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## Goals

1. Remove Arc wrapper from BendersCutPool for direct access
2. Remove HashMap from BendersCutPool (use direct slot indexing)
3. Replace Box<dyn State> with enum dispatch in VisitedStatePool
4. Cleanup deprecated CutData path
5. Final performance validation

---

## Prerequisites

- Sprint 2 complete (training loop using staging path)
- Golden tests passing
- DHAT shows zero allocations

---

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| [T-069](./ticket-069-remove-arc.md) | Remove Arc wrapper from BendersCutPool | 3 | T-068 | ⬜ |
| [T-070](./ticket-070-remove-hashmap.md) | Remove HashMap from BendersCutPool | 3 | T-069 | ⬜ |
| [T-071](./ticket-071-concrete-state-enum.md) | Create ConcreteState enum | 5 | None | ⬜ |
| [T-072](./ticket-072-migrate-state-pool.md) | Migrate VisitedStatePool to enum dispatch | 5 | T-071 | ⬜ |
| [T-073](./ticket-073-cleanup-deprecated.md) | Cleanup deprecated CutData path | 2 | T-070, T-072 | ⬜ |
| [T-074](./ticket-074-final-validation.md) | Final performance validation | 3 | T-073 | ⬜ |

**Total Points**: 21

---

## Key Changes

### BendersCutPool

**Before**:
```rust
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    pub active_cut_indices: HashMap<usize, usize>,
    ...
}
```

**After**:
```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,  // Direct storage
    // No HashMap - use cut.slot_index for model row lookup
    ...
}
```

### VisitedStatePool

**Before**:
```rust
pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}
```

**After**:
```rust
pub enum ConcreteState {
    Storage(StorageStateCore),
    StorageAndInflow(StorageAndInflowStateCore),
}

pub struct VisitedStatePool {
    pub pool: Vec<ConcreteState>,
}
```

---

## Expected Benefits

| Optimization | Expected Impact |
|--------------|-----------------|
| Arc removal | ~1-2% faster (no refcount) |
| HashMap removal | ~2-3% faster (no hashing) |
| Enum dispatch | ~1-2% faster (no vtable) |
| **Combined** | **~5-10% faster** |

---

## Risk Mitigation

- **T-069 (Arc removal)**: Careful audit of all `Arc<BendersCut>` usage
- **T-071 (Enum)**: Match statements may add complexity
- **Golden tests**: Run after each ticket

---

## Definition of Done

- [ ] All tickets complete
- [ ] No Arc/HashMap in BendersCutPool
- [ ] No Box<dyn State> in VisitedStatePool
- [ ] Deprecated code removed
- [ ] Golden tests pass
- [ ] ≥5% speedup measured
- [ ] All 549+ tests pass
