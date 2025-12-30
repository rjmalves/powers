# Sprint 4: Pool Architecture Refinement

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: 🔄 In Progress

---

## Problem Statement

The current `ConcreteState` enum stores `StateLayout` redundantly in each `StorageAndInflow` variant:

```rust
pub enum ConcreteState {
    Storage { core: StateCore, num_hydros: usize },
    StorageAndInflow {
        core: StateCore,
        num_hydros: usize,
        layout: StateLayout,  // ❌ DUPLICATED for every state in the pool!
    },
}
```

**Memory Impact**: For a pool of 500 states with 10 hydros:
- `StateLayout` = 2×`Vec<usize>` + 1×`usize` ≈ 48 bytes stack + heap allocations
- `per_hydro_dims`: 10 × 8 bytes = 80 bytes heap
- `offsets`: 11 × 8 bytes = 88 bytes heap
- **Per state overhead**: ~216 bytes
- **Total redundant allocation**: 500 × 216 bytes = **108 KB of duplicated metadata**

The layout is **node-level metadata** (same for all states in a node's pool), not per-state data.

---

## Goals

1. **Eliminate layout duplication** by storing it once in the pool
2. **Simplify state representation** to pure coefficient data
3. **Improve cache efficiency** by removing metadata from hot data
4. **Remove ConcreteState enum** (no longer needed with shared layout)
5. **Maintain backward compatibility** with existing interfaces

---

## Design: Shared Layout Architecture

### Target State

```rust
/// Pure state data - no metadata redundancy
#[derive(Debug, Clone)]
pub struct StateData {
    pub coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

/// Pool with shared layout (stored once)
pub struct VisitedStatePool {
    /// State data storage
    pub pool: Vec<StateData>,
    /// Shared layout for all states (stored once)
    pub layout: Option<StateLayout>,
    /// State type for all states in pool
    pub state_type: StateTypeId,
    /// Number of hydros
    pub num_hydros: usize,
}
```

### Benefits

| Metric | Before (Sprint 3) | After (Sprint 4) |
|--------|-------------------|------------------|
| Layout storage | 500 copies | 1 copy |
| Memory per state | ~216 bytes overhead | 0 bytes overhead |
| Heap allocations | 1000 (2 per state) | 2 (shared layout) |
| Cache efficiency | Poor (metadata interleaved) | Good (pure data) |
| Enum dispatch | Required | Eliminated |

---

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| [T-075](./ticket-075-create-state-data.md) | Create StateData struct (pure coefficient data) | 2 | T-072 | ✅ |
| [T-076](./ticket-076-refactor-state-pool.md) | Refactor VisitedStatePool to shared layout | 5 | T-075 | ✅ |
| [T-077](./ticket-077-update-fcf-state-access.md) | Update FCF for shared layout state access | 3 | T-076 | ✅ |
| [T-078](./ticket-078-remove-concrete-state.md) | Remove ConcreteState enum | 2 | T-077 | ✅ |
| [T-079](./ticket-079-benchmark-memory.md) | Benchmark memory usage and performance | 2 | T-078 | ⬜ |

**Total Points**: 14

---

## Implementation Order

```
T-075: StateData struct
    │
    ▼
T-076: Refactor VisitedStatePool
    │
    ▼
T-077: Update FCF state access
    │
    ▼
T-078: Remove ConcreteState
    │
    ▼
T-079: Benchmark and validate
```

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API breaking changes | Medium | Medium | Add new methods first, deprecate old |
| State access patterns change | Low | Medium | Audit all pool[slot] usages |
| Layout context needed at callsites | Medium | Low | Pool provides layout accessor |
| Test updates required | High | Low | Methodical test updates |

---

## Key Files

| Component | Location | Changes |
|-----------|----------|---------|
| StateData (new) | `src/state.rs` | Create new struct |
| VisitedStatePool | `src/state.rs:871-1020` | Major refactor |
| ConcreteState | `src/state.rs:480-787` | Remove after migration |
| StateConfig | `src/state.rs:789-869` | Simplify |
| FutureCostFunction | `src/fcf.rs:180-700` | Update state access |
| BendersCutPool | `src/cut.rs:460-520` | Update `update_cut_and_state_slots` |

---

## Testing Strategy

1. **Unit tests**: New tests for StateData and updated pool
2. **Migration tests**: Ensure new API produces same results as old
3. **Integration tests**: All existing tests must pass
4. **Golden tests**: Bit-for-bit identical results
5. **Memory benchmark**: Measure actual memory reduction

---

## Definition of Done

- [ ] All tickets complete
- [ ] StateLayout stored once per pool (not per state)
- [ ] ConcreteState enum removed
- [ ] Memory usage reduced (measurable)
- [ ] All 573+ tests pass
- [ ] Golden tests pass
- [ ] No performance regression
