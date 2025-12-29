# Sprint 1: State Consolidation

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

State representations are used throughout the SDDP algorithm. Any change that modifies state values will corrupt the algorithm.

**All numerical values must remain EXACTLY the same.** Golden tests after every change.

---

## Goals

1. **Primary**: Reduce code duplication in `state.rs` via `StateCore` composition
2. **Primary**: Document State-Cut 1:1 relationship and slot indexing
3. **Secondary**: Add pool-compatible trait extensions for Epic 5
4. **Validation**: Bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-038](./ticket-038-analyze-state-structure.md) | Analyze state.rs structure and duplication | 3 | Yes | Epic 3 | ⬜ |
| [T-039](./ticket-039-document-state-cut-relationship.md) | Document State-Cut 1:1 relationship and slot indexing | 2 | Yes | T-038 | ⬜ |
| [T-040](./ticket-040-extract-state-utilities.md) | Extract common state utilities (StateCore) | 3 | Yes | T-038 | ⬜ |
| [T-041](./ticket-041-consolidate-storage-state.md) | Consolidate StorageState methods | 3 | Yes | T-040 | ⬜ |
| [T-042](./ticket-042-consolidate-inflow-state.md) | Consolidate StorageAndInflowState methods | 3 | Yes | T-040 | ⬜ |
| [T-043](./ticket-043-pool-compatible-extensions.md) | Add pool-compatible trait extensions | 3 | Yes | T-041, T-042 | ⬜ |
| [T-044](./ticket-044-state-extraction-module.md) | Create/document state extraction module | 2 | Yes | T-043 | ⬜ |

**Total Points**: 19

---

## Technical Approach

### StateCore Composition Pattern

Extract common fields into `StateCore` struct:

```rust
/// Common state fields shared by all State implementations.
pub struct StateCore {
    pub dimension: usize,
    pub state_coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}
```

Both `StorageState` and `StorageAndInflowState` embed this struct:

```rust
pub struct StorageState {
    core: StateCore,
}

pub struct StorageAndInflowState {
    core: StateCore,
    dimension: usize,     // num_hydros (different from core.dimension!)
    layout: StateLayout,
}
```

### Estimated Line Reduction

| Section | Before | After | Savings |
|---------|--------|-------|---------|
| Common getters/setters | ~60 × 2 = 120 | ~60 (in StateCore) | ~60 |
| Common method bodies | ~40 × 2 = 80 | ~40 + delegation | ~30 |
| Total | ~200 duplicated | ~100 shared | ~100 lines (~3%) |

**Note**: The 30% reduction target may be ambitious. More realistic: 5-10% reduction plus significant deduplication of logic.

---

## Parallelization

```
T-038 (Analyze) ──→ T-039 (Document Cut-State) ──────────────────────────┐
                └──→ T-040 (StateCore) ──→ T-041 (Storage) ────────┐     │
                                       └──→ T-042 (Inflow) ────────├──→ T-043 (Pool Ext) ──→ T-044 (Extraction)
```

- T-039 runs in parallel with T-040
- T-041 and T-042 can run in parallel after T-040

---

## Verification Protocol

After EVERY ticket:

```bash
cargo build -j1 && RUST_TEST_THREADS=1 cargo test -j1 && ./scripts/golden-tests.sh verify
```

---

## Definition of Done

- [ ] All 7 tickets complete
- [ ] `StateCore` struct implemented and used by both state types
- [ ] State-Cut relationship documented
- [ ] Pool-compatible extensions added
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Ready for Sprint 2 (FCF wrapper removal)
