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

1. **Primary**: Reduce code duplication in `state.rs`
2. **Primary**: Consolidate common utilities
3. **Secondary**: Document allocation points for Epic 5
4. **Validation**: Bit-for-bit identical outputs

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-028](./ticket-028-analyze-state-structure.md) | Analyze state.rs structure and duplication | 2 | Yes | Epic 3 | ⬜ |
| [T-029](./ticket-029-extract-state-utilities.md) | Extract common state utilities | 3 | Yes | T-028 | ⬜ |
| [T-030](./ticket-030-consolidate-storage-state.md) | Consolidate StorageState methods | 3 | Yes | T-029 | ⬜ |
| [T-031](./ticket-031-consolidate-inflow-state.md) | Consolidate StorageAndInflowState methods | 3 | Yes | T-029 | ⬜ |
| [T-032](./ticket-032-document-allocation-points.md) | Document trait object allocation points | 2 | Yes | T-030, T-031 | ⬜ |

**Total Points**: 13

---

## Parallelization

```
T-028 (Analyze) ──→ T-029 (Utilities) ──→ T-030 (Storage) ────────┐
                                     └──→ T-031 (Inflow) ─────────├──→ T-032 (Document)
```

T-030 and T-031 can run in parallel after T-029.

---

## Definition of Done

- [ ] All 5 tickets complete
- [ ] `state.rs` reduced by ≥30%
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] Allocation points documented
