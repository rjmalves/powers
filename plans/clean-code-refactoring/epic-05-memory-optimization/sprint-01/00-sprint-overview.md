# Sprint 1: Pool Implementation

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

Memory optimization changes how data is stored and reused, NOT what data is computed.

**All numerical values must remain EXACTLY the same.** Golden tests after every change. If any memory corruption or incorrect values appear, **STOP immediately**.

---

## Goals

1. **Primary**: Implement CutPool for zero-allocation cut management
2. **Primary**: Implement RealizationPool for trajectory buffers
3. **Primary**: Integrate pools into training loop
4. **Validation**: Verify zero allocations with DHAT

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-033](./ticket-033-implement-cut-pool.md) | Implement CutPool | 5 | Yes | Epic 4 | ⬜ |
| [T-034](./ticket-034-implement-realization-pool.md) | Implement RealizationPool | 5 | Yes | Epic 4 | ⬜ |
| [T-035](./ticket-035-integrate-pools.md) | Integrate pools into training loop | 3 | Yes | T-033, T-034 | ⬜ |
| [T-036](./ticket-036-verify-zero-allocation.md) | Verify zero-allocation hot paths | 3 | Yes | T-035 | ⬜ |

**Total Points**: 16

---

## Parallelization

```
T-033 (CutPool) ─────────────────────┐
                                     ├──→ T-035 (Integrate) ──→ T-036 (Verify)
T-034 (RealizationPool) ─────────────┘
```

T-033 and T-034 can run in parallel.

---

## Definition of Done

- [ ] All 4 tickets complete
- [ ] Pools integrated and functional
- [ ] Zero allocations verified with DHAT
- [ ] Golden tests pass
- [ ] Performance improved or unchanged
