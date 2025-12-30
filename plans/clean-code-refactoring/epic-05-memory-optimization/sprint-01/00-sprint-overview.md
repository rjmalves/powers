# Sprint 1: Handler Staging Buffers

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ✅ Complete

---

## Goals

1. Create `CutStagingBuffer` struct for per-handler cut/state staging
2. Integrate staging buffers into `SddpTrainHandler`
3. Implement parallel-then-sequential pattern in coordinator
4. Preserve all existing functionality (tests must pass)

---

## Tickets

| ID | Title | Points | Dependencies | Status |
|----|-------|--------|--------------|--------|
| [T-060](./ticket-060-create-staging-buffer.md) | Create CutStagingBuffer struct | 2 | None | ✅ |
| [T-061](./ticket-061-add-staging-to-handler.md) | Add staging buffer to SddpTrainHandler | 2 | T-060 | ✅ |
| [T-062](./ticket-062-compute-into-staging.md) | Implement compute_cut_into_staging() | 5 | T-061 | ✅ |
| [T-063](./ticket-063-update-from-staging.md) | Add update_from_staging() to pools | 3 | T-060 | ✅ |
| [T-064](./ticket-064-parallel-then-sequential.md) | Update coordinator for parallel-then-sequential | 5 | T-062, T-063 | ✅ |

**Total Points**: 17

---

## Key Insight

The critical innovation is separating **parallel computation** from **pool update**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1a: Parallel (par_iter_mut on handlers)                  │
│  Each handler: Solve LP → Extract duals → Copy to staging       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    rayon sync barrier
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1b: Sequential (deterministic order)                     │
│  for handler in handlers: pool.update_from_staging(&staging)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

- **From Previous Sprints**: Epic 4 complete (FCF pools preallocated)
- **Existing Infrastructure**:
  - `CutComputationBuffers` (thread-local)
  - `evaluate_cut_ref()` (returns references)
  - `update_cut_and_state_slots()` (direct copy)

---

## Testing Strategy

1. **Unit tests** for CutStagingBuffer operations
2. **Integration tests** that handler staging produces same results as old path
3. **Golden tests** must pass after T-064

---

## Definition of Done

- [x] All tickets complete
- [x] All 554 tests passing
- [x] No new allocations in cut computation
- [x] Parallel execution preserved in Phase 1a
- [x] Code reviewed (clippy clean)
