# Memory Stability Implementation Plan

**Goal**: Achieve zero memory allocation during SDDP training via index-based pool preallocation.

**Status**: 🟢 In Progress

## Quick Navigation

- [Master Plan](./00-master-plan.md) - Architecture and strategy overview

### Epics

| Epic | Description | Status |
|------|-------------|--------|
| [Epic 1: Preallocated Cut Pool](./epic-01-preallocated-cut-pool/00-epic-overview.md) | Preallocate all BendersCut instances | ✅ Complete |
| [Epic 2: Preallocated State Pool](./epic-02-preallocated-state-pool/00-epic-overview.md) | Preallocate all State instances | ✅ Complete |
| [Epic 3: Cut Cloning Elimination](./epic-03-cut-cloning-elimination/00-epic-overview.md) | Use Arc for cut sharing | ✅ Complete |
| [Epic 4: Validation & Profiling](./epic-04-validation-profiling/00-epic-overview.md) | Verify memory stability | 🔄 In Progress |

## Progress Tracking

### Epic 1: Preallocated Cut Pool (Sprint 1)
- [x] TICKET-001: Add slot computation utility function ✅
- [x] TICKET-002: Add BendersCut::update() method for in-place modification ✅
- [x] TICKET-003: Implement BendersCutPool::preallocate() ✅
- [x] TICKET-004: Add FCF::preallocate_pools() method ✅
- [x] TICKET-005: Update add_cuts_batch() to use slot-based access ✅
- [x] TICKET-006: Add populated flag for preallocated cuts ✅

### Epic 2: Preallocated State Pool (Sprint 1)
- [x] TICKET-007: Add State trait methods for in-place updates ✅
- [x] TICKET-008: Implement update methods for StorageState ✅
- [x] TICKET-009: Implement update methods for StorageAndInflowState ✅
- [x] TICKET-010: Implement VisitedStatePool::preallocate() ✅
- [x] TICKET-011: Update state addition to use slot-based access ✅

### Epic 3: Cut Cloning Elimination (Sprint 1)
- [x] TICKET-012: Add atomic fields to BendersCut for thread-safe access ✅
- [x] TICKET-013: Change BendersCutPool to use Arc<BendersCut> ✅

**Implementation Notes**:
- Changed `BendersCutPool::pool` from `Vec<BendersCut>` to `Vec<Arc<BendersCut>>`
- Added atomic fields: `active` (AtomicBool), `non_dominated_state_count` (AtomicUsize), `slot_index` (AtomicUsize)
- Handler cut application now clones Arc (~16 bytes) instead of full struct (~1KB)
- Methods like `add_cut_to_model` now take `&BendersCut` (immutable) since mutations use atomic operations

### Epic 4: Validation & Profiling (Sprint 1)
- [x] TICKET-014: Memory profiling on example 05 ✅
- [x] TICKET-015: Algorithm correctness validation ✅
- [ ] TICKET-016: Document remaining memory sources

**Memory Profiling Results (TICKET-014)**:
- Peak RSS on example 05: ~5.15 GB (similar to baseline)
- Runtime: ~2 min 10 sec
- The Arc-based cut pool is working correctly, but memory growth continues from other sources

**Algorithm Correctness (TICKET-015)**:
- Example 01: Expected cost $2500 ✓
- Example 05: Expected cost ~$1.019e8 ✓
- All tests pass (507/507)

**Remaining Memory Sources Identified**:
1. HiGHS internal memory (~3+ GB for 16 handlers × 60 stages)
2. Realization cloning in forward/backward passes
3. Training loop Vec allocations (130+ in sddp/mod.rs)

## Dependency Graph

```
TICKET-001 ─┬─> TICKET-003 ─┬─> TICKET-004 ─> TICKET-005 ─> TICKET-006
            │               │
TICKET-002 ─┘               │
                            v
TICKET-007 ─> TICKET-008 ─┬─> TICKET-010 ─> TICKET-011
            └─> TICKET-009 ┘
                            │
                            v
TICKET-012 ─────────────────> TICKET-013
                            │
                            v
TICKET-014 ─> TICKET-015 ─> TICKET-016
```

## Key Files

| File | Purpose |
|------|---------|
| `src/cut.rs` | BendersCut and BendersCutPool |
| `src/state.rs` | State trait and implementations |
| `src/fcf.rs` | FutureCostFunction |
| `src/sddp/mod.rs` | Training loop, cut/state addition |
