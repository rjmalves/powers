# Clean Code Refactoring for HPC Performance

Refactoring the POWE.RS codebase into clean, modular Rust code to enable zero-allocation hot paths and improved performance while maintaining **bit-for-bit algorithmic correctness**.

> ⚠️ **CRITICAL**: Read the [Critical Principles](./00-master-plan.md#️-critical-principles-correctness-first-then-performance) section before starting ANY work.

---

## Quick Navigation

### Master Plan
- [00-master-plan.md](./00-master-plan.md) - Architecture overview, phases, and design decisions

### Epics

| Epic | Name | Duration | Status |
|------|------|----------|--------|
| 1 | [Foundation](./epic-01-foundation/00-epic-overview.md) | 2 weeks | ✅ Complete |
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | ✅ Complete |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 2 weeks | ⚠️ **Integration Pending** |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~17-18 weeks

---

## ⚠️ Epic 5 - Implementation Gap Identified

See [EPIC_05_IMPLEMENTATION_ANALYSIS.md](../docs/EPIC_05_IMPLEMENTATION_ANALYSIS.md) for full analysis.

**Summary**: Zero-allocation infrastructure was implemented, but the training loop still uses the allocating path.

### What Was Implemented ✅
- `update_cut_and_state_slots()` - Direct copy to preallocated slots
- `compute_cut_into_slot()` - Zero-allocation cut computation  
- `finalize_cuts_batch()` - Batch finalization for slots
- `compute_cut_into_slot_for_backward_step()` - Full backward step
- `compute_cuts_into_slots()` - Coordinator method

### What Was NOT Implemented ❌
- Training loop in `backward_pass.rs` still calls old allocating path
- New methods exist but are never called in production
- ~18 MB allocations per training run NOT eliminated

### Required: T-055 - Wire Zero-Allocation Path

**Priority**: High - The work is 80% done, the remaining 20% delivers 100% of the value.

---

## Current Focus: Epic 5 Completion → Epic 6

### Sprint 2: Training Loop Integration ⬜ Required

| ID | Title | Status |
|----|-------|--------|
| T-055 | Wire zero-allocation path into training loop | ⬜ **NEXT** |
| T-056 | Verify zero allocations with DHAT | ⬜ |

### Sprint 1: Infrastructure ✅ Complete

| ID | Title | Status |
|----|-------|--------|
| [T-050](./epic-05-memory-optimization/sprint-01/ticket-050-direct-cut-slot-update.md) | Add direct cut slot update method | ✅ |
| [T-051](./epic-05-memory-optimization/sprint-01/ticket-051-compute-cut-into-slot.md) | Add compute_cut_into_slot to State trait | ✅ |
| [T-052](./epic-05-memory-optimization/sprint-01/ticket-052-update-backward-pass.md) | Update backward pass to use direct slot updates | ⚠️ Partial |
| [T-053](./epic-05-memory-optimization/sprint-01/ticket-053-remove-cutdata-from-hot-path.md) | Remove CutData from hot path | ⚠️ Partial |
| [T-054](./epic-05-memory-optimization/sprint-01/ticket-054-verify-zero-allocations.md) | Verify zero allocations with profiling | ❌ Blocked |

---

## Production Code Path Analysis

### Current Path (ALLOCATING - Still Used)

```
backward_pass.rs:266  →  compute_cuts_parallel()
    → coordinator.rs:237  →  compute_cut_data_for_backward_step()
        → state.rs:1108  →  CutData::from_refs()  ← ALLOCATES 2x Vec<f64>
```

### New Path (ZERO-ALLOC - Not Wired In)

```
[NOT CALLED]  →  compute_cuts_into_slots()
    → coordinator.rs:194  →  compute_cut_into_slot_for_backward_step()
        → state.rs:1133  →  update_cut_and_state_slots()  ← NO ALLOCATION
```

---

## Before You Start Any Ticket

1. **Read the Critical Principles** in the [master plan](./00-master-plan.md)
2. **Run golden output tests**: `./scripts/golden-tests.sh verify`
3. **Read the epic and sprint overviews**
4. **Use single-threaded builds/tests**: `cargo build -j1`, `RUST_TEST_THREADS=1 cargo test -j1`

## After Completing Any Ticket

1. **Build**: `cargo build -j1`
2. **Test**: `RUST_TEST_THREADS=1 cargo test -j1`
3. **Feature test**: `cargo build -j1 --features timing`
4. **Golden tests**: `./scripts/golden-tests.sh verify` ✅ CRITICAL
5. **If ANY failure**: STOP and investigate

---

## Status Legend

- ⬜ Not Started
- 🔄 In Progress
- ⚠️ Needs Attention / Partial
- ❌ Blocked / Requires Rework
- ✅ Complete
- 🔴 Blocked
