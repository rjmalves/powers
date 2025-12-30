# Clean Code Refactoring for HPC Performance

Refactoring the POWE.RS codebase into clean, modular Rust code to enable zero-allocation hot paths and improved performance while maintaining **bit-for-bit algorithmic correctness**.

> ⚠️ **CRITICAL**: Read the [Critical Principles](./00-master-plan.md#️-critical-principles-correctness-first-then-performance) section before starting ANY work.

---

## Quick Navigation

### Master Plan
- [00-master-plan.md](./00-master-plan.md) - Architecture overview, phases, and design decisions

### Architecture Document
- [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md) - Detailed architecture for Epic 5

### Epics

| Epic | Name | Duration | Status |
|------|------|----------|--------|
| 1 | [Foundation](./epic-01-foundation/00-epic-overview.md) | 2 weeks | ✅ Complete |
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | ✅ Complete |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 6 weeks | 🔄 **In Progress** |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~21-22 weeks

---

## Current Focus: Epic 5 - Parallel Zero-Allocation

Epic 5 has been **completely revised** with a new architecture that preserves parallelism while achieving zero allocations. See [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md).

### Sprint 1: Handler Staging Buffers ⬜

| ID | Title | Points | Status |
|----|-------|--------|--------|
| [T-060](./epic-05-memory-optimization/sprint-01/ticket-060-create-staging-buffer.md) | Create CutStagingBuffer struct | 2 | ⬜ |
| [T-061](./epic-05-memory-optimization/sprint-01/ticket-061-add-staging-to-handler.md) | Add staging buffer to SddpTrainHandler | 2 | ⬜ |
| [T-062](./epic-05-memory-optimization/sprint-01/ticket-062-compute-into-staging.md) | Implement compute_cut_into_staging() | 5 | ⬜ |
| [T-063](./epic-05-memory-optimization/sprint-01/ticket-063-update-from-staging.md) | Add update_from_staging() to pools | 3 | ⬜ |
| [T-064](./epic-05-memory-optimization/sprint-01/ticket-064-parallel-then-sequential.md) | Update coordinator for parallel-then-sequential | 5 | ⬜ |

### Sprint 2: Training Loop Integration ⬜

| ID | Title | Points | Status |
|----|-------|--------|--------|
| [T-065](./epic-05-memory-optimization/sprint-02/ticket-065-wire-backward-pass.md) | Wire zero-allocation path into training loop | 5 | ⬜ |
| [T-066](./epic-05-memory-optimization/sprint-02/ticket-066-golden-tests.md) | Golden tests validation | 2 | ⬜ |
| [T-067](./epic-05-memory-optimization/sprint-02/ticket-067-benchmark-parallel.md) | Benchmark parallel vs sequential | 3 | ⬜ |
| [T-068](./epic-05-memory-optimization/sprint-02/ticket-068-dhat-profiling.md) | DHAT profiling to verify zero allocations | 3 | ⬜ |

### Sprint 3: Pool Memory Model Optimization ⬜

| ID | Title | Points | Status |
|----|-------|--------|--------|
| [T-069](./epic-05-memory-optimization/sprint-03/ticket-069-remove-arc.md) | Remove Arc wrapper from BendersCutPool | 3 | ⬜ |
| [T-070](./epic-05-memory-optimization/sprint-03/ticket-070-remove-hashmap.md) | Remove HashMap from BendersCutPool | 3 | ⬜ |
| [T-071](./epic-05-memory-optimization/sprint-03/ticket-071-concrete-state-enum.md) | Create ConcreteState enum | 5 | ⬜ |
| [T-072](./epic-05-memory-optimization/sprint-03/ticket-072-migrate-state-pool.md) | Migrate VisitedStatePool to enum dispatch | 5 | ⬜ |
| [T-073](./epic-05-memory-optimization/sprint-03/ticket-073-cleanup-deprecated.md) | Cleanup deprecated CutData path | 2 | ⬜ |
| [T-074](./epic-05-memory-optimization/sprint-03/ticket-074-final-validation.md) | Final performance validation | 3 | ⬜ |

---

## Key Architecture: Parallel-Then-Sequential Pattern

```
┌────────────────────────────────────────────────────────────────┐
│  Phase 1a: Parallel (par_iter_mut on handlers)                 │
│  Each handler: Solve LP → Extract duals → Copy to staging buf  │
└────────────────────────────────────────────────────────────────┘
                             │
                   rayon sync barrier
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  Phase 1b: Sequential (deterministic order)                    │
│  for handler in handlers: pool.update_from_staging(&staging)   │
└────────────────────────────────────────────────────────────────┘
```

**Benefits**:
- Full parallelism preserved in compute-heavy Phase 1a
- Zero allocations (staging buffers are preallocated)
- Deterministic reproducibility via ordered Phase 1b

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
