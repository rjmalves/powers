# Clean Code Refactoring for HPC Performance

Refactoring the POWE.RS codebase into clean, modular Rust code to enable zero-allocation hot paths and improved performance while maintaining **bit-for-bit algorithmic correctness**.

> ⚠️ **CRITICAL**: Read the [Critical Principles](./00-master-plan.md#️-critical-principles-correctness-first-then-performance) section before starting ANY work.

---

## Quick Navigation

### Master Plan
- [00-master-plan.md](./00-master-plan.md) - Architecture overview, phases, and design decisions

### Key Documents
- [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md) - Detailed architecture for Epic 5
- [HOT_PATH_ALLOCATION_AUDIT.md](../../docs/HOT_PATH_ALLOCATION_AUDIT.md) - DHAT profiling analysis and findings

### Epics

| Epic | Name | Duration | Status |
|------|------|----------|--------|
| 1 | [Foundation](./epic-01-foundation/00-epic-overview.md) | 2 weeks | ✅ Complete |
| 2 | [Core Extraction](./epic-02-core-extraction/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 3 | [Algorithm Separation](./epic-03-algorithm-separation/00-epic-overview.md) | 4-5 weeks | ✅ Complete |
| 4 | [State Simplification](./epic-04-state-simplification/00-epic-overview.md) | 3 weeks | ✅ Complete |
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 8 sprints | 🔄 **In Progress** |
| 6 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 7 | [Performance Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~24-26 weeks

---

## Current Focus: Epic 5 - Memory Optimization

Epic 5 has been extended with Sprints 6-9 based on **DHAT profiling findings** that revealed 94.7% of allocations come from HiGHS, not Rust code.

### Sprint Progress

| Sprint | Focus | Status |
|--------|-------|--------|
| 1-4 | Handler Staging Buffers & Pool Optimization | ✅ Complete |
| 5 | Deterministic Memory Allocation | ✅ Complete |
| 6 | [HiGHS Solver Memory Optimization](./epic-05-memory-optimization/sprint-06/00-sprint-overview.md) | ✅ Complete |
| 7 | [Rust Application Allocation Optimization](./epic-05-memory-optimization/sprint-07/00-sprint-overview.md) | ✅ Complete |
| 8 | [Validation and Documentation](./epic-05-memory-optimization/sprint-08/00-sprint-overview.md) | ✅ Complete |
| 9 | [RSS Stabilization via Allocator Strategy](./epic-05-memory-optimization/sprint-09/00-sprint-overview.md) | ✅ Complete |

### Sprint 9: RSS Stabilization - COMPLETE ✅

**CRITICAL FINDING**: glibc is the best allocator! Alternative allocators (mimalloc, jemalloc) perform WORSE.

See [ALLOCATOR_COMPARISON.md](../../docs/ALLOCATOR_COMPARISON.md) for full analysis.

| ID | Title | Points | Status |
|----|-------|--------|--------|
| [T-130](./epic-05-memory-optimization/sprint-09/ticket-130-rss-measurement-harness.md) | RSS Measurement Harness | 3 | ✅ Complete |
| [T-131](./epic-05-memory-optimization/sprint-09/ticket-131-test-mimalloc.md) | Test mimalloc RSS | 3 | ✅ NOT RECOMMENDED |
| [T-132](./epic-05-memory-optimization/sprint-09/ticket-132-add-jemalloc.md) | Add jemalloc Dependency | 2 | ✅ Complete |
| [T-133](./epic-05-memory-optimization/sprint-09/ticket-133-test-jemalloc.md) | Test jemalloc RSS | 3 | ✅ NOT RECOMMENDED |
| [T-134](./epic-05-memory-optimization/sprint-09/ticket-134-test-malloc-trim.md) | Test malloc_trim | 2 | ⏭️ Skipped (already in Sprint 8) |
| [T-135](./epic-05-memory-optimization/sprint-09/ticket-135-compare-allocators.md) | Compare Allocators | 2 | ✅ glibc WINS |
| [T-136](./epic-05-memory-optimization/sprint-09/ticket-136-default-allocator.md) | Set Default Allocator | 3 | ⏭️ Skipped (glibc already default) |
| [T-137](./epic-05-memory-optimization/sprint-09/ticket-137-validate-tests.md) | Validate Tests | 2 | ⏭️ Skipped |
| [T-138](./epic-05-memory-optimization/sprint-09/ticket-138-performance-benchmark.md) | Performance Benchmark | 3 | ⏭️ Skipped |
| [T-139](./epic-05-memory-optimization/sprint-09/ticket-139-document-allocator.md) | Document Allocator | 2 | ✅ Complete |
| [T-140](./epic-05-memory-optimization/sprint-09/ticket-140-rss-ci-check.md) | RSS CI Check | 3 | 📋 Optional |

### Sprint 6: HiGHS Solver Memory Optimization

### Sprint 7: Rust Application Allocation Optimization

| ID | Title | Points |
|----|-------|--------|
| [T-094](./epic-05-memory-optimization/sprint-07/ticket-094-uniform-prob-buffer.md) | Preallocated probability buffers | 3 |
| [T-095](./epic-05-memory-optimization/sprint-07/ticket-095-scenario-sampling-buffers.md) | Thread-local scenario sampling buffers | 3 |
| [T-096](./epic-05-memory-optimization/sprint-07/ticket-096-remove-noises-to-vec.md) | Remove noises.to_vec() clone | 2 |
| [T-097](./epic-05-memory-optimization/sprint-07/ticket-097-state-staging-buffer.md) | State staging buffer for cut computation | 5 |
| [T-098](./epic-05-memory-optimization/sprint-07/ticket-098-forward-costs-move.md) | Replace forward_costs.clone() with move | 1 |
| [T-099](./epic-05-memory-optimization/sprint-07/ticket-099-hashset-to-bitvec.md) | Replace HashSet with BitVec | 3 |
| [T-100](./epic-05-memory-optimization/sprint-07/ticket-100-trajectory-buffer.md) | Preallocate trajectory buffer | 3 |
| [T-101](./epic-05-memory-optimization/sprint-07/ticket-101-dhat-verification.md) | DHAT verification | 3 |

### Sprint 8: Validation and Documentation

| ID | Title | Points |
|----|-------|--------|
| [T-102](./epic-05-memory-optimization/sprint-08/ticket-102-dhat-comparison.md) | Comprehensive DHAT comparison | 3 |
| [T-103](./epic-05-memory-optimization/sprint-08/ticket-103-rss-stability.md) | RSS stability verification | 2 |
| [T-104](./epic-05-memory-optimization/sprint-08/ticket-104-performance-benchmark.md) | Performance benchmark comparison | 3 |
| [T-105](./epic-05-memory-optimization/sprint-08/ticket-105-update-memory-docs.md) | Update MEMORY_BEHAVIOR.md | 3 |
| [T-106](./epic-05-memory-optimization/sprint-08/ticket-106-remove-deprecated.md) | Remove deprecated code paths | 2 |
| [T-107](./epic-05-memory-optimization/sprint-08/ticket-107-user-monitoring-guide.md) | Create user monitoring guide | 2 |

---

## Key Finding: DHAT Profiling Results

**94.7% of heap allocations come from HiGHS LP solver:**

| Component | Bytes | Percentage |
|-----------|-------|------------|
| HiGHS Solver (HEkk/HFactor) | 83.5 GB | 94.7% |
| HiGHS Presolve | 2.8 GB | 3.2% |
| Rust Application | 1.8 GB | 2.0% |

See [HOT_PATH_ALLOCATION_AUDIT.md](../../docs/HOT_PATH_ALLOCATION_AUDIT.md) for full analysis.

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
