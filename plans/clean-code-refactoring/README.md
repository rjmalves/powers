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
| 5 | [Memory Optimization](./epic-05-memory-optimization/00-epic-overview.md) | 9 sprints | ✅ **Complete** |
| 6 | **Performance Evaluation Infrastructure** | 12 weeks | ⬜ Not Started |
| 7 | [Test Modernization](./epic-06-test-modernization/00-epic-overview.md) | 2 weeks | ⬜ Not Started |
| 8 | [Final Validation](./epic-07-performance-validation/00-epic-overview.md) | 1 week | ⬜ Not Started |

**Total Duration**: ~30-32 weeks

---

## 🆕 Epic 6: Performance Evaluation Infrastructure

**STATUS**: Ready to begin

Epic 6 is a comprehensive enterprise-grade profiling suite that must be completed before Test Modernization and Final Validation. It provides the tools needed to properly evaluate performance during those phases.

### [📁 Full Plan: performance-evaluation-infrastructure](../performance-evaluation-infrastructure/README.md)

| Sub-Epic | Name | Duration | Points |
|----------|------|----------|--------|
| 6.1 | Core Profiling Framework | 3 weeks | 42 |
| 6.2 | CPU & Execution Profiling | 2 weeks | 23 |
| 6.3 | Memory Profiling Suite | 2 weeks | 24 |
| 6.4 | Parallelism & Scalability Analysis | 2 weeks | 23 |
| 6.5 | Visualization & Reporting Dashboard | 2 weeks | 28 |
| 6.6 | Integration & Documentation | 1 week | 18 |

### Key Deliverables

- **CLI Tool**: `powers-profile run|compare|dashboard|scaling`
- **Collectors**: CPU (FlameGraph), Memory (DHAT/Massif/RSS), Parallel scaling
- **Outputs**: JSON (machine-readable), Markdown (reports), HTML (Plotly dashboards)
- **Scalability**: Tested from 1 to 192 cores (AWS c7a.48xlarge)

---

## Epic 5 - Memory Optimization: ✅ COMPLETE

### Key Findings

**glibc is the best allocator!** Alternative allocators (mimalloc, jemalloc) perform WORSE.

See [ALLOCATOR_COMPARISON.md](../../docs/ALLOCATOR_COMPARISON.md) and [SPRINT_09_FINAL_REPORT.md](../../docs/SPRINT_09_FINAL_REPORT.md).

| Allocator | Final RSS | Recommendation |
|-----------|-----------|----------------|
| glibc + malloc_trim | 253 MB | ✅ **Default** |
| mimalloc | 838 MB | ❌ 3.3x worse |
| jemalloc | 775 MB | ❌ 3.1x worse |

### Sprint Summary

| Sprint | Focus | Status |
|--------|-------|--------|
| 1-4 | Handler Staging Buffers & Pool Optimization | ✅ Complete |
| 5 | Deterministic Memory Allocation | ✅ Complete |
| 6 | HiGHS Solver Memory Optimization | ✅ Complete |
| 7 | Rust Application Allocation Optimization | ✅ Complete |
| 8 | Validation and Documentation | ✅ Complete |
| 9 | RSS Stabilization via Allocator Strategy | ✅ Complete |

### DHAT Profiling Results

**94.7% of heap allocations come from HiGHS LP solver:**

| Component | Bytes | Percentage |
|-----------|-------|------------|
| HiGHS Solver (HEkk/HFactor) | 83.5 GB | 94.7% |
| HiGHS Presolve | 2.8 GB | 3.2% |
| Rust Application | 1.8 GB | 2.0% |

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

---

*Last Updated: 2026-01-02*
