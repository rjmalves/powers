# Epic 5: Parallel Zero-Allocation Memory Optimization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Architecture Report**: [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md)
> **Allocation Audit**: [HOT_PATH_ALLOCATION_AUDIT.md](../../../docs/HOT_PATH_ALLOCATION_AUDIT.md)
> **Duration**: 8 sprints (16 weeks)
> **Status**: 🔄 In Progress (Sprint 6 complete, Sprints 7-8 planned)

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

This epic implements **parallel zero-allocation cut computation** for SDDP training, targeting production workloads with 500+ forward passes on 192+ core systems. 

### Key Finding: DHAT Analysis (Sprint 5 Discovery)

**94.7% of heap allocations come from the HiGHS LP solver**, not Rust application code:

| Component | Bytes Allocated | Percentage |
|-----------|-----------------|------------|
| HiGHS Solver (HEkk/HFactor) | 83.5 GB | 94.7% |
| HiGHS Presolve | 2.8 GB | 3.2% |
| Rust Application | 1.8 GB | 2.0% |
| Parquet I/O | 12.9 MB | 0.0% |

This finding drove the addition of Sprints 6-8 to address HiGHS-specific optimizations.

### Key Innovation: Handler Staging Buffers (Sprint 1-4)

Each `SddpTrainHandler` gets a lightweight staging buffer (~1.6 KB) that holds one computed cut and state. This enables:
- **Phase 1a**: Parallel cut computation (each handler → own staging buffer)
- **Phase 1b**: Sequential pool update (deterministic order, just copies)

---

## Goals

1. **Zero transient allocations** in cut computation hot path
2. **Full parallelism preserved** in Phase 1 cut computation
3. **Deterministic reproducibility** across runs (required constraint)
4. **Optimized pool memory model** (eliminate HashMap, Arc overhead)
5. **HiGHS allocation reduction** ≥30% (NEW - Sprint 6)
6. **Rust allocation reduction** ≥50% (NEW - Sprint 7)
7. **~5-15% training speedup** from combined optimizations

## Non-Goals

- SoA conversion (deferred - complexity vs benefit)
- Algorithm changes
- New external dependencies
- Lock-free concurrent pool updates (too complex, not needed)
- Modifying HiGHS source code

---

## Sprint Overview

### Sprints 1-4: Foundation & Architecture (COMPLETE ✅)

| Sprint | Focus | Status |
|--------|-------|--------|
| Sprint 1 | Handler Staging Buffers | ✅ Complete |
| Sprint 2 | Training Loop Integration | ✅ Complete |
| Sprint 3 | Pool Memory Model Optimization | ✅ Complete |
| Sprint 4 | Pool Architecture Refinement | ✅ Complete |

### Sprint 5: Deterministic Memory Allocation (COMPLETE ✅)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-080 | Audit and eliminate remaining add_row calls | 5 | ✅ |
| T-081 | Thread-local buffers for try_add_row | 3 | ✅ |
| T-082 | HiGHS solver warmup after preallocation | 3 | ✅ |
| T-083 | Preallocate coordinator result buffers | 3 | ✅ |
| T-084 | Preallocate trajectory buffers | 5 | ✅ |
| T-085 | DHAT profiling verification | 3 | 🔄 Led to Sprint 6-8 |
| T-086 | Benchmark and document | 2 | ✅ |

**Total**: 24 points

### Sprint 6: HiGHS Solver Memory Optimization (NEW ⬜)

DHAT revealed 94.7% of allocations from HiGHS. This sprint targets HiGHS-specific optimizations.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-087 | Investigate HiGHS warm-start API | 5 | ⬜ |
| T-088 | Implement batch changeRowBounds | 3 | ⬜ |
| T-089 | Integrate batch bound updates | 5 | ⬜ |
| T-090 | Verify HiGHS debug mode disabled | 2 | ⬜ |
| T-091 | Evaluate presolve settings | 3 | ⬜ |
| T-092 | Disable HiGHS internal threading | 2 | ⬜ |
| T-093 | DHAT verification | 3 | ⬜ |

**Total**: 23 points

**Target**: ≥30% reduction in HiGHS allocations

### Sprint 7: Rust Application Allocation Optimization (NEW ⬜)

Target the remaining 2% from Rust application code.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-094 | Preallocated probability buffers | 3 | ⬜ |
| T-095 | Thread-local scenario sampling buffers | 3 | ⬜ |
| T-096 | Remove noises.to_vec() clone | 2 | ⬜ |
| T-097 | State staging buffer for cut computation | 5 | ⬜ |
| T-098 | Replace forward_costs.clone() with move | 1 | ⬜ |
| T-099 | Replace HashSet with BitVec | 3 | ⬜ |
| T-100 | Preallocate trajectory buffer | 3 | ⬜ |
| T-101 | DHAT verification | 3 | ⬜ |

**Total**: 23 points

**Target**: ≥50% reduction in Rust allocations

### Sprint 8: Validation and Documentation (NEW ⬜)

Final verification and documentation.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-102 | Comprehensive DHAT comparison | 3 | ⬜ |
| T-103 | RSS stability verification | 2 | ⬜ |
| T-104 | Performance benchmark comparison | 3 | ⬜ |
| T-105 | Update MEMORY_BEHAVIOR.md | 3 | ⬜ |
| T-106 | Remove deprecated code paths | 2 | ⬜ |
| T-107 | Create user monitoring guide | 2 | ⬜ |

**Total**: 15 points

---

## Architecture Overview

### Current State (After Sprint 5)

```
Phase 1a: par_iter_mut → staging buffers (no allocation)
Phase 1b: Sequential copy to pools (deterministic order)
Phase 2:  Cut selection on updated slots
Phase 3:  Apply cuts (parallel)
```

### DHAT Findings (Driving Sprint 6-8)

Top HiGHS allocation sites:
1. `HFactor::setupGeneral` - 44.9% (factorization setup per solve)
2. `HEkk::computeDual` - 22.7% (dual simplex work arrays)
3. `changeRowBounds` - 3.4% (3 million individual calls)
4. `debugDualSimplex` - 0.06% (debug string allocations)

Top Rust allocation sites:
1. `uniform_prob_by_count()` - Per-cut probability vectors
2. `sample_scenario()` - Per-iteration scenario vectors
3. `state.clone()` - Per-cut state cloning
4. HashSet allocations in cut selection

---

## Dependencies

- **Requires**:
  - Epic 4 complete ✅ (FCF simplified, pools preallocated)
  - Existing infrastructure: `CutComputationBuffers`, `compute_cut_into_slot()`

- **Enables**:
  - Epic 7: Performance Validation (final verification)

---

## Acceptance Criteria

### Sprints 1-5 (COMPLETE)
- [x] Staging buffer infrastructure
- [x] Pool memory model optimized
- [x] No Arc/HashMap in pools
- [x] Thread-local buffers for edge cases
- [x] 567+ tests pass

### Sprint 6 (HiGHS Optimization)
- [ ] HiGHS warm-start investigated
- [ ] Batch bound updates implemented
- [ ] Debug mode verified disabled
- [ ] ≥30% HiGHS allocation reduction

### Sprint 7 (Rust Optimization)
- [ ] Preallocated buffers for probabilities, scenarios
- [ ] Eliminated clones in hot path
- [ ] ≥50% Rust allocation reduction

### Sprint 8 (Validation)
- [ ] DHAT comparison complete
- [ ] RSS stability verified
- [ ] No performance regression
- [ ] Documentation complete

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence | Low | **CRITICAL** | Golden tests after every change |
| HiGHS warm-start not available | Medium | Medium | Document findings, alternatives |
| Batch API behavior differs | Low | Medium | Comprehensive testing |
| Buffer sizing incorrect | Low | Low | Conservative sizing + resize |

---

## Memory Budget

| Component | Size | Count | Total |
|-----------|------|-------|-------|
| CutStagingBuffer | ~1.6 KB | 500 handlers | ~800 KB |
| Thread-local buffers | ~10 KB | 192 threads | ~1.9 MB |
| Eliminated allocations | ~18 MB/run | - | **-18 MB** |
| Expected HiGHS reduction | ~30% | - | **-25 GB** |
| Expected Rust reduction | ~50% | - | **-0.9 GB** |

---

## Key Files

| Component | Location |
|-----------|----------|
| BendersCutPool | `src/cut.rs:276-510` |
| VisitedStatePool | `src/state.rs:480-592` |
| CutComputationBuffers | `src/memory/buffers.rs:78-180` |
| SddpTrainHandler | `src/sddp/mod.rs:323-450` |
| ParallelHandlerCoordinator | `src/algorithm/coordinator.rs:46-220` |
| HiGHS options | `src/subproblem.rs:set_default_solver_options()` |
| Allocation audit | `docs/HOT_PATH_ALLOCATION_AUDIT.md` |

---

## Definition of Done

- [ ] All sprint acceptance criteria met
- [ ] DHAT shows significant allocation reduction
- [ ] RSS stable after warmup phase
- [ ] Golden tests pass with all paths
- [ ] 567+ tests pass
- [ ] Benchmarks show no regression
- [ ] Architecture documented
- [ ] Deprecated paths removed
