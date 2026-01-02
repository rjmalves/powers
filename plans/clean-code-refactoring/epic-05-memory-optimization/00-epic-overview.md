# Epic 5: Parallel Zero-Allocation Memory Optimization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Architecture Report**: [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md)
> **Allocation Audit**: [HOT_PATH_ALLOCATION_AUDIT.md](../../../docs/HOT_PATH_ALLOCATION_AUDIT.md)
> **Sprint 6 Analysis**: [DHAT_SPRINT6_ANALYSIS.md](../../../docs/DHAT_SPRINT6_ANALYSIS.md)
> **Sprint 7 Analysis**: [DHAT_SPRINT7_ANALYSIS.md](../../../docs/DHAT_SPRINT7_ANALYSIS.md)
> **RSS Investigation**: [HIGHS_RSS_MEMORY_INVESTIGATION.md](../../../docs/HIGHS_RSS_MEMORY_INVESTIGATION.md)
> **Duration**: 8 sprints (16 weeks)
> **Status**: ⚠️ Sprint 8 complete but RSS objectives not met - needs investigation

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

This epic implements **parallel zero-allocation cut computation** for SDDP training, targeting production workloads with 500+ forward passes on 192+ core systems.

### Cumulative Results (Sprint 6 + Sprint 7)

| Metric | Before Sprint 6 | After Sprint 7 | Total Reduction |
|--------|-----------------|----------------|-----------------|
| Total Bytes Allocated | 88.19 GB | 45.43 GB | **48.5%** |
| Total Allocation Blocks | 159.0 M | 43.3 M | **72.8%** |
| HFactor::setupGeneral | 39.58 GB | <0.01 GB | **>99%** |
| changeRowBounds blocks | 98.4 M | 0.4 M | **99.6%** |

### Sprint 8 Results (2025-12-31)

| Metric | Post-Sprint 7 | Sprint 8 | Change |
|--------|---------------|----------|--------|
| Total Bytes Allocated | 45.43 GB | 50.32 GB | **+10.8%** |
| Total Allocation Blocks | 43.3 M | 46.6 M | **+7.6%** |
| Sum of Max Bytes | 468.4 MB | 526.6 MB | **+12.4%** |

**Analysis**: DHAT regression is expected due to per-iteration Model creation (more allocations over lifetime). However, manual RSS observation suggests RSS stability objective was **not achieved**. Further investigation needed.

### Key Findings

1. **Sprint 6 Discovery**: `reuse_forward_basis()` was counterproductive - disabling it eliminated 95% of HFactor allocations
2. **Sprint 7 Conclusion**: HEkkDual (41.9 GB) is inherent to HiGHS dual simplex - cannot be reduced without solver modifications
3. **RSS Growth Analysis**: HiGHS internal buffers grow but never shrink - **Per-Iteration Model Architecture** proposed
4. **Sprint 8 Issue**: Per-iteration Model lifecycle implemented, but RSS may still be growing - needs proper RSS measurement (not DHAT)

---

## Goals

1. **Zero transient allocations** in cut computation hot path ✅
2. **Full parallelism preserved** in Phase 1 cut computation ✅
3. **Deterministic reproducibility** across runs ✅
4. **Optimized pool memory model** ✅
5. **HiGHS allocation reduction** ≥30% → **48.5% achieved** ✅
6. **Rust allocation reduction** → **Complete** (Sprint 7)
7. **RSS memory stability** → ⚠️ **Needs Investigation** (Sprint 8 architecture complete, RSS validation pending)

## Non-Goals

- SoA conversion (deferred - complexity vs benefit)
- Modifying HiGHS source code
- Lock-free concurrent pool updates

---

## Sprint Overview

### Sprints 1-5: Foundation & Architecture (COMPLETE ✅)

| Sprint | Focus | Status |
|--------|-------|--------|
| Sprint 1 | Handler Staging Buffers | ✅ Complete |
| Sprint 2 | Training Loop Integration | ✅ Complete |
| Sprint 3 | Pool Memory Model Optimization | ✅ Complete |
| Sprint 4 | Pool Architecture Refinement | ✅ Complete |
| Sprint 5 | Deterministic Memory Allocation | ✅ Complete |

### Sprint 6: HiGHS Solver Memory Optimization (COMPLETE ✅)

**Results**: 48.5% byte reduction, 72.7% block reduction (exceeded 30% target)

| Ticket | Title | Status |
|--------|-------|--------|
| T-087 | Investigate HiGHS warm-start API | ✅ |
| T-088 | Implement batch changeRowBounds | ✅ |
| T-089 | Integrate batch bound updates | ✅ |
| T-090-T-093 | Debug/presolve/threading verification | ✅ |

### Sprint 7: Comprehensive Memory Optimization (COMPLETE ✅)

**Results**: Validated Sprint 6 gains, confirmed HEkkDual as inherent, 8/10 tickets completed

| Ticket | Title | Status |
|--------|-------|--------|
| T-094 | Remove `reuse_forward_basis()` code entirely | ✅ |
| T-095 | Document HiGHS basis reuse guidelines | ✅ |
| T-096 | Investigate HEkkDual allocation sources | ✅ (inherent) |
| T-097 | Investigate HSimplexNla debug allocations | ✅ (inherent) |
| T-098 | Batch cut constraint bound updates | ✅ |
| T-099 | Preallocated probability buffers | ✅ |
| T-100 | Thread-local scenario sampling buffers | ⏸️ Deferred → T-100-r |
| T-101 | Eliminate noises.to_vec() clones | ✅ |
| T-102 | Replace HashSet with BitVec | ⏸️ Deferred → T-102-r |
| T-103 | DHAT verification | ✅ |

### Sprint 8 (REVISED): Per-Iteration Model Architecture with Optional Basis ⚠️

**Focus**: Fundamental architecture change - keep Problem as source of truth, create Model per iteration with optional basis for simulation reproducibility.

> ⚠️ **SUPERSEDES** original Sprint 8 (Model Rebuild Strategy)
> ⚠️ **STATUS**: Architecture complete, but RSS objectives not validated

**Key Changes**:
1. `Problem` = persistent LP definition (source of truth)
2. `Model` = transient per-iteration (created/dropped each iteration)
3. `StoredBasis` = optional, for warm-starting (training only)
4. Dual cut updates (both Problem and Model during backward pass)
5. Optional basis for simulation reproducibility from loaded FCF

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-110 | Implement `Problem::create_model()` | 5 | ✅ Complete |
| T-111 | Add Problem modification methods | 3 | ✅ Complete |
| T-112 | Implement StoredBasis and basis transfer | 3 | ✅ Complete |
| T-113 | Refactor Subproblem for dual storage | 5 | ✅ Complete |
| T-114 | Per-iteration lifecycle with optional basis | 5 | ✅ Complete |
| T-115 | Dual cut update (Problem + Model) | 3 | ✅ Complete |
| T-116 | Update realize_and_solve | 3 | ✅ Complete |
| T-117 | Training loop integration | 5 | ✅ Complete |
| T-118 | Simulation mode configuration | 3 | ✅ Complete |
| T-100-r | Scenario sampling indices buffer (revised) | 3 | 📋 Deferred |
| T-102-r | CutIdSet type implementation | 3 | 📋 Deferred |
| T-119 | Determinism test (with/without basis) | 3 | ✅ Complete |
| T-120 | RSS verification tests | 3 | ⚠️ Needs Investigation |
| T-121 | Performance benchmarks | 3 | ✅ Complete |
| T-122 | Golden test validation | 2 | ✅ Complete (pre-existing issues excluded) |
| T-123 | Training loop integration | 5 | ✅ Complete |
| T-124 | Simulation integration | 3 | ✅ Complete |
| T-125 | E2E validation | 3 | ✅ Complete |
| T-126 | Memory regression test | 2 | ⚠️ DHAT shows regression |

**Total**: 52 points (~3 weeks)

**DHAT Results (2025-12-31)**:
- Total bytes: +10.8% (expected due to per-iteration Model creation)
- RSS stability: **Needs investigation** (DHAT cannot measure this)

**Actual Outcomes**:
- ✅ Clean architecture (Problem as source of truth)
- ✅ Simulation reproducibility (optional basis)
- ✅ Enables future FCF persistence feature
- ✅ <5% overhead from per-iteration Model creation
- ⚠️ Memory reclaimed every iteration - **NEEDS VERIFICATION**

See: [Sprint 8 Revised Overview](./sprint-08-revised/00-sprint-overview.md)

---

## Sprint 8 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ITERATION LIFECYCLE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Training (use_basis=true):                                          │
│    1. create_iteration_model(true) → Apply cached basis             │
│    2. Forward + Backward passes → update_cut_dual() for cuts        │
│    3. finalize_iteration(true) → Cache basis, drop Model            │
│                                                                      │
│  Simulation (use_basis=false):                                       │
│    1. clear_cached_basis() → Ensure reproducibility                 │
│    2. create_iteration_model(false) → Cold-start                    │
│    3. Forward pass only                                              │
│    4. finalize_iteration(false) → Drop Model, don't cache           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual Cut Update Pattern

During backward pass, cuts update **both** Problem and Model:
- Problem: For next iteration's Model
- Model: For current backward pass (stage t-1 cut needed at stage t-2)

---

## Superseded Tickets (Original Sprint 8)

| Original ID | Title | Status |
|-------------|-------|--------|
| T-104 | Implement Subproblem::rebuild_model() | ❌ Superseded by T-113/T-114 |
| T-105 | Add RSS monitoring utilities | ⏸️ Merged into T-120 |
| T-107 | Integrate rebuild into training loop | ❌ Superseded by T-117 |
| T-108 | Add rebuild configuration options | ❌ Superseded by T-118 |
| T-106 | DHAT verification & benchmarking | ⏸️ Merged into T-121 |

---

## DHAT Findings Summary

### Before Sprint 6 (Baseline)
| Component | Bytes | Percentage |
|-----------|-------|------------|
| HiGHS (total) | 83.5 GB | 94.7% |
| Rust/Powers | 6.12 GB | 5.3% |
| **Total** | 88.19 GB | 100% |

### After Sprint 7 (Current)
| Component | Bytes | Percentage | vs Baseline |
|-----------|-------|------------|-------------|
| HEkkDual/HEkk | 41.90 GB | 92.2% | **Inherent** |
| Other HiGHS | 2.01 GB | 4.4% | Minimized |
| Powers/SDDP | 0.57 GB | 1.2% | -90% |
| Other | 0.95 GB | 2.1% | - |
| **Total** | 45.43 GB | 100% | **-48.5%** |

### Key Insight

**92.2% of remaining allocations are HEkkDual** (HiGHS dual simplex working vectors). This is inherent to the algorithm and cannot be reduced without modifying HiGHS source code.

---

## Acceptance Criteria

### Completed
- [x] Staging buffer infrastructure (Sprints 1-4)
- [x] Pool memory model optimized (Sprints 3-4)
- [x] Thread-local buffers for edge cases (Sprint 5)
- [x] HiGHS allocation reduction ≥30% → **48.5%** (Sprint 6)
- [x] `reuse_forward_basis()` removed (Sprint 7)
- [x] HEkkDual investigation complete (Sprint 7)
- [x] Batch cut bounds implemented (Sprint 7)

### Sprint 8 (Complete)
- [x] Per-iteration Model lifecycle implemented
- [x] Problem as persistent source of truth
- [x] Optional basis for simulation
- [x] Dual cut updates working
- [x] Determinism verified (results same ±/- basis)
- [x] Performance benchmarks acceptable (<5% overhead)
- [x] Golden tests pass (pre-existing issues excluded)

### Sprint 8 (Needs Investigation)
- [ ] ⚠️ RSS verified to decrease between iterations - **DHAT cannot measure, needs RSS profiling**
- [ ] Scenario indices buffer (T-100-r) - Deferred
- [ ] CutIdSet type (T-102-r) - Deferred

---

## Definition of Done

- [x] ≥30% HiGHS allocation reduction → **48.5% achieved**
- [ ] ⚠️ RSS stable between iterations - **Needs proper RSS measurement (not DHAT)**
- [x] Golden tests pass with all paths (pre-existing issues excluded)
- [x] 589+ tests pass
- [x] Simulation reproducible without basis (infrastructure ready)
- [x] Architecture documented
- [x] Sprint 8 core tickets complete (T-110 through T-125)
- [ ] T-126 Memory regression - DHAT shows expected regression due to architecture change

---

## Key Files

| Component | Location |
|-----------|----------|
| BendersCutPool | `src/cut.rs` |
| VisitedStatePool | `src/state.rs` |
| CutComputationBuffers | `src/memory/buffers.rs` |
| SddpTrainHandler | `src/sddp/mod.rs` |
| Subproblem | `src/subproblem.rs` |
| Solver wrapper | `src/solver.rs` |
| StoredBasis | `src/solver.rs` (Sprint 8) |
| CutIdSet | `src/memory/cut_id_set.rs` (Sprint 8) |

---

## Architecture Documentation

| Document | Purpose |
|----------|---------|
| [Sprint 8 Revised Overview](./sprint-08-revised/00-sprint-overview.md) | Per-iteration Model architecture |
| [Sprint 8 Revision Summary](./sprint-08-revised/SPRINT_8_REVISION_SUMMARY.md) | Design evolution |
| [HOT_PATH_ALLOCATION_AUDIT.md](../../../docs/HOT_PATH_ALLOCATION_AUDIT.md) | Original allocation audit |
| [HIGHS_WARM_START_INVESTIGATION.md](../../../docs/HIGHS_WARM_START_INVESTIGATION.md) | Basis reuse investigation |
| [HEKKDUAL_INVESTIGATION.md](../../../docs/HEKKDUAL_INVESTIGATION.md) | HEkkDual/HSimplexNla investigation |
| [DHAT_SPRINT6_ANALYSIS.md](../../../docs/DHAT_SPRINT6_ANALYSIS.md) | Sprint 6 results |
| [DHAT_SPRINT7_ANALYSIS.md](../../../docs/DHAT_SPRINT7_ANALYSIS.md) | Sprint 7 results |
| [HIGHS_RSS_MEMORY_INVESTIGATION.md](../../../docs/HIGHS_RSS_MEMORY_INVESTIGATION.md) | RSS growth analysis |
| [Sprint 08 RSS Analysis](../../../docs/SPRINT_08_RSS_ANALYSIS.md) | Sprint 8 RSS investigation results |
| [Sprint 9 Overview](./sprint-09/00-sprint-overview.md) | RSS Stabilization via Allocator Strategy |

---

## Sprint 9: RSS Stabilization via Allocator Strategy 📋

**Focus**: Test alternative allocators (mimalloc, jemalloc, malloc_trim) to achieve stable RSS between iterations.

> **Status**: 📋 Planned
> **Root Cause**: glibc malloc does not return freed memory to the OS

**Key Discovery from Sprint 8**:
The per-iteration Model architecture is correctly implemented, but RSS continues to grow because glibc malloc does not return freed memory to the OS. This is documented glibc behavior.

**Solution Approach**:
1. Test mimalloc (already optional dependency)
2. Add and test jemalloc
3. Test malloc_trim as fallback
4. Compare results and select winner
5. Make winning allocator the default

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-130 | Create RSS measurement test harness | 3 | 📋 |
| T-131 | Test mimalloc allocator RSS behavior | 3 | 📋 |
| T-132 | Add jemalloc as optional dependency | 2 | 📋 |
| T-133 | Test jemalloc allocator RSS behavior | 3 | 📋 |
| T-134 | Test malloc_trim after finalize_iteration | 2 | 📋 |
| T-135 | Compare allocator results and select winner | 2 | 📋 |
| T-136 | Make winning allocator the default | 3 | 📋 |
| T-137 | Validate all tests pass with new default | 2 | 📋 |
| T-138 | Performance benchmark with new allocator | 3 | 📋 |
| T-139 | Document allocator configuration | 2 | 📋 |
| T-140 | Add RSS stability CI check | 3 | 📋 |

**Total**: 28 points (~2 weeks)

**Success Criteria**:
- [ ] RSS stable between iterations (no growth after warmup)
- [ ] Best allocator becomes default
- [ ] All 589+ tests pass with new default
- [ ] Performance benchmarks show no regression (>5% slower)

See: [Sprint 9 Overview](./sprint-09/00-sprint-overview.md)
