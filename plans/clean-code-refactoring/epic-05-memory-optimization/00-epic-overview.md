# Epic 5: Parallel Zero-Allocation Memory Optimization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Architecture Report**: [PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md](../../../docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md)
> **Duration**: 3 sprints (6 weeks)
> **Status**: 🔄 Major Revision - Restarting

---

## ⚠️ CRITICAL REMINDER

**Algorithm correctness is non-negotiable.** Memory optimization must not change any numerical results. Golden tests must pass after every change.

If any test fails or results diverge: **STOP and investigate before proceeding.**

---

## Executive Summary

This epic implements **parallel zero-allocation cut computation** for SDDP training, targeting production workloads with 500+ forward passes on 192+ core systems. The design preserves full parallelism while eliminating ~18 MB of transient allocations per training run.

### Key Innovation: Handler Staging Buffers

Each `SddpTrainHandler` gets a lightweight staging buffer (~1.6 KB) that holds one computed cut and state. This enables:
- **Phase 1a**: Parallel cut computation (each handler → own staging buffer)
- **Phase 1b**: Sequential pool update (deterministic order, just copies)

This architecture preserves **full parallelism** while achieving **zero allocation**.

---

## What Was Previously Attempted

Sprint 1 (old) implemented sequential zero-allocation APIs:
- `update_cut_and_state_slots()` ✅
- `compute_cut_into_slot()` ✅
- `compute_cuts_into_slots()` ✅ (but sequential due to mutable pool access)

**Problem**: The sequential approach loses parallelism in Phase 1. For 500 forward passes on 192 cores, this is unacceptable.

**Solution**: Handler staging buffers enable parallel-then-sequential execution.

---

## Goals

1. **Zero transient allocations** in cut computation hot path
2. **Full parallelism preserved** in Phase 1 cut computation
3. **Deterministic reproducibility** across runs (required constraint)
4. **Optimized pool memory model** (eliminate HashMap, Arc overhead)
5. **~5-15% training speedup** from combined optimizations

## Non-Goals

- SoA conversion (deferred - complexity vs benefit)
- Algorithm changes
- New external dependencies
- Lock-free concurrent pool updates (too complex, not needed)

---

## Architecture Overview

### Current State (Problematic)

```
Phase 1: par_iter_mut → CutData { Vec, Vec } → ALLOCATES
Phase 2: Sequential copy to pools → copies then drops allocations
```

### Target State (This Epic)

```
Phase 1a: par_iter_mut → staging buffers (no allocation)
Phase 1b: Sequential copy to pools (deterministic order)
Phase 2:  Cut selection on updated slots
Phase 3:  Apply cuts (parallel)
```

### New Data Structures

```rust
/// Per-handler staging buffer (~1.6 KB for 100-dim state)
pub struct CutStagingBuffer {
    pub cut_coefficients: Vec<f64>,      // Preallocated
    pub cut_rhs: f64,
    pub state_coefficients: Vec<f64>,    // Preallocated
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub timing: BackwardPhase1Timing,
}
```

---

## Sprint Overview

### Sprint 1: Handler Staging Buffers (Foundation)

Create the staging buffer infrastructure and wire into handlers.

| Ticket | Title | Points |
|--------|-------|--------|
| T-060 | Create CutStagingBuffer struct | 2 |
| T-061 | Add staging buffer to SddpTrainHandler | 2 |
| T-062 | Implement compute_cut_into_staging() on handler | 5 |
| T-063 | Add update_from_staging() to pools | 3 |
| T-064 | Update ParallelHandlerCoordinator for parallel-then-sequential | 5 |

**Total**: 17 points

### Sprint 2: Training Loop Integration

Wire the new path into production and validate.

| Ticket | Title | Points |
|--------|-------|--------|
| T-065 | Update backward_pass.rs to use staging path | 5 |
| T-066 | Golden tests validation | 2 |
| T-067 | Benchmark parallel vs sequential | 3 |
| T-068 | DHAT profiling to verify zero allocations | 3 |

**Total**: 13 points

### Sprint 3: Pool Memory Model Optimization

Eliminate HashMap and Arc overhead for additional performance.

| Ticket | Title | Points |
|--------|-------|--------|
| T-069 | Remove Arc wrapper from BendersCutPool | 3 |
| T-070 | Remove HashMap from BendersCutPool | 3 |
| T-071 | Create ConcreteState enum for VisitedStatePool | 5 |
| T-072 | Migrate VisitedStatePool to enum dispatch | 5 |
| T-073 | Cleanup deprecated CutData path | 2 |
| T-074 | Final performance validation | 3 |

**Total**: 21 points

### Sprint 4: Pool Architecture Refinement

Eliminate layout duplication in state pool for optimal memory efficiency.

| Ticket | Title | Points |
|--------|-------|--------|
| T-075 | Create StateData struct (pure coefficient data) | 2 |
| T-076 | Refactor VisitedStatePool to shared layout | 5 |
| T-077 | Update FCF for shared layout state access | 3 |
| T-078 | Remove ConcreteState enum | 2 |
| T-079 | Benchmark memory usage and performance | 2 |

**Total**: 14 points

**Rationale**: Sprint 3's `ConcreteState` enum stored `StateLayout` redundantly in each state.
Sprint 4 refactors to store layout once in the pool, eliminating ~100KB of redundant allocations
for typical workloads (500 states × ~200 bytes layout overhead).


---

## Dependencies

- **Requires**:
  - Epic 4 complete ✅ (FCF simplified, pools preallocated)
  - Existing infrastructure: `CutComputationBuffers`, `compute_cut_into_slot()`

- **Enables**:
  - Epic 7: Performance Validation (final verification)

---

## Acceptance Criteria

### Sprint 1 Completion
- [x] `CutStagingBuffer` struct implemented with tests
- [x] `SddpTrainHandler` contains staging buffer
- [x] `compute_cut_into_staging()` method works
- [x] `ParallelHandlerCoordinator` uses parallel-then-sequential pattern
- [x] All 549+ tests pass

### Sprint 2 Completion
- [x] Training loop uses staging buffer path
- [x] Golden tests pass (bit-for-bit identical)
- [ ] DHAT shows zero allocations in cut computation (deferred to Epic 7)
- [ ] Benchmark shows no regression (deferred to Epic 7)

### Sprint 3 Completion
- [x] No Arc wrapper in BendersCutPool
- [x] No HashMap in BendersCutPool
- [x] VisitedStatePool uses enum dispatch
- [x] `CutData` path removed from production
- [ ] 5-15% speedup measured (deferred to Epic 7)

### Sprint 4 Completion
- [x] StateData struct replaces ConcreteState in pool
- [x] StateLayout stored once per pool (not per state)
- [x] ConcreteState enum removed
- [x] Memory usage reduced (verified by test)
- [x] All 567+ tests pass

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence | Low | **CRITICAL** | Golden tests after every change |
| Arc removal breaks sharing | Medium | Medium | Careful audit of all usages |
| Borrow checker conflicts | Medium | Medium | May need RefCell in edge cases |
| Enum dispatch overhead | Low | Low | Benchmark confirms jump tables fast |
| Phase 1b sequential bottleneck | Low | Low | Copy is ~0.1ms for 500 handlers |

---

## Memory Budget

| Component | Size | Count | Total |
|-----------|------|-------|-------|
| CutStagingBuffer | ~1.6 KB | 500 handlers | ~800 KB |
| Thread-local buffers | ~10 KB | 192 threads | ~1.9 MB |
| Eliminated allocations | ~18 MB/run | - | **-18 MB** |

**Net: ~15 MB reduction per training run**

---

## Key Files

| Component | Location |
|-----------|----------|
| BendersCutPool | `src/cut.rs:276-510` |
| VisitedStatePool | `src/state.rs:480-592` |
| CutComputationBuffers | `src/memory/buffers.rs:78-180` |
| SddpTrainHandler | `src/sddp/mod.rs:323-450` |
| ParallelHandlerCoordinator | `src/algorithm/coordinator.rs:46-220` |
| backward_pass execution | `src/algorithm/backward_pass.rs:250-315` |

---

## Definition of Done

- [ ] All sprint acceptance criteria met
- [ ] Zero allocations in cut computation verified by DHAT
- [ ] Golden tests pass with new path
- [ ] 549+ tests pass
- [ ] Benchmarks show ≥5% improvement
- [ ] Architecture documented
- [ ] Deprecated paths removed
