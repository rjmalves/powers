# Master Plan: Preallocation Refactoring for HPC Scalability

## Executive Summary

Complete the memory preallocation infrastructure in POWE.RS to achieve 100% deterministic memory allocation for HPC environments. This enables predictable performance, memory-locked pages, and better cache utilization on systems with hundreds of cores.

## Goals & Non-Goals

### Goals

- **100% Memory Determinism**: Zero allocations during SDDP training hot paths
- **15-25% Runtime Improvement**: Eliminate allocation overhead and improve cache locality
- **HPC-Ready Scalability**: Predictable memory footprint for cluster deployment
- **Maintain Correctness**: Solution quality unchanged (objective within 0.001%)

### Non-Goals (Explicit Scope Exclusions)

- Full SoA graph refactoring (hybrid approach only, per GRAPH_TO_SOA_REFACTORING_ANALYSIS.md)
- Markovian graph topology changes (preserve current graph structure)
- HiGHS internal optimization (focus on API-level preallocation)
- Test infrastructure fixes (rely on examples for validation)

## Architecture Overview

### Current State (Post Epic 1c)

The codebase has achieved full memory determinism for cut operations:
- **Memory Module** (`src/memory/`): Simplified to `CutComputationBuffers` only (~437 lines) ✅
- **Cut Computation Buffers**: Hardened with capacity enforcement ✅
- **HiGHS Constraint Preallocation**: Epic 1 complete with 23% improvement ✅
- **Deterministic Slots**: Formula-based slot calculation, no dynamic tracking ✅
- **State/Subproblem Separation**: Clean extraction pattern ✅

**Removed in Epic 1c**:
- `SizingInfo` (1,574 lines) - never used in production
- `DeepSizeEstimate` (474 lines) - incorrect assumptions
- `Buffer<T>`, `BufferPool<T>`, `ThreadLocalBuffers` - unused

**Remaining Gaps**:
1. FCF: `with_capacity()` exists but not consistently used in production
2. Handler Data: Hot path data not in contiguous blocks

### Target State

After implementation:
- **HiGHS Constraints**: Pre-allocated at training start, deterministic slot placement ✅
- **No Fallbacks**: Panic on slot exhaustion (not fallback) ✅
- **Deterministic Slots**: `slot = (iteration - 1) * num_forward_passes + forward_pass_idx` ✅
- **FCF Pools**: Always created with `with_capacity()` using runtime parameters
- **Handler Blocks**: Hot data (loads, inflows, storage) in contiguous SoA blocks
- **Memory Profile**: Flat after initialization (±1% variation)

### Key Design Decisions

1. **Deterministic Cut Slot Calculation**: Use formula `(iteration-1) * num_fp + fp_idx` instead of dynamic slot allocation. Each (iteration, forward_pass) pair gets exactly one slot. ✅

2. **No Dynamic Fallback**: If slot calculation exceeds preallocated count, panic. This forces correct sizing and ensures memory determinism. ✅

3. **Store Slot in Cut**: Add `slot_index: Option<usize>` to `BendersCut` for O(1) deactivation lookup (no linear search). ✅

4. **Hybrid SoA Approach**: Keep graph for topology (cold path), add contiguous blocks for hot data (loads, inflows, storage). Preserves Markovian graph extensibility.

5. **Runtime Parameter Threading**: Since `SizingInfo` was removed, FCF preallocation uses runtime parameters (`num_forward_passes`, `num_iterations`, `max_state_dim`) computed inline.

## Technical Approach

### Core Abstractions

- **`compute_cut_slot(iteration, forward_pass_idx)`**: Deterministic O(1) slot calculation ✅
- **`BendersCut.slot_index`**: Stored slot for O(1) deactivation ✅
- **`CutComputationBuffers`**: Thread-local buffers with capacity enforcement ✅
- **`RealizationBlock`**: Contiguous storage for stage-wise load/inflow/storage data
- **`SubproblemBlock`**: Contiguous storage for subproblem hot data

### Data Flow

```
Training start:
    ↓
Calculate max_cuts = num_iterations × num_forward_passes
    ↓
Subproblem::preallocate_cut_constraints(max_cuts, num_forward_passes)
    ↓
Training loop (zero allocations):
    - add_cut: compute_cut_slot(iter, fp) → change coefficients + bounds
    - remove_cut: cut.slot_index → relax bounds
    - No slot tracking, no free list
```

### Parallelism Strategy

- Each thread has deterministic memory footprint
- No allocation contention between threads
- Thread-local buffers initialized via `rayon::broadcast()` ✅
- Cut slots computed per-subproblem (no cross-thread contention)

### Performance Strategy

- **HiGHS**: Eliminate dynamic `Highs_addRow` calls ✅ (23% achieved)
- **Slot Tracking Removal**: Eliminate Vec/HashMap overhead ✅ (included in Epic 1b)
- **Memory Module Cleanup**: Remove 87% of dead code ✅ (Epic 1c)
- **FCF**: Eliminate Vec/HashMap reallocation (1-3% gain expected)
- **Handler Blocks**: Improve cache locality for hot data (6-10% gain expected)
- **Total**: 25-35% cumulative improvement expected

## Phases & Milestones

| Phase | Epic | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | HiGHS Constraint Preallocation | 2 weeks | ✅ Complete - 23% improvement |
| 1b | Full Memory Determinism | 1 week | ✅ Complete - Deterministic slots, no fallbacks |
| 1c | Memory Module Cleanup | 3-5 days | ✅ Complete - 87% code reduction, hardened buffers |
| 2 | FCF Full Preallocation | 0 days | ✅ Complete - Already implemented via reserve() |
| 3 | Handler-Level SoA Blocks | 2-3 weeks | ⬜ **Next** - Hot data in contiguous blocks |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HiGHS API behavior change | Low | High | Pin HiGHS version, extensive testing |
| Numerical instability from relaxed bounds | Low | Medium | Test with diverse problem sizes |
| Cache effects vary by hardware | Medium | Low | Benchmark on target HPC system |
| Presolve removes inactive constraints | Medium | Medium | Disable presolve or use minimal bounds |

## Success Metrics

- [x] **Performance**: ≥15% runtime improvement on 07-par-model example ✅ (23% achieved)
- [x] **Memory Module Cleanup**: 87% code reduction ✅ (3,424 → 437 lines)
- [x] **Capacity Enforcement**: Cut buffers panic on overflow ✅
- [x] **FCF Preallocation**: All FCF pools have capacity reserved before training ✅
- [ ] **Memory Determinism**: Zero allocations during training (massif validation)
- [ ] **Memory Profile**: Flat plateau after initialization (±1%)
- [x] **Correctness**: Examples 01 and 07 produce expected results ✅
- [ ] **Cache**: ≥20% reduction in L1/L2 cache misses (optional, Epic 3)

## Validation Approach

Since tests may be broken, validation relies on:

1. **Examples 01 and 07**: Quick smoke tests after each change
2. **Convergence Check**: Lower bound matches baseline
3. **Memory Profiling**: `valgrind --tool=massif` for memory profile
4. **Performance Benchmarks**: `hyperfine` for timing comparison
5. **Code Verification**: `grep` for removed patterns

## Timeline

- **Week 1-2**: Epic 1 - HiGHS Constraint Preallocation ✅ Complete
- **Week 3**: Epic 1b - Full Memory Determinism ✅ Complete
- **Week 3-4**: Epic 1c - Memory Module Cleanup ✅ Complete
- **2025-12-26**: Epic 2 - FCF Full Preallocation ✅ Complete (already implemented)
- **Next**: Epic 3 - Handler-Level SoA Blocks ⬜

**Total Duration**: 5-7 weeks

## References

1. `PREALLOCATION_STATUS_2025_12.md` - Current status report
2. `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - HiGHS API details
3. `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - Hybrid approach recommendation
4. `MEMORY_MODULE_ANALYSIS.md` - Analysis of unused memory module code (historical)
5. `src/memory/` - Memory infrastructure (CutComputationBuffers only)
