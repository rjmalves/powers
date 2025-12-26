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

### Current State

The codebase has made significant progress on preallocation:
- **Memory Module** (`src/memory/`): `SizingInfo`, `Buffer`, `BufferPool`, `ThreadLocalBuffers` ✅
- **Cut Computation Buffers**: TICKET-006b complete with 31% improvement ✅
- **Deep Memory Estimation**: `DeepSizeEstimate` trait ✅
- **State/Subproblem Separation**: Clean extraction pattern ✅

**Remaining Gaps**:
1. HiGHS Solver: Dynamic constraint allocation during training
2. FCF: `with_capacity()` exists but not consistently used
3. Handler Data: Hot path data not in contiguous blocks

### Target State

After implementation:
- **HiGHS Constraints**: Pre-allocated at training start, modified via coefficient updates
- **FCF Pools**: Always created with `with_capacity()` from `SizingInfo`
- **Handler Blocks**: Hot data (loads, inflows, storage) in contiguous SoA blocks
- **Memory Profile**: Flat after initialization (±1% variation)

### Key Design Decisions

1. **HiGHS Preallocation via Relaxed Bounds**: Use `[-∞, ∞]` bounds to deactivate placeholder constraints, then activate via `Highs_changeCoeff` and `Highs_changeRowBounds`. This is the industry-standard pattern (Gurobi, CPLEX).

2. **Slot Reuse for Cut Selection**: Maintain free list of available slots when cuts are removed, enabling O(1) slot allocation without growing constraint matrix.

3. **Hybrid SoA Approach**: Keep graph for topology (cold path), add contiguous blocks for hot data (loads, inflows, storage). Preserves Markovian graph extensibility.

## Technical Approach

### Core Abstractions

- **`CutSlotManager`**: Tracks preallocated HiGHS constraint slots, free list, and mappings
- **`SizingInfo` Extensions**: Add `estimate_max_cuts_per_node()` and `estimate_lp_dimensions()`
- **`RealizationBlock`**: Contiguous storage for stage-wise load/inflow/storage data
- **`SubproblemBlock`**: Contiguous storage for subproblem hot data

### Data Flow

```
SizingInfo::from_input()
    ↓
Calculate max_cuts, LP dimensions
    ↓
Subproblem::preallocate_cut_constraints()  [HiGHS Highs_addRows]
    ↓
Training loop (zero allocations):
    - add_cut: Highs_changeCoeff + Highs_changeRowBounds
    - remove_cut: Highs_changeRowBounds (relax to [-∞, ∞])
    - slot reuse: free_cut_slots stack
```

### Parallelism Strategy

- Each thread has deterministic memory footprint
- No allocation contention between threads
- Thread-local buffers already implemented (TICKET-006b)
- Cut slots managed per-subproblem (no cross-thread contention)

### Performance Strategy

- **HiGHS**: Eliminate dynamic `Highs_addRow` calls (3-8% gain expected)
- **FCF**: Eliminate Vec/HashMap reallocation (1-3% gain expected)
- **Handler Blocks**: Improve cache locality for hot data (6-10% gain expected)
- **Total**: 15-25% cumulative improvement

## Phases & Milestones

| Phase | Epic | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | HiGHS Constraint Preallocation | 2 weeks | Zero HiGHS allocations during training |
| 2 | FCF Full Preallocation | 1 week | All FCF instances use `with_capacity()` |
| 3 | Handler-Level SoA Blocks | 2-3 weeks | Hot data in contiguous blocks |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HiGHS API behavior change | Low | High | Pin HiGHS version, extensive testing |
| Numerical instability from relaxed bounds | Low | Medium | Test with diverse problem sizes |
| Cut slot exhaustion | Low | Low | Graceful fallback to dynamic allocation |
| Cache effects vary by hardware | Medium | Low | Benchmark on target HPC system |
| Presolve removes inactive constraints | Medium | Medium | Disable presolve or use minimal bounds |

## Success Metrics

- [ ] **Memory Determinism**: Zero allocations during training (massif validation)
- [ ] **Memory Profile**: Flat plateau after initialization (±1%)
- [ ] **Performance**: ≥15% runtime improvement on 07-par-model example
- [ ] **Correctness**: All examples produce identical results
- [ ] **Cache**: ≥20% reduction in L1/L2 cache misses (optional)

## Validation Approach

Since tests may be broken, validation relies on:

1. **Examples 01 and 07**: Quick smoke tests after each change
2. **Convergence Check**: Lower bound matches baseline
3. **Memory Profiling**: `valgrind --tool=massif` for memory profile
4. **Performance Benchmarks**: `hyperfine` for timing comparison

## Timeline

- **Week 1-2**: Epic 1 - HiGHS Constraint Preallocation
- **Week 3**: Epic 2 - FCF Full Preallocation
- **Week 4-6**: Epic 3 - Handler-Level SoA Blocks

**Total Duration**: 4-6 weeks

## References

1. `PREALLOCATION_STATUS_2025_12.md` - Current status report
2. `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - HiGHS API details
3. `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - Hybrid approach recommendation
4. `src/memory/` - Memory infrastructure documentation
