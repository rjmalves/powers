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
- **HiGHS Constraint Preallocation**: Epic 1 complete with 23% improvement ✅

**Remaining Gaps**:
1. ~~HiGHS Solver: Dynamic constraint allocation during training~~ ✅ Fixed in Epic 1
2. **Dynamic fallback**: Epic 1 has fallback to dynamic allocation (needs removal)
3. **Slot tracking complexity**: Free list and mapping overhead (needs simplification)
4. FCF: `with_capacity()` exists but not consistently used
5. Handler Data: Hot path data not in contiguous blocks

### Target State

After implementation:
- **HiGHS Constraints**: Pre-allocated at training start, deterministic slot placement
- **No Fallbacks**: Panic on slot exhaustion (not fallback)
- **Deterministic Slots**: `slot = (iteration - 1) * num_forward_passes + forward_pass_idx`
- **FCF Pools**: Always created with `with_capacity()` from `SizingInfo`
- **Handler Blocks**: Hot data (loads, inflows, storage) in contiguous SoA blocks
- **Memory Profile**: Flat after initialization (±1% variation)

### Key Design Decisions

1. **Deterministic Cut Slot Calculation**: Use formula `(iteration-1) * num_fp + fp_idx` instead of dynamic slot allocation. Each (iteration, forward_pass) pair gets exactly one slot.

2. **No Dynamic Fallback**: If slot calculation exceeds preallocated count, panic. This forces correct sizing and ensures memory determinism.

3. **Store Slot in Cut**: Add `slot_index: Option<usize>` to `BendersCut` for O(1) deactivation lookup (no linear search).

4. **Hybrid SoA Approach**: Keep graph for topology (cold path), add contiguous blocks for hot data (loads, inflows, storage). Preserves Markovian graph extensibility.

## Technical Approach

### Core Abstractions

- **`compute_cut_slot(iteration, forward_pass_idx)`**: Deterministic O(1) slot calculation
- **`BendersCut.slot_index`**: Stored slot for O(1) deactivation
- **`SizingInfo` Extensions**: Add `estimate_max_cuts_per_node()` and `estimate_lp_dimensions()`
- **`RealizationBlock`**: Contiguous storage for stage-wise load/inflow/storage data
- **`SubproblemBlock`**: Contiguous storage for subproblem hot data

### Data Flow

```
SizingInfo::from_input()
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
- Thread-local buffers already implemented (TICKET-006b)
- Cut slots computed per-subproblem (no cross-thread contention)

### Performance Strategy

- **HiGHS**: Eliminate dynamic `Highs_addRow` calls (3-8% gain achieved: 23%)
- **Slot Tracking Removal**: Eliminate Vec/HashMap overhead (minor gain expected)
- **FCF**: Eliminate Vec/HashMap reallocation (1-3% gain expected)
- **Handler Blocks**: Improve cache locality for hot data (6-10% gain expected)
- **Total**: 15-25% cumulative improvement (23% already achieved)

## Phases & Milestones

| Phase | Epic | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | HiGHS Constraint Preallocation | 2 weeks | ✅ Complete - 23% improvement |
| 1b | Full Memory Determinism | 1 week | ✅ Complete - Deterministic slots, no fallbacks |
| 1c | Memory Module Cleanup | 3-5 days | ⬜ **Next** - Remove dead code, harden cut buffers |
| 2 | FCF Full Preallocation | 1 week | All FCF instances use `with_capacity()` |
| 3 | Handler-Level SoA Blocks | 2-3 weeks | Hot data in contiguous blocks |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| HiGHS API behavior change | Low | High | Pin HiGHS version, extensive testing |
| Numerical instability from relaxed bounds | Low | Medium | Test with diverse problem sizes |
| ~~Cut slot exhaustion~~ | ~~Low~~ | ~~Low~~ | **Removed**: Panic instead of fallback |
| Cache effects vary by hardware | Medium | Low | Benchmark on target HPC system |
| Presolve removes inactive constraints | Medium | Medium | Disable presolve or use minimal bounds |

## Success Metrics

- [x] **Performance**: ≥15% runtime improvement on 07-par-model example ✅ (23% achieved)
- [ ] **Memory Determinism**: Zero allocations during training (massif validation)
- [ ] **No Fallbacks**: Zero `add_cut_constraint_to_model()` calls during training
- [ ] **Memory Profile**: Flat plateau after initialization (±1%)
- [ ] **Correctness**: All examples produce identical results
- [ ] **Cache**: ≥20% reduction in L1/L2 cache misses (optional)

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
- **Week 3-4**: Epic 1c - Memory Module Cleanup ⬜ **Next**
- **Week 4-5**: Epic 2 - FCF Full Preallocation
- **Week 5-7**: Epic 3 - Handler-Level SoA Blocks

**Total Duration**: 5-7 weeks

## References

1. `PREALLOCATION_STATUS_2025_12.md` - Current status report
2. `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - HiGHS API details
3. `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - Hybrid approach recommendation
4. `MEMORY_MODULE_ANALYSIS.md` - Analysis of unused memory module code
5. `src/memory/` - Memory infrastructure (simplified after Epic 1c)
