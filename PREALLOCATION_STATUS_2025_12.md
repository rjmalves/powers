# POWE.RS Preallocation Refactoring Status Report

**Date**: 2025-12-26  
**Branch**: `feature/sizing-info-per-node`  
**Prepared by**: Performance Analysis Session  
**Previous Reports**: `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md`, `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md`

---

## Executive Summary

### Current State Assessment

The codebase has made **significant progress** on the preallocation infrastructure since the previous analysis (November 2025). Key accomplishments:

| Area | Status | Notes |
|------|--------|-------|
| Memory Module (`src/memory/`) | ✅ **Implemented** | `SizingInfo`, `Buffer`, `BufferPool`, `ThreadLocalBuffers` |
| Cut Computation Buffers | ✅ **Implemented** | TICKET-006b complete with 31% improvement on large systems |
| Deep Memory Estimation | ✅ **Implemented** | `DeepSizeEstimate` trait with proper heap tracking |
| FCF Preallocation | ⚠️ **Partial** | `with_capacity()` method exists but not fully utilized |
| HiGHS Solver Preallocation | ❌ **Not Started** | Cut constraint slots not preallocated |
| Graph-to-SoA Refactoring | ❌ **Not Started** | Hybrid approach recommended but not implemented |

### Remaining Work for HPC Scalability

To achieve deterministic memory allocation for HPC environments with hundreds of cores:

1. **HiGHS Constraint Preallocation** (Priority 1) - ~2 weeks
2. **FCF Full Preallocation** (Priority 2) - ~1 week  
3. **Handler-Level SoA Blocks** (Priority 3) - ~2-3 weeks

**Expected Total Improvement**: 15-25% runtime reduction + 100% memory determinism

---

## Detailed Codebase Analysis

### 1. Memory Infrastructure (✅ Complete)

**Location**: `src/memory/`

The memory module is well-architected with:

```
src/memory/
├── mod.rs          # Module facade, exports, integration tests
├── sizing.rs       # SizingInfo, NodeSizing, MemoryBreakdown
├── buffers.rs      # Buffer, BufferPool, ThreadLocalBuffers, CutComputationBuffers
└── deep_sizing.rs  # DeepSizeEstimate trait
```

**Key Structures**:

```rust
// SizingInfo computes all buffer dimensions from input at startup
pub struct SizingInfo {
    pub node_sizing: Vec<NodeSizing>,       // Per-node heterogeneous dimensions
    pub max_state_dimension: usize,          // Largest state across all nodes
    pub max_scenarios_per_node: usize,       // Maximum branching factor
    pub max_subproblem_vars: usize,          // Maximum LP variables
    pub num_hydros: usize,
    pub num_stages: usize,
    pub max_iterations: usize,
    pub num_forward_passes: usize,
    // ... etc
}
```

**Thread-Local Buffers** (TICKET-006b - Complete):

```rust
// Eliminates 99% of allocations in cut computation hot path
pub struct CutComputationBuffers {
    pub coefficients: Vec<f64>,           // Preallocated cut coefficients
    pub contributions_outer: Vec<Vec<f64>>, // Branching contributions
}

// Usage in state.rs evaluate_cut():
with_cut_buffers(|buffers| {
    buffers.reset_for_cut(dimension, num_branchings);
    // Zero allocations in hot path
});
```

### 2. Subproblem Structure (⚠️ Needs Preallocation Work)

**Location**: `src/subproblem.rs` (~6,400 lines)

**Current State**:
- Lag data structures properly typed: `LoadLagData`, `InflowLagData`
- Two-phase preprocessing model implemented: `prepare_from_trajectory()` + `realize_and_solve()`
- Uncertainty observation data precomputed: `UncertaintyObservationData`

**Preallocation Gaps**:

```rust
// src/subproblem.rs - Cut constraint addition is dynamic
fn add_cut_constraint_to_model(...) {
    let mut factors = Vec::<(usize, f64)>::with_capacity(self.dimension + 1);
    // ...
    model.add_row(cut.rhs.., factors);  // ❌ Dynamic HiGHS allocation
}
```

**Required Changes**:
1. Add `first_preallocated_cut_row: usize` to `Subproblem`
2. Add `num_preallocated_cuts: usize` and `next_available_cut_slot: usize`
3. Implement `preallocate_cut_constraints()` method
4. Modify `add_cut_constraint_to_model()` to use `Highs_changeCoeff`

### 3. State Implementation (✅ Well-Designed)

**Location**: `src/state.rs` (~2,400 lines)

**Current Architecture**:
- Clean extraction pattern: State provides values, Subproblem updates model
- Two implementations: `StorageState`, `StorageAndInflowState`
- `StateLayout` for heterogeneous AR orders with O(1) slice access

```rust
pub struct StateLayout {
    pub per_hydro_dims: Vec<usize>,  // [dim₀, dim₁, ..., dimₙ]
    pub offsets: Vec<usize>,          // Cumulative offsets for O(1) access
    pub total_dim: usize,
}
```

**Preallocation Status**: ✅ State coefficients are preallocated vectors with known dimensions.

### 4. Future Cost Function (⚠️ Partial Preallocation)

**Location**: `src/fcf.rs`

**Current State**:
```rust
impl FutureCostFunction {
    pub fn with_capacity(num_forward_passes, num_iterations, max_state_dim) -> Self {
        // Preallocates cut_pool and state_pool
        // But NOT used consistently in instance construction
    }
}
```

**Gaps**:
1. `FutureCostFunction::with_capacity()` exists but not always called
2. Cut pool HashMap may still grow dynamically
3. State pool preallocation needs verification

### 5. Graph Structure (✅ Already Optimal)

**Location**: `src/graph.rs` (~260 lines)

The graph structure is already efficient:
- Nodes stored contiguously in `Vec<Node<T>>`
- O(1) access via direct indexing
- Sequential access patterns benefit from prefetching

**Conclusion**: Full SoA refactoring is **NOT** recommended. The hybrid approach from the original analysis remains valid.

### 6. SDDP Handler Structure (⚠️ Needs SoA Blocks)

**Location**: `src/sddp/mod.rs`, `src/sddp/instance.rs`

**Current Pattern**:
```rust
pub struct SddpTrainHandler {
    subproblem_graph: DirectedGraph<Subproblem>,
    realization_graph: DirectedGraph<Realization>,
    branching_graph: DirectedGraph<Vec<Realization>>,
    // ...
}
```

**For HPC Scale**: Handler-level SoA blocks would improve cache locality:
```rust
// Proposed (from GRAPH_TO_SOA_REFACTORING_ANALYSIS.md)
pub struct RealizationBlock {
    all_loads: Vec<f64>,      // Contiguous: [stage0_bus0, stage0_bus1, ...]
    all_inflows: Vec<f64>,    // Contiguous: [stage0_h0, stage0_h1, ...]
    stage_offsets: Vec<(usize, usize)>,
}
```

---

## What Has Changed Since Previous Analysis

### Completed Work (November 2025 - December 2025)

1. **TICKET-006b: Thread-Local Cut Buffers** ✅
   - Implemented `CutComputationBuffers` with thread-local storage
   - 31% improvement validated on large systems
   - 99% reduction in allocations per iteration

2. **Memory Module Infrastructure** ✅
   - `SizingInfo` computes all dimensions from input
   - `Buffer` and `BufferPool` abstractions
   - Deep memory estimation with `DeepSizeEstimate` trait

3. **State Extraction Pattern** ✅
   - STATE-REFACTOR-003/004/005 completed
   - Clean separation: State extracts, Subproblem updates model
   - Idempotent `prepare_from_trajectory()` + `realize_and_solve()`

4. **Lag Buffer Management** ✅
   - Type-safe `LoadLagData` and `InflowLagData` structures
   - Separated buffers by entity type (no unified iteration)
   - Precomputed constraint indices

5. **Cleanup** ✅
   - Removed 20+ obsolete documentation files
   - Removed unused `BackwardPassBuffers`
   - Consolidated ticket tracking

### Still Pending

1. **HiGHS Constraint Preallocation** (HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md)
2. **FCF Full Preallocation** (TICKET-006d partial)
3. **Handler-Level SoA Blocks** (GRAPH_TO_SOA_REFACTORING_ANALYSIS.md)

---

## Revised Implementation Plan

### Phase 1: HiGHS Solver Preallocation (Priority 1, ~2 weeks)

**Goal**: Zero HiGHS allocations during training hot path

**Implementation Steps**:

1. **Week 1: Core Infrastructure**
   ```rust
   // Add to Subproblem struct
   first_preallocated_cut_row: usize,
   num_preallocated_cuts: usize,
   next_available_cut_slot: usize,
   cut_slot_to_id: Vec<Option<usize>>,
   free_cut_slots: Vec<usize>,
   ```

2. **Week 1: Preallocation Method**
   ```rust
   impl Subproblem {
       pub fn preallocate_cut_constraints(
           &mut self,
           max_cuts: usize,
           state_dimension: usize,
       ) -> Result<(), String> {
           // Use Highs_addRows with placeholder constraints
           // Bounds: [-∞, ∞] (inactive by default)
           // Coefficients: 0.0 (placeholders)
       }
   }
   ```

3. **Week 2: Cut Addition/Removal via Coefficient Updates**
   ```rust
   fn add_cut_constraint_to_model(...) {
       let slot = self.allocate_cut_slot();
       let row = self.first_preallocated_cut_row + slot;
       
       // Use Highs_changeCoeff instead of Highs_addRow
       model.change_coefficient(row, variables.alpha, 1.0);
       for (hydro_id, &var) in variables.stored_volume.iter().enumerate() {
           model.change_coefficient(row, var, -cut.coefficients[hydro_id]);
       }
       
       // Activate constraint by setting bounds
       model.change_rows_bounds(row, cut.rhs, f64::INFINITY);
   }
   ```

4. **Week 2: Integration with SddpInstance**
   - Compute max_cuts at training start
   - Call `preallocate_cut_constraints()` for all subproblems
   - Validate with extensive testing

**Expected Impact**:
- Memory growth: +14 MB → 0 MB during training
- Runtime improvement: 3-8%
- Cache miss reduction: ~30%

### Phase 2: FCF Full Preallocation (Priority 2, ~1 week)

**Goal**: Ensure `FutureCostFunction::with_capacity()` is used everywhere

**Changes Required**:

1. **SddpBuilder modifications**:
   ```rust
   // In builder.rs - ensure preallocation
   let fcf = FutureCostFunction::with_capacity(
       sizing.num_forward_passes,
       sizing.max_iterations,
       sizing.max_state_dimension,
   );
   ```

2. **HashMap preallocation**:
   ```rust
   // In cut.rs - BendersCutPool
   impl BendersCutPool {
       pub fn with_capacity(num_cuts: usize) -> Self {
           Self {
               pool: Vec::with_capacity(num_cuts),
               id_to_index: HashMap::with_capacity(num_cuts),
               // ...
           }
       }
   }
   ```

**Expected Impact**: 1-3% runtime improvement

### Phase 3: Handler-Level SoA Blocks (Priority 3, ~2-3 weeks)

**Goal**: Improve cache locality for hot data without full graph refactoring

**Approach**: Hybrid as recommended in GRAPH_TO_SOA_REFACTORING_ANALYSIS.md

1. Keep graph for topology (cold path)
2. Create contiguous blocks for hot data:
   ```rust
   pub struct RealizationBlock {
       all_loads: Vec<f64>,      // [stage0_bus0, ..., stageN_busM]
       all_inflows: Vec<f64>,    // [stage0_h0, ..., stageN_hM]
       all_storage: Vec<f64>,    // [stage0_h0, ..., stageN_hM]
       stage_offsets: Vec<StageOffsets>,
   }
   
   pub struct SddpTrainHandler {
       topology: DirectedGraph<NodeMetadata>,  // Small, cold
       realizations: RealizationBlock,          // Hot data, contiguous
       subproblems: SubproblemBlock,            // Hot data, contiguous
   }
   ```

**Expected Impact**: 6-10% runtime improvement

---

## Sizing Information Computation

For complete preallocation, the following dimensions must be computed at startup:

### From System Definition
- `num_hydros`, `num_thermals`, `num_buses`, `num_lines`
- Per-hydro AR orders (from temporal models)

### From Configuration
- `num_stages`, `num_forward_passes`, `num_iterations`
- `num_branchings_per_stage` (from scenario config)

### Derived Dimensions
```rust
// Maximum cuts per node (conservative estimate)
max_cuts = num_iterations * num_forward_passes * 1.2  // 20% buffer

// HiGHS model dimensions
num_lp_rows = num_hydros + num_buses + 2*num_lines + max_cuts
num_lp_cols = 3*num_hydros + num_thermals + 2*num_buses + 2*num_lines + 1

// State dimension (heterogeneous)
state_dim[node] = num_hydros + sum(inflow_ar_orders)  // for storage_and_inflow
```

### Current Sizing Implementation

`SizingInfo::from_input()` already computes most dimensions:
```rust
// From src/memory/sizing.rs
pub fn from_input(system, graph, config) -> Self {
    // ✅ num_hydros, num_thermals, num_buses, num_lines
    // ✅ Per-node state dimensions (including AR orders)
    // ✅ max_iterations, num_forward_passes
    // ⚠️ Missing: max_cuts estimate
    // ⚠️ Missing: LP row/column counts with cut slots
}
```

**Required Addition**:
```rust
impl SizingInfo {
    pub fn estimate_max_cuts_per_node(&self) -> usize {
        // Conservative: all cuts survive without selection
        self.max_iterations * self.num_forward_passes
    }
    
    pub fn estimate_lp_dimensions(&self) -> (usize, usize) {
        let base_rows = self.num_hydros + self.num_buses + 2 * self.num_lines;
        let max_cuts = self.estimate_max_cuts_per_node();
        let total_rows = base_rows + max_cuts;
        
        let total_cols = 3 * self.num_hydros 
                       + self.num_thermals 
                       + 2 * self.num_buses 
                       + 2 * self.num_lines 
                       + 1;  // alpha
        
        (total_rows, total_cols)
    }
}
```

---

## Validation Checklist

Before starting implementation, verify:

- [ ] All tests pass: `cargo test`
- [ ] Benchmarks available: `cargo bench`
- [ ] Profiling scripts work: `scripts/profile_baseline.sh`

During implementation:

- [ ] `cargo fmt` after each change
- [ ] `cargo clippy -- -D warnings` clean
- [ ] No memory growth during training (validate with massif)
- [ ] Solution quality unchanged (objective within 0.001%)

After implementation:

- [ ] All tests still pass
- [ ] Memory profile is flat during training
- [ ] Runtime improvement ≥ 10%
- [ ] Documented in CHANGELOG.md

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HiGHS API behavior change | Low | High | Pin HiGHS version, test extensively |
| Numerical instability from relaxed bounds | Low | Medium | Test with diverse problem sizes |
| Cut slot exhaustion | Low | Low | Graceful fallback to dynamic allocation |
| Cache effects vary by hardware | Medium | Low | Benchmark on target HPC system |

---

## Conclusion

The POWE.RS codebase is **well-prepared** for the final preallocation work:

1. **Memory infrastructure is complete** - `SizingInfo`, buffers, thread-locals
2. **State/Subproblem separation is clean** - Extraction pattern established
3. **Lag data is properly typed** - No more unified entity iteration

**Recommended Next Steps**:

1. Begin **HiGHS Constraint Preallocation** (highest impact, well-documented API)
2. Validate on representative problems before moving to Phase 2
3. Measure cache performance to confirm SoA block benefit

**Timeline**: 4-6 weeks for full preallocation achieving:
- 100% memory determinism
- 15-25% runtime improvement
- HPC-ready scalability

---

## References

1. `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - Hybrid approach recommendation
2. `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - HiGHS API details
3. `src/memory/mod.rs` - Memory infrastructure documentation
4. HiGHS C API: `Highs_addRows`, `Highs_changeCoeff`, `Highs_changeRowBounds`
