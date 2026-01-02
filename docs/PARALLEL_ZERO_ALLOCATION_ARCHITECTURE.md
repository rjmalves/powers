# Parallel Zero-Allocation Architecture for SDDP Training

**Date**: 2025-12-30  
**Status**: Proposed  
**Author**: Architecture Analysis  

---

## Executive Summary

This document proposes a comprehensive architectural refactoring to achieve **parallel zero-allocation cut computation** in the SDDP training loop. The analysis covers:

1. **Handler-level staging buffers** for parallel cut computation with sequential pool updates
2. **Pool memory model optimization** to eliminate HashMap overhead and unnecessary indirection
3. **Reproducibility-preserving design** for deterministic results across runs

The refactoring is recommended for production use cases with 500+ forward passes on 192+ core systems.

---

## Part 1: Current State Analysis

### 1.1 Allocation Hotspot

The current training loop allocates ~18 MB per run via `CutData::from_refs()`:

```rust
// fcf.rs - Current allocation point
pub fn from_refs(...) -> Self {
    Self {
        cut_coefficients: cut_coefficients.to_vec(),  // ALLOCATES
        state_coefficients: state_coefficients.to_vec(),  // ALLOCATES
        ...
    }
}
```

This creates 2 × `Vec<f64>` per cut × `num_forward_passes` × `num_stages` × `num_iterations`.

### 1.2 Current Parallelism Model

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Par Compute (ALLOCATES)                               │
│                                                                 │
│  par_iter_mut on handlers:                                      │
│    Each handler → CutData { Vec, Vec } → collected              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: Sequential Copy to Pools                              │
│                                                                 │
│  for cut_data in cut_data_vec:                                  │
│      pool.update_from_cut_data(cut_data)  // copies then drops  │
└─────────────────────────────────────────────────────────────────┘
```

**Problem**: The allocations in Phase 1 are immediately copied and discarded in Phase 2.

### 1.3 Existing Infrastructure (Underutilized)

| Component | Location | Status |
|-----------|----------|--------|
| `CutComputationBuffers` | `src/memory/buffers.rs` | ✅ Thread-local, working |
| `evaluate_cut_ref()` | `src/state.rs` | ✅ Returns references, no allocation |
| `CutEvalResult<'a>` | `src/cut.rs` | ✅ Holds references only |
| `BendersCutPool::preallocate()` | `src/cut.rs` | ✅ Preallocates slots |
| `VisitedStatePool::preallocate()` | `src/state.rs` | ✅ Preallocates slots |
| `compute_cut_into_slot()` | `src/state.rs` | ✅ Zero-alloc, but sequential |

The infrastructure exists but is **not wired for parallel execution**.

---

## Part 2: Proposed Architecture - Handler Staging Buffers

### 2.1 Core Insight

Parallel computation and global pool update are **separable phases**:

- **Phase 1a**: Compute cut coefficients in parallel (each handler uses thread-local buffers)
- **Phase 1b**: Copy results to global pools sequentially (deterministic order)

The key is that each handler needs only **one buffer slot** for its current computation, since we process stages sequentially and the buffer content is copied before moving to the next stage.

### 2.2 Handler Staging Buffer Design

```rust
/// Staging area for one cut + one state computation.
/// Lives in SddpTrainHandler, reused across all stages within an iteration.
///
/// Memory: ~2 × state_dim × 8 bytes ≈ 1.6 KB for 100-dimension state
pub struct CutStagingBuffer {
    /// Computed cut coefficients (copied from thread-local CutComputationBuffers)
    pub cut_coefficients: Vec<f64>,
    /// Computed cut RHS
    pub cut_rhs: f64,
    /// State coefficients (copied from State::coefficients())
    pub state_coefficients: Vec<f64>,
    /// Iteration tracking (1-based)
    pub iteration: usize,
    /// Forward pass index (0-based)
    pub forward_pass_idx: usize,
    /// Timing from this computation
    pub timing: BackwardPhase1Timing,
}

impl CutStagingBuffer {
    /// Create staging buffer with preallocated capacity.
    pub fn new(state_dim: usize) -> Self {
        Self {
            cut_coefficients: vec![0.0; state_dim],
            state_coefficients: vec![0.0; state_dim],
            cut_rhs: 0.0,
            iteration: 0,
            forward_pass_idx: 0,
            timing: BackwardPhase1Timing::default(),
        }
    }

    /// Copy computed results into staging area.
    /// Called at end of parallel cut computation, while still holding
    /// the thread-local buffer.
    #[inline]
    pub fn stage_from(
        &mut self,
        eval_result: &CutEvalResult,
        state_coefficients: &[f64],
        timing: BackwardPhase1Timing,
    ) {
        let cut_len = eval_result.coefficients.len();
        let state_len = state_coefficients.len();
        
        self.cut_coefficients[..cut_len].copy_from_slice(eval_result.coefficients);
        self.state_coefficients[..state_len].copy_from_slice(state_coefficients);
        self.cut_rhs = eval_result.rhs;
        self.iteration = eval_result.iteration;
        self.forward_pass_idx = eval_result.forward_pass_idx;
        self.timing = timing;
    }
}
```

### 2.3 Updated Handler Structure

```rust
pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,
    
    // Existing optional history fields...
    forward_detail_history: Option<Vec<ForwardPassDetail>>,
    backward_detail_history: Option<Vec<BackwardPassDetail>>,
    preserve_forward_detail: bool,
    preserve_backward_detail: bool,
    
    // NEW: Staging buffer for zero-allocation parallel cut computation
    cut_staging: CutStagingBuffer,
}
```

### 2.4 New Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1a: Parallel Computation (par_iter_mut on handlers)          │
│                                                                     │
│  Handler[0]                Handler[1]           ...  Handler[N-1]   │
│  ┌────────────────┐        ┌────────────────┐       ┌────────────────┐
│  │ 1. Solve LP    │        │ 1. Solve LP    │       │ 1. Solve LP    │
│  │ 2. Extract     │        │ 2. Extract     │       │ 2. Extract     │
│  │    duals       │        │    duals       │       │    duals       │
│  │ 3. thread_     │        │ 3. thread_     │       │ 3. thread_     │
│  │    local buf   │        │    local buf   │       │    local buf   │
│  │ 4. Copy to     │        │ 4. Copy to     │       │ 4. Copy to     │
│  │    staging     │        │    staging     │       │    staging     │
│  └────────────────┘        └────────────────┘       └────────────────┘
│         │                        │                        │          │
│         └────────────────────────┴────────────────────────┘          │
│                                  │                                   │
│                    rayon sync barrier (collect timings)              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1b: Sequential Pool Update (deterministic order)             │
│                                                                     │
│  for (fp_idx, handler) in handlers.iter().enumerate() {             │
│      // Handlers already ordered by forward_pass_idx (0, 1, 2, ...) │
│      let slot = compute_slot(iteration, fp_idx, num_forward_passes);│
│      global_cut_pool.update_from_staging(&handler.cut_staging);     │
│      global_state_pool.update_from_staging(&handler.cut_staging);   │
│  }                                                                  │
│  // Time: ~N × 2 × copy_from_slice ≈ 0.1 ms for 500 handlers        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 2: Cut Selection (sequential, on sorted slot indices)        │
│  Phase 3: Apply Cuts to Handlers (parallel)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.5 Memory Analysis for 500 Forward Passes

| Component | Size Per Unit | Count | Total |
|-----------|---------------|-------|-------|
| CutStagingBuffer | ~1.6 KB (100-dim state) | 500 | **~800 KB** |
| Thread-local CutComputationBuffers | ~10 KB | 192 threads | **~1.9 MB** |
| Global cut pool | Already allocated | 1 | 0 extra |
| Global state pool | Already allocated | 1 | 0 extra |

**Compare to current**: 18+ MB per training run in transient allocations.

### 2.6 Reproducibility Guarantee

The design ensures bit-for-bit reproducibility:

1. **Parallel phase order-independent**: Each handler writes only to its own staging buffer
2. **Sequential phase deterministic**: Handlers iterated in fixed `forward_pass_idx` order (0, 1, 2, ...)
3. **Slot computation deterministic**: `slot = (iteration - 1) * num_forward_passes + forward_pass_idx`
4. **Cut selection already deterministic**: Uses epsilon-based comparison with ID tie-breaking

---

## Part 3: Pool Memory Model Optimization

### 3.1 Current Pool Structure Issues

#### Issue 1: HashMap for Active Cut Indices

```rust
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    pub active_cut_indices: HashMap<usize, usize>,  // cut_id → model_constraint_index
    pub total_cut_count: usize,
    num_forward_passes: usize,
}
```

**Problems**:

1. **HashMap lookup overhead**: O(1) amortized, but hashing has constant factor ~50-100ns
2. **HashMap iteration overhead**: Non-contiguous memory access in `update_cut_pool_on_remove()`
3. **Redundant with slot_index**: Cuts already store their `slot_index` atomically
4. **Index shifting on removal**: When a cut is removed, all subsequent indices must be decremented

#### Issue 2: Arc Wrapper Around BendersCut

```rust
pub pool: Vec<Arc<BendersCut>>
```

**Analysis**:

The Arc is used because:
1. Cuts are shared during domination evaluation (read-only access)
2. `Arc::get_mut()` is used during preallocated updates

**Problem**: `Arc::get_mut()` requires exclusive ownership, adding a runtime check. With preallocation, we control all references—the Arc adds unnecessary overhead.

#### Issue 3: Box<dyn State> in VisitedStatePool

```rust
pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}
```

**Analysis**:

Dynamic dispatch via `Box<dyn State>` adds:
1. Vtable pointer lookup on every method call (~1-3ns)
2. Heap indirection (cache miss potential)
3. Clone requires trait object machinery

**Observation**: In a given problem, all states are the same concrete type. Dynamic dispatch is unnecessary if we use generics or enum dispatch.

### 3.2 Proposed Pool Redesign

#### 3.2.1 Replace HashMap with Direct Slot Mapping

Since slots are computed deterministically from `(iteration, forward_pass_idx)`, we can:

1. **Use slot_index stored in BendersCut** for model constraint lookup
2. **Track active status via atomic bool** (already exists)
3. **Eliminate HashMap entirely**

```rust
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,  // Direct storage, no Arc
    pub total_cut_count: usize,
    num_forward_passes: usize,
    /// Bitset for O(1) active status check (optional optimization)
    active_bitset: Option<Vec<u64>>,
}
```

**Benefits**:
- No hashing overhead
- Cache-friendly linear memory
- No HashMap rehashing concerns

#### 3.2.2 Replace Arc<BendersCut> with Direct Storage

With preallocated pools, we control all references:

```rust
// Current: Arc wrapper
pub pool: Vec<Arc<BendersCut>>

// Proposed: Direct storage
pub pool: Vec<BendersCut>
```

**Migration**:
1. Remove `Arc::get_mut()` calls—direct mutable access
2. Use references (`&BendersCut`) for read-only sharing
3. Atomic fields (`AtomicBool`, `AtomicUsize`) remain for concurrent reads

**Risk**: Must ensure no code holds `Arc<BendersCut>` across mutable pool operations. Audit required.

#### 3.2.3 Replace Box<dyn State> with Enum Dispatch

```rust
/// State without dynamic dispatch.
pub enum ConcreteState {
    Storage(StorageStateCore),
    StorageAndInflow(StorageAndInflowStateCore),
}

pub struct VisitedStatePool {
    pub pool: Vec<ConcreteState>,
}
```

**Benefits**:
- No vtable lookup
- Contiguous memory (better cache behavior)
- Known sizes at compile time

**Tradeoff**: Requires match statements, but these compile to efficient jump tables.

### 3.3 Comparison: Current vs Proposed

| Aspect | Current | Proposed | Improvement |
|--------|---------|----------|-------------|
| Cut lookup | HashMap O(1) + hash | Direct index O(1) | ~50-100ns saved |
| Cut iteration | Vec + Arc indirection | Direct Vec | Cache-friendly |
| State access | Box + vtable | Direct/enum | No vtable |
| Memory layout | Fragmented | Contiguous | Better prefetch |
| Active check | HashMap contains | Atomic bool | No hash |

### 3.4 HashMap Removal: Constraint Index Tracking

The HashMap maps `cut_id → model_constraint_index`. With preallocation:

- **Model constraint rows are preallocated** in HiGHS during initialization
- **Row index = first_preallocated_cut_row + slot_index**
- **slot_index is stored in each cut** via `cut.slot_index`

The mapping is thus:
```rust
fn get_model_row_for_cut(&self, cut: &BendersCut) -> usize {
    self.first_preallocated_cut_row + cut.get_slot_index().unwrap()
}
```

No HashMap needed.

### 3.5 Handling Cut Removal (Domination)

When a cut is dominated and removed from the model:

**Current approach**:
1. Remove from HashMap
2. Iterate HashMap, decrement indices > removed_index
3. O(n) per removal

**Problem**: With 500 forward passes × 100 iterations = 50,000 cuts, this is expensive.

**Proposed approach**:
Since we use preallocation with bound relaxation (not actual row deletion):
1. Set `cut.active = false`
2. Call `model.change_rows_bounds(row, -∞, +∞)` to deactivate
3. **No index shifting needed**—rows are never deleted

This is already implemented in `deactivate_cut_constraint()`. The HashMap is only needed for the non-preallocated path (which we're deprecating).

---

## Part 4: Implementation Roadmap

### Phase 1: Handler Staging Buffers (Priority: High)

| Ticket | Description | Effort | Risk |
|--------|-------------|--------|------|
| T-101 | Create `CutStagingBuffer` struct | 0.5 days | Low |
| T-102 | Add staging buffer to `SddpTrainHandler` | 0.5 days | Low |
| T-103 | Implement `compute_cut_into_staging()` method | 1 day | Medium |
| T-104 | Add `update_from_staging()` to pools | 0.5 days | Low |
| T-105 | Update `ParallelHandlerCoordinator` for parallel-then-sequential | 1 day | Medium |
| T-106 | Wire into `backward_pass.rs` | 1 day | Medium |
| T-107 | Golden tests and benchmarks | 1 day | Low |

**Total**: ~5.5 days

### Phase 2: Pool Memory Model (Priority: Medium)

| Ticket | Description | Effort | Risk |
|--------|-------------|--------|------|
| T-201 | Remove Arc wrapper from BendersCutPool | 1 day | Medium |
| T-202 | Audit Arc<BendersCut> usage across codebase | 0.5 days | Low |
| T-203 | Remove HashMap from BendersCutPool | 1 day | Medium |
| T-204 | Update FCF to use direct pool access | 0.5 days | Low |
| T-205 | Create ConcreteState enum | 1 day | Medium |
| T-206 | Migrate VisitedStatePool to enum dispatch | 1.5 days | Medium |
| T-207 | Performance validation | 1 day | Low |

**Total**: ~6.5 days

### Phase 3: Cleanup and Documentation

| Ticket | Description | Effort | Risk |
|--------|-------------|--------|------|
| T-301 | Remove deprecated CutData path | 0.5 days | Low |
| T-302 | Remove deprecated compute_cut_data() | 0.5 days | Low |
| T-303 | Update architecture documentation | 0.5 days | Low |
| T-304 | Add profiling instrumentation | 0.5 days | Low |

**Total**: ~2 days

### Total Effort: ~14 days (3 weeks)

---

## Part 5: Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical divergence | Low | **Critical** | Golden tests after every change |
| Arc removal breaks shared access | Medium | Medium | Careful audit of all BendersCut usages |
| Enum dispatch adds match overhead | Low | Low | Benchmark shows jump tables are fast |
| Phase 1b sequential bottleneck | Low | Low | Copy is ~0.1ms for 500 handlers |
| Borrow checker conflicts | Medium | Medium | May need RefCell or unsafe in edge cases |

---

## Part 6: Expected Performance Impact

### Memory

| Metric | Current | After Phase 1 | After Phase 2 |
|--------|---------|---------------|---------------|
| Transient allocations/run | ~18 MB | **~0** | ~0 |
| Peak RSS overhead | Allocation churn | Flat | Flat |
| Handler staging | 0 | ~800 KB | ~800 KB |

### CPU

| Operation | Current | After Phase 1 | After Phase 2 |
|-----------|---------|---------------|---------------|
| Phase 1 (cut compute) | Parallel + alloc | Parallel, zero-alloc | Parallel, zero-alloc |
| Phase 1b (pool update) | N/A | Sequential copy | Sequential copy (faster) |
| Cut lookup | HashMap | HashMap | Direct index |
| State access | Vtable dispatch | Vtable dispatch | Enum dispatch |

### Expected Speedup

Conservative estimate: **5-15%** faster training loop
- Allocation elimination: 3-5%
- Better cache locality: 2-5%
- HashMap removal: 1-3%
- Vtable elimination: 1-2%

---

## Part 7: Conclusion and Recommendation

### Recommendation: **Proceed with Full Refactoring**

The proposed architecture is worth implementing because:

1. **Your use case demands it**: 500 forward passes on 192 threads requires parallel execution
2. **Infrastructure is 80% ready**: Thread-local buffers, preallocated pools, slot computation all exist
3. **Reproducibility is preserved**: Deterministic ordering in Phase 1b
4. **Memory overhead is minimal**: ~800 KB for staging vs ~18 MB eliminated
5. **Pool optimizations are safe**: Direct storage eliminates unnecessary indirection
6. **Risk is manageable**: Golden tests catch any numerical divergence

### Suggested Execution Order

1. **Phase 1 first**: Handler staging buffers unlock parallel zero-allocation
2. **Validate with benchmarks**: Confirm memory elimination and no regression
3. **Phase 2 next**: Pool optimizations for additional CPU gains
4. **Phase 3 last**: Cleanup deprecated code paths

---

## Appendix A: Key Code Locations

| Component | File | Lines |
|-----------|------|-------|
| BendersCutPool | `src/cut.rs` | 276-510 |
| VisitedStatePool | `src/state.rs` | 480-592 |
| CutComputationBuffers | `src/memory/buffers.rs` | 78-180 |
| compute_cut_into_slot() | `src/state.rs` | 1118-1151 |
| SddpTrainHandler | `src/sddp/mod.rs` | 323-450 |
| ParallelHandlerCoordinator | `src/algorithm/coordinator.rs` | 46-220 |
| backward_pass execution | `src/algorithm/backward_pass.rs` | 250-315 |
| FutureCostFunction | `src/fcf.rs` | 56-175 |

## Appendix B: Existing Tests to Leverage

- `src/cut.rs` tests: Cut pool preallocation, slot computation
- `src/state.rs` tests: State pool preallocation, compute_cut_into_slot equivalence
- `src/fcf.rs` tests: Domination evaluation, batch cut selection
- `tests/test_cut_generation_correctness.rs`: Mathematical properties
- Golden tests: Bit-for-bit reproducibility

## Appendix C: Benchmark Targets

| Benchmark | Baseline | Target | Critical |
|-----------|----------|--------|----------|
| `sddp_e2e/05-large-scale` | Current | No regression | Yes |
| `training_loop/backward_pass` | Current | +10% | No |
| Memory peak (heaptrack) | Current | -15 MB | Yes |
| Allocations/iter (DHAT) | ~7500 | ~100 | Yes |
