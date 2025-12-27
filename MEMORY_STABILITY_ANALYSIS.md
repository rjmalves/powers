# Memory Stability Analysis: Achieving Perfect Memory Determinism in POWE.RS

**Date**: 2025-12-27  
**Example**: 05-large-scale-brazilian  
**Goal**: Identify remaining sources of memory growth and propose solutions for complete memory stability

---

## Executive Summary

After implementing two major plans (preallocation-refactoring and memory-growth-prevention), the application still exhibits RSS growth during training on example 05. Previous analysis concluded this was "legitimate memory usage" - **this conclusion is incorrect** for an SDDP algorithm with cut selection.

**Key Insight**: SDDP with cut selection is fundamentally a **bounded memory algorithm**. After reading input data, we know:
- System size (156 hydros, 121 thermals, 5 buses)
- Graph structure (60 stages, 20 branchings per stage)  
- Training configuration (8 iterations, 16 forward passes)
- State implementation per node (StorageState or StorageAndInflowState)
- Maximum cuts and states per node = `num_iterations × num_forward_passes`

With proper preallocation, memory should be **completely stable** after initialization.

---

## Problem Statement

### Example 05 Configuration

| Parameter | Value |
|-----------|-------|
| Stages | 60 (59 study + 1 pre-study) |
| Branchings per stage | 20 |
| Forward passes | 16 |
| Iterations | 8 |
| Hydros | 156 |
| Thermals | 121 |
| Buses | 5 |
| State type | `storage` (not `storage_and_inflow`) |
| Threads | 16 |

### Observed Memory Behavior

From `MEMORY_GROWTH_ANALYSIS.md`:

| Iteration | Active Cuts | Memory (MB) | Delta (MB/iter) |
|-----------|-------------|-------------|-----------------|
| 1 | 944 | 2017.0 | baseline |
| 8 | 5643 | 5033.9 | +375 avg |

**Total growth**: ~3 GB over 8 iterations (~375 MB/iteration average)

### Expected Behavior (Correct Implementation)

| Phase | Memory | Notes |
|-------|--------|-------|
| Input parsing | ~50 MB | System, graph, recourse data |
| Handler creation | ~2-3 GB | 16 handlers × 60 subproblems × HiGHS models |
| Pool preallocation | +~500 MB | All cuts and states preallocated |
| Training | +0 MB | Update in place, no new allocations |
| Final | ~2.5-3.5 GB | Same as after preallocation |

---

## Root Cause Analysis

### 1. Cut Pool: Dynamic Allocation on Push ❌ NOT PREALLOCATED

**Location**: `src/fcf.rs:118-124`

```rust
pub fn add_cut(&mut self, new_cut: cut::BendersCut) {
    self.cut_pool.pool.push(new_cut);  // NEW BendersCut allocated on heap
}
```

**Current behavior**: 
- `Vec::reserve()` preallocates the Vec's internal pointer array
- But each `BendersCut` is still **heap-allocated when constructed**
- The coefficients `Vec<f64>` inside each cut is also allocated per construction

**What we know at training start**:
- Exact number of cuts: `num_iterations × num_forward_passes` per FCF node
- Exact coefficient dimension: `state.dimension()` per node
- Exact cut structure: same for all cuts at a node

**Solution**: Preallocate all `BendersCut` instances with their coefficient vectors, then **update values in place** using `(iteration, forward_pass_idx)` as the index.

### 2. State Pool: Box<dyn State> Allocation on Push ❌ NOT PREALLOCATED

**Location**: `src/state.rs:247-249`

```rust
pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,  // NEW Box allocation per state
}
```

**Current behavior**:
- Each `add_state()` call allocates a new `Box<dyn State>`
- The state's internal coefficient vector is also allocated

**What we know at training start**:
- Exact number of states: `num_iterations × num_forward_passes` per FCF node
- Exact state implementation: determined by `state_choice` per node
- Exact state dimension: `state.dimension()` per node

**Important**: We must keep ALL visited states because cut selection may return previously deactivated cuts. States are never removed - but that doesn't mean we can't preallocate them.

**Solution**: Preallocate all state instances with their coefficient vectors, then **update values in place** using `(iteration, forward_pass_idx)` as the index.

### 3. Cut Cloning for Handler Application ❌ UNNECESSARY ALLOCATION

**Location**: `src/sddp/mod.rs:2166-2180`

```rust
let cuts: Vec<(usize, crate::cut::BendersCut)> =
    aggregated_result
        .new_cut_ids
        .iter()
        .chain(aggregated_result.returning_cut_ids.iter())
        .filter_map(|&cut_id| {
            fcf_locked.cut_pool.pool.get(cut_id)
                .map(|cut| (cut_id, cut.clone()))  // FULL CLONE per cut
        })
        .collect();
```

**Impact**: ~80 MB transient allocations over 8 iterations.

**Solution**: Use `Arc<BendersCut>` or pass references to avoid cloning.

### 4. HiGHS Internal Memory ⚠️ LIMITED CONTROL

**What we've already done**:
- Disabled presolve (`presolve = "off"`)
- Preallocated cut constraint rows via `preallocate_cut_constraints()`
- Cuts are added by modifying coefficients and bounds, not adding rows

**What remains**: HiGHS still manages internal working memory for:
- Basis factorization tables
- Simplex iteration workspace
- Sparse matrix index structures

**Potential optimizations**:
- Experiment with `simplex_update_limit` to control refactorization frequency
- Consider `simplex_dualize` settings for memory efficiency
- Profile HiGHS allocations separately to identify specific parameters

---

## Proposed Solution: Index-Based Pool Access

### Core Concept

Instead of:
```rust
// Current: Allocate new cut, push to pool
let cut = BendersCut::new(id, coefficients, rhs, iteration, forward_pass_idx);
fcf.add_cut(cut);  // Heap allocation
```

Do:
```rust
// Proposed: Access preallocated cut by (iteration, forward_pass_idx), update in place
let slot = fcf.compute_slot(iteration, forward_pass_idx);
let cut = &mut fcf.cut_pool.pool[slot];
cut.update(coefficients, rhs);  // No allocation
```

### Slot Computation

The slot for any cut/state is deterministic:

```rust
/// Compute slot index for (iteration, forward_pass_idx) pair.
/// 
/// Formula: slot = (iteration - 1) * num_forward_passes + forward_pass_idx
/// 
/// Example with num_forward_passes = 16:
///   (1, 0)  → slot 0
///   (1, 15) → slot 15
///   (2, 0)  → slot 16
///   (8, 15) → slot 127
#[inline]
fn compute_slot(iteration: usize, forward_pass_idx: usize, num_forward_passes: usize) -> usize {
    debug_assert!(iteration >= 1);
    debug_assert!(forward_pass_idx < num_forward_passes);
    (iteration - 1) * num_forward_passes + forward_pass_idx
}
```

### Implementation for BendersCutPool

```rust
pub struct BendersCutPool {
    /// Preallocated cuts. Slot = (iteration-1) * num_fp + fp_idx
    pub pool: Vec<BendersCut>,
    
    /// Maps cut_id → index in solver model constraints
    pub active_cut_indices: HashMap<usize, usize>,
    
    /// Total cuts added (same as pool.len() after preallocation)
    pub total_cut_count: usize,
    
    /// Number of forward passes (for slot computation)
    num_forward_passes: usize,
}

impl BendersCutPool {
    /// Preallocate all cuts for the entire training run.
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        state_dimension: usize,
    ) -> Self {
        let total_cuts = num_iterations * num_forward_passes;
        
        // Preallocate all cuts with their coefficient vectors
        let pool: Vec<BendersCut> = (0..total_cuts)
            .map(|id| BendersCut {
                id,
                coefficients: vec![0.0; state_dimension],  // Preallocated
                rhs: 0.0,
                active: false,  // Initially inactive
                non_dominated_state_count: 0,
                iteration: 0,
                forward_pass_idx: 0,
                slot_index: None,
            })
            .collect();
        
        Self {
            pool,
            active_cut_indices: HashMap::with_capacity(total_cuts),
            total_cut_count: 0,
            num_forward_passes,
        }
    }
    
    /// Update cut at slot computed from (iteration, forward_pass_idx).
    /// No allocation - modifies preallocated cut in place.
    pub fn update_cut(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        coefficients: &[f64],
        rhs: f64,
    ) -> usize {
        let slot = compute_slot(iteration, forward_pass_idx, self.num_forward_passes);
        
        let cut = &mut self.pool[slot];
        cut.coefficients.copy_from_slice(coefficients);
        cut.rhs = rhs;
        cut.iteration = iteration;
        cut.forward_pass_idx = forward_pass_idx;
        cut.active = true;
        cut.non_dominated_state_count = 1;  // Reset for new cut
        
        // Update tracking
        if slot >= self.total_cut_count {
            self.total_cut_count = slot + 1;
        }
        
        slot  // Return slot as cut_id
    }
}
```

### Implementation for VisitedStatePool

```rust
pub struct VisitedStatePool {
    /// Preallocated states. Slot = (iteration-1) * num_fp + fp_idx
    /// Using Vec<StorageState> or Vec<StorageAndInflowState> for concrete types
    pub pool: Vec<Box<dyn State>>,
    
    /// Number of forward passes (for slot computation)
    num_forward_passes: usize,
}

impl VisitedStatePool {
    /// Preallocate all states for the entire training run.
    pub fn preallocate<S: State + Clone + 'static>(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &S,
    ) -> Self {
        let total_states = num_iterations * num_forward_passes;
        
        // Preallocate all states from template (same dimension, zeroed coefficients)
        let pool: Vec<Box<dyn State>> = (0..total_states)
            .map(|_| {
                let mut state = template_state.clone();
                state.reset_to_zero();  // Zero out coefficients but keep capacity
                Box::new(state) as Box<dyn State>
            })
            .collect();
        
        Self {
            pool,
            num_forward_passes,
        }
    }
    
    /// Update state at slot computed from (iteration, forward_pass_idx).
    /// No allocation - modifies preallocated state in place.
    pub fn update_state(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        coefficients: &[f64],
    ) -> &mut Box<dyn State> {
        let slot = compute_slot(iteration, forward_pass_idx, self.num_forward_passes);
        
        let state = &mut self.pool[slot];
        state.update_coefficients(coefficients);
        state.set_iteration(iteration);
        state.set_forward_pass_idx(forward_pass_idx);
        
        state
    }
}
```

### Required Trait Extensions

```rust
pub trait State: Send + Sync {
    // Existing methods...
    
    /// Update coefficient values in place (no allocation).
    fn update_coefficients(&mut self, coefficients: &[f64]);
    
    /// Reset coefficients to zero while preserving capacity.
    fn reset_to_zero(&mut self);
}
```

---

## Integration Points

### 1. FCF Initialization in Training

**Current** (`src/sddp/mod.rs:1795-1803`):
```rust
let max_cuts = num_forward_passes * num_iterations;
let max_states = num_forward_passes * num_iterations;

for fcf_node in self.future_cost_function_graph.iter_nodes() {
    let mut fcf = fcf_node.data.lock().unwrap();
    fcf.cut_pool.pool.reserve(max_cuts);
    fcf.cut_pool.active_cut_indices.reserve(max_cuts);
    fcf.state_pool.pool.reserve(max_states);
}
```

**Proposed**:
```rust
for fcf_node in self.future_cost_function_graph.iter_nodes() {
    let node_data = /* get corresponding NodeData */;
    let state_dim = /* compute from state_choice and system */;
    
    let mut fcf = fcf_node.data.lock().unwrap();
    *fcf = FutureCostFunction::preallocate(
        num_iterations,
        num_forward_passes,
        state_dim,
        &node_data.state_choice,
        &node_data.system,
    );
}
```

### 2. Cut Addition in Backward Pass

**Current** (`src/fcf.rs:278-295`):
```rust
for pair in cut_state_pairs.into_iter() {
    let mut cut = pair.cut;
    let mut state = pair.state;
    
    cut.id = self.cut_pool.total_cut_count;
    new_cut_ids.insert(cut.id);
    self.update_cut_pool_on_add(cut.id);
    // ...
    self.add_cut(cut);  // Allocation here!
    self.add_state(state);  // Allocation here!
}
```

**Proposed**:
```rust
for pair in cut_state_pairs.into_iter() {
    let iteration = pair.cut.iteration;
    let forward_pass_idx = pair.cut.forward_pass_idx;
    
    // Update preallocated cut in place
    let cut_slot = self.cut_pool.update_cut(
        iteration,
        forward_pass_idx,
        &pair.cut.coefficients,
        pair.cut.rhs,
    );
    new_cut_ids.insert(cut_slot);
    self.update_cut_pool_on_add(cut_slot);
    
    // Update preallocated state in place
    self.state_pool.update_state(
        iteration,
        forward_pass_idx,
        pair.state.coefficients(),
    );
    
    // Evaluate domination using slot-based access
    self.eval_new_cut_domination(cut_slot);
}
```

### 3. Cut Cloning Elimination

**Current** (`src/sddp/mod.rs:2166-2180`):
```rust
let cuts: Vec<(usize, crate::cut::BendersCut)> = /* clone cuts */;
```

**Proposed Option A**: Use `Arc<BendersCut>` in pool:
```rust
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    // ...
}

// Clone Arc (cheap) instead of cloning cut data
let cuts: Vec<(usize, Arc<BendersCut>)> = cut_ids
    .iter()
    .filter_map(|&id| pool.get(id).map(|cut| (id, Arc::clone(cut))))
    .collect();
```

**Proposed Option B**: Pass references and restructure:
```rust
// Keep lock and pass references directly to handlers
// Requires restructuring the parallel handler application
```

---

## Memory Savings Estimate

### Current Allocations Per Iteration (Example 05)

| Component | Per Iter | Over 8 Iter | Type |
|-----------|----------|-------------|------|
| BendersCut objects | ~1.2 MB | ~10 MB | Heap alloc |
| Box<dyn State> objects | ~1.4 MB | ~11 MB | Heap alloc |
| Cut coefficient Vecs | ~12 MB | ~96 MB | Nested alloc |
| State coefficient Vecs | ~12 MB | ~96 MB | Nested alloc |
| Cut cloning | ~10 MB | ~80 MB | Transient |
| **Total Rust allocations** | ~36 MB | ~293 MB | |

### After Preallocation

| Component | At Init | During Training | Type |
|-----------|---------|-----------------|------|
| BendersCut objects | ~21 MB | 0 | Preallocated |
| State objects | ~22 MB | 0 | Preallocated |
| Cut cloning | 0 | ~1 MB (Arc) | Cheap |
| **Total** | ~43 MB | ~1 MB | |

**Savings**: ~290 MB of allocations eliminated during training.

### Remaining Growth Sources

| Component | Per Iter | Notes |
|-----------|----------|-------|
| HiGHS internal | ~50 MB? | External, limited control |
| HashMap entries | ~0.5 MB | Grows with active cuts |
| Allocator overhead | Variable | RSS vs actual usage |

---

## Implementation Plan

### Phase 1: Preallocated Cut Pool (1 week)

1. Add `BendersCutPool::preallocate()` method
2. Add `BendersCut::update()` method for in-place modification
3. Modify `add_cuts_batch()` to use slot-based access
4. Update cut domination evaluation for slot-based pool
5. Validate numerical correctness against baseline

### Phase 2: Preallocated State Pool (1 week)

1. Add `State::update_coefficients()` and `State::reset_to_zero()` to trait
2. Implement for `StorageState` and `StorageAndInflowState`
3. Add `VisitedStatePool::preallocate()` method
4. Modify state addition to use slot-based access
5. Validate cut selection still works correctly

### Phase 3: Cut Cloning Elimination (3 days)

1. Change `BendersCutPool::pool` to `Vec<Arc<BendersCut>>`
2. Update all access patterns for Arc
3. Modify handler application to clone Arc instead of data
4. Verify thread safety with parallel execution

### Phase 4: Validation and Profiling (3 days)

1. Run memory profiling on example 05
2. Compare RSS growth before/after
3. Validate algorithm correctness (lower bounds, cut counts)
4. Document any remaining HiGHS-related growth

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Memory growth/iteration | ~375 MB | < 10 MB |
| Peak RSS (example 05) | ~5 GB | < 3 GB |
| Allocations during training | ~36 MB/iter | < 1 MB/iter |
| Algorithm correctness | Baseline | Identical lower bounds |

---

## Verification Commands

```bash
# Memory profile during training
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Detailed allocation tracking
valgrind --tool=massif --pages-as-heap=yes \
  ./target/release/powers run examples/05-large-scale-brazilian

# Compare lower bounds before/after
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | grep "lower"
```

---

## Conclusion

The memory growth in example 05 is preventable. The key insight is that **we know everything about the training process upfront**:

- Number of iterations and forward passes → exact cut/state count
- State implementation per node → exact coefficient dimensions
- Graph structure → exact FCF count

By preallocating all cuts and states at training start and updating them in place using `(iteration, forward_pass_idx)` as the access key, we can achieve **zero allocation** during the training loop for SDDP data structures.

The remaining growth will be limited to:
1. HiGHS internal memory (external, partially addressed)
2. HashMap entry storage (minor, can be bounded)
3. Allocator fragmentation (use mimalloc for better behavior)

With these changes, POWE.RS can achieve near-perfect memory stability, making it suitable for HPC environments with strict memory budgets.

---

## References

1. `MEMORY_GROWTH_ANALYSIS.md` - Original root cause analysis
2. `MEMORY_MODULE_ANALYSIS.md` - Memory module review
3. `ADVANCED_PREALLOCATION_ANALYSIS.md` - Phase 2 preallocation plan
4. `HIGHS_SOLVER_PREALLOCATION_ANALYSIS.md` - HiGHS constraint preallocation
5. `GRAPH_TO_SOA_REFACTORING_ANALYSIS.md` - SoA refactoring analysis
6. `PREALLOCATION_STATUS_2025_12.md` - Current preallocation status
7. `plans/memory-growth-prevention/` - Previous implementation plan
8. `plans/preallocation-refactoring/` - Previous preallocation plan
