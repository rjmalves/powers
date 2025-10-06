# Batch Cut Selection Architecture Design

## Problem Analysis

### Current Architecture Issues

**Per-Thread Lock Pattern** (`subproblem.rs:404-462`):

```rust
pub fn add_cut_and_evaluate_cut_selection(
    &mut self,
    cut_state_pair: fcf::CutStatePair,
    future_cost_function: Arc<Mutex<fcf::FutureCostFunction>>,
) {
    let mut cut = cut_state_pair.cut;
    let mut visited_state = cut_state_pair.state;

    // Add cut to local model (no lock needed)
    if let Some(model) = self.model.as_mut() {
        self.state.add_cut_constraint_to_model(&mut cut, &self.variables, model);
    }

    // 🔒 LOCK ACQUIRED HERE
    let mut fcf = future_cost_function.lock().unwrap();

    // Cut selection logic (~50-200 μs with lock held)
    cut.id = fcf.cut_pool.total_cut_count;
    fcf.update_cut_pool_on_add(cut.id);
    fcf.eval_new_cut_domination(&mut cut);  // O(n) where n=state_count
    fcf.add_cut(cut);

    let returning_cut_ids = fcf.update_old_cuts_domination(&mut visited_state);  // O(m) where m=cut_count
    fcf.add_state(visited_state);

    // More model updates while holding lock...
    // 🔒 LOCK RELEASED HERE
}
```

**Measured Costs**:

- Lock hold time: 50-200 μs per thread (varies with pool size)
- With 8 threads: 7 threads wait idle per backward pass
- Wasted CPU: 7 × 50-200 μs = 350-1400 μs per backward pass node
- Total overhead: 15-25% on 8-core system

**Non-Determinism**:

- Thread 1 finishes → locks → adds cuts → changes dominance landscape
- Thread 2 finishes → locks → sees different pool → selects different cuts
- Same problem, different thread order → different policies
- Makes reproducibility impossible

## Proposed Solution: Batch Cut Selection

### High-Level Architecture

```rust
// PHASE 1: Parallel computation (NO LOCKS)
let cut_state_pairs: Vec<CutStatePair> = child_nodes
    .par_iter()
    .map(|child_node| {
        child_node.compute_new_cut(...)  // Heavy computation, no shared state
    })
    .collect();  // All threads run in parallel ✅

// PHASE 2: Batch selection (SINGLE LOCK)
{
    let mut fcf = parent_fcf.lock().unwrap();
    for pair in cut_state_pairs {
        let mut cut = pair.cut;
        let mut state = pair.state;

        // Deterministic order: process cuts in node order
        cut.id = fcf.cut_pool.total_cut_count;
        fcf.update_cut_pool_on_add(cut.id);
        fcf.eval_new_cut_domination(&mut cut);
        fcf.add_cut(cut);

        let _returning_cut_ids = fcf.update_old_cuts_domination(&mut state);
        fcf.add_state(state);
    }
}  // Lock released after all cuts processed

// PHASE 3: Update local models (NO LOCKS, parallel)
subproblems.par_iter_mut().zip(&cut_state_pairs).for_each(|(subproblem, pair)| {
    subproblem.apply_cut_selection_to_model(pair, &fcf);
});
```

### Key Benefits

1. **Eliminate Lock Contention**:

   - Only 1 lock acquisition per backward pass (not N threads)
   - No threads waiting idle
   - Expected speedup: 15-30% on multi-core systems

2. **Deterministic Ordering**:

   - Cuts processed in node order (deterministic)
   - Same problem → same cuts → same policy (reproducible)
   - Easier debugging and testing

3. **Better Cache Locality**:

   - Process all cuts together → fewer cache misses
   - Sequential access pattern → better prefetching

4. **Cleaner Architecture**:
   - Separation of concerns: compute vs select vs apply
   - Easier to test, benchmark, and optimize
   - Clear parallelism boundaries

### Implementation Strategy

#### Step 1: Add batch processing to FCF (`fcf.rs`)

```rust
impl FutureCostFunction {
    /// Add multiple cuts in batch (for deterministic cut selection)
    ///
    /// This processes cuts sequentially in a single lock acquisition,
    /// eliminating lock contention and ensuring deterministic ordering.
    ///
    /// PERFORMANCE: O(n × m) where n=cuts, m=existing_states
    ///              But single lock → no contention overhead
    pub fn add_cuts_batch(&mut self, cut_state_pairs: Vec<CutStatePair>) -> Vec<CutSelectionResult> {
        let mut results = Vec::with_capacity(cut_state_pairs.len());

        for pair in cut_state_pairs {
            let mut cut = pair.cut;
            let mut state = pair.state;

            // Assign ID and evaluate dominance
            cut.id = self.cut_pool.total_cut_count;
            self.update_cut_pool_on_add(cut.id);
            self.eval_new_cut_domination(&mut cut);
            self.add_cut(cut);

            // Update with new state
            let returning_cut_ids = self.update_old_cuts_domination(&mut state);
            self.add_state(state);

            // Identify cuts to remove
            let removing_cut_ids: Vec<usize> = self
                .cut_pool
                .pool
                .iter()
                .filter(|cut| cut.non_dominated_state_count <= 0 && cut.active)
                .map(|cut| cut.id)
                .collect();

            results.push(CutSelectionResult {
                cut_id: cut.id,
                returning_cut_ids,
                removing_cut_ids,
            });
        }

        results
    }
}

/// Result of batch cut selection for one cut
pub struct CutSelectionResult {
    pub cut_id: usize,
    pub returning_cut_ids: Vec<usize>,
    pub removing_cut_ids: Vec<usize>,
}
```

#### Step 2: Update SDDP backward pass (`sddp/mod.rs`)

```rust
// Replace per-thread locking with batch processing
fn backward_step_at_node_batch(
    &mut self,
    node_id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: bool,
    future_cost_function_graph: &graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
) -> Result<(), String> {
    // PHASE 1: Compute cuts in parallel (NO LOCK)
    let cut_state_pairs: Vec<(usize, fcf::CutStatePair)> = past_node_ids
        .par_iter()
        .map(|&child_id| {
            let child_data_node = node_data_graph.get_node(child_id)?;
            let child_subproblem_node = &self.subproblem_graph.nodes[child_id];

            let cut_state_pair = child_subproblem_node.data.compute_new_cut(
                &forward_trajectory,
                branching_realizations,
                child_data_node.data.risk_measure.as_ref(),
            );

            Ok((child_id, cut_state_pair))
        })
        .collect::<Result<Vec<_>, String>>()?;

    // PHASE 2: Batch cut selection (SINGLE LOCK, deterministic order)
    let parent_fcf_node = future_cost_function_graph.get_node(node_id)?;
    let selection_results = {
        let mut fcf = parent_fcf_node.data.lock().unwrap();
        fcf.add_cuts_batch(cut_state_pairs.into_iter().map(|(_, pair)| pair).collect())
    };  // Lock released here

    // PHASE 3: Apply to local models in parallel (NO LOCK)
    selection_results
        .par_iter()
        .for_each(|result| {
            let parent_subproblem = &mut self.subproblem_graph.get_node_mut(node_id).unwrap();
            parent_subproblem.data.apply_cut_selection_result(result, parent_fcf_node);
        });

    Ok(())
}
```

#### Step 3: Add model update method to Subproblem (`subproblem.rs`)

```rust
impl Subproblem {
    /// Apply cut selection results to local solver model
    ///
    /// This is called after batch cut selection to update the model
    /// with new/returning/removing cuts.
    pub fn apply_cut_selection_result(
        &mut self,
        result: &fcf::CutSelectionResult,
        fcf: &Arc<Mutex<fcf::FutureCostFunction>>,
    ) {
        let fcf_locked = fcf.lock().unwrap();

        // Add new cut to model
        if let Some(model) = self.model.as_mut() {
            let mut cut = fcf_locked.cut_pool.pool[result.cut_id].clone();
            self.state.add_cut_constraint_to_model(&mut cut, &self.variables, model);
        }

        // Return cuts to model
        for &cut_id in &result.returning_cut_ids {
            let cut = &fcf_locked.cut_pool.pool[cut_id];
            if let Some(model) = self.model.as_mut() {
                self.state.add_cut_constraint_to_model(
                    &mut cut.clone(),
                    &self.variables,
                    model,
                );
            }
        }

        // Remove cuts from model
        for &cut_id in &result.removing_cut_ids {
            let cut_index = fcf_locked.get_active_cut_index_by_id(cut_id);
            let row_index = self.first_cut_row_index() + cut_index;
            if let Some(model) = self.model.as_mut() {
                model.delete_row(row_index).unwrap();
            }
        }

        drop(fcf_locked);
    }
}
```

### Backward Compatibility

To maintain backward compatibility during transition:

1. **Feature Flag**: Enable batch selection via `batch-cut-selection` feature
2. **Keep Both Implementations**: Old API still works, new API opt-in
3. **Gradual Migration**: Test extensively before making default

```rust
// In sddp/mod.rs
#[cfg(feature = "batch-cut-selection")]
fn backward_step_implementation(...) {
    self.backward_step_at_node_batch(...)
}

#[cfg(not(feature = "batch-cut-selection"))]
fn backward_step_implementation(...) {
    self.backward_step_at_node_per_thread(...)  // Current implementation
}
```

## Performance Expectations

### Expected Improvements

| Metric                     | Current (Per-Thread) | Batch (Proposed) | Improvement  |
| -------------------------- | -------------------- | ---------------- | ------------ |
| Lock acquisitions/backward | N (threads)          | 1                | -87.5% (8→1) |
| Wasted CPU time            | 350-1400 μs          | 0 μs             | -100%        |
| Total backward pass time   | ~2000 μs             | ~1500 μs         | -25%         |
| Deterministic ordering     | ❌                   | ✅               | ✅           |
| Reproducibility            | ❌                   | ✅               | ✅           |

### Benchmarking Plan

1. **Lock Contention**: Measure with 1, 2, 4, 8 threads
2. **Scaling**: Test with 10, 100, 1000, 10000 cuts
3. **Full SDDP**: 12-stage problem, 30 iterations
4. **Reproducibility**: Same seed → same policy

## Testing Strategy

### Unit Tests (15 tests)

1. **Batch Processing Correctness**:

   - `test_batch_selection_same_as_sequential`: Verify batch gives same cuts as sequential
   - `test_batch_deterministic_ordering`: Same input → same output order
   - `test_batch_empty_pool`: Works with no existing cuts
   - `test_batch_single_cut`: Single cut processed correctly

2. **Edge Cases**:
   - `test_batch_identical_cuts`: Handle duplicate cuts
   - `test_batch_dominated_cuts`: All cuts dominated immediately
   - `test_batch_large_batch`: 1000 cuts at once

### Integration Tests (8 tests)

1. **Full SDDP Runs**:
   - `test_batch_convergence_12stage`: Converges to correct bounds
   - `test_batch_deterministic_policy`: Same seed → same policy
   - `test_batch_vs_perthread_equivalence`: Both produce valid policies

### Performance Tests (20 benchmarks - already implemented!)

See `benches/cut_selection.rs`:

- `bench_cut_selection_scaling`
- `bench_thread_contention`
- `bench_batch_vs_perthread`

## Implementation Checklist

- [ ] Add `CutSelectionResult` struct to `fcf.rs`
- [ ] Implement `add_cuts_batch()` in `FutureCostFunction`
- [ ] Add `apply_cut_selection_result()` to `Subproblem`
- [ ] Implement `backward_step_at_node_batch()` in SDDP
- [ ] Add feature flag `batch-cut-selection`
- [ ] Write 15 unit tests
- [ ] Write 8 integration tests
- [ ] Run full benchmark suite
- [ ] Profile with flamegraph
- [ ] Document performance gains
- [ ] Update TESTING.md and CHANGELOG.md

## Risk Assessment

**Low Risk**:

- Pure refactoring, same algorithmic behavior
- Extensive testing planned
- Backward compatibility via feature flag
- Expected gains well-understood (lock contention elimination)

**Potential Issues**:

- Need to ensure cut.id is not used after move (save before add_cut)
- Model updates still need FCF lock (but brief)
- Need careful Vec pre-allocation for `cut_state_pairs`

## Next Steps

1. ✅ Create design document (this file)
2. Implement `CutSelectionResult` and `add_cuts_batch()`
3. Update SDDP backward pass
4. Write unit tests
5. Run benchmarks and profile
6. Measure performance gains
7. Document and finalize
