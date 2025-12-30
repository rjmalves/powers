# [T-062] Implement compute_cut_into_staging() on Handler

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Handler Staging Buffers](./00-sprint-overview.md)
> **Dependencies**: [T-061](./ticket-061-add-staging-to-handler.md)
> **Blocks**: [T-064](./ticket-064-parallel-then-sequential.md)

## Files to Read Before Starting

- `src/sddp/mod.rs:777-880` - `compute_cut_into_slot_for_backward_step()` (existing pattern)
- `src/sddp/mod.rs:700-775` - `compute_cut_data_for_backward_step()` (current allocating path)
- `src/state.rs:1118-1151` - `compute_cut_into_slot()` (zero-alloc path)
- `src/memory/buffers.rs` - `CutStagingBuffer::stage_from()`

---

## Context

### Background

The handler needs a method that computes a cut and stages the result into its local buffer, without touching the global pools. This enables parallel execution—each handler computes independently, then results are merged sequentially.

### Current Flow (Allocating)

```
compute_cut_data_for_backward_step()
    → solve branchings
    → state.compute_cut_data()
        → CutData::from_refs() ← ALLOCATES
    → return CutData
```

### Target Flow (Zero-Allocation)

```
compute_cut_into_staging()
    → solve branchings
    → state.evaluate_cut_ref() ← returns references
    → staging.stage_from() ← copies to handler buffer
    → return timing only
```

---

## Specification

### New Method on SddpTrainHandler

```rust
/// Compute cut and stage into handler's local buffer.
///
/// This method is designed for parallel execution. Each handler:
/// 1. Solves LP for all branching scenarios
/// 2. Extracts duals and computes cut coefficients
/// 3. Copies result to handler's staging buffer
///
/// The staging buffer contents are later copied to global pools
/// in a sequential loop (for deterministic ordering).
///
/// # Arguments
///
/// * `id` - Node ID for this stage
/// * `past_node_ids` - Node IDs from forward trajectory
/// * `node_data_graph` - Graph with node data
/// * `saa` - Scenario tree for branching counts
/// * `iteration` - Current iteration (1-based)
/// * `forward_pass_idx` - Forward pass index (0-based)
///
/// # Returns
///
/// Timing information from the computation. The cut data is in `self.cut_staging`.
pub fn compute_cut_into_staging(
    &mut self,
    id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::ScenarioTree,
    iteration: usize,
    forward_pass_idx: usize,
) -> Result<BackwardPhase1Timing, String>;
```

### Implementation Approach

Based on existing `compute_cut_into_slot_for_backward_step()` but:
1. Uses `evaluate_cut_ref()` instead of `compute_cut_into_slot()`
2. Stages result to handler buffer instead of global pools
3. Returns only timing (cut data is in staging buffer)

---

## Acceptance Criteria

- [ ] Method `compute_cut_into_staging()` implemented
- [ ] After call, `self.cut_staging` contains valid cut data
- [ ] No heap allocations during computation (except LP solver internals)
- [ ] Produces identical cut coefficients as `compute_cut_data_for_backward_step()`
- [ ] All existing tests pass

---

## Implementation Guide

### Step 1: Add method signature

In `src/sddp/mod.rs`, in `impl SddpTrainHandler`:

```rust
pub fn compute_cut_into_staging(
    &mut self,
    id: usize,
    past_node_ids: &[usize],
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::ScenarioTree,
    iteration: usize,
    forward_pass_idx: usize,
) -> Result<BackwardPhase1Timing, String> {
    // Implementation here
}
```

### Step 2: Copy LP solving logic

Copy the LP solving and trajectory extraction from `compute_cut_into_slot_for_backward_step()`:

```rust
let mut timing = BackwardPhase1Timing::default();

let model_preprocessing_start = std::time::Instant::now();

// Build forward trajectory (same as existing)
let node_forward_trajectory: Vec<&subproblem::Realization> = past_node_ids
    .iter()
    .map(|&past_id| {
        self.realization_graph
            .get_node(past_id)
            .map(|node| &node.data)
            .ok_or_else(|| format!("Could not find realization for past_node {}", past_id))
    })
    .collect::<Result<_, _>>()?;

let num_branchings = saa.get_branching_count_at_stage(id).ok_or_else(|| {
    format!("Missing branching count for node {} in backward pass", id)
})?;
timing.model_preprocessing_time = model_preprocessing_start.elapsed();

// Solve all branchings (same as existing)
let branchings_timing = solve_all_branchings(
    &mut self.subproblem_graph,
    &mut self.branching_graph,
    id,
    num_branchings,
    &node_forward_trajectory,
    saa,
)?;
timing.solver_time = branchings_timing.solver_time;
```

### Step 3: Compute cut into staging

Instead of calling `compute_cut_into_slot()`, use `evaluate_cut_ref()` and stage:

```rust
let model_postprocessing_start = std::time::Instant::now();

let branching_node_data = &self
    .branching_graph
    .get_node(id)
    .ok_or_else(|| format!("Could not find branching realizations for node {}", id))?
    .data;

let child_data_node = node_data_graph.get_node(id).ok_or_else(|| {
    format!("Could not find node data for node {}", id)
})?;
let child_subproblem_node = self.subproblem_graph.get_node_mut(id).ok_or_else(|| {
    format!("Could not find subproblem for node {}", id)
})?;

// Get state reference
let state = &mut child_subproblem_node.data.state;

// Set tracking fields
state.set_iteration(iteration);
state.set_forward_pass_idx(forward_pass_idx);

// Evaluate cut using thread-local buffers (returns references)
use crate::memory::with_cut_buffers;

with_cut_buffers(|buffers| {
    let eval_result = state.evaluate_cut_ref(
        child_data_node.data.risk_measure.as_ref(),
        branching_node_data,
        buffers,
    );
    
    // Stage into handler buffer
    self.cut_staging.stage_from(
        &eval_result,
        state.coefficients(),
        iteration,
        forward_pass_idx,
        timing.clone(),  // Or construct timing after
    );
});

timing.model_postprocessing_time = model_postprocessing_start.elapsed();

Ok(timing)
```

### Step 4: Handle backward detail history

If `preserve_backward_detail` is enabled, capture branching realizations (copy from existing method):

```rust
if self.preserve_backward_detail {
    if let Some(ref mut history) = self.backward_detail_history {
        for (branching_idx, realization) in branching_node_data.iter().enumerate() {
            history.push(BackwardPassDetail {
                iteration,
                forward_pass_idx,
                stage_id: id as isize,
                training_state_id: 0,
                branching_idx,
                realization: realization.clone(),
            });
        }
    }
}
```

---

## Testing Requirements

### Unit Tests

Add test in `src/sddp/mod.rs` tests section:

```rust
#[test]
fn test_compute_cut_into_staging() {
    // Setup minimal handler (use existing test patterns)
    // ...
    
    // Call compute_cut_into_staging
    let timing = handler.compute_cut_into_staging(
        stage_id,
        &past_node_ids,
        &node_data_graph,
        &saa,
        1,  // iteration
        0,  // forward_pass_idx
    ).unwrap();
    
    // Verify staging buffer is populated
    assert!(handler.staging_buffer().populated);
    assert!(!handler.staging_buffer().cut_coefficients.iter().all(|&x| x == 0.0));
}
```

### Equivalence Test

Verify staging produces same results as allocating path:

```rust
#[test]
fn test_staging_matches_cutdata() {
    // Setup handler
    // ...
    
    // Compute via allocating path
    let (cut_data, _timing1) = handler.compute_cut_data_for_backward_step(
        stage_id, past_node_ids, node_data_graph, saa, 1, 0
    ).unwrap();
    
    // Compute via staging path
    let _timing2 = handler.compute_cut_into_staging(
        stage_id, past_node_ids, node_data_graph, saa, 1, 0
    ).unwrap();
    let staging = handler.staging_buffer();
    
    // Compare
    assert_eq!(staging.cut_coefficients[..cut_data.cut_coefficients.len()], 
               cut_data.cut_coefficients[..]);
    assert_eq!(staging.state_coefficients[..cut_data.state_coefficients.len()],
               cut_data.state_coefficients[..]);
    assert!((staging.cut_rhs - cut_data.cut_rhs).abs() < 1e-12);
}
```

---

## Pitfalls to Avoid

- ⚠️ **Timing struct clone**: `BackwardPhase1Timing` may not implement `Clone`. Either derive it or construct timing before `with_cut_buffers`.
- ⚠️ **Borrow checker in closure**: `with_cut_buffers` takes a closure. `self.cut_staging` borrow may conflict. May need to restructure.
- ⚠️ **Thread-local initialization**: Ensure `initialize_cut_buffers()` was called (rayon broadcast).

### Borrow Checker Strategy

If the closure causes issues, compute first, then stage:

```rust
// Option A: Extract results before staging
let (coeffs, rhs) = with_cut_buffers(|buffers| {
    let eval = state.evaluate_cut_ref(risk_measure, branchings, buffers);
    // Clone references to owned data? No, we want to avoid allocation.
    // Return what we need
    (eval.coefficients.to_vec(), eval.rhs)  // This allocates! Bad.
});

// Option B: Restructure to avoid self borrow in closure
// Get staging buffer pointer before closure
let staging = &mut self.cut_staging;
let state_coeffs = state.coefficients(); // Get reference

with_cut_buffers(|buffers| {
    let eval = state.evaluate_cut_ref(risk_measure, branchings, buffers);
    staging.stage_from(&eval, state_coeffs, iteration, forward_pass_idx, timing);
});
```

Option B works because we capture `staging` and `state_coeffs` before the closure.

---

## Documentation Requirements

- [ ] Doc comment on method
- [ ] Update handler documentation to mention staging path

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Core parallel-enabling method. Main complexity is borrow checker management and ensuring equivalence with allocating path.

---

## Definition of Done

- [ ] Method implemented
- [ ] Staging buffer populated after call
- [ ] Produces identical results to allocating path
- [ ] No new heap allocations (except LP solver)
- [ ] All existing tests pass (549+)
- [ ] Doc comments complete
