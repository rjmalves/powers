# [TICKET-004] Modify FCF initialization to use preallocation

> **Epic**: [Epic 1: Preallocated Cut Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-003](./ticket-003-implement-cutpool-preallocate.md)
> **Blocks**: [TICKET-005](./ticket-005-update-add-cuts-batch.md)

## Context

### Background

FCF (FutureCostFunction) initialization currently uses `reserve()` which only preallocates the Vec capacity. We need to use the new `BendersCutPool::preallocate()` to fully preallocate all cuts.

### Relation to Epic

Integrates the cut pool preallocation into the training initialization flow.

### Current State

```rust
// src/sddp/mod.rs:1795-1803
let max_cuts = num_forward_passes * num_iterations;
let max_states = num_forward_passes * num_iterations;

for fcf_node in self.future_cost_function_graph.iter_nodes() {
    let mut fcf = fcf_node.data.lock().unwrap();
    fcf.cut_pool.pool.reserve(max_cuts);
    fcf.cut_pool.active_cut_indices.reserve(max_cuts);
    fcf.state_pool.pool.reserve(max_states);
}
```

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training loop and current FCF initialization
- `src/fcf.rs` - FutureCostFunction structure
- `src/cut.rs` - BendersCutPool with new preallocate() method
- `MEMORY_STABILITY_ANALYSIS.md` - Integration points section

## Specification

### Inputs

For each FCF node, we need:
- `num_iterations: usize` - Training iterations
- `num_forward_passes: usize` - Forward passes per iteration
- `state_dimension: usize` - Coefficient dimension for this node's state

### Behavior

1. For each FCF node in the graph:
   - Get the corresponding node's state dimension from `state_choice` and system
   - Create a new `BendersCutPool::preallocate(num_iterations, num_forward_passes, state_dimension)`
   - Assign to `fcf.cut_pool`
2. State pool preallocation is handled in Epic 2

### Error Handling

- State dimension must be > 0
- Panic if preallocation fails (out of memory)

## Acceptance Criteria

- [ ] All FCF cut pools fully preallocated at training start
- [ ] State dimension correctly computed for each node
- [ ] Existing reserve() calls replaced with preallocate()
- [ ] Training runs successfully with preallocated pools
- [ ] All existing tests pass

## Implementation Guide

### Suggested Approach

1. Add method to `FutureCostFunction` for full preallocation:
   ```rust
   impl FutureCostFunction {
       pub fn preallocate_pools(
           num_iterations: usize,
           num_forward_passes: usize,
           state_dimension: usize,
       ) -> Self {
           Self {
               cut_pool: BendersCutPool::preallocate(
                   num_iterations,
                   num_forward_passes,
                   state_dimension,
               ),
               state_pool: VisitedStatePool::with_capacity(
                   num_iterations * num_forward_passes,
               ),
           }
       }
   }
   ```

2. Modify FCF initialization in `src/sddp/mod.rs`:
   ```rust
   for fcf_node in self.future_cost_function_graph.iter_nodes() {
       let node_data = /* get corresponding NodeData */;
       let state_dim = /* compute from state_choice and system */;
       
       let mut fcf = fcf_node.data.lock().unwrap();
       *fcf = FutureCostFunction::preallocate_pools(
           num_iterations,
           num_forward_passes,
           state_dim,
       );
   }
   ```

3. Computing state dimension:
   - For `storage` state: `state_dim = system.hydros.len()`
   - For `storage_and_inflow` state: Use `total_state_dim()` from `state.rs`

### Key Files to Modify

- `src/fcf.rs`: Add `preallocate_pools()` method
- `src/sddp/mod.rs`: Modify FCF initialization (~line 1795)

### Pitfalls to Avoid

- ⚠️ State dimension varies per node based on `state_choice` setting
- ⚠️ Must get state dimension before creating FCF (not from existing FCF state)
- ⚠️ Don't break parallel handler preallocation

## Testing Requirements

### Unit Tests

- [ ] Test `FutureCostFunction::preallocate_pools()` creates correct structures
- [ ] Test state dimension computation for both state types

### Integration Tests

- [ ] Run example-01 and verify training completes
- [ ] Run example-05 and verify training completes
- [ ] Verify lower bounds match baseline

## Documentation Requirements

- [ ] Update `FutureCostFunction` doc comments
- [ ] Add comment explaining state dimension computation

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Integration with existing code, state dimension computation needs care
