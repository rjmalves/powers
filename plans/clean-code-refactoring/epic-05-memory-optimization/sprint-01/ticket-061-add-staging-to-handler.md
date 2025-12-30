# [T-061] Add Staging Buffer to SddpTrainHandler

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Handler Staging Buffers](./00-sprint-overview.md)
> **Dependencies**: [T-060](./ticket-060-create-staging-buffer.md)
> **Blocks**: [T-062](./ticket-062-compute-into-staging.md)

## Files to Read Before Starting

- `src/sddp/mod.rs:323-450` - `SddpTrainHandler` struct and `new()`
- `src/memory/buffers.rs` - `CutStagingBuffer` (from T-060)
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Architecture context

---

## Context

### Background

Each `SddpTrainHandler` needs its own staging buffer to enable parallel cut computation. The buffer is created during handler initialization and reused across all stages and iterations.

### Current State

```rust
pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,
    forward_detail_history: Option<Vec<ForwardPassDetail>>,
    preserve_forward_detail: bool,
    backward_detail_history: Option<Vec<BackwardPassDetail>>,
    preserve_backward_detail: bool,
}
```

### Target State

```rust
pub struct SddpTrainHandler {
    // ... existing fields ...
    
    /// Staging buffer for zero-allocation parallel cut computation.
    /// Holds one cut + state, reused across stages.
    cut_staging: CutStagingBuffer,
}
```

---

## Specification

### Changes to SddpTrainHandler

1. Add `cut_staging: CutStagingBuffer` field
2. Initialize in `new()` with appropriate state dimension
3. Add accessor method `staging_buffer_mut()`

### State Dimension Calculation

The staging buffer needs the maximum state dimension across all nodes. This is already computed during handler initialization for other purposes.

```rust
// In SddpTrainHandler::new()
let max_state_dim = node_data_graph
    .iter_nodes()
    .map(|node| state::calculate_state_dimension(&node.data))
    .max()
    .unwrap_or(0);

let cut_staging = CutStagingBuffer::new(max_state_dim);
```

---

## Acceptance Criteria

- [ ] `SddpTrainHandler` has `cut_staging` field
- [ ] Field initialized in `new()` with correct dimension
- [ ] Accessor method `staging_buffer_mut()` returns `&mut CutStagingBuffer`
- [ ] All existing handler tests pass
- [ ] Memory overhead is ~1.6 KB per handler (for 100-dim state)

---

## Implementation Guide

### Step 1: Add import

In `src/sddp/mod.rs`, add:
```rust
use crate::memory::CutStagingBuffer;
```

### Step 2: Add field to struct

```rust
pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,
    forward_detail_history: Option<Vec<ForwardPassDetail>>,
    preserve_forward_detail: bool,
    backward_detail_history: Option<Vec<BackwardPassDetail>>,
    preserve_backward_detail: bool,
    
    /// Staging buffer for zero-allocation parallel cut computation.
    ///
    /// Holds one computed cut + state coefficients. Reused across all
    /// backward pass stages within an iteration. Each handler has its
    /// own buffer, enabling parallel cut computation.
    cut_staging: CutStagingBuffer,
}
```

### Step 3: Calculate state dimension helper

Add helper function (or use existing if available):

```rust
fn calculate_max_state_dimension(
    node_data_graph: &graph::DirectedGraph<NodeData>,
) -> usize {
    node_data_graph
        .iter_nodes()
        .map(|node| {
            state::calculate_state_dimension(
                &node.data.system,
                &node.data.uncertainty_models,
            )
        })
        .max()
        .unwrap_or(0)
}
```

Note: Check if `state::calculate_state_dimension` exists or needs parameters adjusted.

### Step 4: Initialize in new()

In `SddpTrainHandler::new()`, add before the final `Ok(Self { ... })`:

```rust
// Calculate max state dimension for staging buffer
let max_state_dim = calculate_max_state_dimension(node_data_graph);
let cut_staging = CutStagingBuffer::new(max_state_dim);
```

And add to the struct initialization:
```rust
Ok(Self {
    subproblem_graph,
    realization_graph,
    branching_graph,
    forward_detail_history,
    preserve_forward_detail,
    backward_detail_history,
    preserve_backward_detail,
    cut_staging,  // NEW
})
```

### Step 5: Add accessor

```rust
impl SddpTrainHandler {
    /// Get mutable reference to staging buffer.
    ///
    /// Used during parallel cut computation to stage results
    /// before sequential copy to global pools.
    #[inline]
    pub fn staging_buffer_mut(&mut self) -> &mut CutStagingBuffer {
        &mut self.cut_staging
    }
    
    /// Get immutable reference to staging buffer.
    #[inline]
    pub fn staging_buffer(&self) -> &CutStagingBuffer {
        &self.cut_staging
    }
}
```

---

## Testing Requirements

### Unit Tests

The handler tests are mostly integration-level. Add a simple test to verify staging buffer exists:

```rust
#[test]
fn test_handler_has_staging_buffer() {
    // Create minimal node data graph for test
    // ... (use existing test setup patterns)
    
    let handler = SddpTrainHandler::new(
        &node_data_graph,
        &initial_condition,
        &saa,
        false,
        false,
        1,
        1,
    ).unwrap();
    
    // Verify staging buffer has correct capacity
    let staging = handler.staging_buffer();
    assert!(staging.cut_coefficients.capacity() > 0);
}
```

### Integration Tests

No new integration tests needed—existing tests verify handler creation works.

---

## Pitfalls to Avoid

- ⚠️ **State dimension calculation**: Make sure to use the same dimension as cut pools. Check existing `FutureCostFunction::preallocate_pools()` for reference.
- ⚠️ **Order of initialization**: `cut_staging` must be initialized after node graphs are created.
- ⚠️ **Zero-dimension edge case**: Handle case where state dimension is 0 (shouldn't happen in practice, but be defensive).

---

## Documentation Requirements

- [ ] Doc comment on `cut_staging` field
- [ ] Doc comments on accessor methods
- [ ] Update handler module documentation if it exists

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple field addition and initialization. Main complexity is finding correct state dimension.

---

## Definition of Done

- [ ] Field added to struct
- [ ] Initialized correctly in `new()`
- [ ] Accessor methods added
- [ ] All existing tests pass (549+)
- [ ] Doc comments complete
