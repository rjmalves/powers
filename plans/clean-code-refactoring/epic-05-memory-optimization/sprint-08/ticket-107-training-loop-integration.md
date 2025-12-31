# [T-107] Integrate Model Rebuild into Training Loop

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Dependencies**: T-104, T-105
> **Blocks**: T-108, T-106
> **Priority**: 1 (Critical)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `src/sddp/mod.rs` - Training loop in `SddpAlgorithm::train()`
- `src/subproblem.rs` - `rebuild_model()` from T-104
- `src/memory/rss.rs` - RSS monitoring from T-105
- Sprint overview for architecture diagram

---

## Context

### Background

T-104 implements `Subproblem::rebuild_model()` and T-105 provides RSS monitoring. This ticket integrates them into the training loop with proper coordination across handlers.

### Integration Points

1. **Handler level**: Each `SddpTrainHandler` manages multiple subproblems
2. **Coordinator level**: `ParallelHandlerCoordinator` manages all handlers
3. **Algorithm level**: `SddpAlgorithm::train()` orchestrates iterations

---

## Specification

### New Methods

```rust
impl SddpTrainHandler {
    /// Rebuild all subproblem models in this handler to reclaim memory.
    ///
    /// Extracts active cuts from each stage's FCF, rebuilds the HiGHS model,
    /// and restores the cuts.
    ///
    /// # Arguments
    ///
    /// * `node_data_graph` - Graph with system/temporal model data
    /// * `fcf_graph` - Future cost function graph (source of cut pools)
    /// * `remaining_iterations` - Remaining iterations for cut sizing
    /// * `num_forward_passes` - Forward passes per iteration
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if any subproblem fails to rebuild.
    pub fn rebuild_all_models(
        &mut self,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        fcf_graph: &graph::DirectedGraph<fcf::FutureCostFunction>,
        remaining_iterations: usize,
        num_forward_passes: usize,
    ) -> Result<(), String>;
}
```

### Training Loop Integration

```rust
// In SddpAlgorithm::train(), after backward pass and before next iteration:

const MODEL_REBUILD_INTERVAL: usize = 100;  // Configurable

for index in 0..num_iterations {
    let iteration = index + 1;
    
    // ... existing forward/backward pass logic ...
    
    // Periodic model rebuild for memory reclaim
    if iteration % MODEL_REBUILD_INTERVAL == 0 && iteration < num_iterations {
        let remaining_iterations = num_iterations - iteration;
        
        // Log RSS before rebuild
        let rss_before = crate::memory::get_rss_bytes();
        
        // Rebuild all handler models in parallel
        let rebuild_start = Instant::now();
        coordinator.handlers_mut()
            .par_iter_mut()
            .try_for_each(|handler| {
                handler.rebuild_all_models(
                    &self.node_data_graph,
                    &self.future_cost_function_graph,
                    remaining_iterations,
                    num_forward_passes,
                )
            })?;
        let rebuild_time = rebuild_start.elapsed();
        
        // Log RSS after rebuild
        let rss_after = crate::memory::get_rss_bytes();
        
        if let (Some(before), Some(after)) = (rss_before, rss_after) {
            let freed = before.saturating_sub(after);
            log::info!(
                "Model rebuild at iter {}: {} -> {} (freed {}) in {:?}",
                iteration,
                crate::memory::format_rss(before),
                crate::memory::format_rss(after),
                crate::memory::format_rss(freed),
                rebuild_time
            );
        }
    }
}
```

### Handler Implementation

```rust
impl SddpTrainHandler {
    pub fn rebuild_all_models(
        &mut self,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        fcf_graph: &graph::DirectedGraph<fcf::FutureCostFunction>,
        remaining_iterations: usize,
        num_forward_passes: usize,
    ) -> Result<(), String> {
        // Collect node IDs to avoid borrowing issues
        let node_ids: Vec<usize> = self.subproblem_graph
            .iter_nodes()
            .map(|n| n.id)
            .collect();
        
        for node_id in node_ids {
            // Get node data for rebuild
            let node_data = node_data_graph.get_node(node_id)
                .ok_or_else(|| format!("Node {} not found in data graph", node_id))?;
            
            // Get FCF for this node to extract active cuts
            let fcf_node = fcf_graph.get_node(node_id)
                .ok_or_else(|| format!("Node {} not found in FCF graph", node_id))?;
            
            // Get subproblem
            let subproblem = self.subproblem_graph.get_node_mut(node_id)
                .ok_or_else(|| format!("Node {} not found in subproblem graph", node_id))?;
            
            // Extract temporal models
            let temporal_models: Vec<_> = node_data.data.uncertainty_models
                .iter()
                .cloned()
                .collect();
            
            // Extract active cuts
            let active_cuts = subproblem.data.extract_active_cuts(
                &fcf_node.data.cut_pool.pool
            );
            
            // Rebuild
            subproblem.data.rebuild_model(
                &active_cuts,
                &node_data.data.system,
                &temporal_models,
                remaining_iterations,
                num_forward_passes,
            )?;
        }
        
        Ok(())
    }
}
```

---

## Acceptance Criteria

- [ ] `rebuild_all_models()` implemented in `SddpTrainHandler`
- [ ] Training loop calls rebuild at configured interval
- [ ] RSS logging shows memory before/after rebuild
- [ ] Rebuild timing is logged
- [ ] Training produces same results with rebuild enabled
- [ ] Golden tests pass

---

## Implementation Guide

### Suggested Approach

1. **Add `rebuild_all_models()` to `SddpTrainHandler`** in `src/sddp/mod.rs`

2. **Modify training loop** in `SddpAlgorithm::train()`:
   - Add rebuild interval constant (initially hardcoded, T-108 makes configurable)
   - Add rebuild block after backward pass processing
   - Add RSS logging around rebuild

3. **Test incrementally**:
   - First test rebuild_all_models in isolation
   - Then test training loop integration with small iteration count
   - Finally run full golden tests

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Add `rebuild_all_models()`, modify `train()` |

### Pitfalls to Avoid

- ⚠️ **Don't rebuild on last iteration**: Unnecessary work
- ⚠️ **Pass correct remaining_iterations**: For proper cut slot sizing
- ⚠️ **Handle parallel rebuild errors**: Use `try_for_each` not `for_each`
- ⚠️ **Borrow checker**: Collect node IDs before mutating subproblems

---

## Testing Requirements

### Unit Tests

- [ ] `rebuild_all_models()` succeeds with no cuts
- [ ] `rebuild_all_models()` succeeds with active cuts
- [ ] Error handling for missing nodes

### Integration Tests

- [ ] Training with rebuild at interval 10 produces same final bounds
- [ ] Golden tests pass with rebuild enabled
- [ ] RSS decreases after rebuild (manual verification)

---

## Documentation Requirements

- [ ] Doc comments for `rebuild_all_models()`
- [ ] Comment explaining rebuild interval choice
- [ ] Update training documentation if exists

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Integration work building on T-104 infrastructure

---

## Definition of Done

- [ ] `rebuild_all_models()` implemented
- [ ] Training loop integration complete
- [ ] RSS logging working
- [ ] Tests passing
- [ ] Golden tests passing
- [ ] PR merged
