# [T-064] Update Coordinator for Parallel-Then-Sequential Pattern

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Handler Staging Buffers](./00-sprint-overview.md)
> **Dependencies**: [T-062](./ticket-062-compute-into-staging.md), [T-063](./ticket-063-update-from-staging.md)
> **Blocks**: [T-065](../sprint-02/ticket-065-wire-backward-pass.md) (Sprint 2)

## Files to Read Before Starting

- `src/algorithm/coordinator.rs:176-221` - Existing `compute_cuts_into_slots()` (sequential)
- `src/algorithm/coordinator.rs:224-267` - `compute_cuts_parallel()` (parallel but allocating)
- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Architecture context

---

## Context

### Background

The coordinator manages parallel execution across train handlers. Currently:
- `compute_cuts_parallel()`: Parallel but allocates `CutData`
- `compute_cuts_into_slots()`: Zero-alloc but sequential (due to mutable pool access)

We need a new method that combines both: parallel computation, zero allocation.

### Current Sequential Path

```rust
// coordinator.rs:192
for (fp_idx, handler) in self.handlers.iter_mut().enumerate() {
    let (slot, timing) = handler.compute_cut_into_slot_for_backward_step(...)?;
    slots.push(slot);
}
```

### Target Parallel-Then-Sequential Path

```rust
// Phase 1a: Parallel compute into staging buffers
let timings: Vec<_> = self.handlers
    .par_iter_mut()
    .map(|handler| handler.compute_cut_into_staging(...))
    .collect::<Result<_, _>>()?;

// Phase 1b: Sequential copy to pools (deterministic order)
for handler in self.handlers.iter() {
    let slot = cut_pool.update_from_staging(handler.staging_buffer(), state_pool);
    slots.push(slot);
}
```

---

## Specification

### New Trait Method

Add to `BackwardStageProcessor` trait:

```rust
/// Compute cuts into staging buffers (parallel), then copy to pools (sequential).
///
/// This method enables parallel cut computation while maintaining
/// deterministic reproducibility via sequential pool updates.
///
/// # Arguments
///
/// * `stage_ctx` - Context for this backward stage
/// * `fcf_graph` - FCF graph for pool access
///
/// # Returns
///
/// Slot indices and aggregated timing.
fn compute_cuts_parallel_into_slots(
    &mut self,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase1SlotResult, String>;
```

### Implementation on ParallelHandlerCoordinator

```rust
fn compute_cuts_parallel_into_slots(
    &mut self,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase1SlotResult, String> {
    let parent_id = stage_ctx.parent_id.ok_or_else(|| ...)?;
    
    let parent_fcf_node = fcf_graph.get_node_mut(parent_id).ok_or_else(|| ...)?;
    let fcf = &mut parent_fcf_node.data;
    
    let phase1_begin = Instant::now();
    
    // Phase 1a: Parallel compute into staging buffers
    let timings: Vec<BackwardPhase1Timing> = self.handlers
        .par_iter_mut()
        .map(|handler| {
            handler.compute_cut_into_staging(
                stage_ctx.stage_id,
                stage_ctx.past_node_ids,
                stage_ctx.node_data_graph,
                stage_ctx.saa,
                stage_ctx.iteration,
                // forward_pass_idx from handler? Need to track this.
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    
    // Phase 1b: Sequential copy to pools (deterministic order)
    let mut slots = Vec::with_capacity(self.num_forward_passes);
    for (fp_idx, handler) in self.handlers.iter().enumerate() {
        let slot = fcf.cut_pool.update_from_staging(
            handler.staging_buffer(),
            &mut fcf.state_pool,
        );
        slots.push(slot);
    }
    
    let phase1_wall_time = phase1_begin.elapsed();
    
    // Aggregate timing
    let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
    let solver_calls = self.num_forward_passes * num_branchings;
    let timing = self.scale_timing(&timings, phase1_wall_time, solver_calls);
    
    // Sort for deterministic ordering (should already be in order)
    slots.sort_unstable();
    
    Ok(Phase1SlotResult { slots, timing })
}
```

---

## Acceptance Criteria

- [ ] New method `compute_cuts_parallel_into_slots()` added to trait
- [ ] Implemented on `ParallelHandlerCoordinator`
- [ ] Phase 1a runs in parallel (`par_iter_mut`)
- [ ] Phase 1b runs sequentially in forward_pass_idx order
- [ ] Produces identical slots and timing as existing methods
- [ ] All existing tests pass

---

## Implementation Guide

### Step 1: Add trait method

In `src/algorithm/processor.rs`, add to `BackwardStageProcessor` trait:

```rust
/// Compute cuts into staging buffers (parallel), then copy to pools (sequential).
///
/// This is the preferred method for production use. It provides:
/// - Full parallelism in cut computation (Phase 1a)
/// - Zero heap allocations
/// - Deterministic reproducibility via sequential pool updates (Phase 1b)
fn compute_cuts_parallel_into_slots(
    &mut self,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase1SlotResult, String>;
```

### Step 2: Track forward_pass_idx in handlers

Handlers need to know their forward_pass_idx. Options:
1. Pass it during `compute_cut_into_staging()` call
2. Store it in handler during construction
3. Use `enumerate()` in the parallel loop

Best option: Use enumerate + pass to method:

```rust
self.handlers
    .par_iter_mut()
    .enumerate()
    .map(|(fp_idx, handler)| {
        handler.compute_cut_into_staging(
            stage_ctx.stage_id,
            stage_ctx.past_node_ids,
            stage_ctx.node_data_graph,
            stage_ctx.saa,
            stage_ctx.iteration,
            fp_idx,  // Pass forward_pass_idx
        )
    })
```

### Step 3: Implement on coordinator

In `src/algorithm/coordinator.rs`, add implementation:

```rust
fn compute_cuts_parallel_into_slots(
    &mut self,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase1SlotResult, String> {
    let parent_id = stage_ctx.parent_id.ok_or_else(|| {
        format!(
            "No parent ID for stage {} (stage_idx {})",
            stage_ctx.stage_id, stage_ctx.stage_idx
        )
    })?;

    let parent_fcf_node = fcf_graph.get_node_mut(parent_id).ok_or_else(|| {
        format!("Could not find FCF for parent node {}", parent_id)
    })?;
    let fcf = &mut parent_fcf_node.data;

    let phase1_begin = Instant::now();

    // Phase 1a: Parallel compute into staging buffers
    // Each handler writes to its own staging buffer - no contention
    let timings: Vec<BackwardPhase1Timing> = self.handlers
        .par_iter_mut()
        .enumerate()
        .map(|(fp_idx, handler)| {
            handler.compute_cut_into_staging(
                stage_ctx.stage_id,
                stage_ctx.past_node_ids,
                stage_ctx.node_data_graph,
                stage_ctx.saa,
                stage_ctx.iteration,
                fp_idx,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Phase 1b: Sequential copy to pools
    // Deterministic order (0, 1, 2, ...) for reproducibility
    let mut slots = Vec::with_capacity(self.num_forward_passes);
    for handler in self.handlers.iter() {
        let slot = fcf.cut_pool.update_from_staging(
            handler.staging_buffer(),
            &mut fcf.state_pool,
        );
        slots.push(slot);
    }

    let phase1_wall_time = phase1_begin.elapsed();

    // Calculate solver calls
    let num_branchings = stage_ctx.get_branching_count().unwrap_or(1);
    let solver_calls = self.num_forward_passes * num_branchings;

    let timing = self.scale_timing(&timings, phase1_wall_time, solver_calls);

    // Sort for deterministic ordering (already should be in order by construction)
    slots.sort_unstable();

    Ok(Phase1SlotResult { slots, timing })
}
```

### Step 4: Verify borrow checker

The tricky part: `fcf` is borrowed mutably for pools, but we're iterating handlers. This should work because:
- Phase 1a: Only handlers are borrowed (par_iter_mut), fcf not accessed
- Phase 1b: Handlers borrowed immutably (iter), fcf borrowed mutably

If borrow checker complains, may need to restructure:
```rust
// Get pools first
let (cut_pool, state_pool) = {
    let fcf_node = fcf_graph.get_node_mut(parent_id)?;
    (&mut fcf_node.data.cut_pool, &mut fcf_node.data.state_pool)
};

// Then use in Phase 1b
```

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_compute_cuts_parallel_into_slots() {
    // Setup coordinator with multiple handlers
    // ...
    
    let result = coordinator.compute_cuts_parallel_into_slots(
        &stage_ctx,
        &mut fcf_graph,
    ).unwrap();
    
    // Verify correct number of slots
    assert_eq!(result.slots.len(), num_forward_passes);
    
    // Verify slots are in sorted order
    let mut sorted = result.slots.clone();
    sorted.sort();
    assert_eq!(result.slots, sorted);
    
    // Verify timing is aggregated
    assert!(result.timing.solver > Duration::ZERO);
}
```

### Equivalence Test

```rust
#[test]
fn test_parallel_into_slots_matches_sequential() {
    // Setup two identical coordinators
    // ...
    
    // Method 1: Sequential (existing)
    let result1 = coord1.compute_cuts_into_slots(&stage_ctx, &mut fcf1)?;
    
    // Method 2: Parallel-then-sequential (new)
    let result2 = coord2.compute_cuts_parallel_into_slots(&stage_ctx, &mut fcf2)?;
    
    // Should produce identical slots
    assert_eq!(result1.slots, result2.slots);
    
    // Verify pool contents match
    for slot in result1.slots.iter() {
        let cut1 = &fcf1.cut_pool.pool[*slot];
        let cut2 = &fcf2.cut_pool.pool[*slot];
        assert_eq!(cut1.coefficients, cut2.coefficients);
        assert_eq!(cut1.rhs, cut2.rhs);
    }
}
```

---

## Pitfalls to Avoid

- ⚠️ **Rayon parallel order**: `par_iter_mut().enumerate()` preserves indices but execution order is non-deterministic. This is fine—each handler has its own staging buffer.
- ⚠️ **Borrow checker**: May need careful structuring to satisfy borrow rules between Phase 1a and 1b.
- ⚠️ **Handler staging buffer access**: In Phase 1b, we only need immutable access to staging buffers.

---

## Documentation Requirements

- [ ] Doc comments on new trait method
- [ ] Explain parallel-then-sequential pattern
- [ ] Reference architecture document

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Core integration point. Complexity in borrow checker management and ensuring parallel correctness.

---

## Definition of Done

- [ ] Trait method added
- [ ] Implementation complete
- [ ] Parallel execution verified (timing should show parallel speedup)
- [ ] Produces identical results to sequential path
- [ ] All existing tests pass (549+)
- [ ] Doc comments complete
