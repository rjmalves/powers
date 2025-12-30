# [T-065] Update backward_pass.rs to Use Staging Path

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Training Loop Integration](./00-sprint-overview.md)
> **Dependencies**: [T-064](../sprint-01/ticket-064-parallel-then-sequential.md)
> **Blocks**: [T-066](./ticket-066-golden-tests.md), [T-067](./ticket-067-benchmark-parallel.md), [T-068](./ticket-068-dhat-profiling.md)

## Files to Read Before Starting

- `src/algorithm/backward_pass.rs:250-315` - `execute_stage()` function
- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait
- `src/algorithm/coordinator.rs` - Implementation of trait methods
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Full architecture

---

## Context

### Background

This is the critical integration ticket that wires the new parallel zero-allocation path into production. All previous tickets built infrastructure—this ticket delivers the value.

### Current Code

```rust
// backward_pass.rs:266
let phase1 = processor.compute_cuts_parallel(stage_ctx)?;

// backward_pass.rs:285
let phase2 = processor.select_cuts_batch(phase1.cut_data, stage_ctx, fcf_graph)?;
```

### Target Code

```rust
// Use new parallel-then-sequential zero-allocation path
let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;

// Phase 2 works with slot indices instead of CutData
let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;
```

---

## Specification

### Changes Required

1. **Replace Phase 1 call**: `compute_cuts_parallel()` → `compute_cuts_parallel_into_slots()`
2. **Replace Phase 2 call**: `select_cuts_batch()` → `select_cuts_from_slots()`
3. **Update timing accumulation**: May need adjustment for new timing fields
4. **Add `select_cuts_from_slots` to trait**: If not already present

### select_cuts_from_slots Implementation

This method runs cut selection on already-updated pool slots:

```rust
fn select_cuts_from_slots(
    &mut self,
    slots: Vec<usize>,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase2Result, String> {
    let parent_id = stage_ctx.parent_id.ok_or_else(|| ...)?;
    
    let phase2_begin = Instant::now();
    
    let parent_fcf = fcf_graph.get_node_mut(parent_id)?;
    
    // Run domination/selection on the updated slots
    let batch_result = parent_fcf.data.finalize_cuts_batch(
        &slots,
        stage_ctx.enable_cut_selection,
    );
    
    let cut_selection_time = phase2_begin.elapsed();
    
    // Rest of Phase 2 (FCF state update, cut application prep)
    // Similar to existing select_cuts_batch but works with slots
    
    Ok(Phase2Result {
        batch_result,
        cut_selection_time,
        fcf_update_time: ...,
        cut_cloning_time: ...,
        cuts_to_apply: ...,
    })
}
```

---

## Acceptance Criteria

- [ ] `execute_stage()` uses `compute_cuts_parallel_into_slots()`
- [ ] `execute_stage()` uses `select_cuts_from_slots()`
- [ ] Timing accumulation works correctly
- [ ] All 549+ tests pass
- [ ] Golden tests pass (bit-for-bit identical)
- [ ] No `CutData::from_refs()` calls during training

---

## Implementation Guide

### Step 1: Add select_cuts_from_slots to trait

In `src/algorithm/processor.rs`:

```rust
/// Select cuts from already-updated pool slots.
///
/// Called after `compute_cuts_parallel_into_slots()` has populated
/// the pool slots. Runs domination evaluation and prepares Phase 3.
fn select_cuts_from_slots(
    &mut self,
    slots: Vec<usize>,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase2Result, String>;
```

### Step 2: Implement in coordinator

In `src/algorithm/coordinator.rs`:

```rust
fn select_cuts_from_slots(
    &mut self,
    slots: Vec<usize>,
    stage_ctx: &BackwardStageContext,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<Phase2Result, String> {
    let parent_id = stage_ctx.parent_id.ok_or_else(|| {
        format!(
            "No parent ID for stage {} (stage_idx {})",
            stage_ctx.stage_id, stage_ctx.stage_idx
        )
    })?;

    let phase2_begin = Instant::now();

    let parent_fcf_node = fcf_graph.get_node_mut(parent_id).ok_or_else(|| {
        format!("Could not find FCF for parent node {}", parent_id)
    })?;

    // Run batch finalization on slots
    let batch_result = parent_fcf_node.data.finalize_cuts_batch(
        &slots,
        stage_ctx.enable_cut_selection,
    );

    let cut_selection_time = phase2_begin.elapsed();

    // FCF state update timing
    let fcf_update_begin = Instant::now();
    // ... any FCF state updates needed
    let fcf_update_time = fcf_update_begin.elapsed();

    // Prepare cuts for Phase 3 application
    let cut_clone_begin = Instant::now();
    let cuts_to_apply: Vec<Arc<BendersCut>> = batch_result
        .new_cut_ids
        .iter()
        .chain(batch_result.returning_cut_ids.iter())
        .map(|&id| Arc::clone(&parent_fcf_node.data.cut_pool.pool[id]))
        .collect();
    let cut_cloning_time = cut_clone_begin.elapsed();

    Ok(Phase2Result {
        batch_result,
        cut_selection_time,
        fcf_update_time,
        cut_cloning_time,
        cuts_to_apply,
    })
}
```

### Step 3: Update backward_pass.rs

In `src/algorithm/backward_pass.rs`, modify `execute_stage()`:

```rust
fn execute_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &BackwardPassTimingAccumulator,
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<(), String> {
    // Phase 1: Zero-allocation parallel cut computation
    let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;

    // Accumulate Phase 1 timing
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_preprocessing,
        phase1.timing.model_preprocessing,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.solver,
        phase1.timing.solver,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_postprocessing,
        phase1.timing.model_postprocessing,
    );
    timing.increment_solver_calls(phase1.timing.solver_calls);

    // Phase 2: Cut selection on updated slots
    let phase2 = processor.select_cuts_from_slots(phase1.slots, stage_ctx, fcf_graph)?;

    // Accumulate Phase 2 timing
    BackwardPassTimingAccumulator::add_duration(
        &timing.cut_selection,
        phase2.cut_selection_time,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.fcf_state_update,
        phase2.fcf_update_time,
    );
    BackwardPassTimingAccumulator::add_duration(
        &timing.cut_cloning,
        phase2.cut_cloning_time,
    );

    // Update result counts
    result.cuts_added += phase2.batch_result.new_cut_ids.len();
    result.cuts_removed += phase2.batch_result.removing_cut_ids.len();
    result.cuts_returned += phase2.batch_result.returning_cut_ids.len();

    // Phase 3: Parallel cut application
    let handler_time = processor.apply_cuts_parallel(&phase2, stage_ctx)?;
    BackwardPassTimingAccumulator::add_duration(
        &timing.handler_application,
        handler_time,
    );

    Ok(())
}
```

### Step 4: Mark deprecated methods

Add deprecation warnings to old methods:

```rust
#[deprecated(since = "0.3.0", note = "Use compute_cuts_parallel_into_slots for zero-allocation path")]
fn compute_cuts_parallel(...) -> Result<Phase1Result, String>;

#[deprecated(since = "0.3.0", note = "Use select_cuts_from_slots for zero-allocation path")]
fn select_cuts_batch(...) -> Result<Phase2Result, String>;
```

---

## Testing Requirements

### Compile and Run Tests

```bash
RUST_TEST_THREADS=1 cargo test -j1
```

All 549+ tests must pass.

### Golden Tests

```bash
./scripts/golden-tests.sh verify
```

Must be bit-for-bit identical.

### Manual Verification

Add temporary logging to verify new path is used:

```rust
#[cfg(debug_assertions)]
eprintln!("Using parallel-into-slots path");
```

---

## Pitfalls to Avoid

- ⚠️ **Timing field names**: Ensure timing fields match between `Phase1SlotResult` and accumulator.
- ⚠️ **Backward compatibility**: Keep old methods for now (deprecated). Remove in Sprint 3.
- ⚠️ **Phase2Result structure**: May need to adjust `Phase2Result` to work with slots instead of `CutData`.

---

## Documentation Requirements

- [ ] Update module documentation for backward_pass.rs
- [ ] Add migration note to CHANGELOG.md
- [ ] Reference architecture document

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Critical integration point. Must verify correctness thoroughly.

---

## Definition of Done

- [ ] Training loop uses new path
- [ ] All tests pass (549+)
- [ ] Golden tests pass
- [ ] Deprecated methods marked
- [ ] Doc comments updated
- [ ] CHANGELOG updated
