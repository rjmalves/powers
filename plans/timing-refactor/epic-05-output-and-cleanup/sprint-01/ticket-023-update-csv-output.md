# [T-023] Update CSV training output

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 4 complete  
> **Blocks**: [T-025](./ticket-025-remove-old-metrics.md)

## Files to Read Before Starting

- `src/output/csv/training.rs` - Current CSV writer
- `src/timing/output.rs` - New timing output types
- `src/sddp/mod.rs` - Updated `IterationResult` struct
- `plans/timing-refactor/00-master-plan.md` - Output schema mapping

## Context

### Background

After Epic 4, `IterationResult` uses `IterationTimingOutput` instead of separate `ForwardPassTiming` and `BackwardPassTiming`. The CSV writer needs to be updated to read from the new timing structure.

### Current State

`src/output/csv/training.rs` reads timing like:
```rust
forward_saa_sampling_ms: result.forward_timing.saa_sampling_time.as_millis() as u64,
backward_preprocessing_ms: result.backward_timing.backward_preprocessing_time.as_millis() as u64,
```

### Target State

```rust
forward_saa_sampling_ms: result.timing.forward.saa_sampling.as_millis() as u64,
// backward_preprocessing_ms removed
backward_problem_update_ms: result.timing.backward.problem_update.as_millis() as u64,
```

## Specification

### Field Changes

| Old Access | New Access |
|------------|------------|
| `result.forward_timing.saa_sampling_time` | `result.timing.forward.saa_sampling` |
| `result.forward_timing.model_preprocessing_time` | `result.timing.forward.model_preprocessing` |
| `result.forward_timing.solver_time` | `result.timing.forward.solver` |
| `result.forward_timing.model_postprocessing_time` | `result.timing.forward.model_postprocessing` |
| `result.forward_timing.forward_postprocessing_time` | `result.timing.forward.postprocessing` |
| `result.forward_timing.total_time` | `result.timing.forward.total` |
| `result.backward_timing.backward_preprocessing_time` | **Remove** |
| `result.backward_timing.model_preprocessing_time` | `result.timing.backward.model_preprocessing` |
| `result.backward_timing.solver_time` | `result.timing.backward.solver` |
| `result.backward_timing.model_postprocessing_time` | `result.timing.backward.model_postprocessing` |
| `result.backward_timing.cut_selection_time` | `result.timing.backward.cut_selection` |
| `result.backward_timing.fcf_state_update_time` | `result.timing.backward.problem_update` |
| `result.backward_timing.cut_cloning_time` | **Remove** (merged) |
| `result.backward_timing.handler_application_time` | **Remove** (merged) |
| `result.backward_timing.total_time` | `result.timing.backward.total` |
| `result.iteration_time` | `result.timing.total` |

### Fields to Remove from Output

- `backward_preprocessing_ms` - was misnamed, no equivalent
- `backward_cut_cloning_ms` - merged into `problem_update`
- `backward_handler_application_ms` - merged into `problem_update`

### Fields to Rename

- `backward_fcf_state_update_ms` → `backward_problem_update_ms` (or keep old name for compatibility)

## Acceptance Criteria

- [ ] CSV writer compiles with new timing types
- [ ] All field accesses updated to new paths
- [ ] Removed fields no longer in output struct
- [ ] `cargo test -p powers-rs csv` passes
- [ ] Output CSV has correct column headers
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Update TrainingOutput struct

```rust
#[derive(serde::Serialize)]
struct TrainingOutput {
    // Convergence metrics
    iteration: usize,
    lower_bound: f64,
    policy_cost: f64,
    policy_std: f64,
    gap_percent: f64,

    // Forward pass timing (milliseconds)
    forward_saa_sampling_ms: u64,
    forward_model_preprocessing_ms: u64,
    forward_solver_ms: u64,
    forward_model_postprocessing_ms: u64,
    forward_postprocessing_ms: u64,
    forward_total_ms: u64,

    // Backward pass timing (milliseconds)
    // Note: backward_preprocessing_ms REMOVED
    backward_model_preprocessing_ms: u64,
    backward_solver_ms: u64,
    backward_model_postprocessing_ms: u64,
    backward_cut_selection_ms: u64,
    backward_problem_update_ms: u64,  // Was fcf_state_update + cut_cloning + handler_application
    backward_total_ms: u64,

    // Total
    iteration_total_ms: u64,
}
```

### Step 2: Update serialize calls

```rust
wtr.serialize(TrainingOutput {
    // ... convergence metrics ...

    forward_saa_sampling_ms: result.timing.forward.saa_sampling.as_millis() as u64,
    forward_model_preprocessing_ms: result.timing.forward.model_preprocessing.as_millis() as u64,
    forward_solver_ms: result.timing.forward.solver.as_millis() as u64,
    forward_model_postprocessing_ms: result.timing.forward.model_postprocessing.as_millis() as u64,
    forward_postprocessing_ms: result.timing.forward.postprocessing.as_millis() as u64,
    forward_total_ms: result.timing.forward.total.as_millis() as u64,

    backward_model_preprocessing_ms: result.timing.backward.model_preprocessing.as_millis() as u64,
    backward_solver_ms: result.timing.backward.solver.as_millis() as u64,
    backward_model_postprocessing_ms: result.timing.backward.model_postprocessing.as_millis() as u64,
    backward_cut_selection_ms: result.timing.backward.cut_selection.as_millis() as u64,
    backward_problem_update_ms: result.timing.backward.problem_update.as_millis() as u64,
    backward_total_ms: result.timing.backward.total.as_millis() as u64,

    iteration_total_ms: result.timing.total.as_millis() as u64,
})?;
```

### Pitfalls to Avoid

- ⚠️ Column name change (`backward_problem_update_ms`) may break downstream tools
- ⚠️ Consider keeping old column names with new data for compatibility
- ⚠️ Test with actual training run to verify output

## Testing Requirements

### Unit Tests

- Test that TrainingOutput serializes correctly

### Integration Tests

```bash
cargo test -p powers-rs training
cargo test -p powers-rs output
```

### Validation Tests

- Run training with CSV output enabled
- Verify columns match expected schema
- Verify timing values are reasonable

## Documentation Requirements

- [ ] Update docstring on `write_training_results`
- [ ] Document schema changes in function comments

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward field renaming
