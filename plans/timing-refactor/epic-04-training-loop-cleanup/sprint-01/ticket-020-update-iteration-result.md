# [T-020] Update IterationResult struct

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-018](./ticket-018-replace-forward-timing.md), [T-019](./ticket-019-replace-backward-timing.md)  
> **Blocks**: [T-021](./ticket-021-remove-legacy-timing.md)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Current `IterationResult` struct definition (line ~177)
- `src/timing/output.rs` - `IterationTimingOutput`, `ForwardTimingOutput`, `BackwardTimingOutput`
- `src/output/csv/training.rs` - CSV writer that reads from `IterationResult`

## Context

### Background

`IterationResult` currently uses the legacy `ForwardPassTiming` and `BackwardPassTiming` structs. This ticket updates it to use the new output types from the timing module, which have a cleaner structure and match the new timing hierarchy.

### Current State

```rust
// sddp/mod.rs
pub struct IterationResult {
    pub iteration: usize,
    pub lower_bound: f64,
    pub forward_costs: Vec<f64>,
    pub iteration_time: Duration,
    pub forward_timing: ForwardPassTiming,    // Legacy
    pub backward_timing: BackwardPassTiming,  // Legacy
    pub num_solver_calls: usize,
    pub num_cuts_added: usize,
    pub num_cuts_removed: usize,
    pub num_cuts_returned: usize,
    pub num_active_cuts: usize,
}
```

### Target State

```rust
pub struct IterationResult {
    pub iteration: usize,
    pub lower_bound: f64,
    pub forward_costs: Vec<f64>,
    pub timing: IterationTimingOutput,  // New unified timing
    pub num_cuts_added: usize,
    pub num_cuts_removed: usize,
    pub num_cuts_returned: usize,
    pub num_active_cuts: usize,
}
```

## Specification

### Changes Required

1. **Import new types**: Add `use crate::timing::IterationTimingOutput;`
2. **Update struct**: Replace `forward_timing` + `backward_timing` + `iteration_time` + `num_solver_calls` with single `timing` field
3. **Update construction**: Where `IterationResult` is created, use `timing.to_output()`
4. **Update access patterns**: Update any code that reads from old timing fields

### Field Migration

| Old Field | New Field |
|-----------|-----------|
| `iteration_time` | `timing.total` |
| `forward_timing.saa_sampling_time` | `timing.forward.saa_sampling` |
| `forward_timing.model_preprocessing_time` | `timing.forward.model_preprocessing` |
| `forward_timing.solver_time` | `timing.forward.solver` |
| `forward_timing.model_postprocessing_time` | `timing.forward.model_postprocessing` |
| `forward_timing.forward_postprocessing_time` | `timing.forward.postprocessing` |
| `forward_timing.total_time` | `timing.forward.total` |
| `backward_timing.backward_preprocessing_time` | **Removed** |
| `backward_timing.model_preprocessing_time` | `timing.backward.model_preprocessing` |
| `backward_timing.solver_time` | `timing.backward.solver` |
| `backward_timing.model_postprocessing_time` | `timing.backward.model_postprocessing` |
| `backward_timing.cut_selection_time` | `timing.backward.cut_selection` |
| `backward_timing.fcf_state_update_time` | `timing.backward.problem_update` (combined) |
| `backward_timing.cut_cloning_time` | `timing.backward.problem_update` (combined) |
| `backward_timing.handler_application_time` | `timing.backward.problem_update` (combined) |
| `backward_timing.total_time` | `timing.backward.total` |
| `num_solver_calls` | `timing.solver_calls` |

### New Fields Available

| New Field | Purpose |
|-----------|---------|
| `timing.model_allocation` | Time for model creation |
| `timing.model_cleanup` | Time for model destruction |
| `timing.forward.parallel_wall` | Wall time of parallel section |
| `timing.forward.parallel_overhead` | Scheduling overhead |
| `timing.forward.solver_max` | Max solver time (load balance) |

## Acceptance Criteria

- [ ] `IterationResult` uses `IterationTimingOutput` instead of separate timing fields
- [ ] All construction sites updated to use `timing.to_output()`
- [ ] No compilation errors in `sddp/` module
- [ ] Output writers compile (may have errors, fixed in Epic 5)
- [ ] `cargo build` succeeds
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Update struct definition

```rust
// sddp/mod.rs
use crate::timing::IterationTimingOutput;

pub struct IterationResult {
    pub iteration: usize,
    pub lower_bound: f64,
    pub forward_costs: Vec<f64>,
    pub timing: IterationTimingOutput,  // New
    pub num_cuts_added: usize,
    pub num_cuts_removed: usize,
    pub num_cuts_returned: usize,
    pub num_active_cuts: usize,
}
```

### Step 2: Update construction

Find where `IterationResult` is created and update:

```rust
// OLD
IterationResult {
    iteration,
    lower_bound,
    forward_costs,
    iteration_time,
    forward_timing,
    backward_timing,
    num_solver_calls: forward_solver_calls + backward_solver_calls,
    ...
}

// NEW
IterationResult {
    iteration,
    lower_bound,
    forward_costs,
    timing: timing.to_output(),  // From NewIterationTiming
    ...
}
```

### Step 3: Temporarily break output writers

The output writers in `src/output/csv/training.rs` and `src/output/parquet/` will fail to compile. Add `#[allow(dead_code)]` or comment out temporarily. Epic 5 will fix these.

### Pitfalls to Avoid

- ⚠️ Don't try to fix output writers in this ticket - that's Epic 5
- ⚠️ `backward_preprocessing_time` is removed - no equivalent in new schema
- ⚠️ Phase 3 fields are combined into `problem_update`

## Testing Requirements

### Unit Tests

- Test `IterationResult` construction with new timing

### Integration Tests

- Compilation test only at this stage
- Full test suite may fail until Epic 5

```bash
cargo build -p powers-rs
```

## Documentation Requirements

- [ ] Update doc comments on `IterationResult` struct

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward struct update, output writers fixed later
