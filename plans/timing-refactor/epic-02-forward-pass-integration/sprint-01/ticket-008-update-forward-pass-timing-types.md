# [T-008] Update forward_pass.rs timing types

> **Epic**: [Epic 2: Forward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 1 complete  
> **Blocks**: [T-009](./ticket-009-update-sddp-handler-forward-methods.md)

## Files to Read Before Starting

- `src/timing/trajectory.rs` - New `TrajectoryTiming` definition (the one to use)
- `src/algorithm/context.rs` - Current `TrajectoryTiming` definition (to be removed)
- `src/algorithm/forward_pass.rs` - Current forward pass implementation
- `src/timing/mod.rs` - Module exports

## Context

### Background

The forward pass currently imports `TrajectoryTiming` from `algorithm/context.rs`. Epic 1 created a new `TrajectoryTiming` in `src/timing/trajectory.rs` with identical API. This ticket updates `forward_pass.rs` to use the timing module's version.

### Current State

```rust
// src/algorithm/forward_pass.rs line 31-33
use crate::algorithm::context::{
    ForwardPassContext, ForwardPassResult, TrajectoryTiming,
};
```

The `aggregate_trajectory_timings` function at the bottom of `forward_pass.rs` (lines 235-261) uses the old `ForwardTiming` from `timing::metrics`.

## Specification

### Changes Required

1. **Update import**: Change `TrajectoryTiming` import from `algorithm::context` to `timing::TrajectoryTiming`
2. **Update aggregation function**: Modify `aggregate_trajectory_timings` to work with new timing types

### Expected Import After Change

```rust
use crate::algorithm::context::{ForwardPassContext, ForwardPassResult};
use crate::timing::TrajectoryTiming;
```

### Aggregation Function Update

The `aggregate_trajectory_timings` function should continue to work since both old and new `TrajectoryTiming` have:
- `model_preprocessing: Cell<Duration>`
- `solver: Cell<Duration>`  
- `model_postprocessing: Cell<Duration>`
- `solver_calls: Cell<usize>`

The target `ForwardTiming` type should use `crate::timing::ForwardTiming` (the old one is still needed for now until Epic 4).

## Acceptance Criteria

- [ ] `forward_pass.rs` imports `TrajectoryTiming` from `crate::timing`
- [ ] No import of `TrajectoryTiming` from `algorithm::context` in `forward_pass.rs`
- [ ] `forward_pass::execute()` still works with passed `&TrajectoryTiming`
- [ ] `aggregate_trajectory_timings` compiles and works
- [ ] All existing tests in `forward_pass.rs` pass
- [ ] `cargo test --lib` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Update the import

In `src/algorithm/forward_pass.rs`, change line 31-33 from:

```rust
use crate::algorithm::context::{
    ForwardPassContext, ForwardPassResult, TrajectoryTiming,
};
```

To:

```rust
use crate::algorithm::context::{ForwardPassContext, ForwardPassResult};
use crate::timing::TrajectoryTiming;
```

### Step 2: Verify API compatibility

Check that the new `TrajectoryTiming` has the same methods used in `forward_pass.rs`:
- `model_preprocessing` field (Cell<Duration>) - used on line 109
- `add_solver_time()` - used on line 169
- `add_model_postprocessing()` - used on line 170
- `increment_solver_calls()` - used on line 171
- `get_solver_calls()` - used on line 90

### Step 3: Run tests

```bash
cargo test -p powers-rs forward_pass -- --nocapture
cargo clippy --all-targets
```

### Pitfalls to Avoid

- ⚠️ Don't remove `TrajectoryTiming` from `algorithm/context.rs` yet - that's T-010
- ⚠️ The aggregation function target uses the OLD `ForwardTiming` from `timing::metrics` - keep that for now

## Testing Requirements

### Unit Tests

Existing tests in `forward_pass.rs` should pass without modification:
- `test_step_timing_default`
- `test_aggregate_trajectory_timings_empty`
- `test_aggregate_trajectory_timings_single`
- `test_aggregate_trajectory_timings_average`

### Integration Tests

Run the full test suite to ensure no regressions:
```bash
cargo test
```

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Simple import change with compatible API
