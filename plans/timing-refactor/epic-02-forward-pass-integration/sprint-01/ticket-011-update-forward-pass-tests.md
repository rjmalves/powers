# [T-011] Update forward pass tests

> **Epic**: [Epic 2: Forward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-010](./ticket-010-remove-legacy-forward-timing.md)  
> **Blocks**: None (Epic 2 complete after this)

## Files to Read Before Starting

- `src/algorithm/forward_pass.rs` - Unit tests at bottom
- `src/algorithm/context.rs` - Unit tests at bottom (after removal)
- `tests/` - Integration tests that may use forward pass timing
- `src/timing/trajectory.rs` - TrajectoryTiming tests (ensure no duplication)

## Context

### Background

After removing duplicate `TrajectoryTiming` from `context.rs`, tests may need updating. This ticket ensures all tests compile, pass, and there's no duplicate test coverage.

### Current Test Locations

1. `src/algorithm/forward_pass.rs` - Tests for `aggregate_trajectory_timings`
2. `src/algorithm/context.rs` - Tests for `TrajectoryTiming` (to be removed)
3. `src/timing/trajectory.rs` - New `TrajectoryTiming` tests

## Specification

### Test Consolidation

1. **Remove duplicate tests**: Delete `TrajectoryTiming` tests from `context.rs` (covered by `timing/trajectory.rs`)
2. **Update import paths**: Ensure forward_pass tests use `timing::TrajectoryTiming`
3. **Verify coverage**: Ensure all functionality is still tested

### Tests to Keep (in timing/trajectory.rs)

- `test_new_is_zero`
- `test_cpu_time`
- `test_reset`
- `test_increment_solver_calls`
- `test_add_methods`

### Tests to Remove (from context.rs)

- `test_trajectory_timing_default` (duplicate)
- `test_trajectory_timing_increment_solver_calls` (duplicate)
- `test_trajectory_timing_add_durations` (duplicate)

### Tests to Keep (in forward_pass.rs)

- `test_step_timing_default` (for internal `StepTiming`)
- `test_aggregate_trajectory_timings_empty`
- `test_aggregate_trajectory_timings_single`
- `test_aggregate_trajectory_timings_average`

## Acceptance Criteria

- [ ] No duplicate tests for `TrajectoryTiming`
- [ ] All timing functionality tested in `timing/` module
- [ ] Forward pass aggregation tests pass
- [ ] `cargo test` reports no failures
- [ ] Test coverage for TrajectoryTiming is complete

## Implementation Guide

### Step 1: Identify duplicate tests

```bash
grep -n "test_trajectory_timing" src/algorithm/context.rs
grep -n "test_" src/timing/trajectory.rs
```

### Step 2: Remove duplicates from context.rs

Delete the following test functions from `src/algorithm/context.rs`:
- `test_trajectory_timing_default`
- `test_trajectory_timing_increment_solver_calls`
- `test_trajectory_timing_add_durations`

### Step 3: Update forward_pass.rs test imports

Ensure the test module uses:
```rust
use crate::timing::TrajectoryTiming;
```

### Step 4: Run all tests

```bash
cargo test -p powers-rs trajectory
cargo test -p powers-rs forward_pass
cargo test -p powers-rs context
cargo test
```

### Pitfalls to Avoid

- ⚠️ Don't remove backward pass tests from context.rs (BackwardStageTiming, BackwardPassResult)
- ⚠️ Keep `ForwardPassResult` tests in context.rs

## Testing Requirements

### Verification Commands

```bash
# Check no duplicate test names
cargo test 2>&1 | grep -i "trajectory_timing"

# Run all timing tests
cargo test -p powers-rs timing

# Run full suite
cargo test
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Mostly deletion and import updates
