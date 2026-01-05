# [T-016] Update backward pass tests

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-015](./ticket-015-remove-legacy-backward-timing.md)  
> **Blocks**: None (Epic 3 complete after this)

## Files to Read Before Starting

- `src/algorithm/backward_pass.rs` - Tests at bottom
- `src/algorithm/context.rs` - Tests at bottom (backward-related)
- `src/timing/backward.rs` - Tests for new types
- `tests/` - Integration tests

## Context

### Background

After removing legacy backward timing types (T-015), tests need updating to use new types and verify the integration works correctly.

### Test Categories

1. **Unit tests in backward_pass.rs**: Update to use `NewBackwardTiming`
2. **Unit tests in context.rs**: Remove tests for deleted `BackwardStageTiming`
3. **Unit tests in timing/backward.rs**: Already exist for new types (from Epic 1)
4. **Integration tests**: Verify backward pass timing collection

## Specification

### Tests to Update (backward_pass.rs)

Update tests to use `NewBackwardTiming` instead of `BackwardPassTimingAccumulator`:

- `test_timing_accumulator_default` → `test_backward_timing_new`
- `test_timing_accumulator_increment_solver_calls` → `test_add_solver_calls`
- `test_timing_accumulator_add_duration` → test phase1 accumulation
- `test_timing_snapshot` → `test_to_output`
- `test_timing_snapshot_is_independent` → keep similar pattern

### Tests to Remove (context.rs)

- `test_backward_stage_timing_default` (type removed)
- `test_backward_stage_timing_add` (type removed)

### Tests to Keep (context.rs)

- `test_backward_pass_result_new` (BackwardPassResult still exists)
- `test_backward_stage_context_is_first_stage` (context still exists)

## Acceptance Criteria

- [ ] All backward pass unit tests pass
- [ ] No tests reference removed types
- [ ] Coverage maintained for new timing types
- [ ] `cargo test` passes with no failures
- [ ] No duplicate tests between modules

## Implementation Guide

### Step 1: Update backward_pass.rs tests

Replace old tests with new type usage:

```rust
#[test]
fn test_backward_timing_default() {
    let timing = NewBackwardTiming::new();
    assert_eq!(timing.phase1.solver.get(), Duration::ZERO);
    assert_eq!(timing.get_solver_calls(), 0);
}

#[test]
fn test_backward_timing_accumulation() {
    let timing = NewBackwardTiming::new();
    
    // Simulate Phase 1 accumulation
    timing.phase1.solver.set(Duration::from_millis(100));
    timing.phase1.model_preprocessing.set(Duration::from_millis(50));
    
    assert_eq!(timing.phase1.solver.get(), Duration::from_millis(100));
    assert_eq!(timing.phase1.model_preprocessing.get(), Duration::from_millis(50));
}

#[test]
fn test_backward_timing_to_output() {
    let timing = NewBackwardTiming::new();
    timing.phase1.solver.set(Duration::from_millis(200));
    timing.phase2.cut_selection.set(Duration::from_millis(50));
    timing.add_solver_calls(25);
    
    let output = timing.to_output();
    
    assert_eq!(output.solver, Duration::from_millis(200));
    assert_eq!(output.cut_selection, Duration::from_millis(50));
    assert_eq!(output.solver_calls, 25);
}
```

### Step 2: Remove tests for deleted types from context.rs

Delete:
- `test_backward_stage_timing_default`
- `test_backward_stage_timing_add`

### Step 3: Verify no duplicates with timing/backward.rs

Check that tests in `timing/backward.rs` and `backward_pass.rs` don't overlap significantly.

### Step 4: Run all tests

```bash
cargo test -p powers-rs backward
cargo test -p powers-rs context
cargo test -p powers-rs timing
cargo test
```

### Pitfalls to Avoid

- ⚠️ Keep tests for `BackwardPassResult` in context.rs
- ⚠️ Don't duplicate tests already in `timing/backward.rs`
- ⚠️ Ensure test imports are updated

## Testing Requirements

### Final Verification

```bash
# All timing tests
cargo test -p powers-rs timing -- --nocapture

# Backward pass specific
cargo test -p powers-rs backward_pass -- --nocapture

# Full suite
cargo test

# Clippy
cargo clippy --all-targets
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Test updates follow directly from type changes
