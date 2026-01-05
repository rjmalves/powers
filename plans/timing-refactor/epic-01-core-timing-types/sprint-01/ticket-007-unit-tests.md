# T-007: Add Comprehensive Unit Tests

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-005](./ticket-005-aggregation-methods.md), [T-006](./ticket-006-output-types.md)  
> **Blocks**: None (enables Epic 2)

## Files to Read Before Starting

- All files in `src/timing/` created by T-001 through T-006
- `src/timing/guard.rs` - Existing TimingGuard tests (pattern to follow)

## Context

### Background

This ticket consolidates and expands test coverage for the new timing infrastructure. While each previous ticket added basic tests, this ticket ensures comprehensive coverage including edge cases, integration scenarios, and TimingGuard interaction.

### Current State

Basic tests exist in each timing module. Need to add:
- Edge case tests
- TimingGuard integration tests
- Cross-module integration tests
- Documentation tests (doctests)

## Specification

### Test Categories

1. **Edge Cases**: Zero values, single trajectory, overflow protection
2. **TimingGuard Integration**: Verify guards work with new types
3. **Reset Behavior**: Ensure reset works across nested structures
4. **Conversion Round-trips**: Verify to_output produces expected values
5. **Thread Safety**: Verify Cell usage is correct (not truly thread-safe, but single-thread mutation works)

### Tests to Add

#### Edge Case Tests (`src/timing/forward.rs`)

```rust
#[test]
fn test_single_trajectory() {
    let ft = ForwardTiming::new(1);
    ft.parallel.trajectories[0].solver.set(Duration::from_millis(100));
    ft.parallel.wall.set(Duration::from_millis(120));
    ft.parallel.compute_aggregates();
    
    assert_eq!(ft.parallel.solver_avg.get(), Duration::from_millis(100));
    assert_eq!(ft.parallel.solver_max.get(), Duration::from_millis(100));
    assert_eq!(ft.parallel.overhead.get(), Duration::from_millis(20));
}

#[test]
fn test_all_zero_trajectories() {
    let ft = ForwardTiming::new(3);
    ft.parallel.wall.set(Duration::from_millis(10));
    ft.parallel.compute_aggregates();
    
    assert_eq!(ft.parallel.cpu_total.get(), Duration::ZERO);
    assert_eq!(ft.parallel.solver_avg.get(), Duration::ZERO);
    assert_eq!(ft.parallel.overhead.get(), Duration::from_millis(10));
}

#[test]
fn test_overhead_saturates_to_zero() {
    // CPU time > wall time (parallel speedup)
    let ft = ForwardTiming::new(2);
    ft.parallel.trajectories[0].solver.set(Duration::from_millis(100));
    ft.parallel.trajectories[1].solver.set(Duration::from_millis(100));
    ft.parallel.wall.set(Duration::from_millis(50)); // Less than CPU total
    ft.parallel.compute_aggregates();
    
    assert_eq!(ft.parallel.overhead.get(), Duration::ZERO);
}
```

#### TimingGuard Integration Tests

```rust
// In src/timing/guard.rs or a separate integration test file
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::timing::{ForwardTiming, TrajectoryTiming, IterationTiming};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_guard_with_trajectory_timing() {
        let timing = TrajectoryTiming::new();
        
        {
            let _guard = TimingGuard::new(&timing.model_preprocessing);
            thread::sleep(Duration::from_millis(5));
        }
        
        let elapsed = timing.model_preprocessing.get();
        assert!(elapsed >= Duration::from_millis(5));
        assert!(elapsed < Duration::from_millis(50)); // Reasonable upper bound
    }

    #[test]
    fn test_guard_accumulates() {
        let timing = TrajectoryTiming::new();
        
        for _ in 0..3 {
            let _guard = TimingGuard::new(&timing.solver);
            thread::sleep(Duration::from_millis(2));
        }
        
        let elapsed = timing.solver.get();
        assert!(elapsed >= Duration::from_millis(6));
    }

    #[test]
    fn test_guard_with_forward_timing() {
        let timing = ForwardTiming::new(2);
        
        {
            let _guard = TimingGuard::new(&timing.preprocessing.saa_sampling);
            thread::sleep(Duration::from_millis(5));
        }
        
        assert!(timing.preprocessing.saa_sampling.get() >= Duration::from_millis(5));
    }

    #[test]
    fn test_guard_with_iteration_timing() {
        let timing = IterationTiming::new(2);
        
        {
            let _guard = TimingGuard::new(&timing.model_allocation);
            thread::sleep(Duration::from_millis(5));
        }
        
        assert!(timing.model_allocation.get() >= Duration::from_millis(5));
    }
}
```

#### Reset Behavior Tests

```rust
#[test]
fn test_nested_reset() {
    let it = IterationTiming::new(3);
    
    // Set various values
    it.model_allocation.set(Duration::from_millis(10));
    it.forward.preprocessing.saa_sampling.set(Duration::from_millis(20));
    it.forward.parallel.wall.set(Duration::from_millis(100));
    it.forward.parallel.trajectories[0].solver.set(Duration::from_millis(50));
    it.forward.parallel.trajectories[1].solver.set(Duration::from_millis(60));
    it.backward.phase1.solver.set(Duration::from_millis(30));
    it.backward.solver_calls.set(10);
    
    // Reset
    it.reset();
    
    // Verify all zeroed
    assert_eq!(it.model_allocation.get(), Duration::ZERO);
    assert_eq!(it.forward.preprocessing.saa_sampling.get(), Duration::ZERO);
    assert_eq!(it.forward.parallel.wall.get(), Duration::ZERO);
    assert_eq!(it.forward.parallel.trajectories[0].solver.get(), Duration::ZERO);
    assert_eq!(it.forward.parallel.trajectories[1].solver.get(), Duration::ZERO);
    assert_eq!(it.backward.phase1.solver.get(), Duration::ZERO);
    assert_eq!(it.backward.solver_calls.get(), 0);
}
```

#### Full Workflow Integration Test

```rust
#[test]
fn test_full_iteration_workflow() {
    let timing = IterationTiming::new(2);
    
    // Simulate model allocation
    timing.model_allocation.set(Duration::from_millis(10));
    
    // Simulate SAA sampling
    timing.forward.preprocessing.saa_sampling.set(Duration::from_millis(5));
    
    // Simulate parallel forward pass
    timing.forward.parallel.wall.set(Duration::from_millis(100));
    timing.forward.parallel.trajectories[0].model_preprocessing.set(Duration::from_millis(10));
    timing.forward.parallel.trajectories[0].solver.set(Duration::from_millis(80));
    timing.forward.parallel.trajectories[0].model_postprocessing.set(Duration::from_millis(5));
    timing.forward.parallel.trajectories[0].solver_calls.set(5);
    
    timing.forward.parallel.trajectories[1].model_preprocessing.set(Duration::from_millis(12));
    timing.forward.parallel.trajectories[1].solver.set(Duration::from_millis(90));
    timing.forward.parallel.trajectories[1].model_postprocessing.set(Duration::from_millis(6));
    timing.forward.parallel.trajectories[1].solver_calls.set(5);
    
    // Compute aggregates
    timing.forward.parallel.compute_aggregates();
    
    // Forward postprocessing
    timing.forward.postprocessing.detail_capturing.set(Duration::from_millis(2));
    timing.forward.compute_total();
    
    // Backward pass
    timing.backward.phase1.solver.set(Duration::from_millis(200));
    timing.backward.phase2.cut_selection.set(Duration::from_millis(10));
    timing.backward.phase3.problem_update.set(Duration::from_millis(5));
    timing.backward.solver_calls.set(50);
    timing.backward.total.set(Duration::from_millis(215));
    
    // Model cleanup
    timing.model_cleanup.set(Duration::from_millis(3));
    
    // Compute total
    timing.compute_total();
    
    // Convert to output
    let output = timing.to_output();
    
    // Verify output
    assert_eq!(output.model_allocation, Duration::from_millis(10));
    assert_eq!(output.forward.saa_sampling, Duration::from_millis(5));
    assert_eq!(output.forward.solver, Duration::from_millis(85)); // avg(80, 90)
    assert_eq!(output.forward.solver_max, Duration::from_millis(90));
    assert_eq!(output.forward.solver_calls, 10);
    assert_eq!(output.backward.solver, Duration::from_millis(200));
    assert_eq!(output.backward.solver_calls, 50);
    assert_eq!(output.solver_calls, 60); // 10 + 50
}
```

## Acceptance Criteria

- [ ] Edge case tests for zero values, single trajectory
- [ ] TimingGuard integration tests with new types
- [ ] Reset behavior tests for nested structures
- [ ] Full workflow integration test
- [ ] All tests pass with `cargo test`
- [ ] Test coverage for all public methods

## Implementation Guide

### Suggested Approach

1. Add edge case tests to existing test modules
2. Add TimingGuard integration tests
3. Add reset behavior tests
4. Add full workflow integration test
5. Run `cargo test` to verify all pass

### Key Files to Modify

- `src/timing/trajectory.rs`: Add edge case tests
- `src/timing/forward.rs`: Add edge case and integration tests
- `src/timing/backward.rs`: Add tests
- `src/timing/iteration.rs`: Add workflow integration test
- `src/timing/guard.rs`: Add integration tests with new types

### Pitfalls to Avoid

- ⚠️ Sleep-based tests can be flaky - use reasonable bounds
- ⚠️ Don't test timing accuracy too precisely (system-dependent)

## Testing Requirements

This ticket IS the testing - ensure all tests pass!

```bash
cargo test timing::
```

## Documentation Requirements

- [ ] Test names clearly describe what they test
- [ ] Comments in complex tests explaining the scenario

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Tests are straightforward, building on existing patterns

## Definition of Done

- [ ] All new tests written
- [ ] All tests passing
- [ ] Test coverage verified (manual review of public API)
- [ ] `cargo clippy` clean on test code
