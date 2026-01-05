# T-006: Create Output Conversion Types

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-004](./ticket-004-iteration-timing.md)  
> **Blocks**: [T-007](./ticket-007-unit-tests.md)

## Files to Read Before Starting

- `src/timing/iteration.rs` - IterationTiming (from T-004)
- `src/timing/forward.rs` - ForwardTiming (from T-002)
- `src/timing/backward.rs` - BackwardTiming (from T-003)
- `src/sddp/mod.rs:37-57` - Current ForwardPassTiming, BackwardPassTiming
- `src/output/csv/training.rs:88-115` - Current output schema
- `plans/timing-refactor/00-master-plan.md` - Output schema mapping

## Context

### Background

The accumulator structs use `Cell<Duration>` for interior mutability during execution. Output structs need plain `Duration` for serialization and storage in results. We define output types and `to_output()` conversion methods.

### Current State

`ForwardPassTiming` and `BackwardPassTiming` in `sddp/mod.rs` are the current output types. We're creating new versions with the updated schema.

## Specification

### File: `src/timing/output.rs`

```rust
//! Output types for timing data.
//!
//! These types use plain `Duration` (not `Cell<Duration>`) for serialization
//! and storage in results. Created via `to_output()` methods on accumulator types.

use std::time::Duration;

/// Forward pass timing for output/results.
///
/// Contains aggregated statistics from parallel execution.
#[derive(Debug, Clone, Default)]
pub struct ForwardTimingOutput {
    /// SAA sampling time (preprocessing).
    pub saa_sampling: Duration,

    /// Average model preprocessing time per trajectory.
    pub model_preprocessing: Duration,

    /// Average solver time per trajectory.
    pub solver: Duration,

    /// Average model postprocessing time per trajectory.
    pub model_postprocessing: Duration,

    /// Postprocessing time (detail capturing).
    pub postprocessing: Duration,

    /// Total forward pass time.
    pub total: Duration,

    /// Wall-clock time for parallel section.
    pub parallel_wall: Duration,

    /// Parallel overhead (wall - cpu_total).
    pub parallel_overhead: Duration,

    /// Maximum solver time across trajectories.
    pub solver_max: Duration,

    /// Total solver calls across all trajectories.
    pub solver_calls: usize,
}

/// Backward pass timing for output/results.
#[derive(Debug, Clone, Default)]
pub struct BackwardTimingOutput {
    /// Phase 1 model preprocessing time.
    pub model_preprocessing: Duration,

    /// Phase 1 solver time.
    pub solver: Duration,

    /// Phase 1 model postprocessing time.
    pub model_postprocessing: Duration,

    /// Cut selection time (Phase 2).
    pub cut_selection: Duration,

    /// Problem update time (Phase 3).
    pub problem_update: Duration,

    /// Total backward pass time.
    pub total: Duration,

    /// Total solver calls.
    pub solver_calls: usize,
}

/// Iteration timing for output/results.
#[derive(Debug, Clone, Default)]
pub struct IterationTimingOutput {
    /// Model allocation time.
    pub model_allocation: Duration,

    /// Forward pass timing.
    pub forward: ForwardTimingOutput,

    /// Backward pass timing.
    pub backward: BackwardTimingOutput,

    /// Model cleanup time.
    pub model_cleanup: Duration,

    /// Total iteration time.
    pub total: Duration,

    /// Total solver calls (forward + backward).
    pub solver_calls: usize,
}
```

### Add `to_output()` methods to accumulators

In `src/timing/forward.rs`:

```rust
use super::output::ForwardTimingOutput;

impl ForwardTiming {
    /// Convert to output format.
    ///
    /// # Panics
    ///
    /// Panics if `compute_aggregates()` hasn't been called on the parallel section.
    pub fn to_output(&self) -> ForwardTimingOutput {
        ForwardTimingOutput {
            saa_sampling: self.preprocessing.saa_sampling.get(),
            model_preprocessing: self.parallel.model_preprocessing_avg.get(),
            solver: self.parallel.solver_avg.get(),
            model_postprocessing: self.parallel.model_postprocessing_avg.get(),
            postprocessing: self.postprocessing.detail_capturing.get(),
            total: self.total.get(),
            parallel_wall: self.parallel.wall.get(),
            parallel_overhead: self.parallel.overhead.get(),
            solver_max: self.parallel.solver_max.get(),
            solver_calls: self.parallel.total_solver_calls(),
        }
    }
}
```

In `src/timing/backward.rs`:

```rust
use super::output::BackwardTimingOutput;

impl BackwardTiming {
    /// Convert to output format.
    pub fn to_output(&self) -> BackwardTimingOutput {
        BackwardTimingOutput {
            model_preprocessing: self.phase1.model_preprocessing.get(),
            solver: self.phase1.solver.get(),
            model_postprocessing: self.phase1.model_postprocessing.get(),
            cut_selection: self.phase2.cut_selection.get(),
            problem_update: self.phase3.problem_update.get(),
            total: self.total.get(),
            solver_calls: self.solver_calls.get(),
        }
    }
}
```

In `src/timing/iteration.rs`:

```rust
use super::output::IterationTimingOutput;

impl IterationTiming {
    /// Convert to output format.
    ///
    /// Call `compute_total()` and `forward.parallel.compute_aggregates()` first.
    pub fn to_output(&self) -> IterationTimingOutput {
        IterationTimingOutput {
            model_allocation: self.model_allocation.get(),
            forward: self.forward.to_output(),
            backward: self.backward.to_output(),
            model_cleanup: self.model_cleanup.get(),
            total: self.total.get(),
            solver_calls: self.total_solver_calls(),
        }
    }
}
```

## Acceptance Criteria

- [ ] `ForwardTimingOutput` struct with all required fields
- [ ] `BackwardTimingOutput` struct with all required fields
- [ ] `IterationTimingOutput` struct with nested output types
- [ ] `ForwardTiming::to_output()` converts correctly
- [ ] `BackwardTiming::to_output()` converts correctly
- [ ] `IterationTiming::to_output()` converts correctly
- [ ] All output types derive `Debug, Clone, Default`
- [ ] Doc comments on all types
- [ ] Exported from `src/timing/mod.rs`

## Implementation Guide

### Suggested Approach

1. Create `src/timing/output.rs` with output types
2. Add `to_output()` methods to each accumulator type
3. Update `mod.rs` to export output types
4. Verify with `cargo build`

### Key Files to Modify

- `src/timing/output.rs`: Create new file
- `src/timing/forward.rs`: Add `to_output()` method
- `src/timing/backward.rs`: Add `to_output()` method
- `src/timing/iteration.rs`: Add `to_output()` method
- `src/timing/mod.rs`: Add module and exports

### Pitfalls to Avoid

- ⚠️ Don't forget to call `.get()` on all `Cell` values
- ⚠️ Ensure `compute_aggregates()` is called before `to_output()`

## Testing Requirements

### Unit Tests

```rust
// In src/timing/output.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_timing_output_default() {
        let out = ForwardTimingOutput::default();
        assert_eq!(out.total, Duration::ZERO);
        assert_eq!(out.solver_calls, 0);
    }

    #[test]
    fn test_backward_timing_output_default() {
        let out = BackwardTimingOutput::default();
        assert_eq!(out.total, Duration::ZERO);
    }
}

// In src/timing/forward.rs (add to existing tests)
#[test]
fn test_to_output() {
    let ft = ForwardTiming::new(2);
    ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
    ft.parallel.wall.set(Duration::from_millis(100));
    ft.parallel.trajectories[0].solver.set(Duration::from_millis(40));
    ft.parallel.trajectories[1].solver.set(Duration::from_millis(60));
    ft.postprocessing.detail_capturing.set(Duration::from_millis(5));
    ft.total.set(Duration::from_millis(115));

    ft.parallel.compute_aggregates();
    let out = ft.to_output();

    assert_eq!(out.saa_sampling, Duration::from_millis(10));
    assert_eq!(out.solver, Duration::from_millis(50)); // avg
    assert_eq!(out.solver_max, Duration::from_millis(60));
    assert_eq!(out.total, Duration::from_millis(115));
}

// In src/timing/iteration.rs (add to existing tests)
#[test]
fn test_to_output() {
    let it = IterationTiming::new(1);
    it.model_allocation.set(Duration::from_millis(10));
    it.forward.total.set(Duration::from_millis(100));
    it.backward.total.set(Duration::from_millis(50));
    it.backward.solver_calls.set(5);
    it.model_cleanup.set(Duration::from_millis(5));
    it.total.set(Duration::from_millis(165));

    it.forward.parallel.compute_aggregates();
    let out = it.to_output();

    assert_eq!(out.model_allocation, Duration::from_millis(10));
    assert_eq!(out.total, Duration::from_millis(165));
    assert_eq!(out.backward.solver_calls, 5);
}
```

## Documentation Requirements

- [ ] Module-level doc explaining purpose of output types
- [ ] Doc comments on each output struct
- [ ] Doc comments on `to_output()` methods explaining preconditions

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward data structure mapping

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Output types exported
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
