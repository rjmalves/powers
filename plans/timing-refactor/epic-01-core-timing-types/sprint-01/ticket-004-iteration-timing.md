# T-004: Create IterationTiming Struct

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-002](./ticket-002-forward-timing.md), [T-003](./ticket-003-backward-timing.md)  
> **Blocks**: [T-006](./ticket-006-output-types.md)

## Files to Read Before Starting

- `src/timing/forward.rs` - ForwardTiming (from T-002)
- `src/timing/backward.rs` - BackwardTiming (from T-003)
- `src/timing/metrics.rs:136-180` - Existing IterationTiming (reference)
- `plans/timing-refactor/00-master-plan.md` - Schema definitions

## Context

### Background

`IterationTiming` is the top-level timing struct created at the start of each SDDP iteration. It owns `ForwardTiming` and `BackwardTiming`, plus tracks model allocation/cleanup time. At iteration end, it's converted to an output format.

### Current State

An `IterationTiming` exists in `metrics.rs` but with the old flat structure. We're creating a new version with nested types.

## Specification

### File: `src/timing/iteration.rs`

```rust
use std::cell::Cell;
use std::time::Duration;

use super::{BackwardTiming, ForwardTiming};

/// Complete timing for one SDDP training iteration.
///
/// This is the top-level timing struct - created at iteration start,
/// passed to sub-operations, and converted to output format at iteration end.
///
/// # Usage
///
/// ```ignore
/// let timing = IterationTiming::new(num_forward_passes);
///
/// // Model allocation
/// {
///     let _guard = TimingGuard::new(&timing.model_allocation);
///     create_models();
/// }
///
/// // Forward pass
/// forward_pass(&timing.forward);
///
/// // Backward pass
/// backward_pass(&timing.backward);
///
/// // Convert to output
/// let output = timing.to_output();
/// ```
pub struct IterationTiming {
    /// Time to create solver Models from cached Problems.
    pub model_allocation: Cell<Duration>,

    /// Forward pass timing.
    pub forward: ForwardTiming,

    /// Backward pass timing.
    pub backward: BackwardTiming,

    /// Time to cleanup solver Models at iteration end.
    pub model_cleanup: Cell<Duration>,

    /// Total iteration wall-clock time.
    pub total: Cell<Duration>,
}

impl IterationTiming {
    /// Create new IterationTiming with preallocated trajectory Vec.
    ///
    /// # Arguments
    ///
    /// * `num_forward_passes` - Number of forward passes (trajectories) per iteration
    pub fn new(num_forward_passes: usize) -> Self {
        Self {
            model_allocation: Cell::new(Duration::ZERO),
            forward: ForwardTiming::new(num_forward_passes),
            backward: BackwardTiming::new(),
            model_cleanup: Cell::new(Duration::ZERO),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total iteration time from components.
    ///
    /// Should be called at the end of an iteration before converting to output.
    pub fn compute_total(&self) {
        let total = self.model_allocation.get()
            + self.forward.total.get()
            + self.backward.total.get()
            + self.model_cleanup.get();
        self.total.set(total);
    }

    /// Reset all timing values for reuse.
    ///
    /// Call this at the start of each iteration if reusing the same struct.
    pub fn reset(&self) {
        self.model_allocation.set(Duration::ZERO);
        self.forward.reset();
        self.backward.reset();
        self.model_cleanup.set(Duration::ZERO);
        self.total.set(Duration::ZERO);
    }

    /// Get total solver calls (forward + backward).
    pub fn total_solver_calls(&self) -> usize {
        self.forward.parallel.total_solver_calls() + self.backward.get_solver_calls()
    }
}
```

### TrainingTiming (Full Training Run)

Also add to `src/timing/iteration.rs`:

```rust
/// Timing for the complete training run.
///
/// Contains preprocessing/postprocessing times and per-iteration timing.
pub struct TrainingTiming {
    /// Preprocessing before iterations begin (graph construction, warmup).
    pub preprocessing: Cell<Duration>,

    /// Per-iteration timing (preallocated, length = num_iterations).
    pub iterations: Vec<IterationTiming>,

    /// Postprocessing after iterations complete.
    pub postprocessing: Cell<Duration>,

    /// Total training time.
    pub total: Cell<Duration>,
}

impl TrainingTiming {
    /// Create new TrainingTiming with preallocated iteration Vec.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Number of forward passes per iteration
    pub fn new(num_iterations: usize, num_forward_passes: usize) -> Self {
        Self {
            preprocessing: Cell::new(Duration::ZERO),
            iterations: (0..num_iterations)
                .map(|_| IterationTiming::new(num_forward_passes))
                .collect(),
            postprocessing: Cell::new(Duration::ZERO),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total training time.
    pub fn compute_total(&self) {
        let iter_total: Duration = self.iterations.iter()
            .map(|it| it.total.get())
            .sum();
        let total = self.preprocessing.get() + iter_total + self.postprocessing.get();
        self.total.set(total);
    }
}
```

## Acceptance Criteria

- [ ] `IterationTiming` struct with all fields as specified
- [ ] `IterationTiming::new(num_forward_passes)` preallocates correctly
- [ ] `compute_total()` sums all components
- [ ] `reset()` zeros all fields recursively
- [ ] `total_solver_calls()` returns combined count
- [ ] `TrainingTiming` struct with iteration Vec
- [ ] `TrainingTiming::new(num_iterations, num_forward_passes)` preallocates
- [ ] Doc comments on all types and methods
- [ ] Exported from `src/timing/mod.rs`

## Implementation Guide

### Suggested Approach

1. Create `src/timing/iteration.rs`
2. Define `IterationTiming` and `TrainingTiming`
3. Implement all methods
4. Add `mod iteration;` and exports to `mod.rs`
5. Verify with `cargo build`

### Key Files to Modify

- `src/timing/iteration.rs`: Create new file
- `src/timing/mod.rs`: Add module and exports

### Pitfalls to Avoid

- ⚠️ Ensure `new()` preallocates the trajectory Vec via ForwardTiming
- ⚠️ Don't implement `to_output()` yet - that's T-006

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iteration_timing_new() {
        let it = IterationTiming::new(5);
        assert_eq!(it.forward.parallel.trajectories.len(), 5);
        assert_eq!(it.total.get(), Duration::ZERO);
    }

    #[test]
    fn test_compute_total() {
        let it = IterationTiming::new(2);
        it.model_allocation.set(Duration::from_millis(10));
        it.forward.total.set(Duration::from_millis(100));
        it.backward.total.set(Duration::from_millis(50));
        it.model_cleanup.set(Duration::from_millis(5));
        it.compute_total();
        assert_eq!(it.total.get(), Duration::from_millis(165));
    }

    #[test]
    fn test_reset() {
        let it = IterationTiming::new(2);
        it.model_allocation.set(Duration::from_millis(10));
        it.forward.preprocessing.saa_sampling.set(Duration::from_millis(5));
        it.reset();
        assert_eq!(it.model_allocation.get(), Duration::ZERO);
        assert_eq!(it.forward.preprocessing.saa_sampling.get(), Duration::ZERO);
    }

    #[test]
    fn test_total_solver_calls() {
        let it = IterationTiming::new(2);
        it.forward.parallel.trajectories[0].solver_calls.set(5);
        it.forward.parallel.trajectories[1].solver_calls.set(5);
        it.backward.solver_calls.set(10);
        assert_eq!(it.total_solver_calls(), 20);
    }

    #[test]
    fn test_training_timing_new() {
        let tt = TrainingTiming::new(10, 5);
        assert_eq!(tt.iterations.len(), 10);
        assert_eq!(tt.iterations[0].forward.parallel.trajectories.len(), 5);
    }

    #[test]
    fn test_training_timing_compute_total() {
        let tt = TrainingTiming::new(2, 1);
        tt.preprocessing.set(Duration::from_millis(100));
        tt.iterations[0].total.set(Duration::from_millis(50));
        tt.iterations[1].total.set(Duration::from_millis(50));
        tt.postprocessing.set(Duration::from_millis(10));
        tt.compute_total();
        assert_eq!(tt.total.get(), Duration::from_millis(210));
    }
}
```

## Documentation Requirements

- [ ] Doc comments on `IterationTiming` with usage example
- [ ] Doc comments on `TrainingTiming`
- [ ] Doc comments on all public methods

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Composition of existing types, straightforward

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Exported from timing module
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
