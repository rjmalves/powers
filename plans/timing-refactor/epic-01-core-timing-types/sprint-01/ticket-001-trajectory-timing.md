# T-001: Create TrajectoryTiming Struct

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [T-002](./ticket-002-forward-timing.md)

## Files to Read Before Starting

- `src/timing/mod.rs` - Current module structure
- `src/timing/guard.rs` - TimingGuard implementation
- `src/algorithm/context.rs:151-190` - Existing TrajectoryTiming (to be replaced)

## Context

### Background

The forward pass executes N trajectories in parallel. Each trajectory needs its own timing accumulator to track model preprocessing, solver, and postprocessing time. This struct will be stored in a preallocated `Vec` and accessed by index during parallel execution.

### Current State

`TrajectoryTiming` exists in `algorithm/context.rs` but will be moved and slightly modified.

## Specification

### Struct Definition

```rust
// src/timing/trajectory.rs

use std::cell::Cell;
use std::time::Duration;

/// Per-trajectory timing collected during parallel forward pass.
///
/// Uses `Cell<Duration>` for interior mutability with `TimingGuard`.
/// Each trajectory in a parallel forward pass has its own instance,
/// stored in a preallocated Vec for zero-allocation timing.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryTiming {
    /// Time preparing the subproblem model (state injection, cut updates).
    pub model_preprocessing: Cell<Duration>,

    /// Time in LP solver.
    pub solver: Cell<Duration>,

    /// Time extracting solution (primal/dual values, state update).
    pub model_postprocessing: Cell<Duration>,

    /// Number of solver calls in this trajectory.
    pub solver_calls: Cell<usize>,
}
```

### Methods

```rust
impl TrajectoryTiming {
    /// Create a new trajectory timing with all fields zeroed.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total CPU time for this trajectory.
    #[inline]
    pub fn cpu_time(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
    }

    /// Reset all timing values to zero.
    #[inline]
    pub fn reset(&self) {
        self.model_preprocessing.set(Duration::ZERO);
        self.solver.set(Duration::ZERO);
        self.model_postprocessing.set(Duration::ZERO);
        self.solver_calls.set(0);
    }

    /// Increment solver call count.
    #[inline]
    pub fn increment_solver_calls(&self) {
        self.solver_calls.set(self.solver_calls.get() + 1);
    }

    /// Get solver call count.
    #[inline]
    pub fn get_solver_calls(&self) -> usize {
        self.solver_calls.get()
    }

    /// Add solver time (for internal timing from realize_and_solve).
    #[inline]
    pub fn add_solver_time(&self, duration: Duration) {
        self.solver.set(self.solver.get() + duration);
    }

    /// Add model postprocessing time.
    #[inline]
    pub fn add_model_postprocessing(&self, duration: Duration) {
        self.model_postprocessing.set(self.model_postprocessing.get() + duration);
    }
}
```

## Acceptance Criteria

- [ ] `TrajectoryTiming` struct defined in `src/timing/trajectory.rs`
- [ ] All methods implemented with `#[inline]` hints
- [ ] `cpu_time()` returns sum of three timing fields
- [ ] `reset()` zeros all fields
- [ ] Struct derives `Debug, Clone, Default`
- [ ] Doc comments on struct and all public methods
- [ ] Exported from `src/timing/mod.rs`

## Implementation Guide

### Suggested Approach

1. Create `src/timing/trajectory.rs`
2. Define struct and methods as specified
3. Add `mod trajectory;` and `pub use trajectory::TrajectoryTiming;` to `mod.rs`
4. Run `cargo build` to verify compilation

### Key Files to Modify

- `src/timing/trajectory.rs`: Create new file
- `src/timing/mod.rs`: Add module and export

### Patterns to Follow

- Follow existing pattern in `src/timing/guard.rs` for style
- Use `Cell<Duration>` pattern from `src/timing/metrics.rs`

### Pitfalls to Avoid

- ⚠️ Don't implement `Copy` - struct contains `Cell` which is not `Copy`
- ⚠️ Don't remove the existing `TrajectoryTiming` in `algorithm/context.rs` yet (Epic 4)

## Testing Requirements

### Unit Tests

Add to `src/timing/trajectory.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_zero() {
        let t = TrajectoryTiming::new();
        assert_eq!(t.model_preprocessing.get(), Duration::ZERO);
        assert_eq!(t.solver.get(), Duration::ZERO);
        assert_eq!(t.model_postprocessing.get(), Duration::ZERO);
        assert_eq!(t.solver_calls.get(), 0);
    }

    #[test]
    fn test_cpu_time() {
        let t = TrajectoryTiming::new();
        t.model_preprocessing.set(Duration::from_millis(10));
        t.solver.set(Duration::from_millis(20));
        t.model_postprocessing.set(Duration::from_millis(5));
        assert_eq!(t.cpu_time(), Duration::from_millis(35));
    }

    #[test]
    fn test_reset() {
        let t = TrajectoryTiming::new();
        t.solver.set(Duration::from_millis(100));
        t.solver_calls.set(5);
        t.reset();
        assert_eq!(t.solver.get(), Duration::ZERO);
        assert_eq!(t.solver_calls.get(), 0);
    }

    #[test]
    fn test_increment_solver_calls() {
        let t = TrajectoryTiming::new();
        t.increment_solver_calls();
        t.increment_solver_calls();
        assert_eq!(t.get_solver_calls(), 2);
    }

    #[test]
    fn test_add_methods() {
        let t = TrajectoryTiming::new();
        t.add_solver_time(Duration::from_millis(10));
        t.add_solver_time(Duration::from_millis(20));
        assert_eq!(t.solver.get(), Duration::from_millis(30));

        t.add_model_postprocessing(Duration::from_millis(5));
        assert_eq!(t.model_postprocessing.get(), Duration::from_millis(5));
    }
}
```

## Documentation Requirements

- [ ] Doc comment on struct explaining purpose and thread safety
- [ ] Doc comments on all public methods
- [ ] Example in struct doc showing TimingGuard usage

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple struct definition, similar to existing code

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Exported from timing module
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
