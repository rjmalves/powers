# T-002: Create ForwardTiming Hierarchy

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-001](./ticket-001-trajectory-timing.md)  
> **Blocks**: [T-004](./ticket-004-iteration-timing.md), [T-005](./ticket-005-aggregation-methods.md)

## Files to Read Before Starting

- `src/timing/trajectory.rs` - TrajectoryTiming (from T-001)
- `src/timing/metrics.rs` - Existing ForwardTiming (to be replaced)
- `src/sddp/mod.rs:37-65` - Legacy ForwardPassTiming structs
- `plans/timing-refactor/00-master-plan.md` - Schema definitions

## Context

### Background

The forward pass has three phases: preprocessing (SAA sampling), parallel trajectory execution, and postprocessing. The new `ForwardTiming` hierarchy captures this structure explicitly with nested structs.

### Current State

`ForwardTiming` exists in `src/timing/metrics.rs` with a flat structure. We're creating a new hierarchical version in `src/timing/forward.rs`.

## Specification

### File: `src/timing/forward.rs`

```rust
use std::cell::Cell;
use std::time::Duration;

use super::TrajectoryTiming;

/// Forward pass timing with hierarchical structure.
///
/// The forward pass has three phases:
/// 1. Preprocessing (sequential): SAA sampling
/// 2. Parallel execution: N trajectories solving in parallel
/// 3. Postprocessing (sequential): Detail capturing
pub struct ForwardTiming {
    /// Sequential preprocessing phase.
    pub preprocessing: ForwardPreprocessingTiming,

    /// Parallel trajectory execution.
    pub parallel: ForwardParallelTiming,

    /// Sequential postprocessing phase.
    pub postprocessing: ForwardPostprocessingTiming,

    /// Total forward pass time (wall clock).
    pub total: Cell<Duration>,
}

/// Forward pass preprocessing (sequential, before parallel section).
#[derive(Debug, Clone, Default)]
pub struct ForwardPreprocessingTiming {
    /// Time spent sampling scenarios from SAA tree.
    pub saa_sampling: Cell<Duration>,
}

/// Forward pass parallel section timing.
///
/// Stores BOTH wall time and individual trajectory times:
/// - `wall`: What we observe from outside the parallel section
/// - `trajectories`: Raw per-trajectory timing for statistical analysis
/// - Computed fields populated after parallel section completes
pub struct ForwardParallelTiming {
    /// Wall-clock time for entire parallel section.
    pub wall: Cell<Duration>,

    /// Individual trajectory timings (preallocated).
    pub trajectories: Vec<TrajectoryTiming>,

    // Computed fields (populated by compute_aggregates):

    /// Sum of all trajectory CPU times.
    pub cpu_total: Cell<Duration>,

    /// Parallel overhead: wall - cpu_total.
    pub overhead: Cell<Duration>,

    /// Average model preprocessing time per trajectory.
    pub model_preprocessing_avg: Cell<Duration>,

    /// Average solver time per trajectory.
    pub solver_avg: Cell<Duration>,

    /// Average model postprocessing time per trajectory.
    pub model_postprocessing_avg: Cell<Duration>,

    /// Maximum solver time across trajectories.
    pub solver_max: Cell<Duration>,
}

/// Forward pass postprocessing (sequential, after parallel section).
#[derive(Debug, Clone, Default)]
pub struct ForwardPostprocessingTiming {
    /// Time spent capturing trajectory details when requested.
    pub detail_capturing: Cell<Duration>,
}
```

### Constructor Methods

```rust
impl ForwardTiming {
    /// Create new ForwardTiming with preallocated trajectory Vec.
    pub fn new(num_trajectories: usize) -> Self {
        Self {
            preprocessing: ForwardPreprocessingTiming::default(),
            parallel: ForwardParallelTiming::new(num_trajectories),
            postprocessing: ForwardPostprocessingTiming::default(),
            total: Cell::new(Duration::ZERO),
        }
    }

    /// Compute total forward time from phases.
    pub fn compute_total(&self) {
        let total = self.preprocessing.saa_sampling.get()
            + self.parallel.wall.get()
            + self.postprocessing.detail_capturing.get();
        self.total.set(total);
    }

    /// Reset all timing values.
    pub fn reset(&self) {
        self.preprocessing.saa_sampling.set(Duration::ZERO);
        self.parallel.reset();
        self.postprocessing.detail_capturing.set(Duration::ZERO);
        self.total.set(Duration::ZERO);
    }
}

impl ForwardParallelTiming {
    /// Create with preallocated trajectory Vec.
    pub fn new(num_trajectories: usize) -> Self {
        Self {
            wall: Cell::new(Duration::ZERO),
            trajectories: (0..num_trajectories)
                .map(|_| TrajectoryTiming::new())
                .collect(),
            cpu_total: Cell::new(Duration::ZERO),
            overhead: Cell::new(Duration::ZERO),
            model_preprocessing_avg: Cell::new(Duration::ZERO),
            solver_avg: Cell::new(Duration::ZERO),
            model_postprocessing_avg: Cell::new(Duration::ZERO),
            solver_max: Cell::new(Duration::ZERO),
        }
    }

    /// Reset all fields including trajectories.
    pub fn reset(&self) {
        self.wall.set(Duration::ZERO);
        for t in &self.trajectories {
            t.reset();
        }
        self.cpu_total.set(Duration::ZERO);
        self.overhead.set(Duration::ZERO);
        self.model_preprocessing_avg.set(Duration::ZERO);
        self.solver_avg.set(Duration::ZERO);
        self.model_postprocessing_avg.set(Duration::ZERO);
        self.solver_max.set(Duration::ZERO);
    }

    /// Get total solver calls across all trajectories.
    pub fn total_solver_calls(&self) -> usize {
        self.trajectories.iter().map(|t| t.get_solver_calls()).sum()
    }
}
```

Note: `compute_aggregates()` will be implemented in T-005.

## Acceptance Criteria

- [ ] `ForwardTiming` struct with nested `preprocessing`, `parallel`, `postprocessing`
- [ ] `ForwardPreprocessingTiming` with `saa_sampling` field
- [ ] `ForwardParallelTiming` with trajectory Vec and computed fields
- [ ] `ForwardPostprocessingTiming` with `detail_capturing` field
- [ ] `ForwardTiming::new(num_trajectories)` preallocates Vec
- [ ] `compute_total()` sums phase times
- [ ] `reset()` methods for all structs
- [ ] Doc comments on all structs and methods
- [ ] Exported from `src/timing/mod.rs`

## Implementation Guide

### Suggested Approach

1. Create `src/timing/forward.rs`
2. Define all structs as specified
3. Implement constructors and reset methods
4. Add `mod forward;` and exports to `mod.rs`
5. Verify with `cargo build`

### Key Files to Modify

- `src/timing/forward.rs`: Create new file
- `src/timing/mod.rs`: Add module and exports

### Pitfalls to Avoid

- ⚠️ Don't implement `compute_aggregates()` here - that's T-005
- ⚠️ Don't derive `Clone` for `ForwardParallelTiming` (Vec of non-Copy)
- ⚠️ Ensure `new()` preallocates, not `default()`

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_timing_new_preallocates() {
        let ft = ForwardTiming::new(10);
        assert_eq!(ft.parallel.trajectories.len(), 10);
    }

    #[test]
    fn test_compute_total() {
        let ft = ForwardTiming::new(2);
        ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
        ft.parallel.wall.set(Duration::from_millis(100));
        ft.postprocessing.detail_capturing.set(Duration::from_millis(5));
        ft.compute_total();
        assert_eq!(ft.total.get(), Duration::from_millis(115));
    }

    #[test]
    fn test_reset() {
        let ft = ForwardTiming::new(2);
        ft.preprocessing.saa_sampling.set(Duration::from_millis(10));
        ft.parallel.trajectories[0].solver.set(Duration::from_millis(50));
        ft.reset();
        assert_eq!(ft.preprocessing.saa_sampling.get(), Duration::ZERO);
        assert_eq!(ft.parallel.trajectories[0].solver.get(), Duration::ZERO);
    }

    #[test]
    fn test_total_solver_calls() {
        let ft = ForwardTiming::new(3);
        ft.parallel.trajectories[0].solver_calls.set(5);
        ft.parallel.trajectories[1].solver_calls.set(5);
        ft.parallel.trajectories[2].solver_calls.set(5);
        assert_eq!(ft.parallel.total_solver_calls(), 15);
    }
}
```

## Documentation Requirements

- [ ] Module-level doc explaining forward pass timing hierarchy
- [ ] Doc comments on all structs explaining their role
- [ ] Doc comments on all public methods

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Multiple structs but straightforward definitions

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Exported from timing module
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
