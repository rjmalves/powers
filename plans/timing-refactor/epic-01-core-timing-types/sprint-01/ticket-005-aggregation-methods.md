# T-005: Implement Aggregation Methods

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-002](./ticket-002-forward-timing.md)  
> **Blocks**: [T-007](./ticket-007-unit-tests.md)

## Files to Read Before Starting

- `src/timing/forward.rs` - ForwardParallelTiming (from T-002)
- `src/timing/trajectory.rs` - TrajectoryTiming (from T-001)
- `src/sddp/mod.rs:1984-2005` - Current rescaling logic (to be replaced)

## Context

### Background

After the parallel forward section completes, we need to compute aggregate statistics from the individual trajectory timings: averages, maximum, total CPU time, and parallel overhead. This replaces the current "recalibrate" scaling code.

**Key difference from current code**: We do NOT redistribute timing proportionally. We compute:
- `cpu_total` = sum of all trajectory CPU times
- `overhead` = wall - cpu_total (can be negative if wall < cpu_total due to parallelism)
- Averages are simple arithmetic means

### Current State

The current code in `sddp/mod.rs:1984-2005` redistributes timing proportionally to wall time. This will be removed.

## Specification

### Add to `src/timing/forward.rs`

```rust
impl ForwardParallelTiming {
    /// Compute aggregate statistics from trajectory timings.
    ///
    /// Call this AFTER the parallel forward section completes.
    /// Populates: `cpu_total`, `overhead`, `*_avg`, `solver_max`.
    ///
    /// # Panics
    ///
    /// Panics if `trajectories` is empty.
    pub fn compute_aggregates(&self) {
        let n = self.trajectories.len();
        assert!(n > 0, "Cannot compute aggregates with zero trajectories");

        // Sum CPU times
        let cpu_total: Duration = self.trajectories.iter()
            .map(|t| t.cpu_time())
            .sum();
        self.cpu_total.set(cpu_total);

        // Compute overhead (can be negative conceptually, but Duration is unsigned)
        // If wall < cpu_total (due to parallelism), overhead is zero
        let wall = self.wall.get();
        let overhead = wall.saturating_sub(cpu_total);
        self.overhead.set(overhead);

        // Compute sums for averaging
        let total_model_prep: Duration = self.trajectories.iter()
            .map(|t| t.model_preprocessing.get())
            .sum();
        let total_solver: Duration = self.trajectories.iter()
            .map(|t| t.solver.get())
            .sum();
        let total_model_post: Duration = self.trajectories.iter()
            .map(|t| t.model_postprocessing.get())
            .sum();

        // Compute averages
        let n_u32 = n as u32;
        self.model_preprocessing_avg.set(total_model_prep / n_u32);
        self.solver_avg.set(total_solver / n_u32);
        self.model_postprocessing_avg.set(total_model_post / n_u32);

        // Compute max solver time
        let max_solver = self.trajectories.iter()
            .map(|t| t.solver.get())
            .max()
            .unwrap_or(Duration::ZERO);
        self.solver_max.set(max_solver);
    }

    /// Check if aggregates have been computed.
    ///
    /// Returns true if `cpu_total` is non-zero OR if all trajectories have zero CPU time.
    pub fn aggregates_computed(&self) -> bool {
        self.cpu_total.get() > Duration::ZERO
            || self.trajectories.iter().all(|t| t.cpu_time() == Duration::ZERO)
    }
}
```

### Optional: Aggregation Utilities

Create `src/timing/aggregation.rs` for reusable utilities:

```rust
//! Aggregation utilities for timing statistics.

use std::time::Duration;

/// Compute average of durations.
#[inline]
pub fn duration_avg(durations: &[Duration]) -> Duration {
    if durations.is_empty() {
        return Duration::ZERO;
    }
    let total: Duration = durations.iter().sum();
    total / durations.len() as u32
}

/// Compute max of durations.
#[inline]
pub fn duration_max(durations: &[Duration]) -> Duration {
    durations.iter().copied().max().unwrap_or(Duration::ZERO)
}

/// Compute sum of durations.
#[inline]
pub fn duration_sum(durations: &[Duration]) -> Duration {
    durations.iter().sum()
}
```

## Acceptance Criteria

- [ ] `ForwardParallelTiming::compute_aggregates()` implemented
- [ ] Computes `cpu_total` as sum of trajectory CPU times
- [ ] Computes `overhead` as `wall.saturating_sub(cpu_total)`
- [ ] Computes `model_preprocessing_avg`, `solver_avg`, `model_postprocessing_avg`
- [ ] Computes `solver_max`
- [ ] Panics on empty trajectories (or returns early - document choice)
- [ ] `aggregates_computed()` helper method
- [ ] No timing redistribution (verify the old scaling code is NOT replicated)
- [ ] Doc comments explaining the computation
- [ ] Unit tests covering edge cases

## Implementation Guide

### Suggested Approach

1. Add `compute_aggregates()` to `ForwardParallelTiming` in `forward.rs`
2. Optionally create `aggregation.rs` for utilities
3. Add tests for various scenarios
4. Verify with `cargo test`

### Key Files to Modify

- `src/timing/forward.rs`: Add method
- `src/timing/aggregation.rs`: Optional new file
- `src/timing/mod.rs`: Export if adding aggregation module

### Pitfalls to Avoid

- ⚠️ Do NOT replicate the proportional redistribution from `sddp/mod.rs`
- ⚠️ Use `saturating_sub` to avoid panic on underflow
- ⚠️ Handle division by zero (empty trajectories)

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_aggregates_basic() {
        let parallel = ForwardParallelTiming::new(3);
        
        // Set up trajectory timings
        parallel.trajectories[0].model_preprocessing.set(Duration::from_millis(10));
        parallel.trajectories[0].solver.set(Duration::from_millis(100));
        parallel.trajectories[0].model_postprocessing.set(Duration::from_millis(5));

        parallel.trajectories[1].model_preprocessing.set(Duration::from_millis(20));
        parallel.trajectories[1].solver.set(Duration::from_millis(150));
        parallel.trajectories[1].model_postprocessing.set(Duration::from_millis(10));

        parallel.trajectories[2].model_preprocessing.set(Duration::from_millis(15));
        parallel.trajectories[2].solver.set(Duration::from_millis(120));
        parallel.trajectories[2].model_postprocessing.set(Duration::from_millis(8));

        // Wall time is less than CPU total (parallel speedup)
        parallel.wall.set(Duration::from_millis(200));

        parallel.compute_aggregates();

        // CPU total = (10+100+5) + (20+150+10) + (15+120+8) = 115 + 180 + 143 = 438
        assert_eq!(parallel.cpu_total.get(), Duration::from_millis(438));
        
        // Overhead = 200 - 438 = 0 (saturating)
        assert_eq!(parallel.overhead.get(), Duration::ZERO);

        // Averages: model_prep = (10+20+15)/3 = 15
        assert_eq!(parallel.model_preprocessing_avg.get(), Duration::from_millis(15));
        
        // solver_avg = (100+150+120)/3 = 123.33... ≈ 123ms
        assert_eq!(parallel.solver_avg.get(), Duration::from_millis(123));

        // solver_max = 150
        assert_eq!(parallel.solver_max.get(), Duration::from_millis(150));
    }

    #[test]
    fn test_compute_aggregates_with_overhead() {
        let parallel = ForwardParallelTiming::new(2);
        
        parallel.trajectories[0].solver.set(Duration::from_millis(50));
        parallel.trajectories[1].solver.set(Duration::from_millis(50));

        // Wall time greater than CPU total (scheduling overhead)
        parallel.wall.set(Duration::from_millis(150));

        parallel.compute_aggregates();

        assert_eq!(parallel.cpu_total.get(), Duration::from_millis(100));
        assert_eq!(parallel.overhead.get(), Duration::from_millis(50));
    }

    #[test]
    #[should_panic(expected = "Cannot compute aggregates with zero trajectories")]
    fn test_compute_aggregates_empty_panics() {
        let parallel = ForwardParallelTiming::new(0);
        parallel.compute_aggregates();
    }

    #[test]
    fn test_aggregates_computed() {
        let parallel = ForwardParallelTiming::new(2);
        
        // Before computation
        parallel.trajectories[0].solver.set(Duration::from_millis(50));
        assert!(!parallel.aggregates_computed());

        // After computation
        parallel.compute_aggregates();
        assert!(parallel.aggregates_computed());
    }
}
```

## Documentation Requirements

- [ ] Doc comment on `compute_aggregates()` explaining the algorithm
- [ ] Note that overhead uses `saturating_sub` (no negative values)
- [ ] Explain difference from old redistribution approach

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Core algorithm work, needs careful testing

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] No proportional redistribution (verified by code review)
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
