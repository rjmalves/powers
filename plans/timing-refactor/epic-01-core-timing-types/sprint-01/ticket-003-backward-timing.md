# T-003: Create BackwardTiming Hierarchy

> **Epic**: [Core Timing Types](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [T-004](./ticket-004-iteration-timing.md)

## Files to Read Before Starting

- `src/timing/metrics.rs` - Existing BackwardTiming (reference)
- `src/algorithm/backward_pass.rs:48-120` - Current BackwardPassTimingAccumulator
- `plans/timing-refactor/00-master-plan.md` - Schema definitions

## Context

### Background

The backward pass iterates stages in reverse order with three phases per stage:
1. Phase 1: Parallel cut computation (model prep, solver, postprocessing, cut computation)
2. Phase 2: Sequential cut selection
3. Phase 3: Parallel problem update

The new `BackwardTiming` captures this phase structure explicitly.

### Current State

`BackwardPassTimingAccumulator` exists in `algorithm/backward_pass.rs` with a flat structure. We're creating a hierarchical version.

## Specification

### File: `src/timing/backward.rs`

```rust
use std::cell::Cell;
use std::time::Duration;

/// Backward pass timing with per-phase breakdown.
///
/// The backward pass iterates stages in reverse order:
/// - For each stage: Phase 1 → Phase 2 → Phase 3
/// - Timing is accumulated across all stages
pub struct BackwardTiming {
    /// Phase 1: Parallel cut computation (summed across stages).
    pub phase1: BackwardPhase1Timing,

    /// Phase 2: Sequential cut selection (summed across stages).
    pub phase2: BackwardPhase2Timing,

    /// Phase 3: Parallel problem update (summed across stages).
    pub phase3: BackwardPhase3Timing,

    /// Total backward pass time (wall clock).
    pub total: Cell<Duration>,

    /// Total solver calls across all stages.
    pub solver_calls: Cell<usize>,
}

/// Backward Phase 1: Parallel cut computation.
///
/// For each stage, branchings are solved in parallel across handlers.
/// Times are accumulated across all stages.
#[derive(Debug, Clone, Default)]
pub struct BackwardPhase1Timing {
    /// Model preprocessing time (state injection, noise realization).
    pub model_preprocessing: Cell<Duration>,

    /// LP solver time.
    pub solver: Cell<Duration>,

    /// Model postprocessing time (cut coefficient extraction).
    pub model_postprocessing: Cell<Duration>,

    /// Cut aggregation and risk measure application.
    pub cut_computation: Cell<Duration>,
}

/// Backward Phase 2: Sequential cut selection.
#[derive(Debug, Clone, Default)]
pub struct BackwardPhase2Timing {
    /// Time spent in cut selection algorithm.
    pub cut_selection: Cell<Duration>,
}

/// Backward Phase 3: Parallel problem update.
#[derive(Debug, Clone, Default)]
pub struct BackwardPhase3Timing {
    /// Time applying cuts to handler models and problems.
    pub problem_update: Cell<Duration>,
}
```

### Constructor and Helper Methods

```rust
impl BackwardTiming {
    /// Create new BackwardTiming with all fields zeroed.
    pub fn new() -> Self {
        Self {
            phase1: BackwardPhase1Timing::default(),
            phase2: BackwardPhase2Timing::default(),
            phase3: BackwardPhase3Timing::default(),
            total: Cell::new(Duration::ZERO),
            solver_calls: Cell::new(0),
        }
    }

    /// Reset all timing values.
    pub fn reset(&self) {
        self.phase1.reset();
        self.phase2.cut_selection.set(Duration::ZERO);
        self.phase3.problem_update.set(Duration::ZERO);
        self.total.set(Duration::ZERO);
        self.solver_calls.set(0);
    }

    /// Increment solver call count.
    #[inline]
    pub fn add_solver_calls(&self, count: usize) {
        self.solver_calls.set(self.solver_calls.get() + count);
    }

    /// Get total solver calls.
    #[inline]
    pub fn get_solver_calls(&self) -> usize {
        self.solver_calls.get()
    }
}

impl Default for BackwardTiming {
    fn default() -> Self {
        Self::new()
    }
}

impl BackwardPhase1Timing {
    /// Reset all Phase 1 timing values.
    pub fn reset(&self) {
        self.model_preprocessing.set(Duration::ZERO);
        self.solver.set(Duration::ZERO);
        self.model_postprocessing.set(Duration::ZERO);
        self.cut_computation.set(Duration::ZERO);
    }

    /// Add timing from a single stage's Phase 1.
    pub fn add(&self, other: &BackwardPhase1Timing) {
        self.model_preprocessing.set(
            self.model_preprocessing.get() + other.model_preprocessing.get()
        );
        self.solver.set(self.solver.get() + other.solver.get());
        self.model_postprocessing.set(
            self.model_postprocessing.get() + other.model_postprocessing.get()
        );
        self.cut_computation.set(
            self.cut_computation.get() + other.cut_computation.get()
        );
    }

    /// Total Phase 1 time.
    pub fn total(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
            + self.cut_computation.get()
    }
}
```

## Acceptance Criteria

- [ ] `BackwardTiming` struct with `phase1`, `phase2`, `phase3`, `total`, `solver_calls`
- [ ] `BackwardPhase1Timing` with 4 timing fields
- [ ] `BackwardPhase2Timing` with `cut_selection`
- [ ] `BackwardPhase3Timing` with `problem_update`
- [ ] `new()` and `default()` implementations
- [ ] `reset()` methods for all structs
- [ ] `BackwardPhase1Timing::add()` for accumulating per-stage timing
- [ ] Doc comments on all structs and methods
- [ ] Exported from `src/timing/mod.rs`

## Implementation Guide

### Suggested Approach

1. Create `src/timing/backward.rs`
2. Define all structs as specified
3. Implement constructors, reset, and add methods
4. Add `mod backward;` and exports to `mod.rs`
5. Verify with `cargo build`

### Key Files to Modify

- `src/timing/backward.rs`: Create new file
- `src/timing/mod.rs`: Add module and exports

### Pitfalls to Avoid

- ⚠️ Don't remove existing `BackwardPassTimingAccumulator` yet (Epic 4)
- ⚠️ Ensure `add()` method handles Cell correctly

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_timing_new() {
        let bt = BackwardTiming::new();
        assert_eq!(bt.total.get(), Duration::ZERO);
        assert_eq!(bt.solver_calls.get(), 0);
    }

    #[test]
    fn test_reset() {
        let bt = BackwardTiming::new();
        bt.phase1.solver.set(Duration::from_millis(100));
        bt.solver_calls.set(10);
        bt.reset();
        assert_eq!(bt.phase1.solver.get(), Duration::ZERO);
        assert_eq!(bt.solver_calls.get(), 0);
    }

    #[test]
    fn test_add_solver_calls() {
        let bt = BackwardTiming::new();
        bt.add_solver_calls(5);
        bt.add_solver_calls(3);
        assert_eq!(bt.get_solver_calls(), 8);
    }

    #[test]
    fn test_phase1_add() {
        let p1 = BackwardPhase1Timing::default();
        p1.solver.set(Duration::from_millis(10));

        let p2 = BackwardPhase1Timing::default();
        p2.solver.set(Duration::from_millis(20));
        p2.model_preprocessing.set(Duration::from_millis(5));

        p1.add(&p2);
        assert_eq!(p1.solver.get(), Duration::from_millis(30));
        assert_eq!(p1.model_preprocessing.get(), Duration::from_millis(5));
    }

    #[test]
    fn test_phase1_total() {
        let p1 = BackwardPhase1Timing::default();
        p1.model_preprocessing.set(Duration::from_millis(10));
        p1.solver.set(Duration::from_millis(20));
        p1.model_postprocessing.set(Duration::from_millis(5));
        p1.cut_computation.set(Duration::from_millis(3));
        assert_eq!(p1.total(), Duration::from_millis(38));
    }
}
```

## Documentation Requirements

- [ ] Module-level doc explaining backward pass timing hierarchy
- [ ] Doc comments on all structs explaining the 3-phase architecture
- [ ] Doc comments on all public methods

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simpler than forward timing (no trajectory Vec)

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Exported from timing module
- [ ] Doc comments complete
- [ ] `cargo clippy` clean
