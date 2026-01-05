//! Backward pass timing hierarchy.

use std::cell::Cell;
use std::time::Duration;

use super::output::BackwardTimingOutput;

/// Backward pass timing with per-phase breakdown.
///
/// The backward pass iterates stages in reverse order:
/// - For each stage: Phase 1 → Phase 2 → Phase 3
/// - Timing is accumulated across all stages
///
/// # Example
///
/// ```ignore
/// use powers_rs::timing::{NewBackwardTiming, TimingGuard};
///
/// let timing = NewBackwardTiming::new();
///
/// // Phase 1: Cut computation (accumulated across stages)
/// {
///     let _guard = TimingGuard::new(&timing.phase1.solver);
///     // ... solve branching ...
/// }
///
/// // Phase 2: Cut selection
/// {
///     let _guard = TimingGuard::new(&timing.phase2.cut_selection);
///     // ... select cuts ...
/// }
/// ```
pub struct NewBackwardTiming {
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

impl NewBackwardTiming {
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

impl Default for NewBackwardTiming {
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
    ///
    /// This accumulates timing across stages during the backward pass.
    pub fn add(&self, other: &BackwardPhase1Timing) {
        self.model_preprocessing.set(
            self.model_preprocessing.get() + other.model_preprocessing.get(),
        );
        self.solver.set(self.solver.get() + other.solver.get());
        self.model_postprocessing.set(
            self.model_postprocessing.get() + other.model_postprocessing.get(),
        );
        self.cut_computation
            .set(self.cut_computation.get() + other.cut_computation.get());
    }

    /// Total Phase 1 time.
    pub fn total(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
            + self.cut_computation.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_timing_new() {
        let bt = NewBackwardTiming::new();
        assert_eq!(bt.total.get(), Duration::ZERO);
        assert_eq!(bt.solver_calls.get(), 0);
    }

    #[test]
    fn test_reset() {
        let bt = NewBackwardTiming::new();
        bt.phase1.solver.set(Duration::from_millis(100));
        bt.solver_calls.set(10);
        bt.reset();
        assert_eq!(bt.phase1.solver.get(), Duration::ZERO);
        assert_eq!(bt.solver_calls.get(), 0);
    }

    #[test]
    fn test_add_solver_calls() {
        let bt = NewBackwardTiming::new();
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
