//! Per-trajectory timing for parallel forward pass execution.

use std::cell::Cell;
use std::time::Duration;

/// Per-trajectory timing collected during parallel forward pass.
///
/// Uses `Cell<Duration>` for interior mutability with `TimingGuard`.
/// Each trajectory in a parallel forward pass has its own instance,
/// stored in a preallocated Vec for zero-allocation timing.
///
/// # Thread Safety
///
/// This type is NOT thread-safe for concurrent mutation. However, it is
/// designed for use in `par_iter` contexts where each thread has exclusive
/// access to its own `TrajectoryTiming` instance via indexed access.
///
/// # Example with TimingGuard
///
/// ```ignore
/// use powers_rs::timing::{TrajectoryTiming, TimingGuard};
///
/// let timing = TrajectoryTiming::new();
///
/// // Model preprocessing phase
/// {
///     let _guard = TimingGuard::new(&timing.model_preprocessing);
///     // ... prepare subproblem ...
/// }
///
/// // Solver time comes from realize_and_solve
/// let solve_result = subproblem.realize_and_solve(...);
/// timing.add_solver_time(solve_result.solver_time);
/// timing.increment_solver_calls();
/// ```
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

impl TrajectoryTiming {
    /// Create a new trajectory timing with all fields zeroed.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Total CPU time for this trajectory.
    ///
    /// Returns the sum of model preprocessing, solver, and model postprocessing times.
    #[inline]
    pub fn cpu_time(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
    }

    /// Reset all timing values to zero.
    ///
    /// Useful for reusing timing structures across iterations.
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
    ///
    /// # Arguments
    ///
    /// * `duration` - Additional solver time to accumulate
    #[inline]
    pub fn add_solver_time(&self, duration: Duration) {
        self.solver.set(self.solver.get() + duration);
    }

    /// Add model postprocessing time.
    ///
    /// # Arguments
    ///
    /// * `duration` - Additional postprocessing time to accumulate
    #[inline]
    pub fn add_model_postprocessing(&self, duration: Duration) {
        self.model_postprocessing
            .set(self.model_postprocessing.get() + duration);
    }
}

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
