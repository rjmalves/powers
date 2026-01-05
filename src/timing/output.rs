//! Output types for timing data.
//!
//! These types use plain `Duration` (not `Cell<Duration>`) for serialization
//! and storage in results. Created via `to_output()` methods on accumulator types.

use std::time::Duration;

/// Forward pass timing for output/results.
///
/// Contains aggregated statistics from parallel execution.
/// All `Cell<Duration>` values from the accumulator are converted to plain `Duration`.
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
///
/// Contains timing from all three phases accumulated across stages.
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
///
/// Top-level timing for a single SDDP training iteration.
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
        assert_eq!(out.solver_calls, 0);
    }

    #[test]
    fn test_iteration_timing_output_default() {
        let out = IterationTimingOutput::default();
        assert_eq!(out.total, Duration::ZERO);
        assert_eq!(out.solver_calls, 0);
    }
}
