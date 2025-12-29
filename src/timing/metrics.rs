//! Timing metrics and structured timing data for SDDP execution.

use std::cell::Cell;
use std::time::Duration;

/// All timing metrics tracked in SDDP execution.
/// Organized hierarchically by algorithm phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimingMetric {
    // Forward pass - sequential components
    SaaSampling,
    ForwardPostprocessing,

    // Forward pass - per-trajectory (parallel)
    ForwardModelPrep,
    ForwardSolver,
    ForwardExtraction,

    // Forward pass - parallel overhead (computed, not measured)
    ForwardParallelWallTime,
    ForwardParallelOverhead,

    // Backward pass - sequential components
    BackwardPreprocessing,
    CutSelection,
    FcfUpdate,
    HandlerApplication,

    // Backward pass - per-stage (sequential within stage, parallel across branchings)
    BackwardModelPrep,
    BackwardSolver,
    BackwardExtraction,
    CutComputation,
}

impl TimingMetric {
    /// Total number of timing metrics.
    pub const COUNT: usize = 16;

    /// Get the array index for this metric.
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Forward pass timing with explicit parallel overhead tracking.
///
/// ⚠️ CRITICAL: `model_preprocessing`, `solver`, and `model_postprocessing` contain
/// the PRECISE measured values. They are NEVER overwritten or redistributed.
/// `parallel_overhead` is COMPUTED as the difference between wall time and CPU time.
#[derive(Debug, Clone, Default)]
pub struct ForwardTiming {
    /// Time spent in SAA sampling (sequential).
    pub saa_sampling: Cell<Duration>,

    /// Wall-clock time for the parallel forward pass section.
    pub parallel_wall_time: Cell<Duration>,

    /// Total CPU time spent in model preprocessing across all trajectories.
    pub model_preprocessing: Cell<Duration>,

    /// Total CPU time spent in solver across all trajectories.
    pub solver: Cell<Duration>,

    /// Total CPU time spent in model postprocessing across all trajectories.
    pub model_postprocessing: Cell<Duration>,

    /// Computed parallel overhead: `parallel_wall_time - sum(CPU times)`.
    pub parallel_overhead: Cell<Duration>,

    /// Time spent in forward pass postprocessing (sequential).
    pub postprocessing: Cell<Duration>,
}

impl ForwardTiming {
    /// Get the total measured CPU time for the parallel section.
    pub fn total_cpu_time(&self) -> Duration {
        self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
    }

    /// Get the total time including sequential components.
    pub fn total_time(&self) -> Duration {
        self.saa_sampling.get()
            + self.parallel_wall_time.get()
            + self.postprocessing.get()
    }
}

/// Backward pass timing.
#[derive(Debug, Clone, Default)]
pub struct BackwardTiming {
    /// Time spent in backward pass preprocessing.
    pub preprocessing: Cell<Duration>,

    /// Total CPU time spent in model preprocessing across all stages.
    pub model_preprocessing: Cell<Duration>,

    /// Total CPU time spent in solver across all stages.
    pub solver: Cell<Duration>,

    /// Total CPU time spent in model postprocessing across all stages.
    pub model_postprocessing: Cell<Duration>,

    /// Total CPU time spent computing cuts.
    pub cut_computation: Cell<Duration>,

    /// Time spent in cut selection.
    pub cut_selection: Cell<Duration>,

    /// Time spent updating the FCF.
    pub fcf_update: Cell<Duration>,

    /// Time spent applying handlers.
    pub handler_application: Cell<Duration>,
}

impl BackwardTiming {
    /// Get the total time for the backward pass.
    pub fn total_time(&self) -> Duration {
        self.preprocessing.get()
            + self.model_preprocessing.get()
            + self.solver.get()
            + self.model_postprocessing.get()
            + self.cut_computation.get()
            + self.cut_selection.get()
            + self.fcf_update.get()
            + self.handler_application.get()
    }
}

/// Complete timing for one SDDP iteration.
#[derive(Debug, Clone, Default)]
pub struct IterationTiming {
    /// Forward pass timing.
    pub forward: ForwardTiming,

    /// Backward pass timing.
    pub backward: BackwardTiming,

    /// Total iteration time (wall clock).
    pub total: Cell<Duration>,
}

impl IterationTiming {
    /// Compute parallel overhead for forward pass.
    ///
    /// Call this AFTER all trajectory timings have been collected.
    /// Computes: `parallel_overhead = parallel_wall_time - total_cpu_time`
    pub fn compute_forward_parallel_overhead(&self) {
        let cpu_time = self.forward.total_cpu_time();
        let wall_time = self.forward.parallel_wall_time.get();
        let overhead = wall_time.saturating_sub(cpu_time);
        self.forward.parallel_overhead.set(overhead);
    }

    /// Reset all timing values to zero.
    pub fn reset(&self) {
        self.forward.saa_sampling.set(Duration::ZERO);
        self.forward.parallel_wall_time.set(Duration::ZERO);
        self.forward.model_preprocessing.set(Duration::ZERO);
        self.forward.solver.set(Duration::ZERO);
        self.forward.model_postprocessing.set(Duration::ZERO);
        self.forward.parallel_overhead.set(Duration::ZERO);
        self.forward.postprocessing.set(Duration::ZERO);

        self.backward.preprocessing.set(Duration::ZERO);
        self.backward.model_preprocessing.set(Duration::ZERO);
        self.backward.solver.set(Duration::ZERO);
        self.backward.model_postprocessing.set(Duration::ZERO);
        self.backward.cut_computation.set(Duration::ZERO);
        self.backward.cut_selection.set(Duration::ZERO);
        self.backward.fcf_update.set(Duration::ZERO);
        self.backward.handler_application.set(Duration::ZERO);

        self.total.set(Duration::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_metric_index_unique() {
        let mut seen = std::collections::HashSet::new();
        let metrics = [
            TimingMetric::SaaSampling,
            TimingMetric::ForwardPostprocessing,
            TimingMetric::ForwardModelPrep,
            TimingMetric::ForwardSolver,
            TimingMetric::ForwardExtraction,
            TimingMetric::ForwardParallelWallTime,
            TimingMetric::ForwardParallelOverhead,
            TimingMetric::BackwardPreprocessing,
            TimingMetric::CutSelection,
            TimingMetric::FcfUpdate,
            TimingMetric::HandlerApplication,
            TimingMetric::BackwardModelPrep,
            TimingMetric::BackwardSolver,
            TimingMetric::BackwardExtraction,
            TimingMetric::CutComputation,
        ];

        for metric in metrics {
            assert!(
                seen.insert(metric.index()),
                "Duplicate index for {:?}",
                metric
            );
        }
    }

    #[test]
    fn test_timing_metric_count() {
        assert_eq!(TimingMetric::COUNT, 16);
    }

    #[test]
    fn test_forward_timing_total_cpu_time() {
        let timing = ForwardTiming::default();
        timing.model_preprocessing.set(Duration::from_millis(100));
        timing.solver.set(Duration::from_millis(200));
        timing.model_postprocessing.set(Duration::from_millis(50));

        assert_eq!(timing.total_cpu_time(), Duration::from_millis(350));
    }

    #[test]
    fn test_iteration_timing_compute_parallel_overhead() {
        let timing = IterationTiming::default();

        // Simulate parallel execution with overhead
        timing
            .forward
            .parallel_wall_time
            .set(Duration::from_millis(500));
        timing
            .forward
            .model_preprocessing
            .set(Duration::from_millis(100));
        timing.forward.solver.set(Duration::from_millis(200));
        timing
            .forward
            .model_postprocessing
            .set(Duration::from_millis(50));

        timing.compute_forward_parallel_overhead();

        // Overhead = 500 - (100 + 200 + 50) = 150
        assert_eq!(
            timing.forward.parallel_overhead.get(),
            Duration::from_millis(150)
        );
    }

    #[test]
    fn test_iteration_timing_reset() {
        let timing = IterationTiming::default();
        timing.forward.solver.set(Duration::from_millis(100));
        timing.backward.solver.set(Duration::from_millis(200));
        timing.total.set(Duration::from_millis(300));

        timing.reset();

        assert_eq!(timing.forward.solver.get(), Duration::ZERO);
        assert_eq!(timing.backward.solver.get(), Duration::ZERO);
        assert_eq!(timing.total.get(), Duration::ZERO);
    }
}
