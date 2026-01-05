//! Timing metrics enum for feature-gated detailed timing.

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
}
