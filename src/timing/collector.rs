//! Timing collector trait and implementations.

use std::time::Duration;

use super::metrics::TimingMetric;

/// Trait for timing collection strategies.
///
/// This abstraction enables:
/// - Testing without actual timing (`NullTimingCollector`)
/// - Thread-safe collection (`AtomicTimingCollector`)
/// - Single-threaded collection (Cell-based)
/// - Feature-gated elimination
pub trait TimingCollector: Send + Sync {
    /// Record a duration for the given metric.
    fn record(&self, metric: TimingMetric, duration: Duration);

    /// Get the current value for a metric.
    fn get(&self, metric: TimingMetric) -> Duration;

    /// Reset all metrics to zero.
    fn reset(&self);
}

/// No-op implementation for when timing is disabled.
///
/// All methods compile to nothing, providing zero runtime overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct NullTimingCollector;

impl TimingCollector for NullTimingCollector {
    #[inline(always)]
    fn record(&self, _metric: TimingMetric, _duration: Duration) {}

    #[inline(always)]
    fn get(&self, _metric: TimingMetric) -> Duration {
        Duration::ZERO
    }

    #[inline(always)]
    fn reset(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_collector_is_noop() {
        let collector = NullTimingCollector;

        // Record should do nothing
        collector.record(TimingMetric::ForwardSolver, Duration::from_secs(100));

        // Get should always return zero
        assert_eq!(collector.get(TimingMetric::ForwardSolver), Duration::ZERO);

        // Reset should do nothing (no panic)
        collector.reset();
    }

    #[test]
    fn test_null_collector_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NullTimingCollector>();
    }
}
