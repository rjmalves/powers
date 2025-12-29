//! Thread-safe atomic timing collector.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::collector::TimingCollector;
use super::metrics::TimingMetric;

/// Thread-safe timing collector using atomic operations.
///
/// Suitable for parallel sections where multiple threads record timing.
/// Uses relaxed ordering for performance - timing values don't need
/// strict synchronization guarantees.
pub struct AtomicTimingCollector {
    /// Store nanoseconds as u64 for atomic operations.
    metrics: [AtomicU64; TimingMetric::COUNT],
}

impl AtomicTimingCollector {
    /// Create a new atomic timing collector with all metrics at zero.
    pub fn new() -> Self {
        Self {
            metrics: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
}

impl Default for AtomicTimingCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingCollector for AtomicTimingCollector {
    fn record(&self, metric: TimingMetric, duration: Duration) {
        let nanos = duration.as_nanos() as u64;
        self.metrics[metric.index()].fetch_add(nanos, Ordering::Relaxed);
    }

    fn get(&self, metric: TimingMetric) -> Duration {
        let nanos = self.metrics[metric.index()].load(Ordering::Relaxed);
        Duration::from_nanos(nanos)
    }

    fn reset(&self) {
        for metric in &self.metrics {
            metric.store(0, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_atomic_collector_single_thread() {
        let collector = AtomicTimingCollector::new();

        collector
            .record(TimingMetric::ForwardSolver, Duration::from_millis(100));
        collector
            .record(TimingMetric::ForwardSolver, Duration::from_millis(50));

        assert_eq!(
            collector.get(TimingMetric::ForwardSolver),
            Duration::from_millis(150)
        );
    }

    #[test]
    fn test_atomic_collector_reset() {
        let collector = AtomicTimingCollector::new();

        collector
            .record(TimingMetric::ForwardSolver, Duration::from_millis(100));
        collector
            .record(TimingMetric::BackwardSolver, Duration::from_millis(200));

        collector.reset();

        assert_eq!(collector.get(TimingMetric::ForwardSolver), Duration::ZERO);
        assert_eq!(collector.get(TimingMetric::BackwardSolver), Duration::ZERO);
    }

    #[test]
    fn test_atomic_collector_multiple_threads() {
        use std::sync::Arc;

        let collector = Arc::new(AtomicTimingCollector::new());
        let num_threads = 4;
        let records_per_thread = 100;
        let duration_per_record = Duration::from_micros(10);

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let collector = Arc::clone(&collector);
                thread::spawn(move || {
                    for _ in 0..records_per_thread {
                        collector.record(
                            TimingMetric::ForwardSolver,
                            duration_per_record,
                        );
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let expected =
            Duration::from_micros(10 * num_threads * records_per_thread);
        assert_eq!(collector.get(TimingMetric::ForwardSolver), expected);
    }

    #[test]
    fn test_atomic_collector_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AtomicTimingCollector>();
    }
}
