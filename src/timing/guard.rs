//! RAII timing guard for timing instrumentation.

use std::cell::Cell;
use std::time::{Duration, Instant};

/// Timing guard that records duration on drop.
///
/// The guard accumulates elapsed time into the target `Cell<Duration>`,
/// adding to any existing value rather than replacing it.
///
/// # Example
///
/// ```ignore
/// use std::cell::Cell;
/// use std::time::Duration;
/// use powers_rs::timing::TimingGuard;
///
/// let target = Cell::new(Duration::ZERO);
/// {
///     let _guard = TimingGuard::new(&target);
///     // ... do work ...
/// }
/// println!("Elapsed: {:?}", target.get());
/// ```
pub struct TimingGuard<'a> {
    start: Instant,
    target: &'a Cell<Duration>,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard that will add elapsed time to `target` on drop.
    #[inline(always)]
    pub fn new(target: &'a Cell<Duration>) -> Self {
        Self {
            start: Instant::now(),
            target,
        }
    }
}

impl Drop for TimingGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.target.set(self.target.get() + elapsed);
    }
}

/// Macro for clean timing scope creation.
///
/// Creates a `TimingGuard` that accumulates elapsed time into the specified field.
///
/// # Usage
///
/// ```ignore
/// use powers_rs::time_scope;
///
/// struct Timing {
///     operation: Cell<Duration>,
/// }
///
/// let timing = Timing { operation: Cell::new(Duration::ZERO) };
/// {
///     time_scope!(timing, operation);
///     // ... do work ...
/// }
/// ```
#[macro_export]
macro_rules! time_scope {
    ($timing:expr, $field:ident) => {
        let _guard = $crate::timing::TimingGuard::new(&$timing.$field);
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::thread;

    #[test]
    fn test_timing_guard_accumulates() {
        let target = Cell::new(Duration::ZERO);

        {
            let _guard = TimingGuard::new(&target);
            thread::sleep(Duration::from_millis(10));
        }

        let elapsed = target.get();
        assert!(elapsed >= Duration::from_millis(10));
        assert!(elapsed < Duration::from_millis(100)); // Reasonable upper bound
    }

    #[test]
    fn test_timing_guard_adds_to_existing() {
        let target = Cell::new(Duration::from_millis(100));

        {
            let _guard = TimingGuard::new(&target);
            thread::sleep(Duration::from_millis(10));
        }

        let elapsed = target.get();
        assert!(elapsed >= Duration::from_millis(110));
    }

    #[test]
    fn test_multiple_guards_accumulate() {
        let target = Cell::new(Duration::ZERO);

        for _ in 0..3 {
            let _guard = TimingGuard::new(&target);
            thread::sleep(Duration::from_millis(5));
        }

        let elapsed = target.get();
        assert!(elapsed >= Duration::from_millis(15));
    }

    #[test]
    fn test_timing_guard_compiles_without_feature() {
        // This test verifies the code compiles in both modes
        let target = Cell::new(Duration::ZERO);
        let _guard = TimingGuard::new(&target);
        // In non-timing mode, this should be a no-op
    }
}

// T-007: Integration tests with new timing types
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_guard_with_trajectory_timing() {
        use crate::timing::TrajectoryTiming;
        use std::thread;
        use std::time::Duration;

        let timing = TrajectoryTiming::new();

        {
            let _guard = TimingGuard::new(&timing.model_preprocessing);
            thread::sleep(Duration::from_millis(5));
        }

        let elapsed = timing.model_preprocessing.get();
        assert!(elapsed >= Duration::from_millis(5));
        assert!(elapsed < Duration::from_millis(50)); // Reasonable upper bound
    }

    #[test]
    fn test_guard_accumulates() {
        use crate::timing::TrajectoryTiming;
        use std::thread;
        use std::time::Duration;

        let timing = TrajectoryTiming::new();

        for _ in 0..3 {
            let _guard = TimingGuard::new(&timing.solver);
            thread::sleep(Duration::from_millis(2));
        }

        let elapsed = timing.solver.get();
        assert!(elapsed >= Duration::from_millis(6));
    }

    #[test]
    fn test_guard_with_forward_timing() {
        use crate::timing::NewForwardTiming;
        use std::thread;
        use std::time::Duration;

        let timing = NewForwardTiming::new(2);

        {
            let _guard = TimingGuard::new(&timing.preprocessing.saa_sampling);
            thread::sleep(Duration::from_millis(5));
        }

        assert!(
            timing.preprocessing.saa_sampling.get() >= Duration::from_millis(5)
        );
    }

    #[test]
    fn test_guard_with_iteration_timing() {
        use crate::timing::NewIterationTiming;
        use std::thread;
        use std::time::Duration;

        let timing = NewIterationTiming::new(2);

        {
            let _guard = TimingGuard::new(&timing.model_allocation);
            thread::sleep(Duration::from_millis(5));
        }

        assert!(timing.model_allocation.get() >= Duration::from_millis(5));
    }
}
