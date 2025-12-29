//! RAII timing guard for zero-cost timing instrumentation.

use std::cell::Cell;
use std::time::Duration;
#[cfg(feature = "timing")]
use std::time::Instant;

/// Zero-cost timing guard that records duration on drop.
///
/// When the `timing` feature is disabled, this compiles to nothing.
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
    #[cfg(feature = "timing")]
    start: Instant,
    #[cfg(feature = "timing")]
    target: &'a Cell<Duration>,
    #[cfg(not(feature = "timing"))]
    _phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a> TimingGuard<'a> {
    /// Create a new timing guard that will add elapsed time to `target` on drop.
    ///
    /// When `timing` feature is disabled, this is a no-op.
    #[inline(always)]
    #[cfg(feature = "timing")]
    pub fn new(target: &'a Cell<Duration>) -> Self {
        Self {
            start: Instant::now(),
            target,
        }
    }

    /// Create a new timing guard (no-op when timing feature is disabled).
    #[inline(always)]
    #[cfg(not(feature = "timing"))]
    #[allow(clippy::unused_self)]
    pub fn new(_target: &'a Cell<Duration>) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "timing")]
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
/// When the `timing` feature is disabled, this compiles to nothing.
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

    #[cfg(feature = "timing")]
    use std::thread;

    #[test]
    #[cfg(feature = "timing")]
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
    #[cfg(feature = "timing")]
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
    #[cfg(feature = "timing")]
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
