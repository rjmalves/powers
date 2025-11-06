// Monotonicity assertions for SDDP algorithm validation
//
// These assertions verify critical SDDP properties like monotonic lower bounds.

/// Assert that a sequence of values is monotonically non-decreasing
///
/// This is a **fundamental SDDP property**: lower bounds must never decrease.
/// Small violations (< tolerance) are allowed due to LP solver numerical errors.
///
/// # Arguments
/// - `values`: Sequence to check (typically lower bounds)
/// - `tolerance`: Maximum allowed decrease (typically 1e-6 for LP solvers)
///
/// # Panics
/// If any value decreases by more than tolerance
///
/// # Example
/// ```
/// use powers_rs::tests::utils::assert_monotonic_non_decreasing;
///
/// let lower_bounds = vec![100.0, 105.0, 105.2, 110.0];
/// assert_monotonic_non_decreasing(&lower_bounds, 1e-6);
/// ```
#[track_caller]
pub fn assert_monotonic_non_decreasing(values: &[f64], tolerance: f64) {
    for (i, window) in values.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];

        if curr < prev - tolerance {
            panic!(
                "Monotonicity violated at index {}: {:.10} -> {:.10} (decrease: {:.10}, tolerance: {:.10})",
                i + 1,
                prev,
                curr,
                prev - curr,
                tolerance
            );
        }
    }
}

/// Assert that a sequence is strictly monotonic (always increasing)
///
/// Useful for testing strictly improving algorithms or scenario costs.
///
/// # Arguments
/// - `values`: Sequence to check
/// - `min_increase`: Minimum required increase per step
///
/// # Panics
/// If any consecutive values don't increase by at least min_increase
///
/// # Example
/// ```
/// let costs = vec![10.0, 15.0, 21.0, 30.0];
/// assert_monotonic_increasing(&costs, 5.0);
/// ```
#[track_caller]
pub fn assert_monotonic_increasing(values: &[f64], min_increase: f64) {
    for (i, window) in values.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];
        let increase = curr - prev;

        if increase < min_increase {
            panic!(
                "Insufficient increase at index {}: {:.10} -> {:.10} (increase: {:.10}, minimum: {:.10})",
                i + 1,
                prev,
                curr,
                increase,
                min_increase
            );
        }
    }
}

/// Assert that a sequence is monotonically non-increasing
///
/// Useful for testing gap convergence or decreasing cost-to-go functions.
///
/// # Arguments
/// - `values`: Sequence to check
/// - `tolerance`: Maximum allowed increase
///
/// # Panics
/// If any value increases by more than tolerance
#[track_caller]
pub fn assert_monotonic_non_increasing(values: &[f64], tolerance: f64) {
    for (i, window) in values.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];

        if curr > prev + tolerance {
            panic!(
                "Monotonic non-increasing violated at index {}: {:.10} -> {:.10} (increase: {:.10}, tolerance: {:.10})",
                i + 1,
                prev,
                curr,
                curr - prev,
                tolerance
            );
        }
    }
}

/// Check if a sequence is monotonic non-decreasing (without panicking)
///
/// Returns true if monotonic within tolerance, false otherwise.
/// Useful for conditional logic in tests.
///
/// # Arguments
/// - `values`: Sequence to check
/// - `tolerance`: Maximum allowed decrease
///
/// # Returns
/// `true` if monotonic, `false` otherwise
pub fn is_monotonic_non_decreasing(values: &[f64], tolerance: f64) -> bool {
    values.windows(2).all(|w| w[1] >= w[0] - tolerance)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_non_decreasing_pass() {
        let values = vec![1.0, 2.0, 2.0, 3.0, 3.5];
        assert_monotonic_non_decreasing(&values, 1e-10);
    }

    #[test]
    fn test_monotonic_non_decreasing_with_tolerance() {
        // Small decrease within tolerance should pass
        let values = vec![1.0, 2.0, 2.0 - 5e-7, 3.0];
        assert_monotonic_non_decreasing(&values, 1e-6);
    }

    #[test]
    #[should_panic(expected = "Monotonicity violated")]
    fn test_monotonic_non_decreasing_fail() {
        let values = vec![1.0, 2.0, 1.5, 3.0];
        assert_monotonic_non_decreasing(&values, 1e-10);
    }

    #[test]
    fn test_monotonic_increasing_pass() {
        let values = vec![1.0, 6.0, 11.0, 16.0];
        assert_monotonic_increasing(&values, 5.0);
    }

    #[test]
    #[should_panic(expected = "Insufficient increase")]
    fn test_monotonic_increasing_fail() {
        let values = vec![1.0, 6.0, 10.0, 16.0]; // Only 4.0 increase in third step
        assert_monotonic_increasing(&values, 5.0);
    }

    #[test]
    fn test_monotonic_non_increasing_pass() {
        let values = vec![10.0, 8.0, 8.0, 5.0, 2.0];
        assert_monotonic_non_increasing(&values, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Monotonic non-increasing violated")]
    fn test_monotonic_non_increasing_fail() {
        let values = vec![10.0, 8.0, 9.0, 5.0];
        assert_monotonic_non_increasing(&values, 1e-10);
    }

    #[test]
    fn test_is_monotonic_check() {
        let monotonic = vec![1.0, 2.0, 2.0, 3.0];
        assert!(is_monotonic_non_decreasing(&monotonic, 1e-10));

        let not_monotonic = vec![1.0, 2.0, 1.5, 3.0];
        assert!(!is_monotonic_non_decreasing(&not_monotonic, 1e-10));
    }

    #[test]
    fn test_empty_and_single_element() {
        // Empty and single element sequences are trivially monotonic
        assert_monotonic_non_decreasing(&[], 1e-10);
        assert_monotonic_non_decreasing(&[42.0], 1e-10);
        assert!(is_monotonic_non_decreasing(&[], 1e-10));
        assert!(is_monotonic_non_decreasing(&[42.0], 1e-10));
    }
}
