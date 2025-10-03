// Custom assertions for testing numerical and physical properties
//
// These assertions provide clear error messages and handle floating-point
// comparisons with appropriate tolerances.

/// Default tolerance for floating-point comparisons
///
/// Set to 1e-10 for typical numerical precision in LP solvers.
/// Can be overridden in individual assertions.
pub const DEFAULT_TOLERANCE: f64 = 1e-10;

/// Assert that two floating-point values are approximately equal
///
/// # Arguments
/// - `a`: First value
/// - `b`: Second value
/// - `tolerance`: Maximum absolute difference allowed
///
/// # Panics
/// If |a - b| > tolerance
///
/// # Example
/// ```
/// use powers_rs::tests::utils::assert_float_approx_eq;
/// assert_float_approx_eq!(1.0, 1.0 + 1e-11, 1e-10);
/// ```
#[track_caller]
pub fn assert_float_approx_eq(a: f64, b: f64, tolerance: f64) {
    let diff = (a - b).abs();
    assert!(
        diff <= tolerance,
        "Values not approximately equal: {} ≉ {} (diff: {}, tolerance: {})",
        a,
        b,
        diff,
        tolerance
    );
}

/// Assert that two vectors are approximately equal element-wise
///
/// # Arguments
/// - `a`: First vector
/// - `b`: Second vector
/// - `tolerance`: Maximum absolute difference allowed per element
///
/// # Panics
/// If vectors have different lengths or any element pair exceeds tolerance
///
/// # Performance Note
/// Uses iterator for zero-overhead iteration. No allocations.
#[track_caller]
pub fn assert_vec_approx_eq(a: &[f64], b: &[f64], tolerance: f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "Vectors have different lengths: {} vs {}",
        a.len(),
        b.len()
    );

    for (i, (&a_i, &b_i)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (a_i - b_i).abs();
        assert!(
            diff <= tolerance,
            "Vectors differ at index {}: {} ≉ {} (diff: {}, tolerance: {})",
            i,
            a_i,
            b_i,
            diff,
            tolerance
        );
    }
}

/// Assert that a state vector is within specified bounds
///
/// # Arguments
/// - `state`: State vector to check
/// - `lower_bounds`: Lower bounds for each state variable
/// - `upper_bounds`: Upper bounds for each state variable
/// - `tolerance`: Tolerance for bound violations (allows for numerical errors)
///
/// # Panics
/// If state is outside bounds beyond tolerance
///
/// # Example
/// ```
/// use powers_rs::tests::utils::assert_state_within_bounds;
///
/// let state = vec![50.0, 75.0];
/// let lower = vec![0.0, 0.0];
/// let upper = vec![100.0, 100.0];
/// assert_state_within_bounds(&state, &lower, &upper, 1e-6);
/// ```
#[track_caller]
pub fn assert_state_within_bounds(
    state: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    tolerance: f64,
) {
    assert_eq!(
        state.len(),
        lower_bounds.len(),
        "State dimension doesn't match lower bounds dimension"
    );
    assert_eq!(
        state.len(),
        upper_bounds.len(),
        "State dimension doesn't match upper bounds dimension"
    );

    for i in 0..state.len() {
        let val = state[i];
        let lower = lower_bounds[i];
        let upper = upper_bounds[i];

        assert!(
            val >= lower - tolerance,
            "State[{}] = {} is below lower bound {} (tolerance: {})",
            i,
            val,
            lower,
            tolerance
        );

        assert!(
            val <= upper + tolerance,
            "State[{}] = {} is above upper bound {} (tolerance: {})",
            i,
            val,
            upper,
            tolerance
        );
    }
}

/// Assert that a value is within a range
///
/// # Arguments
/// - `value`: Value to check
/// - `min`: Minimum allowed value (inclusive)
/// - `max`: Maximum allowed value (inclusive)
///
/// # Panics
/// If value is outside [min, max]
#[track_caller]
pub fn assert_in_range(value: f64, min: f64, max: f64) {
    assert!(
        value >= min && value <= max,
        "Value {} is not in range [{}, {}]",
        value,
        min,
        max
    );
}

/// Assert that all values in a vector are finite (not NaN or Inf)
///
/// # Arguments
/// - `values`: Vector to check
/// - `name`: Name for error messages
///
/// # Panics
/// If any value is NaN or Inf
#[track_caller]
pub fn assert_all_finite(values: &[f64], name: &str) {
    for (i, &val) in values.iter().enumerate() {
        assert!(
            val.is_finite(),
            "{} at index {} is not finite: {}",
            name,
            i,
            val
        );
    }
}

/// Assert that a vector is not empty
///
/// # Arguments
/// - `vec`: Vector to check
/// - `name`: Name for error message
///
/// # Panics
/// If vector is empty
#[track_caller]
pub fn assert_not_empty<T>(vec: &[T], name: &str) {
    assert!(!vec.is_empty(), "{} is empty", name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_approx_eq() {
        assert_float_approx_eq(1.0, 1.0, 1e-10);
        assert_float_approx_eq(1.0, 1.0 + 1e-11, 1e-10);
        assert_float_approx_eq(1.0, 1.0 - 1e-11, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Values not approximately equal")]
    fn test_float_approx_eq_fails() {
        assert_float_approx_eq(1.0, 2.0, 1e-10);
    }

    #[test]
    fn test_vec_approx_eq() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0 + 1e-11, 2.0 - 1e-11, 3.0];
        assert_vec_approx_eq(&a, &b, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Vectors differ at index")]
    fn test_vec_approx_eq_fails() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 4.0];
        assert_vec_approx_eq(&a, &b, 1e-10);
    }

    #[test]
    #[should_panic(expected = "Vectors have different lengths")]
    fn test_vec_approx_eq_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_vec_approx_eq(&a, &b, 1e-10);
    }

    #[test]
    fn test_state_within_bounds() {
        let state = vec![50.0, 75.0];
        let lower = vec![0.0, 0.0];
        let upper = vec![100.0, 100.0];
        assert_state_within_bounds(&state, &lower, &upper, 1e-6);
    }

    #[test]
    #[should_panic(expected = "is below lower bound")]
    fn test_state_below_bound_fails() {
        let state = vec![50.0, -1.0]; // Second value below bound
        let lower = vec![0.0, 0.0];
        let upper = vec![100.0, 100.0];
        assert_state_within_bounds(&state, &lower, &upper, 1e-6);
    }

    #[test]
    #[should_panic(expected = "is above upper bound")]
    fn test_state_above_bound_fails() {
        let state = vec![50.0, 101.0]; // Second value above bound
        let lower = vec![0.0, 0.0];
        let upper = vec![100.0, 100.0];
        assert_state_within_bounds(&state, &lower, &upper, 1e-6);
    }

    #[test]
    fn test_in_range() {
        assert_in_range(50.0, 0.0, 100.0);
        assert_in_range(0.0, 0.0, 100.0); // Boundary
        assert_in_range(100.0, 0.0, 100.0); // Boundary
    }

    #[test]
    #[should_panic(expected = "is not in range")]
    fn test_in_range_fails() {
        assert_in_range(101.0, 0.0, 100.0);
    }

    #[test]
    fn test_all_finite() {
        let values = vec![1.0, 2.0, 3.0, -1.0, 0.0];
        assert_all_finite(&values, "test values");
    }

    #[test]
    #[should_panic(expected = "is not finite")]
    fn test_all_finite_nan_fails() {
        let values = vec![1.0, f64::NAN, 3.0];
        assert_all_finite(&values, "test values");
    }

    #[test]
    #[should_panic(expected = "is not finite")]
    fn test_all_finite_inf_fails() {
        let values = vec![1.0, f64::INFINITY, 3.0];
        assert_all_finite(&values, "test values");
    }

    #[test]
    fn test_not_empty() {
        let vec = vec![1, 2, 3];
        assert_not_empty(&vec, "test vector");
    }

    #[test]
    #[should_panic(expected = "is empty")]
    fn test_not_empty_fails() {
        let vec: Vec<i32> = vec![];
        assert_not_empty(&vec, "test vector");
    }
}
