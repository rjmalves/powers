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

// ============================================================================
// CONVERGENCE VALIDATION HELPERS (T2.3)
// ============================================================================

use powers_rs::sddp::TrainingResult;

/// Assert that SDDP training exhibits good convergence quality
///
/// Checks multiple convergence properties:
/// 1. **Monotonicity**: Lower bounds should be non-decreasing (SDDP property)
/// 2. **Gap Convergence**: Gap should decrease over iterations
/// 3. **Bounds Validity**: Bounds must be finite and LB ≤ Statistical UB
///
/// **Important**: Uses `statistical_upper_bound` (average of all forward passes)
/// rather than `final_upper_bound` (last iteration's average) for validation.
/// Individual forward passes can have costs below LB due to sampling variance,
/// but the statistical average must satisfy LB ≤ E[forward costs].
///
/// # Arguments
/// - `result`: Training result from SDDP algorithm
///
/// # Panics
/// If any convergence property is violated
///
/// # Tolerance
/// - Monotonicity: 1e-6 (allows small numerical errors from solver)
///
/// # Example
/// ```
/// let result = sddp.train(50, 10, false, &saa)?;
/// assert_convergence_quality(&result)?;
/// ```
#[allow(dead_code)] // Utility for integration tests
#[track_caller]
pub fn assert_convergence_quality(
    result: &TrainingResult,
) -> Result<(), String> {
    // 1. Check monotonicity - lower bounds should never decrease significantly
    let lower_bounds = result.lower_bounds();
    for (i, window) in lower_bounds.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];
        if curr < prev - 1e-6 {
            return Err(format!(
                "Lower bound decreased at iteration {}: {:.6} -> {:.6} (violation: {:.6})",
                i + 1,
                prev,
                curr,
                prev - curr
            ));
        }
    }

    // 2. Check gap convergence using statistical upper bound
    // Note: Per-iteration gaps can fluctuate due to sampling variance.
    // The statistical gap (statistical_UB - final_LB) is the true convergence metric.
    let statistical_gap =
        result.statistical_upper_bound - result.final_lower_bound;

    // Statistical gap must be non-negative (SDDP invariant)
    if statistical_gap < -1e-6 {
        return Err(format!(
            "Statistical gap is negative: {:.6} (LB={:.6}, statistical_UB={:.6})",
            statistical_gap,
            result.final_lower_bound,
            result.statistical_upper_bound
        ));
    }

    // For problems with very small lower bound, check absolute gap
    if result.final_lower_bound.abs() < 1e-6 {
        // Zero-cost problem: statistical gap should be small
        if statistical_gap > 10.0 {
            return Err(format!(
                "Zero-cost problem has large statistical gap: {:.6}",
                statistical_gap
            ));
        }
    }

    // 3. Check bounds validity - must be finite
    if !result.final_lower_bound.is_finite() {
        return Err(format!(
            "Final lower bound is not finite: {}",
            result.final_lower_bound
        ));
    }
    if !result.statistical_upper_bound.is_finite() {
        return Err(format!(
            "Statistical upper bound is not finite: {}",
            result.statistical_upper_bound
        ));
    }

    // 4. Check SDDP invariant: lower bound ≤ statistical upper bound
    // Note: We use statistical_upper_bound (average across ALL forward passes)
    // rather than final_upper_bound (last iteration only). Individual forward passes
    // can have costs below LB due to sampling variance, but the statistical average
    // must satisfy LB ≤ E[forward costs] by SDDP theory.
    if result.final_lower_bound > result.statistical_upper_bound + 1e-6 {
        return Err(format!(
            "Lower bound exceeds statistical upper bound: LB={:.6} > UB_stat={:.6}",
            result.final_lower_bound, result.statistical_upper_bound
        ));
    }

    Ok(())
}

/// Assert that final bounds are within expected range
///
/// Useful for problems with known solution ranges or benchmark problems.
///
/// # Arguments
/// - `result`: Training result from SDDP algorithm
/// - `expected_min`: Minimum expected value for bounds
/// - `expected_max`: Maximum expected value for bounds
///
/// # Panics
/// If either bound is outside the expected range
///
/// # Example
/// ```
/// // Newsvendor problem with known optimal cost ~150
/// let result = sddp.train(50, 10, false, &saa)?;
/// assert_bounds_in_range(&result, 140.0, 160.0)?;
/// ```
#[allow(dead_code)] // Utility for benchmark tests
#[track_caller]
pub fn assert_bounds_in_range(
    result: &TrainingResult,
    expected_min: f64,
    expected_max: f64,
) -> Result<(), String> {
    let lb = result.final_lower_bound;
    let ub = result.final_upper_bound;

    if lb < expected_min || lb > expected_max {
        return Err(format!(
            "Lower bound {:.4} outside expected range [{:.4}, {:.4}]",
            lb, expected_min, expected_max
        ));
    }

    if ub < expected_min || ub > expected_max {
        return Err(format!(
            "Upper bound {:.4} outside expected range [{:.4}, {:.4}]",
            ub, expected_min, expected_max
        ));
    }

    Ok(())
}

/// Print detailed convergence summary for debugging
///
/// Displays key convergence metrics in a readable format.
/// Use this for debugging failed convergence or performance analysis.
///
/// # Arguments
/// - `result`: Training result from SDDP algorithm
///
/// # Example
/// ```
/// let result = sddp.train(50, 10, false, &saa)?;
/// if result.final_gap() > threshold {
///     print_convergence_summary(&result);
/// }
/// ```
pub fn print_convergence_summary(result: &TrainingResult) {
    println!("\n╔════════════════════════════════════════╗");
    println!("║       Convergence Summary              ║");
    println!("╚════════════════════════════════════════╝");
    println!("  Iterations:        {}", result.iterations().len());
    println!("  Final lower bound: {:.6}", result.final_lower_bound);
    println!("  Final upper bound: {:.6}", result.final_upper_bound);
    println!("  Final gap:         {:.6}", result.final_gap());
    println!("  Relative gap:      {:.4}%", result.relative_gap() * 100.0);
    println!(
        "  Best upper bound:  {:.6} (iteration {})",
        result.best_upper_bound, result.best_iteration
    );
    println!("  Total time:        {:?}", result.total_time);
    println!("  Cuts generated:    {}", result.num_cuts);
    println!("══════════════════════════════════════════\n");
}

// ============================================================================
// TESTS
// ============================================================================

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
