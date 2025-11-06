// Cut validation utilities for SDDP algorithm testing
//
// Validates Benders cuts satisfy mathematical properties:
// - Cut height at training state equals objective value
// - Storage coefficients are non-positive (marginal value of water)
// - Cut provides valid lower bound

use powers_rs::cut::BendersCut;

/// Assert that a Benders cut is valid at its training state
///
/// **Most critical SDDP property**: A cut's height at the state where it was
/// generated must equal the LP objective value at that state. This is the
/// fundamental correctness check for cut generation.
///
/// # Arguments
/// - `cut`: The Benders cut to validate
/// - `training_state`: State vector where cut was generated
/// - `objective_value`: LP objective value at training_state
/// - `tolerance`: Tolerance for floating-point comparison (typically 1e-4 for LP solvers)
///
/// # Panics
/// If cut height deviates from objective by more than tolerance
///
/// # Mathematical Property
/// For a cut: `θ = rhs + coefficients' * state`
/// At training_state: cut should evaluate to objective_value
///
/// # Example
/// ```ignore
/// use powers_rs::cut::BendersCut;
/// let cut = BendersCut::new(0, vec![-2.0], 100.0, 1, 0);
/// let training_state = vec![10.0];
/// let objective = 80.0; // 100.0 + (-2.0 * 10.0) = 80.0
/// assert_cut_validity(&cut, &training_state, objective, 1e-4);
/// ```
#[track_caller]
pub fn assert_cut_validity(
    cut: &BendersCut,
    training_state: &[f64],
    objective_value: f64,
    tolerance: f64,
) {
    let cut_height = cut.eval_height_at_state(training_state);
    let diff = (cut_height - objective_value).abs();

    if diff > tolerance {
        panic!(
            "Cut is invalid at training state!\n  \
             Cut height: {:.10}\n  \
             Objective:  {:.10}\n  \
             Difference: {:.10} (tolerance: {:.10})\n  \
             This indicates a bug in dual extraction or cut generation.",
            cut_height, objective_value, diff, tolerance
        );
    }
}

/// Assert that storage coefficients in a cut are non-positive
///
/// **Fundamental SDDP property**: ∂V/∂storage ≤ 0 for cost minimization.
/// More water stored should never increase future cost (marginal value of water is non-positive).
///
/// # Arguments
/// - `storage_coefficients`: Cut coefficients corresponding to storage state variables
/// - `tolerance`: Tolerance for numerical errors (typically 1e-8)
///
/// # Panics
/// If any storage coefficient is positive beyond tolerance
///
/// # Mathematical Basis
/// In cost minimization, the cost-to-go function V(storage) is non-increasing in storage:
/// - More water → more flexibility → lower (or equal) expected cost
/// - Therefore: ∂V/∂storage ≤ 0
///
/// # Example
/// ```ignore
/// // Extract storage coefficients from cut
/// let storage_coeffs = &cut.coefficients[0..n_hydros];
/// assert_storage_coefficients_negative(storage_coeffs, 1e-8);
/// ```
#[track_caller]
pub fn assert_storage_coefficients_negative(
    storage_coefficients: &[f64],
    tolerance: f64,
) {
    for (i, &coef) in storage_coefficients.iter().enumerate() {
        if coef > tolerance {
            panic!(
                "Storage coefficient[{}] = {:.10} is positive (tolerance: {:.10})!\n  \
                 This violates ∂V/∂storage ≤ 0 for cost minimization.\n  \
                 Possible causes:\n  \
                 - Incorrect dual extraction (sign error)\n  \
                 - LP formulation error\n  \
                 - Numerical instability in solver",
                i, coef, tolerance
            );
        }
    }
}

/// Assert that a cut provides a valid lower bound
///
/// Checks that cut evaluation at any state provides a value ≤ true objective.
/// In practice, we can only check this at the training state where we know the objective.
///
/// For more thorough validation, use assert_cut_validity which checks exact equality.
///
/// # Arguments
/// - `cut`: The Benders cut
/// - `state`: State to evaluate at
/// - `true_objective`: Known objective at this state
/// - `tolerance`: Tolerance (cut can be slightly above due to numerical errors)
#[track_caller]
pub fn assert_cut_lower_bound(
    cut: &BendersCut,
    state: &[f64],
    true_objective: f64,
    tolerance: f64,
) {
    let cut_value = cut.eval_height_at_state(state);

    // Cut should provide lower bound: cut_value ≤ true_objective + tolerance
    if cut_value > true_objective + tolerance {
        panic!(
            "Cut violates lower bound property!\n  \
             Cut value:       {:.10}\n  \
             True objective:  {:.10}\n  \
             Excess:          {:.10} (tolerance: {:.10})",
            cut_value,
            true_objective,
            cut_value - true_objective,
            tolerance
        );
    }
}

/// Check if storage coefficients are non-positive (without panicking)
///
/// Returns true if all coefficients satisfy ∂V/∂storage ≤ 0, false otherwise.
///
/// # Arguments
/// - `storage_coefficients`: Coefficients to check
/// - `tolerance`: Tolerance for numerical errors
///
/// # Returns
/// `true` if all non-positive, `false` if any positive
pub fn are_storage_coefficients_negative(
    storage_coefficients: &[f64],
    tolerance: f64,
) -> bool {
    storage_coefficients.iter().all(|&c| c <= tolerance)
}

/// Validate cut dimension matches state dimension
///
/// Ensures cut has correct number of coefficients for the state space.
///
/// # Arguments
/// - `cut`: Cut to validate
/// - `expected_dimension`: Expected state dimension
///
/// # Panics
/// If dimensions don't match
#[track_caller]
pub fn assert_cut_dimension(cut: &BendersCut, expected_dimension: usize) {
    let cut_dim = cut.coefficients.len();
    if cut_dim != expected_dimension {
        panic!(
            "Cut dimension mismatch!\n  \
             Cut has {} coefficients\n  \
             Expected {} (state dimension)",
            cut_dim, expected_dimension
        );
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use powers_rs::cut::BendersCut;

    #[test]
    fn test_cut_validity_pass() {
        // Create a cut: θ = 100 + (-2.0) * s
        // At training state s=10: θ = 100 + (-2.0 * 10) = 80
        let cut = BendersCut::new(0, vec![-2.0], 100.0, 1, 0);
        let training_state = vec![10.0];
        let objective = 80.0;

        assert_cut_validity(&cut, &training_state, objective, 1e-4);
    }

    #[test]
    #[should_panic(expected = "Cut is invalid at training state")]
    fn test_cut_validity_fail() {
        let cut = BendersCut::new(0, vec![-2.0], 100.0, 1, 0);
        let training_state = vec![10.0];
        let wrong_objective = 85.0; // Should be 80.0

        assert_cut_validity(&cut, &training_state, wrong_objective, 1e-4);
    }

    #[test]
    fn test_storage_coefficients_negative_pass() {
        let coeffs = vec![-1.0, -2.5, -0.1, 0.0]; // All non-positive
        assert_storage_coefficients_negative(&coeffs, 1e-8);
    }

    #[test]
    #[should_panic(expected = "is positive")]
    fn test_storage_coefficients_negative_fail() {
        let coeffs = vec![-1.0, 0.5, -0.1]; // 0.5 is positive
        assert_storage_coefficients_negative(&coeffs, 1e-8);
    }

    #[test]
    fn test_cut_lower_bound_pass() {
        // Cut: θ = 100 - 2*s
        let cut = BendersCut::new(0, vec![-2.0], 100.0, 1, 0);
        let state = vec![5.0];
        let true_objective = 90.0; // At s=5: 100 + (-2*5) = 90

        assert_cut_lower_bound(&cut, &state, true_objective, 1e-4);
    }

    #[test]
    #[should_panic(expected = "violates lower bound")]
    fn test_cut_lower_bound_fail() {
        // Cut says θ = 90, but true objective is 80
        // This means cut is too optimistic (invalid)
        let cut = BendersCut::new(0, vec![-2.0], 100.0, 1, 0);
        let state = vec![5.0];
        let true_objective = 80.0; // Lower than cut predicts

        assert_cut_lower_bound(&cut, &state, true_objective, 1e-4);
    }

    #[test]
    fn test_are_storage_coefficients_negative_check() {
        assert!(are_storage_coefficients_negative(
            &vec![-1.0, -2.0, 0.0],
            1e-8
        ));
        assert!(!are_storage_coefficients_negative(
            &vec![-1.0, 0.5, 0.0],
            1e-8
        ));
    }

    #[test]
    fn test_cut_dimension() {
        let cut = BendersCut::new(0, vec![-1.0, -2.0, -3.0], 100.0, 1, 0);
        assert_cut_dimension(&cut, 3);
    }

    #[test]
    #[should_panic(expected = "Cut dimension mismatch")]
    fn test_cut_dimension_fail() {
        let cut = BendersCut::new(0, vec![-1.0, -2.0], 100.0, 1, 0);
        assert_cut_dimension(&cut, 3); // Expected 3, got 2
    }
}
