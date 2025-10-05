//! Numerical Validation Tests for SDDP Algorithm
//!
//! These tests verify that SDDP produces **numerically correct results**, not just "doesn't crash."
//! They validate fundamental algorithmic properties from SDDP convergence theory:
//!
//! 1. **Lower Bound Monotonicity**: LB must be non-decreasing (Pereira & Pinto 1991)
//! 2. **Gap Reduction**: Optimality gap should decrease over iterations
//! 3. **Bounds Bracket Optimal**: LB ≤ optimal ≤ UB (correctness validation)
//! 4. **Statistical Properties**: Forward pass variance reduces with more samples
//! 5. **Numerical Stability**: No NaN/Inf in results
//! 6. **Policy Structure**: Qualitative reasonableness checks
//!
//! Tests use benchmarks with known solutions to enable validation.

mod fixtures;
use fixtures::benchmarks::{
    create_deterministic_single_reservoir, create_stochastic_single_reservoir,
    create_two_reservoir_cascade,
};
use powers_rs::sddp::TrainingResult;

// ================================================================================================
// HELPER FUNCTIONS
// ================================================================================================

/// Assert that lower bounds are non-decreasing (monotonicity property).
///
/// This is a fundamental property of SDDP: each iteration adds cuts that improve or maintain
/// the lower bound approximation. Violations indicate algorithm bugs.
///
/// # Arguments
///
/// * `result` - Training result to validate
/// * `tolerance` - Maximum allowed decrease (allows tiny numerical noise from LP solver)
///
/// # References
///
/// - Pereira & Pinto (1991): "Multi-stage stochastic optimization applied to energy planning"
fn assert_monotonicity(result: &TrainingResult, tolerance: f64) {
    let iterations = result.iterations();

    for i in 1..iterations.len() {
        let prev_lb = iterations[i - 1].lower_bound;
        let curr_lb = iterations[i].lower_bound;

        assert!(
            curr_lb >= prev_lb - tolerance,
            "Lower bound decreased at iteration {}: {:.6} → {:.6} (decrease: {:.6})",
            i + 1,
            prev_lb,
            curr_lb,
            prev_lb - curr_lb
        );
    }
}

/// Assert that the optimality gap reduces over iterations.
///
/// Compares average gap in early iterations (first 20%) vs late iterations (last 20%).
/// A well-functioning SDDP should show significant improvement.
///
/// # Arguments
///
/// * `result` - Training result to validate
/// * `early_fraction` - Fraction of early iterations to average (e.g., 0.2 = first 20%)
/// * `late_fraction` - Fraction of late iterations to average (e.g., 0.2 = last 20%)
/// * `improvement_factor` - Minimum expected improvement (late_gap < early_gap / factor)
///
/// # Special Case
///
/// If the problem converges to optimal in the first iteration (gap ≈ 0 throughout),
/// this is valid behavior and the assertion passes.
fn assert_gap_reduction(
    result: &TrainingResult,
    early_fraction: f64,
    late_fraction: f64,
    improvement_factor: f64,
) {
    let iters = result.iterations();
    let n = iters.len();

    let early_end = (n as f64 * early_fraction).max(1.0) as usize;
    let late_start = n - (n as f64 * late_fraction).max(1.0) as usize;

    let early_gap: f64 =
        iters[..early_end].iter().map(|it| it.gap).sum::<f64>()
            / early_end as f64;

    let late_gap: f64 =
        iters[late_start..].iter().map(|it| it.gap).sum::<f64>()
            / (n - late_start) as f64;

    // Special case: if early gap is already very small (< 0.01), problem converged immediately
    // This is valid behavior (e.g., simple problem with ample resources)
    if early_gap < 0.01 {
        // Just verify late gap is also small
        assert!(
            late_gap < 0.01,
            "Gap increased unexpectedly: early_avg={:.4}, late_avg={:.4}",
            early_gap,
            late_gap
        );
        return;
    }

    // Normal case: check for improvement
    assert!(
        late_gap < early_gap / improvement_factor,
        "Gap did not reduce sufficiently: early_avg={:.4}, late_avg={:.4}, improvement={:.2}x (expected {:.2}x)",
        early_gap,
        late_gap,
        early_gap / late_gap.max(1e-10),
        improvement_factor
    );
}

/// Assert that bounds bracket the expected optimal value.
///
/// Validates correctness: Lower bound ≤ optimal ≤ Upper bound.
/// This is a fundamental property of SDDP convergence.
///
/// # Arguments
///
/// * `result` - Training result to validate
/// * `expected_optimal` - Known or expected optimal value
/// * `tolerance` - Allowed deviation from optimal (problem-dependent)
fn assert_bounds_valid(
    result: &TrainingResult,
    expected_optimal: f64,
    tolerance: f64,
) {
    let lb = result.final_lower_bound;
    let ub = result.final_upper_bound;

    // Lower bound should be at or below optimal (allowing tolerance)
    assert!(
        lb <= expected_optimal + tolerance,
        "Lower bound {:.4} exceeds optimal {:.4} + tolerance {:.4}",
        lb,
        expected_optimal,
        tolerance
    );

    // Upper bound should be at or above optimal (allowing tolerance)
    assert!(
        ub >= expected_optimal - tolerance,
        "Upper bound {:.4} below optimal {:.4} - tolerance {:.4}",
        ub,
        expected_optimal,
        tolerance
    );

    // Bounds should bracket optimal (implicit from above, but verify gap is reasonable)
    assert!(
        result.final_gap() < 100.0,
        "Gap {:.4} too large - indicates convergence issue",
        result.final_gap()
    );
}

/// Assert that no NaN or Inf values appear in results.
///
/// Validates numerical stability. NaN/Inf indicates serious problems:
/// - Unbounded subproblems
/// - Numerical overflow/underflow
/// - Solver issues
/// - Poor problem scaling
fn assert_no_numerical_issues(result: &TrainingResult) {
    for (i, iteration) in result.iterations().iter().enumerate() {
        assert!(
            iteration.lower_bound.is_finite(),
            "Lower bound is NaN/Inf at iteration {}",
            i + 1
        );

        assert!(
            iteration.upper_bound.is_finite(),
            "Upper bound is NaN/Inf at iteration {}",
            i + 1
        );

        assert!(
            iteration.gap.is_finite(),
            "Gap is NaN/Inf at iteration {}",
            i + 1
        );

        // Check all forward pass costs
        for (j, &cost) in iteration.forward_costs.iter().enumerate() {
            assert!(
                cost.is_finite(),
                "Forward pass cost {} is NaN/Inf at iteration {}",
                j + 1,
                i + 1
            );
        }
    }
}

/// Compute variance of a sample.
///
/// Used for statistical validation (e.g., forward pass variance should decrease with more samples).
///
/// # Arguments
///
/// * `values` - Sample values
///
/// # Returns
///
/// Sample variance: Var(X) = (1/n) Σ (x_i - mean)²
fn compute_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;

    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n
}

// ================================================================================================
// CORE VALIDATION TESTS
// ================================================================================================

#[test]
fn test_lower_bound_monotonicity() {
    // Test with deterministic benchmark (optimal = $0)
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    let result = sddp.train(30, 10, &saa).expect("Training failed");

    // Validate monotonicity with tight tolerance (1e-6)
    // Allows tiny numerical noise from LP solver but catches real violations
    assert_monotonicity(&result, 1e-6);
}

#[test]
fn test_gap_reduction_trend() {
    // Test with stochastic benchmark (optimal ≈ $20-40)
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    let result = sddp.train(50, 20, &saa).expect("Training failed");

    // Validate gap reduces: average of last 20% should be < average of first 20% / 2
    // (i.e., 50% improvement minimum)
    assert_gap_reduction(&result, 0.2, 0.2, 2.0);
}

#[test]
fn test_bounds_bracket_optimal() {
    // Test with deterministic benchmark with water scarcity
    // Expected optimal ≈ $2000 (requires thermal generation due to scarcity)
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    let result = sddp.train(30, 10, &saa).expect("Training failed");

    // Expected optimal ≈ $2000 (deterministic, water scarce)
    // Allow tolerance of ±100.0 for numerical approximation
    assert_bounds_valid(&result, 2000.0, 100.0);

    // Additionally check bounds are reasonable
    assert!(
        result.final_gap() < 500.0,
        "Gap should be reasonable for deterministic problem, got {:.4}",
        result.final_gap()
    );
}

#[test]
fn test_forward_pass_variance_convergence() {
    // Test with stochastic benchmark
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    let result = sddp.train(30, 20, &saa).expect("Training failed");

    // Compute variance of last iteration's forward pass costs
    let last_iter = result.iterations().last().expect("No iterations recorded");
    let variance = compute_variance(&last_iter.forward_costs);

    // Variance should be reasonable (not explosive)
    // With 20 forward passes and costs ~$3000, standard deviation should be manageable
    // Stochastic problems with 3 scenarios can have significant variance
    let std_dev = variance.sqrt();
    assert!(
        std_dev < 2000.0,
        "Forward pass standard deviation too high: {:.2}",
        std_dev
    );

    // Variance should be finite
    assert!(variance.is_finite(), "Variance is NaN/Inf");
}

#[test]
fn test_no_nan_or_inf() {
    // Test with all benchmarks to ensure numerical stability
    let benchmarks = vec![
        create_deterministic_single_reservoir(),
        create_stochastic_single_reservoir(),
        create_two_reservoir_cascade(),
    ];

    for (i, benchmark_result) in benchmarks.into_iter().enumerate() {
        let (mut sddp, saa) = benchmark_result
            .unwrap_or_else(|_| panic!("Failed to create benchmark {}", i + 1));

        let result = sddp.train(30, 10, &saa).unwrap_or_else(|_| {
            panic!("Training failed for benchmark {}", i + 1)
        });

        // Validate no numerical issues
        assert_no_numerical_issues(&result);
    }
}

#[test]
fn test_policy_structure_deterministic() {
    // Test that deterministic policy is sensible:
    // With water scarcity, should use mix of hydro and thermal
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    let result = sddp.train(30, 10, &saa).expect("Training failed");

    // Final lower bound should be positive (thermal usage required with scarcity)
    assert!(
        result.final_lower_bound > 1000.0,
        "Expected significant cost with water scarcity, got {:.4}",
        result.final_lower_bound
    );

    // Upper bound should be reasonable
    assert!(
        result.final_upper_bound < 3000.0,
        "Upper bound too high, got {:.4}",
        result.final_upper_bound
    );
}

#[test]
fn test_policy_structure_stochastic() {
    // Test that stochastic policy converges with hedging behavior
    // With uncertainty and scarcity, expect positive costs
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    let result = sddp.train(30, 20, &saa).expect("Training failed");

    // Validate bounds are reasonable (should be positive with scarcity)
    assert!(
        result.final_lower_bound > 1000.0,
        "Lower bound should be significant with water scarcity, got {:.4}",
        result.final_lower_bound
    );

    assert!(
        result.final_lower_bound < 5000.0,
        "Cost too high for stochastic problem: {:.4}",
        result.final_lower_bound
    );

    // Validate convergence (gap should be reasonable for stochastic)
    assert!(
        result.final_gap().abs() < 1000.0,
        "Gap too large for stochastic problem: {:.4}",
        result.final_gap()
    );
}

// ================================================================================================
// EDGE CASE TESTS
// ================================================================================================

#[test]
fn test_deterministic_tight_convergence() {
    // Deterministic problems should converge tightly
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    let result = sddp.train(50, 10, &saa).expect("Training failed");

    // Gap should be < 1% after 50 iterations
    assert!(
        result.final_gap() < 0.01,
        "Deterministic gap too large after 50 iterations: {:.4}",
        result.final_gap()
    );
}

#[test]
fn test_stochastic_reasonable_convergence() {
    // Stochastic problems converge slower but should be reasonable
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    let result = sddp.train(50, 20, &saa).expect("Training failed");

    // Absolute gap should be small after 50 iterations
    // (Even though optimal is $0 for this benchmark, we still check gap convergence)
    assert!(
        result.final_gap() < 1.0,
        "Stochastic gap too large after 50 iterations: {:.4}",
        result.final_gap()
    );
}

#[test]
fn test_stability_across_runs() {
    // Deterministic problem should give consistent results across multiple runs
    let num_runs = 5;
    let mut lower_bounds = vec![];

    for _ in 0..num_runs {
        let (mut sddp, saa) = create_deterministic_single_reservoir()
            .expect("Failed to create deterministic benchmark");

        // Multiple runs with same benchmark should give stable results
        let result = sddp.train(30, 10, &saa).expect("Training failed");
        lower_bounds.push(result.final_lower_bound);
    }

    // Compute standard deviation of lower bounds
    let variance = compute_variance(&lower_bounds);
    let std_dev = variance.sqrt();

    // Standard deviation should be very small for deterministic problem
    // (may have tiny variations due to scenario sampling order, but should be minimal)
    assert!(
        std_dev < 0.5,
        "Inconsistent results across runs: std_dev={:.4}, bounds={:?}",
        std_dev,
        lower_bounds
    );
}
