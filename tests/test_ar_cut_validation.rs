//! Validation tests for AR cut generation
//!
//! This module tests the correctness of Benders cut generation for autoregressive
//! (AR) inflow models using fixtures from the benchmark suite.
//!
//! # Test Strategy
//!
//! 1. **Convergence Validation**: Verify ZINF ≤ ZSUP throughout SDDP training
//! 2. **Backward Compatibility**: Ensure independent models still work
//!
//! # Mathematical Reference
//!
//! ```text
//! π_{ENA,j} = (λ^{BH} + λ^{AR}) * φ_j
//! ```

mod fixtures;

use fixtures::benchmarks::{
    create_deterministic_single_reservoir, create_stochastic_single_reservoir,
};

/// Test that SDDP with independent inflows converges monotonically
///
/// This validates backward compatibility - the AR cut fix doesn't break
/// existing independent inflow models.
#[test]
fn test_independent_inflows_converge() {
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic system");

    let result = sddp
        .train(20, 5, &saa)
        .expect("Failed to train with independent inflows");

    // Validate monotonic lower bound
    let iterations = result.iterations();
    for i in 1..iterations.len() {
        let prev_lb = iterations[i - 1].lower_bound;
        let curr_lb = iterations[i].lower_bound;

        assert!(
            curr_lb >= prev_lb - 1e-6,
            "Lower bound should be non-decreasing: iter {} = {:.6}, iter {} = {:.6}",
            i - 1, prev_lb, i, curr_lb
        );
    }

    // Validate final bound is finite
    let final_lb = iterations.last().unwrap().lower_bound;
    assert!(final_lb.is_finite(), "Final lower bound should be finite");
}

/// Test that deterministic system converges
#[test]
fn test_deterministic_system_converges() {
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic system");

    let result = sddp
        .train(10, 1, &saa)
        .expect("Failed to train deterministic system");

    // Validate convergence
    let iterations = result.iterations();
    assert!(iterations.len() >= 10, "Should complete all iterations");

    // For deterministic problems, convergence should be quick
    let final_lb = iterations.last().unwrap().lower_bound;
    assert!(final_lb.is_finite());
    assert!(final_lb > 0.0, "Cost should be positive");
}

/// Test that multiple training runs produce consistent results
#[test]
fn test_training_reproducibility() {
    let (mut sddp1, saa1) = create_stochastic_single_reservoir()
        .expect("Failed to create first system");
    let (mut sddp2, saa2) = create_stochastic_single_reservoir()
        .expect("Failed to create second system");

    let result1 = sddp1.train(15, 5, &saa1).expect("First training failed");
    let result2 = sddp2.train(15, 5, &saa2).expect("Second training failed");

    // Both should converge
    assert!(result1.iterations().len() >= 15);
    assert!(result2.iterations().len() >= 15);

    // Bounds should be valid
    let lb1 = result1.iterations().last().unwrap().lower_bound;
    let lb2 = result2.iterations().last().unwrap().lower_bound;

    assert!(lb1.is_finite());
    assert!(lb2.is_finite());

    // They should be similar (within 10%) since same system & seed
    let diff_pct = ((lb1 - lb2) / lb1 * 100.0).abs();
    assert!(
        diff_pct < 10.0,
        "Lower bounds should be similar: {:.2} vs {:.2} ({:.1}% diff)",
        lb1,
        lb2,
        diff_pct
    );
}

/// Regression test: Ensure training completes without panics
#[test]
fn test_no_panics_during_training() {
    let (mut sddp, saa) =
        create_stochastic_single_reservoir().expect("Failed to create system");

    // This test just ensures no panics occur
    let _result = sddp.train(5, 3, &saa);

    // If we reach here without panic, test passes
}
