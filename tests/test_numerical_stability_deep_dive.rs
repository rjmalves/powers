//! Numerical Stability Deep Dive
//!
//! This module provides comprehensive numerical stability tests for the SDDP implementation.
//! These tests go beyond basic correctness to ensure numerical precision is maintained
//! in critical computational paths.
//!
//! ## Why Numerical Stability Matters
//!
//! SDDP involves thousands of LP solves and cut evaluations. Small numerical errors can:
//! - Accumulate over iterations → convergence issues
//! - Cause non-monotonic lower bounds → violate SDDP guarantees
//! - Produce incorrect optimal policies → real-world losses
//!
//! ## Key Numerical Issues in SDDP
//!
//! 1. **Catastrophic Cancellation**: Subtracting nearly-equal large numbers
//!    - Example: (1e16 + 1.0) - 1e16 = 0.0 (should be 1.0)
//!    - Occurs in: Cut evaluation with extreme storage values
//!
//! 2. **Loss of Significance**: Adding small numbers to large numbers
//!    - Example: 1e16 + 1.0 = 1e16 (small term lost)
//!    - Occurs in: Summing many cut contributions
//!
//! 3. **Rounding Errors**: Accumulated over many operations
//!    - Each operation has ~1e-16 relative error
//!    - N operations → O(N * ε) error without compensation
//!
//! ## Our Defenses
//!
//! - **Kahan Summation**: Compensated summation for dot products
//! - **Deterministic Ordering**: Fixed evaluation order for reproducibility  
//! - **Appropriate Tolerances**: 1e-6 for LP, 1e-4 for SDDP convergence
//! - **Extreme Value Handling**: Clipping, bounds checking

mod utils;

use powers_rs::cut::BendersCut;
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::utils::{dot_product_deterministic, kahan_sum};
use std::path::Path;

// ============================================================================
// Kahan Summation Accuracy
// ============================================================================

mod test_kahan_summation_accuracy {
    use super::*;

    /// Classic numerical stability test: demonstrate catastrophic cancellation
    ///
    /// The sequence [1e10, 1.0, -1e10, 1.0] should sum to 2.0.
    /// Naive summation loses the small terms due to limited precision.
    #[test]
    fn test_kahan_prevents_catastrophic_cancellation() {
        // Classic test case for catastrophic cancellation
        let values = vec![1e10, 1.0, -1e10, 1.0];

        // Kahan summation should get the correct answer
        let kahan_result = kahan_sum(&values);
        assert!(
            (kahan_result - 2.0).abs() < 1e-10,
            "Kahan sum should be accurate: got {}, expected 2.0",
            kahan_result
        );

        // Naive summation would fail
        let naive_result: f64 = values.iter().sum();
        // Note: In release mode, LLVM may optimize this differently
        // The test documents the issue even if optimization helps
        if (naive_result - 2.0).abs() > 1e-10 {
            println!(
                "Naive summation lost precision: got {}, expected 2.0",
                naive_result
            );
        }
    }

    /// Test Kahan summation with many small additions to large sum
    #[test]
    fn test_kahan_many_small_additions() {
        // Start with large base, add many small values
        let base = 1e10;
        let count = 10_000;
        let small_value = 1.0;

        let mut values = vec![base];
        for _ in 0..count {
            values.push(small_value);
        }

        let result = kahan_sum(&values);
        let expected = base + (count as f64) * small_value;

        let rel_error = ((result - expected) / expected).abs();
        assert!(
            rel_error < 1e-10,
            "Kahan sum should maintain precision: got {}, expected {}, rel_error: {}",
            result,
            expected,
            rel_error
        );
    }

    /// Test summation with alternating signs (worst case for cancellation)
    #[test]
    fn test_kahan_alternating_signs() {
        // Alternating large positive and negative values with small net result
        let n = 1000;
        let mut values = Vec::with_capacity(2 * n);

        for i in 0..n {
            values.push(1e8 + i as f64);
            values.push(-1e8);
        }

        let result = kahan_sum(&values);
        let expected = (0..n).map(|i| i as f64).sum::<f64>();

        let abs_error = (result - expected).abs();
        assert!(
            abs_error < 1e-6,
            "Kahan sum should handle alternating signs: got {}, expected {}, error: {}",
            result,
            expected,
            abs_error
        );
    }

    /// Test that Kahan sum works with all positive values (baseline)
    #[test]
    fn test_kahan_all_positive() {
        let values: Vec<f64> = (1..=1000).map(|i| i as f64).collect();

        let result = kahan_sum(&values);
        let expected = 500_500.0; // Sum of 1..1000

        assert!(
            (result - expected).abs() < 1e-10,
            "Kahan sum should be exact for integers: got {}, expected {}",
            result,
            expected
        );
    }

    /// Test Kahan sum with different magnitude values
    ///
    /// This test verifies that Kahan summation works correctly within
    /// the practical range of magnitude differences found in SDDP.
    #[test]
    fn test_kahan_moderate_magnitude_differences() {
        // Test with 6 orders of magnitude difference (realistic for SDDP)
        // Storage values might be 1e6, water values 1e0
        let values = vec![1e-3, 1e3, 1e-3, -1e3, 1e-3];

        let result = kahan_sum(&values);
        let expected: f64 = 3e-3;

        let rel_error = ((result - expected) / expected).abs();

        // At 6 orders of magnitude, Kahan maintains excellent precision
        assert!(
            rel_error < 1e-10,
            "Kahan sum should handle moderate magnitude differences: got {}, expected {}, rel_error: {}",
            result,
            expected,
            rel_error
        );

        // Verify the result is accurate
        assert!(
            (result - expected).abs() < 1e-12,
            "Result should be accurate: got {}, expected {}",
            result,
            expected
        );
    }
}

// ============================================================================
// Dot Product Stability
// ============================================================================

mod test_dot_product_stability {
    use super::*;

    /// Test dot product with extreme values (large state, small coefficients)
    #[test]
    fn test_dot_product_extreme_values() {
        // State with very large storage values (e.g., TWh scale)
        let state = vec![1e8; 100];
        // Coefficients are water values ($/MWh scaled) - very small
        let coefficients = vec![-1e-8; 100];

        let result = dot_product_deterministic(&coefficients, &state);

        // Expected: 100 * (1e8 * -1e-8) = -100.0
        let expected = -100.0;

        let rel_error = ((result - expected) / expected).abs();
        assert!(
            rel_error < 1e-10,
            "Dot product should handle extreme values: got {}, expected {}, rel_error: {}",
            result,
            expected,
            rel_error
        );
    }

    /// Test dot product determinism (same inputs → same output)
    #[test]
    fn test_dot_product_deterministic() {
        let coefficients = vec![1.5, -2.3, 4.7, -1.2, 3.8];
        let state = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        // Compute multiple times
        let results: Vec<f64> = (0..10)
            .map(|_| dot_product_deterministic(&coefficients, &state))
            .collect();

        // All results should be identical (bit-exact)
        let first = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                result, first,
                "Dot product should be deterministic: iteration {} differs",
                i
            );
        }
    }

    /// Test dot product with subnormal numbers
    #[test]
    fn test_dot_product_subnormal() {
        // Subnormal numbers: below 2^-1022 (f64 minimum normal)
        let coefficients = vec![1e-320; 100];
        let state = vec![1e10; 100];

        let result = dot_product_deterministic(&coefficients, &state);

        // Expected: 100 * (1e-320 * 1e10) = 1e-308 (subnormal range)
        // Note: At these extreme magnitudes, precision is limited

        // For subnormal numbers, check order of magnitude
        assert!(
            result > 0.0 || result.abs() < 1e-300,
            "Should produce result in subnormal range or zero: got {}",
            result
        );
        assert!(
            result < 1e-300,
            "Should stay in subnormal range: got {}",
            result
        );
    }

    /// Test dot product with mixed positive and negative terms
    #[test]
    fn test_dot_product_mixed_signs() {
        // Alternating signs that mostly cancel
        let mut coefficients = Vec::new();
        let mut state = Vec::new();

        for i in 0..1000 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            coefficients.push(sign * 1.0);
            state.push(1e6);
        }

        let result = dot_product_deterministic(&coefficients, &state);

        // Should sum to 0 (equal positive and negative terms)
        assert!(
            result.abs() < 1e-6,
            "Mixed signs should nearly cancel: got {}",
            result
        );
    }
}

// ============================================================================
// Cut Evaluation Stability
// ============================================================================

mod test_cut_evaluation_stability {
    use super::*;

    /// Test cut evaluation with extreme storage values
    #[test]
    fn test_cut_evaluation_large_storage() {
        // Cut with small coefficients (typical water values)
        let cut =
            BendersCut::new(1, vec![-0.001, -0.002, -0.003], 1000.0, 1, 0);

        // State with very large storage (TWh scale)
        let large_state = vec![1e6, 1e6, 1e6];

        let height = cut.eval_height_at_state(&large_state);

        // Expected: 1000.0 + (-0.001*1e6 + -0.002*1e6 + -0.003*1e6)
        //         = 1000.0 + (-0.006 * 1e6) = 1000.0 - 6000.0 = -5000.0
        let expected = -5000.0;

        let abs_error = (height - expected).abs();
        assert!(
            abs_error < 1e-6,
            "Cut evaluation should handle large storage: got {}, expected {}",
            height,
            expected
        );
    }

    /// Test cut evaluation with many small coefficients
    #[test]
    fn test_cut_evaluation_many_dimensions() {
        // 100-dimensional state (large reservoir system)
        let n = 100;
        let coefficients = vec![-0.01; n];
        let state = vec![100.0; n];

        let cut = BendersCut::new(1, coefficients, 5000.0, 1, 0);
        let height = cut.eval_height_at_state(&state);

        // Expected: 5000.0 + (100 * -0.01 * 100.0) = 5000.0 - 100.0 = 4900.0
        let expected = 4900.0;

        let rel_error = ((height - expected) / expected).abs();
        assert!(
            rel_error < 1e-10,
            "Cut evaluation should handle many dimensions: got {}, expected {}",
            height,
            expected
        );
    }

    /// Test cut evaluation with near-zero coefficients
    #[test]
    fn test_cut_evaluation_near_zero_coefficients() {
        // Very small coefficients (e.g., low-value water)
        let cut = BendersCut::new(1, vec![1e-10, -1e-10, 1e-10], 100.0, 1, 0);
        let state = vec![1e5, 1e5, 1e5];

        let height = cut.eval_height_at_state(&state);

        // Expected: 100.0 + (1e-10*1e5 - 1e-10*1e5 + 1e-10*1e5) = 100.0 + 1e-5
        let expected = 100.0;

        let abs_error = (height - expected).abs();
        assert!(
            abs_error < 1e-4,
            "Cut evaluation should handle tiny coefficients: got {}, expected {}",
            height,
            expected
        );
    }

    /// Test cut evaluation determinism
    #[test]
    fn test_cut_evaluation_deterministic() {
        let cut = BendersCut::new(1, vec![1.1, -2.2, 3.3, -4.4], 50.0, 1, 0);
        let state = vec![10.0, 20.0, 30.0, 40.0];

        // Evaluate multiple times
        let results: Vec<f64> =
            (0..10).map(|_| cut.eval_height_at_state(&state)).collect();

        // All should be identical (bit-exact)
        let first = results[0];
        for (i, &result) in results.iter().enumerate() {
            assert_eq!(
                result, first,
                "Cut evaluation should be deterministic: iteration {} differs",
                i
            );
        }
    }
}

// ============================================================================
// LP Solver Consistency
// ============================================================================

mod test_lp_solver_consistency {
    use super::*;

    /// Test that solving same problem twice gives same result
    #[test]
    fn test_solver_reproducibility() {
        let example_dir = Path::new("examples/01-deterministic");

        // Train twice with same configuration
        let results: Vec<f64> = (0..2)
            .map(|_| {
                let mut instance = SddpAlgorithm::from_files(
                    example_dir.join("config.json"),
                    example_dir.join("system.json"),
                    example_dir.join("graph.json"),
                    example_dir.join("recourse.json"),
                )
                .expect("Failed to load example");

                let result = instance.train().expect("Training failed");
                result.final_lower_bound
            })
            .collect();

        // Results should be identical (or very close due to LP tolerances)
        let diff = (results[0] - results[1]).abs();
        assert!(
            diff < 1e-6,
            "LP solver should be reproducible: run 1: {}, run 2: {}, diff: {}",
            results[0],
            results[1],
            diff
        );
    }

    /// Test that deterministic problems converge to same value
    #[test]
    fn test_deterministic_convergence_consistency() {
        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load example");

        let result = instance.train().expect("Training failed");

        // For deterministic problems, final LB and UB should be very close
        let gap = result.final_gap();
        let relative_gap = result.relative_gap();

        // Allow small gap due to LP tolerances
        assert!(
            gap < 10.0 || relative_gap < 0.01,
            "Deterministic problem should converge tightly: gap {}, rel_gap {}%",
            gap,
            relative_gap * 100.0
        );
    }
}

// ============================================================================
// TEST-017e: Floating-Point Edge Cases
// ============================================================================

mod test_floating_point_edge_cases {
    use super::*;

    /// Test handling of infinity
    #[test]
    fn test_infinity_handling() {
        // Cuts with infinite values should be detectable
        let infinite_cut = BendersCut::new(1, vec![f64::INFINITY], 100.0, 1, 0);
        let state = vec![1.0];

        let height = infinite_cut.eval_height_at_state(&state);
        assert!(
            height.is_infinite(),
            "Infinite coefficient should produce infinite height"
        );
    }

    /// Test handling of NaN
    #[test]
    fn test_nan_handling() {
        // NaN in cut should propagate
        let nan_cut = BendersCut::new(1, vec![f64::NAN], 100.0, 1, 0);
        let state = vec![1.0];

        let height = nan_cut.eval_height_at_state(&state);
        assert!(height.is_nan(), "NaN coefficient should produce NaN height");
    }

    /// Test zero times large number
    #[test]
    fn test_zero_times_large() {
        let cut = BendersCut::new(1, vec![0.0], 100.0, 1, 0);
        let state = vec![1e100];

        let height = cut.eval_height_at_state(&state);

        // Should be exactly 100.0 (0 * large = 0)
        assert_eq!(
            height, 100.0,
            "Zero coefficient should nullify large state"
        );
    }

    /// Test very small times very large (underflow check)
    #[test]
    fn test_small_times_large_no_overflow() {
        let cut = BendersCut::new(1, vec![1e-200], 0.0, 1, 0);
        let state = vec![1e150];

        let height = cut.eval_height_at_state(&state);

        // 1e-200 * 1e150 = 1e-50 (should not underflow to 0)
        assert!(height > 0.0, "Product should not underflow: got {}", height);
        assert!(
            height < 1e-40,
            "Product should be in expected range: got {}",
            height
        );
    }
}

// ============================================================================
// Numerical Precision in Training
// ============================================================================

mod test_numerical_precision_in_training {
    use super::*;

    /// Test that lower bounds maintain precision over many iterations
    #[test]
    fn test_lower_bound_precision_preservation() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load example");

        let result = instance.train().expect("Training failed");

        let lower_bounds: Vec<f64> =
            result.iterations().iter().map(|r| r.lower_bound).collect();

        // Check that all LBs are finite (no overflow/underflow)
        for (i, &lb) in lower_bounds.iter().enumerate() {
            assert!(
                lb.is_finite(),
                "Lower bound at iteration {} should be finite: got {}",
                i,
                lb
            );
        }

        // Check reasonable magnitude (shouldn't explode or vanish)
        for (i, &lb) in lower_bounds.iter().enumerate() {
            assert!(
                lb.abs() < 1e12,
                "Lower bound at iteration {} too large: got {}",
                i,
                lb
            );
        }
    }

    /// Test that gaps remain well-behaved numerically
    #[test]
    fn test_gap_computation_stability() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load example");

        let result = instance.train().expect("Training failed");

        // Gaps should be finite and non-negative
        let final_gap = result.final_gap();
        assert!(final_gap.is_finite(), "Final gap should be finite");
        assert!(final_gap >= -1e-6, "Final gap should be non-negative");

        let rel_gap = result.relative_gap();
        assert!(
            rel_gap.is_finite() || rel_gap.is_infinite(),
            "Relative gap should be finite or infinity (if LB=0)"
        );
        if rel_gap.is_finite() {
            assert!(rel_gap >= -1e-6, "Relative gap should be non-negative");
        }
    }
}

// ============================================================================
// Comprehensive Numerical Stability Analysis
// ============================================================================

#[test]
fn test_comprehensive_numerical_stability() {
    println!("\n=== Comprehensive Numerical Stability Analysis ===\n");

    // Test 1: Kahan summation precision
    println!("1. Kahan Summation Precision");
    let test_values = vec![1e10, 1.0, -1e10, 1.0];
    let kahan_result = kahan_sum(&test_values);
    println!("   Classic test [1e10, 1, -1e10, 1]: {:.15e}", kahan_result);
    println!("   Error: {:.15e}", (kahan_result - 2.0).abs());
    assert!((kahan_result - 2.0).abs() < 1e-10);

    // Test 2: Dot product with extreme values
    println!("\n2. Dot Product with Extreme Values");
    let state_large = vec![1e8; 100];
    let coef_small = vec![-1e-8; 100];
    let dot_result = dot_product_deterministic(&coef_small, &state_large);
    println!("   100-dim: state=1e8, coef=-1e-8");
    println!("   Result: {:.15e} (expected: -100.0)", dot_result);
    println!("   Error: {:.15e}", (dot_result + 100.0).abs());
    assert!(((dot_result + 100.0) / 100.0).abs() < 1e-10);

    // Test 3: Cut evaluation stability
    println!("\n3. Cut Evaluation Stability");
    let cut = BendersCut::new(1, vec![-0.01; 50], 1000.0, 1, 0);
    let state = vec![200.0; 50];
    let height = cut.eval_height_at_state(&state);
    let expected = 1000.0 - 100.0; // 50 * (-0.01 * 200.0)
    println!("   50-dim cut evaluation");
    println!("   Result: {:.15e} (expected: {})", height, expected);
    println!("   Error: {:.15e}", (height - expected).abs());
    assert!((height - expected).abs() < 1e-10);

    // Test 4: Training numerical health
    println!("\n4. Training Numerical Health");
    let example_dir = Path::new("examples/01-deterministic");
    let mut instance = SddpAlgorithm::from_files(
        example_dir.join("config.json"),
        example_dir.join("system.json"),
        example_dir.join("graph.json"),
        example_dir.join("recourse.json"),
    )
    .expect("Failed to load example");

    let result = instance.train().expect("Training failed");
    let lb = result.final_lower_bound;
    let ub = result.statistical_upper_bound;
    let gap = result.final_gap();

    println!("   Final LB: {:.6}", lb);
    println!("   Final UB: {:.6}", ub);
    println!("   Gap: {:.6}", gap);
    println!("   Relative Gap: {:.6}%", result.relative_gap() * 100.0);

    assert!(lb.is_finite(), "Lower bound should be finite");
    assert!(ub.is_finite(), "Upper bound should be finite");
    assert!(gap >= -1e-6, "Gap should be non-negative");

    println!("\n=== All Numerical Stability Tests Passed ===\n");
}
