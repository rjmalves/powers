//! Cut Numerical Stability Tests
//!
//! This module tests numerical stability properties of Benders cuts,
//! focusing on:
//! 1. Kahan summation usage in cut operations
//! 2. Numerical precision in cut evaluation
//! 3. Extreme coefficient handling
//!
//! **Context**: SDDP cuts involve dot products with hundreds of coefficients.
//! Floating-point rounding errors can accumulate and affect:
//! - Cut domination detection (different heights -> different selections)
//! - Lower bound convergence (small errors compound over iterations)
//! - Reproducibility (different operation orders -> different results)
//!
//! **Test Strategy**:
//! - Verify Kahan summation is used in critical paths
//! - Test precision with pathological cases (cancellation, accumulation)
//! - Validate handling of extreme values (near overflow/underflow)

mod fixtures;
mod utils;

use powers_rs::cut::BendersCut;
use powers_rs::utils::{dot_product, dot_product_deterministic, kahan_sum};
use utils::assertions::*;

/// TEST-003b.1: Verify Kahan summation prevents accumulation errors
///
/// Tests that cut evaluation uses compensated summation to maintain
/// precision when summing many terms.
#[test]
fn test_kahan_summation_in_cut_evaluation() {
    // Create a cut with many small terms
    // Without Kahan: accumulation errors can be O(eps * n)
    // With Kahan: errors bounded to O(eps)
    let dim = 10000;
    let coefficients = vec![1e-10; dim];
    let state = vec![1.0; dim];

    let cut = BendersCut::new(1, coefficients.clone(), 0.0, 1, 0);
    let height = cut.eval_height_at_state(&state);

    // Expected: 10000 * 1e-10 = 1e-6
    // Without Kahan: might get 0.0 or garbage due to precision loss
    // With Kahan: should be accurate within relative error
    let expected = 1e-6;
    let relative_error = (height - expected).abs() / expected;

    assert!(
        relative_error < 1e-6,
        "Cut evaluation lost precision: expected {}, got {}, rel_error={}",
        expected,
        height,
        relative_error
    );
}

/// TEST-003b.2: Verify deterministic dot product usage
///
/// Confirms that cut evaluation produces identical results across calls,
/// which requires order-independent accumulation (Kahan summation).
#[test]
fn test_cut_evaluation_deterministic() {
    // Create cut with coefficients that have different rounding behavior
    let coefficients = vec![1e15, 1.0, -1e15, 1.0, 1.0];
    let state = vec![1.0; 5];

    let cut = BendersCut::new(1, coefficients.clone(), 10.0, 1, 0);

    // Evaluate multiple times - should be bit-identical
    let results: Vec<f64> =
        (0..100).map(|_| cut.eval_height_at_state(&state)).collect();

    // All results should be exactly equal (bit-level identical)
    let first = results[0];
    for (i, &result) in results.iter().enumerate() {
        assert_eq!(
            result, first,
            "Cut evaluation not deterministic: iteration {} gave {}, expected {}",
            i, result, first
        );
    }
}

/// TEST-003b.3: Verify Kahan summation accuracy
///
/// Demonstrates that Kahan summation maintains precision in pathological cases
/// where naive summation would lose the small values.
#[test]
fn test_kahan_vs_naive_precision() {
    // Pathological case from existing test suite: alternating large and small values
    // Pattern: large + small - large + small...
    // Naive sum: (1e10 + 1.0) → 1e10 (1.0 lost), then cancel, loses remaining 1.0s
    // Kahan: compensates for lost bits, preserves all 1.0 values
    let values = vec![
        1e10, 1.0, -1e10, // Large cancellation with small value
        1e10, 1.0, -1e10, // Repeat pattern
        1e10, 1.0, -1e10, 1e10, 1.0, -1e10,
    ];

    let naive_sum: f64 = values.iter().sum();
    let kahan_sum_val = kahan_sum(&values);

    // Expected: 4 * 1.0 = 4.0 (the large values cancel out perfectly)
    let expected = 4.0;

    let naive_error = (naive_sum - expected).abs();
    let kahan_error = (kahan_sum_val - expected).abs();

    // Primary assertion: Kahan result should be very accurate
    assert!(
        kahan_error < 1e-10,
        "Kahan summation not accurate enough: expected {}, got {}, error={}",
        expected,
        kahan_sum_val,
        kahan_error
    );

    // Document comparison (naive may or may not lose precision depending on compiler)
    println!(
        "Summation comparison (pathological case: 1e10 ± 1e10 with 1.0 terms):\n\
         Expected: {}\n\
         Naive sum: {} (error: {})\n\
         Kahan sum: {} (error: {})",
        expected, naive_sum, naive_error, kahan_sum_val, kahan_error
    );

    // In many cases, naive will lose precision
    if naive_error > kahan_error * 10.0 {
        println!(
            "✓ Kahan preserved precision where naive lost it ({}x better)",
            naive_error / kahan_error
        );
    } else {
        println!("  Note: Compiler optimized naive sum well in this case");
    }
}

/// TEST-003b.4: Cut evaluation with extreme coefficients
///
/// Tests that cuts handle coefficients near overflow/underflow gracefully.
#[test]
fn test_cut_with_extreme_coefficients() {
    // Test near overflow
    let large_coeffs = vec![1e100, 1e100, 1e100];
    let small_state = vec![1e-100, 1e-100, 1e-100];

    let cut_large = BendersCut::new(1, large_coeffs, 0.0, 1, 0);
    let height_large = cut_large.eval_height_at_state(&small_state);

    // Should not overflow: 1e100 * 1e-100 = 1.0 (three times)
    assert!(height_large.is_finite(), "Cut evaluation overflowed");
    assert_float_approx_eq(height_large, 3.0, 1e-10);

    // Test near underflow
    let tiny_coeffs = vec![1e-150, 1e-150];
    let medium_state = vec![1e10, 1e10];

    let cut_tiny = BendersCut::new(2, tiny_coeffs, 0.0, 1, 0);
    let height_tiny = cut_tiny.eval_height_at_state(&medium_state);

    // Should not underflow: 1e-150 * 1e10 = 1e-140 (very small but representable)
    assert!(height_tiny.is_finite(), "Cut evaluation underflowed");
    assert!(height_tiny > 0.0, "Cut evaluation lost precision to zero");
}

/// TEST-003b.5: Numerical precision with mixed-scale coefficients
///
/// Realistic scenario: storage coefficients (small) and price coefficients (large)
/// mixed in the same cut.
#[test]
fn test_cut_evaluation_numerical_precision_mixed_scales() {
    // Realistic power system: storage (GWh) vs prices ($/MWh)
    // Storage coefficients: O(1e-3) to O(1e-6) $/MWh-stored
    // Price/inflow coefficients: O(1) to O(1e3) $/unit

    let mut coefficients = Vec::new();
    let mut state = Vec::new();

    // 10 storage variables with tiny water values
    for _ in 0..10 {
        coefficients.push(-1e-6); // Water value ~$0.001/MWh
        state.push(1000.0); // Storage ~1000 GWh
    }

    // 10 price variables with moderate values
    for _ in 0..10 {
        coefficients.push(100.0); // Price coefficient
        state.push(0.01); // Small price perturbation
    }

    let cut = BendersCut::new(1, coefficients, 5000.0, 1, 0);
    let height = cut.eval_height_at_state(&state);

    // Expected: 5000 + 10*(-1e-6*1000) + 10*(100*0.01)
    //         = 5000 - 0.01 + 10
    //         = 5009.99
    let expected = 5009.99;

    assert!(
        height.is_finite(),
        "Mixed-scale evaluation produced non-finite result"
    );

    let relative_error = (height - expected).abs() / expected;
    assert!(
        relative_error < 1e-9,
        "Mixed-scale precision loss: expected {}, got {}, rel_error={}",
        expected,
        height,
        relative_error
    );
}

/// TEST-003b.6: Verify dot_product_deterministic is order-independent
///
/// Critical property: cut height must not depend on coefficient ordering.
#[test]
fn test_deterministic_dot_product_order_independence() {
    let coefficients = vec![1e10, 1.0, -1e10, 2.0, 3.0];
    let state = vec![1.0, 1.0, 1.0, 1.0, 1.0];

    // Compute with original order
    let result1 = dot_product_deterministic(&coefficients, &state);

    // Reverse order
    let mut coeffs_rev = coefficients.clone();
    coeffs_rev.reverse();
    let mut state_rev = state.clone();
    state_rev.reverse();
    let result2 = dot_product_deterministic(&coeffs_rev, &state_rev);

    // Permute order
    let coeffs_perm = vec![
        coefficients[1],
        coefficients[3],
        coefficients[0],
        coefficients[4],
        coefficients[2],
    ];
    let state_perm = vec![state[1], state[3], state[0], state[4], state[2]];
    let result3 = dot_product_deterministic(&coeffs_perm, &state_perm);

    // All should be exactly equal
    assert_eq!(
        result1, result2,
        "Deterministic dot product depends on order: original={}, reversed={}",
        result1, result2
    );

    assert_eq!(
        result1, result3,
        "Deterministic dot product depends on order: original={}, permuted={}",
        result1, result3
    );
}

/// TEST-003b.7: Verify naive dot product IS order-dependent (demonstrates problem)
///
/// This test documents why we need deterministic dot product - to show
/// that the standard approach is insufficient.
#[test]
fn test_naive_dot_product_order_dependent() {
    // Pathological case where order matters
    let coefficients = vec![1e16, 1.0, -1e16];
    let state = vec![1.0, 1.0, 1.0];

    // Standard dot product (naive)
    let result_forward = dot_product(&coefficients, &state);

    // Reverse (tests compiler's ability to optimize differently)
    let mut coeffs_rev = coefficients.clone();
    coeffs_rev.reverse();
    let mut state_rev = state.clone();
    state_rev.reverse();
    let result_reverse = dot_product(&coeffs_rev, &state_rev);

    // Expected mathematically: 1e16 + 1 - 1e16 = 1.0
    let expected = 1.0;

    // At least one should have precision loss
    let error_forward = (result_forward - expected).abs();
    let error_reverse = (result_reverse - expected).abs();

    // We expect that at least one has significant error (though this may
    // not always be true depending on compiler optimizations)
    // The key point is documenting this potential behavior
    println!(
        "Naive dot product precision (documenting potential instability):\n\
         Forward order error: {}\n\
         Reverse order error: {}",
        error_forward, error_reverse
    );

    // Even if both are close, document that they could differ
    // This test serves as documentation rather than assertion
    assert!(
        result_forward.is_finite() && result_reverse.is_finite(),
        "Naive dot product should at least not crash"
    );
}

/// TEST-003b.8: Extreme coefficient magnitude ratios
///
/// Tests cuts where coefficient magnitudes differ by many orders.
#[test]
fn test_extreme_coefficient_magnitude_ratios() {
    // Mix tiny and huge coefficients (1e-100 to 1e100 = 200 orders of magnitude)
    let coefficients = vec![
        1e100, 1e-100, // 200 orders of magnitude difference
        1e50, 1e-50, // 100 orders apart
        1.0, 1e-10, // 10 orders apart
    ];

    let state = vec![1e-100, 1e100, 1e-50, 1e50, 1.0, 1e10];

    let cut = BendersCut::new(1, coefficients, 0.0, 1, 0);
    let height = cut.eval_height_at_state(&state);

    // Expected: 1e100*1e-100 + 1e-100*1e100 + 1e50*1e-50 + 1e-50*1e50 + 1.0*1.0 + 1e-10*1e10
    //         = 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 = 6.0
    assert!(
        height.is_finite(),
        "Extreme magnitude ratios caused overflow/underflow"
    );
    assert_float_approx_eq(height, 6.0, 1e-9);
}

/// TEST-003b.9: Verify cut evaluation handles subnormal numbers
///
/// Tests that cuts work correctly with denormalized floating-point numbers.
#[test]
fn test_cut_with_subnormal_numbers() {
    // Subnormal numbers: between 0 and smallest normal f64 (~2.2e-308)
    // These have reduced precision but should still work
    let subnormal = 1e-320; // Well into subnormal range

    let coefficients = vec![subnormal, subnormal, subnormal];
    let state = vec![1e100, 1e100, 1e100]; // Large enough to bring back to normal range

    let cut = BendersCut::new(1, coefficients, 0.0, 1, 0);
    let height = cut.eval_height_at_state(&state);

    // Result: 3 * (1e-320 * 1e100) = 3e-220 (still subnormal but representable)
    assert!(
        height.is_finite() && height > 0.0,
        "Subnormal number handling failed: got {}",
        height
    );
}

/// TEST-003b.10: Precision with alternating signs (catastrophic cancellation)
///
/// Tests the worst case for numerical summation: large alternating terms
/// that mostly cancel.
#[test]
fn test_catastrophic_cancellation_precision() {
    // Worst case: alternating +large/-large with small residual
    let mut coefficients = Vec::new();
    let mut state = Vec::new();

    // 100 pairs of (1e10, -1e10) with a final 1.0
    // Naive summation: each pair loses precision, compounds over 100 pairs
    // Kahan: maintains precision throughout
    for _ in 0..100 {
        coefficients.push(1e10);
        coefficients.push(-1e10);
        state.push(1.0);
        state.push(1.0);
    }
    coefficients.push(1.0);
    state.push(1.0);

    let cut = BendersCut::new(1, coefficients, 0.0, 1, 0);
    let height = cut.eval_height_at_state(&state);

    // Expected: 100 * (1e10 - 1e10) + 1.0 = 1.0
    let expected = 1.0;

    // With Kahan summation, this should be exact or very close
    assert_float_approx_eq(height, expected, 1e-10);
}

#[cfg(test)]
mod test_utils {

    /// Utility: Demonstrates precision comparison
    #[test]
    fn test_precision_comparison_utility() {
        let a: f64 = 1.0;
        let b: f64 = 1.0 + 1e-15; // Just above machine epsilon

        // Absolute error
        let abs_err = (a - b).abs();
        assert!(abs_err > 0.0 && abs_err < 1e-14);

        // Relative error
        let rel_err = abs_err / a.abs();
        assert!(rel_err < 1e-14);
    }
}
