//! This module tests edge cases and boundary conditions for Benders cuts:
//! 1. Cuts with zero coefficients (constant functions)
//! 2. Cuts with single coefficients (1-dimensional systems)
//! 3. Cut domination detection logic
//! 4. Cut cloning and equality semantics
//!
//! **Context**: SDDP algorithms must handle degenerate cases gracefully.
//! Edge cases often reveal bugs in cut selection, domination, and evaluation.
//!
//! **Test Strategy**:
//! - Test boundary conditions (empty, single element, all zeros)
//! - Verify mathematical correctness in degenerate cases
//! - Validate domination logic with explicit examples
//! - Test Rust trait implementations (Clone, Debug)

mod fixtures;
mod utils;

use powers_rs::cut::{BendersCut, BendersCutPool};
use utils::assertions::*;

/// TEST-003c.1: Cut with all zero coefficients (constant lower bound)
///
/// Tests that cuts with zero coefficients represent constant functions
/// correctly. This is a valid mathematical case: V(x) = constant.
#[test]
fn test_cut_with_zero_coefficients() {
    let cut = BendersCut::new(1, vec![0.0, 0.0, 0.0], 100.0, 1, 0);

    // Verify structure
    assert_eq!(cut.coefficients.len(), 3);
    assert!(cut.coefficients.iter().all(|&c| c == 0.0));
    assert_eq!(cut.rhs, 100.0);

    // Evaluate at various states - should always return RHS
    let states = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0],
        vec![100.0, 200.0, 300.0],
        vec![-50.0, -100.0, -150.0],
    ];

    for state in &states {
        let height = cut.eval_height_at_state(state);
        assert_float_approx_eq(height, 100.0, 1e-12);
    }
}

/// TEST-003c.2: Cut with empty coefficients vector
///
/// Tests cuts that represent pure constants with no state dependence.
/// This can occur in single-stage problems or terminal nodes.
#[test]
fn test_cut_with_empty_coefficients() {
    let cut = BendersCut::new(42, vec![], 250.0, 1, 0);

    assert_eq!(cut.id, 42);
    assert!(cut.coefficients.is_empty());
    assert_eq!(cut.rhs, 250.0);
    assert!(cut.active);

    // Evaluate at empty state
    let height = cut.eval_height_at_state(&[]);
    assert_float_approx_eq(height, 250.0, 1e-12);
}

/// TEST-003c.3: Cut with single coefficient (1D state space)
///
/// Tests simplest non-trivial case: one state variable.
/// Common in educational examples and small test systems.
#[test]
fn test_cut_with_single_coefficient() {
    let cut = BendersCut::new(1, vec![-2.5], 50.0, 1, 0);

    assert_eq!(cut.coefficients.len(), 1);
    assert_eq!(cut.coefficients[0], -2.5);
    assert_eq!(cut.rhs, 50.0);

    // Test evaluation: 50 + (-2.5)*state
    let test_cases = vec![
        (0.0, 50.0),   // state=0 → 50 + 0 = 50
        (10.0, 25.0),  // state=10 → 50 - 25 = 25
        (20.0, 0.0),   // state=20 → 50 - 50 = 0
        (30.0, -25.0), // state=30 → 50 - 75 = -25
    ];

    for (state, expected) in test_cases {
        let height = cut.eval_height_at_state(&[state]);
        assert_float_approx_eq(height, expected, 1e-10);
    }
}

/// TEST-003c.4: Cut with single non-zero coefficient among many zeros
///
/// Tests sparse cuts where only one dimension is active.
#[test]
fn test_cut_with_mostly_zero_coefficients() {
    // Only middle coefficient is non-zero
    let cut = BendersCut::new(1, vec![0.0, 0.0, -1.5, 0.0, 0.0], 100.0, 1, 0);

    // State where only non-zero coefficient matters
    let state = vec![10.0, 20.0, 40.0, 50.0, 60.0];
    let height = cut.eval_height_at_state(&state);

    // Expected: 100 + (-1.5 * 40) = 100 - 60 = 40
    assert_float_approx_eq(height, 40.0, 1e-10);

    // Verify other dimensions don't affect result
    let state2 = vec![0.0, 0.0, 40.0, 0.0, 0.0];
    let height2 = cut.eval_height_at_state(&state2);
    assert_float_approx_eq(height2, 40.0, 1e-10);
}

/// TEST-003c.5: Cut domination - one cut dominates another everywhere
///
/// Tests explicit domination scenario: cut A provides tighter lower bound
/// than cut B at all feasible states.
#[test]
fn test_cut_domination_strict() {
    // Two parallel cuts (same slope, different intercepts)
    // Cut 1: V(x) = 100 - 2x  (dominates)
    // Cut 2: V(x) = 80 - 2x   (dominated - always 20 units lower)
    let cut1 = BendersCut::new(1, vec![-2.0], 100.0, 1, 0);
    let cut2 = BendersCut::new(2, vec![-2.0], 80.0, 1, 0);

    // Test at multiple states - cut1 should always be higher (tighter)
    let states = vec![0.0, 10.0, 20.0, 30.0, 50.0];

    for state in states {
        let h1 = cut1.eval_height_at_state(&[state]);
        let h2 = cut2.eval_height_at_state(&[state]);

        assert!(
            h1 > h2 + 1e-10,
            "Cut 1 should dominate cut 2 at state {}: h1={}, h2={}",
            state,
            h1,
            h2
        );

        // Verify the gap is constant (20 units)
        assert_float_approx_eq(h1 - h2, 20.0, 1e-10);
    }
}

/// TEST-003c.6: Cut domination - cuts intersect (neither dominates)
///
/// Tests that cuts with different slopes are not dominated even if one
/// is higher at some states.
#[test]
fn test_cut_domination_intersection() {
    // Two cuts that intersect:
    // Cut 1: V(x) = 50 - 1x   (higher at low x)
    // Cut 2: V(x) = 30 - 0.5x (higher at high x)
    // Intersection at x=40: both give 10.0
    let cut1 = BendersCut::new(1, vec![-1.0], 50.0, 1, 0);
    let cut2 = BendersCut::new(2, vec![-0.5], 30.0, 1, 0);

    // At x=0: cut1 = 50, cut2 = 30 (cut1 dominates)
    let h1_at_0 = cut1.eval_height_at_state(&[0.0]);
    let h2_at_0 = cut2.eval_height_at_state(&[0.0]);
    assert!(h1_at_0 > h2_at_0);

    // At x=40: both equal (intersection point)
    let h1_at_40 = cut1.eval_height_at_state(&[40.0]);
    let h2_at_40 = cut2.eval_height_at_state(&[40.0]);
    assert_float_approx_eq(h1_at_40, h2_at_40, 1e-10);
    assert_float_approx_eq(h1_at_40, 10.0, 1e-10);

    // At x=100: cut1 = -50, cut2 = -20 (cut2 dominates)
    let h1_at_100 = cut1.eval_height_at_state(&[100.0]);
    let h2_at_100 = cut2.eval_height_at_state(&[100.0]);
    assert!(h2_at_100 > h1_at_100);

    // Conclusion: neither cut dominates the other globally
}

/// TEST-003c.7: Cut domination in 2D - partial domination
///
/// Tests domination logic in multi-dimensional state space where
/// domination may only occur in certain regions.
#[test]
fn test_cut_domination_multidimensional() {
    // 2D state space: (x, y)
    // Cut 1: V(x,y) = 100 - 2x - 3y
    // Cut 2: V(x,y) = 90 - 2x - 3y  (strictly dominated - parallel, lower)
    // Cut 3: V(x,y) = 100 - 1x - 1y  (intersects - different slope)

    let cut1 = BendersCut::new(1, vec![-2.0, -3.0], 100.0, 1, 0);
    let cut2 = BendersCut::new(2, vec![-2.0, -3.0], 90.0, 1, 0);
    let cut3 = BendersCut::new(3, vec![-1.0, -1.0], 100.0, 1, 0);

    // Cut1 strictly dominates Cut2 everywhere
    let test_states = vec![
        vec![0.0, 0.0],
        vec![10.0, 0.0],
        vec![0.0, 10.0],
        vec![10.0, 10.0],
        vec![20.0, 20.0],
    ];

    for state in &test_states {
        let h1 = cut1.eval_height_at_state(state);
        let h2 = cut2.eval_height_at_state(state);
        assert!(
            h1 > h2 + 1e-10,
            "Cut 1 should dominate Cut 2 at state {:?}",
            state
        );
        assert_float_approx_eq(h1 - h2, 10.0, 1e-10);
    }

    // Cut1 vs Cut3: neither dominates globally
    // At (0,0): cut1 = 100, cut3 = 100 (equal)
    let h1_origin = cut1.eval_height_at_state(&[0.0, 0.0]);
    let h3_origin = cut3.eval_height_at_state(&[0.0, 0.0]);
    assert_float_approx_eq(h1_origin, h3_origin, 1e-10);

    // At (10,0): cut1 = 80, cut3 = 90 (cut3 higher)
    let h1_10_0 = cut1.eval_height_at_state(&[10.0, 0.0]);
    let h3_10_0 = cut3.eval_height_at_state(&[10.0, 0.0]);
    assert!(h3_10_0 > h1_10_0);

    // At (0,10): cut1 = 70, cut3 = 90 (cut3 higher)
    let h1_0_10 = cut1.eval_height_at_state(&[0.0, 10.0]);
    let h3_0_10 = cut3.eval_height_at_state(&[0.0, 10.0]);
    assert!(h3_0_10 > h1_0_10);
}

/// TEST-003c.8: Cut cloning creates independent copies
///
/// Tests that cloning a cut produces a deep copy with independent data.
#[test]
fn test_cut_clone_independence() {
    let cut1 = BendersCut::new(1, vec![1.0, 2.0, 3.0], 10.0, 5, 2);
    let mut cut2 = cut1.clone();

    // Verify initial equality
    assert_eq!(cut2.id, cut1.id);
    assert_eq!(cut2.coefficients, cut1.coefficients);
    assert_eq!(cut2.rhs, cut1.rhs);
    assert_eq!(cut2.iteration, cut1.iteration);
    assert_eq!(cut2.forward_pass_idx, cut1.forward_pass_idx);

    // Modify clone
    cut2.coefficients[0] = 99.0;
    cut2.rhs = 999.0;
    cut2.active = false;

    // Original should be unchanged
    assert_eq!(cut1.coefficients[0], 1.0);
    assert_eq!(cut1.rhs, 10.0);
    assert!(cut1.active);

    // Clone should be modified
    assert_eq!(cut2.coefficients[0], 99.0);
    assert_eq!(cut2.rhs, 999.0);
    assert!(!cut2.active);
}

/// TEST-003c.9: Cut evaluation with negative RHS
///
/// Tests cuts with negative intercepts (can occur in cost minimization).
#[test]
fn test_cut_with_negative_rhs() {
    let cut = BendersCut::new(1, vec![-1.0, -2.0], -50.0, 1, 0);

    assert_eq!(cut.rhs, -50.0);

    // At state (0,0): height = -50
    let height_origin = cut.eval_height_at_state(&[0.0, 0.0]);
    assert_float_approx_eq(height_origin, -50.0, 1e-10);

    // At state (10,20): height = -50 + (-1*10) + (-2*20) = -50 - 10 - 40 = -100
    let height = cut.eval_height_at_state(&[10.0, 20.0]);
    assert_float_approx_eq(height, -100.0, 1e-10);
}

/// TEST-003c.10: Cut evaluation at boundary - maximum representable state
///
/// Tests evaluation doesn't overflow with large but valid states.
#[test]
fn test_cut_at_large_feasible_state() {
    // Realistic: large reservoir storage (10,000 GWh) with small water value
    let cut = BendersCut::new(1, vec![-0.001], 1000.0, 1, 0);

    let large_storage = 10_000.0;
    let height = cut.eval_height_at_state(&[large_storage]);

    // Expected: 1000 + (-0.001 * 10000) = 1000 - 10 = 990
    assert!(height.is_finite());
    assert_float_approx_eq(height, 990.0, 1e-9);
}

/// TEST-003c.11: Cut pool with zero cuts (empty pool)
///
/// Tests that empty cut pool handles operations gracefully.
#[test]
fn test_cut_pool_empty() {
    let pool = BendersCutPool::new();

    assert!(pool.pool.is_empty());
    assert!(pool.active_cut_indices.is_empty());
    assert_eq!(pool.total_cut_count, 0);
}

/// TEST-003c.12: Cut pool with single cut
///
/// Tests minimum viable cut pool with one cut.
#[test]
fn test_cut_pool_single_cut() {
    let mut pool = BendersCutPool::new();

    let cut = BendersCut::new(1, vec![-1.0], 50.0, 1, 0);
    pool.pool.push(cut);
    pool.total_cut_count = 1;

    assert_eq!(pool.pool.len(), 1);
    assert_eq!(pool.total_cut_count, 1);
    assert_eq!(pool.pool[0].id, 1);
}

/// TEST-003c.13: Cut with dimension mismatch detection
///
/// Tests that evaluating a cut at a state with wrong dimension
/// is handled appropriately (should panic).
#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn test_cut_dimension_mismatch() {
    let cut = BendersCut::new(1, vec![1.0, 2.0, 3.0], 10.0, 1, 0);

    // Try to evaluate at wrong dimension (should panic)
    let _ = cut.eval_height_at_state(&[1.0, 2.0]); // 2D state for 3D cut
}

/// TEST-003c.14: Cut coefficients contain NaN
///
/// Tests behavior when cut coefficients are NaN (should be detectable).
#[test]
fn test_cut_with_nan_coefficient() {
    let cut = BendersCut::new(1, vec![1.0, f64::NAN, 3.0], 10.0, 1, 0);

    // Coefficient is NaN
    assert!(cut.coefficients[1].is_nan());

    // Evaluation produces NaN
    let height = cut.eval_height_at_state(&[1.0, 2.0, 3.0]);
    assert!(
        height.is_nan(),
        "Cut with NaN coefficient should produce NaN height"
    );
}

/// TEST-003c.15: Cut coefficients contain infinity
///
/// Tests behavior when cut coefficients are infinite.
#[test]
fn test_cut_with_infinite_coefficient() {
    let cut = BendersCut::new(1, vec![f64::INFINITY, 2.0], 10.0, 1, 0);

    assert!(cut.coefficients[0].is_infinite());

    // Evaluation with finite state
    let height = cut.eval_height_at_state(&[0.0, 1.0]);
    // 10 + (INFINITY * 0) + (2 * 1) → 10 + NaN + 2 = NaN
    // Note: INFINITY * 0.0 = NaN in IEEE 754
    assert!(height.is_nan() || height.is_infinite());
}

/// TEST-003c.16: Cut metadata preservation
///
/// Tests that cut metadata (iteration, forward_pass_idx) is preserved correctly.
#[test]
fn test_cut_metadata_preservation() {
    let iteration = 42;
    let fp_idx = 7;
    let cut = BendersCut::new(1, vec![1.0], 10.0, iteration, fp_idx);

    assert_eq!(cut.iteration, iteration);
    assert_eq!(cut.forward_pass_idx, fp_idx);
    assert_eq!(cut.non_dominated_state_count, 1); // Default
    assert!(cut.active); // Default
}

/// TEST-003c.17: Cut activation/deactivation
///
/// Tests cut active flag manipulation.
#[test]
fn test_cut_active_flag() {
    let mut cut = BendersCut::new(1, vec![1.0], 10.0, 1, 0);

    // Initially active
    assert!(cut.active);

    // Deactivate
    cut.active = false;
    assert!(!cut.active);

    // Reactivate
    cut.active = true;
    assert!(cut.active);

    // Active flag doesn't affect evaluation
    let height_active = cut.eval_height_at_state(&[5.0]);
    cut.active = false;
    let height_inactive = cut.eval_height_at_state(&[5.0]);
    assert_float_approx_eq(height_active, height_inactive, 1e-12);
}

/// TEST-003c.18: Multiple cuts with same coefficients, different RHS
///
/// Tests distinguishing between cuts that differ only in RHS.
#[test]
fn test_cuts_same_coefficients_different_rhs() {
    let coeffs = vec![-1.0, -2.0];
    let cut1 = BendersCut::new(1, coeffs.clone(), 100.0, 1, 0);
    let cut2 = BendersCut::new(2, coeffs.clone(), 90.0, 1, 0);
    let cut3 = BendersCut::new(3, coeffs.clone(), 80.0, 1, 0);

    // All have same slope, different intercepts
    assert_eq!(cut1.coefficients, cut2.coefficients);
    assert_eq!(cut2.coefficients, cut3.coefficients);
    assert_ne!(cut1.rhs, cut2.rhs);
    assert_ne!(cut2.rhs, cut3.rhs);

    // At same state, heights form an ordering
    let state = vec![10.0, 20.0];
    let h1 = cut1.eval_height_at_state(&state);
    let h2 = cut2.eval_height_at_state(&state);
    let h3 = cut3.eval_height_at_state(&state);

    assert!(h1 > h2 && h2 > h3, "Heights should be ordered by RHS");
    assert_float_approx_eq(h1 - h2, 10.0, 1e-10);
    assert_float_approx_eq(h2 - h3, 10.0, 1e-10);
}

#[cfg(test)]
mod cut_pool_edge_cases {
    use super::*;

    /// Test cut pool iteration order is deterministic
    #[test]
    fn test_cut_pool_deterministic_order() {
        let mut pool = BendersCutPool::new();

        // Add cuts in specific order
        for i in 0..5 {
            let cut = BendersCut::new(i, vec![i as f64], i as f64 * 10.0, 1, 0);
            pool.pool.push(cut);
        }
        pool.total_cut_count = 5;

        // Verify order is preserved
        for i in 0..5 {
            assert_eq!(pool.pool[i].id, i);
        }
    }

    /// Test cut pool capacity growth
    #[test]
    fn test_cut_pool_capacity_growth() {
        let mut pool = BendersCutPool::new();

        // Add many cuts to trigger reallocation
        for i in 0..100 {
            let cut = BendersCut::new(i, vec![1.0], 10.0, 1, 0);
            pool.pool.push(cut);
        }

        assert_eq!(pool.pool.len(), 100);
        assert!(pool.pool.capacity() >= 100);
    }
}
