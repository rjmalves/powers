// Comprehensive unit tests for Benders cut operations (T1.2)
//
// Tests cover:
// - Cut creation and validation
// - Cut evaluation at various states
// - Numerical stability and edge cases
// - Cut pool operations
//
// PERFORMANCE NOTE: Cut evaluation is in the hot path of SDDP (called thousands
// of times per iteration). These tests verify correctness while being mindful
// of performance characteristics.

// Import test infrastructure from T1.1
mod fixtures;
mod utils;

use utils::assertions::*;

// Access cut module directly (now public in test builds)
use powers_rs::cut::{BendersCut, BendersCutPool};

/// Tests for BendersCut creation and validation
mod test_cut_creation {
    use super::*;

    #[test]
    fn test_basic_creation() {
        let cut = BendersCut::new(1, vec![1.0, 2.0, 3.0], 10.0, 1, 0);

        assert_eq!(cut.id, 1);
        assert_eq!(cut.coefficients, vec![1.0, 2.0, 3.0]);
        assert_eq!(cut.rhs, 10.0);
        assert!(cut.active);
        assert_eq!(cut.non_dominated_state_count, 1);
    }

    #[test]
    fn test_empty_coefficients() {
        // Constant function (no state dependence)
        let cut = BendersCut::new(42, vec![], 100.0, 1, 0);

        assert_eq!(cut.id, 42);
        assert!(cut.coefficients.is_empty());
        assert_eq!(cut.rhs, 100.0);
    }

    #[test]
    fn test_single_coefficient() {
        let cut = BendersCut::new(1, vec![5.0], 10.0, 1, 0);

        assert_eq!(cut.coefficients.len(), 1);
        assert_eq!(cut.coefficients[0], 5.0);
    }

    #[test]
    fn test_large_dimension() {
        // PERFORMANCE: Power systems typically have 50-100 state variables
        let dim = 100;
        let coefficients: Vec<f64> = (0..dim).map(|i| i as f64).collect();

        let cut = BendersCut::new(1, coefficients.clone(), 0.0, 1, 0);

        assert_eq!(cut.coefficients.len(), dim);
        assert_eq!(cut.coefficients, coefficients);
    }

    #[test]
    fn test_zero_coefficients() {
        let cut = BendersCut::new(1, vec![0.0, 0.0, 0.0], 10.0, 1, 0);

        assert!(cut.coefficients.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn test_negative_coefficients() {
        let cut = BendersCut::new(1, vec![-1.0, -2.0, -3.0], 10.0, 1, 0);

        assert!(cut.coefficients.iter().all(|&c| c < 0.0));
    }

    #[test]
    fn test_mixed_sign_coefficients() {
        let cut = BendersCut::new(1, vec![-1.0, 0.0, 1.0], 0.0, 1, 0);

        assert_eq!(cut.coefficients[0], -1.0);
        assert_eq!(cut.coefficients[1], 0.0);
        assert_eq!(cut.coefficients[2], 1.0);
    }

    #[test]
    fn test_extreme_values() {
        let cut = BendersCut::new(1, vec![1e10, 1e-10], 1e8, 1, 0);

        assert_eq!(cut.coefficients[0], 1e10);
        assert_eq!(cut.coefficients[1], 1e-10);
        assert_eq!(cut.rhs, 1e8);
    }

    #[test]
    fn test_default_state() {
        let cut = BendersCut::new(99, vec![1.0], 50.0, 1, 0);

        assert!(cut.active);
        assert_eq!(cut.non_dominated_state_count, 1);
    }
}

/// Tests for cut evaluation
mod test_cut_evaluation {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let cut = BendersCut::new(1, vec![2.0, 3.0], 10.0, 1, 0);
        let state = vec![5.0, 7.0];

        // 10.0 + 2.0*5.0 + 3.0*7.0 = 10.0 + 10.0 + 21.0 = 41.0
        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 41.0, 1e-10);
    }

    #[test]
    fn test_zero_state() {
        let cut = BendersCut::new(1, vec![1.0, 2.0, 3.0], 42.0, 1, 0);
        let state = vec![0.0, 0.0, 0.0];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 42.0, 1e-10);
    }

    #[test]
    fn test_zero_coefficients() {
        let cut = BendersCut::new(1, vec![0.0, 0.0], 100.0, 1, 0);
        let state = vec![999.0, 888.0];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 100.0, 1e-10);
    }

    #[test]
    fn test_negative_coefficients() {
        let cut = BendersCut::new(1, vec![-2.0, -3.0], 10.0, 1, 0);
        let state = vec![5.0, 7.0];

        // 10.0 - 2.0*5.0 - 3.0*7.0 = -21.0
        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, -21.0, 1e-10);
    }

    #[test]
    fn test_empty_cut() {
        let cut = BendersCut::new(1, vec![], 50.0, 1, 0);
        let state = vec![];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 50.0, 1e-10);
    }

    #[test]
    fn test_affine_property() {
        // height(s1) - height(s2) = coeff^T * (s1 - s2)
        let cut = BendersCut::new(1, vec![2.0, 3.0, 5.0], 7.0, 1, 0);

        let s1 = vec![10.0, 20.0, 30.0];
        let s2 = vec![5.0, 15.0, 25.0];

        let h1 = cut.eval_height_at_state(&s1);
        let h2 = cut.eval_height_at_state(&s2);

        let diff = h1 - h2;
        let expected = 2.0 * 5.0 + 3.0 * 5.0 + 5.0 * 5.0; // 50.0

        assert_float_approx_eq(diff, expected, 1e-10);
    }

    #[test]
    fn test_linearity() {
        // cut(k*s) - cut(0) = k * (cut(s) - cut(0))
        let cut = BendersCut::new(1, vec![2.0, 3.0], 11.0, 1, 0);
        let s = vec![4.0, 5.0];

        let h0 = cut.eval_height_at_state(&[0.0, 0.0]);
        let h1 = cut.eval_height_at_state(&s);
        let h2 = cut.eval_height_at_state(&[8.0, 10.0]);

        assert_float_approx_eq(h2 - h0, 2.0 * (h1 - h0), 1e-10);
    }
}

/// Tests for numerical stability
mod test_numerical_stability {
    use super::*;

    #[test]
    fn test_large_values() {
        let cut = BendersCut::new(1, vec![100.0, 200.0], 1000.0, 1, 0);
        let state = vec![10000.0, 20000.0];

        let height = cut.eval_height_at_state(&state);
        let expected = 1000.0 + 1.0e6 + 4.0e6; // 5,001,000.0
        assert_float_approx_eq(height, expected, expected * 1e-10);
    }

    #[test]
    fn test_small_values() {
        let cut = BendersCut::new(1, vec![1e-6, 2e-6], 1e-7, 1, 0);
        let state = vec![1e-3, 2e-3];

        let height = cut.eval_height_at_state(&state);
        assert!(height.is_finite());
        assert!(height > 0.0);
    }

    #[test]
    fn test_mixed_scales() {
        // Common: storage (GWh) vs price ($/MWh)
        let cut = BendersCut::new(1, vec![1e-6, 1e6], 1.0, 1, 0);
        let state = vec![1e6, 1e-6];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 3.0, 1e-9); // 1 + 1 + 1
    }

    #[test]
    fn test_catastrophic_cancellation() {
        let cut = BendersCut::new(1, vec![1e10, -1e10], 1.0, 1, 0);
        let state = vec![1.0, 1.0];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 1.0, 1e-6);
    }

    #[test]
    fn test_no_nan() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0, 1, 0);
        let state = vec![f64::MAX / 10.0, f64::MAX / 10.0];

        let height = cut.eval_height_at_state(&state);
        assert!(!height.is_nan());
    }

    #[test]
    fn test_many_terms() {
        // Test rounding error accumulation
        let dim = 1000;
        let cut = BendersCut::new(1, vec![1.0; dim], 0.0, 1, 0);
        let state = vec![0.001; dim];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 1.0, 1e-9);
    }
}

/// Tests for edge cases
mod test_edge_cases {
    use super::*;

    #[test]
    fn test_high_dimension() {
        let dim = 200;
        let coeffs: Vec<f64> = (1..=dim).map(|i| i as f64).collect();
        let state = vec![1.0; dim];

        let cut = BendersCut::new(1, coeffs, 0.0, 1, 0);
        let height = cut.eval_height_at_state(&state);

        let expected: f64 = (1..=dim).map(|i| i as f64).sum();
        assert_float_approx_eq(height, expected, 1e-6);
    }

    #[test]
    fn test_alternating_signs() {
        let cut = BendersCut::new(1, vec![1.0, -1.0, 1.0, -1.0], 0.0, 1, 0);
        let state = vec![2.0, 2.0, 2.0, 2.0];

        let height = cut.eval_height_at_state(&state);
        assert_float_approx_eq(height, 0.0, 1e-10);
    }
}

/// Tests for BendersCutPool
mod test_cut_pool {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = BendersCutPool::new();

        assert_eq!(pool.pool.len(), 0);
        assert_eq!(pool.active_cut_indices.len(), 0);
        assert_eq!(pool.total_cut_count, 0);
    }

    #[test]
    fn test_pool_empty() {
        let pool = BendersCutPool::new();

        assert!(pool.pool.is_empty());
        assert!(pool.active_cut_indices.is_empty());
    }
}
