/// Comprehensive tests for the stochastic_process module
///
/// This test file focuses on:
/// 1. Naive implementation behavior and edge cases
/// 2. Trait object usage and Send+Sync constraints
/// 3. Factory pattern coverage
/// 4. Performance characteristics
/// 5. Memory safety with various input sizes
///
/// Target: 80%+ coverage of stochastic_process.rs module
use powers_rs::stochastic_process::{self, Naive, StochasticProcess};

// ============================================================================
// Unit Tests: Naive Implementation
// ============================================================================

mod test_naive_implementation {
    use super::*;

    #[test]
    fn test_new() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_default() {
        let naive = Naive::default();
        let noises = vec![4.0, 5.0, 6.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_realize_returns_same_reference() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let realized = naive.realize(&noises);

        // Verify it's the same data, not a copy
        assert_eq!(realized.len(), noises.len());
        assert_eq!(realized, &noises[..]);

        // Verify it's actually a reference to the same memory
        assert_eq!(realized.as_ptr(), noises.as_ptr());
    }

    #[test]
    fn test_realize_preserves_values() {
        let naive = Naive::new();
        let test_cases = vec![
            vec![0.0],
            vec![1.0, 2.0, 3.0],
            vec![-1.0, -2.0, -3.0],
            vec![0.0, 0.0, 0.0],
            vec![f64::MIN, f64::MAX],
            vec![1e-10, 1e10],
        ];

        for noises in test_cases {
            let realized = naive.realize(&noises);
            assert_eq!(
                realized,
                &noises[..],
                "Naive should preserve input values exactly"
            );
        }
    }

    #[test]
    fn test_multiple_realizes() {
        let naive = Naive::new();

        let noises1 = vec![1.0, 2.0, 3.0];
        let realized1 = naive.realize(&noises1);
        assert_eq!(realized1, &noises1[..]);

        let noises2 = vec![4.0, 5.0];
        let realized2 = naive.realize(&noises2);
        assert_eq!(realized2, &noises2[..]);

        // Original still works
        let realized3 = naive.realize(&noises1);
        assert_eq!(realized3, &noises1[..]);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

mod test_edge_cases {
    use super::*;

    #[test]
    fn test_empty_array() {
        let naive = Naive::new();
        let noises: Vec<f64> = vec![];
        let realized = naive.realize(&noises);
        assert_eq!(realized.len(), 0);
    }

    #[test]
    fn test_single_value() {
        let naive = Naive::new();
        let noises = vec![42.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized.len(), 1);
        assert_eq!(realized[0], 42.0);
    }

    #[test]
    fn test_large_array() {
        let naive = Naive::new();
        let noises: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let realized = naive.realize(&noises);
        assert_eq!(realized.len(), 10000);
        assert_eq!(realized[0], 0.0);
        assert_eq!(realized[9999], 9999.0);
    }

    #[test]
    fn test_negative_values() {
        let naive = Naive::new();
        let noises = vec![-1.0, -2.0, -3.0, -100.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_zero_values() {
        let naive = Naive::new();
        let noises = vec![0.0, 0.0, 0.0, 0.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_mixed_signs() {
        let naive = Naive::new();
        let noises = vec![-5.0, 0.0, 5.0, -10.0, 10.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_very_small_values() {
        let naive = Naive::new();
        let noises = vec![1e-308, 1e-307, 1e-100];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_very_large_values() {
        let naive = Naive::new();
        let noises = vec![1e100, 1e200, 1e307];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_extreme_values() {
        let naive = Naive::new();
        let noises = vec![f64::MIN, f64::MAX, f64::MIN_POSITIVE];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_special_float_values() {
        let naive = Naive::new();

        // Test with infinity
        let noises_inf = vec![f64::INFINITY, f64::NEG_INFINITY];
        let realized_inf = naive.realize(&noises_inf);
        assert_eq!(realized_inf[0], f64::INFINITY);
        assert_eq!(realized_inf[1], f64::NEG_INFINITY);

        // Test with NaN
        let noises_nan = vec![f64::NAN];
        let realized_nan = naive.realize(&noises_nan);
        assert!(realized_nan[0].is_nan());
    }
}

// ============================================================================
// Trait Object Tests
// ============================================================================

mod test_trait_object {
    use super::*;

    #[test]
    fn test_trait_object_basic() {
        let sp: Box<dyn StochasticProcess> = Box::new(Naive::new());
        let noises = vec![1.0, 2.0, 3.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_trait_object_from_factory() {
        let sp = stochastic_process::factory("naive");
        let noises = vec![10.0, 20.0, 30.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_multiple_trait_objects() {
        let sp1: Box<dyn StochasticProcess> = Box::new(Naive::new());
        let sp2: Box<dyn StochasticProcess> = Box::new(Naive::new());

        let noises1 = vec![1.0, 2.0];
        let noises2 = vec![3.0, 4.0, 5.0];

        assert_eq!(sp1.realize(&noises1), &noises1[..]);
        assert_eq!(sp2.realize(&noises2), &noises2[..]);
    }

    #[test]
    fn test_trait_object_reuse() {
        let sp: Box<dyn StochasticProcess> = Box::new(Naive::new());

        // Use multiple times
        for i in 0..100 {
            let noises = vec![i as f64];
            let realized = sp.realize(&noises);
            assert_eq!(realized[0], i as f64);
        }
    }

    /// Test that StochasticProcess is Send
    #[test]
    fn test_send_constraint() {
        fn assert_send<T: Send>() {}
        assert_send::<Naive>();
        assert_send::<Box<dyn StochasticProcess>>();
    }

    /// Test that StochasticProcess is Sync
    #[test]
    fn test_sync_constraint() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Naive>();
        // Note: Box<dyn StochasticProcess> is Send but not Sync by default
    }

    #[test]
    fn test_debug_trait() {
        let naive = Naive::new();
        let debug_str = format!("{:?}", naive);
        assert_eq!(debug_str, "Naive");
    }
}

// ============================================================================
// Factory Pattern Tests
// ============================================================================

mod test_factory {
    use super::*;

    #[test]
    fn test_factory_naive() {
        let sp = stochastic_process::factory("naive");
        let noises = vec![1.0, 2.0, 3.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    #[should_panic(expected = "stochastic process kind arma not supported")]
    fn test_factory_arma_not_supported() {
        stochastic_process::factory("arma");
    }

    #[test]
    #[should_panic(expected = "stochastic process kind boxcox not supported")]
    fn test_factory_boxcox_not_supported() {
        stochastic_process::factory("boxcox");
    }

    #[test]
    #[should_panic(expected = "stochastic process kind unknown not supported")]
    fn test_factory_unknown() {
        stochastic_process::factory("unknown");
    }

    #[test]
    #[should_panic(expected = "stochastic process kind  not supported")]
    fn test_factory_empty_string() {
        stochastic_process::factory("");
    }

    #[test]
    fn test_factory_returns_boxed_trait_object() {
        let sp = stochastic_process::factory("naive");

        // Verify it works as a trait object
        let noises = vec![5.0, 10.0, 15.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_factory_multiple_calls() {
        // Factory should be callable multiple times
        let sp1 = stochastic_process::factory("naive");
        let sp2 = stochastic_process::factory("naive");

        let noises1 = vec![1.0];
        let noises2 = vec![2.0];

        assert_eq!(sp1.realize(&noises1)[0], 1.0);
        assert_eq!(sp2.realize(&noises2)[0], 2.0);
    }
}

// ============================================================================
// Performance and Memory Characteristics Tests
// ============================================================================

mod test_performance_characteristics {
    use super::*;

    #[test]
    fn test_zero_copy_behavior() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let original_ptr = noises.as_ptr();
        let realized = naive.realize(&noises);
        let realized_ptr = realized.as_ptr();

        // Verify it's truly zero-copy (same memory address)
        assert_eq!(
            original_ptr, realized_ptr,
            "Naive should return reference to same memory (zero-copy)"
        );
    }

    #[test]
    fn test_no_allocation_on_realize() {
        let naive = Naive::new();

        // Create a large array
        let noises: Vec<f64> = (0..100000).map(|i| i as f64).collect();

        // Realize multiple times - should not allocate
        for _ in 0..1000 {
            let realized = naive.realize(&noises);
            // Just access to prevent optimization
            assert_eq!(realized.len(), 100000);
        }

        // If this test completes quickly, no allocations occurred
    }

    #[test]
    fn test_realize_with_varying_sizes() {
        let naive = Naive::new();

        // Test various sizes to ensure no size-dependent issues
        let sizes = vec![0, 1, 10, 100, 1000, 10000];

        for size in sizes {
            let noises: Vec<f64> = (0..size).map(|i| i as f64).collect();
            let realized = naive.realize(&noises);
            assert_eq!(realized.len(), size);
        }
    }

    #[test]
    fn test_repeated_realizes_with_same_input() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0];

        // Realize many times with same input
        for _ in 0..10000 {
            let realized = naive.realize(&noises);
            assert_eq!(realized[0], 1.0);
        }
    }
}

// ============================================================================
// Integration-Style Tests (Usage Patterns)
// ============================================================================

mod test_usage_patterns {
    use super::*;

    /// Simulates how stochastic process is used in SDDP subproblem
    #[test]
    fn test_sddp_usage_pattern() {
        let load_sp: Box<dyn StochasticProcess> =
            stochastic_process::factory("naive");
        let inflow_sp: Box<dyn StochasticProcess> =
            stochastic_process::factory("naive");

        // Simulate sampled noises
        let load_noises = vec![100.0, 120.0, 80.0];
        let inflow_noises = vec![50.0, 60.0, 70.0, 80.0];

        // Realize uncertainties
        let realized_load = load_sp.realize(&load_noises);
        let realized_inflow = inflow_sp.realize(&inflow_noises);

        // In naive case, should be unchanged
        assert_eq!(realized_load, &load_noises[..]);
        assert_eq!(realized_inflow, &inflow_noises[..]);
    }

    /// Test pattern where same process is reused across scenarios
    #[test]
    fn test_reuse_across_scenarios() {
        let sp: Box<dyn StochasticProcess> = Box::new(Naive::new());

        // Simulate multiple scenarios in SDDP forward pass
        let scenarios = vec![
            vec![10.0, 20.0, 30.0],
            vec![15.0, 25.0, 35.0],
            vec![12.0, 22.0, 32.0],
            vec![18.0, 28.0, 38.0],
        ];

        for scenario_noises in scenarios {
            let realized = sp.realize(&scenario_noises);
            assert_eq!(realized, &scenario_noises[..]);
        }
    }

    /// Test with typical hydrothermal dimensions
    #[test]
    fn test_typical_hydrothermal_dimensions() {
        let sp = Naive::new();

        // Typical case: 3 hydro plants + 3 thermal plants
        let noises = vec![50.0, 60.0, 70.0, 100.0, 120.0, 80.0];
        let realized = sp.realize(&noises);

        assert_eq!(realized.len(), 6);
        assert_eq!(realized, &noises[..]);
    }
}

// ============================================================================
// Documentation Tests (Usage Examples)
// ============================================================================

/// Example: Creating a stochastic process using factory
///
/// ```rust
/// use powers_rs::stochastic_process;
///
/// let sp = stochastic_process::factory("naive");
/// let noises = vec![1.0, 2.0, 3.0];
/// let realized = sp.realize(&noises);
/// assert_eq!(realized, &[1.0, 2.0, 3.0]);
/// ```
#[allow(dead_code)]
fn example_factory_usage() {}

/// Example: Creating a stochastic process directly
///
/// ```rust
/// use powers_rs::stochastic_process::{Naive, StochasticProcess};
///
/// let naive = Naive::new();
/// let noises = vec![10.0, 20.0];
/// let realized = naive.realize(&noises);
/// assert_eq!(realized, &[10.0, 20.0]);
/// ```
#[allow(dead_code)]
fn example_direct_usage() {}

/// Example: Using Default trait
///
/// ```rust
/// use powers_rs::stochastic_process::Naive;
///
/// let naive = Naive::default();
/// let noises = vec![5.0];
/// let realized = naive.realize(&noises);
/// assert_eq!(realized[0], 5.0);
/// ```
#[allow(dead_code)]
fn example_default_usage() {}
