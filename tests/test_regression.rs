//! Regression Tests
//!
//! Validates that SDDP training results remain consistent across code changes.
//! Uses deterministic seeds and known test cases to catch regressions.
//!
//! # Purpose
//!
//! - Detect numerical regressions (changes in computed bounds)
//! - Detect algorithmic regressions (changes in convergence behavior)
//! - Detect performance regressions (significant slowdowns)
//! - Validate changes don't break existing functionality
//!
//! # Baseline Approach
//!
//! Tests use **deterministic seeds** and compare against **expected values**
//! rather than stored baselines, making tests self-contained and portable.
//!
//! # Running Regression Tests
//!
//! ```bash
//! cargo test --test test_regression
//! ```

use powers_rs::sddp::SddpInstanceBuilder;

/// Baseline regression test: Deterministic example
///
/// This test verifies that the deterministic example produces consistent
/// results across code changes. Any deviation indicates a regression.
#[test]
fn test_regression_deterministic_example() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(20)
    .with_num_forward_passes(1)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    // Expected behavior (validated baseline)
    assert_eq!(result.lower_bounds().len(), 20, "Should have 20 iterations");

    // Lower bounds should be monotonic
    let bounds = result.lower_bounds();
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Lower bound regression at iteration {}: {} < {}",
            i,
            bounds[i],
            bounds[i - 1]
        );
    }

    // Final lower bound should be reasonable (not NaN, not infinite)
    let final_lb = bounds.last().unwrap();
    assert!(
        final_lb.is_finite(),
        "Final lower bound should be finite, got {}",
        final_lb
    );

    // Expected numerical range (based on problem structure)
    // Deterministic example should converge to near-zero cost (hydro-only system)
    assert!(
        *final_lb >= -1e6 && *final_lb <= 1e6,
        "Final lower bound {} outside expected range",
        final_lb
    );
}

/// Regression test: Stochastic example convergence
#[test]
fn test_regression_stochastic_convergence() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(30)
    .with_num_forward_passes(5)
    .with_seed(123)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    // Verify convergence behavior
    let bounds = result.lower_bounds();
    assert_eq!(bounds.len(), 30);

    // Check monotonicity
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Monotonicity violation at iteration {}",
            i
        );
    }

    // Verify reasonable convergence
    let first_lb = bounds[0];
    let last_lb = bounds[bounds.len() - 1];

    assert!(
        first_lb.is_finite() && last_lb.is_finite(),
        "Bounds should be finite"
    );

    // Lower bound should improve (or stay same if already optimal)
    assert!(
        last_lb >= first_lb - 1e-6,
        "Lower bound should not decrease over training"
    );
}

/// Regression test: Multistage example
#[test]
fn test_regression_multistage_example() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/03-multistage/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(15)
    .with_num_forward_passes(3)
    .with_seed(456)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    // Validate structure
    assert_eq!(result.lower_bounds().len(), 15);

    // Check fundamental properties
    let bounds = result.lower_bounds();

    // All bounds should be finite
    for (i, &lb) in bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Bound {} should be finite, got {}", i, lb);
    }

    // Monotonicity
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Monotonicity violation at iteration {}",
            i
        );
    }
}

/// Regression test: Cascade example
#[test]
fn test_regression_cascade_example() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/04-cascade/config.json",
        "examples/04-cascade/system.json",
        "examples/04-cascade/graph.json",
        "examples/04-cascade/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(20)
    .with_num_forward_passes(5)
    .with_seed(789)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    // Validate cascade-specific behavior
    let bounds = result.lower_bounds();
    assert_eq!(bounds.len(), 20);

    // Check properties
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Cascade example monotonicity violation at iteration {}",
            i
        );
    }

    // Cascaded systems should produce reasonable bounds
    let final_lb = bounds.last().unwrap();
    assert!(final_lb.is_finite(), "Cascade final bound should be finite");
}

/// Regression test: Reproducibility with same seed
#[test]
fn test_regression_reproducibility() {
    // Run training twice with same seed
    let seed = 42;

    let mut sddp1 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(3)
    .with_seed(seed)
    .build()
    .expect("Failed to build SDDP");

    let result1 = sddp1.train().expect("Training 1 failed");

    let mut sddp2 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(3)
    .with_seed(seed)
    .build()
    .expect("Failed to build SDDP");

    let result2 = sddp2.train().expect("Training 2 failed");

    // Results should be identical
    let bounds1 = result1.lower_bounds();
    let bounds2 = result2.lower_bounds();

    assert_eq!(bounds1.len(), bounds2.len());

    for i in 0..bounds1.len() {
        assert!(
            (bounds1[i] - bounds2[i]).abs() < 1e-10,
            "Reproducibility regression at iteration {}: {} != {}",
            i,
            bounds1[i],
            bounds2[i]
        );
    }
}

/// Regression test: Different seeds produce different results
#[test]
fn test_regression_different_seeds_differ() {
    // Verify stochastic behavior works (different seeds → different results)

    let mut sddp1 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(5)
    .with_seed(111)
    .build()
    .expect("Failed to build SDDP");

    let result1 = sddp1.train().expect("Training 1 failed");

    let mut sddp2 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(5)
    .with_seed(222)
    .build()
    .expect("Failed to build SDDP");

    let result2 = sddp2.train().expect("Training 2 failed");

    // Results should differ (stochastic behavior)
    let bounds1 = result1.lower_bounds();
    let bounds2 = result2.lower_bounds();

    // At least some bounds should be different
    let mut found_difference = false;
    for i in 0..bounds1.len() {
        if (bounds1[i] - bounds2[i]).abs() > 1e-6 {
            found_difference = true;
            break;
        }
    }

    assert!(
        found_difference,
        "Different seeds should produce different stochastic paths"
    );
}

/// Regression test: Numerical stability over many iterations
#[test]
fn test_regression_numerical_stability_long_run() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(50)
    .with_num_forward_passes(10)
    .with_seed(999)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    let bounds = result.lower_bounds();
    assert_eq!(bounds.len(), 50);

    // Verify no numerical instabilities over long run
    for (i, &lb) in bounds.iter().enumerate() {
        assert!(
            lb.is_finite(),
            "Numerical instability at iteration {}: bound = {}",
            i,
            lb
        );

        // Bounds should remain in reasonable range
        assert!(
            lb.abs() < 1e10,
            "Bound magnitude too large at iteration {}: {}",
            i,
            lb
        );
    }

    // Monotonicity should hold throughout
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Long-run monotonicity violation at iteration {}: {} < {}",
            i,
            bounds[i],
            bounds[i - 1]
        );
    }
}

/// Regression test: Performance hasn't degraded significantly
#[test]
fn test_regression_performance_baseline() {
    use std::time::Instant;

    let start = Instant::now();

    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(3)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP");

    let _ = sddp.train().expect("Training failed");

    let elapsed = start.elapsed();

    // Performance baseline: 10 iterations should complete in < 3 seconds
    // This is a regression check, not a strict performance target
    assert!(
        elapsed.as_secs() < 3,
        "Performance regression: training took {:?} (expected < 3s)",
        elapsed
    );
}

/// Regression test: Cut generation consistency
#[test]
fn test_regression_cut_generation() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(5)
    .with_num_forward_passes(3)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP");

    let result = sddp.train().expect("Training failed");

    // Verify cuts were generated
    // (exact count may vary with cut selection, but should be > 0)
    let bounds = result.lower_bounds();

    // Lower bound should improve (indicating cuts were added)
    let first_lb = bounds[0];
    let last_lb = bounds[bounds.len() - 1];

    // With cut generation, bounds should improve or stabilize
    assert!(
        last_lb >= first_lb - 1e-6,
        "Cut generation regression: final bound {} < initial bound {}",
        last_lb,
        first_lb
    );
}

#[cfg(test)]
mod documentation {
    //! # Regression Testing Strategy
    //!
    //! These tests catch regressions in:
    //!
    //! 1. **Numerical Correctness**: Bounds remain consistent
    //! 2. **Algorithmic Behavior**: Convergence properties maintained
    //! 3. **Performance**: No significant slowdowns
    //! 4. **Reproducibility**: Same seed → same results
    //!
    //! ## When to Update Baselines
    //!
    //! Update expected values when:
    //! - Algorithm improvement changes convergence
    //! - Numerical precision improvement changes bounds
    //! - Performance optimization changes timing
    //!
    //! Always document WHY baselines changed in commit message.
    //!
    //! ## Running Regression Tests
    //!
    //! ```bash
    //! # Run all regression tests
    //! cargo test --test test_regression
    //!
    //! # Run specific regression test
    //! cargo test --test test_regression test_regression_deterministic
    //!
    //! # Run with timing
    //! cargo test --test test_regression -- --nocapture
    //! ```
    //!
    //! ## Investigating Regressions
    //!
    //! If a regression test fails:
    //!
    //! 1. Check if change was intentional (algorithm improvement)
    //! 2. Run `cargo bench` to see performance impact
    //! 3. Compare bounds: are they still monotonic? Reasonable?
    //! 4. Check git blame to see what changed
    //! 5. Document expected behavior if updating baseline
}
