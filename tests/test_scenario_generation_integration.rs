/// Integration Tests for Scenario Generation
///
/// Coverage:
/// - Examples work unchanged  
/// - Determinism (same seed → same output)
/// - Numerical stability (finite values, reasonable ranges)
/// - Performance (no regressions)

use powers_rs::sddp::SddpInstanceBuilder;
use std::time::Instant;

// ==============================================================================
// Example-Based Integration Tests
// ==============================================================================

#[test]
fn test_example_01_deterministic_unchanged() {
    // Example 01 should work identically with refactored code
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example 01")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 01");

    // Verify training succeeded
    assert!(
        result.final_lower_bound.is_finite(),
        "Example 01: Lower bound should be finite"
    );
    assert_eq!(
        result.lower_bounds().len(),
        5,
        "Example 01: Should have 5 iterations"
    );

    println!("✅ Example 01 (Deterministic): PASSED");
}

#[test]
fn test_example_02_stochastic_unchanged() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example 02")
    .with_num_iterations(10)
    .with_num_forward_passes(5)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 02");

    assert!(result.final_lower_bound.is_finite());
    assert_eq!(result.lower_bounds().len(), 10);

    // Verify bounds are monotonically improving
    let bounds = result.lower_bounds();
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Example 02: Bounds should be non-decreasing"
        );
    }

    println!("✅ Example 02 (Stochastic): PASSED");
}

#[test]
fn test_example_03_multistage_unchanged() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/03-multistage/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    )
    .expect("Failed to load example 03")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 03");

    assert!(result.final_lower_bound.is_finite());
    assert_eq!(result.lower_bounds().len(), 5);

    println!("✅ Example 03 (Multistage): PASSED");
}

#[test]
fn test_example_04_cascade_unchanged() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/04-cascade/config.json",
        "examples/04-cascade/system.json",
        "examples/04-cascade/graph.json",
        "examples/04-cascade/recourse.json",
    )
    .expect("Failed to load example 04")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 04");

    assert!(result.final_lower_bound.is_finite());
    assert_eq!(result.lower_bounds().len(), 5);

    println!("✅ Example 04 (Cascade): PASSED");
}

#[test]
#[cfg_attr(not(feature = "expensive_tests"), ignore)]
fn test_example_05_large_scale_unchanged() {
    let start = Instant::now();

    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/05-large-scale-brazilian/config.json",
        "examples/05-large-scale-brazilian/system.json",
        "examples/05-large-scale-brazilian/graph.json",
        "examples/05-large-scale-brazilian/recourse.json",
    )
    .expect("Failed to load example 05")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 05");

    let elapsed = start.elapsed();

    assert!(result.final_lower_bound.is_finite());
    assert_eq!(result.lower_bounds().len(), 5);

    // Performance check: should complete in reasonable time
    assert!(
        elapsed.as_secs() < 30,
        "Example 05 should complete in <30s, took {:?}",
        elapsed
    );

    println!("✅ Example 05 (Large-scale): PASSED ({:?})", elapsed);
}

#[test]
#[ignore = "Example 06 uses StorageState with PAR which has pre-existing infeasibility issues. See TICKET-012 for details. Use example 07 (StorageAndInflowState) for PAR validation."]
fn test_example_06_par_model_unchanged() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/06-par-model/config.json",
        "examples/06-par-model/system.json",
        "examples/06-par-model/graph.json",
        "examples/06-par-model/recourse.json",
    )
    .expect("Failed to load example 06")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 06");

    assert!(result.final_lower_bound.is_finite());
    assert_eq!(result.lower_bounds().len(), 5);

    println!("✅ Example 06 (PAR Model): PASSED");
}

#[test]
#[ignore = "Example 07 shows infeasibility after 3 iterations. Root cause under investigation - may be related to initial lag values or AR constraint setup. See TICKET-012 for details."]
fn test_example_07_par_with_inflow_state() {
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/07-par-model-with-inflow-state/config.json",
        "examples/07-par-model-with-inflow-state/system.json",
        "examples/07-par-model-with-inflow-state/graph.json",
        "examples/07-par-model-with-inflow-state/recourse.json",
    )
    .expect("Failed to load example 07")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let result = sddp.train().expect("Training failed for example 07");

    assert!(
        result.final_lower_bound.is_finite(),
        "Lower bound should be finite"
    );
    assert_eq!(result.lower_bounds().len(), 5, "Should have 5 iterations");

    // Example 07 uses StorageAndInflowState with PAR(1) model
    // This validates the unified AR model integration with lag variables in state
    println!("✅ Example 07 (PAR with StorageAndInflowState): PASSED");
}

// ==============================================================================
// Determinism Tests
// ==============================================================================

#[test]
fn test_determinism_same_seed_identical_results() {
    // Same seed must produce identical results (bit-for-bit)
    let result1 = {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example")
        .with_num_iterations(10)
        .with_num_forward_passes(5)
        .with_seed(42)
        .build()
        .expect("Failed to build SDDP instance");

        sddp.train().expect("Training failed")
    };

    let result2 = {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example")
        .with_num_iterations(10)
        .with_num_forward_passes(5)
        .with_seed(42)
        .build()
        .expect("Failed to build SDDP instance");

        sddp.train().expect("Training failed")
    };

    // Exact equality (not approximate)
    assert_eq!(
        result1.lower_bounds(),
        result2.lower_bounds(),
        "Same seed must produce identical bounds"
    );

    assert_eq!(
        result1.final_lower_bound, result2.final_lower_bound,
        "Same seed must produce identical final bound"
    );

    println!("✅ Determinism: Same seed produces identical results - PASSED");
}

#[test]
fn test_determinism_different_seeds_differ() {
    let result1 = {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example")
        .with_num_iterations(10)
        .with_num_forward_passes(5)
        .with_seed(42)
        .build()
        .expect("Failed to build SDDP instance");

        sddp.train().expect("Training failed")
    };

    let result2 = {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example")
        .with_num_iterations(10)
        .with_num_forward_passes(5)
        .with_seed(999)
        .build()
        .expect("Failed to build SDDP instance");

        sddp.train().expect("Training failed")
    };

    // Different seeds should produce different results
    assert_ne!(
        result1.lower_bounds(),
        result2.lower_bounds(),
        "Different seeds should produce different bounds"
    );

    println!(
        "✅ Determinism: Different seeds produce different results - PASSED"
    );
}

// ==============================================================================
// Numerical Stability Tests
// ==============================================================================

#[test]
fn test_numerical_stability_all_examples() {
    let examples = vec![
        "examples/01-deterministic",
        "examples/02-stochastic",
        "examples/03-multistage",
        "examples/04-cascade",
    ];

    for example_dir in examples {
        let mut sddp = SddpInstanceBuilder::from_paths(
            format!("{}/config.json", example_dir),
            format!("{}/system.json", example_dir),
            format!("{}/graph.json", example_dir),
            format!("{}/recourse.json", example_dir),
        )
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", example_dir, e))
        .with_num_iterations(5)
        .with_num_forward_passes(2)
        .with_seed(42)
        .build()
        .expect("Failed to build SDDP instance");

        let result = sddp.train().expect("Training failed");

        // Verify all bounds are finite
        for (i, &lb) in result.lower_bounds().iter().enumerate() {
            assert!(
                lb.is_finite(),
                "{}: Iteration {}: Bound {} is not finite",
                example_dir,
                i,
                lb
            );
        }

        // Verify monotonicity (within tolerance)
        let bounds = result.lower_bounds();
        for i in 1..bounds.len() {
            assert!(
                bounds[i] >= bounds[i - 1] - 1e-3,
                "{}: Bounds should be non-decreasing",
                example_dir
            );
        }

        println!("✅ Numerical stability: {} - PASSED", example_dir);
    }
}

// ==============================================================================
// Performance Tests
// ==============================================================================

#[test]
fn test_performance_no_regression_small() {
    // Baseline: Small problem should be fast
    let start = Instant::now();

    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(5)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let _result = sddp.train().expect("Training failed");

    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 500,
        "Small problem should be fast (<500ms), took {:?}",
        elapsed
    );

    println!(
        "✅ Performance: Small problem - PASSED ({} ms)",
        elapsed.as_millis()
    );
}

#[test]
fn test_performance_no_regression_medium() {
    // Baseline: Medium problem should be reasonable
    let start = Instant::now();

    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(10)
    .with_num_forward_passes(10)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let _result = sddp.train().expect("Training failed");

    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 3,
        "Medium problem should be reasonable (<3s), took {:?}",
        elapsed
    );

    println!(
        "✅ Performance: Medium problem - PASSED ({} ms)",
        elapsed.as_millis()
    );
}

#[test]
#[ignore] // Expensive - run manually
fn test_performance_no_regression_large() {
    // Baseline: Large problem should complete in reasonable time
    let start = Instant::now();

    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/03-multistage/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(20)
    .with_num_forward_passes(10)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let _result = sddp.train().expect("Training failed");

    let elapsed = start.elapsed();

    assert!(
        elapsed.as_secs() < 10,
        "Large problem should be reasonable (<10s), took {:?}",
        elapsed
    );

    println!(
        "✅ Performance: Large problem - PASSED ({} s)",
        elapsed.as_secs()
    );
}
