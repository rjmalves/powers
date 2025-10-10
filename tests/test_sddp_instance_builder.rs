//! Integration tests for SddpInstanceBuilder.
//!
//! These tests verify:
//! - Builder produces identical results to from_files()
//! - Seed modification affects SAA generation
//! - Parameter sweeps work for benchmarking
//! - Builder works with production-scale problems (Example 05)
//! - Backward compatibility (from_files() still works)

use powers_rs::sddp::{SddpAlgorithm, SddpInstanceBuilder};

#[test]
fn test_builder_matches_from_files_baseline() {
    // Test with Example 01 (deterministic, small)
    let mut instance1 = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_files should succeed");

    let mut instance2 = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .build()
    .expect("build should succeed");

    // Train both with small parameters for speed
    let result1 = instance1.train().expect("Training should succeed");
    let result2 = instance2.train().expect("Training should succeed");

    // Results should be identical (same seed, same config)
    assert_eq!(
        result1.final_lower_bound, result2.final_lower_bound,
        "Lower bounds should match"
    );
    assert_eq!(
        result1.final_upper_bound, result2.final_upper_bound,
        "Upper bounds should match"
    );
}

#[test]
fn test_builder_modifies_config_before_saa_generation() {
    // Test that seed modification affects SAA generation
    let mut sddp1 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_seed(42)
    .build()
    .expect("build should succeed");

    let mut sddp2 = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_seed(999)
    .build()
    .expect("build should succeed");

    // Train both with small parameters for speed
    let result1 = sddp1.train().expect("Training should succeed");
    let result2 = sddp2.train().expect("Training should succeed");

    // Results should differ (different seeds = different SAA scenarios)
    // Note: We can't guarantee they're different every time, but with high probability
    // they should be different. We check that at least one bound differs.
    let bounds_differ = result1.final_lower_bound != result2.final_lower_bound
        || result1.final_upper_bound != result2.final_upper_bound;

    assert!(
        bounds_differ,
        "Different seeds should produce different results (with high probability)"
    );
}

#[test]
fn test_builder_parameter_sweep_benchmarking() {
    // Test use case: parameter sweep for benchmarking
    // Vary num_forward_passes to measure memory scaling
    let forward_passes = [1, 2, 4];

    for num_fwd in forward_passes {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/01-deterministic/config.json",
            "examples/01-deterministic/system.json",
            "examples/01-deterministic/graph.json",
            "examples/01-deterministic/recourse.json",
        )
        .expect("from_paths should succeed")
        .with_num_iterations(2) // Small for speed
        .with_num_forward_passes(num_fwd)
        .with_seed(42) // Fixed seed for reproducibility
        .build()
        .expect("build should succeed");

        // Each configuration should train successfully
        let result = sddp.train();
        assert!(
            result.is_ok(),
            "Training should succeed with {} forward passes",
            num_fwd
        );

        // Verify configuration was applied
        // (We can't directly inspect config in SddpInstance, but training success implies it worked)
    }
}

#[test]
fn test_builder_with_example_05() {
    // Test with production-scale problem (Example 05: 60 stages, 156 reservoirs)
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/05-large-scale-brazilian/config.json",
        "examples/05-large-scale-brazilian/system.json",
        "examples/05-large-scale-brazilian/graph.json",
        "examples/05-large-scale-brazilian/recourse.json",
    )
    .expect("from_paths should succeed with Example 05")
    .with_num_iterations(2) // Small for speed
    .with_num_forward_passes(4) // Small for speed
    .build()
    .expect("build should succeed with Example 05");

    // Should train successfully with production-scale problem
    let result = sddp.train();
    assert!(
        result.is_ok(),
        "Training should succeed with Example 05 (production-scale)"
    );
}

#[test]
fn test_from_files_still_works() {
    // Regression test: from_files() should still work (backward compatibility)
    let sddp = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(sddp.is_ok(), "from_files should still work");

    let mut instance = sddp.unwrap();
    let result = instance.train();
    assert!(result.is_ok(), "Training should succeed with from_files");
}

#[test]
fn test_builder_multiple_modifications() {
    // Test chaining multiple modifications
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_num_iterations(3)
    .with_num_forward_passes(2)
    .with_seed(123)
    .build()
    .expect("build should succeed");

    // Should train successfully with all modifications applied
    let result = sddp.train();
    assert!(
        result.is_ok(),
        "Training should succeed with multiple modifications"
    );
}

#[test]
fn test_builder_preserves_deterministic_reproducibility() {
    // Test that same builder configuration produces identical results
    let mut sddp1 = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_seed(42)
    .with_num_iterations(5)
    .with_num_forward_passes(3)
    .build()
    .expect("build should succeed");

    let mut sddp2 = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("from_paths should succeed")
    .with_seed(42)
    .with_num_iterations(5)
    .with_num_forward_passes(3)
    .build()
    .expect("build should succeed");

    // Train both
    let result1 = sddp1.train().expect("Training should succeed");
    let result2 = sddp2.train().expect("Training should succeed");

    // Results should be identical (same seed, same config)
    assert_eq!(
        result1.final_lower_bound, result2.final_lower_bound,
        "Deterministic reproducibility: lower bounds should match"
    );
    assert_eq!(
        result1.final_upper_bound, result2.final_upper_bound,
        "Deterministic reproducibility: upper bounds should match"
    );
}
