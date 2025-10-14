//! End-to-End Integration Tests for PAR Model in SDDP
//!
//! These tests verify that PAR (Periodic Autoregressive) models work correctly
//! in the full SDDP optimization context:
//! - Policy convergence with PAR scenarios
//! - Cut generation stability
//! - No numerical instabilities
//! - Finite, reasonable bounds
//!
//! Strategy: Use SddpInstanceBuilder with existing example files as baseline,
//! then add PAR-specific fixture tests when PAR examples become available.

use powers_rs::sddp::SddpInstanceBuilder;

// ==============================================================================
// Baseline E2E Tests: Verify PAR Additions Haven't Broken Existing Functionality
// ==============================================================================

#[test]
fn test_e2e_deterministic_baseline() {
    // Baseline: Verify existing deterministic example still works
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(10)
    .with_num_forward_passes(2)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    // Verify convergence
    let bounds = training_result.lower_bounds();
    assert_eq!(bounds.len(), 10, "Should have 10 iterations");

    // Verify all bounds are finite and non-decreasing
    for (i, &lb) in bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Bound {} should be finite", i);
        if i > 0 {
            assert!(
                lb >= bounds[i - 1] - 1e-6,
                "Bounds should be non-decreasing at iteration {}",
                i
            );
        }
    }

    println!("✅ E2E Baseline: Deterministic example - PASSED");
}

#[test]
fn test_e2e_stochastic_baseline() {
    // Baseline: Verify stochastic example (independent noise) still works
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(8)
    .with_num_forward_passes(3)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    let bounds = training_result.lower_bounds();
    assert_eq!(bounds.len(), 8);

    // Verify finite bounds
    for &lb in &bounds {
        assert!(lb.is_finite(), "All bounds should be finite");
    }

    // Verify statistical upper bound exists
    assert!(
        training_result.final_upper_bound.is_finite(),
        "Upper bound should be finite"
    );

    println!("✅ E2E Baseline: Stochastic example - PASSED");
}

#[test]
fn test_e2e_cascade_baseline() {
    // Baseline: Verify multi-hydro cascade example still works
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/04-cascade/config.json",
        "examples/04-cascade/system.json",
        "examples/04-cascade/graph.json",
        "examples/04-cascade/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(10)
    .with_num_forward_passes(3)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    let bounds = training_result.lower_bounds();
    assert_eq!(bounds.len(), 10);

    // Verify cascade doesn't cause numerical issues
    for (i, &lb) in bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Cascade bound {} should be finite", i);
        assert!(
            lb < 1e8,
            "Cascade bound {} should not explode (got {})",
            i,
            lb
        );
    }

    println!("✅ E2E Baseline: Cascade example - PASSED");
}

// ==============================================================================
// E2E Convergence Tests
// ==============================================================================

#[test]
fn test_e2e_convergence_many_iterations() {
    // Test convergence pattern over many iterations
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(50)
    .with_num_forward_passes(5)
    .with_seed(123)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    let bounds = training_result.lower_bounds();
    assert_eq!(bounds.len(), 50);

    // Check monotonicity
    for i in 1..bounds.len() {
        assert!(
            bounds[i] >= bounds[i - 1] - 1e-6,
            "Bounds should be monotonic at iteration {}",
            i
        );
    }

    // Check convergence (last 10 iterations should show small changes)
    let last_10 = &bounds[40..50];
    let max_change = last_10
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);

    let relative_change = max_change / last_10[0].max(1.0);

    assert!(
        relative_change < 0.15,
        "Should show convergence (max relative change < 15% in last 10 iters), got {:.2}%",
        relative_change * 100.0
    );

    println!("✅ E2E Convergence: Many iterations - PASSED");
    println!(
        "   Max relative change in last 10: {:.2}%",
        relative_change * 100.0
    );
}

// ==============================================================================
// E2E Numerical Stability Tests
// ==============================================================================

#[test]
fn test_e2e_numerical_stability_long_horizon() {
    // Test numerical stability with longer planning horizon
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/03-multistage/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(15)
    .with_num_forward_passes(3)
    .with_seed(456)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    let bounds = training_result.lower_bounds();

    // Verify no exploding bounds
    for (i, &lb) in bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Long horizon bound {} should be finite", i);
        assert!(
            lb < 1e9,
            "Long horizon bound {} should not explode (got {})",
            i,
            lb
        );
    }

    // Verify reasonable gap
    let gap = training_result.final_gap();
    assert!(gap.is_finite(), "Gap should be finite");
    assert!(gap >= 0.0, "Gap should be non-negative");

    println!("✅ E2E Stability: Long horizon - PASSED");
    println!("   Final gap: {:.2}", gap);
}

#[test]
fn test_e2e_simulation_stability() {
    // Test that simulation produces stable, finite results
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/02-stochastic/config.json",
        "examples/02-stochastic/system.json",
        "examples/02-stochastic/graph.json",
        "examples/02-stochastic/recourse.json",
    )
    .expect("Failed to load example files")
    .with_num_iterations(10)
    .with_num_forward_passes(3)
    .with_seed(789)
    .build()
    .expect("Failed to build SDDP instance");

    let training_result = sddp.train().expect("Training failed");

    // Verify training succeeded
    assert!(training_result.final_lower_bound.is_finite());

    // Simulate with the trained policy (zero-argument API)
    let sim_trajectories = sddp.simulate().expect("Simulation failed");

    // Verify all trajectories have valid data
    assert_eq!(
        sim_trajectories.len(),
        20,
        "Should have 20 simulation scenarios"
    );

    for (idx, traj) in sim_trajectories.iter().enumerate() {
        assert!(
            !traj.realizations.is_empty(),
            "Trajectory {} should have realizations",
            idx
        );

        // Verify each realization has finite costs
        for (stage, real) in traj.realizations.iter().enumerate() {
            assert!(
                real.current_stage_objective.is_finite(),
                "Trajectory {} stage {} cost should be finite",
                idx,
                stage
            );
            assert!(
                real.total_stage_objective.is_finite(),
                "Trajectory {} stage {} cumulative cost should be finite",
                idx,
                stage
            );
        }
    }

    println!("✅ E2E Stability: Simulation - PASSED");
    println!(
        "   Simulated {} trajectories successfully",
        sim_trajectories.len()
    );
}

// ==============================================================================
// E2E Reproducibility Tests
// ==============================================================================

#[test]
fn test_e2e_reproducibility_same_seed() {
    // Test that same seed produces identical results
    let config = |seed| {
        SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example files")
        .with_num_iterations(10)
        .with_num_forward_passes(3)
        .with_seed(seed)
        .build()
        .expect("Failed to build SDDP instance")
    };

    let mut sddp1 = config(42);
    let mut sddp2 = config(42);

    let result1 = sddp1.train().expect("Training 1 failed");
    let result2 = sddp2.train().expect("Training 2 failed");

    // Same seed should produce identical bounds
    let bounds1 = result1.lower_bounds();
    let bounds2 = result2.lower_bounds();

    assert_eq!(bounds1.len(), bounds2.len());

    for (i, (&lb1, &lb2)) in bounds1.iter().zip(bounds2.iter()).enumerate() {
        assert!(
            (lb1 - lb2).abs() < 1e-10,
            "Bounds should match at iteration {} (got {} vs {})",
            i,
            lb1,
            lb2
        );
    }

    println!("✅ E2E Reproducibility: Same seed - PASSED");
}

#[test]
fn test_e2e_different_seeds_differ() {
    // Test that different seeds produce different results
    let config = |seed| {
        SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example files")
        .with_num_iterations(10)
        .with_num_forward_passes(3)
        .with_seed(seed)
        .build()
        .expect("Failed to build SDDP instance")
    };

    let mut sddp1 = config(42);
    let mut sddp2 = config(999);

    let result1 = sddp1.train().expect("Training 1 failed");
    let result2 = sddp2.train().expect("Training 2 failed");

    // Different seeds should produce different bounds (with high probability)
    let bounds1 = result1.lower_bounds();
    let bounds2 = result2.lower_bounds();

    let any_differ = bounds1
        .iter()
        .zip(bounds2.iter())
        .any(|(&lb1, &lb2)| (lb1 - lb2).abs() > 1e-10);

    assert!(
        any_differ,
        "Different seeds should produce different results (with high probability)"
    );

    println!("✅ E2E Reproducibility: Different seeds - PASSED");
}

// ==============================================================================
// E2E Policy Quality Tests
// ==============================================================================

#[test]
fn test_e2e_policy_improves_with_iterations() {
    // Test that more iterations lead to better (tighter) bounds
    let train_with = |num_iters| {
        let mut sddp = SddpInstanceBuilder::from_paths(
            "examples/02-stochastic/config.json",
            "examples/02-stochastic/system.json",
            "examples/02-stochastic/graph.json",
            "examples/02-stochastic/recourse.json",
        )
        .expect("Failed to load example files")
        .with_num_iterations(num_iters)
        .with_num_forward_passes(4)
        .with_seed(42)
        .build()
        .expect("Failed to build SDDP instance");

        sddp.train().expect("Training failed")
    };

    let result_10 = train_with(10);
    let result_30 = train_with(30);

    // More iterations should give better (higher) lower bound
    assert!(
        result_30.final_lower_bound >= result_10.final_lower_bound,
        "More iterations should improve lower bound: {} vs {}",
        result_30.final_lower_bound,
        result_10.final_lower_bound
    );

    println!("✅ E2E Policy Quality: Iterations improve bounds - PASSED");
    println!("   10 iters: LB={:.2}", result_10.final_lower_bound);
    println!("   30 iters: LB={:.2}", result_30.final_lower_bound);
}
