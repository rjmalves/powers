// Integration test for simple 2-stage SDDP problem

mod fixtures;

use fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    generate_2stage_saa,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{NodeData, SddpAlgorithm};
use powers_rs::subproblem::StudyPeriodKind;

mod utils;
#[allow(unused_imports)] // Some imports used only in specific tests
use utils::assertions::{
    assert_bounds_in_range, assert_convergence_quality,
    print_convergence_summary,
};

/// Helper to create a simple 2-stage graph for SDDP
///
/// Graph structure:
/// - Node 0 (PreStudy): Initial condition node
/// - Node 1 (Stage 1): First decision stage, deterministic
/// - Node 2 (Stage 2): Second decision stage, stochastic
///
/// Edges:
/// - 0 → 1 (Initial to first stage)
/// - 1 → 2 (First to second stage)
///
/// This is a simple path graph (no cyclic or Markovian structure).
/// This is fine for a test fixture - System creation is not in the hot path.
fn create_2stage_graph() -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    // Add PreStudy node (id=-1 by convention)
    let pre_study_id = graph
        .add_node(NodeData::new(
            -1,                     // node_id
            0,                      // stage_id
            0,                      // season_id
            "2024-01-01T00:00:00Z", // start_date
            "2024-01-01T00:00:00Z", // end_date (same as start for pre-study)
            StudyPeriodKind::PreStudy,
            create_simple_2stage_system(), // Create system for this node
            "expectation",                 // risk_measure
            "naive",                       // load_stochastic_process
            "naive",                       // inflow_stochastic_process
            "storage",                     // state_variables
        )?)
        .map_err(|e| format!("Failed to add PreStudy node: {:?}", e))?;

    // Add Stage 1 node (Study period)
    let stage1_id = graph
        .add_node(NodeData::new(
            1, // node_id
            1, // stage_id
            1, // season_id
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z", // 1 day duration
            StudyPeriodKind::Study,
            create_simple_2stage_system(), // Create system for this node
            "expectation",
            "naive",
            "naive",
            "storage",
        )?)
        .map_err(|e| format!("Failed to add Stage 1 node: {:?}", e))?;

    // Add Stage 2 node (Study period)
    let stage2_id = graph
        .add_node(NodeData::new(
            2, // node_id
            2, // stage_id
            2, // season_id
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z", // 1 day duration
            StudyPeriodKind::Study,
            create_simple_2stage_system(), // Create system for this node
            "expectation",
            "naive",
            "naive",
            "storage",
        )?)
        .map_err(|e| format!("Failed to add Stage 2 node: {:?}", e))?;

    // Add edges: PreStudy → Stage1 → Stage2
    graph
        .add_edge(pre_study_id, stage1_id)
        .map_err(|e| format!("Failed to add edge PreStudy->Stage1: {:?}", e))?;
    graph
        .add_edge(stage1_id, stage2_id)
        .map_err(|e| format!("Failed to add edge Stage1->Stage2: {:?}", e))?;

    Ok(graph)
}

/// Main integration test: 2-stage SDDP convergence
#[test]
fn test_sddp_2stage_convergence() {
    // ============================================================
    // SETUP PHASE
    // ============================================================

    // Create problem components
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    // Create SDDP algorithm instance
    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let num_iterations = 30;
    let num_forward_passes = 10;

    let result = sddp.train(num_iterations, num_forward_passes, &saa);

    // Training should complete without errors and return convergence data
    assert!(result.is_ok(), "SDDP training failed: {:?}", result.err());
    let training_result = result.unwrap();

    println!("✓ Training completed successfully");
    println!(
        "  Final gap: {:.4}, Best iteration: {}",
        training_result.final_gap(),
        training_result.best_iteration
    );

    assert_convergence_quality(&training_result)
        .expect("Convergence quality check failed");
}
/// Test that SDDP can handle multiple training runs with different seeds
#[test]
fn test_sddp_2stage_multiple_runs() {
    let graph1 = create_2stage_graph().expect("Failed to create graph");
    let graph2 = create_2stage_graph().expect("Failed to create graph");

    let ic1 = create_simple_2stage_initial_condition();
    let ic2 = create_simple_2stage_initial_condition();

    let saa1 = generate_2stage_saa(42);
    let saa2 = generate_2stage_saa(43); // Different seed

    // Run 1
    let mut sddp1 = SddpAlgorithm::new(graph1, ic1, 42)
        .expect("Failed to create SDDP algorithm");
    let result1 = sddp1.train(20, 10, &saa1); // Increased iterations for better statistical UB
    assert!(result1.is_ok(), "First run failed");
    let training1 = result1.unwrap();

    // Run 2 with different seed
    let mut sddp2 = SddpAlgorithm::new(graph2, ic2, 43)
        .expect("Failed to create SDDP algorithm");
    let result2 = sddp2.train(20, 10, &saa2); // Increased iterations for better statistical UB
    assert!(result2.is_ok(), "Second run failed");
    let training2 = result2.unwrap();

    // Validate both runs using convergence helper
    assert_convergence_quality(&training1)
        .expect("First run convergence quality check failed");
    assert_convergence_quality(&training2)
        .expect("Second run convergence quality check failed");
}

/// Test that SDDP handles edge case: single forward pass per iteration
#[test]
fn test_sddp_2stage_single_forward_pass() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    // Run with only 1 forward pass per iteration
    let result = sddp.train(20, 1, &saa);

    assert!(result.is_ok(), "Training with single forward pass failed");
    let training = result.unwrap();
    assert_eq!(training.iterations().len(), 20);
}

/// Test SDDP with more iterations to ensure stability
#[test]
fn test_sddp_2stage_extended_training() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    // Run for more iterations than needed (should converge and stabilize)
    let result = sddp.train(50, 10, &saa);

    assert!(result.is_ok(), "Extended training failed");
    let training = result.unwrap();

    // Extended training should complete all iterations
    assert_eq!(training.iterations().len(), 50);

    // Final gap should be reasonable (not diverging)
    assert!(training.final_gap() < 1e6, "Gap should not diverge");
}

/// Performance characteristic test: Measure runtime
///
/// This doesn't assert anything but prints timing information
/// for performance monitoring. Target: <10 seconds.
#[test]
fn test_sddp_2stage_performance() {
    use std::time::Instant;

    let start = Instant::now();

    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let train_start = Instant::now();
    let result = sddp.train(30, 10, &saa);
    let train_duration = train_start.elapsed();

    assert!(result.is_ok(), "Training failed");
    let training = result.unwrap();

    let total_duration = start.elapsed();

    println!("\n=== Performance Characteristics ===");
    println!("Setup time: {:?}", train_start.duration_since(start));
    println!("Training time: {:?}", train_duration);
    println!("Total time: {:?}", total_duration);
    println!("Iterations: 30");
    println!("Forward passes per iteration: 10");
    println!("Total solver calls: ~600 (2 stages × 300 passes)");
    println!("Final gap: {:.4}", training.final_gap());
    println!("Cuts generated: {}", training.num_cuts);

    // Target: <10 seconds total
    // This is a guideline, not a hard requirement (depends on hardware)
    if total_duration.as_secs() > 10 {
        println!("⚠️  Warning: Test took longer than 10 seconds");
        println!("   This may indicate performance regression");
    } else {
        println!("✓ Performance within target (<10 seconds)");
    }
}

/// Test that lower bounds are non-decreasing across iterations
///
/// **SDDP Property**: Lower bounds should monotonically increase (or stay constant)
/// as more cuts are added to the approximation.
///
/// **Tolerance**: 1e-6 to account for solver numerical precision
#[test]
fn test_convergence_monotonicity() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    // Train with enough iterations to observe convergence behavior
    let result = sddp.train(50, 10, &saa).expect("Training failed");

    // Extract lower bounds
    let lower_bounds = result.lower_bounds();

    // Check monotonicity with small tolerance for numerical errors
    let mut violations = 0;
    for window in lower_bounds.windows(2) {
        let prev = window[0];
        let curr = window[1];

        // Allow small numerical error (1e-6)
        if curr < prev - 1e-6 {
            violations += 1;
        }
    }

    assert_eq!(
        violations, 0,
        "Lower bound decreased {} times (should be monotonically non-decreasing)",
        violations
    );

    // Check that lower bound improved OR stayed at optimal (zero cost)
    let initial = lower_bounds[0];
    let final_lb = lower_bounds[lower_bounds.len() - 1];

    // Allow for problems where optimal cost is zero
    assert!(
        final_lb >= initial - 1e-6,
        "Lower bound should not decrease: {:.6} -> {:.6}",
        initial,
        final_lb
    );
}

/// Test that the gap decreases over iterations
/// Test: Gap convergence to zero
///
/// **Convergence Property**: The statistical gap (statistical_UB - LB) should
/// decrease as the algorithm refines the cost-to-go approximation.
#[test]
fn test_convergence_gap_decrease() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let result = sddp.train(50, 10, &saa).expect("Training failed");

    // Compute statistical gap: statistical_UB - final_LB
    let statistical_gap =
        result.statistical_upper_bound - result.final_lower_bound;

    // For problems with non-zero cost, gap should be reasonably small
    if result.final_lower_bound.abs() > 1e-6 {
        let relative_gap = statistical_gap / result.final_lower_bound.abs();

        // After 50 iterations with 10 forward passes each, gap should be < 20%
        assert!(
            relative_gap < 0.20,
            "Relative gap too large after 50 iterations: {:.2}%",
            relative_gap * 100.0
        );
    } else {
        assert!(
            statistical_gap < 1.0,
            "Zero-cost problem should have small statistical gap: {:.6}",
            statistical_gap
        );
    }
}

/// Test that bounds remain valid throughout training
///
/// **Validity Properties**:
/// 1. Bounds must be finite (not NaN, not Inf)
/// 2. Lower bound ≤ Statistical Upper bound (SDDP invariant)
/// 3. Lower bounds are monotonically non-decreasing
///
/// **Why this matters**: Invalid bounds indicate numerical instability,
/// solver failures, or implementation errors.
///
/// Note: Per-iteration "upper bounds" can be below LB due to sampling variance.
/// Only the statistical upper bound (average across all iterations) must be ≥ LB.
/// Test that bounds remain valid throughout training
///
/// **Validity Properties**:
/// 1. Bounds must be finite (not NaN, not Inf)
/// 2. Lower bound ≤ Statistical Upper bound (SDDP invariant)
/// 3. Lower bounds are monotonically non-decreasing
///
/// **Why this matters**: Invalid bounds indicate numerical instability,
/// solver failures, or implementation errors.
///
/// Note: Per-iteration "upper bounds" can be below LB due to sampling variance.
/// Only the statistical upper bound (average across all iterations) must be ≥ LB.
#[test]
fn test_convergence_bounds_validity() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let result = sddp.train(30, 10, &saa).expect("Training failed");

    // Check final bounds are finite
    assert!(
        result.final_lower_bound.is_finite(),
        "Final lower bound must be finite"
    );

    assert!(
        result.statistical_upper_bound.is_finite(),
        "Statistical upper bound must be finite"
    );

    // Check SDDP invariant: Lower bound ≤ Statistical upper bound
    // Note: Per-iteration upper bounds can be below LB due to sampling variance,
    // but the statistical average (across all iterations) must be above LB.
    assert!(
        result.final_lower_bound <= result.statistical_upper_bound + 1e-6,
        "Lower bound must be ≤ statistical upper bound: LB={:.6}, UB_stat={:.6}",
        result.final_lower_bound,
        result.statistical_upper_bound
    );

    // Check that all lower bounds are finite and monotonic
    let lower_bounds = result.lower_bounds();
    for (i, lb) in lower_bounds.iter().enumerate() {
        assert!(
            lb.is_finite(),
            "Lower bound at iteration {} not finite",
            i + 1
        );
    }

    for (i, window) in lower_bounds.windows(2).enumerate() {
        assert!(
            window[1] >= window[0] - 1e-6,
            "Lower bound decreased at iteration {}: {:.6} -> {:.6}",
            i + 1,
            window[0],
            window[1]
        );
    }
}

/// Test that convergence is stable (no wild oscillations)
///
/// Test: Training convergence stability
///
/// **Stability Property**: Lower bounds should be monotonically non-decreasing
/// (key SDDP property) and not oscillate.
///
/// **Why this matters**: Decreasing lower bounds or wild oscillations indicate
/// numerical instability, cut selection issues, or implementation errors.
///
/// Note: Per-iteration "upper bounds" (forward pass averages) are allowed to
/// fluctuate due to sampling variance - this is normal SDDP behavior.
#[test]
fn test_convergence_stability() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let result = sddp.train(50, 10, &saa).expect("Training failed");

    println!("\n=== Stability Test ===");
    println!("Checking lower bound stability (monotonicity)...");

    let lower_bounds = result.lower_bounds();

    // Check monotonicity: LB should never decrease
    let mut violations = 0;
    for (i, window) in lower_bounds.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];
        if curr < prev - 1e-6 {
            violations += 1;
            if violations <= 3 {
                // Only print first few violations
                println!(
                    "  ⚠️  Iteration {}: LB decreased {:.6} -> {:.6}",
                    i + 2,
                    prev,
                    curr
                );
            }
        }
    }

    assert_eq!(
        violations, 0,
        "Lower bound decreased {} times (should be monotonic)",
        violations
    );

    println!("✓ Lower bounds are stable (monotonically non-decreasing)");
    println!("  Initial LB: {:.6}", lower_bounds[0]);
    println!("  Final LB:   {:.6}", lower_bounds[lower_bounds.len() - 1]);
    println!(
        "  Total improvement: {:.6}",
        lower_bounds[lower_bounds.len() - 1] - lower_bounds[0]
    );
}
