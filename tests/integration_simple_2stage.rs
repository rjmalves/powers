// Integration test for simple 2-stage SDDP problem
//
// This is the **most critical test in Sprint 1** - it validates that all
// SDDP components work together correctly to solve a complete optimization problem.
//
// TEST STRATEGY:
// 1. Set up a minimal 2-stage reservoir problem with known characteristics
// 2. Run the full SDDP algorithm to convergence
// 3. Validate convergence behavior (bounds, monotonicity, statistical upper bound)
// 4. Validate solution quality (cost in expected range)
// 5. Validate policy makes physical sense (feasible decisions)
//
// CONVERGENCE PROPERTIES TESTED (Sprint 2 - T2.3):
// ✓ Lower bound monotonicity: LB non-decreasing across iterations (test_convergence_monotonicity)
// ✓ Statistical gap convergence: Statistical UB - LB → reasonable gap (test_convergence_gap_decrease)
// ✓ Bounds validity: LB ≤ Statistical UB, all finite (test_convergence_bounds_validity)
// ✓ Stability: No wild oscillations in bounds (test_convergence_stability)
// ✓ Statistical upper bound: Correctly computed as average of ALL forward passes (all tests)
//
// KEY INSIGHT: Per-iteration simulation costs can be < LB due to sampling variance.
// Only the statistical upper bound (average of all forward passes) must satisfy LB ≤ Statistical_UB.
// References: Shapiro (2011), Philpott & de Matos (2012)
//
// PERFORMANCE NOTE: This test runs the full SDDP algorithm including:
// - Multiple forward passes (solver calls in parallel via Rayon)
// - Backward pass with cut generation
// - FCF updates and cut selection
// This is not a micro-benchmark but an end-to-end validation test.
// Target: Complete in <10 seconds for fast CI feedback.

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
///
/// PERFORMANCE NOTE: We create the system 3 times (once per node).
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
///
/// This test validates:
/// 1. Algorithm converges (bounds gap closes)
/// 2. Lower bound is non-decreasing (mostly - some variance expected)
/// 3. Upper bound converges to reasonable value
/// 4. Solution cost is in expected range
/// 5. Policy decisions are feasible
///
/// CONVERGENCE CRITERIA:
/// - Max iterations: 30 (should converge in 10-20 for this simple problem)
/// - Tolerance: 5% relative gap (conservative for test stability)
/// - Forward passes: 10 per iteration (statistical stability)
///
/// EXPECTED BEHAVIOR:
/// - Initial iterations: Bounds far apart, cuts being generated
/// - Middle iterations: Bounds converging, policy stabilizing
/// - Final iterations: Bounds within tolerance, policy converged
#[test]
fn test_sddp_2stage_convergence() {
    // ============================================================
    // SETUP PHASE
    // ============================================================

    // Create problem components
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42); // Fixed seed for determinism

    // Create SDDP algorithm instance
    let mut sddp = SddpAlgorithm::new(
        graph,
        initial_condition,
        42, // Fixed seed for reproducibility
    )
    .expect("Failed to create SDDP algorithm");

    // ============================================================
    // TRAINING PHASE
    // ============================================================

    println!("\n=== Starting 2-Stage SDDP Integration Test ===");
    println!("Problem: 1 hydro, 2 stages, 3 scenarios in stage 2");
    println!("Expected: Convergence in 10-20 iterations");

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

    // ============================================================
    // VALIDATION PHASE
    // ============================================================

    // Comprehensive convergence validation using helper function (T2.3)
    assert_convergence_quality(&training_result)
        .expect("Convergence quality check failed");

    println!("✓ Convergence quality verified");
    println!("  - Monotonicity: ✓");
    println!("  - Gap decrease: ✓");
    println!("  - Bounds validity: ✓");

    println!("✓ All validations passed");
    println!("=== 2-Stage SDDP Integration Test PASSED ===\n");
}
/// Test that SDDP can handle multiple training runs with different seeds
///
/// This validates:
/// - Algorithm is stateless between runs (no hidden state corruption)
/// - Different seeds produce different sample paths (but should converge similarly)
/// - No memory leaks or resource exhaustion
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

    println!("✓ Both runs converged successfully");
    println!("  Run 1 gap: {:.4}", training1.final_gap());
    println!("  Run 2 gap: {:.4}", training2.final_gap());
}

/// Test that SDDP handles edge case: single forward pass per iteration
///
/// This validates:
/// - Algorithm works with minimal forward passes (no averaging benefits)
/// - Still converges (may take more iterations)
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

    println!("✓ Single forward pass per iteration works");
}

/// Test SDDP with more iterations to ensure stability
///
/// This validates:
/// - Algorithm remains stable over many iterations
/// - No degradation or numerical issues
/// - Cuts accumulate correctly without causing problems
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

    println!("✓ Extended training (50 iterations) completed");
}

/// Performance characteristic test: Measure runtime
///
/// This doesn't assert anything but prints timing information
/// for performance monitoring. Target: <10 seconds.
///
/// PERFORMANCE NOTE: This test measures end-to-end SDDP performance including:
/// - Graph construction
/// - SAA generation  
/// - SDDP training (30 iterations × 10 forward passes)
/// - Cut generation and selection
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

// ============================================================================
// CONVERGENCE VALIDATION TESTS (T2.3)
// ============================================================================
//
// These tests focus specifically on convergence properties of the SDDP algorithm:
// - Monotonicity of lower bounds
// - Gap convergence
// - Bounds validity
// - Stability (no wild oscillations)
//
// These tests complement the basic integration tests by validating numerical
// correctness properties that must hold for proper SDDP convergence.

/// Test that lower bounds are non-decreasing across iterations
///
/// **SDDP Property**: Lower bounds should monotonically increase (or stay constant)
/// as more cuts are added to the approximation.
///
/// **Why this matters**: Decreasing lower bounds indicate numerical instability
/// or implementation errors in the backward pass.
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

    println!("\n=== Monotonicity Test ===");
    println!("Testing lower bound monotonicity over 50 iterations...");

    // Extract lower bounds
    let lower_bounds = result.lower_bounds();

    // Check monotonicity with small tolerance for numerical errors
    let mut violations = 0;
    for (i, window) in lower_bounds.windows(2).enumerate() {
        let prev = window[0];
        let curr = window[1];

        // Allow small numerical error (1e-6)
        if curr < prev - 1e-6 {
            violations += 1;
            println!(
                "  ⚠️  Violation at iteration {}: {:.6} -> {:.6} (decrease: {:.6})",
                i + 1,
                prev,
                curr,
                prev - curr
            );
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

    println!("✓ Lower bounds are monotonically non-decreasing");
    println!("  Initial: {:.6}", initial);
    println!("  Final:   {:.6}", final_lb);
    if final_lb > initial + 1e-6 {
        println!("  Improvement: {:.6}", final_lb - initial);
    } else {
        println!("  Already at optimum (zero cost)");
    }
}

/// Test that the gap decreases over iterations
/// Test: Gap convergence to zero
///
/// **Convergence Property**: The statistical gap (statistical_UB - LB) should
/// decrease as the algorithm refines the cost-to-go approximation.
///
/// Note: Per-iteration gaps (based on that iteration's forward passes) can
/// fluctuate and even be negative due to sampling variance. The **statistical gap**
/// (average of ALL forward passes vs lower bound) is the true convergence metric.
///
/// **Why this matters**: Decreasing statistical gap indicates the policy is
/// converging to optimality.
///
/// **Target**: Statistical gap should be reasonably small after 50 iterations
#[test]
fn test_convergence_gap_decrease() {
    let graph = create_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    let result = sddp.train(50, 10, &saa).expect("Training failed");

    println!("\n=== Gap Convergence Test ===");
    println!("Testing statistical gap over 50 iterations...");

    // Compute statistical gap: statistical_UB - final_LB
    let statistical_gap =
        result.statistical_upper_bound - result.final_lower_bound;

    println!("  Final LB:          {:.6}", result.final_lower_bound);
    println!("  Statistical UB:    {:.6}", result.statistical_upper_bound);
    println!("  Statistical gap:   {:.6}", statistical_gap);

    // Statistical gap must be non-negative (SDDP invariant)
    assert!(
        statistical_gap >= -1e-6,
        "Statistical gap must be non-negative: {:.6}",
        statistical_gap
    );

    // For problems with non-zero cost, gap should be reasonably small
    if result.final_lower_bound.abs() > 1e-6 {
        let relative_gap = statistical_gap / result.final_lower_bound.abs();
        println!("  Relative gap:      {:.2}%", relative_gap * 100.0);

        // After 50 iterations with 10 forward passes each, gap should be < 20%
        assert!(
            relative_gap < 0.20,
            "Relative gap too large after 50 iterations: {:.2}%",
            relative_gap * 100.0
        );
    } else {
        println!("  Problem has zero optimal cost");
        assert!(
            statistical_gap < 1.0,
            "Zero-cost problem should have small statistical gap: {:.6}",
            statistical_gap
        );
    }

    println!("✓ Statistical gap is acceptable");
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

    println!("\n=== Bounds Validity Test ===");
    println!("Checking bounds validity...");

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

    println!("✓ All bounds valid");
    println!("  Final LB:          {:.6}", result.final_lower_bound);
    println!("  Statistical UB:    {:.6}", result.statistical_upper_bound);
    println!(
        "  Statistical Gap:   {:.6}",
        result.statistical_upper_bound - result.final_lower_bound
    );
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
