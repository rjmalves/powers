//! Risk Measure Integration Tests
//!
//! Tests how different risk measures affect SDDP training behavior.
//!
//! **Current Status**: Only "expectation" risk measure is implemented in the
//! risk_measure factory. CVaR and WorstCase tests are marked as ignored
//! until those risk measures are added to the factory.
//!
//! Key aspects tested:
//! - Training completes successfully with expectation risk measure
//! - Lower bounds remain monotonic (fundamental SDDP guarantee)
//! - Convergence behavior is consistent
//!
//! Note: Detailed risk measure mathematical properties (CVaR bounds, probability
//! adjustments, etc.) are tested in src/risk_measure.rs unit tests.

use crate::fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{NodeData, SddpAlgorithm};
use powers_rs::subproblem::StudyPeriodKind;

/// Helper to create a 2-stage graph with specified risk measure
fn create_test_graph_with_risk(
    risk_measure: &str,
) -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    let pre_study_id = graph
        .add_node(NodeData::new(
            -1,
            0,
            0,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            StudyPeriodKind::PreStudy,
            create_simple_2stage_system(),
            risk_measure,
            std::sync::Arc::new(vec![]),
            "storage",
            1,
        )?)
        .map_err(|e| format!("Failed to add PreStudy node: {:?}", e))?;

    let stage1_id = graph
        .add_node(NodeData::new(
            1,
            1,
            1,
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            StudyPeriodKind::Study,
            create_simple_2stage_system(),
            risk_measure,
            std::sync::Arc::new(vec![]),
            "storage",
            1,
        )?)
        .map_err(|e| format!("Failed to add Stage 1 node: {:?}", e))?;

    graph
        .add_edge(pre_study_id, stage1_id)
        .map_err(|e| format!("Failed to add edge: {:?}", e))?;

    Ok(graph)
}

#[test]
fn test_expectation_converges_normally() {
    let graph = create_test_graph_with_risk("expectation")
        .expect("Failed to create graph with expectation");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = crate::fixtures::generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP algorithm");

    // Run 10 training iterations with 5 forward scenarios each
    let result = sddp.train(10, 5, false, &saa).expect("Training failed");

    let iterations = result.iterations();

    // Should have 10 iterations
    assert_eq!(iterations.len(), 10);

    // Lower bounds should be monotonic (fundamental SDDP guarantee)
    for i in 1..iterations.len() {
        let lb_curr = iterations[i].lower_bound;
        let lb_prev = iterations[i - 1].lower_bound;
        assert!(
            lb_curr >= lb_prev - 1e-6,
            "Expectation: Lower bound decreased from {:.4} to {:.4} at iteration {}",
            lb_prev,
            lb_curr,
            i
        );
    }

    // All bounds should be finite
    for (i, iter) in iterations.iter().enumerate() {
        assert!(
            iter.lower_bound.is_finite(),
            "Iteration {} has non-finite lower bound",
            i
        );
    }
}

#[test]
fn test_expectation_with_many_iterations() {
    // Run more iterations to see convergence behavior
    let graph = create_test_graph_with_risk("expectation")
        .expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = crate::fixtures::generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(20, 8, false, &saa).expect("Training failed");

    let iterations = result.iterations();

    assert_eq!(iterations.len(), 20);

    // Verify monotonicity throughout all iterations
    for i in 1..iterations.len() {
        let lb = iterations[i].lower_bound;
        let lb_prev = iterations[i - 1].lower_bound;
        assert!(
            lb >= lb_prev - 1e-6,
            "LB decreased at iteration {}: {} -> {}",
            i,
            lb_prev,
            lb
        );
    }
}

#[test]
fn test_expectation_deterministic_behavior() {
    // With fixed seed, training should be deterministic
    let saa = crate::fixtures::generate_2stage_saa(123);

    // First run
    let graph1 = create_test_graph_with_risk("expectation").unwrap();
    let mut sddp1 = SddpAlgorithm::new(
        graph1,
        create_simple_2stage_initial_condition(),
        123,
    )
    .expect("Failed to create SDDP");

    let result1 = sddp1.train(5, 3, false, &saa).expect("Training failed");

    // Second run with same seed
    let graph2 = create_test_graph_with_risk("expectation").unwrap();
    let mut sddp2 = SddpAlgorithm::new(
        graph2,
        create_simple_2stage_initial_condition(),
        123,
    )
    .expect("Failed to create SDDP");

    let result2 = sddp2.train(5, 3, false, &saa).expect("Training failed");

    let iters1 = result1.iterations();
    let iters2 = result2.iterations();

    // Both should have same number of iterations
    assert_eq!(iters1.len(), iters2.len());

    // Bounds should be identical (deterministic)
    for i in 0..iters1.len() {
        let lb1 = iters1[i].lower_bound;
        let lb2 = iters2[i].lower_bound;
        assert!(
            (lb1 - lb2).abs() < 1e-10,
            "Bounds differ at iteration {}: {} vs {}",
            i,
            lb1,
            lb2
        );
    }
}

#[test]
fn test_expectation_with_cut_selection() {
    // Test that risk measure works with cut selection enabled
    let graph = create_test_graph_with_risk("expectation").unwrap();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = crate::fixtures::generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp
        .train(10, 5, true, &saa) // enable_cut_selection = true
        .expect("Training with cut selection failed");

    let iterations = result.iterations();
    assert_eq!(iterations.len(), 10);

    // Monotonicity should still hold with cut selection
    for i in 1..iterations.len() {
        let lb = iterations[i].lower_bound;
        let lb_prev = iterations[i - 1].lower_bound;
        assert!(
            lb >= lb_prev - 1e-6,
            "LB decreased with cut selection at iteration {}",
            i
        );
    }
}

#[test]
fn test_risk_measure_timing() {
    // Verify that risk measure computation doesn't cause performance issues
    let graph = create_test_graph_with_risk("expectation").unwrap();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = crate::fixtures::generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let start = std::time::Instant::now();

    let result = sddp.train(10, 5, false, &saa).expect("Training failed");

    let elapsed = start.elapsed();

    // Should complete quickly (< 1s for 10 iterations)
    assert!(
        elapsed.as_secs_f64() < 1.0,
        "Training took too long: {:.3}s",
        elapsed.as_secs_f64()
    );

    let iterations = result.iterations();
    let total_time: f64 = iterations
        .iter()
        .map(|it| {
            it.forward_timing.total_time.as_secs_f64()
                + it.backward_timing.total_time.as_secs_f64()
        })
        .sum();

    println!(
        "Expectation: {} iterations in {:.3}s (SDDP time: {:.3}s)",
        iterations.len(),
        elapsed.as_secs_f64(),
        total_time
    );
}

// ============================================================================
// IGNORED TESTS FOR FUTURE RISK MEASURES
// ============================================================================
// These tests are ignored because CVaR and WorstCase are not yet implemented
// in the risk_measure factory. They serve as specifications for when those
// features are added.

#[test]
#[ignore = "CVaR not yet implemented in risk_measure factory"]
fn test_cvar_converges_with_tail_focus() {
    // CVaR with α=0.3 focuses on worst 30% of scenarios
    // TODO: Implement when CVaR is added to factory
}

#[test]
#[ignore = "WorstCase not yet implemented in risk_measure factory"]
fn test_worst_case_converges_conservatively() {
    // WorstCase gives all weight to worst scenario
    // TODO: Implement when WorstCase is added to factory
}

#[test]
#[ignore = "Multiple risk measures not yet implemented"]
fn test_multiple_risk_measures_all_converge() {
    // Test that all risk measures successfully complete training
    // TODO: Implement when CVaR and WorstCase are added
}

#[test]
#[ignore = "CVaR not yet implemented"]
fn test_cvar_alpha_variations() {
    // Test different α values for CVaR
    // TODO: Implement when CVaR is added to factory
}

#[test]
#[ignore = "Multiple risk measures not yet implemented"]
fn test_risk_measures_produce_valid_bounds() {
    // Verify all risk measures produce finite, positive lower bounds
    // TODO: Implement when CVaR and WorstCase are added
}

#[test]
#[ignore = "Multiple risk measures not yet implemented"]
fn test_risk_measure_timing_overhead() {
    // Verify that different risk measures don't cause performance problems
    // TODO: Implement when CVaR and WorstCase are added
}

#[test]
#[ignore = "CVaR not yet implemented"]
fn test_expectation_vs_cvar_bounds() {
    // Compare convergence behavior between expectation and CVaR
    // TODO: Implement when CVaR is added to factory
}

#[test]
fn test_risk_measure_string_parsing() {
    // Test that the currently supported risk measure string is accepted
    let result = create_test_graph_with_risk("expectation");
    assert!(
        result.is_ok(),
        "Failed to create graph with valid risk measure: expectation"
    );

    // Note: Unsupported risk measures (cvar, worst-case) will cause NodeData
    // creation to panic when the factory is called, so we can't test invalid
    // strings here without catching panics.
}
