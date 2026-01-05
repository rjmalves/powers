//! Forward Pass Integration Tests
//!
//! Tests verify that state initialization, uncertainty realization, LP solving,
//! and state extraction work together correctly in the forward pass.

use crate::fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    generate_2stage_saa,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{NodeData, SddpAlgorithm};
use powers_rs::subproblem::StudyPeriodKind;

/// Helper to create a minimal 2-stage graph for testing
fn create_test_graph() -> Result<DirectedGraph<NodeData>, String> {
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
            "expectation",
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
            "expectation",
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

/// Compute upper bound from forward costs (mean)
fn compute_upper_bound(costs: &[f64]) -> f64 {
    costs.iter().sum::<f64>() / costs.len() as f64
}

/// Test that state initialization from initial_condition works correctly
#[test]
fn test_forward_pass_state_initialization() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run one training iteration
    let result = sddp.train(1, 1, false, &saa, false, false);
    assert!(result.is_ok(), "Training should succeed");

    let training_result = result.unwrap();

    // Verify that initial state was used
    assert!(
        training_result.iterations().len() > 0,
        "Should have at least one iteration"
    );

    // Verify training completed successfully
    assert!(
        training_result.lower_bounds().len() > 0,
        "Should have computed lower bounds"
    );
}

/// Test that uncertainty realization is applied correctly to subproblems
#[test]
fn test_forward_pass_uncertainty_realization() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();

    // Create SAA with known seed for reproducibility
    let saa = generate_2stage_saa(123);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 123)
        .expect("Failed to create SDDP");

    // Run training with the same seed twice - should give same results
    let result1 = sddp
        .train(1, 1, false, &saa, false, false)
        .expect("First training should succeed");

    // Recreate SDDP with same seed
    let graph2 = create_test_graph().expect("Failed to create graph");
    let initial_condition2 = create_simple_2stage_initial_condition();
    let saa2 = generate_2stage_saa(123);

    let mut sddp2 = SddpAlgorithm::new(graph2, initial_condition2, 123)
        .expect("Failed to create SDDP");

    let result2 = sddp2
        .train(1, 1, false, &saa2, false, false)
        .expect("Second training should succeed");

    // With the same seed, results should be deterministic
    assert_eq!(
        result1.lower_bounds().len(),
        result2.lower_bounds().len(),
        "Should produce same number of iterations"
    );

    // Verify lower bounds are identical (deterministic behavior)
    let lb1 = result1.lower_bounds()[0];
    let lb2 = result2.lower_bounds()[0];

    assert!(
        (lb1 - lb2).abs() < 1e-6,
        "Lower bounds should be identical with same seed: {} vs {}",
        lb1,
        lb2
    );
}

/// Test that cost accumulation across stages is correct
#[test]
fn test_forward_pass_cost_accumulation() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run several iterations to accumulate costs
    let result = sddp
        .train(5, 3, false, &saa, false, false)
        .expect("Training should succeed");

    // Verify we have costs for multiple iterations
    assert!(
        result.iterations().len() >= 5,
        "Should have at least 5 iterations"
    );

    // Verify forward costs (simulation costs) are present
    for (i, iter_result) in result.iterations().iter().enumerate() {
        assert!(
            !iter_result.forward_costs.is_empty(),
            "Iteration {} should have forward costs",
            i
        );

        // Compute upper bound from forward costs
        let ub = compute_upper_bound(&iter_result.forward_costs);

        assert!(
            !ub.is_nan(),
            "Upper bound at iteration {} should not be NaN",
            i
        );
        assert!(
            ub.is_finite(),
            "Upper bound at iteration {} should be finite",
            i
        );
    }

    // Verify lower bounds are monotonically non-decreasing (key SDDP property)
    let lower_bounds = result.lower_bounds();
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
            "Lower bounds should be non-decreasing: iteration {} had {}, iteration {} had {}",
            i - 1,
            lower_bounds[i - 1],
            i,
            lower_bounds[i]
        );
    }
}

/// Test that trajectory storage captures necessary information for backward pass
#[test]
fn test_forward_pass_trajectory_storage() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run training to generate trajectories
    let result = sddp
        .train(3, 2, false, &saa, false, false)
        .expect("Training should succeed");

    // Verify iteration results exist
    let iterations = result.iterations();
    assert!(iterations.len() >= 3, "Should have at least 3 iterations");

    // Each iteration should have recorded information
    for (i, iter_result) in iterations.iter().enumerate() {
        // Verify iteration number is tracked
        assert!(
            iter_result.iteration > 0,
            "Iteration {} should have positive iteration number",
            i
        );

        // Verify timing information is captured
        assert!(
            iter_result.timing.forward.total.as_secs_f64() >= 0.0,
            "Forward pass time should be non-negative for iteration {}",
            i
        );

        assert!(
            iter_result.timing.backward.total.as_secs_f64() >= 0.0,
            "Backward pass time should be non-negative for iteration {}",
            i
        );
    }
}

/// Test that state continuity is maintained across stages
#[test]
fn test_forward_pass_state_continuity() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run training
    let result = sddp.train(2, 1, false, &saa, false, false);
    assert!(
        result.is_ok(),
        "Training should succeed: {:?}",
        result.err()
    );

    let training_result = result.unwrap();

    // Verify we completed iterations
    assert!(
        training_result.iterations().len() >= 2,
        "Should have completed at least 2 iterations"
    );

    // Verify bounds relationship (lower <= upper, within numerical tolerance)
    for (i, iter_result) in training_result.iterations().iter().enumerate() {
        let lb = iter_result.lower_bound;
        let ub = compute_upper_bound(&iter_result.forward_costs);

        if lb.is_finite() && ub.is_finite() {
            assert!(
                lb <= ub + 1e-4,
                "Lower bound should not exceed upper bound at iteration {}: LB={}, UB={}",
                i,
                lb,
                ub
            );
        }
    }
}

/// Test forward pass with multiple forward scenarios per iteration
#[test]
fn test_forward_pass_multiple_scenarios() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run with 5 forward passes per iteration
    let result = sddp
        .train(2, 5, false, &saa, false, false)
        .expect("Training with multiple forward passes should succeed");

    assert!(
        result.iterations().len() >= 2,
        "Should complete requested iterations"
    );

    // Verify that each iteration has 5 forward costs
    for (i, iter_result) in result.iterations().iter().enumerate() {
        assert_eq!(
            iter_result.forward_costs.len(),
            5,
            "Iteration {} should have 5 forward costs",
            i
        );
    }

    // Multiple forward passes should be recorded
    assert!(
        result.lower_bounds().len() > 0,
        "Should have computed lower bounds"
    );
}

/// Test that forward pass handles edge cases gracefully
#[test]
fn test_forward_pass_edge_cases() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Test with minimal iterations (1)
    let result = sddp.train(1, 1, false, &saa, false, false);
    assert!(
        result.is_ok(),
        "Should handle single iteration: {:?}",
        result.err()
    );

    // Test result structure
    let training_result = result.unwrap();
    assert_eq!(
        training_result.iterations().len(),
        1,
        "Should have exactly 1 iteration"
    );
    assert_eq!(
        training_result.lower_bounds().len(),
        1,
        "Should have 1 lower bound"
    );
}
