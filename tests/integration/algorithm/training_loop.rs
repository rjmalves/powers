//! Training Loop Integration Tests
//!
//! Tests the main SDDP training loop and iteration management.
//! Verifies that the complete iteration cycle works correctly:
//! forward passes → backward pass → bounds computation → convergence checking.

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

/// Test that training iterations follow the correct sequence:
/// forward passes → backward pass → bounds computation
#[test]
fn test_training_iteration_structure() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run several iterations
    let result = sddp
        .train(5, 3, false, &saa, false, false)
        .expect("Training should succeed");

    let iterations = result.iterations();

    // Verify we got the requested number of iterations
    assert_eq!(
        iterations.len(),
        5,
        "Should have exactly 5 iterations as requested"
    );

    // Verify each iteration has the complete structure
    for (i, iter_result) in iterations.iter().enumerate() {
        // Forward passes executed
        assert!(
            !iter_result.forward_costs.is_empty(),
            "Iteration {} should have forward costs from forward passes",
            i
        );
        assert_eq!(
            iter_result.forward_costs.len(),
            3,
            "Iteration {} should have 3 forward pass costs",
            i
        );

        // Forward timing recorded
        assert!(
            iter_result.timing.forward.total.as_nanos() > 0,
            "Iteration {} should have forward pass timing",
            i
        );

        // Backward pass executed
        assert!(
            iter_result.timing.backward.total.as_nanos() > 0,
            "Iteration {} should have backward pass timing",
            i
        );

        // Bounds computed
        assert!(
            iter_result.lower_bound.is_finite(),
            "Iteration {} should have finite lower bound",
            i
        );

        // Forward costs are valid
        for fc in &iter_result.forward_costs {
            assert!(
                fc.is_finite(),
                "Iteration {} forward costs should be finite",
                i
            );
        }
    }
}

/// Test that lower bound is computed correctly from the root node LP objective
/// after cuts have been added
#[test]
fn test_lower_bound_computation() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp
        .train(10, 2, false, &saa, false, false)
        .expect("Training should succeed");

    let lower_bounds = result.lower_bounds();

    // Verify we have lower bounds for all iterations
    assert_eq!(
        lower_bounds.len(),
        10,
        "Should have 10 lower bounds for 10 iterations"
    );

    // Verify lower bounds are monotonically non-decreasing
    // (key property: cuts improve the approximation)
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
            "Lower bound should be non-decreasing: iteration {} = {}, iteration {} = {}",
            i - 1,
            lower_bounds[i - 1],
            i,
            lower_bounds[i]
        );
    }

    // Verify lower bounds are finite and not NaN
    for (i, lb) in lower_bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Lower bound {} should be finite", i);
        assert!(!lb.is_nan(), "Lower bound {} should not be NaN", i);
    }

    // Verify final lower bound is available
    assert_eq!(
        result.final_lower_bound,
        lower_bounds[lower_bounds.len() - 1],
        "Final lower bound should match last iteration"
    );
}

/// Test that upper bound is computed correctly from forward pass simulation costs
#[test]
fn test_upper_bound_computation() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp
        .train(8, 5, false, &saa, false, false)
        .expect("Training should succeed");

    let iterations = result.iterations();

    // Verify forward costs are tracked for each iteration
    for (i, iter_result) in iterations.iter().enumerate() {
        assert!(
            !iter_result.forward_costs.is_empty(),
            "Iteration {} should have forward costs",
            i
        );

        // Verify all forward costs are finite
        for fc in &iter_result.forward_costs {
            assert!(
                fc.is_finite(),
                "Iteration {} forward costs should be finite",
                i
            );
        }
    }

    // Verify statistical upper bound across all iterations
    let all_costs: Vec<f64> = iterations
        .iter()
        .flat_map(|iter| iter.forward_costs.iter().copied())
        .collect();

    let statistical_ub = all_costs.iter().sum::<f64>() / all_costs.len() as f64;

    assert!(
        (result.statistical_upper_bound - statistical_ub).abs() < 1e-6,
        "Statistical upper bound {} should match average of all forward costs {}",
        result.statistical_upper_bound,
        statistical_ub
    );

    // Verify final upper bound is available
    assert!(
        result.statistical_upper_bound.is_finite(),
        "Final upper bound should be finite"
    );

    // Verify best upper bound is available
    assert!(
        result.best_upper_bound.is_finite(),
        "Best upper bound should be finite"
    );
}

/// Test that iteration results track all required metrics
#[test]
fn test_iteration_result_tracking() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp
        .train(3, 2, false, &saa, false, false)
        .expect("Training should succeed");

    let iterations = result.iterations();

    for (i, iter_result) in iterations.iter().enumerate() {
        // Bounds
        assert!(
            iter_result.lower_bound.is_finite(),
            "Iteration {} should track lower bound",
            i
        );

        // Forward pass data
        assert_eq!(
            iter_result.forward_costs.len(),
            2,
            "Iteration {} should track all forward costs",
            i
        );
        assert!(
            iter_result.timing.forward.total.as_nanos() > 0,
            "Iteration {} should track forward timing",
            i
        );

        // Backward pass data
        assert!(
            iter_result.timing.backward.total.as_nanos() > 0,
            "Iteration {} should track backward timing",
            i
        );

        // Total iteration time
        assert!(
            iter_result.timing.total.as_nanos() > 0,
            "Iteration {} should track total time",
            i
        );

        // Total time should be at least forward + backward time
        let component_time =
            iter_result.timing.forward.total.as_secs_f64()
                + iter_result.timing.backward.total.as_secs_f64();

        assert!(
            iter_result.timing.total.as_secs_f64() >= component_time - 1e-6,
            "Iteration {} total time should be >= forward + backward time",
            i
        );
    }
}

/// Test that training computes final gap correctly
#[test]
fn test_convergence_gap_computation() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // For a simple 2-stage deterministic problem, convergence may happen quickly
    // Run enough iterations that we might see convergence
    let result = sddp
        .train(20, 1, false, &saa, false, false)
        .expect("Training should succeed");

    // Verify final gap is computed correctly
    let computed_gap =
        result.statistical_upper_bound - result.final_lower_bound;
    let final_gap = result.final_gap();

    assert!(
        (final_gap - computed_gap).abs() < 1e-10,
        "Final gap {} should match UB - LB = {}",
        final_gap,
        computed_gap
    );

    // Gap should be non-negative (UB >= LB)
    assert!(
        final_gap >= -1e-6,
        "Final gap should be non-negative: {}",
        final_gap
    );
}

/// Test that training stops at max iterations
#[test]
fn test_convergence_detection_max_iterations() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Set a specific number of iterations
    let max_iterations = 7;

    let result = sddp
        .train(max_iterations, 2, false, &saa, false, false)
        .expect("Training should succeed");

    // Verify we got exactly max_iterations
    assert_eq!(
        result.iterations().len(),
        max_iterations,
        "Training should complete exactly {} iterations",
        max_iterations
    );

    // Verify all iterations completed successfully
    for (i, iter_result) in result.iterations().iter().enumerate() {
        assert!(
            iter_result.lower_bound.is_finite(),
            "Iteration {} should have completed",
            i
        );
    }
}

/// Test complete training workflow with realistic iteration count
#[test]
fn test_complete_training_workflow() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run realistic training
    let result = sddp
        .train(15, 4, false, &saa, false, false)
        .expect("Training should succeed");

    // Verify complete workflow
    assert_eq!(
        result.iterations().len(),
        15,
        "Should complete 15 iterations"
    );

    // Verify bounds converge or improve
    let lower_bounds = result.lower_bounds();
    assert!(
        lower_bounds[lower_bounds.len() - 1] >= lower_bounds[0] - 1e-6,
        "Lower bound should improve or stay same from first to last iteration"
    );

    // Verify final statistics are available
    assert!(result.final_lower_bound.is_finite(), "Final LB available");
    assert!(
        result.statistical_upper_bound.is_finite(),
        "Statistical UB available"
    );
    assert!(result.best_upper_bound.is_finite(), "Best UB available");

    // Verify total training time is positive
    assert!(
        result.total_time.as_secs_f64() > 0.0,
        "Total training time should be positive"
    );

    // Verify cuts were tracked (may be 0 for simple problems)
    let _cuts = result.num_cuts; // Field exists

    // Verify iteration with best UB is tracked
    assert!(
        result.best_iteration < result.iterations().len(),
        "Best iteration should be within bounds"
    );
}

/// Test training with cut selection enabled
#[test]
fn test_training_with_cut_selection() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run training WITH cut selection
    let result = sddp
        .train(10, 3, true, &saa, false, false)
        .expect("Training with cut selection should succeed");

    // Verify cut selection statistics are tracked
    let iterations = result.iterations();

    // Training should still complete successfully with cut selection
    assert_eq!(
        iterations.len(),
        10,
        "Should complete all iterations with cut selection"
    );

    // Bounds should still be valid
    let lower_bounds = result.lower_bounds();
    for (i, lb) in lower_bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Lower bound {} should be finite", i);
    }
}

/// Test training with different forward pass counts
#[test]
fn test_training_variable_forward_passes() {
    // Test with 1 forward pass
    {
        let graph = create_test_graph().expect("Failed to create graph");
        let initial_condition = create_simple_2stage_initial_condition();
        let saa = generate_2stage_saa(42);

        let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
            .expect("Failed to create SDDP");

        let result = sddp
            .train(5, 1, false, &saa, false, false)
            .expect("Training with 1 forward pass should succeed");

        for iter in result.iterations() {
            assert_eq!(
                iter.forward_costs.len(),
                1,
                "Should have 1 forward cost per iteration"
            );
        }
    }

    // Test with 10 forward passes
    {
        let graph = create_test_graph().expect("Failed to create graph");
        let initial_condition = create_simple_2stage_initial_condition();
        let saa = generate_2stage_saa(42);

        let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
            .expect("Failed to create SDDP");

        let result = sddp
            .train(5, 10, false, &saa, false, false)
            .expect("Training with 10 forward passes should succeed");

        for iter in result.iterations() {
            assert_eq!(
                iter.forward_costs.len(),
                10,
                "Should have 10 forward costs per iteration"
            );
        }
    }
}

/// Test that training is deterministic with fixed seed
#[test]
fn test_training_determinism() {
    // First run
    let graph1 = create_test_graph().expect("Failed to create graph");
    let initial_condition1 = create_simple_2stage_initial_condition();
    let saa1 = generate_2stage_saa(12345);

    let mut sddp1 = SddpAlgorithm::new(graph1, initial_condition1, 12345)
        .expect("Failed to create SDDP");

    let result1 = sddp1
        .train(5, 2, false, &saa1, false, false)
        .expect("First training should succeed");

    // Second run with same seed
    let graph2 = create_test_graph().expect("Failed to create graph");
    let initial_condition2 = create_simple_2stage_initial_condition();
    let saa2 = generate_2stage_saa(12345);

    let mut sddp2 = SddpAlgorithm::new(graph2, initial_condition2, 12345)
        .expect("Failed to create SDDP");

    let result2 = sddp2
        .train(5, 2, false, &saa2, false, false)
        .expect("Second training should succeed");

    // Results should be identical
    let lb1 = result1.lower_bounds();
    let lb2 = result2.lower_bounds();

    assert_eq!(
        lb1.len(),
        lb2.len(),
        "Should have same number of iterations"
    );

    for (i, (l1, l2)) in lb1.iter().zip(lb2.iter()).enumerate() {
        assert!(
            (l1 - l2).abs() < 1e-10,
            "Lower bound {} should be identical: {} vs {}",
            i,
            l1,
            l2
        );
    }
}
