mod fixtures;

use fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    generate_2stage_saa,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{NodeData, SddpAlgorithm};
use powers_rs::subproblem::StudyPeriodKind;

/// Helper to create a simple 2-stage graph for testing
fn create_minimal_2stage_graph() -> Result<DirectedGraph<NodeData>, String> {
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
            std::sync::Arc::new(vec![]), // Empty uncertainty models for pre-study
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
            std::sync::Arc::new(vec![]), // Empty uncertainty models for simple test
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
fn test_train_with_zero_iterations() {
    // Test training with 0 iterations (edge case)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Training with 0 iterations returns an error (documented behavior)
    let result = sddp.train(0, 5, false, &saa, false, false, None);

    assert!(
        result.is_err(),
        "Training with 0 iterations should return an error"
    );

    // Verify error message mentions iterations
    let err = result.unwrap_err();
    let err_msg = format!("{:?}", err);
    assert!(
        err_msg.contains("iteration") || err_msg.contains("0"),
        "Error should mention iterations or zero"
    );
}

#[test]
fn test_train_with_zero_forward_passes() {
    // Test training with 0 forward passes per iteration (edge case)
    // Now properly returns an error instead of panicking (bug fixed!)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Training with 0 forward passes should return an error
    let result = sddp.train(1, 0, false, &saa, false, false, None);

    assert!(
        result.is_err(),
        "Training with 0 forward passes should return an error"
    );

    // Verify error message mentions forward passes
    let err = result.unwrap_err();
    assert!(
        err.contains("forward") || err.contains("0"),
        "Error should mention forward passes: {}",
        err
    );
}

#[test]
fn test_simulate_before_training() {
    // Test simulation before any training (should work with zero cuts)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Simulate without training - should work (just use initial policy)
    let result = sddp.simulate(5, &saa);

    assert!(result.is_ok(), "Simulation before training should succeed");
    let simulation_handlers = result.unwrap();
    assert_eq!(
        simulation_handlers.len(),
        5,
        "Should have 5 simulation handlers"
    );
}

#[test]
fn test_simulate_with_zero_scenarios() {
    // Test simulation with 0 scenarios (edge case)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Train first
    sddp.train(2, 5, false, &saa, false, false, None)
        .expect("Training should succeed");

    // Simulate with 0 scenarios
    let result = sddp.simulate(0, &saa);

    match result {
        Err(_) => {}
        Ok(handlers) => {
            assert!(
                handlers.is_empty(),
                "Should have no handlers for 0 scenarios"
            );
        }
    }
}

#[test]
fn test_train_with_single_iteration() {
    // Test training with exactly 1 iteration (edge case)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(1, 5, false, &saa, false, false, None);

    assert!(result.is_ok(), "Training with 1 iteration should succeed");
    let training_result = result.unwrap();
    assert_eq!(
        training_result.iterations().len(),
        1,
        "Should have 1 iteration"
    );
    assert_eq!(
        training_result.lower_bounds().len(),
        1,
        "Should have 1 lower bound"
    );

    // Lower bound should be finite
    assert!(
        training_result.lower_bounds()[0].is_finite(),
        "Lower bound should be finite"
    );

    // Final upper bound from final simulation should be finite
    assert!(
        training_result.statistical_upper_bound.is_finite(),
        "Final upper bound should be finite"
    );
}

#[test]
fn test_train_with_single_forward_pass() {
    // Test training with exactly 1 forward pass per iteration
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(3, 1, false, &saa, false, false, None);

    assert!(
        result.is_ok(),
        "Training with 1 forward pass should succeed"
    );
    let training_result = result.unwrap();
    assert_eq!(
        training_result.iterations().len(),
        3,
        "Should have 3 iterations"
    );

    // Final upper bound should be computed from final simulation
    assert!(
        training_result.statistical_upper_bound.is_finite(),
        "Final upper bound should be finite"
    );
}

#[test]
fn test_multiple_train_calls() {
    // Test calling train() multiple times on same algorithm (should accumulate cuts)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // First training run
    let result1 = sddp.train(2, 5, false, &saa, false, false, None);
    assert!(result1.is_ok(), "First training should succeed");

    // Second training run (should continue from where we left off)
    let result2 = sddp.train(2, 5, false, &saa, false, false, None);
    assert!(result2.is_ok(), "Second training should succeed");

    let training_result2 = result2.unwrap();
    assert_eq!(
        training_result2.iterations().len(),
        2,
        "Second training should have 2 iterations"
    );

    // Lower bound should be at least as good as before (cuts accumulated)
    // We can't easily compare to result1 final LB without storing it,
    // but we can check that bounds are reasonable
    assert!(
        training_result2.lower_bounds()[0].is_finite(),
        "Lower bound should be finite"
    );
}

#[test]
fn test_simulate_with_single_scenario() {
    // Test simulation with exactly 1 scenario
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Train first
    sddp.train(2, 5, false, &saa, false, false, None)
        .expect("Training should succeed");

    // Simulate with 1 scenario
    let result = sddp.simulate(1, &saa);

    assert!(result.is_ok(), "Simulation with 1 scenario should succeed");
    let handlers = result.unwrap();
    assert_eq!(handlers.len(), 1, "Should have 1 handler");
}

#[test]
fn test_convergence_on_first_iteration_trivial_problem() {
    // For a trivial problem, SDDP might converge on first iteration
    // (if problem is so simple that initial cuts are perfect)
    // This test documents that behavior works correctly
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(1, 10, false, &saa, false, false, None);
    assert!(result.is_ok(), "Single iteration training should succeed");

    let training_result = result.unwrap();

    // Check that we have valid bounds even after 1 iteration
    assert_eq!(training_result.iterations().len(), 1);
    assert!(training_result.lower_bounds()[0].is_finite());
    assert!(training_result.statistical_upper_bound.is_finite());

    // LB should be <= UB (basic validity check)
    let lb = training_result.lower_bounds()[0];
    let ub = training_result.statistical_upper_bound;
    assert!(
        lb <= ub + 1e-6, // Allow small numerical tolerance
        "Lower bound {} should be <= upper bound {}",
        lb,
        ub
    );
}

#[test]
fn test_no_convergence_after_max_iterations() {
    // Test that algorithm returns results even if not converged
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run for very few iterations (likely won't converge)
    let result = sddp.train(2, 3, false, &saa, false, false, None);
    assert!(
        result.is_ok(),
        "Training should succeed even without convergence"
    );

    let training_result = result.unwrap();
    assert_eq!(training_result.iterations().len(), 2);

    // Bounds should still be valid (even if not converged)
    for lb in training_result.lower_bounds() {
        assert!(lb.is_finite(), "Lower bound should be finite");
    }
    assert!(
        training_result.statistical_upper_bound.is_finite(),
        "Final upper bound should be finite"
    );
}

#[test]
fn test_bounds_monotonicity_validation() {
    // Test that lower bounds are non-decreasing (fundamental SDDP property)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(5, 10, false, &saa, false, false, None);
    assert!(result.is_ok(), "Training should succeed");

    let training_result = result.unwrap();
    let lower_bounds = training_result.lower_bounds();

    // Check monotonicity (with small tolerance for numerical precision)
    for i in 1..lower_bounds.len() {
        let prev_lb = lower_bounds[i - 1];
        let curr_lb = lower_bounds[i];

        // Allow for small numerical tolerance (1e-6 relative tolerance)
        let tolerance = 1e-6 * prev_lb.abs().max(1.0);
        assert!(
            curr_lb >= prev_lb - tolerance,
            "Lower bound should be non-decreasing: iteration {} LB={:.6} < iteration {} LB={:.6}",
            i - 1,
            prev_lb,
            i,
            curr_lb
        );
    }
}

#[test]
fn test_train_with_sequential_execution() {
    // Test training with 1 thread (sequential execution)
    // This is important for debugging and also tests the non-parallel path

    // Note: Rayon doesn't provide a direct API to set thread count per call,
    // so we test the default behavior. The algorithm should work regardless
    // of thread count.

    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(3, 5, false, &saa, false, false, None);
    assert!(result.is_ok(), "Sequential training should succeed");

    let training_result = result.unwrap();
    assert_eq!(training_result.iterations().len(), 3);

    // Results should be deterministic (given fixed seed)
    // We don't test for specific values, but verify they're reasonable
    for lb in training_result.lower_bounds() {
        assert!(lb.is_finite(), "Lower bound should be finite");
    }
}

#[test]
fn test_train_with_more_forward_passes_than_scenarios() {
    // Test with more forward passes than available scenarios
    // (over-subscription in parallel execution terms)
    let graph = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42); // SAA has some fixed number of scenarios

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Request 10 forward passes (requires resampling if SAA has fewer scenarios)
    let result = sddp.train(2, 10, false, &saa, false, false, None);

    assert!(
        result.is_ok(),
        "Training with more forward passes than scenarios should succeed"
    );
    let training_result = result.unwrap();
    assert_eq!(training_result.iterations().len(), 2);

    // Final upper bound should be computed correctly (from final simulation)
    assert!(training_result.statistical_upper_bound.is_finite());
}

#[test]
fn test_parallel_execution_determinism() {
    // Test that parallel execution with fixed seed produces deterministic results
    let graph1 = create_minimal_2stage_graph().expect("Failed to create graph");
    let graph2 = create_minimal_2stage_graph().expect("Failed to create graph");
    let initial_condition1 = create_simple_2stage_initial_condition();
    let initial_condition2 = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp1 = SddpAlgorithm::new(graph1, initial_condition1, 42)
        .expect("Failed to create SDDP");
    let mut sddp2 = SddpAlgorithm::new(graph2, initial_condition2, 42)
        .expect("Failed to create SDDP");

    // Run both with same parameters and seed
    let result1 = sddp1.train(3, 5, false, &saa, false, false, None);
    let result2 = sddp2.train(3, 5, false, &saa, false, false, None);

    assert!(result1.is_ok() && result2.is_ok(), "Both should succeed");

    let training_result1 = result1.unwrap();
    let training_result2 = result2.unwrap();

    // Lower bounds should be identical (deterministic with fixed seed)
    for i in 0..training_result1.iterations().len() {
        let lb1 = training_result1.lower_bounds()[i];
        let lb2 = training_result2.lower_bounds()[i];

        assert!(
            (lb1 - lb2).abs() < 1e-6,
            "Lower bounds should be deterministic: {} vs {}",
            lb1,
            lb2
        );
    }

    // Final upper bounds from simulation should also be deterministic
    let ub1 = training_result1.statistical_upper_bound;
    let ub2 = training_result2.statistical_upper_bound;

    let rel_diff = (ub1 - ub2).abs() / ub1.abs().max(1.0);
    assert!(
        rel_diff < 1e-6,
        "Final upper bounds should be deterministic: {} vs {} (rel_diff: {:.2e})",
        ub1,
        ub2,
        rel_diff
    );
}
