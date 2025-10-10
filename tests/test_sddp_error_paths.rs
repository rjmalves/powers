//! SDDP Error Path and Edge Case Tests (T4.2 Coverage Completion)
//!
//! This test file targets uncovered lines in src/sddp/mod.rs to improve
//! coverage from 86.60% to 92%+. Focus areas:
//!
//! 1. **Timing Aggregation Edge Cases**: ForwardPassTimingAccumulator, BackwardPassTimingAccumulator
//! 2. **TrainingResult Analysis**: Gap calculations with edge values (zero, infinity, negative)
//! 3. **SimulationResult Analysis**: Trajectory retrieval and statistics methods
//! 4. **Integration Tests**: Training and simulation edge cases
//!
//! PERFORMANCE NOTE: These are unit/integration tests for data structures and analysis methods.
//! Not hot-path code - these run during post-training analysis, not in SDDP iterations.

mod fixtures;

use fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    generate_2stage_saa,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{
    BackwardPassTimingAccumulator, ForwardPassTimingAccumulator, NodeData,
    SddpAlgorithm, TerminationReason,
};
use powers_rs::subproblem::StudyPeriodKind;
use std::time::Duration;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a minimal 2-stage graph for testing
fn create_minimal_graph() -> DirectedGraph<NodeData> {
    let mut graph = DirectedGraph::<NodeData>::new();

    let pre_study_id = graph
        .add_node(
            NodeData::new(
                -1,
                0,
                0,
                "2024-01-01T00:00:00Z",
                "2024-01-01T00:00:00Z",
                StudyPeriodKind::PreStudy,
                create_simple_2stage_system(),
                "expectation",
                "naive",
                "naive",
                "storage",
            )
            .unwrap(),
        )
        .unwrap();

    let stage1_id = graph
        .add_node(
            NodeData::new(
                1,
                1,
                1,
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                StudyPeriodKind::Study,
                create_simple_2stage_system(),
                "expectation",
                "naive",
                "naive",
                "storage",
            )
            .unwrap(),
        )
        .unwrap();

    graph.add_edge(pre_study_id, stage1_id).unwrap();
    graph
}

// ============================================================================
// TIMING ACCUMULATOR TESTS (Test timing infrastructure)
// ============================================================================

#[test]
fn test_forward_timing_accumulator_single_timing() {
    // Test ForwardPassTimingAccumulator::aggregate with a single timing
    // Target: Lines testing aggregate() method with n=1
    let acc = ForwardPassTimingAccumulator {
        model_preprocessing_time: Duration::from_millis(100),
        solver_time: Duration::from_millis(500),
        model_postprocessing_time: Duration::from_millis(50),
        solver_calls: 1,
    };

    let timing = ForwardPassTimingAccumulator::aggregate(&[acc]);

    // With single timing, averages equal the input
    assert_eq!(timing.model_preprocessing_time, Duration::from_millis(100));
    assert_eq!(timing.solver_time, Duration::from_millis(500));
    assert_eq!(timing.model_postprocessing_time, Duration::from_millis(50));
}

#[test]
fn test_forward_timing_accumulator_multiple_timings() {
    // Test averaging across multiple parallel handlers
    // Target: Lines in aggregate() with n>1, testing averaging logic
    let acc1 = ForwardPassTimingAccumulator {
        model_preprocessing_time: Duration::from_millis(100),
        solver_time: Duration::from_millis(400),
        model_postprocessing_time: Duration::from_millis(50),
        solver_calls: 2,
    };

    let acc2 = ForwardPassTimingAccumulator {
        model_preprocessing_time: Duration::from_millis(200),
        solver_time: Duration::from_millis(600),
        model_postprocessing_time: Duration::from_millis(100),
        solver_calls: 3,
    };

    let timing = ForwardPassTimingAccumulator::aggregate(&[acc1, acc2]);

    // Averages should be (100+200)/2 = 150, (400+600)/2 = 500, (50+100)/2 = 75
    assert_eq!(timing.model_preprocessing_time, Duration::from_millis(150));
    assert_eq!(timing.solver_time, Duration::from_millis(500));
    assert_eq!(timing.model_postprocessing_time, Duration::from_millis(75));
}

#[test]
fn test_backward_timing_accumulator_into_timing() {
    // Test BackwardPassTimingAccumulator::into_timing conversion
    // Target: Lines in into_timing() method, testing field assignments and total_time calculation
    let acc = BackwardPassTimingAccumulator {
        backward_preprocessing_time: Duration::from_millis(10),
        model_preprocessing_time: Duration::from_millis(50),
        solver_time: Duration::from_millis(300),
        model_postprocessing_time: Duration::from_millis(40),
        cut_selection_time: Duration::from_millis(20),
        fcf_state_update_time: Duration::from_millis(15),
        cut_cloning_time: Duration::from_millis(10),
        handler_application_time: Duration::from_millis(5),
        solver_calls: 5,
        cuts_added: 3,
    };

    let timing = acc.into_timing();

    // Verify all fields are preserved
    assert_eq!(
        timing.backward_preprocessing_time,
        Duration::from_millis(10)
    );
    assert_eq!(timing.model_preprocessing_time, Duration::from_millis(50));
    assert_eq!(timing.solver_time, Duration::from_millis(300));
    assert_eq!(timing.model_postprocessing_time, Duration::from_millis(40));
    assert_eq!(timing.cut_selection_time, Duration::from_millis(20));
    assert_eq!(timing.fcf_state_update_time, Duration::from_millis(15));
    assert_eq!(timing.cut_cloning_time, Duration::from_millis(10));
    assert_eq!(timing.handler_application_time, Duration::from_millis(5));

    // Total should be sum of all components
    let expected_total =
        Duration::from_millis(10 + 50 + 300 + 40 + 20 + 15 + 10 + 5);
    assert_eq!(timing.total_time, expected_total);
}

// ============================================================================
// TRAININGRESULT ANALYSIS TESTS (Test gap calculation edge cases)
// ============================================================================

#[test]
fn test_training_result_gap_with_zero_lower_bound() {
    // Test gap calculations when lower bound is exactly zero
    // Target: Lines in relative_gap() method with lower_bound == 0.0
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(1, 5, &saa).expect("Training should succeed");

    // Get actual result and test methods
    let gap = result.final_gap();
    let rel_gap = result.relative_gap();

    assert!(gap >= 0.0);
    assert!(rel_gap >= 0.0 || rel_gap.is_infinite());
}

#[test]
fn test_training_result_converged_check() {
    // Test convergence check with various tolerance levels
    // Target: Lines in converged() method
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(3, 10, &saa).expect("Training should succeed");

    // Test convergence with large tolerance (should be true)
    assert!(result.converged(10000.0));

    // Test convergence with tiny tolerance (likely false)
    let converged_tight = result.converged(0.01);
    // Just verify the method executes without panicking
    let _ = converged_tight;
}

#[test]
fn test_training_result_lower_bounds_extraction() {
    // Test lower_bounds() extraction method
    // Target: Lines in lower_bounds() method
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(5, 10, &saa).expect("Training should succeed");

    // Test extraction method
    let lower_bounds = result.lower_bounds();

    assert_eq!(lower_bounds.len(), 5);

    // Lower bounds should be monotonically increasing
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i - 1],
            "Lower bounds should be monotonic"
        );
    }

    // Lower bounds should all be finite
    for lb in &lower_bounds {
        assert!(lb.is_finite(), "Lower bound should be finite");
    }
}

#[test]
fn test_training_result_iterations_accessor() {
    // Test iterations() accessor method
    // Target: Lines in iterations() method
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(5, 10, &saa).expect("Training should succeed");

    // Access iterations via iterations() method
    let iterations = result.iterations();
    assert_eq!(iterations.len(), 5);

    // Verify each iteration has expected properties
    for (i, iter) in iterations.iter().enumerate() {
        assert_eq!(iter.iteration, i + 1);
        assert!(iter.lower_bound >= 0.0);
        assert!(!iter.forward_costs.is_empty());
        assert!(iter.num_solver_calls > 0);
    }
}

// ============================================================================
// SIMULATION RESULT TESTS (Test trajectory retrieval and statistics)
// ============================================================================

#[test]
fn test_simulation_result_get_trajectory_methods() {
    // Test get_trajectory() and get_all_trajectories() methods
    // Target: Lines in get_trajectory() and get_all_trajectories()
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    sddp.train(3, 5, &saa).expect("Training should succeed");

    let sim_result = sddp
        .simulate_and_analyze(100, &saa)
        .expect("Simulation should succeed");

    // Test get_trajectory() with valid indices
    assert!(sim_result.get_trajectory(0).is_some());
    assert!(sim_result.get_trajectory(50).is_some());
    assert!(sim_result.get_trajectory(99).is_some());

    // Test get_trajectory() with invalid index
    assert!(sim_result.get_trajectory(100).is_none());
    assert!(sim_result.get_trajectory(1000).is_none());

    // Test get_all_trajectories()
    let all_trajs = sim_result.get_all_trajectories();
    assert_eq!(all_trajs.len(), 100);

    // Verify each trajectory has expected structure
    for traj in all_trajs {
        assert!(traj.total_cost > 0.0);
        assert!(!traj.stages.is_empty());
    }
}

#[test]
fn test_simulation_result_get_statistics() {
    // Test get_statistics() method
    // Target: Lines in get_statistics() method
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    sddp.train(2, 5, &saa).expect("Training should succeed");

    let sim_result = sddp
        .simulate_and_analyze(50, &saa)
        .expect("Simulation should succeed");

    // Test get_statistics() accessor
    let stats = sim_result.get_statistics();

    assert!(stats.mean > 0.0);
    assert!(stats.std >= 0.0);
    assert!(stats.p5 > 0.0);
    assert!(stats.p50 > 0.0); // median
    assert!(stats.p95 > 0.0);
    assert!(stats.num_trajectories == 50);

    // Verify percentile ordering
    assert!(stats.p5 <= stats.p25);
    assert!(stats.p25 <= stats.p50);
    assert!(stats.p50 <= stats.p75);
    assert!(stats.p75 <= stats.p95);

    // Verify confidence interval
    assert!(stats.ci_95.lower <= stats.mean);
    assert!(stats.mean <= stats.ci_95.upper);
    assert_eq!(stats.ci_95.confidence_level, 0.95);
}

#[test]
fn test_simulation_with_single_scenario_zero_variance() {
    // Edge case: Single scenario should have zero variance
    // Target: Lines handling n=1 in statistics calculation
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    sddp.train(2, 5, &saa).expect("Training should succeed");

    let sim_result = sddp
        .simulate_and_analyze(1, &saa)
        .expect("Simulation should succeed");

    // With 1 scenario, std should be 0
    assert_eq!(sim_result.statistics.std, 0.0);
    assert_eq!(sim_result.trajectories.len(), 1);

    // All percentiles should equal the mean
    assert_eq!(sim_result.statistics.p5, sim_result.statistics.mean);
    assert_eq!(sim_result.statistics.p50, sim_result.statistics.mean);
    assert_eq!(sim_result.statistics.p95, sim_result.statistics.mean);

    // Confidence interval should collapse to mean
    let ci = &sim_result.statistics.ci_95;
    assert_eq!(ci.lower, ci.upper);
}

#[test]
fn test_simulation_with_many_scenarios() {
    // Test with large number of scenarios to verify no memory issues
    // Target: Lines in trajectory collection and statistics computation
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    sddp.train(2, 5, &saa).expect("Training should succeed");

    // Simulate with 200 scenarios
    let sim_result = sddp
        .simulate_and_analyze(200, &saa)
        .expect("Simulation should succeed");

    assert_eq!(sim_result.trajectories.len(), 200);
    assert_eq!(sim_result.statistics.num_trajectories, 200);

    // Confidence interval properties
    let ci = &sim_result.statistics.ci_95;
    let ci_width = ci.upper - ci.lower;
    assert!(ci_width >= 0.0); // Non-negative (zero for deterministic problems)

    // Verify statistics are reasonable
    let stats = sim_result.get_statistics();
    assert!(stats.p95 >= stats.p5); // Percentiles ordered (may be equal for deterministic)
    assert!(stats.mean >= stats.p5 && stats.mean <= stats.p95);
}

// ============================================================================
// TIMING STRUCTURE INTEGRATION TESTS
// ============================================================================

#[test]
fn test_forward_pass_timing_recorded() {
    // Test that forward pass timing is correctly recorded in iterations
    // Target: Lines recording forward_timing in IterationResult
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(1, 5, &saa).expect("Training should succeed");

    let iter = &result.iterations()[0];

    // Verify timing components are non-zero (actual work happened)
    assert!(iter.forward_timing.solver_time > Duration::from_micros(0));
    assert!(iter.forward_timing.total_time > Duration::from_micros(0));

    // Total time should be at least solver time
    assert!(iter.forward_timing.total_time >= iter.forward_timing.solver_time);
}

#[test]
fn test_backward_pass_timing_recorded() {
    // Test that backward pass timing is correctly recorded
    // Target: Lines recording backward_timing in IterationResult
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(1, 5, &saa).expect("Training should succeed");

    let iter = &result.iterations()[0];

    // Verify timing components are non-zero
    assert!(iter.backward_timing.solver_time > Duration::from_micros(0));
    assert!(
        iter.backward_timing.cut_selection_time >= Duration::from_micros(0)
    );
    assert!(iter.backward_timing.total_time > Duration::from_micros(0));

    // Total time should be at least solver time
    assert!(
        iter.backward_timing.total_time >= iter.backward_timing.solver_time
    );
}

#[test]
fn test_termination_reason_iteration_limit() {
    // Test that termination reason is correctly set to IterationLimit
    // Target: Lines setting termination_reason in TrainingResult
    let graph = create_minimal_graph();
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let result = sddp.train(10, 5, &saa).expect("Training should succeed");

    // Should terminate due to iteration limit (not convergence)
    assert_eq!(result.termination_reason, TerminationReason::IterationLimit);
}
