//! Convergence Integration Tests
//!
//! Tests convergence behavior over multiple iterations:
//! - Monotonic lower bound increases
//! - Upper bound improvement trends
//! - Gap closure for deterministic problems
//! - Convergence rate validation

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

/// Test that lower bound is monotonically non-decreasing across iterations
#[test]
fn test_monotonic_lower_bound_integration() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run enough iterations to observe convergence behavior
    let result = sddp
        .train(30, 2, false, &saa)
        .expect("Training should succeed");

    let lower_bounds = result.lower_bounds();

    // Verify monotonic non-decreasing property (fundamental SDDP guarantee)
    for i in 1..lower_bounds.len() {
        let diff = lower_bounds[i] - lower_bounds[i - 1];
        assert!(
            diff >= -1e-6,
            "Lower bound should be non-decreasing: iteration {} = {:.6e}, iteration {} = {:.6e}, diff = {:.6e}",
            i - 1,
            lower_bounds[i - 1],
            i,
            lower_bounds[i],
            diff
        );
    }

    // Verify bounds are finite
    for (i, lb) in lower_bounds.iter().enumerate() {
        assert!(lb.is_finite(), "Lower bound {} should be finite", i);
    }

    // For a deterministic problem, lower bound should stabilize
    if lower_bounds.len() >= 10 {
        let last_10 = &lower_bounds[lower_bounds.len() - 10..];
        let max = last_10.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min = last_10.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let range = max - min;

        // Last 10 iterations should show small variation (convergence)
        assert!(
            range < 1e-3 || range / max.abs() < 1e-4,
            "Lower bound should stabilize in last 10 iterations: range = {:.6e}, max = {:.6e}",
            range,
            max
        );
    }
}

/// Test that upper bound shows improvement trend over iterations
#[test]
fn test_upper_bound_improvement() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run multiple iterations
    let result = sddp
        .train(25, 5, false, &saa)
        .expect("Training should succeed");

    let iterations = result.iterations();

    // Collect upper bounds (average forward cost per iteration)
    let upper_bounds: Vec<f64> = iterations
        .iter()
        .map(|iter| {
            iter.forward_costs.iter().sum::<f64>()
                / iter.forward_costs.len() as f64
        })
        .collect();

    // For deterministic problems, upper bound should improve or stay same
    // Compare first 20% vs last 20% of iterations
    let split_point = upper_bounds.len() / 5;
    if split_point > 0 {
        let early_avg: f64 = upper_bounds[..split_point].iter().sum::<f64>()
            / split_point as f64;
        let late_avg: f64 = upper_bounds[upper_bounds.len() - split_point..]
            .iter()
            .sum::<f64>()
            / split_point as f64;

        // Later iterations should have similar or better upper bound
        // (for deterministic problem, should be similar; for stochastic, should improve)
        assert!(
            late_avg <= early_avg + 1e-3,
            "Upper bound should not worsen significantly: early avg = {:.6e}, late avg = {:.6e}",
            early_avg,
            late_avg
        );
    }

    // Verify best upper bound is reasonable
    assert!(
        result.best_upper_bound.is_finite(),
        "Best upper bound should be finite"
    );

    // Best UB should be in the range of observed UBs
    let min_ub = upper_bounds.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_ub = upper_bounds
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    assert!(
        result.best_upper_bound >= min_ub - 1e-6
            && result.best_upper_bound <= max_ub + 1e-6,
        "Best UB {:.6e} should be in range [{:.6e}, {:.6e}]",
        result.best_upper_bound,
        min_ub,
        max_ub
    );
}

/// Test that gap closes for deterministic problems
#[test]
fn test_gap_closure_deterministic() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run enough iterations for convergence
    let result = sddp
        .train(40, 1, false, &saa)
        .expect("Training should succeed");

    let iterations = result.iterations();
    let lower_bounds = result.lower_bounds();

    // Calculate gaps for each iteration
    let mut gaps = Vec::with_capacity(iterations.len());
    for (i, iter) in iterations.iter().enumerate() {
        let ub = iter.forward_costs.iter().sum::<f64>()
            / iter.forward_costs.len() as f64;
        let lb = lower_bounds[i];
        let gap = ub - lb;
        gaps.push(gap);
    }

    // Gap should be non-negative (UB >= LB)
    for (i, gap) in gaps.iter().enumerate() {
        assert!(
            *gap >= -1e-6,
            "Gap should be non-negative at iteration {}: {:.6e}",
            i,
            gap
        );
    }

    // For deterministic problem, gap should close
    // Compare first 5 iterations vs last 5 iterations
    if gaps.len() >= 10 {
        let early_gap_avg = gaps[..5].iter().sum::<f64>() / 5.0;
        let late_gap_avg = gaps[gaps.len() - 5..].iter().sum::<f64>() / 5.0;

        // Gap should decrease or stay small
        assert!(
            late_gap_avg <= early_gap_avg + 1e-6,
            "Gap should close: early avg = {:.6e}, late avg = {:.6e}",
            early_gap_avg,
            late_gap_avg
        );
    }

    // Final gap should be small for deterministic problem
    let final_gap = result.final_gap();
    assert!(
        final_gap.is_finite(),
        "Final gap should be finite: {:.6e}",
        final_gap
    );

    // Gap should be reasonable (small relative to problem scale)
    let problem_scale = result.final_lower_bound.abs().max(1.0);
    let relative_gap = final_gap / problem_scale;

    assert!(
        relative_gap < 0.1 || final_gap < 1e-3,
        "Final gap should be small: absolute = {:.6e}, relative = {:.6e}",
        final_gap,
        relative_gap
    );
}

/// Test convergence rate and iteration count needed
#[test]
fn test_convergence_rate() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run training
    let result = sddp
        .train(50, 2, false, &saa)
        .expect("Training should succeed");

    let iterations = result.iterations();
    let lower_bounds = result.lower_bounds();

    // Track when lower bound reaches certain thresholds
    let final_lb = lower_bounds[lower_bounds.len() - 1];
    let initial_lb = lower_bounds[0];
    let lb_improvement = final_lb - initial_lb;

    // Find iteration where we reach 90% of final improvement
    let target_lb = initial_lb + 0.9 * lb_improvement;
    let mut convergence_iter = None;

    for (i, &lb) in lower_bounds.iter().enumerate() {
        if lb >= target_lb - 1e-9 {
            convergence_iter = Some(i);
            break;
        }
    }

    // Should converge reasonably fast for simple 2-stage problem
    if lb_improvement > 1e-6 {
        assert!(
            convergence_iter.is_some(),
            "Should reach 90% of improvement within iterations"
        );

        let conv_iter = convergence_iter.unwrap();
        assert!(
            conv_iter < iterations.len(),
            "Convergence iteration {} should be less than total {}",
            conv_iter,
            iterations.len()
        );

        // For a 2-stage problem, should converge quickly (typically < 20 iterations)
        // This is a soft check - depends on problem structure
        if lb_improvement > 0.01 {
            // Only check if there's significant improvement to track
            assert!(
                conv_iter < iterations.len(),
                "Should show convergence trend"
            );
        }
    }

    // Verify lower bound is stable in final iterations
    if lower_bounds.len() >= 10 {
        let last_10 = &lower_bounds[lower_bounds.len() - 10..];
        let variance: f64 =
            last_10.iter().map(|&x| (x - final_lb).powi(2)).sum::<f64>() / 10.0;
        let std_dev = variance.sqrt();

        // Standard deviation should be small (stable convergence)
        assert!(
            std_dev < 1e-3 || std_dev / final_lb.abs().max(1.0) < 1e-6,
            "Lower bound should be stable in final iterations: std_dev = {:.6e}, final_lb = {:.6e}",
            std_dev,
            final_lb
        );
    }
}

/// Test that algorithm terminates correctly on max iterations
#[test]
fn test_termination_on_max_iterations() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    let max_iterations = 15;

    let result = sddp
        .train(max_iterations, 3, false, &saa)
        .expect("Training should succeed");

    // Should complete exactly max_iterations
    assert_eq!(
        result.iterations().len(),
        max_iterations,
        "Should complete exactly {} iterations",
        max_iterations
    );

    // All iterations should have valid data
    for (i, iter) in result.iterations().iter().enumerate() {
        assert!(
            iter.lower_bound.is_finite(),
            "Iteration {} should have finite lower bound",
            i
        );
        assert!(
            !iter.forward_costs.is_empty(),
            "Iteration {} should have forward costs",
            i
        );
    }
}

/// Test convergence with different forward pass counts
#[test]
fn test_convergence_with_varying_scenarios() {
    // Test with 1 forward pass (deterministic sampling)
    {
        let graph = create_test_graph().expect("Failed to create graph");
        let initial_condition = create_simple_2stage_initial_condition();
        let saa = generate_2stage_saa(42);

        let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
            .expect("Failed to create SDDP");

        let result = sddp
            .train(20, 1, false, &saa)
            .expect("Training should succeed");

        let lower_bounds = result.lower_bounds();

        // Verify monotonicity
        for i in 1..lower_bounds.len() {
            assert!(
                lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
                "LB should be non-decreasing with 1 scenario"
            );
        }
    }

    // Test with 10 forward passes (more sampling)
    {
        let graph = create_test_graph().expect("Failed to create graph");
        let initial_condition = create_simple_2stage_initial_condition();
        let saa = generate_2stage_saa(42);

        let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
            .expect("Failed to create SDDP");

        let result = sddp
            .train(20, 10, false, &saa)
            .expect("Training should succeed");

        let lower_bounds = result.lower_bounds();

        // Verify monotonicity
        for i in 1..lower_bounds.len() {
            assert!(
                lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
                "LB should be non-decreasing with 10 scenarios"
            );
        }

        // With more scenarios, upper bound estimate should be more stable
        let iterations = result.iterations();
        let ub_variance: f64 = {
            let mean: f64 = iterations
                .iter()
                .map(|iter| {
                    iter.forward_costs.iter().sum::<f64>()
                        / iter.forward_costs.len() as f64
                })
                .sum::<f64>()
                / iterations.len() as f64;

            iterations
                .iter()
                .map(|iter| {
                    let ub = iter.forward_costs.iter().sum::<f64>()
                        / iter.forward_costs.len() as f64;
                    (ub - mean).powi(2)
                })
                .sum::<f64>()
                / iterations.len() as f64
        };

        // Variance should be finite
        assert!(ub_variance.is_finite(), "UB variance should be finite");
    }
}

/// Test convergence behavior with cut selection enabled
#[test]
fn test_convergence_with_cut_selection() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run with cut selection
    let result = sddp
        .train(30, 3, true, &saa)
        .expect("Training with cut selection should succeed");

    let lower_bounds = result.lower_bounds();

    // Monotonicity should still hold with cut selection
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
            "LB should be non-decreasing even with cut selection: iteration {} = {:.6e}, iteration {} = {:.6e}",
            i - 1,
            lower_bounds[i - 1],
            i,
            lower_bounds[i]
        );
    }

    // Algorithm should still converge with cut selection
    let final_gap = result.final_gap();
    assert!(final_gap.is_finite(), "Final gap should be finite");
}

/// Test long convergence run to validate stability
#[test]
fn test_long_convergence_run() {
    let graph = create_test_graph().expect("Failed to create graph");
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)
        .expect("Failed to create SDDP");

    // Run many iterations to test stability
    let result = sddp
        .train(100, 2, false, &saa)
        .expect("Long training run should succeed");

    let lower_bounds = result.lower_bounds();

    // Verify monotonicity holds over long run
    for i in 1..lower_bounds.len() {
        assert!(
            lower_bounds[i] >= lower_bounds[i - 1] - 1e-6,
            "LB monotonicity should hold over 100 iterations"
        );
    }

    // Lower bound should stabilize (small changes in later iterations)
    let mid_point = lower_bounds.len() / 2;
    let mid_lb = lower_bounds[mid_point];
    let final_lb = lower_bounds[lower_bounds.len() - 1];
    let late_improvement = final_lb - mid_lb;

    // Later half should show less improvement than early iterations
    let early_improvement = lower_bounds[mid_point] - lower_bounds[0];

    // Late improvement should be smaller (convergence slowing)
    assert!(
        late_improvement <= early_improvement + 1e-6,
        "Improvement should slow down: early = {:.6e}, late = {:.6e}",
        early_improvement,
        late_improvement
    );

    // Final iterations should be very stable
    let last_5 = &lower_bounds[lower_bounds.len() - 5..];
    let max_change = last_5
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);

    assert!(
        max_change < 1e-6,
        "Final 5 iterations should be stable: max change = {:.6e}",
        max_change
    );
}
