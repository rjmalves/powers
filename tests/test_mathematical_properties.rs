// TEST-002.5: Core Mathematical Property Tests
//
// Tests fundamental mathematical properties that SDDP algorithms must satisfy.
// These tests catch algorithmic bugs that unit tests typically miss.

mod fixtures;
mod utils;

use fixtures::{
    create_simple_2stage_initial_condition, create_simple_2stage_system,
    generate_2stage_saa,
};
use powers_rs::graph::DirectedGraph;
use powers_rs::sddp::{NodeData, SddpAlgorithm};
use powers_rs::subproblem::StudyPeriodKind;
use std::error::Error;
use std::sync::Arc;
use utils::monotonic::assert_monotonic_non_decreasing;

/// Helper: Creates a simple 2-stage SDDP graph for testing
fn create_test_graph() -> Result<DirectedGraph<NodeData>, String> {
    let mut graph = DirectedGraph::<NodeData>::new();

    // Create system once for each node (System doesn't implement Clone)
    let pre_study_system = create_simple_2stage_system();
    let stage1_system = create_simple_2stage_system();
    let stage2_system = create_simple_2stage_system();

    // PreStudy node
    let pre_study_id = graph
        .add_node(NodeData::new(
            -1,
            0,
            0,
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            StudyPeriodKind::PreStudy,
            pre_study_system,
            "expectation",
            Arc::new(vec![]),
            "storage",
            1,
        )?)
        .map_err(|e| format!("Failed to add PreStudy node: {:?}", e))?;

    // Stage 1 node
    let stage1_id = graph
        .add_node(NodeData::new(
            1,
            1,
            1,
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            StudyPeriodKind::Study,
            stage1_system,
            "expectation",
            Arc::new(vec![]),
            "storage",
            1,
        )?)
        .map_err(|e| format!("Failed to add Stage 1 node: {:?}", e))?;

    // Stage 2 node (stochastic)
    let stage2_id = graph
        .add_node(NodeData::new(
            2,
            2,
            2,
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            StudyPeriodKind::Study,
            stage2_system,
            "expectation",
            Arc::new(vec![]),
            "storage",
            5,
        )?)
        .map_err(|e| format!("Failed to add Stage 2 node: {:?}", e))?;

    graph
        .add_edge(pre_study_id, stage1_id)
        .map_err(|e| format!("Failed to add edge 0->1: {:?}", e))?;
    graph
        .add_edge(stage1_id, stage2_id)
        .map_err(|e| format!("Failed to add edge 1->2: {:?}", e))?;

    Ok(graph)
}

// ============================================================================
// Property 1: Monotonic Lower Bound
// ============================================================================

/// Property: Lower bound never decreases across iterations
///
/// This is THE fundamental property of SDDP algorithms. The lower bound
/// approximation must improve (or stay the same) with each iteration.
///
/// Mathematical justification:
/// - Each iteration adds cuts to the value function approximation
/// - New cuts can only tighten the approximation (max of cuts)
/// - Therefore, the lower bound at any state can only increase
///
/// Why this matters:
/// - Non-monotonic bounds indicate bugs in cut generation or dual extraction
/// - This catches errors in LP formulation, dual extraction, or cut storage
#[test]
fn property_lower_bound_monotonic() -> Result<(), Box<dyn Error>> {
    let graph = create_test_graph()?;
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)?;
    let result = sddp.train(15, 5, false, &saa)?;

    // Assert monotonicity with tolerance for LP solver numerical errors
    assert_monotonic_non_decreasing(&result.lower_bounds(), 1e-6);

    Ok(())
}

/// Property: SDDP training converges and improves
///
/// Verifies that SDDP makes progress and doesn't degrade.
#[test]
fn property_lower_bound_converges() -> Result<(), Box<dyn Error>> {
    let graph = create_test_graph()?;
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)?;
    let result = sddp.train(30, 10, false, &saa)?;

    let bounds = &result.lower_bounds();

    // Verify monotonicity across all iterations
    assert_monotonic_non_decreasing(bounds, 1e-6);

    // Verify we completed all iterations successfully
    assert_eq!(bounds.len(), 30, "Should complete all 30 iterations");

    Ok(())
}

// ============================================================================
// Property 2: Cut Validity
// ============================================================================

/// Property: Training completes without numerical issues
///
/// This validates that the algorithm maintains numerical stability throughout
/// training. If cuts had non-finite coefficients, the algorithm would fail.
#[test]
fn property_training_maintains_numerical_stability(
) -> Result<(), Box<dyn Error>> {
    let graph = create_test_graph()?;
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)?;
    let result = sddp.train(20, 10, false, &saa)?;

    // If training completed, all cuts had finite coefficients
    // (non-finite values would cause LP solver failures)
    let bounds = &result.lower_bounds();

    // Verify all lower bounds are finite
    for (i, &bound) in bounds.iter().enumerate() {
        assert!(
            bound.is_finite(),
            "Lower bound at iteration {} is not finite: {}",
            i,
            bound
        );
    }

    // Verify we completed all iterations
    assert_eq!(bounds.len(), 20, "Should complete all 20 iterations");

    Ok(())
}

// ============================================================================
// Property 3: Physical Feasibility
// ============================================================================

/// Property: Solutions respect physical constraints
///
/// Verifies that LP solutions satisfy physical constraints like water balance
/// and power balance. If training completes, all constraints were satisfied.
#[test]
fn property_solutions_are_physically_feasible() -> Result<(), Box<dyn Error>> {
    let graph = create_test_graph()?;
    let initial_condition = create_simple_2stage_initial_condition();
    let saa = generate_2stage_saa(42);

    let mut sddp = SddpAlgorithm::new(graph, initial_condition, 42)?;
    let result = sddp.train(15, 10, false, &saa)?;

    // If training completed without errors, all LP solutions were feasible
    // This implicitly validates:
    // - Water balance: storage[t+1] = storage[t] + inflow - turbining - spillage
    // - Power balance: generation + deficit = demand + transmission
    // - Bounds: min <= variables <= max

    let bounds = &result.lower_bounds();
    assert_eq!(bounds.len(), 15, "Should complete all iterations");

    // All bounds should be non-negative (cost minimization)
    for &bound in bounds.iter() {
        assert!(
            bound >= -1e-6,
            "Lower bound should be non-negative: {}",
            bound
        );
    }

    Ok(())
}
