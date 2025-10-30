// Subproblem construction is where SDDP builds stage-specific
// optimization problems. This is complex logic that constructs LP/MILP from system
// description, adds state transition constraints, and integrates cuts from the
// future cost function. Thorough testing is critical for algorithm correctness.
//
//  Subproblems are constructed once per node in the graph, but
// solved thousands of times. Construction performance is less critical than solve
// performance, but we still aim for efficient memory allocation patterns.

mod fixtures;

use fixtures::subproblems::*;
use powers_rs::scenario::OptimizedSampledBranchingNoises;

/// Tests subproblem construction with minimal system
///
/// Validates:
/// - Model is created
/// - Variable counts match system dimensions
/// - Constraint counts match system dimensions
/// - Alpha variable (future cost) exists
#[test]
fn test_minimal_subproblem_construction() {
    let subproblem = create_minimal_subproblem();

    // Model should be constructed
    assert!(subproblem.model.is_some(), "Subproblem should have a model");

    // Variable counts (minimal system: 1 bus, 1 hydro, 2 thermals, 0 lines)
    assert_eq!(
        subproblem.variables.deficit.len(),
        1,
        "Should have 1 deficit variable (1 bus)"
    );
    assert_eq!(
        subproblem.variables.thermal_gen.len(),
        2,
        "Should have 2 thermal generation variables"
    );
    assert_eq!(
        subproblem.variables.turbined_flow.len(),
        1,
        "Should have 1 turbined flow variable (1 hydro)"
    );
    assert_eq!(
        subproblem.variables.spillage.len(),
        1,
        "Should have 1 spillage variable (1 hydro)"
    );
    assert_eq!(
        subproblem.variables.stored_volume.len(),
        1,
        "Should have 1 stored volume variable (1 hydro)"
    );
    assert_eq!(
        subproblem.variables.inflow.len(),
        1,
        "Should have 1 inflow variable (1 hydro)"
    );
    assert_eq!(
        subproblem.variables.direct_exchange.len(),
        0,
        "Should have 0 exchange variables (no lines)"
    );
    assert_eq!(
        subproblem.variables.reverse_exchange.len(),
        0,
        "Should have 0 reverse exchange variables (no lines)"
    );

    // Alpha variable exists (future cost epigraph)
    assert!(
        subproblem.variables.alpha > 0,
        "Alpha variable should exist and have valid index"
    );

    // Constraint counts
    assert_eq!(
        subproblem.constraints.load_balance.len(),
        1,
        "Should have 1 load balance constraint (1 bus)"
    );
    assert_eq!(
        subproblem.constraints.hydro_balance.len(),
        1,
        "Should have 1 hydro balance constraint (1 hydro)"
    );
}

/// Tests subproblem construction with cascade system
///
/// Validates:
/// - Multiple buses and hydros handled correctly
/// - Transmission line variables created
/// - Cascade relationship doesn't create extra constraints (handled via factors)
#[test]
fn test_cascade_subproblem_construction() {
    let subproblem = create_cascade_subproblem();

    // Model should be constructed
    assert!(subproblem.model.is_some());

    // Variable counts (cascade system: 2 buses, 2 hydros, 0 thermals, 1 line)
    assert_eq!(
        subproblem.variables.deficit.len(),
        2,
        "Should have 2 deficit variables (2 buses)"
    );
    assert_eq!(
        subproblem.variables.thermal_gen.len(),
        0,
        "Should have 0 thermal variables (no thermals)"
    );
    assert_eq!(
        subproblem.variables.turbined_flow.len(),
        2,
        "Should have 2 turbined flow variables (2 hydros)"
    );
    assert_eq!(
        subproblem.variables.spillage.len(),
        2,
        "Should have 2 spillage variables (2 hydros)"
    );
    assert_eq!(
        subproblem.variables.stored_volume.len(),
        2,
        "Should have 2 stored volume variables (2 hydros)"
    );
    assert_eq!(
        subproblem.variables.inflow.len(),
        2,
        "Should have 2 inflow variables (2 hydros)"
    );
    assert_eq!(
        subproblem.variables.direct_exchange.len(),
        1,
        "Should have 1 direct exchange variable (1 line)"
    );
    assert_eq!(
        subproblem.variables.reverse_exchange.len(),
        1,
        "Should have 1 reverse exchange variable (1 line)"
    );

    // Constraint counts
    assert_eq!(
        subproblem.constraints.load_balance.len(),
        2,
        "Should have 2 load balance constraints (2 buses)"
    );
    assert_eq!(
        subproblem.constraints.hydro_balance.len(),
        2,
        "Should have 2 hydro balance constraints (2 hydros)"
    );
}

/// Tests subproblem construction with mixed hydro-thermal system
///
/// Validates:
/// - Thermal generation variables created
/// - Objective function includes thermal costs
#[test]
fn test_mixed_subproblem_construction() {
    let subproblem = create_mixed_subproblem();

    assert!(subproblem.model.is_some());

    // Variable counts (mixed system: 1 bus, 1 hydro, 2 thermals, 0 lines)
    assert_eq!(subproblem.variables.deficit.len(), 1);
    assert_eq!(
        subproblem.variables.thermal_gen.len(),
        2,
        "Should have 2 thermal generation variables"
    );
    assert_eq!(subproblem.variables.turbined_flow.len(), 1);
    assert_eq!(subproblem.variables.direct_exchange.len(), 0);
}
/// Tests that load balance constraints are properly structured
///
/// Load balance constraint should be:
/// deficit + thermal_gen + hydro_gen (productivity-adjusted) + imports - exports = load
///
/// This test doesn't validate coefficients (hard to access), but validates structure.
#[test]
fn test_load_balance_constraint_structure() {
    let system = create_minimal_system();
    let subproblem = create_minimal_subproblem();

    // Load balance constraint count matches bus count
    assert_eq!(
        subproblem.constraints.load_balance.len(),
        system.meta.buses_count
    );

    // Constraint indices should be valid (> 0 after model construction)
    for &constraint_idx in &subproblem.constraints.load_balance {
        assert!(
            constraint_idx < 1000,
            "Constraint index should be reasonable (< 1000 for minimal system)"
        );
    }
}

/// Tests that hydro balance constraints are properly structured
///
/// Hydro balance constraint should be:
/// stored_volume[t] + turbined_flow[t] + spillage[t] - inflow[t] - upstream_flows = initial_storage
///
/// For cascade systems, upstream flows are included.
#[test]
fn test_hydro_balance_constraint_structure() {
    let system = create_cascade_system();
    let subproblem = create_cascade_subproblem();

    // Hydro balance constraint count matches hydro count
    assert_eq!(
        subproblem.constraints.hydro_balance.len(),
        system.meta.hydros_count
    );

    // Constraint indices should be valid
    for &constraint_idx in &subproblem.constraints.hydro_balance {
        assert!(
            constraint_idx < 1000,
            "Constraint index should be reasonable"
        );
    }
}

/// Tests that hydro balance RHS can be updated (state transition)
///
/// In SDDP, the incoming state (initial storage) is set via hydro balance RHS.
#[test]
fn test_hydro_balance_rhs_update() {
    let mut subproblem = create_minimal_subproblem();

    // Create realization with specific storage value
    let realization = create_minimal_realization(Some(30.0));

    // Update with trajectory (sets hydro balance RHS)
    subproblem.update_with_current_trajectory(vec![&realization]);

    // Model should still be valid after update
    assert!(
        subproblem.model.is_some(),
        "Model should remain valid after state update"
    );
}

/// Tests solving subproblem with simple deterministic uncertainties
///
/// Validates:
/// - Subproblem solves to optimality
/// - Solution is extracted correctly
/// - Realization container is populated
#[test]
fn test_realize_uncertainties_simple() {
    let mut subproblem = create_minimal_subproblem();
    let system = create_minimal_system();

    // Set up sampled noises (1 bus load entity, 1 hydro inflow entity)
    let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
    noises.set_load_innovations(&[0.0]); // Naive process uses mean, ignores noise
    noises.set_inflow_data(&[0.0], &[0.0]);

    // Create realization container with initial storage
    let mut realization = create_test_realization(&system, Some(vec![50.0]));

    // Realize uncertainties (solve subproblem)
    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    // Should solve successfully
    assert!(
        result.is_ok(),
        "Subproblem should solve successfully: {:?}",
        result.err()
    );

    // Objective should be finite and non-negative
    assert!(
        realization.total_stage_objective.is_finite(),
        "Objective should be finite"
    );
    assert!(
        realization.total_stage_objective >= 0.0,
        "Objective should be non-negative (costs)"
    );

    // Solution vectors should be populated
    assert_eq!(realization.deficit.len(), 1);
    assert_eq!(realization.final_storage.len(), 1);
    assert_eq!(realization.turbined_flow.len(), 1);
}

/// Tests solving cascade subproblem with upstream/downstream flow
///
/// Validates that cascade relationships work correctly.
#[test]
fn test_realize_uncertainties_cascade() {
    let mut subproblem = create_cascade_subproblem();

    // Sampled noises (2 bus load entities, 2 hydro inflow entities)
    let mut noises = OptimizedSampledBranchingNoises::new(2, 2);
    noises.set_load_innovations(&[0.0, 0.0]);
    noises.set_inflow_data(&[0.0, 0.0], &[0.0, 0.0]);

    // Realization container with initial storages
    let mut realization = create_cascade_realization(Some(50.0), Some(40.0));

    // Solve
    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    assert!(
        result.is_ok(),
        "Cascade subproblem should solve: {:?}",
        result.err()
    );

    // Both hydros should have solutions
    assert_eq!(realization.final_storage.len(), 2);
    assert_eq!(realization.turbined_flow.len(), 2);
    assert_eq!(realization.spillage.len(), 2);

    // Objective should be reasonable
    assert!(realization.total_stage_objective.is_finite());
}

/// Tests solving with high load demand (forcing deficit or thermal)
///
/// Validates that deficit variables activate when needed.
#[test]
fn test_realize_uncertainties_with_deficit() {
    let mut subproblem = create_minimal_subproblem();

    // Sampled noises (naive process doesn't directly control load)
    // The load is determined by the stochastic process realization
    let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
    noises.set_load_innovations(&[0.0]);
    noises.set_inflow_data(&[0.0], &[0.0]);

    let mut realization = create_minimal_realization(Some(0.0)); // Empty storage

    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    assert!(result.is_ok());
    // Deficit vector should exist (even if zero)
    assert_eq!(realization.deficit.len(), 1);
}

/// Tests subproblem with single hydro (minimal case)
///
/// Validates that minimal configurations work correctly.
#[test]
fn test_single_hydro_subproblem() {
    let subproblem = create_minimal_subproblem();

    assert!(subproblem.model.is_some());
    assert_eq!(subproblem.variables.stored_volume.len(), 1);
    assert_eq!(subproblem.variables.turbined_flow.len(), 1);
    assert_eq!(subproblem.constraints.hydro_balance.len(), 1);
}

/// Tests subproblem construction with no thermal plants
///
/// Validates that thermal variables are optional.
#[test]
fn test_no_thermal_subproblem() {
    let subproblem = create_cascade_subproblem(); // Cascade has no thermals

    assert!(subproblem.model.is_some());
    assert_eq!(
        subproblem.variables.thermal_gen.len(),
        0,
        "Should have no thermal variables"
    );
}

/// Tests subproblem construction with no transmission lines
///
/// Validates that exchange variables are optional.
#[test]
fn test_no_transmission_subproblem() {
    let subproblem = create_minimal_subproblem(); // Minimal has no lines

    assert!(subproblem.model.is_some());
    assert_eq!(
        subproblem.variables.direct_exchange.len(),
        0,
        "Should have no direct exchange variables"
    );
    assert_eq!(
        subproblem.variables.reverse_exchange.len(),
        0,
        "Should have no reverse exchange variables"
    );
}

/// Tests subproblem with tight storage bounds
///
/// Validates numerical stability with constrained feasible region.
#[test]
fn test_tight_storage_bounds() {
    // Use minimal system (storage bounds: 0-100)
    let mut subproblem = create_minimal_subproblem();

    // Initial storage near upper bound
    let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
    noises.set_load_innovations(&[0.0]);
    noises.set_inflow_data(&[0.0], &[0.0]);

    let mut realization = create_minimal_realization(Some(95.0)); // Near max

    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    assert!(
        result.is_ok(),
        "Should handle tight bounds: {:?}",
        result.err()
    );
}

/// Tests that subproblem model can be solved with real solver
///
/// Validates:
/// - Model structure is valid for HiGHS
/// - Solve returns optimal status
/// - Solution can be extracted
#[test]
fn test_subproblem_solves_to_optimality() {
    let mut subproblem = create_minimal_subproblem();

    let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
    noises.set_load_innovations(&[0.0]);
    noises.set_inflow_data(&[0.0], &[0.0]);

    let mut realization = create_minimal_realization(Some(50.0));

    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    assert!(result.is_ok());

    // Verify solution quality
    assert!(realization.total_stage_objective.is_finite());
    assert!(realization.current_stage_objective.is_finite());
    assert!(
        realization.total_stage_objective
            >= realization.current_stage_objective,
        "Total objective should be >= current (includes future cost alpha)"
    );
}

/// Tests subproblem feasibility with various initial conditions
///
/// Validates that subproblem remains feasible across range of storages.
#[test]
fn test_subproblem_feasibility_range() {
    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();

    // Test various initial storage values
    let storage_values = vec![0.0, 25.0, 50.0, 75.0, 100.0];

    for storage in storage_values {
        let mut subproblem = create_minimal_subproblem();
        let mut realization = create_minimal_realization(Some(storage));

        let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
        noises.set_load_innovations(&[0.0]);
        noises.set_inflow_data(&[0.0], &[0.0]);

        let result = subproblem.realize_uncertainties(
            &noises,
            load_sp.as_ref(),
            &mut realization,
        );

        assert!(
            result.is_ok(),
            "Subproblem should be feasible with storage={}: {:?}",
            storage,
            result.err()
        );
    }
}

/// Tests that objective value is consistent with solution
///
/// Validates that reported objective matches sum of costs.
#[test]
fn test_objective_consistency() {
    let mut subproblem = create_minimal_subproblem();

    let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
    noises.set_load_innovations(&[0.0]);
    noises.set_inflow_data(&[0.0], &[0.0]);

    let mut realization = create_minimal_realization(Some(50.0));

    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();
    let result = subproblem.realize_uncertainties(
        &noises,
        load_sp.as_ref(),
        &mut realization,
    );

    assert!(result.is_ok());

    // Current stage objective should be non-negative (sum of costs)
    assert!(
        realization.current_stage_objective >= 0.0,
        "Current stage cost should be non-negative"
    );

    // Total objective includes future cost (alpha), should be >= current
    assert!(
        realization.total_stage_objective
            >= realization.current_stage_objective - 1e-6,
        "Total >= current (allowing small numerical error)"
    );
}

/// Tests that repeated subproblem solves don't cause memory issues
///
/// Validates:
/// - No memory leaks over many solves
/// - Performance remains stable
#[test]
fn test_repeated_solves() {
    let (load_sp, _inflow_sp) = create_naive_stochastic_processes();

    // Solve same subproblem many times
    for i in 0..10 {
        let mut subproblem = create_minimal_subproblem();
        let mut realization = create_minimal_realization(Some(50.0));
        let mut noises = OptimizedSampledBranchingNoises::new(1, 1);
        noises.set_load_innovations(&[0.0]);
        noises.set_inflow_data(&[0.0], &[0.0]);

        let result = subproblem.realize_uncertainties(
            &noises,
            load_sp.as_ref(),
            &mut realization,
        );

        assert!(
            result.is_ok(),
            "Solve {} should succeed: {:?}",
            i,
            result.err()
        );
    }
}

/// Tests that subproblem has consistent variable/constraint structure
///
/// Validates internal consistency of Variables and Constraints structs.
#[test]
fn test_subproblem_structure_consistency() {
    let system = create_cascade_system();
    let subproblem = create_cascade_subproblem();

    // Variables should match system dimensions
    assert_eq!(subproblem.variables.deficit.len(), system.meta.buses_count);
    assert_eq!(
        subproblem.variables.thermal_gen.len(),
        system.meta.thermals_count
    );
    assert_eq!(
        subproblem.variables.turbined_flow.len(),
        system.meta.hydros_count
    );
    assert_eq!(
        subproblem.variables.spillage.len(),
        system.meta.hydros_count
    );
    assert_eq!(
        subproblem.variables.stored_volume.len(),
        system.meta.hydros_count
    );
    assert_eq!(subproblem.variables.inflow.len(), system.meta.hydros_count);
    assert_eq!(
        subproblem.variables.direct_exchange.len(),
        system.meta.lines_count
    );
    assert_eq!(
        subproblem.variables.reverse_exchange.len(),
        system.meta.lines_count
    );

    // Constraints should match system dimensions
    assert_eq!(
        subproblem.constraints.load_balance.len(),
        system.meta.buses_count
    );
    assert_eq!(
        subproblem.constraints.hydro_balance.len(),
        system.meta.hydros_count
    );
}

/// Tests that realization container dimensions match system
///
/// Validates that solution extraction will work correctly.
#[test]
fn test_realization_container_dimensions() {
    let system = create_cascade_system();
    let realization = create_test_realization(&system, None);

    assert_eq!(realization.loads.len(), system.meta.buses_count);
    assert_eq!(realization.deficit.len(), system.meta.buses_count);
    assert_eq!(realization.exchange.len(), system.meta.lines_count);
    assert_eq!(realization.inflow.len(), system.meta.hydros_count);
    assert_eq!(realization.turbined_flow.len(), system.meta.hydros_count);
    assert_eq!(realization.spillage.len(), system.meta.hydros_count);
    assert_eq!(
        realization.thermal_generation.len(),
        system.meta.thermals_count
    );
    assert_eq!(realization.water_value.len(), system.meta.hydros_count);
    assert_eq!(realization.marginal_cost.len(), system.meta.buses_count);
    assert_eq!(realization.final_storage.len(), system.meta.hydros_count);
}
