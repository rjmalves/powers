mod fixtures;
mod utils;

use fixtures::*;
use utils::assertions::*;

#[test]
fn test_mock_solver_basic() {
    let solver = MockSolver::new();

    // Should be in initial state
    assert_eq!(solver.optimize_call_count(), 0);
    assert_eq!(solver.add_row_call_count(), 0);
}

#[test]
fn test_system_fixtures_available() {
    // Test trivial system JSON is available
    let trivial_json = trivial_system_json();
    assert!(!trivial_json.is_empty());
    assert!(trivial_json.contains("buses"));
    assert!(trivial_json.contains("hydros"));

    // Test simple system JSON is available
    let simple_json = simple_system_json();
    assert!(!simple_json.is_empty());
    assert!(simple_json.contains("buses"));
    assert!(simple_json.contains("hydros"));
}

#[test]
fn test_scenario_fixtures() {
    // Test deterministic scenario
    let det_gen = deterministic_scenario(3, 2, 1);
    let det_saa = det_gen.generate(42);

    // Should have expected structure
    let noises = det_saa.get_noises_by_stage_and_branching(0, 0).unwrap();
    assert_eq!(noises.num_load_entities, 2);
    assert_eq!(noises.num_inflow_entities, 1);

    // Test stochastic scenario
    let stoch_gen = simple_stochastic_scenario(2, 3, 2, 1, 1.0, 0.2, 0.0, 0.3);
    let stoch_saa = stoch_gen.generate(123);

    let noises = stoch_saa.get_noises_by_stage_and_branching(0, 0).unwrap();
    assert_eq!(noises.num_load_entities, 2);
    assert_eq!(noises.num_inflow_entities, 1);

    // Test fan scenario
    let fan_gen = fan_scenario(3, 5, 2, 1);
    let fan_saa = fan_gen.generate(456);

    let noises = fan_saa.get_noises_by_stage_and_branching(0, 0).unwrap();
    assert_eq!(noises.num_load_entities, 2);
    assert_eq!(noises.num_inflow_entities, 1);
}

#[test]
fn test_assertions() {
    // Test float comparison
    assert_float_approx_eq(1.0, 1.0 + 1e-11, 1e-10);

    // Test vector comparison
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0 + 1e-11, 2.0 + 1e-11, 3.0 + 1e-11];
    assert_vec_approx_eq(&a, &b, 1e-10);

    // Test state bounds
    let state = vec![50.0, 75.0];
    let lower = vec![0.0, 0.0];
    let upper = vec![100.0, 100.0];
    assert_state_within_bounds(&state, &lower, &upper, 1e-6);
}

#[test]
fn test_scenario_reproducibility() {
    // Same seed should produce same scenarios
    let generator1 = simple_stochastic_scenario(2, 2, 1, 1, 1.0, 0.1, 0.0, 0.1);
    let generator2 = simple_stochastic_scenario(2, 2, 1, 1, 1.0, 0.1, 0.0, 0.1);

    let saa1 = generator1.generate(42);
    let saa2 = generator2.generate(42);

    let noises1 = saa1.get_noises_by_stage_and_branching(0, 0).unwrap();
    let noises2 = saa2.get_noises_by_stage_and_branching(0, 0).unwrap();

    // Should be identical
    assert_eq!(
        noises1.get_load_innovations(),
        noises2.get_load_innovations()
    );
    assert_eq!(
        noises1.get_inflow_innovations(),
        noises2.get_inflow_innovations()
    );
}
