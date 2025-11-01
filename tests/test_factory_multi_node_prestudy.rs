//! Tests for multi-node pre-study generation via factory API (production path)
//!
//! This test suite verifies PAR-007 implementation for the factory API,
//! which is used by production code (SddpAlgorithm::from_files).
///
/// Key tests:
/// - Verifies factory API loads successfully with storage state (1 pre-study node)
/// - Tests all existing examples work with new multi-node pre-study logic
/// - Verifies state_variables field from JSON is properly read
/// - Tests production code path: from_files() → build() → build_sddp_graph()
///
/// Note: These tests focus on successful loading and training, which indirectly
/// verifies the graph structure is correct. Direct graph inspection tests are
/// in the builder API tests (test_sddp_instance_builder.rs).
use powers_rs::sddp::SddpAlgorithm;
use std::fs;

/// Helper to create file paths for example JSON files
fn json_paths(example: &str) -> (String, String, String, String) {
    let base = format!("examples/{}", example);
    (
        format!("{}/config.json", base),
        format!("{}/system.json", base),
        format!("{}/graph.json", base),
        format!("{}/recourse.json", base),
    )
}

#[test]
fn test_factory_api_loads_with_storage_state() {
    // Use example 01-deterministic which has state_variables: "storage"
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("01-deterministic");

    // Verify graph.json has state_variables: "storage"
    let graph_content =
        fs::read_to_string(&graph_path).expect("Failed to read graph.json");
    assert!(
        graph_content.contains("\"state_variables\": \"storage\""),
        "Example should use storage state"
    );

    // Load via factory API (production path)
    let result = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    );

    assert!(
        result.is_ok(),
        "Factory API should load successfully with storage state: {:?}",
        result.err()
    );
}

#[test]
fn test_factory_api_deterministic_example_trains() {
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("01-deterministic");

    let mut sddp = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    )
    .expect("Failed to load SDDP from files");

    // Training should succeed with new multi-node pre-study logic
    let result = sddp.train();

    assert!(
        result.is_ok(),
        "Training should succeed with new pre-study logic: {:?}",
        result.err()
    );
}

#[test]
fn test_factory_api_multistage_example() {
    // Example 03-multistage has more stages, verify it works
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("03-multistage");

    let result = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    );

    assert!(
        result.is_ok(),
        "Multistage example should load successfully: {:?}",
        result.err()
    );
}

#[test]
fn test_factory_api_cascade_example() {
    // Example 04-cascade has multiple hydros in cascade
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("04-cascade");

    let result = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    );

    assert!(
        result.is_ok(),
        "Cascade example should load successfully: {:?}",
        result.err()
    );
}

#[test]
fn test_factory_api_stochastic_example() {
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("02-stochastic");

    let result = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    );

    assert!(
        result.is_ok(),
        "Stochastic example should load successfully: {:?}",
        result.err()
    );
}

#[test]
fn test_factory_api_all_examples_load() {
    // Verify all examples work with new multi-node pre-study logic
    let examples = vec![
        "01-deterministic",
        "02-stochastic",
        "03-multistage",
        "04-cascade",
    ];

    for example in examples {
        let (config_path, system_path, graph_path, recourse_path) =
            json_paths(example);

        let result = SddpAlgorithm::from_files(
            &config_path,
            &system_path,
            &graph_path,
            &recourse_path,
        );

        assert!(
            result.is_ok(),
            "Example {} should load successfully with valid state_variables: {:?}",
            example,
            result.err()
        );
    }
}

#[test]
fn test_factory_api_example_trains_successfully() {
    // Run a full training cycle to verify the graph structure is correct
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("01-deterministic");

    let mut sddp = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    )
    .expect("Failed to load SDDP from files");

    // Full training should succeed
    let training_result = sddp.train().expect("Training should succeed");

    // Verify we have iterations
    assert!(
        !training_result.iterations().is_empty(),
        "Should have completed iterations"
    );
}

#[test]
fn test_factory_api_with_simulation() {
    // Test full workflow: load, train, simulate
    let (config_path, system_path, graph_path, recourse_path) =
        json_paths("01-deterministic");

    let mut sddp = SddpAlgorithm::from_files(
        &config_path,
        &system_path,
        &graph_path,
        &recourse_path,
    )
    .expect("Failed to load SDDP from files");

    // Train
    let _ = sddp.train().expect("Training should succeed");

    // Simulate (config has num_simulation_scenarios set)
    let simulation_result = sddp.simulate();

    assert!(
        simulation_result.is_ok(),
        "Simulation should succeed with new pre-study logic: {:?}",
        simulation_result.err()
    );
}
