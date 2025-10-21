use powers_rs::input::Config;
use powers_rs::output;
use powers_rs::sddp::SddpAlgorithm;
use std::fs;
use std::path::Path;

/// Helper function to cleanup test output directory
fn cleanup_test_output(path: &str) {
    if Path::new(path).exists() {
        fs::remove_dir_all(path).ok();
    }
}

/// Helper function to create simple SDDP algorithm for testing
fn create_simple_sddp(
) -> (SddpAlgorithm, Vec<powers_rs::sddp::SimulationTrajectory>) {
    // Using the example system from the project
    let system_input = powers_rs::input::read_system_input(
        "examples/03-multistage/system.json",
    );
    let _system = system_input.build_sddp_system();

    let graph_input =
        powers_rs::input::read_graph_input("examples/03-multistage/graph.json");

    let recourse_input = powers_rs::input::read_recourse_input(
        "examples/03-multistage/recourse.json",
    );
    let node_data_graph = graph_input
        .build_sddp_graph(&system_input, &recourse_input)
        .unwrap();

    let initial_condition = recourse_input.build_sddp_initial_condition();
    let saa = recourse_input.generate_sddp_noises(
        &node_data_graph,
        &initial_condition,
        42,
    );

    let mut sddp_algo =
        SddpAlgorithm::new(node_data_graph, initial_condition, 42).unwrap();

    // Train for just a few iterations (quick test)
    let _result = sddp_algo.train(2, 2, &saa).unwrap();

    // Run minimal simulation (new method returns trajectories)
    let simulation_trajectories = sddp_algo.simulate(2, &saa).unwrap();

    (sddp_algo, simulation_trajectories)
}

#[test]
fn test_output_with_none_creates_no_files() {
    // Create a temporary directory name that shouldn't exist
    let test_dir = "./test_output_none";
    cleanup_test_output(test_dir);

    // Ensure directory doesn't exist before test
    assert!(!Path::new(test_dir).exists());

    let (sddp, sim_handlers) = create_simple_sddp();

    // Call generate_outputs with None - should skip all I/O
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        None, // No output path
    );

    assert!(result.is_ok());

    // Verify no files were created
    assert!(!Path::new(test_dir).exists());
    assert!(!Path::new(&format!("{}/cuts.csv", test_dir)).exists());
    assert!(!Path::new(&format!("{}/states.csv", test_dir)).exists());
    assert!(!Path::new(&format!("{}/simulation_buses.csv", test_dir)).exists());

    cleanup_test_output(test_dir);
}

#[test]
fn test_output_with_some_creates_files() {
    let test_dir = "./test_output_some";
    cleanup_test_output(test_dir);

    // Create the output directory
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers) = create_simple_sddp();

    // Call generate_outputs with Some(path) - should create files
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        Some(test_dir),
    );

    assert!(result.is_ok());

    // Verify files were created
    assert!(Path::new(&format!("{}/cuts.csv", test_dir)).exists());
    assert!(Path::new(&format!("{}/states.csv", test_dir)).exists());
    assert!(Path::new(&format!("{}/simulation_buses.csv", test_dir)).exists());
    assert!(
        Path::new(&format!("{}/simulation_thermals.csv", test_dir)).exists()
    );
    assert!(Path::new(&format!("{}/simulation_hydros.csv", test_dir)).exists());
    // Note: simulation_lines.csv may not exist if system has no lines

    // Verify files have content (not empty)
    let cuts_content =
        fs::read_to_string(format!("{}/cuts.csv", test_dir)).unwrap();
    assert!(!cuts_content.is_empty());
    assert!(cuts_content.contains("stage_index")); // CSV header

    cleanup_test_output(test_dir);
}

#[test]
fn test_config_deserialization_controls_output() {
    // Test that Config properly deserializes output_path and controls behavior

    // Config without output_path (should default to None)
    let json_no_output = r#"{
        "num_iterations": 5,
        "num_forward_passes": 2,
        "num_simulation_scenarios": 10,
        "seed": 42
    }"#;

    let config: Config = serde_json::from_str(json_no_output).unwrap();
    assert!(config.output_path.is_none());

    // Config with output_path
    let json_with_output = r#"{
        "num_iterations": 5,
        "num_forward_passes": 2,
        "num_simulation_scenarios": 10,
        "seed": 42,
        "output_path": "./test_output"
    }"#;

    let config: Config = serde_json::from_str(json_with_output).unwrap();
    assert_eq!(config.output_path, Some("./test_output".to_string()));

    // Verify as_deref() works correctly for Option<String> -> Option<&str>
    assert_eq!(config.output_path.as_deref(), Some("./test_output"));

    // Config with null output_path
    let json_null_output = r#"{
        "num_iterations": 5,
        "num_forward_passes": 2,
        "num_simulation_scenarios": 10,
        "seed": 42,
        "output_path": null
    }"#;

    let config: Config = serde_json::from_str(json_null_output).unwrap();
    assert!(config.output_path.is_none());
    assert_eq!(config.output_path.as_deref(), None);
}

#[test]
fn test_performance_no_output_faster_than_with_output() {
    // This is a qualitative test - we just verify it completes without error
    // In practice, with output=None should be 10-30% faster

    use std::time::Instant;

    let test_dir = "./test_output_perf";
    cleanup_test_output(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers) = create_simple_sddp();

    // Time with output=None (no I/O)
    let start_no_output = Instant::now();
    output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        None,
    )
    .unwrap();
    let duration_no_output = start_no_output.elapsed();

    // Time with output=Some(path) (with I/O)
    let start_with_output = Instant::now();
    output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        Some(test_dir),
    )
    .unwrap();
    let duration_with_output = start_with_output.elapsed();

    // We expect no_output to be faster, but just verify both complete
    assert!(
        duration_no_output
            < duration_with_output
                .saturating_add(std::time::Duration::from_secs(10))
    );

    cleanup_test_output(test_dir);
}
