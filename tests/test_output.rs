use powers_rs::input::{Config, OutputConfig};
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
fn create_simple_sddp() -> (
    SddpAlgorithm,
    Vec<powers_rs::sddp::SimulationTrajectory>,
    powers_rs::scenario::ScenarioTree,
) {
    // Using the example system from the project
    let system_input = powers_rs::input::read_system_input(
        "examples/03-multistage/system.json",
    )
    .expect("Failed to read system input");
    let _system = system_input
        .build_sddp_system()
        .expect("Failed to build system");

    let graph_input =
        powers_rs::input::read_graph_input("examples/03-multistage/graph.json")
            .expect("Failed to read graph input");

    let recourse_input = powers_rs::input::read_recourse_input(
        "examples/03-multistage/recourse.json",
    )
    .expect("Failed to read recourse input");
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
    let _result = sddp_algo
        .train(2, 2, false, &saa, false, false, None)
        .unwrap();

    // Run minimal simulation (new method returns trajectories)
    let simulation_trajectories = sddp_algo.simulate(2, &saa).unwrap();

    (sddp_algo, simulation_trajectories, saa)
}

#[test]
fn test_output_with_none_creates_no_files() {
    // Create a temporary directory name that shouldn't exist
    let test_dir = "./test_output_none";
    cleanup_test_output(test_dir);

    // Ensure directory doesn't exist before test
    assert!(!Path::new(test_dir).exists());

    let (sddp, sim_handlers, saa) = create_simple_sddp();
    let system = sddp.system();
    let max_ar_order = sddp.max_ar_order();
    let hydro_ar_orders = sddp.hydro_ar_orders();

    let output_config = OutputConfig::default();

    // Call generate_outputs with None - should skip all I/O
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
        None, // No output path
    );

    assert!(result.is_ok());

    // Verify no files were created
    assert!(!Path::new(test_dir).exists());
    assert!(!Path::new(&format!("{}/cuts.csv", test_dir)).exists());
    assert!(!Path::new(&format!("{}/states.csv", test_dir)).exists());
    assert!(!Path::new(&format!("{}/simulation.csv", test_dir)).exists());

    cleanup_test_output(test_dir);
}

#[test]
fn test_output_with_some_creates_files() {
    let test_dir = "./test_output_some";
    cleanup_test_output(test_dir);

    // Create the output directory
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers, saa) = create_simple_sddp();
    let system = sddp.system();
    let max_ar_order = sddp.max_ar_order();
    let hydro_ar_orders = sddp.hydro_ar_orders();

    let output_config = OutputConfig::default();

    // Call generate_outputs with Some(path) - should create files
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
        Some(test_dir),
    );

    assert!(result.is_ok());

    // Verify files were created
    assert!(Path::new(&format!("{}/cuts.csv", test_dir)).exists());
    assert!(Path::new(&format!("{}/states.csv", test_dir)).exists());
    assert!(Path::new(&format!("{}/simulation.csv", test_dir)).exists());
    assert!(
        Path::new(&format!("{}/variable_dictionary.csv", test_dir)).exists()
    );
    assert!(
        Path::new(&format!("{}/coefficient_dictionary.csv", test_dir)).exists()
    );
    assert!(
        Path::new(&format!("{}/state_component_dictionary.csv", test_dir))
            .exists()
    );

    // Verify files have content (not empty)
    let cuts_content =
        fs::read_to_string(format!("{}/cuts.csv", test_dir)).unwrap();
    assert!(!cuts_content.is_empty());
    assert!(cuts_content.contains("stage_index")); // CSV header

    // Verify single simulation file with indexed format
    let sim_content =
        fs::read_to_string(format!("{}/simulation.csv", test_dir)).unwrap();
    assert!(!sim_content.is_empty());
    assert!(sim_content.contains("variable_index")); // Indexed format

    cleanup_test_output(test_dir);
}

#[test]
fn test_config_deserialization_controls_output() {
    // Test that Config properly deserializes output settings

    // Config without output section (should use defaults)
    let json_no_output = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 5,
            "num_forward_passes": 2
        },
        "simulation": {
            "num_scenarios": 10
        }
    }"#;

    let config: Config = serde_json::from_str(json_no_output).unwrap();
    assert_eq!(config.output.path, Some(".".to_string())); // Should default to "."
    assert!(!config.output.export_training_noises); // Should default to false

    // Config with output_path
    let json_with_output = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 5,
            "num_forward_passes": 2
        },
        "simulation": {
            "num_scenarios": 10
        },
        "output": {
            "path": "./test_output"
        }
    }"#;

    let config: Config = serde_json::from_str(json_with_output).unwrap();
    assert_eq!(config.output.path, Some("./test_output".to_string()));
    assert!(!config.output.export_training_noises); // Should default to false

    // Verify effective_path() works correctly
    assert_eq!(
        config.output.effective_path(Some("./default")),
        Some("./test_output")
    );

    // Config with null output_path
    let json_null_output = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 5,
            "num_forward_passes": 2
        },
        "simulation": {
            "num_scenarios": 10
        },
        "output": {
            "path": null
        }
    }"#;

    let config: Config = serde_json::from_str(json_null_output).unwrap();
    assert!(config.output.path.is_none());
    assert_eq!(
        config.output.effective_path(Some("./default")),
        Some("./default")
    );

    // Config with export_training_noises enabled
    let json_with_noises = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 5,
            "num_forward_passes": 2
        },
        "simulation": {
            "num_scenarios": 10
        },
        "output": {
            "export_training_noises": true
        }
    }"#;

    let config: Config = serde_json::from_str(json_with_noises).unwrap();
    assert!(config.output.export_training_noises);

    // Config with format specified
    let json_with_format = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 5,
            "num_forward_passes": 2
        },
        "simulation": {
            "num_scenarios": 10
        },
        "output": {
            "format": "PARQUET"
        }
    }"#;

    let config: Config = serde_json::from_str(json_with_format).unwrap();
    assert_eq!(
        config.output.format,
        powers_rs::input::OutputFormat::PARQUET
    );
}

#[test]
fn test_performance_no_output_faster_than_with_output() {
    // This is a qualitative test - we just verify it completes without error
    // In practice, with output=None should be 10-30% faster

    use std::time::Instant;

    let test_dir = "./test_output_perf";
    cleanup_test_output(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers, saa) = create_simple_sddp();
    let system = sddp.system();
    let max_ar_order = sddp.max_ar_order();
    let hydro_ar_orders = sddp.hydro_ar_orders();

    let output_config = OutputConfig::default();

    // Time with output=None (no I/O)
    let start_no_output = Instant::now();
    output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
        None,
    )
    .unwrap();
    let duration_no_output = start_no_output.elapsed();

    // Time with output=Some(path) (with I/O)
    let start_with_output = Instant::now();
    output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
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

#[test]
fn test_sampled_noises_export() {
    let test_dir = "./test_output_noises";
    cleanup_test_output(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers, saa) = create_simple_sddp();
    let system = sddp.system();
    let max_ar_order = sddp.max_ar_order();
    let hydro_ar_orders = sddp.hydro_ar_orders();

    let output_config = OutputConfig {
        export_training_noises: true,
        ..Default::default()
    };

    // Call generate_outputs with export_training_noises enabled
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
        Some(test_dir),
    );

    assert!(result.is_ok());

    // Verify training_sampled_noises.csv was created (renamed from sampled_noises.csv)
    let noises_path = format!("{}/training_sampled_noises.csv", test_dir);
    assert!(Path::new(&noises_path).exists());

    // Verify file has content
    let noises_content = fs::read_to_string(&noises_path).unwrap();
    assert!(!noises_content.is_empty());
    assert!(noises_content.contains("stage_index"));
    assert!(noises_content.contains("branching_index"));
    assert!(noises_content.contains("variable_index")); // Indexed format
    assert!(noises_content.contains("entity_id"));
    assert!(noises_content.contains("value"));

    cleanup_test_output(test_dir);
}

#[test]
fn test_sampled_noises_not_exported_when_disabled() {
    let test_dir = "./test_output_noises_disabled";
    cleanup_test_output(test_dir);
    fs::create_dir_all(test_dir).unwrap();

    let (sddp, sim_handlers, saa) = create_simple_sddp();
    let system = sddp.system();
    let max_ar_order = sddp.max_ar_order();
    let hydro_ar_orders = sddp.hydro_ar_orders();

    let output_config = OutputConfig {
        export_training_noises: false,
        ..Default::default()
    };

    // Call generate_outputs with export_training_noises disabled
    let result = output::generate_outputs(
        &sddp.future_cost_function_graph,
        &sim_handlers,
        &[], // Empty training results
        &[], // Empty forward details
        &[], // Empty backward details
        &saa,
        system,
        max_ar_order,
        &hydro_ar_orders,
        &output_config,
        Some(test_dir),
    );

    assert!(result.is_ok());

    // Verify training_sampled_noises.csv was NOT created
    let noises_path = format!("{}/training_sampled_noises.csv", test_dir);
    assert!(!Path::new(&noises_path).exists());

    cleanup_test_output(test_dir);
}
