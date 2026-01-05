//! Factory API tests
//!
//! Tests the SddpAlgorithm factory methods that construct from configuration files.
//! Validates error handling, file loading, and system initialization.

use powers_rs::sddp::SddpAlgorithm;
use std::fs;
use std::path::Path;

/// Test that factory API works with valid example files
#[test]
fn test_factory_api_with_valid_inputs() {
    let result = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(
        result.is_ok(),
        "Factory should succeed with valid inputs: {:?}",
        result.err()
    );

    let sddp = result.unwrap();
    assert_eq!(sddp.config().training.num_iterations, 50);
    assert_eq!(sddp.config().training.num_forward_passes, 1);
    assert_eq!(sddp.config().general.seed, 42);
}

/// Test that factory API can train successfully
#[test]
fn test_factory_api_train() {
    let mut sddp = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Factory should succeed");

    // Train for just 2 iterations to keep test fast
    let result = sddp.train();

    assert!(
        result.is_ok(),
        "Training should succeed: {:?}",
        result.err()
    );

    let training_result = result.unwrap();
    assert_eq!(training_result.iterations().len(), 50);
}

/// Test that factory API validation catches zero iterations
#[test]
fn test_factory_api_validation_zero_iterations() {
    // Create temporary config with zero iterations
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("invalid_config_zero_iter.json");

    let invalid_config = r#"{
        "general": {
            "seed": 0
        },
        "training": {
            "num_iterations": 0,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 128
        }
    }"#;

    fs::write(&config_path, invalid_config).expect("Failed to write test file");

    let result = SddpAlgorithm::from_files(
        &config_path,
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(result.is_err(), "Factory should reject zero iterations");

    if let Err(error) = result {
        let error_msg = format!("{}", error);
        assert!(
            error_msg.contains("num_iterations"),
            "Error should mention num_iterations: {}",
            error_msg
        );
        assert!(
            error_msg.contains("positive"),
            "Error should say 'positive': {}",
            error_msg
        );
    }

    // Cleanup
    let _ = fs::remove_file(&config_path);
}

/// Test that factory API validation catches zero forward passes
#[test]
fn test_factory_api_validation_zero_forward_passes() {
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("invalid_config_zero_passes.json");

    let invalid_config = r#"{
        "general": {
            "seed": 0
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 0
        },
        "simulation": {
            "num_scenarios": 128
        }
    }"#;

    fs::write(&config_path, invalid_config).expect("Failed to write test file");

    let result = SddpAlgorithm::from_files(
        &config_path,
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(result.is_err(), "Factory should reject zero forward passes");

    if let Err(error) = result {
        let error_msg = format!("{}", error);
        assert!(
            error_msg.contains("num_forward_passes"),
            "Error should mention num_forward_passes: {}",
            error_msg
        );
    }

    // Cleanup
    let _ = fs::remove_file(&config_path);
}

/// Test that factory API validation catches zero simulation scenarios
#[test]
fn test_factory_api_validation_zero_simulation() {
    let temp_dir = std::env::temp_dir();
    let config_path = temp_dir.join("invalid_config_zero_sim.json");

    let invalid_config = r#"{
        "general": {
            "seed": 0
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 0
        }
    }"#;

    fs::write(&config_path, invalid_config).expect("Failed to write test file");

    let result = SddpAlgorithm::from_files(
        &config_path,
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(
        result.is_err(),
        "Factory should reject zero simulation scenarios"
    );

    if let Err(error) = result {
        let error_msg = format!("{}", error);
        assert!(
            error_msg.contains("num_scenarios"),
            "Error should mention num_scenarios: {}",
            error_msg
        );
    }

    // Cleanup
    let _ = fs::remove_file(&config_path);
}

/// Test that factory API handles missing files gracefully
#[test]
fn test_factory_api_missing_file() {
    let result = SddpAlgorithm::from_files(
        "nonexistent/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    );

    assert!(result.is_err(), "Should return error for missing file");

    if let Err(error) = result {
        let error_msg = format!("{}", error);
        // Check that error provides context
        assert!(
            error_msg.contains("nonexistent")
                || error_msg.contains("not found"),
            "Error should mention the missing file: {}",
            error_msg
        );
    }
}

/// Test that SddpInstance accessors work correctly
#[test]
fn test_sddp_instance_accessors() {
    let sddp = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Factory should succeed");

    // Test config accessor
    assert_eq!(sddp.config().training.num_iterations, 50);
    assert_eq!(sddp.config().training.num_forward_passes, 1);

    // Test algorithm accessor
    let algorithm = sddp.algorithm();
    assert!(!algorithm.study_period_ids.is_empty());

    // Test SAA accessor
    let _saa = sddp.saa();
}

/// Test that Input::from_paths works with flexible paths
#[test]
fn test_input_from_paths_flexible() {
    use powers_rs::input::Input;

    // Test with Path objects
    let result = Input::from_paths(
        Path::new("examples/01-deterministic/config.json"),
        Path::new("examples/01-deterministic/system.json"),
        Path::new("examples/01-deterministic/graph.json"),
        Path::new("examples/01-deterministic/recourse.json"),
    );

    assert!(
        result.is_ok(),
        "from_paths should work with Path objects: {:?}",
        result.err()
    );

    let input = result.unwrap();
    assert_eq!(input.config.training.num_iterations, 50);
    assert_eq!(input.system.buses.len(), 1);
    assert_eq!(input.graph.nodes.len(), 2);
}

/// Test backward compatibility: Input::build still works
#[test]
fn test_input_build_backward_compatibility() {
    use powers_rs::input::Input;

    let input = Input::build("examples/01-deterministic")
        .expect("Input::build failed");
    assert_eq!(input.config.training.num_iterations, 50);
    assert_eq!(input.system.buses.len(), 1);
}
