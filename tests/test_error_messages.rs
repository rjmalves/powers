//! Error message tests
//!
//! Validates that error messages are clear, actionable, and include relevant context.
//! Tests various error types: validation, solver, graph, I/O errors.

use powers_rs::error::{
    GraphError, IoError, PowersError, SolverError, ValidationError,
};
use powers_rs::input::{read_config_input, Input};
use powers_rs::input_validation::InputValidator;
use powers_rs::sddp::SddpAlgorithm;
use std::fs;
use std::path::Path;

#[test]
fn test_validation_error_includes_file_name() {
    let error = ValidationError::InvalidFieldValue {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        value: "0".to_string(),
        constraint: "must be positive".to_string(),
        suggestion: "Set to at least 1".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("config.json"), "Should mention file name");
}

#[test]
fn test_validation_error_includes_field_name() {
    let error = ValidationError::InvalidFieldValue {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        value: "0".to_string(),
        constraint: "must be positive".to_string(),
        suggestion: "Set to at least 1".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("num_iterations"),
        "Should mention field name"
    );
}

#[test]
fn test_validation_error_includes_value() {
    let error = ValidationError::InvalidFieldValue {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        value: "0".to_string(),
        constraint: "must be positive".to_string(),
        suggestion: "Set to at least 1".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("'0'"), "Should show the invalid value");
}

#[test]
fn test_validation_error_includes_constraint() {
    let error = ValidationError::InvalidFieldValue {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        value: "0".to_string(),
        constraint: "must be positive (> 0)".to_string(),
        suggestion: "Set to at least 1".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("Constraint:"), "Should show constraint");
    assert!(
        message.contains("must be positive"),
        "Should explain constraint"
    );
}

#[test]
fn test_validation_error_includes_suggestion() {
    let error = ValidationError::InvalidFieldValue {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        value: "0".to_string(),
        constraint: "must be positive".to_string(),
        suggestion: "Set num_iterations to at least 1".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("Suggestion:"), "Should show suggestion");
    assert!(
        message.contains("Set num_iterations to at least 1"),
        "Should provide actionable fix"
    );
}

#[test]
fn test_io_error_file_not_found_includes_path() {
    let error = IoError::FileNotFound {
        path: "/nonexistent/config.json".to_string(),
        current_dir: "/home/user/powers".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("/nonexistent/config.json"),
        "Should show file path"
    );
}

#[test]
fn test_io_error_file_not_found_includes_current_dir() {
    let error = IoError::FileNotFound {
        path: "config.json".to_string(),
        current_dir: "/home/user/powers".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("/home/user/powers"),
        "Should show current directory for context"
    );
}

#[test]
fn test_io_error_permission_denied_includes_suggestion() {
    let error = IoError::PermissionDenied {
        path: "/root/config.json".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("chmod"),
        "Should suggest using chmod to fix permissions"
    );
}

#[test]
fn test_solver_error_infeasible_explains_cause() {
    let error = SolverError::Infeasible {
        context: "node 5, iteration 10".to_string(),
        suggestion: "Check generation capacity >= peak load".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("Common causes"),
        "Should list common causes"
    );
    assert!(
        message.contains("infeasible"),
        "Should explain what infeasible means"
    );
}

#[test]
fn test_solver_error_unbounded_explains_cause() {
    let error = SolverError::Unbounded {
        context: "node 3".to_string(),
        suggestion: "Check for missing upper bounds".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("Common causes"),
        "Should list common causes"
    );
    assert!(
        message.contains("unbounded"),
        "Should explain what unbounded means"
    );
}

#[test]
fn test_graph_error_disconnected_lists_unreachable_nodes() {
    let error = GraphError::DisconnectedGraph {
        nodes: "[5, 6, 7]".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("Unreachable nodes"),
        "Should list unreachable nodes"
    );
    assert!(
        message.contains("[5, 6, 7]"),
        "Should show specific node IDs"
    );
}

#[test]
fn test_graph_error_invalid_probability_shows_sum() {
    let error = GraphError::InvalidProbabilitySum {
        node_id: 2,
        sum: 0.85,
        edges: "[(2→3, 0.3), (2→4, 0.25), (2→5, 0.3)]".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("0.85"), "Should show actual sum");
    assert!(message.contains("1.0"), "Should show expected sum");
}

#[test]
fn test_missing_field_error_suggests_fix() {
    let error = ValidationError::MissingField {
        file: "config.json".to_string(),
        field: "num_iterations".to_string(),
        suggestion: "Add \"num_iterations\": 100 to your config file"
            .to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("Suggestion"), "Should provide suggestion");
    assert!(
        message.contains("INPUT-SPECIFICATION"),
        "Should reference documentation"
    );
}

#[test]
fn test_constraint_violation_shows_details() {
    let error = ValidationError::ConstraintViolation {
        file: "system.json".to_string(),
        context: "Thermal 'thermal_1'".to_string(),
        constraint: "min_generation <= max_generation".to_string(),
        details: "min_generation=100.0, max_generation=50.0".to_string(),
        suggestion: "Set max_generation >= 100.0".to_string(),
    };

    let message = format!("{}", error);
    assert!(message.contains("Found:"), "Should show actual values");
    assert!(message.contains("100.0"), "Should show specific values");
    assert!(message.contains("50.0"), "Should show specific values");
}

#[test]
fn test_invalid_reference_lists_available_values() {
    let error = ValidationError::InvalidReference {
        file: "system.json".to_string(),
        context: "Line 'line_1'".to_string(),
        ref_type: "bus_id".to_string(),
        ref_id: "99".to_string(),
        available: "0, 1, 2, 3".to_string(),
        suggestion: "Change to a valid bus ID (0-3)".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("Available"),
        "Should list available values"
    );
    assert!(message.contains("0, 1, 2, 3"), "Should show valid options");
}

#[test]
fn test_json_parse_error_suggests_validation_tool() {
    let error = ValidationError::JsonParseError {
        file: "config.json".to_string(),
        error: "expected `,` or `}` at line 5 column 12".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("jsonlint"),
        "Should suggest JSON validation tool"
    );
}

#[test]
fn test_empty_array_error_suggests_adding_element() {
    let error = ValidationError::EmptyArray {
        file: "system.json".to_string(),
        field: "hydros".to_string(),
        suggestion: "Add at least one hydro unit".to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("at least one"),
        "Should suggest adding element"
    );
}

#[test]
fn test_solver_numerical_error_explains_scaling() {
    let error = SolverError::NumericalError {
        context: "node 10".to_string(),
        suggestion: "Scale problem so coefficients are between 1e-6 and 1e6"
            .to_string(),
    };

    let message = format!("{}", error);
    assert!(
        message.contains("scaling"),
        "Should mention problem scaling"
    );
    assert!(message.contains("1e-6"), "Should suggest specific range");
}

#[test]
fn test_real_zero_iterations_error_is_clear() {
    // Create invalid config file
    let temp_dir = "test_temp_errors";
    fs::create_dir_all(temp_dir).unwrap();
    let config_path = format!("{}/config.json", temp_dir);

    fs::write(
        &config_path,
        r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 0,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 128
        }
    }"#,
    )
    .unwrap();

    // Try to load - should parse successfully but validation should fail
    let config = read_config_input(&config_path)
        .expect("read_config_input failed");
    let validation_result = InputValidator::validate_config_minimal(&config);

    assert!(
        validation_result.is_err(),
        "Validation should fail for zero iterations"
    );

    if let Err(error) = validation_result {
        let message = format!("{}", error);
        assert!(
            message.contains("num_iterations"),
            "Error should mention num_iterations: {}",
            message
        );
        assert!(
            message.contains("positive"),
            "Error should say 'positive': {}",
            message
        );
        assert!(
            message.contains("Suggestion"),
            "Error should provide suggestion: {}",
            message
        );
    }

    // Cleanup
    fs::remove_dir_all(temp_dir).unwrap();
}

#[test]
fn test_real_missing_file_error_is_helpful() {
    // Use Input::from_paths which returns Result instead of read_config_input which panics
    use std::path::Path;
    let result = Input::from_paths(
        Path::new("nonexistent_directory/config.json"),
        Path::new("examples/03-multistage/system.json"),
        Path::new("examples/03-multistage/graph.json"),
        Path::new("examples/03-multistage/recourse.json"),
    );

    assert!(result.is_err(), "Should return error for missing file");

    if let Err(error) = result {
        let message = format!("{}", error);
        // Should be a PowersError with helpful message about the file not found
        assert!(
            message.contains("nonexistent")
                || message.contains("not found")
                || message.contains("No such file"),
            "Error should explain file not found: {}",
            message
        );
    }
}

#[test]
fn test_validation_error_converts_to_powers_error() {
    let validation_error = Box::new(ValidationError::InvalidFieldValue {
        file: "test.json".to_string(),
        field: "field1".to_string(),
        value: "bad".to_string(),
        constraint: "must be numeric".to_string(),
        suggestion: "Use a number".to_string(),
    });

    let powers_error: PowersError = validation_error.into();
    assert!(matches!(powers_error, PowersError::Validation(_)));
}

#[test]
fn test_io_error_converts_to_powers_error() {
    let io_error = Box::new(IoError::FileNotFound {
        path: "test.json".to_string(),
        current_dir: "/tmp".to_string(),
    });

    let powers_error: PowersError = io_error.into();
    assert!(matches!(powers_error, PowersError::Io(_)));
}

#[test]
fn test_solver_error_converts_to_powers_error() {
    let solver_error = Box::new(SolverError::Infeasible {
        context: "node 1".to_string(),
        suggestion: "Check constraints".to_string(),
    });

    let powers_error: PowersError = solver_error.into();
    assert!(matches!(powers_error, PowersError::Solver(_)));
}

#[test]
fn test_graph_error_converts_to_powers_error() {
    let graph_error = Box::new(GraphError::DisconnectedGraph {
        nodes: "[1, 2, 3]".to_string(),
    });

    let powers_error: PowersError = graph_error.into();
    assert!(matches!(powers_error, PowersError::Graph(_)));
}

#[test]
fn test_string_converts_to_powers_error() {
    let powers_error: PowersError = "Generic error".into();
    assert!(matches!(powers_error, PowersError::Other(_)));
}

#[test]
fn test_std_io_error_converts_to_powers_error() {
    let io_err =
        std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let powers_error: PowersError = io_err.into();
    assert!(matches!(powers_error, PowersError::Io(_)));
}

#[test]
fn test_serde_json_error_converts_to_validation_error() {
    let bad_json = "{ invalid json }";
    let json_err =
        serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
    let powers_error: PowersError = json_err.into();
    assert!(matches!(powers_error, PowersError::Validation(_)));
}

#[test]
fn test_factory_api_returns_powers_error_type() {
    let result = SddpAlgorithm::from_files(
        "nonexistent/config.json",
        "examples/03-multistage/system.json",
        "examples/03-multistage/graph.json",
        "examples/03-multistage/recourse.json",
    );

    assert!(result.is_err());
    // Type is PowersError (not String)
    match result {
        Err(_error) => {
            // Successfully got PowersError type
        }
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn test_input_from_paths_returns_powers_error_type() {
    let result = Input::from_paths(
        Path::new("nonexistent/config.json"),
        Path::new("examples/03-multistage/system.json"),
        Path::new("examples/03-multistage/graph.json"),
        Path::new("examples/03-multistage/recourse.json"),
    );

    assert!(result.is_err());
    // Type is PowersError (not String)
    match result {
        Err(_error) => {
            // Successfully got PowersError type
        }
        Ok(_) => panic!("Expected error"),
    }
}

#[test]
fn test_validation_returns_powers_error_type() {
    // Create config with zero value
    let json = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 0,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 128
        }
    }"#;
    let config: powers_rs::input::Config = serde_json::from_str(json).unwrap();

    let result = InputValidator::validate_config_minimal(&config);
    assert!(result.is_err());
    // Type is PowersError (not String)
    let _error: PowersError = result.unwrap_err();
}
