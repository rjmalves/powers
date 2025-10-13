// AR-5: Backward Compatibility Tests
//
// Ensures that the new `noise_models` format (AR-1+) doesn't break existing
// examples that use the legacy `uncertainties` format.
//
// Baseline results captured on 2025-01-02:
// - Example 01 (deterministic): 10 iterations, ~4ms training
// - Example 02 (stochastic):    15 iterations, ~44ms training
// - Example 03 (multistage):    32 iterations, ~809ms training
// - Example 04 (cascade):       8 iterations, ~403ms training

use powers_rs::input::read_recourse_input;
use powers_rs::sddp::SddpAlgorithm;
use std::path::Path;

// ============================================================================
// Helper Functions
// ============================================================================

/// Run a full SDDP example and return training statistics
fn run_example(example_dir: &str) -> Result<TrainingStats, String> {
    let base_path = Path::new(example_dir);

    let config_path = base_path.join("config.json");
    let system_path = base_path.join("system.json");
    let graph_path = base_path.join("graph.json");
    let recourse_path = base_path.join("recourse.json");

    // Load recourse to validate format
    let recourse = read_recourse_input(recourse_path.to_str().unwrap());

    // Validate that old format is used
    assert!(
        recourse.uncertainties.is_some(),
        "Example should use legacy uncertainties format"
    );
    assert!(
        recourse.noise_models.is_none(),
        "Example should NOT use new noise_models format"
    );

    // Use from_files API (factory pattern)
    let mut sddp = SddpAlgorithm::from_files(
        config_path,
        system_path,
        graph_path,
        recourse_path,
    )
    .map_err(|e| format!("{}", e))?;

    let result = sddp.train().map_err(|e| e.to_string())?;

    Ok(TrainingStats {
        num_iterations: result.iterations().len(),
        converged: result.converged(1.0), // 1.0 gap tolerance
        final_gap: result.final_gap(),
        final_lower_bound: result.final_lower_bound,
        final_upper_bound: result.final_upper_bound,
    })
}

#[derive(Debug)]
#[allow(dead_code)] // Fields used for future extensibility
struct TrainingStats {
    num_iterations: usize,
    converged: bool,
    final_gap: f64,
    final_lower_bound: f64,
    final_upper_bound: f64,
}

// ============================================================================
// Example Regression Tests
// ============================================================================

#[test]
fn test_example_01_deterministic_unchanged() {
    // Example 01: Deterministic problem
    // Baseline: 10 iterations
    let result = run_example("examples/01-deterministic");

    assert!(
        result.is_ok(),
        "Example 01 should run successfully: {:?}",
        result.err()
    );

    let stats = result.unwrap();
    assert_eq!(
        stats.num_iterations, 10,
        "Example 01 should converge in 10 iterations (deterministic)"
    );
}

#[test]
fn test_example_02_stochastic_unchanged() {
    // Example 02: Stochastic with 2 hydros
    // Baseline: 15 iterations
    let result = run_example("examples/02-stochastic");

    assert!(
        result.is_ok(),
        "Example 02 should run successfully: {:?}",
        result.err()
    );

    let stats = result.unwrap();

    // Allow small variation in iterations (±2) due to stochastic nature
    assert!(
        stats.num_iterations >= 13 && stats.num_iterations <= 17,
        "Example 02 iterations should be ~15 (got {})",
        stats.num_iterations
    );
}

#[test]
fn test_example_03_multistage_unchanged() {
    // Example 03: Multistage problem
    // Baseline: 32 iterations
    let result = run_example("examples/03-multistage");

    assert!(
        result.is_ok(),
        "Example 03 should run successfully: {:?}",
        result.err()
    );

    let stats = result.unwrap();

    // Allow variation (±3) due to randomness
    assert!(
        stats.num_iterations >= 29 && stats.num_iterations <= 35,
        "Example 03 iterations should be ~32 (got {})",
        stats.num_iterations
    );
}

#[test]
fn test_example_04_cascade_unchanged() {
    // Example 04: Cascade system
    // Baseline: 8 iterations
    let result = run_example("examples/04-cascade");

    assert!(
        result.is_ok(),
        "Example 04 should run successfully: {:?}",
        result.err()
    );

    let stats = result.unwrap();

    // Deterministic problem, should be exact
    assert_eq!(
        stats.num_iterations, 8,
        "Example 04 should converge in 8 iterations"
    );
}

// ============================================================================
// Format Validation Tests
// ============================================================================

#[test]
fn test_legacy_uncertainties_format_parsing() {
    // Old format with uncertainties field
    let json = r#"
    {
        "initial_condition": {
            "storage": [
                {"hydro_id": 0, "value": 50.0}
            ],
            "inflow": []
        },
        "uncertainties": [
            {
                "season_id": 0,
                "num_branchings": 5,
                "distributions": {
                    "load": [
                        {
                            "bus_id": 0,
                            "normal": {"mu": 100.0, "sigma": 20.0}
                        }
                    ],
                    "inflow": [
                        {
                            "hydro_id": 0,
                            "lognormal": {"mu": 3.0, "sigma": 0.5}
                        }
                    ]
                }
            }
        ]
    }
    "#;

    let recourse: powers_rs::input::Recourse =
        serde_json::from_str(json).expect("Old format should parse");

    assert!(
        recourse.uncertainties.is_some(),
        "Old format should have uncertainties"
    );
    assert!(
        recourse.noise_models.is_none(),
        "Old format should NOT have noise_models"
    );

    let uncertainties = recourse.uncertainties.unwrap();
    assert_eq!(uncertainties.len(), 1);
    assert_eq!(uncertainties[0].season_id, 0);
    assert_eq!(uncertainties[0].num_branchings, 5);
}

#[test]
fn test_legacy_format_with_lognormal_distribution() {
    // Verify lognormal distributions still work (common in hydro examples)
    let json = r#"
    {
        "initial_condition": {
            "storage": [
                {"hydro_id": 0, "value": 30.0}
            ],
            "inflow": [
                {"hydro_id": 0, "lag": 1, "value": 20.0}
            ]
        },
        "uncertainties": [
            {
                "season_id": 0,
                "num_branchings": 10,
                "distributions": {
                    "load": [],
                    "inflow": [
                        {
                            "hydro_id": 0,
                            "lognormal": {"mu": 2.996, "sigma": 0.5}
                        }
                    ]
                }
            }
        ]
    }
    "#;

    let recourse: powers_rs::input::Recourse =
        serde_json::from_str(json).expect("Lognormal format should parse");

    assert!(recourse.uncertainties.is_some());
    let uncertainties = recourse.uncertainties.unwrap();
    assert_eq!(uncertainties[0].distributions.inflow.len(), 1);
}

#[test]
fn test_legacy_format_validation_still_works() {
    // Ensure validation catches errors in old format
    use powers_rs::input_validation::InputValidator;

    let json = r#"
    {
        "initial_condition": {
            "storage": [
                {"hydro_id": 0, "value": 30.0}
            ],
            "inflow": []
        },
        "uncertainties": [
            {
                "season_id": 0,
                "num_branchings": 5,
                "distributions": {
                    "load": [],
                    "inflow": [
                        {
                            "hydro_id": 999,
                            "lognormal": {"mu": 3.0, "sigma": 0.5}
                        }
                    ]
                }
            }
        ]
    }
    "#;

    let recourse: powers_rs::input::Recourse =
        serde_json::from_str(json).unwrap();

    // Create minimal system for validation
    let system = powers_rs::input::SystemInput {
        buses: vec![powers_rs::input::BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![powers_rs::input::HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 100.0,
        }],
    };

    // Validation should catch invalid hydro_id
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Validation should catch invalid hydro_id in old format"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("hydro_id") || error.contains("999"),
        "Error should mention invalid hydro_id"
    );
}

// ============================================================================
// Both Formats Rejected Tests
// ============================================================================

#[test]
fn test_reject_both_formats_present() {
    // Should reject if both uncertainties AND noise_models are present
    let json = r#"
    {
        "initial_condition": {
            "storage": [{"hydro_id": 0, "value": 50.0}],
            "inflow": []
        },
        "uncertainties": [
            {
                "season_id": 0,
                "num_branchings": 5,
                "distributions": {
                    "load": [],
                    "inflow": [
                        {"hydro_id": 0, "lognormal": {"mu": 3.0, "sigma": 0.5}}
                    ]
                }
            }
        ],
        "noise_models": [
            {
                "noise_type": "independent",
                "uncertainty_type": "inflow",
                "entity_id": 0,
                "season_id": 0,
                "distribution": {"type": "normal", "mean": 100.0, "std_dev": 20.0},
                "lag_order": null,
                "coefficients": null
            }
        ]
    }
    "#;

    let recourse: powers_rs::input::Recourse =
        serde_json::from_str(json).unwrap();

    // Create minimal system
    let system = powers_rs::input::SystemInput {
        buses: vec![powers_rs::input::BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![powers_rs::input::HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 100.0,
        }],
    };

    // Validation should reject both formats present
    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    assert!(
        result.is_err(),
        "Should reject when both uncertainties and noise_models present"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("uncertainties")
            || error.contains("noise_models")
            || error.contains("both"),
        "Error should mention conflicting formats"
    );
}

#[test]
fn test_reject_neither_format_present() {
    // Should reject if NEITHER uncertainties NOR noise_models are present
    let json = r#"
    {
        "initial_condition": {
            "storage": [{"hydro_id": 0, "value": 50.0}],
            "inflow": []
        }
    }
    "#;

    let recourse: powers_rs::input::Recourse =
        serde_json::from_str(json).unwrap();

    // Create minimal system
    let system = powers_rs::input::SystemInput {
        buses: vec![powers_rs::input::BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![powers_rs::input::HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 100.0,
        }],
    };

    // Validation should reject missing noise specification
    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    assert!(
        result.is_err(),
        "Should reject when neither uncertainties nor noise_models present"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("uncertainties")
            || error.contains("noise_models")
            || error.contains("required"),
        "Error should mention missing noise specification"
    );
}
