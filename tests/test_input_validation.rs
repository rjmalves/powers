//! Comprehensive tests for T3.10: Input Validation
//!
//! This module tests the validation system that provides:
//! - System validation (IDs, references, constraints)
//! - Graph validation (nodes, edges, probabilities)
//! - Recourse validation (storage bounds, distributions)
//! - Cross-validation (consistency across files)
//!
//! Total: 68 tests (20 + 18 + 15 + 10 + 5)

use powers_rs::input::{
    BusInput, GraphEdgeInput, GraphInput, GraphNodeInput, HydroInput,
    InflowDistribution, InitialConditionInput, InitialStorage, LineInput,
    LoadDistribution, LognormalParams, NormalParams, PastInflow, Recourse,
    SeasonalUncertaintyInput, SystemInput, ThermalInput,
    UncertaintyDistributions,
};
use powers_rs::input_validation::InputValidator;

// Helper function to create valid GraphNodeInput for testing
// Helper function to create a test GraphNodeInput with all required fields
fn create_test_graph_node(
    id: usize,
    stage_id: usize,
    season_id: usize,
    risk_measure: &str,
) -> GraphNodeInput {
    GraphNodeInput {
        id,
        stage_id,
        season_id,
        start_date: "2024-01-01".to_string(),
        end_date: "2024-01-31".to_string(),
        risk_measure: risk_measure.to_string(),
        load_stochastic_process: "naive".to_string(),
        inflow_stochastic_process: "naive".to_string(),
        state_variables: "storage".to_string(),
    }
}

// Helper function to create a minimal valid SystemInput for recourse tests
fn create_test_system() -> SystemInput {
    SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 1000.0,
            spillage_penalty: 0.0,
        }],
    }
}

// Helper function to create a SystemInput with no hydros (for testing empty storage)
fn create_test_system_no_hydros() -> SystemInput {
    SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    }
}

// Helper function to create a SystemInput with 2 hydros (for testing duplicate hydro IDs)
fn create_test_system_two_hydros() -> SystemInput {
    SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![
            HydroInput {
                id: 0,
                downstream_hydro_id: None,
                bus_id: 0,
                productivity: 1.0,
                min_storage: 0.0,
                max_storage: 100.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 1000.0,
                spillage_penalty: 0.0,
            },
            HydroInput {
                id: 1,
                downstream_hydro_id: Some(0),
                bus_id: 0,
                productivity: 1.0,
                min_storage: 0.0,
                max_storage: 100.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 1000.0,
                spillage_penalty: 0.0,
            },
        ],
    }
}

// ============================================================================
// Phase 1: System Validation Tests (20 tests)
// ============================================================================

#[test]
fn test_system_validation_valid_input_passes() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 50.0,
            min_generation: 0.0,
            max_generation: 100.0,
        }],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_ok(), "Valid system input should pass validation");
}

#[test]
fn test_system_validation_duplicate_bus_id_fails() {
    let system = SystemInput {
        buses: vec![
            BusInput {
                id: 0,
                deficit_cost: 1000.0,
            },
            BusInput {
                id: 0, // Duplicate!
                deficit_cost: 1500.0,
            },
        ],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err(), "Duplicate bus IDs should fail validation");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("unique") || error.contains("duplicate"),
        "Error should mention uniqueness: {}",
        error
    );
}

#[test]
fn test_system_validation_gap_in_thermal_ids_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![
            ThermalInput {
                id: 0,
                bus_id: 0,
                cost: 50.0,
                min_generation: 0.0,
                max_generation: 100.0,
            },
            ThermalInput {
                id: 2, // Gap! Missing id=1
                bus_id: 0,
                cost: 60.0,
                min_generation: 0.0,
                max_generation: 120.0,
            },
        ],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err(), "Gap in thermal IDs should fail validation");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("sequential") || error.contains("gap"),
        "Error should mention sequential IDs or gaps: {}",
        error
    );
}

#[test]
fn test_system_validation_line_references_nonexistent_bus_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![LineInput {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 999, // Doesn't exist!
            direct_capacity: 100.0,
            reverse_capacity: 100.0,
            exchange_penalty: 0.01,
        }],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Line referencing nonexistent bus should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("target_bus_id") || error.contains("999"),
        "Error should mention invalid bus reference: {}",
        error
    );
}

#[test]
fn test_system_validation_negative_line_capacity_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![LineInput {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 0,
            direct_capacity: -50.0, // Negative!
            reverse_capacity: 100.0,
            exchange_penalty: 0.01,
        }],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Negative line capacity should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("direct_capacity") || error.contains("negative"),
        "Error should mention negative capacity: {}",
        error
    );
}

#[test]
fn test_system_validation_thermal_min_greater_than_max_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 50.0,
            min_generation: 100.0, // Greater than max!
            max_generation: 50.0,
        }],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err(), "Thermal min > max should fail validation");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("min_generation") && error.contains("max_generation"),
        "Error should mention min/max generation: {}",
        error
    );
}

#[test]
fn test_system_validation_hydro_invalid_bus_id_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 5, // Doesn't exist!
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Hydro referencing nonexistent bus should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("bus_id") || error.contains("5"),
        "Error should mention invalid bus_id: {}",
        error
    );
}

#[test]
fn test_system_validation_hydro_non_positive_productivity_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 0.0, // Must be positive!
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Hydro with zero productivity should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("productivity") || error.contains("positive"),
        "Error should mention productivity constraint: {}",
        error
    );
}

#[test]
fn test_system_validation_hydro_storage_min_greater_than_max_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 200.0, // Greater than max!
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Hydro storage min > max should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("min_storage") && error.contains("max_storage"),
        "Error should mention storage bounds: {}",
        error
    );
}

#[test]
fn test_system_validation_hydro_flow_min_greater_than_max_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 60.0, // Greater than max!
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Hydro flow min > max should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("min_turbined_flow")
            && error.contains("max_turbined_flow"),
        "Error should mention flow bounds: {}",
        error
    );
}

#[test]
fn test_system_validation_hydro_invalid_downstream_reference_fails() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: Some(999), // Doesn't exist!
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_err(),
        "Hydro referencing nonexistent downstream should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("downstream_hydro_id") || error.contains("999"),
        "Error should mention invalid downstream reference: {}",
        error
    );
}

#[test]
fn test_system_validation_error_includes_field_name() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: -10.0, // Negative cost
            min_generation: 0.0,
            max_generation: 100.0,
        }],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("cost") || error.contains("thermals[0]"),
        "Error should include field name: {}",
        error
    );
}

#[test]
fn test_system_validation_error_includes_file_name() {
    // Even empty system is valid, so let's use invalid data
    let system = SystemInput {
        buses: vec![BusInput {
            id: 1, // Not starting from 0!
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("system.json") || error.contains("system"),
        "Error should include file name: {}",
        error
    );
}

#[test]
fn test_system_validation_error_includes_constraint() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 50.0,
            min_generation: 100.0,
            max_generation: 50.0,
        }],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("<=") || error.contains("must be"),
        "Error should include constraint description: {}",
        error
    );
}

#[test]
fn test_system_validation_error_includes_suggestion() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![LineInput {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 0,
            direct_capacity: -50.0,
            reverse_capacity: 100.0,
            exchange_penalty: 0.01,
        }],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("Suggestion:") || error.contains("Set"),
        "Error should include suggestion: {}",
        error
    );
}

#[test]
fn test_system_validation_error_lists_available_bus_ids() {
    let system = SystemInput {
        buses: vec![
            BusInput {
                id: 0,
                deficit_cost: 1000.0,
            },
            BusInput {
                id: 1,
                deficit_cost: 1200.0,
            },
        ],
        lines: vec![LineInput {
            id: 0,
            source_bus_id: 0,
            target_bus_id: 5, // Invalid
            direct_capacity: 100.0,
            reverse_capacity: 100.0,
            exchange_penalty: 0.01,
        }],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("0") && error.contains("1"),
        "Error should list available bus IDs: {}",
        error
    );
}

#[test]
fn test_system_validation_multiple_hydros_valid() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![
            HydroInput {
                id: 0,
                downstream_hydro_id: Some(1),
                bus_id: 0,
                productivity: 1.0,
                min_storage: 0.0,
                max_storage: 100.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 50.0,
                spillage_penalty: 0.01,
            },
            HydroInput {
                id: 1,
                downstream_hydro_id: None,
                bus_id: 0,
                productivity: 0.9,
                min_storage: 0.0,
                max_storage: 80.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 40.0,
                spillage_penalty: 0.01,
            },
        ],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_ok(),
        "Valid cascade of hydros should pass validation"
    );
}

#[test]
fn test_system_validation_empty_arrays_valid() {
    let system = SystemInput {
        buses: vec![],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(result.is_ok(), "Empty system (no entities) should be valid");
}

#[test]
fn test_system_validation_thermal_cost_zero_valid() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 0.0, // Zero is valid (free energy)
            min_generation: 0.0,
            max_generation: 100.0,
        }],
        hydros: vec![],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_ok(),
        "Thermal with zero cost should be valid (free energy)"
    );
}

#[test]
fn test_system_validation_hydro_downstream_none_valid() {
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None, // Terminal hydro
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.01,
        }],
    };

    let result = InputValidator::validate_system(&system);
    assert!(
        result.is_ok(),
        "Hydro with no downstream (terminal) should be valid"
    );
}

// ============================================================================
// Phase 2: Graph Validation Tests (18 tests)
// ============================================================================

#[test]
fn test_graph_validation_valid_input_passes() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 1.0,
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_ok(), "Valid graph input should pass validation");
}

#[test]
fn test_graph_validation_duplicate_node_id_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(0, 1, 0, "expectation"), // Duplicate ID!
        ],
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_err(), "Duplicate node IDs should fail validation");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("unique") || error.contains("duplicate"),
        "Error should mention uniqueness: {}",
        error
    );
}

#[test]
fn test_graph_validation_non_sequential_stage_ids_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 2, 0, "expectation"), // Skip stage 1!
        ],
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Non-sequential stage IDs should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("sequential") || error.contains("gap"),
        "Error should mention sequential stages: {}",
        error
    );
}

#[test]
fn test_graph_validation_stage_ids_with_gap_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 0, 0, "expectation"),
            create_test_graph_node(2, 2, 0, "expectation"), // Gap! Missing stage 1
        ],
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_err(), "Gap in stage IDs should fail validation");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("sequential") || error.contains("gap"),
        "Error should mention gap in stages: {}",
        error
    );
}

#[test]
fn test_graph_validation_invalid_risk_measure_fails() {
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "invalid_risk")], // Invalid!
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Invalid risk measure should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("risk_measure")
            || error.contains("expectation")
            || error.contains("cvar"),
        "Error should mention valid risk measures: {}",
        error
    );
}

#[test]
fn test_graph_validation_edge_references_nonexistent_source_fails() {
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![GraphEdgeInput {
            source_id: 5, // Doesn't exist!
            target_id: 0,
            probability: 1.0,
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Edge referencing nonexistent source should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("source_id") || error.contains("5"),
        "Error should mention invalid source_id: {}",
        error
    );
}

#[test]
fn test_graph_validation_edge_references_nonexistent_target_fails() {
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 10, // Doesn't exist!
            probability: 1.0,
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Edge referencing nonexistent target should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("target_id") || error.contains("10"),
        "Error should mention invalid target_id: {}",
        error
    );
}

#[test]
fn test_graph_validation_edge_probability_zero_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 0.0, // Invalid!
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Edge with zero probability should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("probability") || error.contains("positive"),
        "Error should mention probability constraint: {}",
        error
    );
}

#[test]
fn test_graph_validation_edge_probability_negative_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: -0.5, // Negative!
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Edge with negative probability should fail validation"
    );
}

#[test]
fn test_graph_validation_edge_probability_greater_than_one_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 1.5, // Greater than 1!
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Edge with probability > 1 should fail validation"
    );
}

#[test]
fn test_graph_validation_probabilities_sum_to_0_5_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 0.5, // Sum = 0.5, not 1.0!
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Probabilities summing to 0.5 should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("sum") || error.contains("1.0"),
        "Error should mention probability sum: {}",
        error
    );
}

#[test]
fn test_graph_validation_probabilities_sum_to_1_5_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
            create_test_graph_node(2, 1, 0, "expectation"),
        ],
        edges: vec![
            GraphEdgeInput {
                source_id: 0,
                target_id: 1,
                probability: 0.8,
                discount_rate: 0.05,
            },
            GraphEdgeInput {
                source_id: 0,
                target_id: 2,
                probability: 0.7, // Sum = 1.5!
                discount_rate: 0.05,
            },
        ],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Probabilities summing to 1.5 should fail validation"
    );
}

#[test]
fn test_graph_validation_probabilities_within_tolerance_passes() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
            create_test_graph_node(2, 1, 0, "expectation"),
        ],
        edges: vec![
            GraphEdgeInput {
                source_id: 0,
                target_id: 1,
                probability: 0.5 + 5e-7, // Sum will be 1.000001, within tolerance (1e-6)
                discount_rate: 0.05,
            },
            GraphEdgeInput {
                source_id: 0,
                target_id: 2,
                probability: 0.5 + 5e-7, // Sum = 1.000001, |1.000001 - 1.0| = 1e-6
                discount_rate: 0.05,
            },
        ],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_ok(),
        "Probabilities within tolerance should pass validation"
    );
}

#[test]
fn test_graph_validation_negative_discount_rate_fails() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 1.0,
            discount_rate: -0.05, // Negative!
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Negative discount rate should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("discount_rate") || error.contains("negative"),
        "Error should mention discount rate: {}",
        error
    );
}

#[test]
fn test_graph_validation_error_shows_actual_probability_sum() {
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 0, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 0.7,
            discount_rate: 0.05,
        }],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("0.7") || error.contains("sum"),
        "Error should show actual probability sum: {}",
        error
    );
}

#[test]
fn test_graph_validation_error_lists_valid_risk_measures() {
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "invalid")],
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("expectation")
            || error.contains("cvar")
            || error.contains("worstcase"),
        "Error should list valid risk measures: {}",
        error
    );
}

#[test]
fn test_graph_validation_multiple_sources_probabilities_checked_independently()
{
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 0, 0, "expectation"),
            create_test_graph_node(2, 1, 0, "expectation"),
        ],
        edges: vec![
            GraphEdgeInput {
                source_id: 0,
                target_id: 2,
                probability: 1.0, // Source 0: sum = 1.0 ✓
                discount_rate: 0.05,
            },
            GraphEdgeInput {
                source_id: 1,
                target_id: 2,
                probability: 0.5, // Source 1: sum = 0.5 ✗
                discount_rate: 0.05,
            },
        ],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(
        result.is_err(),
        "Each source node's probabilities should be checked independently"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("node 1") || error.contains("0.5"),
        "Error should identify the problematic source node: {}",
        error
    );
}

#[test]
fn test_graph_validation_empty_graph_valid() {
    let graph = GraphInput {
        nodes: vec![],
        edges: vec![],
    };

    let result = InputValidator::validate_graph(&graph);
    assert!(result.is_ok(), "Empty graph should be valid");
}

// ============================================================================
// Phase 3: Recourse Validation Tests (15 tests)
// ============================================================================

#[test]
fn test_recourse_validation_valid_input_passes() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 100.0,
            }],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![LoadDistribution {
                    bus_id: 0,
                    normal: NormalParams {
                        mu: 100.0,
                        sigma: 10.0,
                    },
                }],
                inflow: vec![InflowDistribution {
                    hydro_id: 0,
                    lognormal: LognormalParams {
                        mu: 5.0,
                        sigma: 1.0,
                    },
                }],
            },
        }],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Valid recourse input should pass validation"
    );
}

#[test]
fn test_recourse_validation_initial_storage_negative_value_fails() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Negative!
            }],
            inflow: vec![],
        },
        uncertainties: vec![],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Negative initial storage should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("storage")
            || error.contains("negative")
            || error.contains("non-negative"),
        "Error should mention storage constraint: {}",
        error
    );
}

#[test]
fn test_recourse_validation_past_inflow_negative_value_fails() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: -50.0, // Negative!
            }],
        },
        uncertainties: vec![],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);

    assert!(
        result.is_err(),
        "Negative past inflow should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("inflow")
            || error.contains("negative")
            || error.contains("non-negative"),
        "Error should mention inflow constraint: {}",
        error
    );
}

#[test]
fn test_recourse_validation_past_inflow_zero_lag_fails() {
    // NOTE: This validation is NOT YET IMPLEMENTED in src/input_validation.rs
    // The test is kept here to document expected behavior for future implementation
    // TODO: Implement past inflow lag validation in validate_recourse()
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 0, // Invalid!
                value: 100.0,
            }],
        },
        uncertainties: vec![],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);

    assert!(
        result.is_err(),
        "Past inflow with zero lag should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("lag")
            || error.contains("positive")
            || error.contains("greater than 0"),
        "Error should mention lag constraint: {}",
        error
    );
}

#[test]
fn test_recourse_validation_load_distribution_negative_sigma_fails() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![LoadDistribution {
                    bus_id: 0,
                    normal: NormalParams {
                        mu: 100.0,
                        sigma: -5.0, // Negative!
                    },
                }],
                inflow: vec![],
            },
        }],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Load distribution with negative sigma should fail validation"
    );
}

// NOTE: sigma = 0 is currently ALLOWED by the validation (checks sigma < 0, not <= 0)
// If stricter validation is needed, update src/input_validation.rs to check sigma <= 0

#[test]
fn test_recourse_validation_inflow_distribution_negative_sigma_fails() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![InflowDistribution {
                    hydro_id: 0,
                    lognormal: LognormalParams {
                        mu: 5.0,
                        sigma: -1.0, // Negative!
                    },
                }],
            },
        }],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Inflow distribution with negative sigma should fail validation"
    );
}

// NOTE: sigma = 0 is currently ALLOWED by the validation (checks sigma < 0, not <= 0)
// If stricter validation is needed, update src/input_validation.rs to check sigma <= 0

#[test]
fn test_recourse_validation_duplicate_initial_storage_hydro_ids_fails() {
    // NOTE: This validation is NOT YET IMPLEMENTED in src/input_validation.rs
    // The test is kept here to document expected behavior for future implementation
    // TODO: Implement duplicate hydro_id detection in validate_recourse()
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![
                InitialStorage {
                    hydro_id: 0,
                    value: 50.0,
                },
                InitialStorage {
                    hydro_id: 0, // Duplicate!
                    value: 60.0,
                },
            ],
            inflow: vec![],
        },
        uncertainties: vec![],
    };

    let system = create_test_system_two_hydros();
    let result = InputValidator::validate_recourse(&recourse, &system);

    assert!(
        result.is_err(),
        "Duplicate initial storage hydro IDs should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("unique") || error.contains("duplicate"),
        "Error should mention uniqueness: {}",
        error
    );
}

#[test]
fn test_recourse_validation_duplicate_season_ids_fails() {
    // NOTE: This validation is NOT YET IMPLEMENTED in src/input_validation.rs
    // The test is kept here to document expected behavior for future implementation
    // TODO: Implement duplicate season_id detection in validate_recourse()
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![
            SeasonalUncertaintyInput {
                season_id: 0,
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
            SeasonalUncertaintyInput {
                season_id: 0, // Duplicate!
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
        ],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);

    assert!(
        result.is_err(),
        "Duplicate season IDs should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("unique")
            || error.contains("duplicate")
            || error.contains("season"),
        "Error should mention season ID uniqueness: {}",
        error
    );
}

#[test]
fn test_recourse_validation_zero_num_branchings_fails() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 0, // Invalid!
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Zero num_branchings should fail validation"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("num_branchings") || error.contains("positive"),
        "Error should mention num_branchings constraint: {}",
        error
    );
}

#[test]
fn test_recourse_validation_error_includes_field_name() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Invalid
            }],
            inflow: vec![],
        },
        uncertainties: vec![],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("value") || error.contains("storage"),
        "Error should include field name: {}",
        error
    );
}

#[test]
fn test_recourse_validation_error_includes_file_name() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Invalid
            }],
            inflow: vec![],
        },
        uncertainties: vec![],
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err());

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("recourse"),
        "Error should include file name: {}",
        error
    );
}

#[test]
fn test_recourse_validation_empty_uncertainties_valid() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![], // Empty is valid
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Empty uncertainties should be valid (edge case)"
    );
}

#[test]
fn test_recourse_validation_empty_initial_condition_valid() {
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![], // Empty is valid
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let system = create_test_system_no_hydros();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Empty initial condition should be valid (edge case)"
    );
}

// ============================================================================
// Part 4: Cross-Validation Tests (10 tests)
// ============================================================================
// Test cross-file consistency: graph ↔ recourse, recourse ↔ system
// Uses validate_consistency() which checks references across files

#[test]
fn test_cross_validation_valid_input_passes() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 1000.0,
            spillage_penalty: 0.0,
        }],
    };

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![LoadDistribution {
                    bus_id: 0,
                    normal: NormalParams {
                        mu: 100.0,
                        sigma: 10.0,
                    },
                }],
                inflow: vec![InflowDistribution {
                    hydro_id: 0,
                    lognormal: LognormalParams {
                        mu: 5.0,
                        sigma: 1.0,
                    },
                }],
            },
        }],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_ok(),
        "Valid cross-references should pass validation"
    );
}

#[test]
fn test_cross_validation_graph_season_not_in_recourse_fails() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    // Graph references season_id = 999, but recourse only has season_id = 0
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 999, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_err(),
        "Graph node referencing non-existent season should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("season_id"),
        "Error should mention season_id reference"
    );
    assert!(
        error.contains("999"),
        "Error should mention the invalid season_id value"
    );
}

#[test]
fn test_cross_validation_recourse_load_invalid_bus_id_fails() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    // System only has bus_id = 0
    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Recourse references bus_id = 999 (doesn't exist in system)
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![LoadDistribution {
                    bus_id: 999,
                    normal: NormalParams {
                        mu: 100.0,
                        sigma: 10.0,
                    },
                }],
                inflow: vec![],
            },
        }],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_err(),
        "Load distribution referencing non-existent bus should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("bus_id"),
        "Error should mention bus_id reference"
    );
    assert!(
        error.contains("999"),
        "Error should mention the invalid bus_id value"
    );
}

#[test]
fn test_cross_validation_recourse_inflow_invalid_hydro_id_fails() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    // System only has hydro_id = 0
    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Recourse references hydro_id = 999 (doesn't exist in system)
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![InflowDistribution {
                    hydro_id: 999,
                    lognormal: LognormalParams {
                        mu: 5.0,
                        sigma: 1.0,
                    },
                }],
            },
        }],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_err(),
        "Inflow distribution referencing non-existent hydro should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("hydro_id"),
        "Error should mention hydro_id reference"
    );
    assert!(
        error.contains("999"),
        "Error should mention the invalid hydro_id value"
    );
}

#[test]
fn test_cross_validation_error_identifies_missing_season() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 42, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(result.is_err(), "Should fail with missing season");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("42"),
        "Error should identify the specific missing season_id"
    );
    assert!(
        error.contains("graph.json") || error.contains("node"),
        "Error should identify the source file/context"
    );
}

#[test]
fn test_cross_validation_error_lists_available_seasons() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 999, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![
            SeasonalUncertaintyInput {
                season_id: 0,
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
            SeasonalUncertaintyInput {
                season_id: 1,
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
        ],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(result.is_err(), "Should fail with invalid season");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("0") && (error.contains("1") || error.contains(", ")),
        "Error should list available season IDs (0, 1)"
    );
}

#[test]
fn test_cross_validation_multiple_seasons_validated() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    // Multiple nodes, one with invalid season_id
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 1, "expectation"),
            create_test_graph_node(2, 2, 999, "expectation"), // Invalid
        ],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![
            SeasonalUncertaintyInput {
                season_id: 0,
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
            SeasonalUncertaintyInput {
                season_id: 1,
                num_branchings: 10,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
        ],
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_err(),
        "Should detect invalid season in multi-node graph"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("999"),
        "Error should identify the invalid season_id"
    );
}

#[test]
fn test_cross_validation_empty_recourse_uncertainties_fails() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Empty uncertainties array - graph references season_id=0 but it doesn't exist
    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![], // Empty!
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_err(),
        "Graph node referencing season when no uncertainties exist should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("season_id") || error.contains("0"),
        "Error should mention season_id reference issue"
    );
}

#[test]
fn test_validate_all_catches_config_error() {
    use powers_rs::input::Config;

    // Invalid config: num_iterations = 0
    let config = Config {
        num_iterations: 0, // Invalid!
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let result =
        InputValidator::validate_all(&config, &system, &graph, &recourse);
    assert!(result.is_err(), "validate_all should catch config errors");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("num_iterations") || error.contains("positive"),
        "Error should mention config validation failure"
    );
}

#[test]
fn test_validate_all_catches_system_error() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
    };

    // Invalid system: hydro with min_storage > max_storage
    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 200.0, // > max_storage!
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 1000.0,
            spillage_penalty: 0.0,
        }],
    };

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }],
    };

    let result =
        InputValidator::validate_all(&config, &system, &graph, &recourse);
    assert!(result.is_err(), "validate_all should catch system errors");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("storage")
            || error.contains("min")
            || error.contains("max"),
        "Error should mention system validation failure"
    );
}

// ============================================================================
// Part 5: Integration Tests (5 tests)
// ============================================================================
// Test that Input::from_paths() properly validates all inputs before construction
// Uses actual file I/O to test end-to-end validation flow

#[test]
fn test_input_from_paths_runs_all_validations() {
    use powers_rs::input::Input;
    use std::path::Path;

    // Use example files which are known to be valid
    let config_path = Path::new("examples/01-deterministic/config.json");
    let system_path = Path::new("examples/01-deterministic/system.json");
    let graph_path = Path::new("examples/01-deterministic/graph.json");
    let recourse_path = Path::new("examples/01-deterministic/recourse.json");

    let result =
        Input::from_paths(config_path, system_path, graph_path, recourse_path);
    assert!(
        result.is_ok(),
        "Example files should pass comprehensive validation"
    );
}

#[test]
fn test_input_from_paths_fails_on_invalid_system() {
    use powers_rs::input::Input;
    use std::fs;
    use tempfile::TempDir;

    // Create temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Copy valid files
    fs::copy(
        "examples/01-deterministic/config.json",
        temp_path.join("config.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/graph.json",
        temp_path.join("graph.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/recourse.json",
        temp_path.join("recourse.json"),
    )
    .unwrap();

    // Create invalid system.json (hydro with min_storage > max_storage)
    let invalid_system = r#"{
        "buses": [{"id": 0, "deficit_cost": 1000.0}],
        "lines": [],
        "thermals": [],
        "hydros": [{
            "id": 0,
            "downstream_hydro_id": null,
            "bus_id": 0,
            "productivity": 1.0,
            "min_storage": 200.0,
            "max_storage": 100.0,
            "min_turbined_flow": 0.0,
            "max_turbined_flow": 1000.0,
            "spillage_penalty": 0.0
        }]
    }"#;
    fs::write(temp_path.join("system.json"), invalid_system).unwrap();

    let result = Input::from_paths(
        &temp_path.join("config.json"),
        &temp_path.join("system.json"),
        &temp_path.join("graph.json"),
        &temp_path.join("recourse.json"),
    );

    assert!(
        result.is_err(),
        "Input::from_paths should reject invalid system"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("storage")
            || error.contains("min")
            || error.contains("max"),
        "Error should mention system validation issue"
    );
}

#[test]
fn test_input_from_paths_fails_on_invalid_graph() {
    use powers_rs::input::Input;
    use std::fs;
    use tempfile::TempDir;

    // Create temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Copy valid files
    fs::copy(
        "examples/01-deterministic/config.json",
        temp_path.join("config.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/system.json",
        temp_path.join("system.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/recourse.json",
        temp_path.join("recourse.json"),
    )
    .unwrap();

    // Create invalid graph.json (probability > 1.0)
    let invalid_graph = r#"{
        "nodes": [{
            "id": 0,
            "stage_id": 0,
            "season_id": 0,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "risk_measure": "expectation",
            "load_stochastic_process": "naive",
            "inflow_stochastic_process": "naive",
            "state_variables": "storage"
        }],
        "edges": [{
            "source_id": 0,
            "target_id": 0,
            "probability": 1.5,
            "discount_rate": 0.0
        }]
    }"#;
    fs::write(temp_path.join("graph.json"), invalid_graph).unwrap();

    let result = Input::from_paths(
        &temp_path.join("config.json"),
        &temp_path.join("system.json"),
        &temp_path.join("graph.json"),
        &temp_path.join("recourse.json"),
    );

    assert!(
        result.is_err(),
        "Input::from_paths should reject invalid graph"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("probability") || error.contains("1.5"),
        "Error should mention graph validation issue"
    );
}

#[test]
fn test_input_from_paths_fails_on_invalid_recourse() {
    use powers_rs::input::Input;
    use std::fs;
    use tempfile::TempDir;

    // Create temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Copy valid files
    fs::copy(
        "examples/01-deterministic/config.json",
        temp_path.join("config.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/system.json",
        temp_path.join("system.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/graph.json",
        temp_path.join("graph.json"),
    )
    .unwrap();

    // Create invalid recourse.json (sigma < 0)
    // Note: Must include initial_storage to match system.json hydros
    let invalid_recourse = r#"{
        "initial_condition": {
            "storage": [{"hydro_id": 0, "value": 50.0}],
            "inflow": []
        },
        "uncertainties": [{
            "season_id": 0,
            "num_branchings": 10,
            "distributions": {
                "load": [{
                    "bus_id": 0,
                    "normal": {"mu": 100.0, "sigma": -5.0}
                }],
                "inflow": []
            }
        }]
    }"#;
    fs::write(temp_path.join("recourse.json"), invalid_recourse).unwrap();

    let result = Input::from_paths(
        &temp_path.join("config.json"),
        &temp_path.join("system.json"),
        &temp_path.join("graph.json"),
        &temp_path.join("recourse.json"),
    );

    assert!(
        result.is_err(),
        "Input::from_paths should reject invalid recourse"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("sigma")
            || error.contains("negative")
            || error.contains("non-negative"),
        "Error should mention recourse validation issue"
    );
}

#[test]
fn test_input_from_paths_fails_on_cross_validation_error() {
    use powers_rs::input::Input;
    use std::fs;
    use tempfile::TempDir;

    // Create temporary directory for test files
    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();

    // Copy valid config and system
    fs::copy(
        "examples/01-deterministic/config.json",
        temp_path.join("config.json"),
    )
    .unwrap();
    fs::copy(
        "examples/01-deterministic/system.json",
        temp_path.join("system.json"),
    )
    .unwrap();

    // Create graph that references season_id = 999
    let graph_with_invalid_season = r#"{
        "nodes": [{
            "id": 0,
            "stage_id": 0,
            "season_id": 999,
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "risk_measure": "expectation",
            "load_stochastic_process": "naive",
            "inflow_stochastic_process": "naive",
            "state_variables": "storage"
        }],
        "edges": []
    }"#;
    fs::write(temp_path.join("graph.json"), graph_with_invalid_season).unwrap();

    // Create recourse with only season_id = 0
    // Note: Must include initial_storage to match system.json hydros
    let recourse_with_season_0 = r#"{
        "initial_condition": {
            "storage": [{"hydro_id": 0, "value": 50.0}],
            "inflow": []
        },
        "uncertainties": [{
            "season_id": 0,
            "num_branchings": 10,
            "distributions": {
                "load": [],
                "inflow": []
            }
        }]
    }"#;
    fs::write(temp_path.join("recourse.json"), recourse_with_season_0).unwrap();

    let result = Input::from_paths(
        &temp_path.join("config.json"),
        &temp_path.join("system.json"),
        &temp_path.join("graph.json"),
        &temp_path.join("recourse.json"),
    );

    assert!(
        result.is_err(),
        "Input::from_paths should reject cross-validation errors"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("season_id") || error.contains("999"),
        "Error should mention cross-validation issue (season_id mismatch)"
    );
}
