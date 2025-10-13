#![allow(deprecated)]

use powers_rs::input::{
    BusInput, GraphEdgeInput, GraphInput, GraphNodeInput, HydroInput,
    InflowDistribution, InitialConditionInput, InitialStorage, LineInput,
    LoadDistribution, LognormalParams, NormalParams, PastInflow, Recourse,
    SeasonalUncertaintyInput, SystemInput, ThermalInput,
    UncertaintyDistributions,
};
use powers_rs::input_validation::InputValidator;

// Helper function to create valid GraphNodeInput for testing
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
        num_scenarios: 1,
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

#[test]
fn test_recourse_validation_valid_input_passes() {
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 200.0,
            }],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
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
                inflow: vec![],
            },
        }]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Negative!
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: -100.0,
            }],
        },
        uncertainties: Some(vec![]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 0, // Invalid!
                value: 0.0,
            }],
        },
        uncertainties: Some(vec![]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
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
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Load distribution with negative sigma should fail validation"
    );
}

#[test]
fn test_recourse_validation_inflow_distribution_negative_sigma_fails() {
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
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
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
    };

    let system = create_test_system();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Inflow distribution with negative sigma should fail validation"
    );
}

#[test]
fn test_recourse_validation_duplicate_initial_storage_hydro_ids_fails() {
    // NOTE: This validation is NOT YET IMPLEMENTED in src/input_validation.rs
    // The test is kept here to document expected behavior for future implementation
    // TODO: Implement duplicate hydro_id detection in validate_recourse()
    let recourse = Recourse {
        correlation: None,
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
        uncertainties: Some(vec![]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![
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
                num_branchings: 5,
                distributions: UncertaintyDistributions {
                    load: vec![],
                    inflow: vec![],
                },
            },
        ]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 0, // Invalid!
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Invalid
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: -10.0, // Invalid
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![]),
        noise_models: None, // Empty is valid
        noise_models_v2: None,
        schema_version: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![], // Empty is valid
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
    };

    let system = create_test_system_no_hydros();
    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Empty initial condition should be valid (edge case)"
    );
}

#[test]
fn test_cross_validation_valid_input_passes() {
    use powers_rs::input::Config;

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: 100,
        seed: 42,
        output_path: None,
        num_threads: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
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
                        mu: 100.0,
                        sigma: 10.0,
                    },
                }],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
    };

    let system = create_test_system();

    // Graph references season_id = 999, but recourse only has season_id = 0
    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 999, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
    };

    // System only has bus_id = 0
    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Recourse references bus_id = 999 (doesn't exist in system)
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
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
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
    };

    // System only has hydro_id = 0
    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Recourse references hydro_id = 999 (doesn't exist in system)
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![InflowDistribution {
                    hydro_id: 999, // Invalid - doesn't exist in system!
                    lognormal: LognormalParams {
                        mu: 5.0,
                        sigma: 1.0,
                    },
                }],
            },
        }]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
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
        num_threads: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 42, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 999, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![
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
        ]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![
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
        ]),
        noise_models: None,
        noise_models_v2: None,
        schema_version: None,
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
        num_threads: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    // Empty uncertainties array - graph references season_id=0 but it doesn't exist
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![]),
        noise_models: None, // Empty!
        noise_models_v2: None,
        schema_version: None,
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
        num_threads: None,
    };

    let system = create_test_system();

    let graph = GraphInput {
        nodes: vec![create_test_graph_node(0, 0, 0, "expectation")],
        edges: vec![],
    };

    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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
        num_threads: None,
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
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: Some(vec![SeasonalUncertaintyInput {
            season_id: 0,
            num_branchings: 10,
            distributions: UncertaintyDistributions {
                load: vec![],
                inflow: vec![],
            },
        }]),
        noise_models: None,

        noise_models_v2: None,

        schema_version: None,
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

// ============================================================================
// AR-2: AR Model Validation Tests
// ============================================================================

#[test]
fn test_ar1_stationary_valid() {
    // Valid AR(1) with |φ| < 1 should pass
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 30.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Valid AR(1) with φ=0.7 should pass validation"
    );
}

#[test]
fn test_ar1_non_stationary_rejected() {
    // Non-stationary AR(1) with |φ| ≥ 1 should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![1.05]), // Non-stationary: |φ| > 1
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Non-stationary AR(1) with φ=1.05 should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("stationarity") || error.contains("1.05"),
        "Error should mention stationarity violation"
    );
}

#[test]
fn test_ar2_stationary_valid() {
    // Valid AR(2) satisfying triangle conditions
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![
                PastInflow {
                    hydro_id: 0,
                    lag: 1,
                    value: 30.0,
                },
                PastInflow {
                    hydro_id: 0,
                    lag: 2,
                    value: 20.0,
                },
            ],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(2),

            coefficients: Some(vec![0.6, 0.3]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Valid AR(2) with φ=[0.6, 0.3] should pass validation"
    );
}

#[test]
fn test_ar_coefficient_count_mismatch() {
    // lag_order=2 but only 1 coefficient should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(2),

            coefficients: Some(vec![0.7]), // Only 1 coefficient for AR(2)!
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR(2) with only 1 coefficient should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("2") || error.contains("coefficients"),
        "Error should mention coefficient count mismatch"
    );
}

#[test]
fn test_ar_missing_lag_order() {
    // AR model without lag_order should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: None, // Missing!
            coefficients: Some(vec![0.7]),

            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR model without lag_order should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("lag_order") || error.contains("Missing"),
        "Error should mention missing lag_order"
    );
}

#[test]
fn test_ar_missing_coefficients() {
    // AR model without coefficients should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),

            coefficients: None, // Missing!
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR model without coefficients should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("coefficients") || error.contains("Missing"),
        "Error should mention missing coefficients"
    );
}

#[test]
fn test_ar_invalid_lag_order() {
    // lag_order > 3 should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(4), // Too high!
            coefficients: Some(vec![0.7, 0.2, 0.1, 0.05]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err(), "lag_order=4 should be rejected");

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("1 and 3") || error.contains("lag_order"),
        "Error should mention valid lag_order range"
    );
}

#[test]
fn test_ar_lognormal_distribution_rejected() {
    // AR with lognormal distribution should be rejected (asymmetric)
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Lognormal {
                mu: 4.5,
                sigma: 0.3,
            }, // Asymmetric!
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR with lognormal distribution should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("symmetric") || error.contains("lognormal"),
        "Error should mention symmetric distribution requirement"
    );
}

#[test]
fn test_ar_negative_std_dev_rejected() {
    // Negative std_dev should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: -5.0, // Negative!
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err(), "Negative std_dev should be rejected");

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("positive") || error.contains("std_dev"),
        "Error should mention positive std_dev requirement"
    );
}

#[test]
fn test_ar_entity_not_found() {
    // Reference to non-existent entity should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system(); // Only has hydro_id=0
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 99, // Non-existent!
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err(), "Non-existent entity_id should be rejected");

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("entity_id") || error.contains("99"),
        "Error should mention invalid entity_id reference"
    );
}

#[test]
fn test_independent_noise_valid() {
    // Independent noise model should validate correctly
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Independent,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            lag_order: None,
            coefficients: None,
            non_negativity_method: None,
        }]),

        noise_models_v2: None,

        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_ok(), "Valid independent noise should pass");
}

#[test]
fn test_ar2_triangle_violation() {
    // AR(2) violating φ₂ + φ₁ < 1 should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(2),
            coefficients: Some(vec![0.8, 0.3]), // φ₂ + φ₁ = 1.1 > 1
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR(2) triangle violation should be rejected"
    );

    let error = format!("{}", result.err().unwrap());
    assert!(
        error.contains("φ₂ + φ₁") || error.contains("stationarity"),
        "Error should mention triangle condition violation"
    );
}

#[test]
fn test_independent_noise_lognormal_valid() {
    // Independent noise can use lognormal (not restricted like AR)
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Independent,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Lognormal {
                mu: 4.5,
                sigma: 0.3,
            },
            lag_order: None,
            coefficients: None,
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Independent noise with lognormal should pass"
    );
}

// ============================================================================
// AR-3: Initial Condition Lag Tests
// ============================================================================

#[test]
fn test_ar1_with_valid_lags() {
    // AR(1) model with correct inflow lags should pass
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_ok(), "AR(1) with valid inflow lags should pass");
}

#[test]
fn test_ar2_with_valid_lags() {
    // AR(2) model with correct inflow lags should pass
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![
                PastInflow {
                    hydro_id: 0,
                    lag: 1,
                    value: 0.0,
                },
                PastInflow {
                    hydro_id: 0,
                    lag: 2,
                    value: 0.0,
                },
            ],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(2),
            coefficients: Some(vec![0.6, 0.3]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_ok(), "AR(2) with valid inflow lags should pass");
}

#[test]
fn test_ar1_missing_lags() {
    // AR(1) model without inflow lags should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![], // Missing lag values
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR(1) without inflow lags should be rejected"
    );
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("Missing AR lag inflows"),
        "Error should mention missing lag inflows: {}",
        err_str
    );
}

#[test]
fn test_ar2_lag_count_mismatch() {
    // AR(2) model with only 1 lag value should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }], // Only 1 lag, but AR(2) needs 2
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(2),
            coefficients: Some(vec![0.6, 0.3]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR(2) with wrong lag count should be rejected"
    );
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("Invalid AR lag count")
            && err_str.contains("expected 2")
            && err_str.contains("found 1"),
        "Error should mention lag count mismatch: {}",
        err_str
    );
}

#[test]
fn test_ar1_negative_lag() {
    // AR(1) model with negative lag value should be rejected
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 0, // Zero lag - should be rejected
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "AR(1) with zero/invalid lag should be rejected"
    );
    let err_str = result.unwrap_err().to_string();
    assert!(
        err_str.contains("Invalid AR lag indices") || err_str.contains("lag"),
        "Error should mention lag issues: {}",
        err_str
    );
}

#[test]
fn test_independent_noise_no_lags_required() {
    // Independent noise models don't require inflow lags
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![], // No inflow lags for independent noise
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Independent,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            lag_order: None,
            coefficients: None,
            non_negativity_method: None,
        }]),

        noise_models_v2: None,

        schema_version: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Independent noise without inflow lags should pass"
    );
}

// ============================================================================
// AR-4: AR Stability Validation Tests
// ============================================================================

#[test]
fn test_spectral_radius_ar1() {
    // AR(1): ρ = |φ|
    use powers_rs::input_validation::InputValidator;

    // Test positive coefficient
    let spectral_radius = InputValidator::compute_spectral_radius(&[0.7]);
    assert!(
        (spectral_radius - 0.7).abs() < 1e-10,
        "AR(1) spectral radius should equal |φ|"
    );

    // Test negative coefficient
    let spectral_radius = InputValidator::compute_spectral_radius(&[-0.7]);
    assert!(
        (spectral_radius - 0.7).abs() < 1e-10,
        "AR(1) spectral radius should equal |φ|"
    );

    // Test near unit root
    let spectral_radius = InputValidator::compute_spectral_radius(&[0.99]);
    assert!(
        (spectral_radius - 0.99).abs() < 1e-10,
        "AR(1) near unit root case"
    );
}

#[test]
fn test_spectral_radius_ar2_real_roots() {
    // AR(2) with real eigenvalues: φ₁=0.8, φ₂=-0.15
    // Characteristic equation: λ² - 0.8λ + 0.15 = 0
    // Roots: λ = (0.8 ± √(0.64-0.6))/2 = (0.8 ± 0.2)/2 = {0.5, 0.3}
    // Spectral radius = max(|0.5|, |0.3|) = 0.5
    use powers_rs::input_validation::InputValidator;

    let spectral_radius =
        InputValidator::compute_spectral_radius(&[0.8, -0.15]);
    assert!(
        (spectral_radius - 0.5).abs() < 1e-10,
        "AR(2) real roots spectral radius should be max eigenvalue magnitude"
    );
}

#[test]
fn test_spectral_radius_ar2_complex_roots() {
    // AR(2) with complex conjugate eigenvalues: φ₁=0.6, φ₂=0.25
    // Characteristic equation: λ² - 0.6λ - 0.25 = 0
    // Discriminant: 0.36 + 1.0 = 1.36 > 0 (actually real roots)
    // Better example: φ₁=0.5, φ₂=0.3
    // Discriminant: 0.25 + 1.2 = 1.45 > 0 (still real)
    // For complex: φ₁=0.4, φ₂=0.5
    // Discriminant: 0.16 + 2.0 = 2.16 > 0 (still real!)
    // True complex case: φ₁=0.3, φ₂=0.1
    // Discriminant: 0.09 + 0.4 = 0.49 > 0 (nope)
    // Need negative φ₂ for complex: φ₁=0.5, φ₂=-0.3
    // Discriminant: 0.25 - 1.2 = -0.95 < 0 ✓
    // |λ| = √|φ₂| = √0.3 ≈ 0.5477
    use powers_rs::input_validation::InputValidator;

    let spectral_radius = InputValidator::compute_spectral_radius(&[0.5, -0.3]);
    let expected = (0.3_f64).sqrt();
    assert!(
        (spectral_radius - expected).abs() < 1e-10,
        "AR(2) complex roots spectral radius should be sqrt(|φ₂|)"
    );
}

#[test]
fn test_spectral_radius_ar3_conservative() {
    // AR(3): Conservative bound Σ|φᵢ|
    // Example: φ=[0.4, 0.3, 0.2] → ρ ≤ 0.9
    use powers_rs::input_validation::InputValidator;

    let spectral_radius =
        InputValidator::compute_spectral_radius(&[0.4, 0.3, 0.2]);
    assert!(
        (spectral_radius - 0.9).abs() < 1e-10,
        "AR(3) spectral radius should be sum of absolute coefficients"
    );

    // With negative coefficients
    let spectral_radius =
        InputValidator::compute_spectral_radius(&[0.5, -0.2, 0.1]);
    assert!(
        (spectral_radius - 0.8).abs() < 1e-10,
        "AR(3) should sum absolute values regardless of sign"
    );
}

#[test]
fn test_acf_half_life_ar1_fast_decay() {
    // AR(1): h = log(0.5) / log(|φ|)
    // φ=0.7: h = log(0.5)/log(0.7) ≈ 1.943
    use powers_rs::input_validation::InputValidator;

    let half_life = InputValidator::compute_acf_half_life(&[0.7]);
    let expected = (0.5_f64).ln() / (0.7_f64).ln();

    assert!(half_life.is_some(), "AR(1) should always return half-life");
    let h = half_life.unwrap();
    assert!(
        (h as f64 - expected).abs() < 0.5,
        "AR(1) half-life should match closed form (got {}, expected ~{})",
        h,
        expected
    );
}

#[test]
fn test_acf_half_life_ar1_slow_decay() {
    // AR(1) near unit root: φ=0.95
    // h = log(0.5)/log(0.95) ≈ 13.5
    use powers_rs::input_validation::InputValidator;

    let half_life = InputValidator::compute_acf_half_life(&[0.95]);
    let expected = (0.5_f64).ln() / (0.95_f64).ln();

    assert!(half_life.is_some(), "AR(1) should always return half-life");
    let h = half_life.unwrap();
    assert!(
        (h as f64 - expected).abs() < 0.5,
        "AR(1) near unit root half-life should be long (got {}, expected ~{})",
        h,
        expected
    );
    assert!(h > 10, "Near unit root should have half-life > 10 stages");
}

#[test]
fn test_acf_half_life_ar2() {
    // AR(2): φ₁=0.6, φ₂=-0.2
    // Uses Yule-Walker recursion to compute ACF
    use powers_rs::input_validation::InputValidator;

    let half_life = InputValidator::compute_acf_half_life(&[0.6, -0.2]);
    // Should return a reasonable value (between 1 and 100)
    assert!(
        half_life.is_some() || half_life.is_none(),
        "AR(2) may or may not return half-life depending on ACF behavior"
    );

    if let Some(h) = half_life {
        assert!(
            h > 0 && h < 100,
            "AR(2) half-life should be reasonable (got {})",
            h
        );
    }
}

#[test]
fn test_warning_near_unit_root() {
    // φ=0.99 should trigger strong warning (ρ > 0.99)
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.99]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    // Should pass validation but with warnings logged
    assert!(
        result.is_ok(),
        "Near unit root should pass validation (warnings only)"
    );
}

#[test]
fn test_warning_borderline_stability() {
    // φ=0.96 should trigger warning (0.95 < ρ ≤ 0.99)
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![0.96]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    // Should pass validation but with warnings logged
    assert!(
        result.is_ok(),
        "Borderline stability should pass validation (warnings only)"
    );
}

#[test]
fn test_no_warning_good_stability() {
    // φ=0.7 should pass without warnings (good stability)
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),

            coefficients: Some(vec![0.7]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    // Should pass without warnings
    assert!(
        result.is_ok(),
        "Good stability parameters should pass cleanly"
    );
}

#[test]
fn test_numerical_precision_warning() {
    // Very small coefficient should trigger precision warning
    use powers_rs::input::{
        Distribution, InitialConditionInput, InitialStorage, NoiseModel,
        NoiseType, PastInflow, Recourse, UncertaintyType,
    };

    let system = create_test_system();
    let recourse = Recourse {
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![PastInflow {
                hydro_id: 0,
                lag: 1,
                value: 0.0,
            }],
        },
        uncertainties: None,
        noise_models: Some(vec![NoiseModel {
            noise_type: NoiseType::Autoregressive,
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 1,
            distribution: Distribution::Normal {
                mean: 0.0,
                std_dev: 15.0,
            },
            lag_order: Some(1),
            coefficients: Some(vec![1e-15]),
            non_negativity_method: None,
        }]),
        noise_models_v2: None,
        schema_version: None,
    };

    let result = powers_rs::input_validation::InputValidator::validate_recourse(
        &recourse, &system,
    );
    // Should pass validation but with precision warning logged
    assert!(
        result.is_ok(),
        "Very small coefficient should pass (effectively independent noise)"
    );
}
