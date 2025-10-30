use powers_rs::input::{
    BusInput, GraphEdgeInput, GraphInput, GraphNodeInput, HydroInput,
    LineInput, SystemInput, ThermalInput,
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

// ============================================================================
// Tests for CLEANUP-003: Cross-file consistency validation
// ============================================================================

#[test]
fn test_recourse_validation_valid_entity_references_pass() {
    use powers_rs::input::{
        InitialConditionInput, InitialStorage, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
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
            max_turbined_flow: 50.0,
            spillage_penalty: 1.0,
        }],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 0,
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainty_specifications: vec![
            UncertaintySpecification {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0, // Valid: matches hydro_id 0
                temporal_model: TemporalModelInput::PeriodicAr {
                    num_seasons: 1,
                    ar_orders: vec![1],
                    ar_coefficients: vec![vec![0.7]],
                    seasonal_means: vec![100.0],
                    seasonal_stds: vec![20.0],
                },
                marginal_distribution: Some(MarginalDistribution::Normal {
                    mean: 100.0,
                    std_dev: 20.0,
                }),
                seasonal_distributions: None,
            },
            UncertaintySpecification {
                uncertainty_type: UncertaintyType::Load,
                entity_id: 0, // Valid: matches bus_id 0
                temporal_model: TemporalModelInput::Independent,
                marginal_distribution: None,
                seasonal_distributions: Some(vec![
                    powers_rs::input::SeasonalDistribution {
                        season_id: 0,
                        distribution: MarginalDistribution::Normal {
                            mean: 80.0,
                            std_dev: 8.0,
                        },
                    },
                ]),
            },
        ],
        correlation: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_ok(),
        "Valid entity references should pass: {:?}",
        result
    );
}

#[test]
fn test_recourse_validation_invalid_inflow_entity_id_fails() {
    use powers_rs::input::{
        InitialConditionInput, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
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
            max_turbined_flow: 50.0,
            spillage_penalty: 1.0,
        }],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 99, // Invalid: no hydro with id 99
            temporal_model: TemporalModelInput::PeriodicAr {
                num_seasons: 1,
                ar_orders: vec![1],
                ar_coefficients: vec![vec![0.7]],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![20.0],
            },
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }),
            seasonal_distributions: None,
        }],
        correlation: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err(), "Invalid inflow entity_id should fail");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("entity_id") && error.contains("99"),
        "Error should mention invalid entity_id: {}",
        error
    );
}

#[test]
fn test_recourse_validation_invalid_load_entity_id_fails() {
    use powers_rs::input::{
        InitialConditionInput, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
    };

    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 5, // Invalid: no bus with id 5
            temporal_model: TemporalModelInput::Independent,
            marginal_distribution: None,
            seasonal_distributions: Some(vec![
                powers_rs::input::SeasonalDistribution {
                    season_id: 0,
                    distribution: MarginalDistribution::Normal {
                        mean: 80.0,
                        std_dev: 8.0,
                    },
                },
            ]),
        }],
        correlation: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(result.is_err(), "Invalid load entity_id should fail");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("entity_id") && error.contains("5"),
        "Error should mention invalid entity_id: {}",
        error
    );
}

#[test]
fn test_recourse_validation_invalid_initial_storage_hydro_id_fails() {
    use powers_rs::input::{
        InitialConditionInput, InitialStorage, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
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
            max_turbined_flow: 50.0,
            spillage_penalty: 1.0,
        }],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![InitialStorage {
                hydro_id: 7, // Invalid: no hydro with id 7
                value: 50.0,
            }],
            inflow: vec![],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0, // Valid entity_id
            temporal_model: TemporalModelInput::PeriodicAr {
                num_seasons: 1,
                ar_orders: vec![1],
                ar_coefficients: vec![vec![0.7]],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![20.0],
            },
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }),
            seasonal_distributions: None,
        }],
        correlation: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Invalid initial_condition storage hydro_id should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("hydro_id") && error.contains("7"),
        "Error should mention invalid hydro_id: {}",
        error
    );
}

#[test]
fn test_recourse_validation_invalid_initial_inflow_hydro_id_fails() {
    use powers_rs::input::{
        InitialConditionInput, MarginalDistribution, PastInflow, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
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
            max_turbined_flow: 50.0,
            spillage_penalty: 1.0,
        }],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![PastInflow {
                hydro_id: 3, // Invalid: no hydro with id 3
                lag: 1,
                value: 100.0,
            }],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0, // Valid entity_id
            temporal_model: TemporalModelInput::PeriodicAr {
                num_seasons: 1,
                ar_orders: vec![1],
                ar_coefficients: vec![vec![0.7]],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![20.0],
            },
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }),
            seasonal_distributions: None,
        }],
        correlation: None,
    };

    let result = InputValidator::validate_recourse(&recourse, &system);
    assert!(
        result.is_err(),
        "Invalid initial_condition inflow hydro_id should fail"
    );

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("hydro_id") && error.contains("3"),
        "Error should mention invalid hydro_id: {}",
        error
    );
}

#[test]
fn test_consistency_validation_invalid_season_id_fails() {
    use powers_rs::input::{
        Config, InitialConditionInput, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
    };

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: None,
        num_threads: None,
        output_path: None,
        seed: 42,
    };

    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    // Graph only has season_id 0 and 1
    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 1, "expectation"),
        ],
        edges: vec![GraphEdgeInput {
            source_id: 0,
            target_id: 1,
            probability: 1.0,
            discount_rate: 0.05,
        }],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelInput::Independent,
            marginal_distribution: None,
            seasonal_distributions: Some(vec![
                powers_rs::input::SeasonalDistribution {
                    season_id: 0, // Valid
                    distribution: MarginalDistribution::Normal {
                        mean: 80.0,
                        std_dev: 8.0,
                    },
                },
                powers_rs::input::SeasonalDistribution {
                    season_id: 99, // Invalid: not in graph
                    distribution: MarginalDistribution::Normal {
                        mean: 85.0,
                        std_dev: 8.5,
                    },
                },
            ]),
        }],
        correlation: None,
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(result.is_err(), "Invalid season_id reference should fail");

    let error = format!("{}", result.unwrap_err());
    assert!(
        error.contains("season_id") && error.contains("99"),
        "Error should mention invalid season_id: {}",
        error
    );
}

#[test]
fn test_consistency_validation_valid_season_ids_pass() {
    use powers_rs::input::{
        Config, InitialConditionInput, MarginalDistribution, Recourse,
        TemporalModelInput, UncertaintySpecification, UncertaintyType,
    };

    let config = Config {
        num_iterations: 10,
        num_forward_passes: 5,
        num_simulation_scenarios: None,
        num_threads: None,
        output_path: None,
        seed: 42,
    };

    let system = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let graph = GraphInput {
        nodes: vec![
            create_test_graph_node(0, 0, 0, "expectation"),
            create_test_graph_node(1, 1, 1, "expectation"),
            create_test_graph_node(2, 2, 2, "expectation"),
        ],
        edges: vec![
            GraphEdgeInput {
                source_id: 0,
                target_id: 1,
                probability: 1.0,
                discount_rate: 0.05,
            },
            GraphEdgeInput {
                source_id: 1,
                target_id: 2,
                probability: 1.0,
                discount_rate: 0.05,
            },
        ],
    };

    let recourse = Recourse {
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![],
        },
        uncertainty_specifications: vec![UncertaintySpecification {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
            temporal_model: TemporalModelInput::Independent,
            marginal_distribution: None,
            seasonal_distributions: Some(vec![
                powers_rs::input::SeasonalDistribution {
                    season_id: 0, // Valid
                    distribution: MarginalDistribution::Normal {
                        mean: 80.0,
                        std_dev: 8.0,
                    },
                },
                powers_rs::input::SeasonalDistribution {
                    season_id: 1, // Valid
                    distribution: MarginalDistribution::Normal {
                        mean: 85.0,
                        std_dev: 8.5,
                    },
                },
                powers_rs::input::SeasonalDistribution {
                    season_id: 2, // Valid
                    distribution: MarginalDistribution::Normal {
                        mean: 90.0,
                        std_dev: 9.0,
                    },
                },
            ]),
        }],
        correlation: None,
    };

    let result = InputValidator::validate_consistency(
        &config, &system, &graph, &recourse,
    );
    assert!(
        result.is_ok(),
        "Valid season_id references should pass: {:?}",
        result
    );
}
