use powers_rs::input::*;
use std::fs;
use std::io::Write;
use tempfile::TempDir;

#[test]
#[should_panic(expected = "Error while reading config file")]
fn test_read_config_input_missing_file() {
    // Test error handling when config file doesn't exist
    read_config_input("/nonexistent/path/config.json");
}

#[test]
#[should_panic(expected = "Error while reading config file")]
fn test_read_system_input_missing_file() {
    // Test error handling when system file doesn't exist
    read_system_input("/nonexistent/path/system.json");
}

#[test]
#[should_panic(expected = "Error while reading graph file")]
fn test_read_graph_input_missing_file() {
    // Test error handling when graph file doesn't exist
    read_graph_input("/nonexistent/path/graph.json");
}

#[test]
#[should_panic(expected = "Error while reading recourse file")]
fn test_read_recourse_input_missing_file() {
    // Test error handling when recourse file doesn't exist
    read_recourse_input("/nonexistent/path/recourse.json");
}

#[test]
#[should_panic]
fn test_read_config_input_malformed_json() {
    // Test error handling with invalid JSON syntax
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("bad_config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(file, "{{ this is not valid JSON }}").unwrap();

    read_config_input(file_path.to_str().unwrap());
}

#[test]
#[should_panic]
fn test_read_config_input_missing_required_fields() {
    // Test error when required fields are missing
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("incomplete_config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(file, "{{}}").unwrap(); // Empty object missing all required fields

    read_config_input(file_path.to_str().unwrap());
}

#[test]
#[should_panic]
fn test_read_system_input_malformed_json() {
    // Test error handling with invalid JSON syntax in system file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("bad_system.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(file, "{{ invalid json syntax").unwrap();

    read_system_input(file_path.to_str().unwrap());
}

#[test]
#[should_panic]
fn test_read_graph_input_malformed_json() {
    // Test error handling with invalid JSON in graph file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("bad_graph.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(file, "[not valid JSON]").unwrap();

    read_graph_input(file_path.to_str().unwrap());
}

#[test]
#[should_panic]
fn test_read_recourse_input_malformed_json() {
    // Test error handling with invalid JSON in recourse file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("bad_recourse.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(file, "{{ malformed: json }}").unwrap();

    read_recourse_input(file_path.to_str().unwrap());
}

#[test]
fn test_config_with_none_output_path() {
    // Test Config with None output_path (default)
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "num_iterations": 10,
        "num_forward_passes": 100,
        "num_simulation_scenarios": 500,
        "seed": 42
    }}"#
    )
    .unwrap();

    let config = read_config_input(file_path.to_str().unwrap());
    assert_eq!(config.num_iterations, 10);
    assert_eq!(config.num_forward_passes, 100);
    assert_eq!(config.num_simulation_scenarios, 500);
    assert_eq!(config.seed, 42);
    assert!(
        config.output_path.is_none(),
        "output_path should default to None"
    );
}

#[test]
fn test_config_with_some_output_path() {
    // Test Config with Some output_path
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "num_iterations": 5,
        "num_forward_passes": 50,
        "num_simulation_scenarios": 200,
        "seed": 123,
        "output_path": "/tmp/output"
    }}"#
    )
    .unwrap();

    let config = read_config_input(file_path.to_str().unwrap());
    assert_eq!(config.output_path, Some("/tmp/output".to_string()));
}

#[test]
fn test_system_input_minimal() {
    // Test minimal valid system with empty arrays
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("minimal_system.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "buses": [],
        "lines": [],
        "thermals": [],
        "hydros": []
    }}"#
    )
    .unwrap();

    let system = read_system_input(file_path.to_str().unwrap());
    assert_eq!(system.buses.len(), 0);
    assert_eq!(system.lines.len(), 0);
    assert_eq!(system.thermals.len(), 0);
    assert_eq!(system.hydros.len(), 0);
}

#[test]
fn test_system_input_with_single_bus() {
    // Test system with single bus
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("single_bus_system.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "buses": [
            {{"id": 0, "deficit_cost": 1000.0}}
        ],
        "lines": [],
        "thermals": [],
        "hydros": []
    }}"#
    )
    .unwrap();

    let system = read_system_input(file_path.to_str().unwrap());
    assert_eq!(system.buses.len(), 1);
    assert_eq!(system.buses[0].id, 0);
    assert_eq!(system.buses[0].deficit_cost, 1000.0);
}

#[test]
#[should_panic]
fn test_system_input_invalid_json_type() {
    // Test type mismatch (string instead of number)
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("invalid_type_system.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "buses": [
            {{"id": "not_a_number", "deficit_cost": 1000.0}}
        ],
        "lines": [],
        "thermals": [],
        "hydros": []
    }}"#
    )
    .unwrap();

    read_system_input(file_path.to_str().unwrap());
}

#[test]
fn test_graph_input_minimal() {
    // Test minimal valid graph with empty nodes and edges
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("minimal_graph.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "nodes": [],
        "edges": []
    }}"#
    )
    .unwrap();

    let graph = read_graph_input(file_path.to_str().unwrap());
    assert_eq!(graph.nodes.len(), 0);
    assert_eq!(graph.edges.len(), 0);
}

#[test]
#[should_panic]
fn test_graph_input_missing_nodes_field() {
    // Test error when nodes field is missing
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("no_nodes_graph.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "edges": []
    }}"#
    )
    .unwrap();

    read_graph_input(file_path.to_str().unwrap());
}

#[test]
#[should_panic]
fn test_recourse_input_missing_initial_condition() {
    // Test error when initial_condition is missing
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("bad_recourse.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "noise_models": []
    }}"#
    )
    .unwrap();

    read_recourse_input(file_path.to_str().unwrap());
}

#[test]
fn test_recourse_input_minimal() {
    // Test minimal valid recourse input
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("minimal_recourse.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "initial_condition": {{
            "storage": [],
            "inflow": []
        }},
        "noise_models": []
    }}"#
    )
    .unwrap();

    let recourse = read_recourse_input(file_path.to_str().unwrap());
    assert_eq!(recourse.initial_condition.storage.len(), 0);
    assert_eq!(recourse.initial_condition.inflow.len(), 0);
    assert_eq!(recourse.noise_models.as_ref().map_or(0, |m| m.len()), 0);
}

#[test]
fn test_deserialize_bus_input() {
    // Test BusInput deserialization
    let json = r#"{"id": 5, "deficit_cost": 2500.5}"#;
    let bus: BusInput = serde_json::from_str(json).unwrap();
    assert_eq!(bus.id, 5);
    assert_eq!(bus.deficit_cost, 2500.5);
}

#[test]
fn test_deserialize_line_input() {
    // Test LineInput deserialization
    let json = r#"{
        "id": 1,
        "source_bus_id": 0,
        "target_bus_id": 1,
        "direct_capacity": 100.0,
        "reverse_capacity": 80.0,
        "exchange_penalty": 10.0
    }"#;
    let line: LineInput = serde_json::from_str(json).unwrap();
    assert_eq!(line.id, 1);
    assert_eq!(line.source_bus_id, 0);
    assert_eq!(line.target_bus_id, 1);
    assert_eq!(line.direct_capacity, 100.0);
    assert_eq!(line.reverse_capacity, 80.0);
    assert_eq!(line.exchange_penalty, 10.0);
}

#[test]
fn test_deserialize_thermal_input() {
    // Test ThermalInput deserialization
    let json = r#"{
        "id": 2,
        "bus_id": 0,
        "cost": 50.0,
        "min_generation": 10.0,
        "max_generation": 100.0
    }"#;
    let thermal: ThermalInput = serde_json::from_str(json).unwrap();
    assert_eq!(thermal.id, 2);
    assert_eq!(thermal.bus_id, 0);
    assert_eq!(thermal.cost, 50.0);
    assert_eq!(thermal.min_generation, 10.0);
    assert_eq!(thermal.max_generation, 100.0);
}

#[test]
fn test_deserialize_hydro_input() {
    // Test HydroInput deserialization with None downstream
    let json = r#"{
        "id": 0,
        "downstream_hydro_id": null,
        "bus_id": 1,
        "productivity": 0.9,
        "min_storage": 0.0,
        "max_storage": 100.0,
        "min_turbined_flow": 0.0,
        "max_turbined_flow": 50.0,
        "spillage_penalty": 0.1
    }"#;
    let hydro: HydroInput = serde_json::from_str(json).unwrap();
    assert_eq!(hydro.id, 0);
    assert_eq!(hydro.downstream_hydro_id, None);
    assert_eq!(hydro.bus_id, 1);
    assert_eq!(hydro.productivity, 0.9);
}

#[test]
fn test_deserialize_hydro_input_with_downstream() {
    // Test HydroInput deserialization with Some downstream
    let json = r#"{
        "id": 1,
        "downstream_hydro_id": 0,
        "bus_id": 1,
        "productivity": 0.85,
        "min_storage": 5.0,
        "max_storage": 80.0,
        "min_turbined_flow": 1.0,
        "max_turbined_flow": 40.0,
        "spillage_penalty": 0.2
    }"#;
    let hydro: HydroInput = serde_json::from_str(json).unwrap();
    assert_eq!(hydro.id, 1);
    assert_eq!(hydro.downstream_hydro_id, Some(0));
}

#[test]
fn test_deserialize_initial_storage() {
    // Test InitialStorage deserialization
    let json = r#"{"hydro_id": 0, "value": 50.0}"#;
    let storage: InitialStorage = serde_json::from_str(json).unwrap();
    assert_eq!(storage.hydro_id, 0);
    assert_eq!(storage.value, 50.0);
}

#[test]
fn test_deserialize_past_inflow() {
    // Test PastInflow deserialization
    let json = r#"{"hydro_id": 1, "lag": 2, "value": 30.5}"#;
    let inflow: PastInflow = serde_json::from_str(json).unwrap();
    assert_eq!(inflow.hydro_id, 1);
    assert_eq!(inflow.lag, 2);
    assert_eq!(inflow.value, 30.5);
}

#[test]
fn test_deserialize_normal_params() {
    // Test NormalParams deserialization
    let json = r#"{"mu": 100.0, "sigma": 10.0}"#;
    let params: NormalParams = serde_json::from_str(json).unwrap();
    assert_eq!(params.mu, 100.0);
    assert_eq!(params.sigma, 10.0);
}

#[test]
fn test_deserialize_load_distribution() {
    // Test LoadDistribution deserialization
    let json = r#"{
        "bus_id": 0,
        "normal": {"mu": 150.0, "sigma": 15.0}
    }"#;
    let load: LoadDistribution = serde_json::from_str(json).unwrap();
    assert_eq!(load.bus_id, 0);
    assert_eq!(load.normal.mu, 150.0);
    assert_eq!(load.normal.sigma, 15.0);
}

#[test]
fn test_config_with_zero_iterations() {
    // Edge case: zero iterations (valid but unusual)
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("zero_iter_config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "num_iterations": 0,
        "num_forward_passes": 1,
        "num_simulation_scenarios": 1,
        "seed": 0
    }}"#
    )
    .unwrap();

    let config = read_config_input(file_path.to_str().unwrap());
    assert_eq!(config.num_iterations, 0);
}

#[test]
fn test_config_with_large_numbers() {
    // Edge case: large numbers
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large_config.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "num_iterations": 1000000,
        "num_forward_passes": 10000,
        "num_simulation_scenarios": 100000,
        "seed": 18446744073709551615
    }}"#
    )
    .unwrap();

    let config = read_config_input(file_path.to_str().unwrap());
    assert_eq!(config.num_iterations, 1000000);
    assert_eq!(config.seed, 18446744073709551615);
}

#[test]
fn test_system_with_multiple_elements() {
    // Test system with multiple buses, lines, thermals, and hydros
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("multi_element_system.json");
    let mut file = fs::File::create(&file_path).unwrap();
    writeln!(
        file,
        r#"{{
        "buses": [
            {{"id": 0, "deficit_cost": 1000.0}},
            {{"id": 1, "deficit_cost": 1200.0}}
        ],
        "lines": [
            {{
                "id": 0,
                "source_bus_id": 0,
                "target_bus_id": 1,
                "direct_capacity": 50.0,
                "reverse_capacity": 50.0,
                "exchange_penalty": 5.0
            }}
        ],
        "thermals": [
            {{
                "id": 0,
                "bus_id": 0,
                "cost": 100.0,
                "min_generation": 0.0,
                "max_generation": 50.0
            }}
        ],
        "hydros": [
            {{
                "id": 0,
                "downstream_hydro_id": null,
                "bus_id": 1,
                "productivity": 1.0,
                "min_storage": 0.0,
                "max_storage": 100.0,
                "min_turbined_flow": 0.0,
                "max_turbined_flow": 50.0,
                "spillage_penalty": 0.01
            }}
        ]
    }}"#
    )
    .unwrap();

    let system = read_system_input(file_path.to_str().unwrap());
    assert_eq!(system.buses.len(), 2);
    assert_eq!(system.lines.len(), 1);
    assert_eq!(system.thermals.len(), 1);
    assert_eq!(system.hydros.len(), 1);
}

#[test]
fn test_build_sddp_system_empty() {
    // Test building system from empty input
    let system_input = SystemInput {
        buses: vec![],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.buses_count, 0);
    assert_eq!(system.meta.lines_count, 0);
    assert_eq!(system.meta.thermals_count, 0);
    assert_eq!(system.meta.hydros_count, 0);
}

#[test]
fn test_build_sddp_system_single_bus() {
    // Test building system with single bus
    let system_input = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 5000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.buses_count, 1);
    assert_eq!(system.buses[0].id, 0);
}

#[test]
fn test_build_sddp_system_with_line() {
    // Test building system with buses and a line
    let system_input = SystemInput {
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
            target_bus_id: 1,
            direct_capacity: 100.0,
            reverse_capacity: 80.0,
            exchange_penalty: 10.0,
        }],
        thermals: vec![],
        hydros: vec![],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.buses_count, 2);
    assert_eq!(system.meta.lines_count, 1);
}

#[test]
fn test_build_sddp_system_with_thermal() {
    // Test building system with thermal unit
    let system_input = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 50.0,
            min_generation: 10.0,
            max_generation: 100.0,
        }],
        hydros: vec![],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.thermals_count, 1);
    assert_eq!(system.thermals[0].id, 0);
    assert_eq!(system.thermals[0].bus_id, 0);
}

#[test]
fn test_build_sddp_system_with_hydro() {
    // Test building system with hydro unit
    let system_input = SystemInput {
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
            productivity: 0.9,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.1,
        }],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.hydros_count, 1);
    assert_eq!(system.hydros[0].id, 0);
}

#[test]
fn test_build_sddp_system_with_cascaded_hydros() {
    // Test building system with cascaded hydro units
    let system_input = SystemInput {
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
                max_turbined_flow: 50.0,
                spillage_penalty: 0.0,
            },
            HydroInput {
                id: 1,
                downstream_hydro_id: Some(0),
                bus_id: 0,
                productivity: 0.95,
                min_storage: 0.0,
                max_storage: 80.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 40.0,
                spillage_penalty: 0.0,
            },
        ],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.hydros_count, 2);
    assert_eq!(system.hydros[1].downstream_hydro_id, Some(0));
}

#[test]
#[should_panic(expected = "ID 0 not found for buses")]
fn test_build_sddp_system_invalid_bus_ids() {
    // Test error when bus IDs don't start from 0
    let system_input = SystemInput {
        buses: vec![BusInput {
            id: 1, // Invalid: should start from 0
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![],
        hydros: vec![],
    };

    system_input.build_sddp_system();
}

#[test]
#[should_panic(expected = "ID 1 not found for lines")]
fn test_build_sddp_system_invalid_line_ids() {
    // Test error when line IDs have gaps
    let system_input = SystemInput {
        buses: vec![
            BusInput {
                id: 0,
                deficit_cost: 1000.0,
            },
            BusInput {
                id: 1,
                deficit_cost: 1000.0,
            },
        ],
        lines: vec![
            LineInput {
                id: 0,
                source_bus_id: 0,
                target_bus_id: 1,
                direct_capacity: 100.0,
                reverse_capacity: 100.0,
                exchange_penalty: 0.0,
            },
            LineInput {
                id: 2, // Invalid: skips ID 1
                source_bus_id: 1,
                target_bus_id: 0,
                direct_capacity: 100.0,
                reverse_capacity: 100.0,
                exchange_penalty: 0.0,
            },
        ],
        thermals: vec![],
        hydros: vec![],
    };

    system_input.build_sddp_system();
}

#[test]
#[should_panic(expected = "ID 0 not found for thermals")]
fn test_build_sddp_system_invalid_thermal_ids() {
    // Test error when thermal IDs are invalid
    let system_input = SystemInput {
        buses: vec![BusInput {
            id: 0,
            deficit_cost: 1000.0,
        }],
        lines: vec![],
        thermals: vec![ThermalInput {
            id: 1, // Invalid: should start from 0
            bus_id: 0,
            cost: 50.0,
            min_generation: 0.0,
            max_generation: 100.0,
        }],
        hydros: vec![],
    };

    system_input.build_sddp_system();
}

#[test]
#[should_panic(expected = "ID 1 not found for hydros")]
fn test_build_sddp_system_invalid_hydro_ids() {
    // Test error when hydro IDs have gaps
    let system_input = SystemInput {
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
                max_turbined_flow: 50.0,
                spillage_penalty: 0.0,
            },
            HydroInput {
                id: 2, // Invalid: skips ID 1
                downstream_hydro_id: Some(0),
                bus_id: 0,
                productivity: 1.0,
                min_storage: 0.0,
                max_storage: 100.0,
                min_turbined_flow: 0.0,
                max_turbined_flow: 50.0,
                spillage_penalty: 0.0,
            },
        ],
    };

    system_input.build_sddp_system();
}

#[test]
fn test_build_sddp_system_complete() {
    // Test building complete system with all elements
    let system_input = SystemInput {
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
            target_bus_id: 1,
            direct_capacity: 150.0,
            reverse_capacity: 120.0,
            exchange_penalty: 5.0,
        }],
        thermals: vec![ThermalInput {
            id: 0,
            bus_id: 0,
            cost: 80.0,
            min_generation: 20.0,
            max_generation: 200.0,
        }],
        hydros: vec![HydroInput {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 1,
            productivity: 0.88,
            min_storage: 10.0,
            max_storage: 150.0,
            min_turbined_flow: 5.0,
            max_turbined_flow: 80.0,
            spillage_penalty: 0.2,
        }],
    };

    let system = system_input.build_sddp_system();
    assert_eq!(system.meta.buses_count, 2);
    assert_eq!(system.meta.lines_count, 1);
    assert_eq!(system.meta.thermals_count, 1);
    assert_eq!(system.meta.hydros_count, 1);
}
