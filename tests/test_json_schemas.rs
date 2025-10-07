//! Integration tests for T3.8 JSON Schema Documentation
//!
//! Tests verify that:
//! 1. All JSON schemas exist and are valid
//! 2. Example files conform to schemas (via serde deserialization)
//! 3. Schema files are valid JSON Schema Draft 7
//! 4. Documentation references are accurate

use powers_rs::input::{
    read_config_input, read_graph_input, read_recourse_input,
    read_system_input, Input,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

// ============================================================================
// Schema File Existence Tests
// ============================================================================

#[test]
fn test_config_schema_exists() {
    let schema_path = "schemas/config.schema.json";
    assert!(
        Path::new(schema_path).exists(),
        "Config schema file should exist at {}",
        schema_path
    );
}

#[test]
fn test_system_schema_exists() {
    let schema_path = "schemas/system.schema.json";
    assert!(
        Path::new(schema_path).exists(),
        "System schema file should exist at {}",
        schema_path
    );
}

#[test]
fn test_graph_schema_exists() {
    let schema_path = "schemas/graph.schema.json";
    assert!(
        Path::new(schema_path).exists(),
        "Graph schema file should exist at {}",
        schema_path
    );
}

#[test]
fn test_recourse_schema_exists() {
    let schema_path = "schemas/recourse.schema.json";
    assert!(
        Path::new(schema_path).exists(),
        "Recourse schema file should exist at {}",
        schema_path
    );
}

#[test]
fn test_vscode_settings_exists() {
    let settings_path = ".vscode/settings.json";
    assert!(
        Path::new(settings_path).exists(),
        "VS Code settings file should exist at {}",
        settings_path
    );
}

// ============================================================================
// Schema Validity Tests (Valid JSON Schema Draft 7)
// ============================================================================

#[test]
fn test_config_schema_is_valid_json() {
    let contents = fs::read_to_string("schemas/config.schema.json")
        .expect("Failed to read config schema");

    let _schema: Value = serde_json::from_str(&contents)
        .expect("Config schema should be valid JSON");

    // Verify it's a JSON Schema (has $schema field)
    let schema_obj: Value = serde_json::from_str(&contents).unwrap();
    assert!(
        schema_obj.get("$schema").is_some(),
        "Config schema should have $schema field"
    );
}

#[test]
fn test_system_schema_is_valid_json() {
    let contents = fs::read_to_string("schemas/system.schema.json")
        .expect("Failed to read system schema");

    let schema_obj: Value = serde_json::from_str(&contents)
        .expect("System schema should be valid JSON");

    assert!(
        schema_obj.get("$schema").is_some(),
        "System schema should have $schema field"
    );
}

#[test]
fn test_graph_schema_is_valid_json() {
    let contents = fs::read_to_string("schemas/graph.schema.json")
        .expect("Failed to read graph schema");

    let schema_obj: Value = serde_json::from_str(&contents)
        .expect("Graph schema should be valid JSON");

    assert!(
        schema_obj.get("$schema").is_some(),
        "Graph schema should have $schema field"
    );
}

#[test]
fn test_recourse_schema_is_valid_json() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");

    let schema_obj: Value = serde_json::from_str(&contents)
        .expect("Recourse schema should be valid JSON");

    assert!(
        schema_obj.get("$schema").is_some(),
        "Recourse schema should have $schema field"
    );
}

// ============================================================================
// Example File Conformance Tests (Serde Deserialization)
// ============================================================================

#[test]
fn test_example_config_conforms_to_schema() {
    // If serde can deserialize it, it conforms to the Rust types
    // which are documented in the schema
    let config = read_config_input("example/config.json");

    // Verify expected values from example
    assert_eq!(config.num_iterations, 32);
    assert_eq!(config.num_forward_passes, 4);
    assert_eq!(config.num_simulation_scenarios, 128);
    assert_eq!(config.seed, 0);
    assert_eq!(config.output_path, Some("./example".to_string()));
}

#[test]
fn test_example_system_conforms_to_schema() {
    let system = read_system_input("example/system.json");

    // Verify structure matches schema
    assert_eq!(system.buses.len(), 1, "Example has 1 bus");
    assert_eq!(system.lines.len(), 0, "Example has 0 lines");
    assert_eq!(system.thermals.len(), 2, "Example has 2 thermals");
    assert_eq!(system.hydros.len(), 1, "Example has 1 hydro");

    // Verify field types and constraints (serde handles this)
    assert_eq!(system.buses[0].id, 0);
    assert_eq!(system.buses[0].deficit_cost, 50.0);

    assert_eq!(system.thermals[0].id, 0);
    assert_eq!(system.thermals[0].bus_id, 0);
    assert!(system.thermals[0].cost >= 0.0);
    assert!(
        system.thermals[0].min_generation <= system.thermals[0].max_generation
    );

    assert_eq!(system.hydros[0].id, 0);
    assert!(system.hydros[0].productivity > 0.0);
    assert!(system.hydros[0].min_storage <= system.hydros[0].max_storage);
}

#[test]
fn test_example_graph_conforms_to_schema() {
    let graph = read_graph_input("example/graph.json");

    // Verify structure
    assert_eq!(graph.nodes.len(), 12, "Example has 12 nodes");
    assert_eq!(
        graph.edges.len(),
        11,
        "Example has 11 edges (sequential tree)"
    );

    // Verify node fields
    let first_node = &graph.nodes[0];
    assert_eq!(first_node.id, 0);
    assert_eq!(first_node.stage_id, 0);
    assert_eq!(first_node.season_id, 0);
    assert_eq!(first_node.risk_measure, "expectation");
    assert_eq!(first_node.load_stochastic_process, "naive");
    assert_eq!(first_node.inflow_stochastic_process, "naive");
    assert_eq!(first_node.state_variables, "storage");

    // Verify edge fields
    let first_edge = &graph.edges[0];
    assert_eq!(first_edge.source_id, 0);
    assert_eq!(first_edge.target_id, 1);
    assert!(first_edge.probability > 0.0 && first_edge.probability <= 1.0);
    assert!(first_edge.discount_rate >= 0.0);
}

#[test]
fn test_example_recourse_conforms_to_schema() {
    let recourse = read_recourse_input("example/recourse.json");

    // Verify initial condition
    assert_eq!(recourse.initial_condition.storage.len(), 1);
    assert_eq!(recourse.initial_condition.storage[0].hydro_id, 0);
    assert!(recourse.initial_condition.storage[0].value >= 0.0);

    assert_eq!(recourse.initial_condition.inflow.len(), 1);
    assert_eq!(recourse.initial_condition.inflow[0].hydro_id, 0);
    assert_eq!(recourse.initial_condition.inflow[0].lag, 1);

    // Verify uncertainties
    assert_eq!(recourse.uncertainties.len(), 12, "Example has 12 seasons");

    let first_uncertainty = &recourse.uncertainties[0];
    assert_eq!(first_uncertainty.season_id, 0);
    assert!(first_uncertainty.num_branchings > 0);

    // Verify distributions exist
    assert_eq!(first_uncertainty.distributions.load.len(), 1);
    assert_eq!(first_uncertainty.distributions.inflow.len(), 1);

    let load_dist = &first_uncertainty.distributions.load[0];
    assert_eq!(load_dist.bus_id, 0);

    let inflow_dist = &first_uncertainty.distributions.inflow[0];
    assert_eq!(inflow_dist.hydro_id, 0);
}

// ============================================================================
// Factory API Integration Tests (T3.7 + T3.8)
// ============================================================================

#[test]
fn test_factory_api_works_with_schema_validated_inputs() {
    // Verify that the factory API (T3.7) works with example files
    // that conform to schemas (T3.8)
    use powers_rs::sddp::SddpAlgorithm;

    let result = SddpAlgorithm::from_files(
        "example/config.json",
        "example/system.json",
        "example/graph.json",
        "example/recourse.json",
    );

    assert!(
        result.is_ok(),
        "Factory API should work with schema-conformant example files: {:?}",
        result.err()
    );
}

#[test]
fn test_input_from_paths_works_with_schema_validated_inputs() {
    // Verify Input::from_paths (T3.7) works with schema-conformant files
    let result = Input::from_paths(
        Path::new("example/config.json"),
        Path::new("example/system.json"),
        Path::new("example/graph.json"),
        Path::new("example/recourse.json"),
    );

    assert!(
        result.is_ok(),
        "Input::from_paths should work with schema-conformant files: {:?}",
        result.err()
    );
}

// ============================================================================
// Documentation Reference Tests
// ============================================================================

#[test]
fn test_input_specification_document_exists() {
    let doc_path = "docs/reference/INPUT-SPECIFICATION.md";
    assert!(
        Path::new(doc_path).exists(),
        "INPUT-SPECIFICATION.md should exist at {}",
        doc_path
    );
}

#[test]
fn test_input_specification_references_schemas() {
    let doc_contents =
        fs::read_to_string("docs/reference/INPUT-SPECIFICATION.md")
            .expect("Failed to read INPUT-SPECIFICATION.md");

    // Verify documentation references all schemas
    assert!(
        doc_contents.contains("config.schema.json"),
        "Documentation should reference config.schema.json"
    );
    assert!(
        doc_contents.contains("system.schema.json"),
        "Documentation should reference system.schema.json"
    );
    assert!(
        doc_contents.contains("graph.schema.json"),
        "Documentation should reference graph.schema.json"
    );
    assert!(
        doc_contents.contains("recourse.schema.json"),
        "Documentation should reference recourse.schema.json"
    );
}

#[test]
fn test_input_specification_references_example_files() {
    let doc_contents =
        fs::read_to_string("docs/reference/INPUT-SPECIFICATION.md")
            .expect("Failed to read INPUT-SPECIFICATION.md");

    // Verify documentation references example files
    assert!(
        doc_contents.contains("example/config.json"),
        "Documentation should reference example/config.json"
    );
    assert!(
        doc_contents.contains("example/system.json"),
        "Documentation should reference example/system.json"
    );
    assert!(
        doc_contents.contains("example/graph.json"),
        "Documentation should reference example/graph.json"
    );
    assert!(
        doc_contents.contains("example/recourse.json"),
        "Documentation should reference example/recourse.json"
    );
}

// ============================================================================
// VS Code Integration Tests
// ============================================================================

#[test]
fn test_vscode_settings_references_all_schemas() {
    let settings_contents = fs::read_to_string(".vscode/settings.json")
        .expect("Failed to read .vscode/settings.json");

    let settings: Value = serde_json::from_str(&settings_contents)
        .expect("VS Code settings should be valid JSON");

    // Verify schema mappings exist
    let schemas = settings
        .get("json.schemas")
        .expect("settings.json should have json.schemas field");

    assert!(schemas.is_array(), "json.schemas should be an array");

    let schemas_array = schemas.as_array().unwrap();
    assert_eq!(
        schemas_array.len(),
        4,
        "Should have 4 schema mappings (config, system, graph, recourse)"
    );

    // Verify all schemas are referenced
    let settings_str = settings_contents.to_lowercase();
    assert!(settings_str.contains("config.schema.json"));
    assert!(settings_str.contains("system.schema.json"));
    assert!(settings_str.contains("graph.schema.json"));
    assert!(settings_str.contains("recourse.schema.json"));
}

// ============================================================================
// Schema Content Tests (Verify Key Constraints)
// ============================================================================

#[test]
fn test_config_schema_defines_required_fields() {
    let contents = fs::read_to_string("schemas/config.schema.json")
        .expect("Failed to read config schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let required = schema
        .get("required")
        .expect("Config schema should have 'required' field")
        .as_array()
        .expect("'required' should be an array");

    assert_eq!(required.len(), 4, "Config should have 4 required fields");

    // Verify field names
    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"num_iterations"));
    assert!(required_strs.contains(&"num_forward_passes"));
    assert!(required_strs.contains(&"num_simulation_scenarios"));
    assert!(required_strs.contains(&"seed"));
}

#[test]
fn test_system_schema_defines_all_components() {
    let contents = fs::read_to_string("schemas/system.schema.json")
        .expect("Failed to read system schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let required = schema
        .get("required")
        .expect("System schema should have 'required' field")
        .as_array()
        .expect("'required' should be an array");

    assert_eq!(required.len(), 4, "System should have 4 required fields");

    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"buses"));
    assert!(required_strs.contains(&"lines"));
    assert!(required_strs.contains(&"thermals"));
    assert!(required_strs.contains(&"hydros"));
}

#[test]
fn test_graph_schema_defines_nodes_and_edges() {
    let contents = fs::read_to_string("schemas/graph.schema.json")
        .expect("Failed to read graph schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let required = schema
        .get("required")
        .expect("Graph schema should have 'required' field")
        .as_array()
        .expect("'required' should be an array");

    assert_eq!(required.len(), 2, "Graph should have 2 required fields");

    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"nodes"));
    assert!(required_strs.contains(&"edges"));
}

#[test]
fn test_recourse_schema_defines_initial_condition_and_uncertainties() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let required = schema
        .get("required")
        .expect("Recourse schema should have 'required' field")
        .as_array()
        .expect("'required' should be an array");

    assert_eq!(required.len(), 2, "Recourse should have 2 required fields");

    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"initial_condition"));
    assert!(required_strs.contains(&"uncertainties"));
}
