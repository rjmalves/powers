use powers_rs::input::{
    read_config_input, read_graph_input, read_recourse_input,
    read_system_input, Input,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

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

#[test]
fn test_example_config_conforms_to_schema() {
    // If serde can deserialize it, it conforms to the Rust types
    // which are documented in the schema
    let config = read_config_input("examples/01-deterministic/config.json");

    // Verify expected values from example
    assert_eq!(config.num_iterations, 10);
    assert_eq!(config.num_forward_passes, 1);
    assert_eq!(config.num_simulation_scenarios, Some(1));
    assert_eq!(config.seed, 42);
    assert_eq!(
        config.output_path,
        Some("./examples/01-deterministic".to_string())
    );
}

#[test]
fn test_example_system_conforms_to_schema() {
    let system = read_system_input("examples/01-deterministic/system.json");

    // Verify structure matches schema
    assert_eq!(system.buses.len(), 1, "Example has 1 bus");
    assert_eq!(system.lines.len(), 0, "Example has 0 lines");
    assert_eq!(system.thermals.len(), 1, "Example has 1 thermal");
    assert_eq!(system.hydros.len(), 1, "Example has 1 hydro");

    // Verify field types and constraints (serde handles this)
    assert_eq!(system.buses[0].id, 0);
    assert_eq!(system.buses[0].deficit_cost, 500.0);

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
    let graph = read_graph_input("examples/01-deterministic/graph.json");

    // Verify structure
    assert_eq!(graph.nodes.len(), 2, "Example has 2 nodes");
    assert_eq!(graph.edges.len(), 1, "Example has 1 edge (sequential tree)");

    // Verify node fields
    let first_node = &graph.nodes[0];
    assert_eq!(first_node.id, 0);
    assert_eq!(first_node.stage_id, 0);
    assert_eq!(first_node.season_id, 0);
    assert_eq!(first_node.risk_measure, "expectation");
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
    let recourse =
        read_recourse_input("examples/01-deterministic/recourse.json");

    // Verify initial condition
    assert_eq!(recourse.initial_condition.storage.len(), 1);
    assert_eq!(recourse.initial_condition.storage[0].hydro_id, 0);
    assert!(recourse.initial_condition.storage[0].value >= 0.0);

    // Initial inflow conditions are optional (empty for deterministic example)
    assert!(recourse.initial_condition.inflow.len() <= 1);

    // Verify uncertainty_specifications
    let specs = recourse.uncertainty_specifications;
    let first_spec = &specs[0];
    assert_eq!(first_spec.entity_id, 0);
}

#[test]
fn test_factory_api_works_with_schema_validated_inputs() {
    // Verify that the factory API (T3.7) works with example files
    // that conform to schemas (T3.8)
    use powers_rs::sddp::SddpAlgorithm;

    let result = SddpAlgorithm::from_files(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
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
        Path::new("examples/01-deterministic/config.json"),
        Path::new("examples/01-deterministic/system.json"),
        Path::new("examples/01-deterministic/graph.json"),
        Path::new("examples/01-deterministic/recourse.json"),
    );

    assert!(
        result.is_ok(),
        "Input::from_paths should work with schema-conformant files: {:?}",
        result.err()
    );
}

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
        doc_contents.contains("examples/01-deterministic/config.json"),
        "Documentation should reference examples/01-deterministic/config.json"
    );
    assert!(
        doc_contents.contains("examples/01-deterministic/system.json"),
        "Documentation should reference examples/01-deterministic/system.json"
    );
    assert!(
        doc_contents.contains("examples/01-deterministic/graph.json"),
        "Documentation should reference examples/01-deterministic/graph.json"
    );
    assert!(
        doc_contents.contains("examples/01-deterministic/recourse.json"),
        "Documentation should reference examples/01-deterministic/recourse.json"
    );
}

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

    assert_eq!(required.len(), 3, "Config should have 3 required fields (num_simulation_scenarios is now optional)");

    // Verify field names
    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"num_iterations"));
    assert!(required_strs.contains(&"num_forward_passes"));
    assert!(
        !required_strs.contains(&"num_simulation_scenarios"),
        "num_simulation_scenarios should not be required"
    );
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
fn test_recourse_schema_defines_uncertainty_specification() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let definitions = schema
        .get("definitions")
        .expect("Schema should have definitions")
        .as_object()
        .unwrap();

    assert!(
        definitions.contains_key("UncertaintySpecification"),
        "Should define UncertaintySpecification"
    );
    assert!(
        definitions.contains_key("SeasonalDistribution"),
        "Should define SeasonalDistribution"
    );
    assert!(
        definitions.contains_key("TemporalModelInput"),
        "Should define TemporalModelInput"
    );
}

#[test]
fn test_recourse_schema_uncertainty_spec_required_fields() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let uncertainty_spec = schema
        .get("definitions")
        .unwrap()
        .get("UncertaintySpecification")
        .expect("Should have UncertaintySpecification definition")
        .as_object()
        .unwrap();

    let required = uncertainty_spec
        .get("required")
        .expect("UncertaintySpecification should have required fields")
        .as_array()
        .unwrap();

    let required_strs: Vec<&str> =
        required.iter().map(|v| v.as_str().unwrap()).collect();

    assert!(required_strs.contains(&"uncertainty_type"));
    assert!(required_strs.contains(&"entity_id"));
    assert!(required_strs.contains(&"temporal_model"));
}

#[test]
fn test_recourse_schema_has_examples() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let examples = schema
        .get("examples")
        .expect("Schema should have examples")
        .as_array()
        .unwrap();

    assert!(
        examples.len() >= 3,
        "Should have at least 3 examples (PAR, independent, mixed)"
    );

    // Verify first example is PAR model
    let example1 = &examples[0];
    assert!(
        example1.get("uncertainty_specifications").is_some(),
        "Example 1 should use new format"
    );

    // Verify second example is independent model
    let example2 = &examples[1];
    let specs = example2
        .get("uncertainty_specifications")
        .unwrap()
        .as_array()
        .unwrap();
    let spec = &specs[0];
    let temporal_model = spec.get("temporal_model").unwrap();
    assert_eq!(
        temporal_model.get("type").unwrap().as_str().unwrap(),
        "independent"
    );
}

#[test]
fn test_recourse_schema_seasonal_distribution_required_fields() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let seasonal_dist = schema
        .get("definitions")
        .unwrap()
        .get("SeasonalDistribution")
        .expect("Should have SeasonalDistribution definition")
        .as_object()
        .unwrap();

    // SeasonalDistribution uses oneOf discriminated union
    let one_of = seasonal_dist
        .get("oneOf")
        .expect("SeasonalDistribution should have oneOf")
        .as_array()
        .unwrap();

    assert_eq!(
        one_of.len(),
        2,
        "Should have Normal and LogNormal3 variants"
    );

    // Check Normal variant
    let normal_variant = one_of[0].as_object().unwrap();
    let normal_required =
        normal_variant.get("required").unwrap().as_array().unwrap();
    let normal_required_strs: Vec<&str> = normal_required
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    assert!(normal_required_strs.contains(&"season_id"));
    assert!(normal_required_strs.contains(&"type"));
    assert!(normal_required_strs.contains(&"mean"));
    assert!(normal_required_strs.contains(&"std_dev"));

    // Check LogNormal3 variant
    let lognormal_variant = one_of[1].as_object().unwrap();
    let lognormal_required = lognormal_variant
        .get("required")
        .unwrap()
        .as_array()
        .unwrap();
    let lognormal_required_strs: Vec<&str> = lognormal_required
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    assert!(lognormal_required_strs.contains(&"season_id"));
    assert!(lognormal_required_strs.contains(&"type"));
    assert!(lognormal_required_strs.contains(&"gamma"));
    assert!(lognormal_required_strs.contains(&"mu"));
    assert!(lognormal_required_strs.contains(&"sigma"));
}

#[test]
fn test_recourse_schema_validation_rules_comprehensive() {
    let contents = fs::read_to_string("schemas/recourse.schema.json")
        .expect("Failed to read recourse schema");
    let schema: Value = serde_json::from_str(&contents).unwrap();

    let validation_rules = schema
        .get("validationRules")
        .expect("Schema should have validationRules")
        .as_object()
        .unwrap();

    assert!(
        validation_rules.contains_key("commonRules"),
        "Should have common rules"
    );
    assert!(
        validation_rules.contains_key("formatRules"),
        "Should have format rules"
    );

    // Verify format rules cover key validation points
    let format_rules = validation_rules
        .get("formatRules")
        .unwrap()
        .as_array()
        .unwrap();

    let rules_text = format_rules
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        rules_text.contains("marginal_distribution"),
        "Should mention marginal_distribution requirement for PAR"
    );
    assert!(
        rules_text.contains("seasonal_distributions"),
        "Should mention seasonal_distributions requirement for independent"
    );
    assert!(
        rules_text.contains("duplicate"),
        "Should mention no duplicate entities"
    );
}
