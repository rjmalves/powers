/// Unit tests for num_threads configuration parameter.
///
/// Tests:
/// - Config deserialization with/without num_threads
/// - Thread pool configuration (explicit, auto, validation)
use powers_rs::input::Config;

#[test]
fn test_config_deserialize_with_num_threads() {
    let json = r#"{
        "general": {
            "seed": 42,
            "num_threads": 4
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 100
        }
    }"#;

    let config: Config = serde_json::from_str(json).expect("Should parse");
    assert_eq!(config.general.num_threads, Some(4));
}

#[test]
fn test_config_deserialize_with_num_threads_null() {
    let json = r#"{
        "general": {
            "seed": 42,
            "num_threads": null
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 100
        }
    }"#;

    let config: Config = serde_json::from_str(json).expect("Should parse");
    assert_eq!(config.general.num_threads, None);
}

#[test]
fn test_config_deserialize_without_num_threads() {
    // Test that configs without num_threads field (defaults to None) still work
    let json = r#"{
        "general": {
            "seed": 42
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 100
        }
    }"#;

    let config: Config = serde_json::from_str(json).expect("Should parse");
    assert_eq!(config.general.num_threads, None);
}

#[test]
fn test_all_example_configs_parse() {
    // Verify all example configs parse correctly
    let examples = vec![
        "examples/01-deterministic/config.json",
        "examples/02-stochastic/config.json",
        "examples/03-multistage/config.json",
        "examples/04-cascade/config.json",
        "examples/05-large-scale-brazilian/config.json",
    ];

    for example in examples {
        let contents = std::fs::read_to_string(example)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", example, e));

        let config: Config = serde_json::from_str(&contents)
            .unwrap_or_else(|e| panic!("Failed to parse {}: {}", example, e));

        // All examples should have num_threads set
        assert!(
            config.general.num_threads.is_some(),
            "{} should have general.num_threads",
            example
        );
    }
}

#[test]
fn test_config_deserialize_with_output_path() {
    // Test that num_threads works with output_path
    let json = r#"{
        "general": {
            "seed": 42,
            "num_threads": 8
        },
        "training": {
            "num_iterations": 10,
            "num_forward_passes": 4
        },
        "simulation": {
            "num_scenarios": 100
        },
        "output": {
            "path": "./output"
        }
    }"#;

    let config: Config = serde_json::from_str(json).expect("Should parse");
    assert_eq!(config.general.num_threads, Some(8));
    assert_eq!(config.output.path, Some("./output".to_string()));
}
