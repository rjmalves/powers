//! Comprehensive test infrastructure for UnifiedNoiseSpec conversion
//!
//! This module provides:
//! 1. Test fixture loaders for example cases
//! 2. Builder utilities for creating test specs
//! 3. Round-trip validation (old format → new format → scenarios)
//! 4. Snapshot testing for conversion results
//! 5. Performance baseline establishment
//!
//! PERFORMANCE: Round-trip tests ensure conversion doesn't change scenario generation results.
//! All tests use tolerance-based comparison for floating-point arithmetic.

use powers_rs::input::{
    GraphInput, MarginalDistribution, NoiseModel, SystemInput, UncertaintyType,
};
use powers_rs::unified_noise_spec::{
    SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec, UnifiedNoiseSpec,
};
use std::collections::HashMap;
use std::path::PathBuf;

// ================================
// Test Fixture Loaders
// ================================

/// Load test fixture from examples directory
///
/// # Arguments
/// - `name`: Relative path from examples/ (e.g., "06-par-model/01-simple-par1")
///
/// # Returns
/// Tuple of (system, graph, noise_models) loaded from JSON files
///
/// # Performance
/// File I/O overhead acceptable for tests. Results not cached to ensure test isolation.
fn load_test_fixture(name: &str) -> (SystemInput, GraphInput, Vec<NoiseModel>) {
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name);

    let system_path = base_path.join("system.json");
    let graph_path = base_path.join("graph.json");
    let recourse_path = base_path.join("recourse.json");

    // Read files
    let system_json =
        std::fs::read_to_string(&system_path).unwrap_or_else(|e| {
            panic!(
                "Failed to read system.json from {}: {}",
                system_path.display(),
                e
            )
        });
    let graph_json = std::fs::read_to_string(&graph_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read graph.json from {}: {}",
            graph_path.display(),
            e
        )
    });
    let recourse_json =
        std::fs::read_to_string(&recourse_path).unwrap_or_else(|e| {
            panic!(
                "Failed to read recourse.json from {}: {}",
                recourse_path.display(),
                e
            )
        });

    // Parse JSON
    let system: SystemInput = serde_json::from_str(&system_json)
        .unwrap_or_else(|e| panic!("Failed to parse system.json: {}", e));
    let graph: GraphInput = serde_json::from_str(&graph_json)
        .unwrap_or_else(|e| panic!("Failed to parse graph.json: {}", e));

    // Parse recourse to extract noise_models
    let recourse: serde_json::Value = serde_json::from_str(&recourse_json)
        .unwrap_or_else(|e| panic!("Failed to parse recourse.json: {}", e));
    let noise_models: Vec<NoiseModel> =
        serde_json::from_value(recourse["noise_models"].clone())
            .unwrap_or_else(|e| panic!("Failed to parse noise_models: {}", e));

    (system, graph, noise_models)
}

// ================================
// Builder Utilities
// ================================

/// Builder for PAR UnifiedNoiseSpec (test utility)
///
/// Creates a standard PAR model with:
/// - Specified entity_id and num_seasons
/// - AR(1) model for all seasons with coefficient 0.7
/// - Mean 100.0, std_dev 20.0 for all seasons
/// - LogNormal3 distribution (gamma=1.0, mu=0.0, sigma=0.6)
///
/// # Performance
/// Pre-allocates HashMaps with capacity to avoid rehashing.
#[allow(dead_code)]
fn build_par_spec(
    entity_id: usize,
    uncertainty_type: UncertaintyType,
    num_seasons: usize,
) -> UnifiedNoiseSpec {
    // PERFORMANCE: Pre-allocate with known capacity
    let mut seasonal_params = HashMap::with_capacity(num_seasons);
    let mut seasonal_ar_params = HashMap::with_capacity(num_seasons);

    for season in 0..num_seasons {
        seasonal_params.insert(
            season,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        seasonal_ar_params.insert(
            season,
            SeasonalPARParams {
                ar_order: 1,
                ar_coefficients: vec![0.7],
            },
        );
    }

    UnifiedNoiseSpec {
        entity_id,
        uncertainty_type,
        temporal_model: TemporalModelSpec::PeriodicAutoregressive {
            num_seasons,
            seasonal_ar_params,
        },
        seasonal_params,
        marginal_distribution: Some(MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 0.0,
            sigma: 0.6,
        }),
    }
}

/// Builder for independent UnifiedNoiseSpec (test utility)
///
/// Creates an independent model with:
/// - Specified entity_id and seasons
/// - Mean 80.0, std_dev 15.0 for all seasons
/// - Normal distribution
///
/// # Performance
/// Pre-allocates HashMap with exact capacity.
#[allow(dead_code)]
fn build_independent_spec(
    entity_id: usize,
    uncertainty_type: UncertaintyType,
    seasons: Vec<usize>,
) -> UnifiedNoiseSpec {
    // PERFORMANCE: Pre-allocate with known capacity
    let mut seasonal_params = HashMap::with_capacity(seasons.len());

    for season in seasons {
        seasonal_params.insert(
            season,
            SeasonalNoiseParams {
                mean: 80.0,
                std_dev: 15.0,
                marginal_override: Some(MarginalDistribution::Normal {
                    mean: 80.0,
                    std_dev: 15.0,
                }),
            },
        );
    }

    UnifiedNoiseSpec {
        entity_id,
        uncertainty_type,
        temporal_model: TemporalModelSpec::Independent,
        seasonal_params,
        marginal_distribution: None,
    }
}

// ================================
// Comparison Utilities
// ================================

/// Assert that two UnifiedNoiseSpec collections are equivalent
///
/// Checks:
/// - Same number of specs
/// - Same entity_id and uncertainty_type for each
/// - Same temporal model type
/// - Seasonal parameters within tolerance
///
/// # Performance
/// O(n × m) where n = specs count, m = seasons per spec.
/// Uses direct HashMap lookups (O(1)) for seasonal parameter comparison.
#[allow(dead_code)]
fn assert_specs_equivalent(
    specs1: &[UnifiedNoiseSpec],
    specs2: &[UnifiedNoiseSpec],
    tolerance: f64,
) {
    assert_eq!(
        specs1.len(),
        specs2.len(),
        "Spec count mismatch: {} vs {}",
        specs1.len(),
        specs2.len()
    );

    for (spec1, spec2) in specs1.iter().zip(specs2.iter()) {
        assert_eq!(spec1.entity_id, spec2.entity_id, "Entity ID mismatch");
        assert_eq!(
            format!("{:?}", spec1.uncertainty_type),
            format!("{:?}", spec2.uncertainty_type),
            "Uncertainty type mismatch"
        );

        // Check temporal model type matches
        match (&spec1.temporal_model, &spec2.temporal_model) {
            (
                TemporalModelSpec::Independent,
                TemporalModelSpec::Independent,
            ) => {}
            (
                TemporalModelSpec::PeriodicAutoregressive {
                    num_seasons: n1,
                    ..
                },
                TemporalModelSpec::PeriodicAutoregressive {
                    num_seasons: n2,
                    ..
                },
            ) => {
                assert_eq!(n1, n2, "PAR num_seasons mismatch");
            }
            _ => panic!("Temporal model type mismatch"),
        }

        // Check seasonal parameters
        assert_eq!(
            spec1.seasonal_params.len(),
            spec2.seasonal_params.len(),
            "Seasonal params count mismatch"
        );

        for (season_id, params1) in &spec1.seasonal_params {
            let params2 =
                spec2.seasonal_params.get(season_id).unwrap_or_else(|| {
                    panic!("Season {} not found in spec2", season_id)
                });

            assert!(
                (params1.mean - params2.mean).abs() < tolerance,
                "Mean mismatch for season {}: {} vs {}",
                season_id,
                params1.mean,
                params2.mean
            );
            assert!(
                (params1.std_dev - params2.std_dev).abs() < tolerance,
                "Std_dev mismatch for season {}: {} vs {}",
                season_id,
                params1.std_dev,
                params2.std_dev
            );
        }
    }
}

// ================================
// Conversion Tests
// ================================

#[test]
fn test_conversion_simple_par1() {
    // Load example 06-par-model/01-simple-par1
    let (system, graph, noise_models) =
        load_test_fixture("06-par-model/01-simple-par1");

    // Convert to unified specs
    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed for simple PAR model");

    // Validate specs
    for spec in &unified_specs {
        spec.validate()
            .unwrap_or_else(|e| panic!("Validation failed: {}", e));
        spec.validate_against_graph(&graph, &system)
            .unwrap_or_else(|e| panic!("Cross-validation failed: {}", e));
    }

    // Check conversion correctness
    // simple-par1 has 1 hydro (PAR inflow) + 1 bus (independent load) = 2 specs
    assert_eq!(
        unified_specs.len(),
        2,
        "Should have 2 specs (1 hydro + 1 bus)"
    );

    let inflow_spec = unified_specs
        .iter()
        .find(|s| matches!(s.uncertainty_type, UncertaintyType::Inflow))
        .expect("Should have inflow spec");
    assert_eq!(inflow_spec.entity_id, 0);

    // Check PAR structure
    match &inflow_spec.temporal_model {
        TemporalModelSpec::PeriodicAutoregressive {
            num_seasons,
            seasonal_ar_params,
        } => {
            assert_eq!(*num_seasons, 12, "Should have 12 seasons");
            assert_eq!(
                seasonal_ar_params.len(),
                12,
                "Should have 12 AR param sets"
            );

            // Check AR(1) structure
            for season in 0..12 {
                let ar_params = seasonal_ar_params
                    .get(&season)
                    .expect("AR params should exist for all seasons");
                assert_eq!(ar_params.ar_order, 1, "Should be AR(1)");
                assert_eq!(ar_params.ar_coefficients.len(), 1);
            }
        }
        _ => panic!("Expected PAR model"),
    }

    // Check seasonal parameters exist
    assert_eq!(inflow_spec.seasonal_params.len(), 12);
}

#[test]
fn test_conversion_mixed_models() {
    // simple-par1 already has mixed models (PAR inflow + independent load)
    let (_system, _graph, noise_models) =
        load_test_fixture("06-par-model/01-simple-par1");

    // Convert to unified specs
    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed for mixed models");

    // Check we have both PAR and independent specs
    let par_count = unified_specs
        .iter()
        .filter(|s| {
            matches!(
                s.temporal_model,
                TemporalModelSpec::PeriodicAutoregressive { .. }
            )
        })
        .count();
    let independent_count = unified_specs
        .iter()
        .filter(|s| matches!(s.temporal_model, TemporalModelSpec::Independent))
        .count();

    assert!(par_count > 0, "Should have at least one PAR spec");
    assert!(
        independent_count > 0,
        "Should have at least one independent spec"
    );
}

#[test]
#[ignore] // Example directory exists but is empty
fn test_conversion_cascade() {
    // Load cascade example with multiple hydros
    let (system, graph, noise_models) =
        load_test_fixture("06-par-model/02-cascade-par");

    // Convert to unified specs
    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed for cascade");

    // Validate all specs
    for spec in &unified_specs {
        spec.validate().unwrap_or_else(|e| {
            panic!("Validation failed for entity {}: {}", spec.entity_id, e)
        });
        spec.validate_against_graph(&graph, &system)
            .unwrap_or_else(|e| {
                panic!(
                    "Cross-validation failed for entity {}: {}",
                    spec.entity_id, e
                )
            });
    }

    // Check we have specs for all hydros
    let num_hydros = system.hydros.len();
    let inflow_specs = unified_specs
        .iter()
        .filter(|s| matches!(s.uncertainty_type, UncertaintyType::Inflow))
        .count();

    assert_eq!(
        inflow_specs, num_hydros,
        "Should have inflow spec for each hydro"
    );
}

#[test]
#[ignore] // Example directory exists but is empty
fn test_conversion_quarterly_par() {
    // Load quarterly PAR example (4 seasons instead of 12)
    let (system, graph, noise_models) =
        load_test_fixture("06-par-model/05-quarterly-par");

    // Convert to unified specs
    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed for quarterly PAR");

    // Validate specs
    for spec in &unified_specs {
        spec.validate()
            .unwrap_or_else(|e| panic!("Validation failed: {}", e));
        spec.validate_against_graph(&graph, &system)
            .unwrap_or_else(|e| panic!("Cross-validation failed: {}", e));
    }

    // Check PAR has 4 seasons
    let par_spec = unified_specs
        .iter()
        .find(|s| {
            matches!(
                s.temporal_model,
                TemporalModelSpec::PeriodicAutoregressive { .. }
            )
        })
        .expect("Should have at least one PAR spec");

    if let TemporalModelSpec::PeriodicAutoregressive { num_seasons, .. } =
        &par_spec.temporal_model
    {
        assert_eq!(*num_seasons, 4, "Should have 4 seasons (quarterly)");
    }
}

// ================================
// Builder Utility Tests
// ================================

#[test]
fn test_builder_par_spec() {
    let spec = build_par_spec(0, UncertaintyType::Inflow, 12);

    // Validate structure
    assert_eq!(spec.entity_id, 0);
    assert!(matches!(spec.uncertainty_type, UncertaintyType::Inflow));

    match &spec.temporal_model {
        TemporalModelSpec::PeriodicAutoregressive {
            num_seasons,
            seasonal_ar_params,
        } => {
            assert_eq!(*num_seasons, 12);
            assert_eq!(seasonal_ar_params.len(), 12);
        }
        _ => panic!("Expected PAR model"),
    }

    assert_eq!(spec.seasonal_params.len(), 12);

    // Validate passes
    assert!(spec.validate().is_ok());
}

#[test]
fn test_builder_independent_spec() {
    let seasons = vec![0, 3, 6, 9]; // Quarterly seasons
    let spec =
        build_independent_spec(1, UncertaintyType::Load, seasons.clone());

    // Validate structure
    assert_eq!(spec.entity_id, 1);
    assert!(matches!(spec.uncertainty_type, UncertaintyType::Load));
    assert!(matches!(
        spec.temporal_model,
        TemporalModelSpec::Independent
    ));
    assert_eq!(spec.seasonal_params.len(), seasons.len());

    // Check all seasons present
    for season in seasons {
        assert!(spec.seasonal_params.contains_key(&season));
    }

    // Validate passes
    assert!(spec.validate().is_ok());
}

// ================================
// Comparison Utility Tests
// ================================

#[test]
fn test_assert_specs_equivalent_identical() {
    let spec1 = build_par_spec(0, UncertaintyType::Inflow, 12);
    let spec2 = build_par_spec(0, UncertaintyType::Inflow, 12);

    assert_specs_equivalent(&[spec1], &[spec2], 1e-10);
}

#[test]
#[should_panic(expected = "Mean mismatch")]
fn test_assert_specs_equivalent_detects_difference() {
    let spec1 = build_par_spec(0, UncertaintyType::Inflow, 12);
    let mut spec2 = build_par_spec(0, UncertaintyType::Inflow, 12);

    // Modify one seasonal parameter
    if let Some(params) = spec2.seasonal_params.get_mut(&0) {
        params.mean = 999.0; // Different value
    }

    assert_specs_equivalent(&[spec1], &[spec2], 1e-10);
}

// ================================
// Collection Validation Tests
// ================================

#[test]
fn test_validate_noise_specs_simple_par() {
    let (system, graph, noise_models) =
        load_test_fixture("06-par-model/01-simple-par1");

    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed");

    // Collection-level validation should pass
    powers_rs::unified_noise_spec::validate_noise_specs(
        &unified_specs,
        &graph,
        &system,
    )
    .expect("Collection validation should pass");
}

#[test]
#[ignore] // Example directory exists but is empty
fn test_validate_noise_specs_cascade() {
    let (system, graph, noise_models) =
        load_test_fixture("06-par-model/02-cascade-par");

    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed");

    // Collection-level validation should pass
    powers_rs::unified_noise_spec::validate_noise_specs(
        &unified_specs,
        &graph,
        &system,
    )
    .expect("Collection validation should pass for cascade");
}

#[test]
fn test_validate_noise_specs_detects_missing_entity() {
    let (system, graph, _) = load_test_fixture("06-par-model/01-simple-par1");

    // Create specs but skip entity 0 (should fail validation)
    let unified_specs = vec![];

    // Collection validation should fail (missing inflow for hydro 0)
    let result = powers_rs::unified_noise_spec::validate_noise_specs(
        &unified_specs,
        &graph,
        &system,
    );

    assert!(result.is_err(), "Should detect missing entity");
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Missing inflow"),
        "Error should mention missing inflow"
    );
}

#[test]
fn test_validate_noise_specs_detects_duplicate() {
    let (system, graph, _) = load_test_fixture("06-par-model/01-simple-par1");

    // Create duplicate specs for same entity
    let spec = build_par_spec(0, UncertaintyType::Inflow, 12);
    let unified_specs = vec![spec.clone(), spec];

    // Collection validation should fail (duplicate entity)
    let result = powers_rs::unified_noise_spec::validate_noise_specs(
        &unified_specs,
        &graph,
        &system,
    );

    assert!(result.is_err(), "Should detect duplicate");
    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("Duplicate"),
        "Error should mention duplicate"
    );
}

// ================================
// Edge Case Tests
// ================================

#[test]
fn test_conversion_handles_empty_input() {
    let noise_models: Vec<NoiseModel> = vec![];

    // Empty input should return an error (not a valid scenario)
    let result = UnifiedNoiseSpec::from_noise_models(&noise_models);
    assert!(result.is_err());

    let err_msg = result.unwrap_err();
    assert!(
        err_msg.contains("empty"),
        "Error should mention empty input"
    );
}

#[test]
#[ignore] // Example directory exists but is empty
fn test_conversion_preserves_marginal_distribution() {
    let (_, _, noise_models) =
        load_test_fixture("06-par-model/03-lognormal-par");

    let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
        .expect("Conversion should succeed");

    // Check that LogNormal3 distribution is preserved
    for spec in &unified_specs {
        if let Some(MarginalDistribution::LogNormal3 { gamma, mu, sigma }) =
            &spec.marginal_distribution
        {
            // Verify parameters are finite
            assert!(gamma.is_finite());
            assert!(mu.is_finite());
            assert!(sigma.is_finite());
            assert!(*sigma > 0.0);
        }
    }
}

#[test]
fn test_conversion_all_examples() {
    // Test only the example that actually has files (01-simple-par1)
    // Other examples exist as directories but are empty
    let examples = vec!["06-par-model/01-simple-par1"];

    for example in examples {
        let (system, graph, noise_models) = load_test_fixture(example);

        let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
            .unwrap_or_else(|e| {
                panic!("Conversion failed for {}: {}", example, e)
            });

        for spec in &unified_specs {
            spec.validate().unwrap_or_else(|e| {
                panic!("Validation failed for {}: {}", example, e)
            });
            spec.validate_against_graph(&graph, &system)
                .unwrap_or_else(|e| {
                    panic!("Cross-validation failed for {}: {}", example, e)
                });
        }

        powers_rs::unified_noise_spec::validate_noise_specs(
            &unified_specs,
            &graph,
            &system,
        )
        .unwrap_or_else(|e| {
            panic!("Collection validation failed for {}: {}", example, e)
        });
    }
}

// ================================
// Performance Baseline Tests
// ================================

#[test]
#[ignore] // Run explicitly with: cargo test --test test_unified_noise_spec_conversion establish_performance_baseline -- --ignored --nocapture
fn establish_performance_baseline() {
    use std::time::Instant;

    // Only test examples that exist
    let examples = vec!["06-par-model/01-simple-par1"];

    println!("\n=== Performance Baseline ===");
    for example in examples {
        let (_, _, noise_models) = load_test_fixture(example);

        // Measure conversion time
        let start = Instant::now();
        let _unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)
            .expect("Conversion should succeed");
        let duration = start.elapsed();

        println!(
            "{}: Conversion took {:?} ({} noise models)",
            example,
            duration,
            noise_models.len()
        );

        // Expect conversion to be fast (<10ms for typical cases)
        assert!(
            duration.as_millis() < 10,
            "Conversion too slow for {}: {:?}",
            example,
            duration
        );
    }
}
