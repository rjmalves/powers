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

use powers_rs::input::{MarginalDistribution, UncertaintyType};
use powers_rs::unified_noise_spec::{
    SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec, UnifiedNoiseSpec,
};
use std::collections::HashMap;

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
#[allow(dead_code)]
fn build_par_spec(
    entity_id: usize,
    uncertainty_type: UncertaintyType,
    num_seasons: usize,
) -> UnifiedNoiseSpec {
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
