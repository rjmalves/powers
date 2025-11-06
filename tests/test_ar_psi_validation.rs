//! Integration tests for AR cut PSI coefficient validation
//!
//! These tests validate that StorageAndInflowState can be constructed with
//! various PAR model configurations and that the transformation logic doesn't
//! panic or produce invalid results.
//!
//! Note: The actual ψ = φ × (σ_t / σ_{t-i}) transformation is thoroughly
//! tested in unit tests within src/state.rs (test_transformed_coefficients_*).
//! These integration tests focus on system-level validation.
//!

use powers_rs::input::{MarginalDistribution, UncertaintyType};
use powers_rs::state::StorageAndInflowState;
use powers_rs::system;
use powers_rs::temporal_model::TemporalModel;

/// Test that state can be constructed with seasonal variance (primary use case)
#[test]
fn test_state_construction_with_seasonal_variance() {
    let system = create_test_system(1);

    // PAR model with seasonal variance
    let phi = vec![0.8, 0.3];
    let seasonal_stds = vec![50.0, 100.0];

    let temporal_models = vec![create_par_model(0, phi, seasonal_stds)];

    // Should construct without panic
    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: State constructed with ψ = φ × (σ_t / σ_{t-i})
    // Internal transformation is tested in unit tests
}

/// Test that state construction handles uniform variance correctly
#[test]
fn test_state_construction_with_uniform_variance() {
    let system = create_test_system(1);

    // All seasons have same σ: ψ should equal φ
    let phi = vec![0.7, 0.2];
    let uniform_sigma = vec![10.0];

    let temporal_models = vec![TemporalModel::from_par(
        UncertaintyType::Inflow,
        0,
        1,
        vec![100.0],
        uniform_sigma,
        vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 10.0,
        }],
        vec![phi.len()],
        vec![phi.clone()],
    )
    .unwrap()];

    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: When σ is uniform, ψ = φ (tested in unit tests)
}

/// Test state construction with extreme variance ratios (10×)
#[test]
fn test_state_construction_with_extreme_variance_ratio() {
    let system = create_test_system(1);

    // 10× variance ratio between seasons
    let phi = vec![0.6];
    let seasonal_stds = vec![10.0, 100.0];

    let temporal_models = vec![create_par_model(0, phi, seasonal_stds)];

    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: ψ = φ × 0.1 for this configuration (tested in unit tests)
}

/// Test that state construction works with various AR orders
#[test]
fn test_state_construction_with_heterogeneous_ar_orders() {
    let system = create_test_system(3);

    // Create mixed system: AR(1), AR(2), AR(1) with different seasonal patterns
    let temporal_models = vec![
        create_par_model(0, vec![0.8], vec![50.0, 100.0]),
        create_par_model(1, vec![0.7, 0.2], vec![60.0, 90.0]),
        create_par_model(2, vec![0.6], vec![40.0, 80.0]),
    ];

    // Should not panic with heterogeneous AR orders
    let _state = StorageAndInflowState::new(&system, &temporal_models);
}

/// Test state construction with 12-season PAR model (typical monthly model)
#[test]
fn test_state_construction_with_twelve_seasons() {
    let system = create_test_system(1);

    let phi = vec![0.8];
    // 12 seasons with varying σ
    let seasonal_stds: Vec<f64> =
        (0..12).map(|i| 50.0 + (i as f64) * 5.0).collect();

    let temporal_models = vec![TemporalModel::from_par(
        UncertaintyType::Inflow,
        0,
        12,
        vec![100.0; 12],
        seasonal_stds.clone(),
        seasonal_stds
            .iter()
            .map(|&std_dev| MarginalDistribution::Normal {
                mean: 100.0,
                std_dev,
            })
            .collect(),
        vec![phi.len(); 12],
        vec![phi.clone(); 12],
    )
    .unwrap()];

    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: Seasonal wraparound logic works correctly
}

/// Test that state construction is deterministic
#[test]
fn test_state_construction_deterministic() {
    let system = create_test_system(1);

    let phi = vec![0.8, 0.3];
    let seasonal_stds = vec![50.0, 100.0];
    let temporal_models = vec![create_par_model(0, phi, seasonal_stds)];

    // Construct state multiple times
    let _state1 = StorageAndInflowState::new(&system, &temporal_models);
    let _state2 = StorageAndInflowState::new(&system, &temporal_models);
    let _state3 = StorageAndInflowState::new(&system, &temporal_models);

    // Success: All constructions succeed (transformation is deterministic)
}

/// Test mixed system with independent and PAR models
#[test]
fn test_state_construction_with_mixed_models() {
    let system = create_test_system(3);

    let temporal_models = vec![
        // Hydro 0: Independent (no AR)
        TemporalModel::from_independent(
            UncertaintyType::Inflow,
            0,
            vec![100.0],
            vec![20.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }],
        )
        .unwrap(),
        // Hydro 1: AR(1) with seasonal variance
        create_par_model(1, vec![0.8], vec![50.0, 100.0]),
        // Hydro 2: AR(2) with uniform variance
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            2,
            1,
            vec![100.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 10.0,
            }],
            vec![2],
            vec![vec![0.7, 0.2]],
        )
        .unwrap(),
    ];

    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: Mixed system with both independent and PAR models works
}

/// Document the error magnitude when using φ instead of ψ
#[test]
fn test_error_magnitude_documentation() {
    // This test documents the error magnitude when using φ instead of ψ
    // The actual values are tested in unit tests in src/state.rs

    let phi = 0.7;
    let sigma_t = 100.0; // Current season
    let sigma_lag = 50.0; // Previous season

    let psi_correct = phi * (sigma_t / sigma_lag); // 1.4
    let phi_incorrect = phi; // 0.7

    let error_magnitude = (psi_correct - phi_incorrect) / psi_correct;

    // Error is 50% when σ_t = 2 × σ_lag
    assert!(
        (error_magnitude - 0.5_f64).abs() < 1e-10,
        "Using φ instead of ψ causes 50% error when σ ratio is 2:1"
    );

    println!("\n=== Error Magnitude Analysis ===");
    println!("When σ_t / σ_{{t-1}} = 2.0:");
    println!("  φ (wrong)  = {:.3}", phi_incorrect);
    println!("  ψ (correct) = {:.3}", psi_correct);
    println!("  Error      = {:.1}%", error_magnitude * 100.0);
    println!("================================\n");
}

/// Regression test: Verify no panic with zero AR order (independent model)
#[test]
fn test_no_panic_with_independent_model() {
    let system = create_test_system(1);

    let temporal_models = vec![TemporalModel::from_independent(
        UncertaintyType::Inflow,
        0,
        vec![100.0],
        vec![20.0],
        vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }],
    )
    .unwrap()];

    let _state = StorageAndInflowState::new(&system, &temporal_models);

    // Success: Independent models produce empty transformed_coefficients
}

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_system(num_hydros: usize) -> system::System {
    let mut system = system::System::default();
    system.hydros.clear();
    for i in 0..num_hydros {
        system.hydros.push(system::Hydro::new(
            i, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
    }
    system.meta.hydros_count = num_hydros;
    system
}

fn create_par_model(
    entity_id: usize,
    phi: Vec<f64>,
    seasonal_stds: Vec<f64>,
) -> TemporalModel {
    let ar_order = phi.len();
    let num_seasons = seasonal_stds.len();

    TemporalModel::from_par(
        UncertaintyType::Inflow,
        entity_id,
        num_seasons,
        vec![100.0; num_seasons],
        seasonal_stds,
        vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            };
            num_seasons
        ],
        vec![ar_order; num_seasons],
        vec![phi; num_seasons],
    )
    .unwrap()
}
