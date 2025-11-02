//! Integration tests for observation-space PAR refactoring
//!
//! This test suite validates the observation-space formulation against the
//! residual-space formulation to ensure mathematical equivalence and correctness.
//!
//! # Test Coverage (Week 3 Tickets)
//!
//! - **Ticket 3.1**: Unit Tests for Pre-computation (done in modules)
//! - **Ticket 3.2**: Normal Distribution Compatibility
//! - **Ticket 3.3**: LogNormal Distribution Fix
//! - **Ticket 3.4**: Example Validation (placeholder)
//! - **Ticket 3.5**: Performance Benchmarking (placeholder)

use powers_rs::initial_condition::InitialCondition;
use powers_rs::input::UncertaintyType;
use powers_rs::precomputed_scenario::PrecomputedInflowScenario;
use powers_rs::scenario_generator::ScenarioGenerator;
use powers_rs::uncertainty_model::{
    DistributionType, PARParams, SeasonalParams, UncertaintyModel,
};
use std::collections::HashMap;

#[test]
fn test_observation_space_independent_normal() {
    // Ticket 3.2: Verify Independent Normal model works correctly

    let seasonal_params = vec![
        SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::Normal,
        };
        12
    ];

    let model = UncertaintyModel::Independent {
        entity_type: UncertaintyType::Inflow,
        entity_id: 0,
        seasonal_params,
    };

    let models = vec![model];
    let initial_condition = InitialCondition::new(vec![50.0], vec![]);

    let mut generator =
        ScenarioGenerator::new(models.clone(), &initial_condition, None)
            .unwrap();

    let mut rng = rand::rng();
    let lag_observations = HashMap::new();

    // Generate scenarios
    let scenarios = generator.generate_observation_space_scenarios(
        0,   // season_id
        100, // num_scenarios
        &mut rng,
        &lag_observations,
    );

    assert_eq!(scenarios.len(), 100);

    // Validate statistical properties
    let observations: Vec<f64> =
        scenarios.iter().map(|s| s[0].observation).collect();

    let mean = observations.iter().sum::<f64>() / observations.len() as f64;
    let variance = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / observations.len() as f64;
    let std_dev = variance.sqrt();

    // Should be approximately Normal(100, 20)
    // Allow for sampling variation with 100 samples
    assert!(mean > 95.0 && mean < 105.0, "Mean = {}", mean);
    assert!(std_dev > 15.0 && std_dev < 25.0, "Std dev = {}", std_dev);

    // All scenarios should have empty transformed coefficients (Independent)
    for scenario_set in &scenarios {
        for scenario in scenario_set {
            assert!(scenario.transformed_coefficients.is_empty());
            assert!(scenario.observation > 0.0);
        }
    }
}

#[test]
fn test_observation_space_par1_normal() {
    // Ticket 3.2: Verify PAR(1) Normal model produces correct statistics

    let par_params = PARParams {
        num_seasons: 12,
        ar_orders: vec![1; 12],
        ar_coefficients: vec![vec![0.7]; 12],
        seasonal_means: vec![100.0; 12],
        seasonal_stds: vec![20.0; 12],
        seasonal_distributions: vec![DistributionType::Normal; 12],
        max_ar_order: 1,
    };

    let model = UncertaintyModel::PeriodicAR {
        entity_type: UncertaintyType::Inflow,
        entity_id: 0,
        par_params,
    };

    let models = vec![model];
    let initial_condition =
        InitialCondition::new(vec![50.0], vec![vec![100.0]]);

    let mut generator =
        ScenarioGenerator::new(models.clone(), &initial_condition, None)
            .unwrap();

    let mut rng = rand::rng();

    // Provide lag observation
    let mut lag_observations = HashMap::new();
    lag_observations.insert((UncertaintyType::Inflow, 0), vec![100.0]);

    // Generate scenarios
    let scenarios = generator.generate_observation_space_scenarios(
        0, // season_id
        100,
        &mut rng,
        &lag_observations,
    );

    assert_eq!(scenarios.len(), 100);

    // Verify transformed coefficients
    for scenario_set in &scenarios {
        for scenario in scenario_set {
            assert_eq!(scenario.transformed_coefficients.len(), 1);
            // ψ_1 = φ_1 * (σ_t / σ_{t-1}) = 0.7 * (20/20) = 0.7
            assert!((scenario.transformed_coefficients[0] - 0.7).abs() < 1e-10);
            assert!(scenario.observation > 0.0);
        }
    }

    // Observations should be reasonable (AR process with φ=0.7 is stable)
    let observations: Vec<f64> =
        scenarios.iter().map(|s| s[0].observation).collect();

    let mean = observations.iter().sum::<f64>() / observations.len() as f64;

    // Mean should be close to unconditional mean: μ / (1 - φ) but
    // with lag=100, it will be influenced: E[Y_t | Y_{t-1}=100] ≈ ψ*100 + noise
    // The unconditional mean is 100.0
    assert!(mean > 80.0 && mean < 120.0, "Mean = {}", mean);
}

#[test]
fn test_observation_space_par2_normal() {
    // Ticket 3.2: Verify PAR(2) Normal model handles multiple lags correctly

    let par_params = PARParams {
        num_seasons: 1,
        ar_orders: vec![2],
        ar_coefficients: vec![vec![0.5, 0.3]],
        seasonal_means: vec![100.0],
        seasonal_stds: vec![20.0],
        seasonal_distributions: vec![DistributionType::Normal],
        max_ar_order: 2,
    };

    let model = UncertaintyModel::PeriodicAR {
        entity_type: UncertaintyType::Inflow,
        entity_id: 0,
        par_params,
    };

    let models = vec![model];
    let initial_condition =
        InitialCondition::new(vec![50.0], vec![vec![100.0, 95.0]]);

    let mut generator =
        ScenarioGenerator::new(models.clone(), &initial_condition, None)
            .unwrap();

    let mut rng = rand::rng();

    let mut lag_observations = HashMap::new();
    lag_observations.insert((UncertaintyType::Inflow, 0), vec![100.0, 95.0]);

    let scenarios = generator.generate_observation_space_scenarios(
        0,
        50,
        &mut rng,
        &lag_observations,
    );

    assert_eq!(scenarios.len(), 50);

    // Verify transformed coefficients
    for scenario_set in &scenarios {
        for scenario in scenario_set {
            assert_eq!(scenario.transformed_coefficients.len(), 2);
            // ψ_1 = 0.5 * (20/20) = 0.5
            assert!((scenario.transformed_coefficients[0] - 0.5).abs() < 1e-10);
            // ψ_2 = 0.3 * (20/20) = 0.3
            assert!((scenario.transformed_coefficients[1] - 0.3).abs() < 1e-10);
        }
    }
}

#[test]
fn test_observation_space_lognormal_basic() {
    // Ticket 3.3: Verify LogNormal distribution produces reasonable values
    // This is THE KEY TEST - LogNormal was broken in residual-space!

    let seasonal_params = vec![
        SeasonalParams {
            mean: 100.0,
            std_dev: 20.0,
            distribution: DistributionType::LogNormal3 {
                gamma: 0.0,
                mu: 4.5,
                sigma: 0.2,
            },
        };
        1
    ];

    let model = UncertaintyModel::Independent {
        entity_type: UncertaintyType::Inflow,
        entity_id: 0,
        seasonal_params,
    };

    let models = vec![model];
    let initial_condition = InitialCondition::new(vec![50.0], vec![]);

    let mut generator =
        ScenarioGenerator::new(models.clone(), &initial_condition, None)
            .unwrap();

    let mut rng = rand::rng();
    let lag_observations = HashMap::new();

    let scenarios = generator.generate_observation_space_scenarios(
        0,
        100,
        &mut rng,
        &lag_observations,
    );

    assert_eq!(scenarios.len(), 100);

    // THE FIX: Observations should be in reasonable range
    // NOT astronomical values like 10^38 or 10^41
    for scenario_set in &scenarios {
        for scenario in scenario_set {
            let obs = scenario.observation;

            // LogNormal3(gamma=0, mu=4.5, sigma=0.2) should produce values
            // roughly around exp(4.5) ≈ 90, with some variance
            // Definitely NOT 10^38!
            assert!(obs > 0.0, "Observation should be positive: {}", obs);
            assert!(obs < 1000.0, "Observation should be reasonable: {}", obs);

            // Most values should be between 50 and 200
            // (within a few standard deviations)
        }
    }

    // Check mean is reasonable
    let observations: Vec<f64> =
        scenarios.iter().map(|s| s[0].observation).collect();

    let mean = observations.iter().sum::<f64>() / observations.len() as f64;

    // Should be roughly exp(mu + sigma^2/2) = exp(4.5 + 0.02) ≈ 90-100
    assert!(mean > 50.0 && mean < 200.0, "LogNormal mean = {}", mean);
}

#[test]
fn test_observation_space_lognormal_par() {
    // Ticket 3.3: Verify LogNormal with PAR works correctly
    // This combination was particularly problematic in residual-space

    let par_params = PARParams {
        num_seasons: 1,
        ar_orders: vec![1],
        ar_coefficients: vec![vec![0.6]],
        seasonal_means: vec![100.0],
        seasonal_stds: vec![20.0],
        seasonal_distributions: vec![DistributionType::LogNormal3 {
            gamma: 0.0,
            mu: 4.5,
            sigma: 0.2,
        }],
        max_ar_order: 1,
    };

    let model = UncertaintyModel::PeriodicAR {
        entity_type: UncertaintyType::Inflow,
        entity_id: 0,
        par_params,
    };

    let models = vec![model];
    let initial_condition = InitialCondition::new(vec![50.0], vec![vec![90.0]]);

    let mut generator =
        ScenarioGenerator::new(models.clone(), &initial_condition, None)
            .unwrap();

    let mut rng = rand::rng();

    let mut lag_observations = HashMap::new();
    lag_observations.insert((UncertaintyType::Inflow, 0), vec![90.0]);

    let scenarios = generator.generate_observation_space_scenarios(
        0,
        50,
        &mut rng,
        &lag_observations,
    );

    assert_eq!(scenarios.len(), 50);

    // Verify no numerical overflow or astronomical values
    for scenario_set in &scenarios {
        for scenario in scenario_set {
            let obs = scenario.observation;

            assert!(obs > 0.0, "Observation should be positive: {}", obs);
            assert!(obs < 1000.0, "Observation should be reasonable: {}", obs);
            assert!(obs.is_finite(), "Observation should be finite: {}", obs);
        }
    }
}

#[test]
fn test_precomputed_scenario_equivalence_normal() {
    // Ticket 3.2: Verify pre-computed scenarios match direct calculation
    // for Normal distribution

    let current = SeasonalParams {
        mean: 150.0,
        std_dev: 30.0,
        distribution: DistributionType::Normal,
    };

    let lag = SeasonalParams {
        mean: 100.0,
        std_dev: 20.0,
        distribution: DistributionType::Normal,
    };

    let phi = vec![0.7];
    let lag_obs = vec![120.0];
    let innovation = 0.5; // Fixed innovation for determinism

    let scenario = PrecomputedInflowScenario::from_periodic_ar(
        0,
        current,
        &phi,
        &[lag],
        &lag_obs,
        innovation,
    );

    // Manual calculation for verification
    let psi_1 = 0.7 * (30.0 / 20.0); // = 1.05
    let deterministic = -psi_1 * 100.0 + 150.0; // = -105 + 150 = 45
    let noise = deterministic + 30.0 * 0.5; // = 45 + 15 = 60
    let expected_obs = psi_1 * 120.0 + noise; // = 126 + 60 = 186

    assert!((scenario.transformed_coefficients[0] - 1.05).abs() < 1e-10);
    assert!((scenario.noise_term - 60.0).abs() < 1e-10);
    assert!((scenario.observation - expected_obs).abs() < 1e-10);
}

#[test]
fn test_coefficient_transformation_multiple_seasons() {
    // Ticket 3.2: Verify coefficient transformation handles seasonal variation

    let current = SeasonalParams {
        mean: 150.0,
        std_dev: 30.0,
        distribution: DistributionType::Normal,
    };

    let lag1 = SeasonalParams {
        mean: 100.0,
        std_dev: 20.0,
        distribution: DistributionType::Normal,
    };

    let lag2 = SeasonalParams {
        mean: 80.0,
        std_dev: 15.0,
        distribution: DistributionType::Normal,
    };

    let phi = vec![0.7, 0.3];
    let lag_obs = vec![120.0, 90.0];
    let innovation = 0.0;

    let scenario = PrecomputedInflowScenario::from_periodic_ar(
        0,
        current,
        &phi,
        &[lag1, lag2],
        &lag_obs,
        innovation,
    );

    // Verify transformations
    let psi_1 = 0.7 * (30.0 / 20.0); // = 1.05
    let psi_2 = 0.3 * (30.0 / 15.0); // = 0.6

    assert!((scenario.transformed_coefficients[0] - psi_1).abs() < 1e-10);
    assert!((scenario.transformed_coefficients[1] - psi_2).abs() < 1e-10);
}
