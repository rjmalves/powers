//! Integration tests for LogNormal3 scenario generation
//!
//! Tests the 3-parameter log-normal transformation for non-negative
//! scenario generation using the 4-stage pipeline:
//! 1. Base Noise Generation (Z ~ N(0,1))
//! 2. Correlation Application (W = L×Z)
//! 3. Marginal Transformation (LogNormal3/Normal)
//! 4. (Not used in these tests - PAR dynamics)
//!
//! Verifies:
//! - Non-negativity guarantee with large sample sizes
//! - Correlation preservation through the pipeline
//! - Statistical properties (mean, variance)
//! - Mixed distributions (LogNormal3 inflows + Normal loads)

use nalgebra::DMatrix;
use powers_rs::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
use powers_rs::correlation_applicator::{
    CorrelationApplicator, CorrelationBlock, EntityRef, UncertaintyType,
};
use powers_rs::input::MarginalDistribution;
use powers_rs::lognormal3::LogNormal3Param;
use powers_rs::marginal_transformer::MarginalTransformer;
use std::collections::HashMap;

#[test]
fn test_lognormal3_ensures_nonnegativity() {
    // Test that LogNormal3 guarantees non-negative samples

    let dist = LogNormal3Param::new(1.0, 4.5, 0.3).expect("Valid parameters");

    // Generate 10,000 samples to ensure statistical significance
    let num_samples = 10_000;
    let mut all_positive = true;
    let mut min_value = f64::INFINITY;
    let mut max_value = f64::NEG_INFINITY;

    for i in 0..num_samples {
        // Use different z values to cover the distribution
        let z = (i as f64 / num_samples as f64 - 0.5) * 6.0; // Covers ~99.7% of normal distribution
        let sample = dist.sample(z);

        if sample < 0.0 {
            all_positive = false;
            eprintln!(
                "FAILURE: Negative sample at i={}: z={}, sample={}",
                i, z, sample
            );
        }

        min_value = min_value.min(sample);
        max_value = max_value.max(sample);
    }

    assert!(all_positive, "All samples must be non-negative");
    assert!(
        min_value >= 1.0,
        "Minimum value should be >= gamma=1.0, got {}",
        min_value
    );

    eprintln!(
        "✅ LogNormal3 non-negativity: {} samples, range [{:.2}, {:.2}]",
        num_samples, min_value, max_value
    );
}

#[test]
fn test_lognormal3_sample_always_positive_extreme_cases() {
    // Test non-negativity with extreme parameter values
    let test_cases = vec![
        ("small gamma", 0.1, 3.0, 0.5),
        ("zero gamma", 0.0, 4.0, 0.3),
        ("large gamma", 10.0, 5.0, 0.2),
        ("large sigma", 1.0, 4.0, 1.5),
        ("small sigma", 1.0, 4.0, 0.1),
    ];

    for (name, gamma, mu, sigma) in test_cases {
        let dist = LogNormal3Param::new(gamma, mu, sigma)
            .unwrap_or_else(|_| panic!("Valid parameters for {}", name));

        // Test with extreme z values
        let z_values = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        for z in z_values {
            let sample = dist.sample(z);
            assert!(
                sample >= 0.0,
                "{}: sample({}) = {} must be non-negative",
                name,
                z,
                sample
            );
            assert!(
                sample >= gamma,
                "{}: sample({}) = {} must be >= gamma={}",
                name,
                z,
                sample,
                gamma
            );
        }

        eprintln!(
            "✅ {}: gamma={}, mu={}, sigma={} - all positive",
            name, gamma, mu, sigma
        );
    }
}

#[test]
fn test_lognormal3_preserves_correlation() {
    // Test that LogNormal3 works correctly with the 4-stage pipeline
    // Validates correlation preservation through: Base Noise → Correlation → Marginal

    // Stage 1: Setup marginal distributions
    let marginals = vec![
        MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.3,
        },
        MarginalDistribution::LogNormal3 {
            gamma: 0.5,
            mu: 4.0,
            sigma: 0.4,
        },
    ];

    // Stage 2: Setup correlation structure
    // Correlation matrix: strong positive correlation (0.8)
    let correlation_matrix =
        DMatrix::from_row_slice(2, 2, &[1.0, 0.8, 0.8, 1.0]);

    let entities = vec![
        EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: 0,
        },
        EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: 1,
        },
    ];

    let block =
        CorrelationBlock::new(entities.clone(), correlation_matrix).unwrap();

    let entity_map: HashMap<EntityRef, usize> =
        entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let applicator = CorrelationApplicator::new(vec![block], entity_map);

    // Stage 3: Setup marginal transformer
    let transformer = MarginalTransformer::new(marginals).unwrap();

    // Generate samples through the pipeline
    let num_scenarios = 1_000;
    let num_entities = 2;
    let seed = 42;

    // Stage 1: Base noise generation
    let base_gen = BaseNoiseGenerator::new(num_scenarios, num_entities, seed);
    let base_samples = base_gen.generate(BaseNoiseMethod::Standard);

    // Stage 2: Apply correlation
    let correlated_samples = applicator.apply_correlation(&base_samples);

    // Stage 3: Apply marginal transformations
    let final_samples = transformer.transform_marginals(&correlated_samples);

    // Extract entity samples
    let mut samples1 = Vec::with_capacity(num_scenarios);
    let mut samples2 = Vec::with_capacity(num_scenarios);

    for scenario in &final_samples {
        samples1.push(scenario[0]);
        samples2.push(scenario[1]);
    }

    // Verify all samples are non-negative
    assert!(
        samples1.iter().all(|&x| x >= 0.0),
        "All samples1 must be non-negative"
    );
    assert!(
        samples2.iter().all(|&x| x >= 0.0),
        "All samples2 must be non-negative"
    );

    // Compute empirical correlation
    let mean1: f64 = samples1.iter().sum::<f64>() / num_scenarios as f64;
    let mean2: f64 = samples2.iter().sum::<f64>() / num_scenarios as f64;

    let mut covariance = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for i in 0..num_scenarios {
        let diff1 = samples1[i] - mean1;
        let diff2 = samples2[i] - mean2;
        covariance += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }

    let empirical_corr = covariance / (var1.sqrt() * var2.sqrt());

    eprintln!("Expected correlation: 0.8");
    eprintln!("Empirical correlation: {:.3}", empirical_corr);
    eprintln!("Mean1: {:.2}, Mean2: {:.2}", mean1, mean2);

    // Allow some tolerance due to sampling variability
    // With 1000 samples, standard error is ~0.03 for correlation
    assert!(
        (empirical_corr - 0.8).abs() < 0.1,
        "Empirical correlation {:.3} should be close to 0.8",
        empirical_corr
    );

    eprintln!(
        "✅ LogNormal3 correlation preserved through pipeline: {:.3} ≈ 0.8",
        empirical_corr
    );
}

#[test]
fn test_lognormal3_mixed_with_normal() {
    // Test mixed distributions: LogNormal3 for inflows, Normal for loads
    // This is the recommended pattern for SDDP applications using the 4-stage pipeline

    let marginals = vec![
        MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.3,
        },
        MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 10.0,
        },
    ];

    // Weak correlation between inflow and load
    let correlation_matrix =
        DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.3, 1.0]);

    let entities = vec![
        EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: 0,
        },
        EntityRef {
            uncertainty_type: UncertaintyType::Load,
            entity_id: 0,
        },
    ];

    let block =
        CorrelationBlock::new(entities.clone(), correlation_matrix).unwrap();

    let entity_map: HashMap<EntityRef, usize> =
        entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let applicator = CorrelationApplicator::new(vec![block], entity_map);
    let transformer = MarginalTransformer::new(marginals).unwrap();

    // Generate samples through the pipeline
    let num_scenarios = 1_000;
    let num_entities = 2;
    let seed = 123;

    let base_gen = BaseNoiseGenerator::new(num_scenarios, num_entities, seed);
    let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
    let correlated_samples = applicator.apply_correlation(&base_samples);
    let final_samples = transformer.transform_marginals(&correlated_samples);

    // Extract samples
    let mut inflow_samples = Vec::with_capacity(num_scenarios);
    let mut load_samples = Vec::with_capacity(num_scenarios);

    for scenario in &final_samples {
        inflow_samples.push(scenario[0]);
        load_samples.push(scenario[1]);
    }

    // Verify inflows are all non-negative (LogNormal3 guarantee)
    assert!(
        inflow_samples.iter().all(|&x| x >= 0.0),
        "All inflow samples must be non-negative"
    );

    // Compute load statistics
    let load_mean: f64 =
        load_samples.iter().sum::<f64>() / num_scenarios as f64;
    let load_std: f64 = (load_samples
        .iter()
        .map(|&x| (x - load_mean).powi(2))
        .sum::<f64>()
        / num_scenarios as f64)
        .sqrt();

    eprintln!("Inflow samples: all non-negative ✅");
    eprintln!("Load mean: {:.2} (expected ~100.0)", load_mean);
    eprintln!("Load std: {:.2} (expected ~10.0)", load_std);

    // Verify load statistics are close to expected
    assert!(
        (load_mean - 100.0).abs() < 2.0,
        "Load mean {:.2} should be close to 100.0",
        load_mean
    );
    assert!(
        (load_std - 10.0).abs() < 1.0,
        "Load std {:.2} should be close to 10.0",
        load_std
    );

    eprintln!("✅ Mixed distributions work correctly through pipeline");
}

/// Test that LogNormal3 sample moments match theoretical values.
///
/// # Theoretical Moments
///
/// For the 3-parameter log-normal distribution LN3(γ, μ, σ):
///
/// ```text
/// E[X] = γ + exp(μ + σ²/2)
/// Var[X] = exp(2μ + σ²) × (exp(σ²) - 1)
/// ```
///
/// # Test Parameters
/// - γ = 2.0, μ = 4.0, σ = 0.5
/// - 10,000 samples using Box-Muller transform
/// - Tolerance: 5% for mean, 10% for variance
#[test]
fn test_lognormal3_correct_moments() {
    let gamma = 2.0;
    let mu = 4.0;
    let sigma = 0.5;

    let dist =
        LogNormal3Param::new(gamma, mu, sigma).expect("Valid parameters");

    // Compute theoretical moments using formulas from doc comment
    let expected_mean = gamma + (mu + sigma * sigma / 2.0).exp();
    let expected_var =
        (2.0 * mu + sigma * sigma).exp() * ((sigma * sigma).exp() - 1.0);

    eprintln!("Theoretical mean: {:.2}", expected_mean);
    eprintln!("Theoretical variance: {:.2}", expected_var);

    // Generate samples using uniform random z values
    let num_samples = 10_000;
    let mut samples = Vec::with_capacity(num_samples);

    // Use a simple LCG for reproducible pseudo-random numbers
    let mut seed = 12345u64;
    for _ in 0..num_samples {
        // Simple LCG: x_{n+1} = (a * x_n + c) mod m
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u = (seed as f64) / (u64::MAX as f64);

        // Box-Muller transform to get normal random variable
        let u1 = u;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = (seed as f64) / (u64::MAX as f64);
        let z =
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        samples.push(dist.sample(z));
    }

    // Compute empirical mean and variance
    let empirical_mean: f64 = samples.iter().sum::<f64>() / num_samples as f64;
    let empirical_var: f64 = samples
        .iter()
        .map(|&x| (x - empirical_mean).powi(2))
        .sum::<f64>()
        / num_samples as f64;

    eprintln!("Empirical mean: {:.2}", empirical_mean);
    eprintln!("Empirical variance: {:.2}", empirical_var);

    // Allow 5% relative error due to sampling variability
    let mean_rel_error = (empirical_mean - expected_mean).abs() / expected_mean;
    let var_rel_error = (empirical_var - expected_var).abs() / expected_var;

    eprintln!("Mean relative error: {:.2}%", mean_rel_error * 100.0);
    eprintln!("Variance relative error: {:.2}%", var_rel_error * 100.0);

    assert!(
        mean_rel_error < 0.05,
        "Empirical mean {:.2} should be within 5% of theoretical {:.2}",
        empirical_mean,
        expected_mean
    );
    assert!(
        var_rel_error < 0.1,
        "Empirical variance {:.2} should be within 10% of theoretical {:.2}",
        empirical_var,
        expected_var
    );

    eprintln!("✅ LogNormal3 moments match theory");
}

#[test]
fn test_lognormal3_multiple_entities() {
    // Test scenario generation with multiple entities (n=10 hydros)
    // This simulates a realistic SDDP scenario with a cascade using the pipeline

    let num_entities = 10;
    let mut marginals = Vec::with_capacity(num_entities);

    // Create LogNormal3 marginals with varying parameters
    for i in 0..num_entities {
        marginals.push(MarginalDistribution::LogNormal3 {
            gamma: 0.5 + (i as f64) * 0.1, // Varying minimums
            mu: 4.0 + (i as f64) * 0.05,   // Varying means
            sigma: 0.3,                    // Same variability
        });
    }

    // Create a valid positive semi-definite correlation matrix
    // Use AR(1)-like structure with immediate neighbor correlation
    let mut corr_data = vec![0.0; num_entities * num_entities];
    for i in 0..num_entities {
        corr_data[i * num_entities + i] = 1.0;
        // Add moderate correlation with immediate neighbors
        if i > 0 {
            let rho = 0.5;
            corr_data[i * num_entities + (i - 1)] = rho;
            corr_data[(i - 1) * num_entities + i] = rho;
        }
    }
    let correlation_matrix =
        DMatrix::from_row_slice(num_entities, num_entities, &corr_data);

    // Setup entities and correlation block
    let entities: Vec<EntityRef> = (0..num_entities)
        .map(|i| EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: i,
        })
        .collect();

    let block =
        CorrelationBlock::new(entities.clone(), correlation_matrix).unwrap();

    let entity_map: HashMap<EntityRef, usize> =
        entities.iter().enumerate().map(|(i, &e)| (e, i)).collect();

    let applicator = CorrelationApplicator::new(vec![block], entity_map);
    let transformer = MarginalTransformer::new(marginals).unwrap();

    // Generate samples through the pipeline
    let num_scenarios = 100;
    let seed = 456;

    let base_gen = BaseNoiseGenerator::new(num_scenarios, num_entities, seed);
    let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
    let correlated_samples = applicator.apply_correlation(&base_samples);
    let final_samples = transformer.transform_marginals(&correlated_samples);

    // Verify all samples are non-negative
    for (scenario_idx, scenario) in final_samples.iter().enumerate() {
        for (entity_idx, &value) in scenario.iter().enumerate() {
            assert!(
                value >= 0.0,
                "Scenario {} Entity {} sample must be non-negative, got {}",
                scenario_idx,
                entity_idx,
                value
            );
        }
    }

    eprintln!(
        "✅ LogNormal3 with {} entities: {} samples, all non-negative",
        num_entities, num_scenarios
    );
}

#[test]
fn test_lognormal3_zero_gamma_edge_case() {
    // Test the edge case where gamma = 0 (no shift)
    // This should still guarantee X >= 0

    let dist = LogNormal3Param::new(0.0, 4.0, 0.5).expect("Valid parameters");

    // Generate samples with negative z values (tail of distribution)
    let z_values = vec![-5.0, -4.0, -3.0, -2.0, -1.0, 0.0];
    for z in z_values {
        let sample = dist.sample(z);
        assert!(
            sample >= 0.0,
            "With gamma=0, sample({}) = {} must still be non-negative",
            z,
            sample
        );
    }

    eprintln!("✅ LogNormal3 with gamma=0: all samples non-negative");
}

#[test]
fn test_lognormal3_inverse_cdf_nonnegativity() {
    // Test that inverse_cdf also produces non-negative values
    // This is used for correlation via copula approach

    let dist = LogNormal3Param::new(1.0, 4.5, 0.3).expect("Valid parameters");

    // Test with uniform samples across [0, 1]
    let u_values: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();

    for u in u_values {
        let sample = dist.inverse_cdf(u);
        assert!(
            sample >= 0.0,
            "inverse_cdf({}) = {} must be non-negative",
            u,
            sample
        );
        assert!(
            sample >= 1.0,
            "inverse_cdf({}) = {} must be >= gamma=1.0",
            u,
            sample
        );
    }

    eprintln!("✅ LogNormal3 inverse_cdf: all samples non-negative");
}
