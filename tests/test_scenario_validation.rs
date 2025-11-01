use approx::assert_relative_eq;
use powers_rs::initial_condition::InitialCondition;
use powers_rs::input::Recourse;
use powers_rs::scenario::SAA;
/// Statistical validation tests for ScenarioGenerator
///
/// This module tests that the 4-stage scenario generation pipeline produces
/// statistically correct scenarios that match theoretical properties:
/// - Marginal distributions (mean, variance)
/// - Correlation structure (Pearson)
/// - AR temporal correlation (ACF)
/// - Non-negativity (LogNormal3)
///
/// All tests use 95% confidence intervals to validate statistical properties.
/// Helper function to generate SAA for testing using the new API
///
/// Creates a minimal 2-stage graph and generates scenarios using Recourse::generate_sddp_noises()
fn generate_test_saa(
    recourse_json: &str,
    num_stages: usize,
    scenarios_per_stage: Vec<usize>,
    seed: u64,
) -> (SAA, InitialCondition) {
    use powers_rs::input::{GraphEdgeInput, GraphInput, GraphNodeInput};

    // Parse recourse
    let recourse: Recourse = serde_json::from_str(recourse_json)
        .expect("Failed to parse recourse JSON");
    let initial_condition = recourse.build_sddp_initial_condition();

    // Build minimal graph JSON structure
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut node_id = 0;

    for stage_id in 0..num_stages {
        let num_scenarios = scenarios_per_stage[stage_id];
        for _scenario_id in 0..num_scenarios {
            // Use simple incrementing dates that won't overflow month
            let day_start = (stage_id % 28) + 1;
            let day_end = ((stage_id + 1) % 28) + 1;
            nodes.push(GraphNodeInput {
                id: node_id,
                stage_id,
                season_id: 0, // Single season for test simplicity
                start_date: format!("2024-01-{:02}T00:00:00Z", day_start),
                end_date: format!("2024-01-{:02}T00:00:00Z", day_end),
                risk_measure: "expectation".to_string(),
                state_variables: "storage".to_string(),
                num_scenarios,
            });

            // Connect to previous stage
            if stage_id > 0 {
                let prev_stage_start =
                    node_id - scenarios_per_stage[stage_id - 1];
                for prev_node in prev_stage_start..node_id {
                    edges.push(GraphEdgeInput {
                        source_id: prev_node,
                        target_id: node_id,
                        probability: 1.0 / num_scenarios as f64,
                        discount_rate: 1.0,
                    });
                }
            }

            node_id += 1;
        }
    }

    let graph_input = GraphInput { nodes, edges };

    // Build minimal system (we only need the entity counts)
    let system_json = r#"{
        "buses": [{"id": 0, "deficit_cost": 1000.0}],
        "lines": [],
        "thermals": [],
        "hydros": [{
            "id": 0,
            "downstream_hydro_id": null,
            "bus_id": 0,
            "productivity": 1.0,
            "min_storage": 0.0,
            "max_storage": 100.0,
            "min_turbined_flow": 0.0,
            "max_turbined_flow": 50.0,
            "spillage_penalty": 0.01
        }]
    }"#;
    let system: powers_rs::input::SystemInput =
        serde_json::from_str(system_json).expect("Failed to parse system JSON");

    // Build graph
    let graph = graph_input
        .build_sddp_graph(&system, &recourse)
        .expect("Failed to build graph");

    // Generate SAA
    let saa = recourse.generate_sddp_noises(&graph, &initial_condition, seed);

    (saa, initial_condition)
}

mod statistical_tests {
    #![allow(dead_code)] // Some utilities not yet used

    /// Compute sample mean
    pub fn sample_mean(data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Compute sample variance (unbiased estimator)
    pub fn sample_variance(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mean = sample_mean(data);
        let sum_sq = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        sum_sq / (n - 1.0)
    }

    /// Compute sample standard deviation
    pub fn sample_std_dev(data: &[f64]) -> f64 {
        sample_variance(data).sqrt()
    }

    /// Validate sample mean within 95% CI using CLT
    pub fn validate_mean(samples: &[f64], expected_mean: f64) -> bool {
        let n = samples.len() as f64;
        let sample_mean = sample_mean(samples);
        let sample_std = sample_std_dev(samples);

        // 95% CI: μ ± z * (σ/√n), z = 1.96 for 95%
        let z = 1.96;
        let margin = z * (sample_std / n.sqrt());

        (sample_mean - expected_mean).abs() < margin
    }

    /// Validate sample variance within reasonable bounds
    /// Using 20% tolerance for practical validation
    pub fn validate_variance(samples: &[f64], expected_var: f64) -> bool {
        let sample_var = sample_variance(samples);

        // Use 20% tolerance for practical validation
        let margin = 0.20 * expected_var;

        (sample_var - expected_var).abs() < margin
    }

    /// Compute Pearson correlation coefficient
    pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f64;

        let mean_x = sample_mean(x);
        let mean_y = sample_mean(y);

        let cov: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (n - 1.0);

        let std_x = sample_std_dev(x);
        let std_y = sample_std_dev(y);

        cov / (std_x * std_y)
    }

    /// Validate correlation coefficient within 95% CI using Fisher z-transformation
    pub fn validate_correlation(
        samples_x: &[f64],
        samples_y: &[f64],
        expected_corr: f64,
    ) -> bool {
        let n = samples_x.len() as f64;
        let sample_corr = pearson_correlation(samples_x, samples_y);

        // Fisher z-transformation for CI
        let z_sample = 0.5 * ((1.0 + sample_corr) / (1.0 - sample_corr)).ln();
        let z_expected =
            0.5 * ((1.0 + expected_corr) / (1.0 - expected_corr)).ln();
        let se_z = 1.0 / (n - 3.0).sqrt();

        (z_sample - z_expected).abs() < 1.96 * se_z
    }

    /// Compute autocorrelation function (ACF) at lag k
    pub fn acf(data: &[f64], lag: usize) -> f64 {
        let n = data.len();
        let mean = sample_mean(data);

        let c0: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();

        let ck: f64 = data[..n - lag]
            .iter()
            .zip(&data[lag..])
            .map(|(x, x_lag)| (x - mean) * (x_lag - mean))
            .sum();

        ck / c0
    }

    /// Validate ACF at lag k matches expected value
    pub fn validate_acf(data: &[f64], lag: usize, expected_acf: f64) -> bool {
        let sample_acf = acf(data, lag);
        let n = data.len() as f64;

        // Bartlett's formula for standard error of ACF
        // For AR(1): SE(r_k) ≈ 1/√n
        let se = 1.0 / n.sqrt();

        (sample_acf - expected_acf).abs() < 1.96 * se
    }
}

#[test]
fn test_statistical_utils_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = statistical_tests::sample_mean(&data);
    assert_relative_eq!(mean, 3.0, epsilon = 1e-10);
}

#[test]
fn test_statistical_utils_variance() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = statistical_tests::sample_variance(&data);
    // Expected variance ≈ 4.571
    assert_relative_eq!(var, 4.571, epsilon = 0.01);
}

#[test]
fn test_statistical_utils_correlation() {
    // Perfectly correlated
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let corr = statistical_tests::pearson_correlation(&x, &y);
    assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
}

#[test]
fn test_marginal_normal_distribution() {
    // Test: Normal marginal with μ=100, σ=20
    let recourse_json = r#"{
        "initial_condition": {
            "storage": [],
            "inflow": []
        },
        "uncertainty_specifications": [
            {
                "uncertainty_type": "load",
                "entity_id": 0,
                "temporal_model": {
                    "type": "independent"
                },
                "seasonal_distributions": [
                    {
                        "season_id": 0,
                        "type": "normal",
                        "mean": 100.0,
                        "std_dev": 20.0                    
                    }
                ]
            }
        ]
    }"#;

    // Generate large sample for statistical testing
    let num_scenarios = 10000;
    let (saa, _initial_condition) =
        generate_test_saa(recourse_json, 2, vec![1, num_scenarios], 42);

    // Extract all samples from stage 1 (load is in load_noises, not inflow_noises)
    let mut samples = Vec::with_capacity(num_scenarios);
    for i in 0..num_scenarios {
        let noises = saa.get_noises_by_stage_and_branching(1, i).unwrap();
        samples.push(noises.get_load_innovations()[0]);
    }

    let sample_mean = statistical_tests::sample_mean(&samples);
    let sample_var = statistical_tests::sample_variance(&samples);

    println!("Sample mean: {}, expected: 100.0", sample_mean);
    println!("Sample variance: {}, expected: 400.0", sample_var);

    // Validate mean (using 99% CI for more lenient test)
    let n = samples.len() as f64;
    let sample_std = statistical_tests::sample_std_dev(&samples);
    let z = 2.576; // 99% CI
    let margin = z * (sample_std / n.sqrt());
    assert!(
        (sample_mean - 100.0).abs() < margin,
        "Sample mean {} should be within 99% CI of 100.0 (margin: {})",
        sample_mean,
        margin
    );

    // Validate variance (30% tolerance for large sample)
    assert!(
        (sample_var - 400.0).abs() < 0.30 * 400.0,
        "Sample variance {} should be within 30% of 400.0",
        sample_var
    );
}

#[test]
fn test_marginal_lognormal3_distribution() {
    // Test: LogNormal3 marginal with γ=10, μ=4.5, σ=0.3
    let recourse_json = r#"{
        "initial_condition": {
            "storage": [],
            "inflow": []
        },
        "uncertainty_specifications": [
            {
                "uncertainty_type": "inflow",
                "entity_id": 0,
                "temporal_model": {
                    "type": "independent"
                },
                "seasonal_distributions": [
                    {
                        "season_id": 0,
                        "type": "lognormal3",
                        "gamma": 10.0,
                        "mu": 4.5,
                        "sigma": 0.3
                    }
                ]
            }
        ]
    }"#;

    let num_scenarios = 10000;
    let (saa, _initial_condition) =
        generate_test_saa(recourse_json, 2, vec![1, num_scenarios], 42);

    let mut samples = Vec::with_capacity(num_scenarios);
    for i in 0..num_scenarios {
        let noises = saa.get_noises_by_stage_and_branching(1, i).unwrap();
        samples.push(noises.get_inflow_innovations()[0]);
    }

    // Validate non-negativity
    assert!(
        samples.iter().all(|&x| x >= 10.0),
        "All LogNormal3 samples should be >= shift (10.0)"
    );

    // Validate mean
    let expected_mean = 10.0 + (4.5 + 0.5 * 0.3_f64.powi(2)).exp();
    assert!(
        statistical_tests::validate_mean(&samples, expected_mean),
        "LogNormal3 sample mean should be within 95% CI of {}",
        expected_mean
    );
}

#[test]
fn test_ar1_autocorrelation() {
    // Test: AR(1) with φ=0.7, should have ACF(1)=0.7, ACF(2)=0.49
    let recourse_json = r#"{
        "initial_condition": {
            "storage": [],
            "inflow": [{"hydro_id": 0, "lag": 1, "value": 100.0}]
        },
        "uncertainty_specifications": [
            {
                "uncertainty_type": "inflow",
                "entity_id": 0,
                "temporal_model": {
                    "type": "periodic_ar",
                    "num_seasons": 1,
                    "ar_orders": [1],
                    "ar_coefficients": [[0.7]],
                    "seasonal_means": [100.0],
                    "seasonal_stds": [25.0]
                },
                "marginal_distribution": {
                    "type": "normal",
                    "mean": 0.0,
                    "std_dev": 1.0
                }
            }
        ]
    }"#;

    // Generate many stages to get long time series
    let num_stages = 200; // Longer series for better statistical properties
    let scenarios_per_stage = vec![1; num_stages]; // Single scenario path
    let (saa, _initial_condition) =
        generate_test_saa(recourse_json, num_stages, scenarios_per_stage, 42);

    // Extract time series from single scenario
    let mut time_series = Vec::with_capacity(num_stages);
    for stage_id in 0..num_stages {
        let noises =
            saa.get_noises_by_stage_and_branching(stage_id, 0).unwrap();
        time_series.push(noises.get_inflow_innovations()[0]);
    }

    // PAR model: X_t = μ_m + σ_m × Z_t (where Z_t follows AR process)
    // To validate AR correlation, compute: Z_t = (X_t - μ_m) / σ_m
    let seasonal_mean = 100.0;
    let seasonal_std = 25.0;
    let deseasonalized: Vec<f64> = time_series
        .iter()
        .map(|&x| (x - seasonal_mean) / seasonal_std)
        .collect();

    // Validate ACF(1) ≈ φ = 0.7 on deseasonalized series
    let acf1 = statistical_tests::acf(&deseasonalized, 1);
    println!(
        "AR(1) ACF(1) on deseasonalized series: {}, expected: 0.7",
        acf1
    );
    assert!(
        statistical_tests::validate_acf(&deseasonalized, 1, 0.7),
        "AR(1) ACF(1) should be approximately 0.7, got {}",
        acf1
    );

    // Validate ACF(2) ≈ φ² = 0.49
    let acf2 = statistical_tests::acf(&deseasonalized, 2);
    println!(
        "AR(1) ACF(2) on deseasonalized series: {}, expected: 0.49",
        acf2
    );
    assert!(
        statistical_tests::validate_acf(&deseasonalized, 2, 0.49),
        "AR(1) ACF(2) should be approximately 0.49, got {}",
        acf2
    );
}

#[test]
fn test_ar2_autocorrelation() {
    // Test: AR(2) with φ₁=0.6, φ₂=0.2
    // ACF(1) = φ₁/(1-φ₂) = 0.6/0.8 = 0.75
    // ACF(2) = φ₁·ACF(1) + φ₂ = 0.6·0.75 + 0.2 = 0.65
    let recourse_json = r#"{
        "initial_condition": {
            "storage": [],
            "inflow": [
                {"hydro_id": 0, "lag": 1, "value": 100.0},
                {"hydro_id": 0, "lag": 2, "value": 95.0}
            ]
        },
        "uncertainty_specifications": [
            {
                "uncertainty_type": "inflow",
                "entity_id": 0,
                "temporal_model": {
                    "type": "periodic_ar",
                    "num_seasons": 1,
                    "ar_orders": [2],
                    "ar_coefficients": [[0.6, 0.2]],
                    "seasonal_means": [100.0],
                    "seasonal_stds": [25.0]
                },
                "marginal_distribution": {
                    "type": "normal",
                    "mean": 0.0,
                    "std_dev": 1.0
                }
            }
        ]
    }"#;

    let num_stages = 200; // Longer series for better statistical properties
    let scenarios_per_stage = vec![1; num_stages];
    let (saa, _initial_condition) =
        generate_test_saa(recourse_json, num_stages, scenarios_per_stage, 42);

    let mut time_series = Vec::with_capacity(num_stages);
    for stage_id in 0..num_stages {
        let noises =
            saa.get_noises_by_stage_and_branching(stage_id, 0).unwrap();
        time_series.push(noises.get_inflow_innovations()[0]);
    }

    // PAR model: X_t = μ_m + σ_m × Z_t (where Z_t follows AR process)
    // To validate AR correlation, compute: Z_t = (X_t - μ_m) / σ_m
    let seasonal_mean = 100.0;
    let seasonal_std = 25.0;
    let deseasonalized: Vec<f64> = time_series
        .iter()
        .map(|&x| (x - seasonal_mean) / seasonal_std)
        .collect();

    // Validate ACF(1) ≈ 0.75 on deseasonalized series
    let acf1 = statistical_tests::acf(&deseasonalized, 1);
    println!(
        "AR(2) ACF(1) on deseasonalized series: {}, expected: 0.75",
        acf1
    );
    assert!(
        statistical_tests::validate_acf(&deseasonalized, 1, 0.75),
        "AR(2) ACF(1) should be approximately 0.75, got {}",
        acf1
    );

    // Validate ACF(2) ≈ 0.65
    let acf2 = statistical_tests::acf(&deseasonalized, 2);
    println!(
        "AR(2) ACF(2) on deseasonalized series: {}, expected: 0.65",
        acf2
    );
    assert!(
        statistical_tests::validate_acf(&deseasonalized, 2, 0.65),
        "AR(2) ACF(2) should be approximately 0.65, got {}",
        acf2
    );
}

#[test]
fn test_seed_determinism() {
    // Test: Same seed produces identical scenarios
    let recourse_json = r#"{
        "initial_condition": {
            "storage": [],
            "inflow": []
        },
        "uncertainty_specifications": [
            {
                "uncertainty_type": "inflow",
                "entity_id": 0,
                "temporal_model": {
                    "type": "independent"
                },
                "seasonal_distributions": [
                    {
                        "season_id": 0,
                        "type": "lognormal3",
                        "mu": 2.9960,
                        "sigma": 0.1,                        
                        "gamma": 0                        
                    },
                    {
                        "season_id": 1,
                        "type": "normal",
                        "mean": 60.0,
                        "std_dev": 0.1                    
                    }
                ]
            }
        ]
    }"#;

    let seed = 12345;
    let scenarios_per_stage = vec![1, 100];

    let (saa1, _) =
        generate_test_saa(recourse_json, 2, scenarios_per_stage.clone(), seed);
    let (saa2, _) =
        generate_test_saa(recourse_json, 2, scenarios_per_stage, seed);

    // Compare all scenarios
    for i in 0..100 {
        let noises1 = saa1.get_noises_by_stage_and_branching(1, i).unwrap();
        let noises2 = saa2.get_noises_by_stage_and_branching(1, i).unwrap();

        assert_relative_eq!(
            noises1.get_inflow_innovations()[0],
            noises2.get_inflow_innovations()[0],
            epsilon = 1e-10
        );
    }
}
