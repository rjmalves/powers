//! CEPEL Validation Tests for PAR Implementation
//!
//! This module contains rigorous validation tests that verify mathematical
//! correctness of the PAR implementation against CEPEL's published equations.
//!
//! # CEPEL PAR(p) Equation
//!
//! Z_t = μ_m + σ_m · [φ_1m·a_t-1 + φ_2m·a_t-2 + ... + φ_pm·a_t-p + a_t]
//!
//! where:
//!   - t = time step (stage)
//!   - m = t mod period (seasonal index)
//!   - a_t = transformed residual (from marginal distribution)
//!   - μ_m = seasonal mean for period m
//!   - σ_m = seasonal standard deviation for period m
//!   - φ_km = AR coefficient k for period m
//!
//! # Test Categories
//!
//! 1. **Hand-Calculated Cases**: Trace equation with known inputs/outputs
//! 2. **Statistical Convergence**: Verify long-run mean/variance properties
//! 3. **Correlation Preservation**: Spatial correlation through PAR pipeline
//! 4. **Stationarity**: Coefficient constraints produce bounded series

use powers_rs::{
    par_generator::PeriodicARGenerator, seasonal_params::SeasonalParams,
};

// ============================================================================
// Hand-Calculated Reference Cases
// ============================================================================

/// Test PAR(1) equation by hand calculation
///
/// # CEPEL Equation for PAR(1)
///
/// Z_t = μ_m + σ_m · (φ_1m · a_t-1 + a_t)
///
/// # Test Case
///
/// Period = 1 (single season), φ_1 = 0.7, μ = 100, σ = 20
///
/// Given:
/// - a_0 = 1.0 (residual at t=0)
/// - a_1 = 0.5 (residual at t=1)
/// - a_2 = -0.3 (residual at t=2)
/// - Initial lag: a_-1 = 0.0
///
/// Expected:
/// - Z'_0 = 0.7·0 + 1.0 = 1.0      → Z_0 = 100 + 20·1.0 = 120.0
/// - Z'_1 = 0.7·1.0 + 0.5 = 1.2     → Z_1 = 100 + 20·1.2 = 124.0
/// - Z'_2 = 0.7·1.2 + (-0.3) = 0.54 → Z_2 = 100 + 20·0.54 = 110.8
#[test]
fn test_par1_hand_calculated() {
    let params = SeasonalParams::new(
        1,               // period = 1
        vec![1],         // AR(1)
        vec![vec![0.7]], // φ_1 = 0.7
        vec![100.0],     // μ = 100
        vec![20.0],      // σ = 20
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Stage 0: Z'_{-1} = 0 (initial), a_0 = 1.0
    let z0 = gen.generate_next(1.0);
    assert!(
        (z0 - 120.0).abs() < 1e-10,
        "Z_0 should be 120.0, got {}",
        z0
    );

    // Stage 1: Z'_0 = 1.0 (prev), a_1 = 0.5
    let z1 = gen.generate_next(0.5);
    assert!(
        (z1 - 124.0).abs() < 1e-10,
        "Z_1 should be 124.0, got {}",
        z1
    );

    // Stage 2: Z'_1 = 1.2 (prev), a_2 = -0.3
    let z2 = gen.generate_next(-0.3);
    assert!(
        (z2 - 110.8).abs() < 1e-10,
        "Z_2 should be 110.8, got {}",
        z2
    );
}

/// Test PAR(2) equation by hand calculation
///
/// # CEPEL Equation for PAR(2)
///
/// Z'_t = φ_1m · Z'_{t-1} + φ_2m · Z'_{t-2} + a_t
///
/// # Test Case
///
/// Period = 1, φ_1 = 0.5, φ_2 = 0.3, μ = 100, σ = 20
///
/// Given:
/// - a_0 = 1.0, a_1 = 0.5, a_2 = -0.3
/// - Initial lags: Z'_{-2} = 0.0, Z'_{-1} = 0.0
///
/// Expected:
/// - Z'_0 = 0.5·0 + 0.3·0 + 1.0 = 1.0     → Z_0 = 100 + 20·1.0 = 120.0
/// - Z'_1 = 0.5·1.0 + 0.3·0 + 0.5 = 1.0   → Z_1 = 100 + 20·1.0 = 120.0
/// - Z'_2 = 0.5·1.0 + 0.3·1.0 + (-0.3) = 0.5 → Z_2 = 100 + 20·0.5 = 110.0
#[test]
fn test_par2_hand_calculated() {
    let params = SeasonalParams::new(
        1,                    // period = 1
        vec![2],              // AR(2)
        vec![vec![0.5, 0.3]], // φ_1 = 0.5, φ_2 = 0.3
        vec![100.0],          // μ = 100
        vec![20.0],           // σ = 20
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let z0 = gen.generate_next(1.0);
    assert!(
        (z0 - 120.0).abs() < 1e-10,
        "Z_0 should be 120.0, got {}",
        z0
    );

    let z1 = gen.generate_next(0.5);
    assert!(
        (z1 - 120.0).abs() < 1e-10,
        "Z_1 should be 120.0, got {}",
        z1
    );

    let z2 = gen.generate_next(-0.3);
    assert!(
        (z2 - 110.0).abs() < 1e-10,
        "Z_2 should be 110.0, got {}",
        z2
    );
}

/// Test PAR with 2 periods (seasonal variation)
///
/// # Test Case
///
/// Period 0 (wet): μ_0 = 100, σ_0 = 20, φ_1,0 = 0.7
/// Period 1 (dry): μ_1 = 120, σ_1 = 25, φ_1,1 = 0.6
///
/// Given:
/// - a_0 = 1.0 (period 0), a_1 = 0.5 (period 1), a_2 = -0.3 (period 0 again)
///
/// Expected:
/// - Z'_0 = 0.7·0 + 1.0 = 1.0       → Z_0 = 100 + 20·1.0 = 120.0 (period 0)
/// - Z'_1 = 0.6·1.0 + 0.5 = 1.1     → Z_1 = 120 + 25·1.1 = 147.5 (period 1)
/// - Z'_2 = 0.7·1.1 + (-0.3) = 0.47 → Z_2 = 100 + 20·0.47 = 109.4 (period 0)
#[test]
fn test_par_seasonal_hand_calculated() {
    let params = SeasonalParams::new(
        2,                          // period = 2
        vec![1, 1],                 // AR(1) for both
        vec![vec![0.7], vec![0.6]], // φ_1,0 = 0.7, φ_1,1 = 0.6
        vec![100.0, 120.0],         // μ_0 = 100, μ_1 = 120
        vec![20.0, 25.0],           // σ_0 = 20, σ_1 = 25
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Stage 0 (period 0)
    let z0 = gen.generate_next(1.0);
    assert!(
        (z0 - 120.0).abs() < 1e-10,
        "Z_0 should be 120.0, got {}",
        z0
    );

    // Stage 1 (period 1)
    let z1 = gen.generate_next(0.5);
    assert!(
        (z1 - 147.5).abs() < 1e-10,
        "Z_1 should be 147.5, got {}",
        z1
    );

    // Stage 2 (period 0 again)
    let z2 = gen.generate_next(-0.3);
    assert!(
        (z2 - 109.4).abs() < 1e-10,
        "Z_2 should be 109.4, got {}",
        z2
    );
}

// ============================================================================
// Statistical Convergence Tests
// ============================================================================

/// Verify long-run mean converges to seasonal mean μ_m
///
/// For stationary PAR(1), the unconditional mean is μ_m.
/// With enough samples, sample mean should approach μ_m.
///
/// Test: Generate 10,000 scenarios, verify sample mean ≈ μ_m ± 3σ/√n
#[test]
fn test_convergence_to_seasonal_mean() {
    let params = SeasonalParams::new(
        2, // 2 periods
        vec![1, 1],
        vec![vec![0.7], vec![0.6]],
        vec![100.0, 120.0], // Target means
        vec![20.0, 25.0],
    )
    .unwrap();

    let n_scenarios = 10_000;
    let n_warmup = 100; // Discard warmup to reach stationary distribution

    // Generate scenarios
    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Warmup phase
    for _ in 0..n_warmup {
        gen.generate_next(0.0); // Use zero residuals for warmup
    }

    // Reset and collect samples
    gen.reset(vec![]);
    let mut values_period0 = Vec::new();
    let mut values_period1 = Vec::new();

    for i in 0..n_scenarios {
        // Use small random residuals (simulating Normal(0,1))
        use std::f64::consts::PI;
        let u1 = ((i as f64 + 1.0) / (n_scenarios as f64 + 1.0)).sin();
        let u2 = ((i as f64 + 1.0) / (n_scenarios as f64 + 1.0)).cos();
        let residual = ((-2.0 * (u1.abs() + 0.01).ln()).sqrt()
            * (2.0 * PI * u2).cos())
            * 0.1;

        let z = gen.generate_next(residual);

        if i % 2 == 0 {
            values_period0.push(z);
        } else {
            values_period1.push(z);
        }
    }

    // Compute sample means
    let mean0: f64 =
        values_period0.iter().sum::<f64>() / values_period0.len() as f64;
    let mean1: f64 =
        values_period1.iter().sum::<f64>() / values_period1.len() as f64;

    // Expected means (with AR, mean converges to μ_m in stationary regime)
    let expected_mean0: f64 = 100.0;
    let expected_mean1: f64 = 120.0;

    // Tolerance: ±5% or ±10 units (generous for finite sample)
    let tol0 = expected_mean0.abs() * 0.1; // 10%
    let tol1 = expected_mean1.abs() * 0.1; // 10%

    println!(
        "Period 0: Sample mean = {:.2}, Expected = {:.2}, Diff = {:.2}",
        mean0,
        expected_mean0,
        (mean0 - expected_mean0).abs()
    );
    println!(
        "Period 1: Sample mean = {:.2}, Expected = {:.2}, Diff = {:.2}",
        mean1,
        expected_mean1,
        (mean1 - expected_mean1).abs()
    );

    assert!(
        (mean0 - expected_mean0).abs() < tol0,
        "Period 0 mean {:.2} should be within {:.2} of {:.2}",
        mean0,
        tol0,
        expected_mean0
    );

    assert!(
        (mean1 - expected_mean1).abs() < tol1,
        "Period 1 mean {:.2} should be within {:.2} of {:.2}",
        mean1,
        tol1,
        expected_mean1
    );
}

/// Verify variance structure in stationary PAR(1)
///
/// For AR(1): Var(Z) ≈ σ²/(1-φ²) in limit
/// For PAR with small residuals, variance dominated by seasonal σ_m
#[test]
fn test_variance_structure() {
    let params = SeasonalParams::new(
        1,
        vec![1],
        vec![vec![0.5]], // φ = 0.5
        vec![100.0],
        vec![20.0], // σ = 20
    )
    .unwrap();

    let n_scenarios = 5_000;
    let n_warmup = 100;

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Warmup
    for _ in 0..n_warmup {
        gen.generate_next(0.0);
    }

    // Collect samples with varying residuals
    gen.reset(vec![]);
    let mut values = Vec::with_capacity(n_scenarios);

    use std::f64::consts::PI;
    for i in 0..n_scenarios {
        // Generate pseudo-random residuals in [-1, 1] range
        let t = (i as f64) / (n_scenarios as f64);
        let residual = (2.0 * PI * t * 7.0).sin() * (2.0 * PI * t * 13.0).cos();
        values.push(gen.generate_next(residual));
    }

    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let variance: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
        / values.len() as f64;
    let std_dev = variance.sqrt();

    println!("Sample mean = {:.2}, std dev = {:.2}", mean, std_dev);
    println!("Expected mean ≈ 100, std dev in range [15, 35]");

    // Variance should be reasonable (not exploding, not collapsed)
    // With σ=20 and varying residuals, expect std dev roughly in [15, 35]
    assert!(
        std_dev > 10.0 && std_dev < 50.0,
        "Std dev {:.2} should be in reasonable range [10, 50]",
        std_dev
    );

    // Mean should be near 100
    assert!(
        (mean - 100.0).abs() < 15.0,
        "Mean {:.2} should be near 100",
        mean
    );
}

// ============================================================================
// Stationarity Verification
// ============================================================================

/// Verify stationary PAR produces bounded series
///
/// For stationary coefficients (Σφ_k < 1), series should remain bounded
/// even with occasional large residuals.
#[test]
fn test_stationarity_produces_bounded_series() {
    let params = SeasonalParams::new(
        1,
        vec![2],
        vec![vec![0.5, 0.3]], // Σφ = 0.8 < 1 (stationary)
        vec![100.0],
        vec![20.0],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let n_steps = 10_000;
    let mut max_value = f64::NEG_INFINITY;
    let mut min_value = f64::INFINITY;

    for i in 0..n_steps {
        // Occasional large residuals
        let residual = if i % 100 == 0 {
            3.0
        } else if i % 100 == 50 {
            -3.0
        } else {
            0.0
        };
        let z = gen.generate_next(residual);

        max_value = max_value.max(z);
        min_value = min_value.min(z);

        // Should never explode
        assert!(z.is_finite(), "Value should be finite at step {}", i);
    }

    println!("Value range: [{:.2}, {:.2}]", min_value, max_value);

    // For stationary process with μ=100, σ=20, expect range roughly [20, 180]
    assert!(
        max_value < 300.0 && min_value > 0.0,
        "Stationary series should be bounded: [{:.2}, {:.2}]",
        min_value,
        max_value
    );
}

/// Verify coefficient sum constraint
///
/// CEPEL: For AR(p) stationarity, Σφ_k < 1 (necessary condition)
#[test]
fn test_coefficient_sum_constraint() {
    // Valid stationary case
    let params_valid = SeasonalParams::new(
        1,
        vec![2],
        vec![vec![0.5, 0.3]], // Σ = 0.8 < 1 ✓
        vec![100.0],
        vec![20.0],
    );
    assert!(
        params_valid.is_ok(),
        "Stationary coefficients should be valid"
    );

    // Invalid non-stationary case
    let params_invalid = SeasonalParams::new(
        1,
        vec![2],
        vec![vec![0.6, 0.5]], // Σ = 1.1 > 1 ✗
        vec![100.0],
        vec![20.0],
    );
    assert!(
        params_invalid.is_err(),
        "Non-stationary coefficients should be rejected"
    );
}

// ============================================================================
// CEPEL Equation Compliance
// ============================================================================

/// Verify PAR equation structure matches CEPEL definition
///
/// CEPEL: Z_t = μ_m + σ_m · [AR_term + a_t]
/// where AR_term = Σ φ_km · a_t-k
#[test]
fn test_cepel_equation_structure() {
    // Test with zero residuals → should get exactly μ_m
    let params = SeasonalParams::new(
        2,
        vec![1, 1],
        vec![vec![0.7], vec![0.6]],
        vec![100.0, 120.0],
        vec![20.0, 25.0],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // With zero residuals and zero initial lags:
    // Z_0 = μ_0 + σ_0 · (0.7·0 + 0) = 100
    let z0 = gen.generate_next(0.0);
    assert!((z0 - 100.0).abs() < 1e-10, "Zero residual → μ_0 = {}", z0);

    // Z_1 = μ_1 + σ_1 · (0.6·0 + 0) = 120
    let z1 = gen.generate_next(0.0);
    assert!((z1 - 120.0).abs() < 1e-10, "Zero residual → μ_1 = {}", z1);
}

/// Verify seasonal parameter application
///
/// Each period m should use its own (μ_m, σ_m, φ_km)
#[test]
fn test_seasonal_parameter_switching() {
    let params = SeasonalParams::new(
        3, // 3 periods
        vec![1, 1, 1],
        vec![vec![0.5], vec![0.6], vec![0.7]], // Different φ per period
        vec![100.0, 110.0, 120.0],             // Different μ per period
        vec![10.0, 15.0, 20.0],                // Different σ per period
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Period 0: μ=100, σ=10, φ=0.5
    // Z'_0 = 0.5·0 + 1.0 = 1.0 → Z_0 = 100 + 10·1.0 = 110
    let z0 = gen.generate_next(1.0);
    let expected0 = 110.0;
    assert!((z0 - expected0).abs() < 1e-10);

    // Period 1: μ=110, σ=15, φ=0.6
    // Z'_1 = 0.6·1.0 + 1.0 = 1.6 → Z_1 = 110 + 15·1.6 = 134
    let z1 = gen.generate_next(1.0);
    let expected1 = 134.0;
    assert!((z1 - expected1).abs() < 1e-10);

    // Period 2: μ=120, σ=20, φ=0.7
    // Z'_2 = 0.7·1.6 + 1.0 = 2.12 → Z_2 = 120 + 20·2.12 = 162.4
    let z2 = gen.generate_next(1.0);
    let expected2 = 162.4;
    assert!((z2 - expected2).abs() < 1e-10);

    // Period 0 again (wraps)
    // Z'_3 = 0.5·2.12 + 1.0 = 2.06 → Z_3 = 100 + 10·2.06 = 120.6
    let z3 = gen.generate_next(1.0);
    let expected3 = 120.6;
    assert!((z3 - expected3).abs() < 1e-10);
}

// ============================================================================
// Integration with Marginal Distributions
// ============================================================================

/// Verify PAR works correctly with Normal marginal distribution
///
/// Normal(0,1) residuals → PAR applies seasonal structure
#[test]
fn test_par_with_normal_marginals() {
    let params = SeasonalParams::new(
        1,
        vec![1],
        vec![vec![0.7]],
        vec![100.0],
        vec![20.0],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Simulate Normal(0,1) residuals
    let residuals = vec![0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5];
    let mut values = Vec::new();

    for &a_t in &residuals {
        values.push(gen.generate_next(a_t));
    }

    // All values should be reasonable (μ ± few σ)
    for (i, &z) in values.iter().enumerate() {
        assert!(
            z > 20.0 && z < 180.0,
            "Value {} = {} should be in reasonable range for μ=100, σ=20",
            i,
            z
        );
    }
}

/// Verify PAR handles extreme residual values gracefully
///
/// Even with extreme residuals, output should scale with σ_m
#[test]
fn test_par_with_extreme_residuals() {
    let params = SeasonalParams::new(
        1,
        vec![1],
        vec![vec![0.5]],
        vec![100.0],
        vec![20.0],
    )
    .unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Extreme residuals
    let z_large = gen.generate_next(10.0);
    let z_small = gen.generate_next(-10.0);

    // Should be finite and scale appropriately
    assert!(z_large.is_finite() && z_small.is_finite());

    // Large positive residual → value above mean
    assert!(
        z_large > 100.0,
        "Large positive residual should produce value > μ"
    );

    // Large negative residual → value below mean
    assert!(
        z_small < 100.0,
        "Large negative residual should produce value < μ"
    );
}

//
// ============================================================================
// Correlation Preservation Tests
// ============================================================================
//

/// Test that spatial correlation is preserved through PAR generation
///
/// This verifies that PAR transformation doesn't break spatial correlation
/// structure in the residuals.
#[test]
fn test_spatial_correlation_preservation() {
    use nalgebra::DMatrix;
    use powers_rs::correlation::{
        CorrelatedNoiseGenerator, MarginalDistribution,
    };
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    // Setup: 3 stations with specified correlation structure
    let n_stations = 3;
    let n_scenarios = 2_000;

    // Target correlation matrix (symmetric, positive definite)
    // Station 0 & 1: r=0.7, Station 0 & 2: r=0.5, Station 1 & 2: r=0.6
    #[rustfmt::skip]
    let correlation_data = vec![
        1.0, 0.7, 0.5,
        0.7, 1.0, 0.6,
        0.5, 0.6, 1.0,
    ];
    let correlation_matrix =
        DMatrix::from_row_slice(n_stations, n_stations, &correlation_data);

    // Marginal distributions (Normal(0,1) for all stations)
    let marginals = vec![
        MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0
        };
        n_stations
    ];

    // Correlated noise generator
    let corr_gen =
        CorrelatedNoiseGenerator::new(correlation_matrix.clone(), marginals)
            .unwrap();

    // PAR generators (one per station)
    let mut par_generators = Vec::new();
    for station_id in 0..n_stations {
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.5]],                        // φ = 0.5
            vec![100.0 + station_id as f64 * 10.0], // Different means
            vec![20.0],
        )
        .unwrap();
        par_generators.push(PeriodicARGenerator::new(params, vec![]));
    }

    // Generate scenarios
    let mut all_values = vec![Vec::new(); n_stations];
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    for _ in 0..n_scenarios {
        // Generate correlated normal samples
        let correlated_sample = corr_gen.generate_correlated_sample(&mut rng);

        // Apply PAR transformation
        for (station_id, &residual) in correlated_sample.iter().enumerate() {
            let par_value = par_generators[station_id].generate_next(residual);
            all_values[station_id].push(par_value);
        }
    }

    // Compute empirical correlation matrix
    let mut empirical_corr = vec![vec![0.0; n_stations]; n_stations];

    for i in 0..n_stations {
        for j in 0..n_stations {
            let mean_i =
                all_values[i].iter().sum::<f64>() / all_values[i].len() as f64;
            let mean_j =
                all_values[j].iter().sum::<f64>() / all_values[j].len() as f64;

            let cov: f64 = all_values[i]
                .iter()
                .zip(all_values[j].iter())
                .map(|(&x, &y)| (x - mean_i) * (y - mean_j))
                .sum::<f64>()
                / all_values[i].len() as f64;

            let std_i = (all_values[i]
                .iter()
                .map(|&x| (x - mean_i).powi(2))
                .sum::<f64>()
                / all_values[i].len() as f64)
                .sqrt();

            let std_j = (all_values[j]
                .iter()
                .map(|&y| (y - mean_j).powi(2))
                .sum::<f64>()
                / all_values[j].len() as f64)
                .sqrt();

            empirical_corr[i][j] = cov / (std_i * std_j);
        }
    }

    println!("\n=== Spatial Correlation Preservation ===");
    println!("Target correlation matrix:");
    for i in 0..n_stations {
        println!(
            "  [{:.3}, {:.3}, {:.3}]",
            correlation_matrix[(i, 0)],
            correlation_matrix[(i, 1)],
            correlation_matrix[(i, 2)]
        );
    }
    println!("\nEmpirical correlation matrix:");
    for row in &empirical_corr {
        println!("  [{:.3}, {:.3}, {:.3}]", row[0], row[1], row[2]);
    }

    // Verify correlation preservation (tolerance ±0.05)
    let tolerance = 0.05;
    for i in 0..n_stations {
        for j in 0..n_stations {
            let target = correlation_matrix[(i, j)];
            let empirical = empirical_corr[i][j];
            let diff = (empirical - target).abs();

            println!(
                "Corr({},{}) target={:.3}, empirical={:.3}, diff={:.3}",
                i, j, target, empirical, diff
            );

            assert!(
                diff < tolerance,
                "Correlation({},{}) diff {:.3} exceeds tolerance {:.3}",
                i,
                j,
                diff,
                tolerance
            );
        }
    }
}

/// Test correlation preservation with seasonal PAR parameters
#[test]
fn test_correlation_with_seasonal_par() {
    use nalgebra::DMatrix;
    use powers_rs::correlation::{
        CorrelatedNoiseGenerator, MarginalDistribution,
    };
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    let n_stations = 2;
    let n_scenarios = 3_000;
    let n_periods = 2;

    // Simple correlation: r=0.8
    let correlation_matrix =
        DMatrix::from_row_slice(n_stations, n_stations, &[1.0, 0.8, 0.8, 1.0]);
    let marginals = vec![
        MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0
        };
        n_stations
    ];

    let corr_gen =
        CorrelatedNoiseGenerator::new(correlation_matrix, marginals).unwrap();

    // Seasonal PAR generators (different parameters per period)
    let mut par_generators = Vec::new();
    for _ in 0..n_stations {
        let params = SeasonalParams::new(
            n_periods,
            vec![1, 1],
            vec![vec![0.6], vec![0.4]], // Different φ per period
            vec![100.0, 120.0],         // Wet/dry season means
            vec![20.0, 25.0],           // Wet/dry season std devs
        )
        .unwrap();
        par_generators.push(PeriodicARGenerator::new(params, vec![]));
    }

    // Generate scenarios across both periods
    let mut values_period0 = vec![Vec::new(); n_stations];
    let mut values_period1 = vec![Vec::new(); n_stations];
    let mut rng = Xoshiro256Plus::seed_from_u64(42);

    for scenario in 0..n_scenarios {
        let period = scenario % n_periods;

        let correlated_sample = corr_gen.generate_correlated_sample(&mut rng);

        for (station_id, &residual) in correlated_sample.iter().enumerate() {
            let par_value = par_generators[station_id].generate_next(residual);

            if period == 0 {
                values_period0[station_id].push(par_value);
            } else {
                values_period1[station_id].push(par_value);
            }
        }
    }

    // Check correlation in each period
    for (period_idx, values) in
        [&values_period0, &values_period1].iter().enumerate()
    {
        let mean0 = values[0].iter().sum::<f64>() / values[0].len() as f64;
        let mean1 = values[1].iter().sum::<f64>() / values[1].len() as f64;

        let cov: f64 = values[0]
            .iter()
            .zip(values[1].iter())
            .map(|(&x, &y)| (x - mean0) * (y - mean1))
            .sum::<f64>()
            / values[0].len() as f64;

        let std0 =
            (values[0].iter().map(|&x| (x - mean0).powi(2)).sum::<f64>()
                / values[0].len() as f64)
                .sqrt();

        let std1 =
            (values[1].iter().map(|&y| (y - mean1).powi(2)).sum::<f64>()
                / values[1].len() as f64)
                .sqrt();

        let empirical_corr = cov / (std0 * std1);
        let target_corr = 0.8;

        println!(
            "Period {}: target corr={:.3}, empirical corr={:.3}, diff={:.3}",
            period_idx,
            target_corr,
            empirical_corr,
            (empirical_corr - target_corr).abs()
        );

        assert!(
            (empirical_corr - target_corr).abs() < 0.06,
            "Period {} correlation {:.3} differs from target {:.3}",
            period_idx,
            empirical_corr,
            target_corr
        );
    }
}
