//! Performance benchmarks for PAR (Periodic Autoregressive) model
//!
//! Measures:
//! - PAR(1), PAR(2), PAR(12) generation speed
//! - Comparison against stationary AR baseline
//! - Memory overhead of lag buffers
//! - Full scenario pipeline performance
//! - Scaling characteristics with different configurations
//!
//! Target: <10% overhead vs stationary AR for typical use cases

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};
use powers_rs::par_generator::PeriodicARGenerator;
use powers_rs::seasonal_params::SeasonalParams;

// ==============================================================================
// Baseline: Stationary AR(1) - Reference Implementation
// ==============================================================================

/// Simple stationary AR(1) for comparison baseline
/// Z_t = μ + φ·(Z_{t-1} - μ) + σ·a_t
struct StationaryAR1 {
    phi: f64,
    mu: f64,
    sigma: f64,
    z_prev: f64,
}

impl StationaryAR1 {
    fn new(phi: f64, mu: f64, sigma: f64) -> Self {
        Self {
            phi,
            mu,
            sigma,
            z_prev: mu,
        }
    }

    #[inline]
    fn generate_next(&mut self, residual: f64) -> f64 {
        let a_prev = (self.z_prev - self.mu) / self.sigma;
        let z = self.mu + self.sigma * (self.phi * a_prev + residual);
        self.z_prev = z;
        z
    }
}

/// Stationary AR(2) for comparison baseline
struct StationaryAR2 {
    phi1: f64,
    phi2: f64,
    mu: f64,
    sigma: f64,
    z_prev1: f64,
    z_prev2: f64,
}

impl StationaryAR2 {
    fn new(phi1: f64, phi2: f64, mu: f64, sigma: f64) -> Self {
        Self {
            phi1,
            phi2,
            mu,
            sigma,
            z_prev1: mu,
            z_prev2: mu,
        }
    }

    #[inline]
    fn generate_next(&mut self, residual: f64) -> f64 {
        let a_prev1 = (self.z_prev1 - self.mu) / self.sigma;
        let a_prev2 = (self.z_prev2 - self.mu) / self.sigma;
        let z = self.mu
            + self.sigma
                * (self.phi1 * a_prev1 + self.phi2 * a_prev2 + residual);
        self.z_prev2 = self.z_prev1;
        self.z_prev1 = z;
        z
    }
}

// ==============================================================================
// Single-Series Generation Benchmarks
// ==============================================================================

/// Benchmark PAR(1) vs stationary AR(1) generation speed
fn bench_par1_vs_stationary_ar1(c: &mut Criterion) {
    let mut group = c.benchmark_group("PAR(1) vs Stationary AR(1)");

    // Common parameters
    let phi = 0.7;
    let mu = 100.0;
    let sigma = 20.0;
    let n_steps = 10_000;

    // Generate residuals once (to ensure fair comparison)
    let residuals: Vec<f64> =
        (0..n_steps).map(|i| (i as f64 * 0.1).sin() * 2.0).collect();

    group.throughput(Throughput::Elements(n_steps as u64));

    // Baseline: Stationary AR(1)
    group.bench_function("Stationary AR(1)", |b| {
        b.iter(|| {
            let mut ar1 = StationaryAR1::new(phi, mu, sigma);
            for &residual in &residuals {
                black_box(ar1.generate_next(residual));
            }
        })
    });

    // PAR(1) with single period (equivalent to stationary)
    group.bench_function("PAR(1) - Single Period", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                1,
                vec![1],
                vec![vec![phi]],
                vec![mu],
                vec![sigma],
            )
            .unwrap();
            let mut par1 = PeriodicARGenerator::new(params, vec![]);
            for &residual in &residuals {
                black_box(par1.generate_next(residual));
            }
        })
    });

    // PAR(1) with 2 periods (typical seasonal case)
    group.bench_function("PAR(1) - 2 Periods", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                2,
                vec![1, 1],
                vec![vec![phi], vec![phi]],
                vec![mu, mu + 20.0],
                vec![sigma, sigma + 5.0],
            )
            .unwrap();
            let mut par1 = PeriodicARGenerator::new(params, vec![]);
            for &residual in &residuals {
                black_box(par1.generate_next(residual));
            }
        })
    });

    // PAR(1) with 12 periods (monthly data)
    group.bench_function("PAR(1) - 12 Periods", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                12,
                vec![1; 12],
                vec![vec![phi]; 12],
                (0..12).map(|i| mu + i as f64 * 10.0).collect(),
                vec![sigma; 12],
            )
            .unwrap();
            let mut par1 = PeriodicARGenerator::new(params, vec![]);
            for &residual in &residuals {
                black_box(par1.generate_next(residual));
            }
        })
    });

    group.finish();
}

/// Benchmark PAR(2) vs stationary AR(2) generation speed
fn bench_par2_vs_stationary_ar2(c: &mut Criterion) {
    let mut group = c.benchmark_group("PAR(2) vs Stationary AR(2)");

    let phi1 = 0.5;
    let phi2 = 0.3;
    let mu = 100.0;
    let sigma = 20.0;
    let n_steps = 10_000;

    let residuals: Vec<f64> =
        (0..n_steps).map(|i| (i as f64 * 0.1).sin() * 2.0).collect();

    group.throughput(Throughput::Elements(n_steps as u64));

    // Baseline: Stationary AR(2)
    group.bench_function("Stationary AR(2)", |b| {
        b.iter(|| {
            let mut ar2 = StationaryAR2::new(phi1, phi2, mu, sigma);
            for &residual in &residuals {
                black_box(ar2.generate_next(residual));
            }
        })
    });

    // PAR(2) with single period
    group.bench_function("PAR(2) - Single Period", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                1,
                vec![2],
                vec![vec![phi1, phi2]],
                vec![mu],
                vec![sigma],
            )
            .unwrap();
            let mut par2 = PeriodicARGenerator::new(params, vec![]);
            for &residual in &residuals {
                black_box(par2.generate_next(residual));
            }
        })
    });

    // PAR(2) with 12 periods
    group.bench_function("PAR(2) - 12 Periods", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                12,
                vec![2; 12],
                vec![vec![phi1, phi2]; 12],
                (0..12).map(|i| mu + i as f64 * 10.0).collect(),
                vec![sigma; 12],
            )
            .unwrap();
            let mut par2 = PeriodicARGenerator::new(params, vec![]);
            for &residual in &residuals {
                black_box(par2.generate_next(residual));
            }
        })
    });

    group.finish();
}

// ==============================================================================
// Varying AR Order Benchmarks
// ==============================================================================

/// Benchmark PAR with different AR orders per period
fn bench_varying_ar_orders(c: &mut Criterion) {
    let mut group = c.benchmark_group("PAR Varying Orders");

    let n_steps = 10_000;
    let residuals: Vec<f64> =
        (0..n_steps).map(|i| (i as f64 * 0.1).sin() * 2.0).collect();

    group.throughput(Throughput::Elements(n_steps as u64));

    // Test different AR orders: p = 1, 2, 3, 4, 6, 12
    for p in [1, 2, 3, 4, 6, 12] {
        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, &p| {
            b.iter(|| {
                let phi_coeffs = vec![0.7 / p as f64; p]; // Ensure stationary
                let params = SeasonalParams::new(
                    12,
                    vec![p; 12],
                    vec![phi_coeffs.clone(); 12],
                    (0..12).map(|i| 100.0 + i as f64 * 10.0).collect(),
                    vec![20.0; 12],
                )
                .unwrap();
                let mut par = PeriodicARGenerator::new(params, vec![]);
                for &residual in &residuals {
                    black_box(par.generate_next(residual));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark PAR with different numbers of periods
fn bench_varying_periods(c: &mut Criterion) {
    let mut group = c.benchmark_group("PAR Varying Periods");

    let n_steps = 10_000;
    let residuals: Vec<f64> =
        (0..n_steps).map(|i| (i as f64 * 0.1).sin() * 2.0).collect();

    group.throughput(Throughput::Elements(n_steps as u64));

    // Test different period counts: 1, 2, 4, 12, 24, 52
    for n_periods in [1, 2, 4, 12, 24, 52] {
        group.bench_with_input(
            BenchmarkId::from_parameter(n_periods),
            &n_periods,
            |b, &n_periods| {
                b.iter(|| {
                    let params = SeasonalParams::new(
                        n_periods,
                        vec![2; n_periods],
                        vec![vec![0.5, 0.3]; n_periods],
                        (0..n_periods)
                            .map(|i| 100.0 + i as f64 * 5.0)
                            .collect(),
                        vec![20.0; n_periods],
                    )
                    .unwrap();
                    let mut par = PeriodicARGenerator::new(params, vec![]);
                    for &residual in &residuals {
                        black_box(par.generate_next(residual));
                    }
                })
            },
        );
    }

    group.finish();
}

// ==============================================================================
// Multi-Station Pipeline Benchmarks
// ==============================================================================

/// Benchmark end-to-end scenario generation with PAR
fn bench_multi_station_scenario_generation(c: &mut Criterion) {
    use nalgebra::DMatrix;
    use powers_rs::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
    use powers_rs::correlation_applicator::{
        CorrelationApplicator, CorrelationBlock, EntityRef, UncertaintyType,
    };
    use powers_rs::input::MarginalDistribution;
    use powers_rs::marginal_transformer::MarginalTransformer;
    use std::collections::HashMap;

    let mut group = c.benchmark_group("Multi-Station Scenario Generation");

    let n_stations = 10;
    let n_scenarios = 1_000;

    // Setup correlation matrix (identity for simplicity)
    let mut corr_data = vec![0.0; n_stations * n_stations];
    for i in 0..n_stations {
        corr_data[i * n_stations + i] = 1.0;
    }
    let correlation_matrix =
        DMatrix::from_row_slice(n_stations, n_stations, &corr_data);

    let marginals = vec![
        MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0
        };
        n_stations
    ];

    // Setup pipeline components
    let entities: Vec<EntityRef> = (0..n_stations)
        .map(|id| EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: id,
        })
        .collect();

    let mut entity_map = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        entity_map.insert(*entity, idx);
    }

    let block =
        CorrelationBlock::new(entities.clone(), correlation_matrix.clone())
            .unwrap();

    let applicator = CorrelationApplicator::new(vec![block], entity_map);
    let transformer = MarginalTransformer::new(marginals).unwrap();

    group.throughput(Throughput::Elements((n_stations * n_scenarios) as u64));

    // Baseline: Independent normal generation (no AR)
    group.bench_function("Baseline - Independent Normal", |b| {
        b.iter(|| {
            let seed = 42;
            let base_gen =
                BaseNoiseGenerator::new(n_scenarios, n_stations, seed);
            let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
            let correlated = applicator.apply_correlation(&base_samples);
            let final_samples = transformer.transform_marginals(&correlated);

            let mut values = vec![Vec::new(); n_stations];
            for scenario in final_samples.iter() {
                for (station, &val) in scenario.iter().enumerate() {
                    values[station].push(val);
                }
            }
            black_box(values);
        })
    });

    // PAR(1) with 2 periods
    group.bench_function("PAR(1) - 2 Periods", |b| {
        b.iter(|| {
            let seed = 42;
            let base_gen =
                BaseNoiseGenerator::new(n_scenarios, n_stations, seed);
            let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
            let correlated = applicator.apply_correlation(&base_samples);
            let residuals = transformer.transform_marginals(&correlated);

            let mut values = vec![Vec::new(); n_stations];

            let mut par_gens: Vec<_> = (0..n_stations)
                .map(|_| {
                    let params = SeasonalParams::new(
                        2,
                        vec![1, 1],
                        vec![vec![0.7], vec![0.6]],
                        vec![100.0, 120.0],
                        vec![20.0, 25.0],
                    )
                    .unwrap();
                    PeriodicARGenerator::new(params, vec![])
                })
                .collect();

            for scenario in residuals.iter() {
                for (station, &residual) in scenario.iter().enumerate() {
                    let val = par_gens[station].generate_next(residual);
                    values[station].push(val);
                }
            }
            black_box(values);
        })
    });

    // PAR(2) with 12 periods
    group.bench_function("PAR(2) - 12 Periods", |b| {
        b.iter(|| {
            let seed = 42;
            let base_gen =
                BaseNoiseGenerator::new(n_scenarios, n_stations, seed);
            let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
            let correlated = applicator.apply_correlation(&base_samples);
            let residuals = transformer.transform_marginals(&correlated);

            let mut values = vec![Vec::new(); n_stations];

            let mut par_gens: Vec<_> = (0..n_stations)
                .map(|_| {
                    let params = SeasonalParams::new(
                        12,
                        vec![2; 12],
                        vec![vec![0.5, 0.3]; 12],
                        (0..12).map(|i| 100.0 + i as f64 * 10.0).collect(),
                        vec![20.0; 12],
                    )
                    .unwrap();
                    PeriodicARGenerator::new(params, vec![])
                })
                .collect();

            for scenario in residuals.iter() {
                for (station, &residual) in scenario.iter().enumerate() {
                    let val = par_gens[station].generate_next(residual);
                    values[station].push(val);
                }
            }
            black_box(values);
        })
    });

    group.finish();
}

// ==============================================================================
// Memory Overhead Benchmarks
// ==============================================================================

/// Benchmark memory allocation overhead
fn bench_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Overhead");

    // Measure construction time (includes buffer allocation)
    group.bench_function("PAR(1) Construction", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                12,
                vec![1; 12],
                vec![vec![0.7]; 12],
                vec![100.0; 12],
                vec![20.0; 12],
            )
            .unwrap();
            black_box(PeriodicARGenerator::new(params, vec![]));
        })
    });

    group.bench_function("PAR(2) Construction", |b| {
        b.iter(|| {
            let params = SeasonalParams::new(
                12,
                vec![2; 12],
                vec![vec![0.5, 0.3]; 12],
                vec![100.0; 12],
                vec![20.0; 12],
            )
            .unwrap();
            black_box(PeriodicARGenerator::new(params, vec![]));
        })
    });

    group.bench_function("PAR(12) Construction", |b| {
        b.iter(|| {
            let phi = vec![0.7 / 12.0; 12]; // Ensure stationary
            let params = SeasonalParams::new(
                12,
                vec![12; 12],
                vec![phi.clone(); 12],
                vec![100.0; 12],
                vec![20.0; 12],
            )
            .unwrap();
            black_box(PeriodicARGenerator::new(params, vec![]));
        })
    });

    // Measure reset cost (should be fast, just memset)
    group.bench_function("PAR(2) Reset", |b| {
        let params = SeasonalParams::new(
            12,
            vec![2; 12],
            vec![vec![0.5, 0.3]; 12],
            vec![100.0; 12],
            vec![20.0; 12],
        )
        .unwrap();
        let mut par = PeriodicARGenerator::new(params, vec![]);

        b.iter(|| {
            par.reset(vec![]);
            black_box(&par);
        })
    });

    group.finish();
}

// ==============================================================================
// Regression: Non-PAR Pipeline Benchmarks
// ==============================================================================

/// Ensure PAR implementation doesn't impact non-PAR code paths
fn bench_regression_non_par_pipeline(c: &mut Criterion) {
    use nalgebra::DMatrix;
    use powers_rs::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
    use powers_rs::correlation_applicator::{
        CorrelationApplicator, CorrelationBlock, EntityRef, UncertaintyType,
    };
    use powers_rs::input::MarginalDistribution;
    use powers_rs::marginal_transformer::MarginalTransformer;
    use std::collections::HashMap;

    let mut group = c.benchmark_group("Regression - Non-PAR");

    let n_stations = 10;
    let n_scenarios = 1_000;

    let mut corr_data = vec![0.0; n_stations * n_stations];
    for i in 0..n_stations {
        corr_data[i * n_stations + i] = 1.0;
    }
    let correlation_matrix =
        DMatrix::from_row_slice(n_stations, n_stations, &corr_data);
    let marginals = vec![
        MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0
        };
        n_stations
    ];

    // Setup pipeline components
    let entities: Vec<EntityRef> = (0..n_stations)
        .map(|id| EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: id,
        })
        .collect();

    let mut entity_map = HashMap::new();
    for (idx, entity) in entities.iter().enumerate() {
        entity_map.insert(*entity, idx);
    }

    let block =
        CorrelationBlock::new(entities.clone(), correlation_matrix.clone())
            .unwrap();
    let applicator = CorrelationApplicator::new(vec![block], entity_map);
    let transformer = MarginalTransformer::new(marginals).unwrap();

    group.throughput(Throughput::Elements((n_stations * n_scenarios) as u64));

    // Pure correlation generation (no PAR)
    group.bench_function("Pure Correlation Generation", |b| {
        b.iter(|| {
            let seed = 42;
            let base_gen =
                BaseNoiseGenerator::new(n_scenarios, n_stations, seed);
            let base_samples = base_gen.generate(BaseNoiseMethod::Standard);
            let correlated = applicator.apply_correlation(&base_samples);
            let final_samples = transformer.transform_marginals(&correlated);

            for scenario in final_samples.iter() {
                black_box(scenario);
            }
        })
    });

    group.finish();
}

// ==============================================================================
// Criterion Configuration
// ==============================================================================

criterion_group!(
    benches,
    bench_par1_vs_stationary_ar1,
    bench_par2_vs_stationary_ar2,
    bench_varying_ar_orders,
    bench_varying_periods,
    bench_multi_station_scenario_generation,
    bench_memory_overhead,
    bench_regression_non_par_pipeline,
);
criterion_main!(benches);
