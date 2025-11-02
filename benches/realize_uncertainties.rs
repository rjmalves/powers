//! Baseline performance benchmarks for PERF-003 and PERF-004
//!
//! **Purpose**: Establish baseline performance metrics before hot path optimization.
//!
//! This benchmark suite measures:
//! 1. Subproblem construction with hydro_data preprocessing (PERF-002)
//! 2. realize_uncertainties() hot path (PERF-004 target: 2-3x speedup)
//! 3. Memory access patterns for cache optimization
//!
//! **Target Metrics (PERF-004)**:
//! - realize_uncertainties 50 hydros: 120-150μs → 40-60μs (2-3x faster)
//! - Forward pass 100 hydros: ~5s → ~2-2.5s (50-60% faster)
//!
//! **Run with**:
//! ```bash
//! cargo bench --bench realize_uncertainties -- --save-baseline before_perf004
//! # After PERF-004:
//! cargo bench --bench realize_uncertainties -- --baseline before_perf004
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::input::UncertaintyType;
use powers_rs::scenario::OptimizedSampledBranchingNoises;
use powers_rs::subproblem::{Realization, StudyPeriodKind, Subproblem};
use powers_rs::system::{Bus, Hydro, System, Thermal};
use powers_rs::uncertainty_model::{
    DistributionType, PARParams, SeasonalParams, UncertaintyModel,
};

// =============================================================================
// Helper Functions - System and Model Creation
// =============================================================================

fn create_system_with_hydros(num_hydros: usize) -> System {
    let bus = Bus::new(0, 500.0);
    let mut hydros = Vec::new();
    for i in 0..num_hydros {
        hydros.push(Hydro::new(i, None, 0, 1.0, 0.0, 100.0, 0.0, 40.0, 0.01));
    }
    let thermal = Thermal::new(0, 0, 50.0, 0.0, 500.0);
    System::new(vec![bus], vec![], vec![thermal], hydros)
}

fn create_ar2_uncertainty_models(num_hydros: usize) -> Vec<UncertaintyModel> {
    let mut models = Vec::new();
    let par_params = PARParams {
        num_seasons: 1,
        ar_orders: vec![2],
        ar_coefficients: vec![vec![0.5, 0.3]],
        seasonal_means: vec![100.0],
        seasonal_stds: vec![20.0],
        seasonal_distributions: vec![DistributionType::Normal],
        max_ar_order: 2,
    };
    for i in 0..num_hydros {
        models.push(UncertaintyModel::PeriodicAR {
            entity_type: UncertaintyType::Inflow,
            entity_id: i,
            par_params: par_params.clone(),
        });
    }
    models
}

fn create_independent_uncertainty_models(
    num_hydros: usize,
) -> Vec<UncertaintyModel> {
    let mut models = Vec::new();
    for i in 0..num_hydros {
        models.push(UncertaintyModel::Independent {
            entity_type: UncertaintyType::Inflow,
            entity_id: i,
            seasonal_params: vec![SeasonalParams {
                mean: 100.0,
                std_dev: 20.0,
                distribution: DistributionType::Normal,
            }],
        });
    }
    models
}

fn create_ar3_uncertainty_models(num_hydros: usize) -> Vec<UncertaintyModel> {
    let mut models = Vec::new();
    let par_params = PARParams {
        num_seasons: 1,
        ar_orders: vec![3],
        ar_coefficients: vec![vec![0.4, 0.3, 0.2]],
        seasonal_means: vec![150.0],
        seasonal_stds: vec![30.0],
        seasonal_distributions: vec![DistributionType::Normal],
        max_ar_order: 3,
    };
    for i in 0..num_hydros {
        models.push(UncertaintyModel::PeriodicAR {
            entity_type: UncertaintyType::Inflow,
            entity_id: i,
            par_params: par_params.clone(),
        });
    }
    models
}

fn create_test_noises(num_hydros: usize) -> OptimizedSampledBranchingNoises {
    // Create empty noise structure and fill with innovations
    let mut noises = OptimizedSampledBranchingNoises::new(1, num_hydros);
    
    // Manually set innovations for testing (zero-mean, unit variance)
    noises.inflow_innovations = vec![0.5; num_hydros];
    noises.load_innovations = vec![0.1; 1]; // Single bus
    
    noises
}

// =============================================================================
// PERF-002 Benchmarks: Subproblem Construction
// =============================================================================

fn bench_subproblem_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("subproblem_construction");
    for &num_hydros in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_hydros),
            &num_hydros,
            |b, &n| {
                let system = create_system_with_hydros(n);
                let models = create_ar2_uncertainty_models(n);
                b.iter(|| {
                    let sp = Subproblem::new_from_uncertainty_models(
                        black_box(&system),
                        black_box("storage"),
                        black_box(&models),
                        black_box(0),
                    );
                    black_box(sp);
                });
            },
        );
    }
    group.finish();
}

fn bench_hydro_data_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydro_data_access");
    for &num_hydros in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_hydros),
            &num_hydros,
            |b, &n| {
                let system = create_system_with_hydros(n);
                let models = create_ar2_uncertainty_models(n);
                let subproblem = Subproblem::new_from_uncertainty_models(
                    &system, "storage", &models, 0,
                );
                b.iter(|| {
                    let mut sum = 0.0;
                    for hydro in &subproblem.hydro_data {
                        sum += hydro.deterministic_noise_base;
                        sum += hydro.seasonal_params.std_dev;
                    }
                    black_box(sum);
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// PERF-003/PERF-004 Benchmarks: realize_uncertainties (HOT PATH)
// =============================================================================

/// Benchmark realize_uncertainties with Independent models (AR order 0)
///
/// **Baseline Expectation**: ~10-20μs for 50 hydros (no lag computation)
fn bench_realize_uncertainties_independent(c: &mut Criterion) {
    let mut group = c.benchmark_group("realize_uncertainties_independent");
    
    for &num_hydros in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_hydros),
            &num_hydros,
            |b, &n| {
                let system = create_system_with_hydros(n);
                let models = create_independent_uncertainty_models(n);
                let mut subproblem = Subproblem::new_from_uncertainty_models(
                    &system, "storage", &models, 0,
                );
                
                // Set initial conditions
                let initial_storage = vec![50.0; n];
                let load = vec![300.0];
                subproblem.set_hydro_balance_rhs(&initial_storage);
                subproblem.set_load_balance_rhs(&load);
                
                let noises = create_test_noises(n);
                let mut realization =
                    Realization::with_capacity(&StudyPeriodKind::Study, &system);
                
                b.iter(|| {
                    subproblem
                        .realize_uncertainties(
                            black_box(&noises),
                            black_box(&mut realization),
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark realize_uncertainties with AR(2) models
///
/// **PERF-003 Baseline Target**: 120-150μs for 50 hydros
/// **PERF-004 Optimized Target**: 40-60μs for 50 hydros (2-3x speedup)
fn bench_realize_uncertainties_ar2(c: &mut Criterion) {
    let mut group = c.benchmark_group("realize_uncertainties_ar2");
    
    for &num_hydros in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_hydros),
            &num_hydros,
            |b, &n| {
                let system = create_system_with_hydros(n);
                let models = create_ar2_uncertainty_models(n);
                let mut subproblem = Subproblem::new_from_uncertainty_models(
                    &system, "storage", &models, 0,
                );
                
                // Set initial conditions
                let initial_storage = vec![50.0; n];
                let load = vec![300.0];
                subproblem.set_hydro_balance_rhs(&initial_storage);
                subproblem.set_load_balance_rhs(&load);
                
                let noises = create_test_noises(n);
                let mut realization =
                    Realization::with_capacity(&StudyPeriodKind::Study, &system);
                
                b.iter(|| {
                    subproblem
                        .realize_uncertainties(
                            black_box(&noises),
                            black_box(&mut realization),
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Benchmark realize_uncertainties with AR(3) models
///
/// **Purpose**: Test scaling with higher AR orders
fn bench_realize_uncertainties_ar3(c: &mut Criterion) {
    let mut group = c.benchmark_group("realize_uncertainties_ar3");
    
    for &num_hydros in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_hydros),
            &num_hydros,
            |b, &n| {
                let system = create_system_with_hydros(n);
                let models = create_ar3_uncertainty_models(n);
                let mut subproblem = Subproblem::new_from_uncertainty_models(
                    &system, "storage", &models, 0,
                );
                
                // Set initial conditions
                let initial_storage = vec![50.0; n];
                let load = vec![300.0];
                subproblem.set_hydro_balance_rhs(&initial_storage);
                subproblem.set_load_balance_rhs(&load);
                
                let noises = create_test_noises(n);
                let mut realization =
                    Realization::with_capacity(&StudyPeriodKind::Study, &system);
                
                b.iter(|| {
                    subproblem
                        .realize_uncertainties(
                            black_box(&noises),
                            black_box(&mut realization),
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_subproblem_construction,
    bench_hydro_data_access,
    bench_realize_uncertainties_independent,
    bench_realize_uncertainties_ar2,
    bench_realize_uncertainties_ar3,
);
criterion_main!(benches);
