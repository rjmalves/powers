//! Baseline performance benchmarks for PERF-003
//!
//! Measures subproblem construction and hydro_data access patterns.
//! These are building blocks for the hot path optimization in PERF-004.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::input::UncertaintyType;
use powers_rs::subproblem::Subproblem;
use powers_rs::system::{Bus, Hydro, System, Thermal};
use powers_rs::uncertainty_model::{
    DistributionType, PARParams, SeasonalParams, UncertaintyModel,
};

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

criterion_group!(
    benches,
    bench_subproblem_construction,
    bench_hydro_data_access
);
criterion_main!(benches);
