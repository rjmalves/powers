//! Benchmarks for AR dynamics performance
//!
//! These benchmarks test deprecated AR models during soft deprecation (PAR-018).

#![allow(deprecated)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::ar_dynamics::ARDynamicsApplicator;
use powers_rs::input::TemporalModel;
use std::collections::HashMap;

fn benchmark_ar_dynamics_ar1(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities, all AR(1)
    let num_scenarios = 1000;
    let num_entities = 10;

    // All entities are AR(1) with φ₁=0.7
    let temporal_models: Vec<TemporalModel> = (0..num_entities)
        .map(|_| TemporalModel::Autoregressive {
            lag_order: 1,
            coefficients: vec![0.7],
        })
        .collect();

    // Initial lags for each entity
    let mut initial_lags = HashMap::new();
    for i in 0..num_entities {
        initial_lags.insert(i, vec![100.0 + i as f64 * 10.0]);
    }

    let applicator =
        ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

    // Innovations from Stage 3 (marginal transformation)
    let innovations: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.5).collect())
        .collect();

    c.bench_function("ar_dynamics_ar1_1000x10", |b| {
        b.iter(|| {
            let _result = applicator.apply_ar_dynamics(black_box(&innovations));
        })
    });
}

fn benchmark_ar_dynamics_ar2(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities, all AR(2)
    let num_scenarios = 1000;
    let num_entities = 10;

    // All entities are AR(2) with φ₁=0.6, φ₂=0.2
    let temporal_models: Vec<TemporalModel> = (0..num_entities)
        .map(|_| TemporalModel::Autoregressive {
            lag_order: 2,
            coefficients: vec![0.6, 0.2],
        })
        .collect();

    // Initial lags for each entity
    let mut initial_lags = HashMap::new();
    for i in 0..num_entities {
        initial_lags
            .insert(i, vec![100.0 + i as f64 * 10.0, 95.0 + i as f64 * 10.0]);
    }

    let applicator =
        ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

    // Innovations from Stage 3
    let innovations: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.5).collect())
        .collect();

    c.bench_function("ar_dynamics_ar2_1000x10", |b| {
        b.iter(|| {
            let _result = applicator.apply_ar_dynamics(black_box(&innovations));
        })
    });
}

fn benchmark_ar_dynamics_mixed(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities, mixed AR(1)/AR(2)/Independent
    let num_scenarios = 1000;
    let num_entities = 10;

    // Mixed temporal models
    let temporal_models: Vec<TemporalModel> = (0..num_entities)
        .map(|i| match i % 3 {
            0 => TemporalModel::Autoregressive {
                lag_order: 1,
                coefficients: vec![0.7],
            },
            1 => TemporalModel::Autoregressive {
                lag_order: 2,
                coefficients: vec![0.6, 0.2],
            },
            _ => TemporalModel::Independent,
        })
        .collect();

    // Initial lags for AR entities
    let mut initial_lags = HashMap::new();
    for i in 0..num_entities {
        match i % 3 {
            0 => {
                initial_lags.insert(i, vec![100.0 + i as f64 * 10.0]);
            }
            1 => {
                initial_lags.insert(
                    i,
                    vec![100.0 + i as f64 * 10.0, 95.0 + i as f64 * 10.0],
                );
            }
            _ => {} // Independent, no lags
        }
    }

    let applicator =
        ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

    // Innovations from Stage 3
    let innovations: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.5).collect())
        .collect();

    c.bench_function("ar_dynamics_mixed_1000x10", |b| {
        b.iter(|| {
            let _result = applicator.apply_ar_dynamics(black_box(&innovations));
        })
    });
}

fn benchmark_ar_dynamics_independent_only(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities, all independent
    let num_scenarios = 1000;
    let num_entities = 10;

    // All entities are independent (no AR)
    let temporal_models: Vec<TemporalModel> = (0..num_entities)
        .map(|_| TemporalModel::Independent)
        .collect();

    let initial_lags = HashMap::new(); // No lags needed

    let applicator =
        ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

    // Innovations from Stage 3
    let innovations: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.5).collect())
        .collect();

    c.bench_function("ar_dynamics_independent_only_1000x10", |b| {
        b.iter(|| {
            let _result = applicator.apply_ar_dynamics(black_box(&innovations));
        })
    });
}

fn benchmark_ar_dynamics_large_scale(c: &mut Criterion) {
    // Setup: 5000 scenarios × 50 entities (Brazilian system scale)
    let num_scenarios = 5000;
    let num_entities = 50;

    // Mixed AR(1) and AR(2)
    let temporal_models: Vec<TemporalModel> = (0..num_entities)
        .map(|i| {
            if i % 2 == 0 {
                TemporalModel::Autoregressive {
                    lag_order: 1,
                    coefficients: vec![0.7],
                }
            } else {
                TemporalModel::Autoregressive {
                    lag_order: 2,
                    coefficients: vec![0.6, 0.2],
                }
            }
        })
        .collect();

    // Initial lags for each entity
    let mut initial_lags = HashMap::new();
    for i in 0..num_entities {
        if i % 2 == 0 {
            initial_lags.insert(i, vec![100.0 + i as f64]);
        } else {
            initial_lags.insert(i, vec![100.0 + i as f64, 95.0 + i as f64]);
        }
    }

    let applicator =
        ARDynamicsApplicator::new(temporal_models, initial_lags).unwrap();

    // Innovations from Stage 3
    let innovations: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.1).collect())
        .collect();

    c.bench_function("ar_dynamics_large_scale_5000x50", |b| {
        b.iter(|| {
            let _result = applicator.apply_ar_dynamics(black_box(&innovations));
        })
    });
}

criterion_group!(
    benches,
    benchmark_ar_dynamics_ar1,
    benchmark_ar_dynamics_ar2,
    benchmark_ar_dynamics_mixed,
    benchmark_ar_dynamics_independent_only,
    benchmark_ar_dynamics_large_scale
);
criterion_main!(benches);
