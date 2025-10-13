use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::input::MarginalDistribution;
use powers_rs::marginal_transformer::MarginalTransformer;

fn benchmark_marginal_transformation(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities (realistic hydrothermal case)
    let num_scenarios = 1000;
    let num_entities = 10;

    // Mix of Normal and LogNormal3 marginals
    let marginals: Vec<MarginalDistribution> = (0..num_entities)
        .map(|i| {
            if i % 2 == 0 {
                MarginalDistribution::Normal {
                    mean: 100.0 + i as f64 * 10.0,
                    std_dev: 20.0,
                }
            } else {
                MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 4.5,
                    sigma: 0.3,
                }
            }
        })
        .collect();

    let transformer = MarginalTransformer::new(marginals).unwrap();

    // Generate correlated standard normal samples (simulating Stage 2 output)
    let correlated_samples: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.1).collect())
        .collect();

    c.bench_function("marginal_transformation_1000x10", |b| {
        b.iter(|| {
            let _transformed =
                transformer.transform_marginals(black_box(&correlated_samples));
        })
    });
}

fn benchmark_normal_only(c: &mut Criterion) {
    let num_scenarios = 1000;
    let num_entities = 10;

    // All Normal marginals
    let marginals: Vec<MarginalDistribution> = (0..num_entities)
        .map(|i| MarginalDistribution::Normal {
            mean: 100.0 + i as f64 * 10.0,
            std_dev: 20.0,
        })
        .collect();

    let transformer = MarginalTransformer::new(marginals).unwrap();

    let correlated_samples: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.1).collect())
        .collect();

    c.bench_function("marginal_normal_only_1000x10", |b| {
        b.iter(|| {
            let _transformed =
                transformer.transform_marginals(black_box(&correlated_samples));
        })
    });
}

fn benchmark_lognormal3_only(c: &mut Criterion) {
    let num_scenarios = 1000;
    let num_entities = 10;

    // All LogNormal3 marginals
    let marginals: Vec<MarginalDistribution> = (0..num_entities)
        .map(|_| MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.3,
        })
        .collect();

    let transformer = MarginalTransformer::new(marginals).unwrap();

    let correlated_samples: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.1).collect())
        .collect();

    c.bench_function("marginal_lognormal3_only_1000x10", |b| {
        b.iter(|| {
            let _transformed =
                transformer.transform_marginals(black_box(&correlated_samples));
        })
    });
}

fn benchmark_large_scale(c: &mut Criterion) {
    // Brazilian system scale: 5000 scenarios × 50 entities
    let num_scenarios = 5000;
    let num_entities = 50;

    let marginals: Vec<MarginalDistribution> = (0..num_entities)
        .map(|i| {
            if i % 2 == 0 {
                MarginalDistribution::Normal {
                    mean: 100.0 + i as f64 * 10.0,
                    std_dev: 20.0,
                }
            } else {
                MarginalDistribution::LogNormal3 {
                    gamma: 1.0,
                    mu: 4.5,
                    sigma: 0.3,
                }
            }
        })
        .collect();

    let transformer = MarginalTransformer::new(marginals).unwrap();

    let correlated_samples: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|_| (0..num_entities).map(|j| (j as f64) * 0.1).collect())
        .collect();

    c.bench_function("marginal_transformation_5000x50", |b| {
        b.iter(|| {
            let _transformed =
                transformer.transform_marginals(black_box(&correlated_samples));
        })
    });
}

criterion_group!(
    benches,
    benchmark_marginal_transformation,
    benchmark_normal_only,
    benchmark_lognormal3_only,
    benchmark_large_scale
);
criterion_main!(benches);
