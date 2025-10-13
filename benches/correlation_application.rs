use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::DMatrix;
use powers_rs::correlation_applicator::{
    CorrelationApplicator, CorrelationBlock, EntityRef, UncertaintyType,
};
use std::collections::HashMap;

fn benchmark_correlation_application(c: &mut Criterion) {
    // Setup: 1000 scenarios × 10 entities with 2 correlation blocks
    let num_scenarios = 1000;
    let num_entities = 10;

    // Block 1: Correlate entities 0-4 (5 hydro inflows, ρ=0.7)
    let correlation_matrix_1 =
        DMatrix::from_fn(5, 5, |i, j| if i == j { 1.0 } else { 0.7 });
    let entities_1: Vec<EntityRef> = (0..5)
        .map(|i| EntityRef {
            uncertainty_type: UncertaintyType::HydroInflow,
            entity_id: i,
        })
        .collect();
    let block_1 =
        CorrelationBlock::new(entities_1, correlation_matrix_1).unwrap();

    // Block 2: Correlate entities 5-7 (3 loads, ρ=0.5)
    let correlation_matrix_2 =
        DMatrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.5 });
    let entities_2: Vec<EntityRef> = (0..3)
        .map(|i| EntityRef {
            uncertainty_type: UncertaintyType::Load,
            entity_id: i,
        })
        .collect();
    let block_2 =
        CorrelationBlock::new(entities_2, correlation_matrix_2).unwrap();

    // Entity mapping (entities 8-9 remain independent)
    let entity_map: HashMap<EntityRef, usize> = (0..5)
        .map(|i| {
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::HydroInflow,
                    entity_id: i,
                },
                i,
            )
        })
        .chain((0..3).map(|i| {
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Load,
                    entity_id: i,
                },
                5 + i,
            )
        }))
        .chain((0..2).map(|i| {
            (
                EntityRef {
                    uncertainty_type: UncertaintyType::Other,
                    entity_id: i,
                },
                8 + i,
            )
        }))
        .collect();

    let applicator =
        CorrelationApplicator::new(vec![block_1, block_2], entity_map);

    // Generate base samples (independent standard normals)
    let base_samples: Vec<Vec<f64>> = (0..num_scenarios)
        .map(|i| {
            (0..num_entities)
                .map(|j| {
                    // Deterministic samples for reproducibility
                    ((i * num_entities + j) as f64
                        / (num_scenarios * num_entities) as f64
                        - 0.5)
                        * 6.0
                })
                .collect()
        })
        .collect();

    // Benchmark
    c.bench_function("apply_correlation_1000x10", |b| {
        b.iter(|| {
            let correlated =
                applicator.apply_correlation(black_box(&base_samples));
            black_box(correlated);
        })
    });
}

criterion_group!(benches, benchmark_correlation_application);
criterion_main!(benches);
