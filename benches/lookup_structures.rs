//! Performance benchmarks for NoiseLookupTable and lookup optimizations
//!
//! Measures:
//! - Lookup table construction time (O(n × s) pre-indexing)
//! - Single parameter lookup speed (O(1) HashMap access)
//! - Bulk parameter retrieval speed (cache-friendly iteration)
//! - Lookup table scaling with number of entities and seasons
//!
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
    Throughput,
};
use powers_rs::input::UncertaintyType;
use powers_rs::scenario::NoiseLookupTable;
use powers_rs::unified_noise_spec::{
    SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec, UnifiedNoiseSpec,
};
use std::collections::HashMap;

// ==============================================================================
// Lookup Table Construction Benchmarks
// ==============================================================================

/// Benchmark: NoiseLookupTable construction with varying entity counts
///
/// Measures the O(n × s) pre-indexing cost during table construction.
/// This is a one-time cost per scenario generation call.
fn bench_lookup_table_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_table_construction");

    // Test different problem sizes
    for (num_entities, num_seasons) in
        &[(5, 12), (10, 12), (20, 12), (50, 12), (100, 12)]
    {
        let total_params = num_entities * num_seasons;
        group.throughput(Throughput::Elements(total_params as u64));

        let label = format!("{}entities_{}seasons", num_entities, num_seasons);

        group.bench_function(&label, |b| {
            // Create synthetic unified noise specs
            let specs = create_synthetic_noise_specs(
                *num_entities,
                *num_seasons,
                false, // Independent model
            );

            b.iter(|| {
                let lookup =
                    NoiseLookupTable::from_unified_specs(black_box(&specs));
                black_box(lookup);
            })
        });
    }

    group.finish();
}

/// Benchmark: Construction comparison between independent and PAR models
///
/// PAR models have the same construction cost (same HashMap size),
/// but we verify there's no unexpected overhead.
fn bench_construction_independent_vs_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_independent_vs_par");

    let num_entities = 50;
    let num_seasons = 12;
    let total_params = num_entities * num_seasons;

    group.throughput(Throughput::Elements(total_params as u64));

    // Independent noise models
    group.bench_function("independent", |b| {
        let specs =
            create_synthetic_noise_specs(num_entities, num_seasons, false);
        b.iter(|| {
            let lookup =
                NoiseLookupTable::from_unified_specs(black_box(&specs));
            black_box(lookup);
        })
    });

    // PAR noise models
    group.bench_function("par", |b| {
        let specs =
            create_synthetic_noise_specs(num_entities, num_seasons, true);
        b.iter(|| {
            let lookup =
                NoiseLookupTable::from_unified_specs(black_box(&specs));
            black_box(lookup);
        })
    });

    group.finish();
}

fn bench_single_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_lookup");

    // Create lookup table with 100 entities × 12 seasons
    let specs = create_synthetic_noise_specs(100, 12, false);
    let lookup = NoiseLookupTable::from_unified_specs(&specs);

    group.throughput(Throughput::Elements(1));

    // Worst-case: Lookup at end of table (would be slowest for linear search)
    group.bench_function("worst_case", |b| {
        b.iter(|| {
            let params = lookup.get_params(
                black_box(UncertaintyType::Inflow),
                black_box(99), // Last entity
                black_box(11), // Last season
            );
            black_box(params);
        })
    });

    // Best-case: Lookup at beginning
    group.bench_function("best_case", |b| {
        b.iter(|| {
            let params = lookup.get_params(
                black_box(UncertaintyType::Inflow),
                black_box(0), // First entity
                black_box(0), // First season
            );
            black_box(params);
        })
    });

    // Average-case: Middle of table
    group.bench_function("average_case", |b| {
        b.iter(|| {
            let params = lookup.get_params(
                black_box(UncertaintyType::Inflow),
                black_box(50), // Middle entity
                black_box(6),  // Middle season
            );
            black_box(params);
        })
    });

    // Temporal model lookup (is_par check)
    group.bench_function("is_par_lookup", |b| {
        b.iter(|| {
            let is_par = lookup.is_par_model(
                black_box(UncertaintyType::Inflow),
                black_box(50),
            );
            black_box(is_par);
        })
    });

    group.finish();
}

/// Benchmark: Lookup scaling with table size
///
/// Verify O(1) complexity: lookup time should be constant regardless of table size.
fn bench_lookup_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lookup_scaling");

    group.throughput(Throughput::Elements(1));

    for num_entities in [10, 50, 100, 500, 1000] {
        let specs = create_synthetic_noise_specs(num_entities, 12, false);
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_entities),
            &lookup,
            |b, lookup| {
                b.iter(|| {
                    // Always lookup same position (middle)
                    let params = lookup.get_params(
                        black_box(UncertaintyType::Inflow),
                        black_box(num_entities / 2),
                        black_box(6),
                    );
                    black_box(params);
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: Bulk parameter retrieval vs repeated single lookups
fn bench_bulk_vs_single_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_vs_single_lookups");

    let num_entities = 100;
    let num_seasons = 12;
    let specs = create_synthetic_noise_specs(num_entities, num_seasons, false);
    let lookup = NoiseLookupTable::from_unified_specs(&specs);

    group.throughput(Throughput::Elements(num_entities as u64));

    // Baseline: Repeated single lookups
    group.bench_function("single_lookups", |b| {
        b.iter(|| {
            let mut params_vec = Vec::new();
            for entity_id in 0..num_entities {
                if let Some(params) = lookup.get_params(
                    UncertaintyType::Inflow,
                    entity_id,
                    6, // Season 6
                ) {
                    params_vec.push(params);
                }
            }
            black_box(params_vec);
        })
    });

    // Optimized: Bulk retrieval
    group.bench_function("bulk_retrieval", |b| {
        b.iter(|| {
            let params =
                lookup.get_all_params_for_season(UncertaintyType::Inflow, 6);
            black_box(params);
        })
    });

    group.finish();
}

/// Benchmark: Bulk retrieval scaling
///
/// Verify bulk retrieval maintains O(n) complexity and has better
/// cache locality than repeated HashMap lookups.
fn bench_bulk_retrieval_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_retrieval_scaling");

    for num_entities in [10, 50, 100, 500, 1000] {
        let specs = create_synthetic_noise_specs(num_entities, 12, false);
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        group.throughput(Throughput::Elements(num_entities as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(num_entities),
            &lookup,
            |b, lookup| {
                b.iter(|| {
                    let params = lookup
                        .get_all_params_for_season(UncertaintyType::Inflow, 6);
                    black_box(params);
                })
            },
        );
    }

    group.finish();
}

// ==============================================================================
// Profiling Helper Benchmarks
// ==============================================================================

/// Benchmark: Profiling helper methods (param_count, entity_count)
///
/// These should be extremely fast (just HashMap.len() calls).
fn bench_profiling_helpers(c: &mut Criterion) {
    let mut group = c.benchmark_group("profiling_helpers");

    let specs = create_synthetic_noise_specs(100, 12, false);
    let lookup = NoiseLookupTable::from_unified_specs(&specs);

    group.bench_function("param_count", |b| {
        b.iter(|| {
            let count = lookup.param_count();
            black_box(count);
        })
    });

    group.bench_function("entity_count", |b| {
        b.iter(|| {
            let count = lookup.entity_count(UncertaintyType::Inflow);
            black_box(count);
        })
    });

    group.finish();
}

// ==============================================================================
// Helper Functions
// ==============================================================================

/// Create synthetic UnifiedNoiseSpec for benchmarking
fn create_synthetic_noise_specs(
    num_entities: usize,
    num_seasons: usize,
    use_par: bool,
) -> Vec<UnifiedNoiseSpec> {
    use powers_rs::input::MarginalDistribution;

    (0..num_entities)
        .map(|entity_id| {
            let mut seasonal_params = HashMap::new();
            for season_id in 0..num_seasons {
                seasonal_params.insert(
                    season_id,
                    SeasonalNoiseParams {
                        mean: 100.0
                            + entity_id as f64 * 5.0
                            + season_id as f64 * 10.0,
                        std_dev: 20.0 + season_id as f64 * 2.0,
                        marginal_override: None,
                    },
                );
            }

            let temporal_model = if use_par {
                let mut par_params = HashMap::new();
                for season_id in 0..num_seasons {
                    par_params.insert(
                        season_id,
                        SeasonalPARParams {
                            ar_order: 1,
                            ar_coefficients: vec![0.7],
                        },
                    );
                }
                TemporalModelSpec::PeriodicAutoregressive {
                    num_seasons,
                    seasonal_ar_params: par_params,
                }
            } else {
                TemporalModelSpec::Independent
            };

            UnifiedNoiseSpec {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id,
                seasonal_params,
                temporal_model,
                marginal_distribution: Some(MarginalDistribution::Normal {
                    mean: 0.0,
                    std_dev: 1.0,
                }),
            }
        })
        .collect()
}

// ==============================================================================
// Criterion Configuration
// ==============================================================================

criterion_group!(
    benches,
    bench_lookup_table_construction,
    bench_construction_independent_vs_par,
    bench_single_lookup,
    bench_lookup_scaling,
    bench_bulk_vs_single_lookups,
    bench_bulk_retrieval_scaling,
    bench_profiling_helpers,
);
criterion_main!(benches);
