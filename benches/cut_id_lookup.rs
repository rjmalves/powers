// Micro-benchmarks for cut ID lookup performance
//
// Compares Vec::position (O(n)) vs HashSet::contains (O(1)) for cut ID lookups
// in typical SDDP scenarios.
//
// Run with:
//   cargo bench --bench cut_id_lookup

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use std::collections::HashSet;

/// Benchmark Vec::iter().position() for various active cut pool sizes and lookup counts
///
/// Simulates the current implementation's O(n×m) behavior where:
/// - n = number of active cuts (100-2000)
/// - m = number of cuts to look up (10-100)
fn bench_vec_position(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_position");

    // Test different pool sizes (typical: 100-1000 at convergence)
    for pool_size in [100, 200, 500, 1000, 2000] {
        // Test different lookup counts (typical: 10-50 cuts removed per iteration)
        for lookup_count in [10, 25, 50, 100] {
            let active_ids: Vec<usize> = (0..pool_size).collect();
            let to_find: Vec<usize> = (0..lookup_count)
                .map(|i| i * pool_size / lookup_count)
                .collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}",
                    pool_size, lookup_count
                )),
                &(active_ids, to_find),
                |b, (active_ids, to_find)| {
                    b.iter(|| {
                        // Simulate the hot loop in apply_aggregated_cut_selection_result
                        let mut found_count = 0;
                        for &id in to_find {
                            if let Some(_pos) =
                                active_ids.iter().position(|&x| x == id)
                            {
                                found_count += 1;
                            }
                        }
                        black_box(found_count);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark HashSet::contains() for the same scenarios
///
/// This is the proposed optimization - O(1) average case lookups.
fn bench_hashset_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashset_contains");

    for pool_size in [100, 200, 500, 1000, 2000] {
        for lookup_count in [10, 25, 50, 100] {
            let active_set: HashSet<usize> = (0..pool_size).collect();
            let to_find: Vec<usize> = (0..lookup_count)
                .map(|i| i * pool_size / lookup_count)
                .collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}",
                    pool_size, lookup_count
                )),
                &(active_set, to_find),
                |b, (active_set, to_find)| {
                    b.iter(|| {
                        let mut found_count = 0;
                        for &id in to_find {
                            if active_set.contains(&id) {
                                found_count += 1;
                            }
                        }
                        black_box(found_count);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark hybrid approach: convert Vec to HashSet once, then do lookups
///
/// This is the recommended Phase 1 implementation - amortizes conversion cost.
fn bench_hybrid_approach(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_vec_to_hashset");

    for pool_size in [100, 200, 500, 1000, 2000] {
        for lookup_count in [10, 25, 50, 100] {
            let active_ids: Vec<usize> = (0..pool_size).collect();
            let to_find: Vec<usize> = (0..lookup_count)
                .map(|i| i * pool_size / lookup_count)
                .collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}",
                    pool_size, lookup_count
                )),
                &(active_ids, to_find),
                |b, (active_ids, to_find)| {
                    b.iter(|| {
                        // Convert Vec to HashSet ONCE (O(n) cost)
                        let active_set: HashSet<usize> =
                            active_ids.iter().copied().collect();

                        // Then do O(1) lookups
                        let mut found_count = 0;
                        for &id in to_find {
                            if active_set.contains(&id) {
                                found_count += 1;
                            }
                        }
                        black_box(found_count);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark worst-case scenario: lookups at the end of the Vec
///
/// Vec::position scans from the beginning, so items at the end are worst-case.
fn bench_vec_position_worst_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_position_worst_case");

    for pool_size in [100, 200, 500, 1000, 2000] {
        for lookup_count in [10, 25, 50, 100] {
            let active_ids: Vec<usize> = (0..pool_size).collect();
            // Look for items at the END of the vec (worst case)
            let to_find: Vec<usize> =
                (pool_size - lookup_count..pool_size).collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}",
                    pool_size, lookup_count
                )),
                &(active_ids, to_find),
                |b, (active_ids, to_find)| {
                    b.iter(|| {
                        let mut found_count = 0;
                        for &id in to_find {
                            if let Some(_pos) =
                                active_ids.iter().position(|&x| x == id)
                            {
                                found_count += 1;
                            }
                        }
                        black_box(found_count);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark best-case scenario: lookups at the beginning of the Vec
fn bench_vec_position_best_case(c: &mut Criterion) {
    let mut group = c.benchmark_group("vec_position_best_case");

    for pool_size in [100, 200, 500, 1000, 2000] {
        for lookup_count in [10, 25, 50, 100] {
            let active_ids: Vec<usize> = (0..pool_size).collect();
            // Look for items at the BEGINNING of the vec (best case)
            let to_find: Vec<usize> = (0..lookup_count).collect();

            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "{}x{}",
                    pool_size, lookup_count
                )),
                &(active_ids, to_find),
                |b, (active_ids, to_find)| {
                    b.iter(|| {
                        let mut found_count = 0;
                        for &id in to_find {
                            if let Some(_pos) =
                                active_ids.iter().position(|&x| x == id)
                            {
                                found_count += 1;
                            }
                        }
                        black_box(found_count);
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark memory overhead: measure HashSet construction time
fn bench_hashset_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hashset_construction");

    for pool_size in [100, 200, 500, 1000, 2000] {
        let active_ids: Vec<usize> = (0..pool_size).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(pool_size),
            &active_ids,
            |b, active_ids| {
                b.iter(|| {
                    let active_set: HashSet<usize> =
                        active_ids.iter().copied().collect();
                    black_box(active_set);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vec_position,
    bench_hashset_contains,
    bench_hybrid_approach,
    bench_vec_position_worst_case,
    bench_vec_position_best_case,
    bench_hashset_construction
);
criterion_main!(benches);
