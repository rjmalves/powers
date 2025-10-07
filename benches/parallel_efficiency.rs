// Parallel efficiency benchmarks for SDDP training
//
// Measures speedup and efficiency across different thread counts to analyze
// parallel scaling characteristics and identify bottlenecks.
//
// **PERFORMANCE**: This benchmark tests Rayon parallelism scaling by running
// the same problem with 1, 2, 4, and 8 threads. Expected efficiency: 70-90%
// (Amdahl's law + synchronization overhead).
//
// Run with: cargo bench --bench parallel_efficiency

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpAlgorithm;
use std::path::Path;

/// Benchmark full SDDP training with different thread counts
///
/// **CRITICAL ANALYSIS**:
/// - Thread counts: 1 (baseline), 2, 4, 8
/// - Example problem has 4 forward passes → optimal speedup at 4 threads
/// - Beyond 4 threads, expect diminishing returns (fewer forward passes than threads)
/// - Backward pass is also parallelized across stages
///
/// **EXPECTED SCALING**:
/// - 1 thread: Baseline (1.0× speedup, 100% efficiency)
/// - 2 threads: ~1.7-1.9× speedup (85-95% efficiency)
/// - 4 threads: ~3.0-3.6× speedup (75-90% efficiency) - matches forward pass count
/// - 8 threads: ~3.5-4.5× speedup (44-56% efficiency) - diminishing returns
///
/// **BOTTLENECKS TO WATCH**:
/// - Sequential parts (Amdahl's law): Problem construction, result aggregation
/// - Synchronization overhead: Thread spawning, cut pool locking
/// - Load imbalance: Some scenarios/stages may take longer than others
fn bench_sddp_training_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scaling");

    // Reduce sample size since SDDP training is expensive
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Test the example problem (12-stage hydrothermal dispatch)
    let config_path = Path::new("example/config.json");
    let system_path = Path::new("example/system.json");
    let graph_path = Path::new("example/graph.json");
    let recourse_path = Path::new("example/recourse.json");

    // Thread counts to test: 1 (baseline), 2, 4, 8
    // Note: Example config has 4 forward passes, so 4 threads should be optimal
    let thread_counts = vec![1, 2, 4, 8];

    for num_threads in thread_counts {
        group.bench_with_input(
            BenchmarkId::new(
                "sddp_training",
                format!("{}threads", num_threads),
            ),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    // Create a thread pool with the specified number of threads
                    // This ensures consistent thread count across iterations
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .build()
                        .expect("Failed to create thread pool");

                    // Run the benchmark within the thread pool
                    pool.install(|| {
                        let mut sddp = SddpAlgorithm::from_files(
                            config_path,
                            system_path,
                            graph_path,
                            recourse_path,
                        )
                        .expect("Failed to create SDDP instance");

                        // Run training (this is the expensive part)
                        // Rayon will use the thread pool we created
                        let result = sddp.train().expect("Training failed");
                        black_box(result);
                    })
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel efficiency metrics
///
/// **EFFICIENCY CALCULATION**:
/// - Speedup = T(1) / T(n) where T(n) is time with n threads
/// - Efficiency = Speedup / n × 100%
/// - Ideal efficiency: 100% (linear scaling)
/// - Good efficiency: 70-90% (typical for real workloads)
/// - Poor efficiency: <50% (too much overhead)
///
/// **ANALYSIS**:
/// After running: `cargo bench --bench parallel_efficiency`
/// Compare the median times:
///   Speedup(2) = Time(1 thread) / Time(2 threads)
///   Efficiency(2) = Speedup(2) / 2 × 100%
///
/// **EXAMPLE**:
///   1 thread:  10.0 seconds → baseline
///   2 threads:  5.5 seconds → 1.82× speedup, 91% efficiency ✅
///   4 threads:  3.0 seconds → 3.33× speedup, 83% efficiency ✅
///   8 threads:  2.2 seconds → 4.55× speedup, 57% efficiency ⚠️
///
/// ⚠️ If efficiency < 50%, investigate:
/// - Amdahl's law (sequential bottlenecks)
/// - Lock contention (cut pool, FCF)
/// - Load imbalance (uneven work distribution)
/// - Thread spawning overhead (pool creation cost)
fn bench_parallel_efficiency_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_efficiency_analysis");

    // Minimal samples for thread pool overhead measurement
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(5)); // Simple micro-benchmark to validate thread pool creation overhead
    group.bench_function("thread_pool_creation_overhead", |b| {
        b.iter(|| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .expect("Failed to create thread pool");

            pool.install(|| {
                // Minimal work to measure pool creation cost
                black_box(42)
            })
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sddp_training_parallel_scaling,
    bench_parallel_efficiency_analysis
);
criterion_main!(benches);
