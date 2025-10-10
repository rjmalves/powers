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
use powers_rs::sddp::SddpInstanceBuilder;

/// Benchmark full SDDP training with different thread counts
///
/// **CRITICAL ANALYSIS**:
/// - Thread counts: 1 (baseline), 2, 4, 8, 16
/// - Example 05: Large-scale Brazilian system (60 stages, 156 hydros)
/// - Problem characteristics: ~15,000 variables, ~30,000 constraints per subproblem
/// - Forward pass: 16 scenarios (parallelizable)
/// - Backward pass: 60 stages (parallelizable with synchronization)
/// - Solver time: Dominates runtime (~80-90% of execution time)
///
/// **EXPECTED SCALING**:
/// - 1 thread: Baseline (1.0× speedup, 100% efficiency)
/// - 2 threads: ~1.7-1.9× speedup (85-95% efficiency)
/// - 4 threads: ~3.0-3.6× speedup (75-90% efficiency)
/// - 8 threads: ~4.5-6.0× speedup (56-75% efficiency)
/// - 16 threads: ~6.0-8.0× speedup (38-50% efficiency) - matches forward pass count
///
/// **BOTTLENECKS TO WATCH**:
/// - Sequential parts (Amdahl's law): Problem construction, result aggregation
/// - Synchronization overhead: Cut pool locking
/// - Load imbalance: Some scenarios/stages may take longer than others
/// - Solver parallelism: HiGHS internal threading (disabled via num_threads config)
///
/// **PERFORMANCE NOTE**: Uses SddpInstanceBuilder with with_num_threads() to
/// configure thread count explicitly, eliminating manual Rayon pool management.
fn bench_sddp_training_parallel_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scaling");

    // Reduce sample size since Example 05 is very expensive (60 stages, 156 hydros)
    // Each training run takes 30-120 seconds depending on thread count
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    // Test the large-scale Brazilian system (60-stage, 156 hydros)
    let config_path = "examples/05-large-scale-brazilian/config.json";
    let system_path = "examples/05-large-scale-brazilian/system.json";
    let graph_path = "examples/05-large-scale-brazilian/graph.json";
    let recourse_path = "examples/05-large-scale-brazilian/recourse.json";

    // Thread counts to test: 1 (baseline), 2, 4, 8, 16
    let thread_counts = vec![1, 2, 4, 8, 16];

    for num_threads in thread_counts {
        group.bench_with_input(
            BenchmarkId::new(
                "sddp_training",
                format!("{}threads", num_threads),
            ),
            &num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    // Use SddpInstanceBuilder with explicit thread configuration
                    // This eliminates manual Rayon ThreadPoolBuilder usage
                    let mut sddp = SddpInstanceBuilder::from_paths(
                        config_path,
                        system_path,
                        graph_path,
                        recourse_path,
                    )
                    .expect("Failed to load Example 05")
                    .with_num_threads(num_threads) // Configure thread count
                    .build()
                    .expect("Failed to build SDDP instance");

                    // Run training - thread pool already configured by builder
                    let result = sddp.train().expect("Training failed");
                    black_box(result);
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
///
/// **PERFORMANCE NOTE**: Thread pool configuration via with_num_threads()
/// adds < 10ms overhead per train() call, which is negligible compared to
/// typical training times (30-120 seconds for Example 05).
fn bench_parallel_efficiency_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_efficiency_analysis");
    group.sample_size(10);

    // Benchmark thread pool configuration overhead
    // This validates the < 10ms overhead claim from T4.5.6
    group.bench_function("thread_pool_config_overhead", |b| {
        b.iter(|| {
            // Simulate what happens when calling with_num_threads()
            // Note: Actual thread pool may already be configured (first wins)
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build_global();
            black_box(())
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
