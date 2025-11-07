// Parallel efficiency benchmarks for SDDP training
//
// Measures speedup and efficiency across different thread counts to analyze
// parallel scaling characteristics and identify bottlenecks.
//
use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpInstanceBuilder;

/// Benchmark full SDDP training with different thread counts
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
/// - Speedup = T(1) / T(n) where T(n) is time with n threads
/// - Efficiency = Speedup / n × 100%
fn bench_parallel_efficiency_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_efficiency_analysis");
    group.sample_size(10);

    // Benchmark thread pool configuration overhead
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
