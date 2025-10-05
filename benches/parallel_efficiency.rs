// Parallel efficiency benchmarks for SDDP training
//
// Measures speedup and efficiency across different thread counts to analyze
// parallel scaling characteristics and identify bottlenecks.
//
// Run with:
//   RAYON_NUM_THREADS=1 cargo bench --bench parallel_efficiency
//   RAYON_NUM_THREADS=2 cargo bench --bench parallel_efficiency
//   ...etc
//
// Or use the helper script:
//   ./scripts/bench_parallel_efficiency.sh

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpAlgorithm;
use std::path::Path;

/// Benchmark full SDDP training with different thread counts
///
/// This measures the end-to-end performance of the SDDP algorithm,
/// including forward pass, backward pass, and cut selection.
fn bench_sddp_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_training");

    // Reduce sample size since SDDP training is expensive
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Get thread count from environment (set by RAYON_NUM_THREADS)
    let thread_count = rayon::current_num_threads();

    // Test the example problem (12-stage hydrothermal dispatch)
    let config_path = Path::new("example/config.json");
    let system_path = Path::new("example/system.json");
    let graph_path = Path::new("example/graph.json");
    let recourse_path = Path::new("example/recourse.json");

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("{}threads", thread_count)),
        &(config_path, system_path, graph_path, recourse_path),
        |b, (config_path, system_path, graph_path, recourse_path)| {
            b.iter(|| {
                let mut sddp = SddpAlgorithm::from_files(
                    config_path,
                    system_path,
                    graph_path,
                    recourse_path,
                )
                .expect("Failed to create SDDP instance");

                // Run training (this is the expensive part)
                let result = sddp.train().expect("Training failed");
                black_box(result);
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_sddp_training);
criterion_main!(benches);
