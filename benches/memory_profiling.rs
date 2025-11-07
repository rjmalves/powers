//! Memory profiling benchmarks for SDDP algorithm
//!
//! ## Profiling Approaches
//!
//! 1. **Lightweight tracking**: Uses /proc/self/status on Linux
//! 2. **Detailed analysis**: Use dhat for heap allocation profiling
//!
//! ## Key Metrics
//!
//! - Peak memory usage (RSS) during training
//! - Memory per training iteration
//! - Memory per cut stored
//! - Memory growth with problem size
//!

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::{SddpAlgorithm, SddpInstanceBuilder};
use powers_rs::system::{Bus, Hydro, System, Thermal};
use std::time::Instant;

/// Get current memory usage from /proc/self/status (Linux only)
///
/// Returns (VmRSS, VmSize) in bytes
/// - VmRSS: Resident Set Size (physical memory)
/// - VmSize: Virtual memory size
#[cfg(target_os = "linux")]
fn get_memory_usage() -> (usize, usize) {
    let status = std::fs::read_to_string("/proc/self/status")
        .expect("Failed to read /proc/self/status");

    let mut vm_rss = 0;
    let mut vm_size = 0;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            vm_rss = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
                * 1024; // Convert KB to bytes
        } else if line.starts_with("VmSize:") {
            vm_size = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0)
                * 1024; // Convert KB to bytes
        }
    }

    (vm_rss, vm_size)
}

#[cfg(not(target_os = "linux"))]
fn get_memory_usage() -> (usize, usize) {
    (0, 0) // Not supported on non-Linux platforms
}

/// Format bytes to human-readable string
fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    let bytes_f = bytes as f64;
    if bytes_f >= GB {
        format!("{:.2} GB", bytes_f / GB)
    } else if bytes_f >= MB {
        format!("{:.2} MB", bytes_f / MB)
    } else if bytes_f >= KB {
        format!("{:.2} KB", bytes_f / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Memory statistics collected during a benchmark
#[derive(Debug, Clone)]
struct MemoryStats {
    initial_rss: usize,
    peak_rss: usize,
    final_rss: usize,
    delta_rss: isize,
    samples: usize,
}

impl MemoryStats {
    fn new() -> Self {
        let (rss, _) = get_memory_usage();
        Self {
            initial_rss: rss,
            peak_rss: rss,
            final_rss: rss,
            delta_rss: 0,
            samples: 0,
        }
    }

    fn sample(&mut self) {
        let (rss, _) = get_memory_usage();
        self.peak_rss = self.peak_rss.max(rss);
        self.final_rss = rss;
        self.samples += 1;
    }

    fn finalize(&mut self) {
        self.delta_rss = self.final_rss as isize - self.initial_rss as isize;
    }

    fn report(&self, label: &str) {
        println!("\n{}", "=".repeat(70));
        println!("Memory Profile: {}", label);
        println!("{}", "=".repeat(70));
        println!("  Initial RSS:  {}", format_bytes(self.initial_rss));
        println!("  Peak RSS:     {}", format_bytes(self.peak_rss));
        println!("  Final RSS:    {}", format_bytes(self.final_rss));
        println!(
            "  Delta RSS:    {} ({})",
            if self.delta_rss >= 0 { "+" } else { "" },
            format_bytes(self.delta_rss.unsigned_abs())
        );
        println!("  Samples:      {}", self.samples);
        println!("{}", "=".repeat(70));
    }
}

/// Create a minimal single-reservoir system for benchmarking
fn create_single_reservoir_system() -> System {
    let bus = Bus::new(0, 500.0);
    let hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 0.01);
    let thermal = Thermal::new(0, 0, 50.0, 0.0, 100.0);
    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

/// Create a 2-stage problem for memory profiling
fn create_2stage_problem() -> (SddpAlgorithm, powers_rs::scenario::SAA) {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0])
        .num_stages(2)
        .deterministic_inflows(vec![vec![15.0], vec![25.0]])
        .deterministic_loads(vec![vec![50.0], vec![50.0]])
        .seed(42)
        .build_with_saa()
        .expect("Failed to create 2-stage problem")
}

/// Create a 12-stage problem for memory profiling
fn create_12stage_problem() -> (SddpAlgorithm, powers_rs::scenario::SAA) {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0])
        .num_stages(12)
        .deterministic_inflows(vec![
            vec![15.0],
            vec![15.0],
            vec![15.0],
            vec![20.0],
            vec![20.0],
            vec![20.0],
            vec![25.0],
            vec![25.0],
            vec![25.0],
            vec![20.0],
            vec![15.0],
            vec![15.0],
        ])
        .deterministic_loads(vec![vec![50.0]; 12])
        .seed(42)
        .build_with_saa()
        .expect("Failed to create 12-stage problem")
}

/// Benchmark: Memory usage during training iteration (2-stage)
fn memory_training_iteration_2stage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_training_iteration");
    group.sample_size(5); // Reduce sample size for faster benchmarking

    group.bench_function("2stage_1iter", |b| {
        b.iter_custom(|iters| {
            let mut stats = MemoryStats::new();
            let mut total_duration = std::time::Duration::ZERO;

            for _ in 0..iters {
                let (mut sddp, saa) = create_2stage_problem();

                let start = Instant::now();
                black_box(sddp.train(1, 1, &saa).expect("Training failed"));
                total_duration += start.elapsed();

                stats.sample();
            }

            stats.finalize();
            stats.report("2-stage training (1 iteration)");

            total_duration
        });
    });

    group.bench_function("2stage_10iter", |b| {
        b.iter_custom(|iters| {
            let mut stats = MemoryStats::new();
            let mut total_duration = std::time::Duration::ZERO;

            for _ in 0..iters {
                let (mut sddp, saa) = create_2stage_problem();

                let start = Instant::now();
                black_box(sddp.train(10, 1, &saa).expect("Training failed"));
                total_duration += start.elapsed();

                stats.sample();
            }

            stats.finalize();
            stats.report("2-stage training (10 iterations)");

            total_duration
        });
    });

    group.finish();
}

/// Benchmark: Memory usage during training iteration (12-stage)
fn memory_training_iteration_12stage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_training_iteration_12stage");
    group.sample_size(5); // Reduce sample size for faster benchmarking

    group.bench_function("12stage_1iter", |b| {
        b.iter_custom(|iters| {
            let mut stats = MemoryStats::new();
            let mut total_duration = std::time::Duration::ZERO;

            for _ in 0..iters {
                let (mut sddp, saa) = create_12stage_problem();

                let start = Instant::now();
                black_box(sddp.train(1, 1, &saa).expect("Training failed"));
                total_duration += start.elapsed();

                stats.sample();
            }

            stats.finalize();
            stats.report("12-stage training (1 iteration)");

            total_duration
        });
    });

    group.bench_function("12stage_10iter", |b| {
        b.iter_custom(|iters| {
            let mut stats = MemoryStats::new();
            let mut total_duration = std::time::Duration::ZERO;

            for _ in 0..iters {
                let (mut sddp, saa) = create_12stage_problem();

                let start = Instant::now();
                black_box(sddp.train(10, 1, &saa).expect("Training failed"));
                total_duration += start.elapsed();

                stats.sample();
            }

            stats.finalize();
            stats.report("12-stage training (10 iterations)");

            total_duration
        });
    });

    group.finish();
}

/// Benchmark: Memory scaling with forward passes (production-scale)
///
/// Measures peak RSS for Example 05 (60 stages, 156 hydros) varying the
/// number of forward passes per iteration. This is the CRITICAL scaling
/// factor for production problems.
///
/// Expected behavior: Memory should scale roughly linearly with
/// num_iterations × num_forward_passes, as each forward pass generates
/// cuts that are stored in the cut pool.
///
/// Uses SddpInstanceBuilder to programmatically vary forward passes
/// without creating multiple config files.
fn memory_production_forward_passes(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_production_fwd_passes");
    group.sample_size(5); // Reduce sample size (each run takes ~45s)

    // Use Example 05 configuration for production-scale profiling
    let config_path = "examples/05-large-scale-brazilian/config.json";
    let system_path = "examples/05-large-scale-brazilian/system.json";
    let graph_path = "examples/05-large-scale-brazilian/graph.json";
    let recourse_path = "examples/05-large-scale-brazilian/recourse.json";

    // Test different forward pass counts: 4, 8, 16 (production-scale values)
    // Memory should scale linearly with num_forward_passes
    for num_fwd in [4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("example05_8iter", format!("{}fwd", num_fwd)),
            num_fwd,
            |b, &num_fwd| {
                b.iter_custom(|iters| {
                    let mut stats = MemoryStats::new();
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        // Use SddpInstanceBuilder to modify forward passes programmatically
                        let mut sddp_instance = SddpInstanceBuilder::from_paths(
                            config_path,
                            system_path,
                            graph_path,
                            recourse_path,
                        )
                        .expect("Failed to load Example 05")
                        .with_num_iterations(8) // Fixed iterations for comparison
                        .with_num_forward_passes(num_fwd) // Vary forward passes
                        .with_num_threads(4) // Fixed thread count for consistent memory measurement
                        .build()
                        .expect("Failed to build SDDP instance");

                        let start = Instant::now();
                        let result = sddp_instance.train().expect("Training failed");
                        black_box(result);
                        total_duration += start.elapsed();

                        stats.sample();
                    }

                    stats.finalize();
                    stats.report(&format!(
                        "Example 05 (60 stages, 156 hydros) - 8 iters × {} fwd passes",
                        num_fwd
                    ));

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory scaling with iteration count (1 forward pass)
///
/// Uses 12-stage single-reservoir problem with varying iterations
/// to isolate iteration scaling effects on a simple problem.
fn memory_scaling_with_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling_iterations");
    group.sample_size(5); // Reduce sample size for faster benchmarking

    for num_iters in [5, 10, 20, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("12stage_1fwd", num_iters),
            num_iters,
            |b, &num_iters| {
                b.iter_custom(|iters| {
                    let mut stats = MemoryStats::new();
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let (mut sddp, saa) = create_12stage_problem();

                        let start = Instant::now();
                        black_box(
                            sddp.train(num_iters, 1, &saa)
                                .expect("Training failed"),
                        );
                        total_duration += start.elapsed();

                        stats.sample();
                    }

                    stats.finalize();
                    stats.report(&format!(
                        "12 stages, 1 reservoir, 1 fwd pass - {} iterations",
                        num_iters
                    ));

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory scaling with forward passes per iteration
///
/// Uses 12-stage single-reservoir problem with 10 iterations
/// and varying forward passes to isolate forward pass scaling.
fn memory_scaling_with_forward_passes(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling_forward_passes");
    group.sample_size(5); // Reduce sample size for faster benchmarking

    // Test: 1, 4, 8, 16, 32 forward passes (doubling pattern)
    for num_fwd in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("12stage_10iter", num_fwd),
            num_fwd,
            |b, &num_fwd| {
                b.iter_custom(|iters| {
                    let mut stats = MemoryStats::new();
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let (mut sddp, saa) = create_12stage_problem();

                        let start = Instant::now();
                        black_box(
                            sddp.train(10, num_fwd, &saa)
                                .expect("Training failed"),
                        );
                        total_duration += start.elapsed();

                        stats.sample();
                    }

                    stats.finalize();
                    stats.report(&format!(
                        "12 stages, 1 reservoir, {} fwd passes - 10 iterations",
                        num_fwd
                    ));

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory scaling with problem complexity (state dimension)
///
/// Uses 12-stage problems with varying numbers of reservoirs
/// to characterize O(N²) scaling hypothesis.
fn memory_scaling_with_state_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling_state_dim");
    group.sample_size(5); // Reduce sample size for faster benchmarking

    // Test: 1, 2, 4, 8 reservoirs (12 stages, 10 iterations, 8 forward passes)
    for num_reservoirs in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("12stage_10iter_8fwd", num_reservoirs),
            num_reservoirs,
            |b, &num_reservoirs| {
                b.iter_custom(|iters| {
                    let mut stats = MemoryStats::new();
                    let mut total_duration = std::time::Duration::ZERO;
                    let n_res = num_reservoirs; // Copy value to avoid lifetime issues

                    for _ in 0..iters {
                        // Create system with multiple reservoirs
                        let system_factory = move || {
                            let bus = Bus::new(0, 500.0);
                            let mut hydros = Vec::new();
                            for i in 0..n_res {
                                hydros.push(Hydro::new(
                                    i,
                                    None,
                                    0,
                                    1.0,
                                    0.0,
                                    100.0,
                                    0.0,
                                    100.0,
                                    0.01,
                                ));
                            }
                            let thermal = Thermal::new(0, 0, 50.0, 0.0, 100.0);
                            System::new(vec![bus], vec![], vec![thermal], hydros)
                        };

                        let initial_storage = vec![20.0; n_res];
                        let inflows = vec![vec![15.0; n_res]; 12];
                        let loads = vec![vec![50.0]; 12];

                        let (mut sddp, saa) = SddpAlgorithm::builder()
                            .system_factory(system_factory)
                            .initial_storage(initial_storage)
                            .num_stages(12)
                            .deterministic_inflows(inflows)
                            .deterministic_loads(loads)
                            .seed(42)
                            .build_with_saa()
                            .expect("Failed to create multi-reservoir problem");

                        let start = Instant::now();
                        black_box(
                            sddp.train(10, 8, &saa).expect("Training failed")
                        );
                        total_duration += start.elapsed();

                        stats.sample();
                    }

                    stats.finalize();
                    stats.report(&format!(
                        "12 stages, {} reservoirs, 8 fwd passes - 10 iterations",
                        num_reservoirs
                    ));

                    total_duration
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    memory_training_iteration_2stage,
    memory_training_iteration_12stage,
    memory_production_forward_passes,
    memory_scaling_with_iterations,
    memory_scaling_with_forward_passes,
    memory_scaling_with_state_dimension,
);
criterion_main!(benches);
