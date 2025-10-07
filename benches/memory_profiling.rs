//! Memory profiling benchmarks for SDDP algorithm
//!
//! This benchmark suite measures memory usage patterns to identify
//! allocation hotspots, memory growth, and optimization opportunities.
//!
//! Run with: `cargo bench --bench memory_profiling`
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

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpAlgorithm;
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
        .deterministic_loads(vec![50.0, 50.0])
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
        .deterministic_loads(vec![50.0; 12])
        .seed(42)
        .build_with_saa()
        .expect("Failed to create 12-stage problem")
}

/// Benchmark: Memory usage during training iteration (2-stage)
fn memory_training_iteration_2stage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_training_iteration");

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

/// Benchmark: Memory growth with iteration count
fn memory_growth_with_iterations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_growth");

    for num_iters in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("12stage", num_iters),
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
                        "12-stage training ({} iterations)",
                        num_iters
                    ));

                    total_duration
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Memory usage with varying problem sizes
fn memory_scaling_with_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_scaling");

    for num_stages in [2, 5, 12, 24].iter() {
        group.bench_with_input(
            BenchmarkId::new("training", num_stages),
            num_stages,
            |b, &num_stages| {
                b.iter_custom(|iters| {
                    let mut stats = MemoryStats::new();
                    let mut total_duration = std::time::Duration::ZERO;

                    for _ in 0..iters {
                        let inflows = vec![vec![20.0]; num_stages];
                        let loads = vec![50.0; num_stages];

                        let (mut sddp, saa) = SddpAlgorithm::builder()
                            .system_factory(create_single_reservoir_system)
                            .initial_storage(vec![20.0])
                            .num_stages(num_stages)
                            .deterministic_inflows(inflows)
                            .deterministic_loads(loads)
                            .seed(42)
                            .build_with_saa()
                            .expect("Failed to create problem");

                        let start = Instant::now();
                        black_box(
                            sddp.train(10, 1, &saa).expect("Training failed"),
                        );
                        total_duration += start.elapsed();

                        stats.sample();
                    }

                    stats.finalize();
                    stats.report(&format!(
                        "{}-stage training (10 iterations)",
                        num_stages
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
    memory_growth_with_iterations,
    memory_scaling_with_stages,
);
criterion_main!(benches);
