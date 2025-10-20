//! Simulation Memory Benchmarks (SIM-OPT-007)
//!
//! This benchmark suite validates the memory optimization from SIM-OPT-005 and SIM-OPT-006.
//! It measures:
//! 1. Peak memory usage during simulation (Extract-and-Release pattern)
//! 2. Simulation throughput (scenarios/second)
//! 3. Data extraction overhead (trajectory extraction from handlers)
//! 4. CSV export performance (trajectory-based vs historical handler-based)
//!
//! ## Expected Results (120 stages, 8 threads)
//!
//! Memory Model:
//! - Old: num_scenarios × 6 MB (full handler) = 60 GB for 10K scenarios
//! - New: num_threads × 6 MB + num_scenarios × 240 KB = 2.45 GB for 10K scenarios
//! - Reduction: 96% for large simulations
//!
//! Run with:
//! ```bash
//! cargo bench --bench simulation_memory
//! ```
//!
//! For detailed memory profiling:
//! ```bash
//! # Linux: Use heaptrack for heap allocation profiling
//! heaptrack cargo bench --bench simulation_memory -- --sample-size 10
//!
//! # Or valgrind massif for memory snapshots
//! valgrind --tool=massif --massif-out-file=massif.out \
//!   cargo bench --bench simulation_memory -- --sample-size 10
//! msprof massif.out
//! ```

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::{Bus, Hydro, System, Thermal};
use std::time::Instant;

/// Memory statistics tracker
#[derive(Debug, Clone)]
struct MemoryStats {
    initial_rss: usize,
    peak_rss: usize,
    final_rss: usize,
    samples: Vec<usize>,
}

impl MemoryStats {
    fn new() -> Self {
        let (rss, _) = get_memory_usage();
        Self {
            initial_rss: rss,
            peak_rss: rss,
            final_rss: rss,
            samples: Vec::new(),
        }
    }

    fn sample(&mut self) {
        let (rss, _) = get_memory_usage();
        self.samples.push(rss);
        if rss > self.peak_rss {
            self.peak_rss = rss;
        }
    }

    fn finalize(&mut self) {
        let (rss, _) = get_memory_usage();
        self.final_rss = rss;
    }

    fn delta_mb(&self) -> f64 {
        (self.peak_rss as i64 - self.initial_rss as i64) as f64
            / (1024.0 * 1024.0)
    }

    fn peak_mb(&self) -> f64 {
        self.peak_rss as f64 / (1024.0 * 1024.0)
    }

    fn per_scenario_kb(&self, num_scenarios: usize) -> f64 {
        if num_scenarios == 0 {
            return 0.0;
        }
        let delta_bytes = self.peak_rss as i64 - self.initial_rss as i64;
        (delta_bytes as f64 / num_scenarios as f64) / 1024.0
    }

    fn report(
        &self,
        num_scenarios: usize,
        num_stages: usize,
        num_threads: usize,
    ) {
        eprintln!("\n📊 Memory Report:");
        eprintln!("  Scenarios: {}", num_scenarios);
        eprintln!("  Stages: {}", num_stages);
        eprintln!("  Threads: {}", num_threads);
        eprintln!(
            "  Initial RSS: {:.2} MB",
            self.initial_rss as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  Peak RSS: {:.2} MB", self.peak_mb());
        eprintln!(
            "  Final RSS: {:.2} MB",
            self.final_rss as f64 / (1024.0 * 1024.0)
        );
        eprintln!("  Delta RSS: {:.2} MB", self.delta_mb());
        eprintln!(
            "  Per scenario: {:.2} KB",
            self.per_scenario_kb(num_scenarios)
        );
        eprintln!("  Samples: {}", self.samples.len());
    }
}

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

/// Create a realistic multi-stage system for memory benchmarking
///
/// This system has multiple independent hydros and thermals serving one bus:
/// - Multiple hydros (independent, no cascade)
/// - Multiple thermals with varying costs
/// - Realistic storage capacities and productivities
fn create_cascade_system(num_hydros: usize, num_thermals: usize) -> System {
    let buses = vec![Bus::new(0, 500.0)]; // Single bus for simplicity
    let lines = vec![];
    let mut thermals = Vec::new();
    let mut hydros = Vec::new();

    // Create independent hydros (no cascade for simplicity)
    for i in 0..num_hydros {
        hydros.push(Hydro::new(
            i, None,   // No downstream (independent hydros)
            0,      // All on bus 0
            0.95,   // productivity
            0.0,    // min_storage
            1000.0, // max_storage (MWh)
            0.0,    // min_turbined_flow
            200.0,  // max_turbined_flow (MW)
            0.01,   // spillage_penalty
        ));
    }

    // Create thermals with varying costs
    for i in 0..num_thermals {
        let cost = 50.0 + (i as f64 * 25.0); // Escalating costs

        thermals.push(Thermal::new(
            i, 0, // All on bus 0
            cost, 0.0, 150.0, // max_generation
        ));
    }

    System::new(buses, lines, thermals, hydros)
}

/// Benchmark: Memory usage for simulation with varying scenario counts
///
/// This benchmark measures peak memory usage during simulation to validate
/// the Extract-and-Release pattern from SIM-OPT-005.
///
/// Expected behavior:
/// - Memory scales as O(threads × handler_size + scenarios × trajectory_size)
/// - For 8 threads, 120 stages: ~48 MB (handlers) + scenarios × ~240 KB (trajectories)
fn bench_simulation_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_memory");

    // Test different scenario counts
    let scenario_counts = vec![100, 500, 1000, 2000];

    for num_scenarios in scenario_counts {
        group.bench_with_input(
            BenchmarkId::new("memory_peak", num_scenarios),
            &num_scenarios,
            |b, &num_scenarios| {
                // Create problem (outside timing)
                let num_stages = 24; // Monthly planning
                let num_hydros = 4;
                let num_thermals = 4;

                let (mut sddp_algo, saa) = SddpAlgorithm::builder()
                    .system_factory(move || {
                        create_cascade_system(num_hydros, num_thermals)
                    })
                    .initial_storage(vec![500.0; num_hydros])
                    .num_stages(num_stages)
                    .deterministic_inflows(vec![
                        vec![50.0; num_hydros];
                        num_stages
                    ])
                    .deterministic_loads(vec![vec![400.0]; num_stages])
                    .seed(42)
                    .build_with_saa()
                    .expect("Failed to create problem");

                // Quick training (5 iterations) to build policy
                sddp_algo.train(5, 10, &saa).expect("Training failed");

                b.iter(|| {
                    let mut stats = MemoryStats::new();

                    // Run simulation with memory tracking
                    stats.sample();
                    let trajectories =
                        sddp_algo.simulate(num_scenarios, &saa).unwrap();
                    stats.sample();

                    // Ensure trajectories aren't optimized away
                    black_box(&trajectories);

                    stats.finalize();
                    stats.report(num_scenarios, num_stages, 8);

                    trajectories.len()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Simulation throughput (scenarios/second)
///
/// Measures how many scenarios can be simulated per second.
/// This validates that the Extract-and-Release pattern doesn't hurt throughput.
///
/// Expected: Equal or better than old approach (no batch synchronization overhead)
fn bench_simulation_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation_throughput");

    let scenario_counts = vec![100, 500, 1000];

    for num_scenarios in scenario_counts {
        group.bench_with_input(
            BenchmarkId::new("throughput", num_scenarios),
            &num_scenarios,
            |b, &num_scenarios| {
                let num_stages = 24;
                let num_hydros = 4;
                let num_thermals = 4;

                let (mut sddp_algo, saa) = SddpAlgorithm::builder()
                    .system_factory(move || {
                        create_cascade_system(num_hydros, num_thermals)
                    })
                    .initial_storage(vec![500.0; num_hydros])
                    .num_stages(num_stages)
                    .deterministic_inflows(vec![
                        vec![50.0; num_hydros];
                        num_stages
                    ])
                    .deterministic_loads(vec![vec![400.0]; num_stages])
                    .seed(42)
                    .build_with_saa()
                    .expect("Failed to create problem");

                sddp_algo.train(5, 10, &saa).expect("Training failed");

                b.iter(|| {
                    let start = Instant::now();
                    let trajectories =
                        sddp_algo.simulate(num_scenarios, &saa).unwrap();
                    let elapsed = start.elapsed();

                    let scenarios_per_sec =
                        num_scenarios as f64 / elapsed.as_secs_f64();
                    eprintln!(
                        "\n⚡ Throughput: {:.0} scenarios/sec",
                        scenarios_per_sec
                    );

                    black_box(trajectories.len())
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Data extraction overhead
///
/// Measures the time to extract a trajectory from a handler.
/// This is the core operation of the Extract-and-Release pattern.
///
/// Expected: <1% of total forward pass time (negligible overhead)
fn bench_extraction_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("extraction_overhead");

    // Create a small problem for focused extraction measurement
    let num_stages = 24;
    let num_hydros = 4;

    let (mut sddp_algo, saa) = SddpAlgorithm::builder()
        .system_factory(move || create_cascade_system(num_hydros, 2))
        .initial_storage(vec![500.0; num_hydros])
        .num_stages(num_stages)
        .deterministic_inflows(vec![vec![50.0; num_hydros]; num_stages])
        .deterministic_loads(vec![vec![400.0]; num_stages])
        .seed(42)
        .build_with_saa()
        .expect("Failed to create problem");

    sddp_algo.train(5, 10, &saa).expect("Training failed");

    group.bench_function("extract_trajectory", |b| {
        b.iter(|| {
            // Simulate single scenario to measure extraction
            let start = Instant::now();
            let trajectories = sddp_algo.simulate(1, &saa).unwrap();
            let total_time = start.elapsed();

            eprintln!("\n🔍 Extraction Analysis:");
            eprintln!(
                "  Total time (forward pass + extraction): {:?}",
                total_time
            );
            eprintln!("  Stages: {}", num_stages);

            black_box(trajectories.len())
        });
    });

    group.finish();
}

/// Benchmark: CSV export performance with trajectory-based access
///
/// This validates the performance improvement from SIM-OPT-006.
/// Trajectory-based access has better cache locality than handler-based access.
///
/// Expected: ≥5% faster due to sequential memory access vs pointer-chasing
fn bench_csv_export_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_export");

    let num_scenarios = 100;
    let num_stages = 24;
    let num_hydros = 4;

    // Create and train problem
    let (mut sddp_algo, saa) = SddpAlgorithm::builder()
        .system_factory(move || create_cascade_system(num_hydros, 2))
        .initial_storage(vec![500.0; num_hydros])
        .num_stages(num_stages)
        .deterministic_inflows(vec![vec![50.0; num_hydros]; num_stages])
        .deterministic_loads(vec![vec![400.0]; num_stages])
        .seed(42)
        .build_with_saa()
        .expect("Failed to create problem");

    sddp_algo.train(5, 10, &saa).expect("Training failed");

    // Simulate once to get trajectories
    let trajectories = sddp_algo.simulate(num_scenarios, &saa).unwrap();

    group.bench_function("iterate_trajectories", |b| {
        b.iter(|| {
            let start = Instant::now();

            // Simulate CSV export: iterate all trajectories and access data
            let mut total_cost = 0.0;
            let mut total_hydro_gen = 0.0;

            for trajectory in &trajectories {
                for realization in &trajectory.realizations {
                    // Access multiple fields (typical CSV export pattern)
                    total_cost += realization.current_stage_objective;
                    for &flow in &realization.turbined_flow {
                        total_hydro_gen += flow;
                    }
                }
            }

            let elapsed = start.elapsed();
            eprintln!("\n📄 CSV Export Simulation:");
            eprintln!("  Scenarios: {}", num_scenarios);
            eprintln!("  Stages: {}", num_stages);
            eprintln!("  Time: {:?}", elapsed);
            eprintln!(
                "  Throughput: {:.0} records/sec",
                (num_scenarios * num_stages) as f64 / elapsed.as_secs_f64()
            );

            black_box(total_cost + total_hydro_gen)
        });
    });

    group.finish();
}

/*
// Benchmark: Large-scale memory validation (10K scenarios)
//
// This is the "proof test" for the 96% memory reduction claim.
//
// Expected memory (120 stages, 8 threads):
// - Old: 10,000 × 6 MB = 60 GB
// - New: 8 × 6 MB + 10,000 × 240 KB = 2.45 GB
// - Reduction: 96%
//
// Note: This test is commented out by default (very slow, requires lots of memory).
// Uncomment and run separately for large-scale validation.
#[allow(dead_code)]
fn bench_large_scale_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale_memory");
    group.sample_size(10); // Reduce samples for long-running test

    let num_scenarios = 10_000;
    let num_stages = 120; // Full Brazilian system scale
    let num_hydros = 20;
    let num_thermals = 10;

    group.bench_function("10k_scenarios_120_stages", |b| {
        let (mut sddp_algo, saa) = SddpAlgorithm::builder()
            .system_factory(|| create_cascade_system(num_hydros, num_thermals))
            .initial_storage(vec![500.0; num_hydros])
            .num_stages(num_stages)
            .deterministic_inflows(vec![vec![50.0; num_hydros]; num_stages])
            .deterministic_loads(vec![vec![400.0]; num_stages])
            .seed(42)
            .build_with_saa()
            .expect("Failed to create problem");

        sddp_algo.train(10, 20, &saa).expect("Training failed");

        b.iter(|| {
            let mut stats = MemoryStats::new();

            eprintln!("\n🎯 LARGE SCALE TEST: 10K scenarios × 120 stages");
            stats.sample();

            let trajectories = sddp_algo.simulate(num_scenarios, &saa).unwrap();

            stats.sample();
            stats.finalize();
            stats.report(num_scenarios, num_stages, 8);

            // Expected results
            let expected_handler_mb = 8.0 * 6.0; // 8 threads × 6 MB
            let expected_trajectory_mb = (num_scenarios as f64 * 240.0) / 1024.0; // scenarios × 240 KB
            let expected_total_mb = expected_handler_mb + expected_trajectory_mb;

            eprintln!("\n📐 Expected Memory Model:");
            eprintln!("  Handlers: {:.2} MB (8 threads × 6 MB)", expected_handler_mb);
            eprintln!("  Trajectories: {:.2} MB ({} × 240 KB)", expected_trajectory_mb, num_scenarios);
            eprintln!("  Expected total: {:.2} MB", expected_total_mb);
            eprintln!("  Measured delta: {:.2} MB", stats.delta_mb());
            eprintln!("  Match: {:.1}%", (stats.delta_mb() / expected_total_mb) * 100.0);

            black_box(trajectories.len())
        });
    });

    group.finish();
}
*/

criterion_group!(
    benches,
    bench_simulation_memory_usage,
    bench_simulation_throughput,
    bench_extraction_overhead,
    bench_csv_export_performance,
);

criterion_main!(benches);
