//! End-to-End SDDP Algorithm Benchmarks
//!
//! Measures the performance of the complete SDDP training and simulation workflow
//! using realistic problem instances (Example 05: Large-scale Brazilian system).
//!
//! ## Problem Size (Example 05)
//!
//! - **156 hydro plants** (realistic cascade)
//! - **121 thermal plants** (large generation mix)
//! - **5 buses** (regional interconnections)
//! - **60 stages** (5 years monthly planning)
//! - **Complex state space** (~156 dimensions)
//!
//! This is representative of production-scale hydrothermal dispatch problems.
//! Performance optimizations must be validated at this scale to be meaningful.
//!
//! ## Benchmark Groups
//!
//! 1. **Single Iteration** - One forward + backward pass (most important!)
//! 2. **Training Phases** - Forward vs backward breakdown
//! 3. **Problem Scaling** - Performance vs training effort
//! 4. **Simulation** - Out-of-sample policy simulation
//!
//! ## Metrics
//!
//! - Wall-clock time per iteration
//! - Solver calls per second
//! - Cuts generated per second
//! - Forward/backward phase timing
//! - Memory pressure (indirectly)
//!
//! ## Usage
//!
//! ```bash
//! # Run all E2E benchmarks (WARNING: Takes 30-60 minutes with 156 hydros!)
//! cargo bench --bench sddp_e2e
//!
//! # Run single iteration (fastest, most important, ~5-10 minutes)
//! cargo bench --bench sddp_e2e single_iteration
//!
//! # Save baseline before optimization
//! cargo bench --bench sddp_e2e -- --save-baseline before_opt
//!
//! # Compare after optimization
//! cargo bench --bench sddp_e2e -- --baseline before_opt
//! ```
//!
//! ## WARNING: Long Runtime
//!
//! These benchmarks use a **production-scale problem** (156 hydros, 60 stages).
//! Each benchmark group takes 5-20 minutes. Budget accordingly.
//!
//! ## Why Example 05?
//!
//! Small test cases (1-2 hydros) can give **misleading results**:
//! - Cache effects don't materialize
//! - Allocation overhead is hidden
//! - Parallelism opportunities invisible
//! - O(n²) algorithms appear fast
//!
//! Example 05 reveals real bottlenecks in production scenarios.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpInstanceBuilder;
use std::path::PathBuf;

// =============================================================================
// Helper: Load Example 05
// =============================================================================

/// Load Example 05: Large-scale Brazilian hydrothermal system
///
/// System characteristics:
/// - 156 hydro plants (realistic cascade)
/// - 121 thermal plants
/// - 5 buses
/// - 60 stages (5 years, monthly)
/// - State dimension: ~156 (storage levels)
///
/// This is a **production-scale problem** representative of real-world
/// hydrothermal dispatch optimization. Performance must be validated at this scale.
fn load_example_05() -> SddpInstanceBuilder {
    let base_path = PathBuf::from("examples/05-large-scale-brazilian");

    SddpInstanceBuilder::from_paths(
        base_path.join("config.json"),
        base_path.join("system.json"),
        base_path.join("graph.json"),
        base_path.join("recourse.json"),
    )
    .expect("Failed to load example 05")
}

// =============================================================================
// Benchmark Group 1: Single Iteration (MOST IMPORTANT!)
// =============================================================================
//
// This is the hot path - every iteration does forward+backward.
// Optimizations should target reducing single iteration time.
// With 156 hydros and 60 stages, this reveals real bottlenecks.

fn bench_single_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_single_iteration");

    // Reduced sample size for large problem (156 hydros takes ~30-60s per iteration!)
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Test with different forward pass counts
    // Note: Example 05 default is 16 forward passes
    for num_forward_passes in [2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("forward_passes", num_forward_passes),
            &num_forward_passes,
            |b, &num_fp| {
                b.iter_with_setup(
                    || {
                        // Setup: Load problem fresh each time
                        let builder = load_example_05();
                        builder
                            .with_num_iterations(1)
                            .with_num_forward_passes(num_fp)
                    },
                    |builder| {
                        // Benchmark: Run one iteration (156 hydros, 60 stages!)
                        let mut instance =
                            builder.build().expect("Build failed");
                        let result = instance.train().expect("Training failed");
                        black_box(result)
                    },
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// Benchmark Group 3: Training Phases (Measure Breakdown)
// =============================================================================

fn bench_training_phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_training_phases");

    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(40));

    group.bench_function("3_iterations_10_forward", |b| {
        b.iter_with_setup(
            || {
                let builder = load_example_05();
                builder
                    .with_num_iterations(3) // Reduced from 5 for large problem
                    .with_num_forward_passes(10)
            },
            |builder| {
                let mut instance = builder.build().expect("Build failed");
                let result = instance.train().expect("Training failed");

                // Extract timing breakdown - this is CRITICAL data!
                let iterations = result.iterations();
                let total_forward: std::time::Duration = iterations
                    .iter()
                    .map(|it| it.forward_timing.total_time)
                    .sum();
                let total_backward: std::time::Duration = iterations
                    .iter()
                    .map(|it| it.backward_timing.total_time)
                    .sum();

                println!("\n=== Phase Breakdown (156 hydros, 60 stages) ===");
                println!("  Forward:  {:?}", total_forward);
                println!("  Backward: {:?}", total_backward);
                println!(
                    "  Ratio:    {:.2}x",
                    total_backward.as_secs_f64() / total_forward.as_secs_f64()
                );

                black_box(result)
            },
        );
    });

    group.finish();
}
// =============================================================================
// Benchmark Group 3: Simulation (Out-of-Sample)
// =============================================================================
//
// Measures policy simulation performance after training.
// Less critical than training, but important for operational deployment.

fn bench_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_simulation");

    group.sample_size(10); // Minimum required by Criterion
    group.measurement_time(std::time::Duration::from_secs(30));

    // Note: Config default is 32 scenarios (from example 05 config.json)
    group.bench_function("32_scenarios", |b| {
        b.iter_with_setup(
            || {
                // Setup: Train a policy (not counted in benchmark time)
                let builder = load_example_05();
                let builder =
                    builder.with_num_iterations(5).with_num_forward_passes(10);
                let mut instance = builder.build().expect("Build failed");
                let _ = instance.train().expect("Training failed");
                instance
            },
            |mut instance| {
                // Benchmark: Run simulation
                let trajectories =
                    instance.simulate().expect("Simulation failed");
                black_box(trajectories)
            },
        );
    });

    group.finish();
}

// =============================================================================
// Benchmark Group 4: Problem Scaling
// =============================================================================
//
// Shows how performance scales with training effort (iterations × forward passes).
// Useful for understanding algorithm complexity and parallelization limits.

fn bench_problem_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("sddp_problem_scaling");

    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(50));

    // Test how performance scales with training effort
    // Note: Even "minimal" is large (156 hydros!)
    let configs = vec![
        ("minimal", 2, 5),  // Quick baseline
        ("default", 3, 10), // Reasonable training
    ];

    for (label, num_iter, num_fp) in configs {
        group.bench_with_input(
            BenchmarkId::new("config", label),
            &(num_iter, num_fp),
            |b, &(iter, fp)| {
                b.iter_with_setup(
                    || {
                        let builder = load_example_05();
                        builder
                            .with_num_iterations(iter)
                            .with_num_forward_passes(fp)
                    },
                    |builder| {
                        let mut instance =
                            builder.build().expect("Build failed");
                        let result = instance.train().expect("Training failed");

                        // Report solver efficiency - key metric!
                        let total_solver_calls: usize = result
                            .iterations()
                            .iter()
                            .map(|it| it.num_solver_calls)
                            .sum();
                        let total_time = result.total_time;
                        let calls_per_sec = total_solver_calls as f64
                            / total_time.as_secs_f64();

                        println!(
                            "\n{} config: {:.1} solver calls/sec (156 hydros!)",
                            label, calls_per_sec
                        );

                        black_box(result)
                    },
                );
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Configuration
// =============================================================================

criterion_group!(
    benches,
    bench_single_iteration, // MOST IMPORTANT - hot path
    bench_training_phases,  // Identifies bottleneck
    bench_simulation,       // Post-training performance
    bench_problem_scaling,  // Scaling characteristics
);

criterion_main!(benches);
