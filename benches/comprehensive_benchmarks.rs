//! Comprehensive Performance Benchmarks Using Production-Scale Examples
//!
//! This benchmark suite uses realistic problem instances from Examples 04 and 05
//! to provide meaningful performance metrics for production deployment.
//!
//! **Why These Examples**:
//! - Example 04: Cascade system (5 hydros, 24 stages, 20 branchings) - Medium complexity
//! - Example 05: Large-scale Brazilian (156 hydros, 60 stages) - Production scale
//!
//! **What We Measure**:
//! - Training iteration time (forward + backward passes) for both systems
//! - Cold start vs. warm iteration performance
//!
//! Run with: `cargo bench --bench comprehensive_benchmarks`
//! View results: `open target/criterion/report/index.html`

use criterion::{
    black_box, criterion_group, criterion_main, Criterion, SamplingMode,
};
use powers_rs::sddp::SddpAlgorithm;

/// Load Example 04 - Cascade System (5 hydros, 24 stages)
///
/// **System Characteristics**:
/// - 5 hydroelectric plants in series (cascade)
/// - 5 thermal plants backup
/// - 2 buses with transmission
/// - 24 stages (monthly, 2 years)
/// - 20 branchings per season (high stochasticity)
///
/// **Why This Example**:
/// - Realistic cascade coordination problem
/// - Medium-scale complexity (good for iteration benchmarks)
/// - Tests network optimization (2 buses + line)
/// - Sufficient complexity without overwhelming CI
fn load_cascade_system() -> powers_rs::sddp::SddpInstance {
    let base_path = "examples/04-cascade";

    SddpAlgorithm::from_files(
        format!("{}/config.json", base_path),
        format!("{}/system.json", base_path),
        format!("{}/graph.json", base_path),
        format!("{}/recourse.json", base_path),
    )
    .expect("Failed to load Example 04 - Cascade System")
}

/// Load Example 05 - Large-Scale Brazilian System (156 hydros, 60 stages)
///
/// **System Characteristics**:
/// - 156 hydroelectric plants across 33 cascades
/// - 121 thermal plants
/// - 5 buses with transmission network
/// - 60 stages (monthly, 5 years)
/// - Production-scale Brazilian hydrothermal system
///
/// **Why This Example**:
/// - Represents real-world deployment scale
/// - Stress-tests all SDDP components
/// - Validates production readiness
/// - Benchmark baseline for performance regressions
fn load_large_scale_system() -> powers_rs::sddp::SddpInstance {
    let base_path = "examples/05-large-scale-brazilian";

    SddpAlgorithm::from_files(
        format!("{}/config.json", base_path),
        format!("{}/system.json", base_path),
        format!("{}/graph.json", base_path),
        format!("{}/recourse.json", base_path),
    )
    .expect("Failed to load Example 05 - Large-Scale Brazilian System")
}

// ============================================================================
// BENCHMARK GROUP 1: Training Iteration Performance
// ============================================================================

/// Benchmark Group 1: Training Iteration Performance - CASCADE SYSTEM
///
/// Measures complete training iterations on Example 04 (5-hydro cascade, 24 stages).
/// This is the primary metric for medium-complexity problems.
fn training_iteration_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_iteration_cascade");

    // Configure for medium-duration benchmarks
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));

    // Benchmark: Single iteration on pre-trained system
    group.bench_function("single_iteration_after_warmup", |b| {
        b.iter_with_setup(
            || {
                // Load system and pre-train with 3 iterations
                let mut sddp = load_cascade_system();
                for _ in 0..3 {
                    let _ = sddp.train();
                }
                sddp
            },
            |mut sddp| {
                // Measure single iteration after warmup
                black_box(sddp.train())
            },
        );
    });

    // Benchmark: First iteration (cold start)
    group.bench_function("single_iteration_cold_start", |b| {
        b.iter_with_setup(load_cascade_system, |mut sddp| {
            black_box(sddp.train())
        });
    });

    group.finish();
}

/// Benchmark Group 2: Training Iteration Performance - LARGE-SCALE SYSTEM
///
/// Measures complete training iterations on Example 05 (156 hydros, 60 stages).
/// This is the production-scale benchmark.
fn training_iteration_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_iteration_large_scale");

    // Configure for very long-running benchmarks
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10); // Minimum samples for statistics
    group.measurement_time(std::time::Duration::from_secs(300)); // 5 minutes per benchmark
    group.warm_up_time(std::time::Duration::from_secs(10));

    // Benchmark: Single iteration on pre-trained system
    group.bench_function("single_iteration_after_warmup", |b| {
        b.iter_with_setup(
            || {
                // Load system and pre-train with 1 iteration (expensive!)
                let mut sddp = load_large_scale_system();
                let _ = sddp.train();
                sddp
            },
            |mut sddp| {
                // Measure single iteration after warmup
                black_box(sddp.train())
            },
        );
    });

    // Benchmark: First iteration (cold start)
    group.bench_function("single_iteration_cold_start", |b| {
        b.iter_with_setup(load_large_scale_system, |mut sddp| {
            black_box(sddp.train())
        });
    });

    group.finish();
}

// ============================================================================
// CRITERION GROUP REGISTRATION
// ============================================================================

criterion_group!(
    comprehensive_benches,
    training_iteration_cascade,
    training_iteration_large_scale,
);

criterion_main!(comprehensive_benches);
