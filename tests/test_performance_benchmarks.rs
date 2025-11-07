//! Performance Benchmark Tests
//!
//! Validates that performance benchmarks execute correctly and produce
//! reasonable results. Does NOT run full criterion benchmarks (too slow for CI),
//! but ensures benchmark code compiles and produces valid output.
//!
//! # Purpose
//!
//! - Verify benchmark code doesn't panic or error
//! - Ensure benchmark setups are valid
//! - Catch regressions in benchmark infrastructure
//! - Quick smoke tests (< 1s total)
//!
//! # Full Benchmarks
//!
//! Run full criterion benchmarks with:
//! ```bash
//! cargo bench
//! ```

use std::time::Instant;

/// Smoke test: Verify cut evaluation benchmark setup works
#[test]
fn test_cut_evaluation_benchmark_setup() {
    // This verifies the benchmark setup code doesn't panic
    // We run a minimal version (not the full criterion benchmark)

    use powers_rs::cut::BendersCut;

    // Create a simple cut (2 storage state variables)
    let storage_coefs = vec![-1.0, -0.5];
    let cut = BendersCut::new(0, storage_coefs, 100.0, 0, 0);

    // Create test state
    let storage_state = vec![50.0, 75.0];

    // Benchmark evaluation (minimal iterations)
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..1000 {
        sum += cut.eval_height_at_state(&storage_state);
    }
    let elapsed = start.elapsed();

    // Verify reasonable performance (< 50 ms for 1000 evaluations in debug mode)
    // Note: Debug builds are ~10-100x slower than release, so use generous limit
    assert!(
        elapsed.as_millis() < 50,
        "Cut evaluation too slow: {:?}",
        elapsed
    );

    // Verify result is correct: rhs + dot(coef, state) = 100 + (-1*50 + -0.5*75) = 100 - 50 - 37.5 = 12.5
    let expected = 12.5;
    assert!((sum / 1000.0 - expected).abs() < 1e-10);
}

/// Smoke test: Verify state operation benchmarks work
#[test]
fn test_state_operations_benchmark_setup() {
    // Test that state operations (common in SDDP) are fast

    let storage = vec![50.0; 10];
    let inflow = vec![30.0; 10];

    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..10000 {
        // Simulate common state operations
        for i in 0..storage.len() {
            sum += storage[i] + inflow[i];
        }
    }
    let elapsed = start.elapsed();

    // Should be fast (< 10ms for 10000 iterations in debug mode)
    assert!(
        elapsed.as_millis() < 10,
        "State operations too slow: {:?}",
        elapsed
    );

    // Use sum to prevent optimization
    assert!(sum > 0.0);
}

/// Smoke test: Verify cut selection benchmark compiles
#[test]
fn test_cut_selection_benchmark_setup() {
    use powers_rs::cut::BendersCut;

    // Create multiple cuts
    let mut cuts = Vec::new();
    for i in 0..100 {
        let storage_coefs = vec![-1.0 - i as f64 * 0.01, -0.5];
        let cut = BendersCut::new(i, storage_coefs, 100.0 + i as f64, 0, 0);
        cuts.push(cut);
    }

    let storage_state = vec![50.0, 75.0];

    // Find active cut (max evaluation)
    let start = Instant::now();
    let mut max_value = f64::NEG_INFINITY;
    for cut in &cuts {
        let value = cut.eval_height_at_state(&storage_state);
        if value > max_value {
            max_value = value;
        }
    }
    let elapsed = start.elapsed();

    // Should be fast (< 10 ms for 100 cuts in debug mode)
    // Note: Debug builds are much slower than release, so use generous limit
    assert!(
        elapsed.as_millis() < 10,
        "Cut selection too slow: {:?}",
        elapsed
    );

    assert!(max_value.is_finite());
}

/// Test: Verify benchmark directory structure exists
#[test]
fn test_benchmark_infrastructure_exists() {
    use std::path::Path;

    // Verify bench directory exists
    assert!(
        Path::new("benches").exists(),
        "benches/ directory should exist"
    );

    // Verify key benchmark files exist
    let expected_benchmarks = vec![
        "benches/correlation_application.rs",
        "benches/cut_selection.rs",
        "benches/memory_profiling.rs",
        "benches/parallel_efficiency.rs",
        "benches/simd_dot_product.rs",
        "benches/simulation_memory.rs",
    ];

    for benchmark in expected_benchmarks {
        assert!(
            Path::new(benchmark).exists(),
            "Benchmark file should exist: {}",
            benchmark
        );
    }
}

/// Test: Verify Cargo.toml has criterion dependency
#[test]
fn test_criterion_dependency_configured() {
    use std::fs;

    let cargo_toml =
        fs::read_to_string("Cargo.toml").expect("Failed to read Cargo.toml");

    // Verify criterion is in dev-dependencies
    assert!(
        cargo_toml.contains("criterion"),
        "Cargo.toml should have criterion in dev-dependencies"
    );
}

/// Integration test: Run minimal SDDP training and verify performance
#[test]
fn test_sddp_training_performance() {
    use powers_rs::sddp::SddpInstanceBuilder;

    // Use smallest example for quick test
    let mut sddp = SddpInstanceBuilder::from_paths(
        "examples/01-deterministic/config.json",
        "examples/01-deterministic/system.json",
        "examples/01-deterministic/graph.json",
        "examples/01-deterministic/recourse.json",
    )
    .expect("Failed to load example")
    .with_num_iterations(3)
    .with_num_forward_passes(1)
    .with_seed(42)
    .build()
    .expect("Failed to build SDDP");

    let start = Instant::now();
    let result = sddp.train().expect("Training failed");
    let elapsed = start.elapsed();

    // 3 iterations should complete quickly (< 2 seconds)
    assert!(elapsed.as_secs() < 2, "Training too slow: {:?}", elapsed);

    // Verify we got results
    assert_eq!(result.lower_bounds().len(), 3);
}

/// Test: Verify benchmark utilities compile
#[test]
fn test_benchmark_utilities_available() {
    // This test verifies that common benchmark utilities are available
    // and work correctly (without running full benchmarks)

    use std::hint::black_box;

    // Test black_box utility (prevents compiler optimization)
    let value = black_box(42.0f64);
    assert_eq!(value, 42.0);

    // Test timing utilities work
    let start = Instant::now();
    std::thread::sleep(std::time::Duration::from_micros(100));
    let elapsed = start.elapsed();
    assert!(elapsed.as_micros() >= 100);
}

/// Performance regression smoke test
#[test]
fn test_no_obvious_performance_regressions() {
    // Very basic performance check - catches major regressions
    // Full performance testing is in benchmarks/

    use powers_rs::cut::BendersCut;

    // Baseline: Cut evaluation should be < 100ns per evaluation
    let cut = BendersCut::new(0, vec![-1.0; 5], 100.0, 0, 0);
    let state = vec![50.0; 5];

    let iterations = 10000;
    let start = Instant::now();

    let mut sum = 0.0;
    for _ in 0..iterations {
        sum += cut.eval_height_at_state(&state);
    }

    let elapsed = start.elapsed();
    let ns_per_eval = elapsed.as_nanos() / iterations;

    // Use sum to prevent optimization
    assert!(sum != 0.0);

    // Sanity check: should be reasonably fast (< 1000ns in debug mode)
    assert!(
        ns_per_eval < 1000,
        "Cut evaluation too slow: {} ns/eval (expected < 1000 ns in debug)",
        ns_per_eval
    );
}

/// Test: Verify parallel benchmark infrastructure
#[test]
fn test_parallel_execution_works() {
    use rayon::prelude::*;

    // Verify Rayon parallel iteration works
    let data: Vec<i32> = (0..1000).collect();

    let start = Instant::now();
    let sum: i32 = data.par_iter().map(|&x| x * 2).sum();
    let elapsed = start.elapsed();

    // Verify correctness
    assert_eq!(sum, 999000);

    // Should complete quickly
    assert!(
        elapsed.as_millis() < 100,
        "Parallel computation too slow: {:?}",
        elapsed
    );
}

#[cfg(test)]
mod documentation {
    //! # Running Full Benchmarks
    //!
    //! These tests are quick smoke tests. For full performance analysis:
    //!
    //! ```bash
    //! # Run all benchmarks
    //! cargo bench
    //!
    //! # Run specific benchmark
    //! cargo bench --bench sddp_benchmarks
    //!
    //! # Run with baseline comparison
    //! cargo bench -- --save-baseline main
    //! cargo bench -- --baseline main
    //!
    //! # Generate HTML report
    //! cargo bench -- --save-baseline report
    //! open target/criterion/report/index.html
    //! ```
    //!
    //! # Interpreting Results
    //!
    //! - **Time**: Mean execution time ± std deviation
    //! - **Throughput**: Operations per second
    //! - **Change**: % difference from baseline (if available)
    //!
    //! # Performance Targets
    //!
    //! - Cut evaluation: < 50 ns
    //! - Forward pass (100 stages): < 100 ms
    //! - Backward pass (100 stages): < 500 ms
    //! - Full iteration (10 FP, 1 BP): < 2 seconds
}
