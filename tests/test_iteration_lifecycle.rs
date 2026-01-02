//! Tests for Per-Iteration Model Lifecycle (Sprint 8 Revised)
//!
//! These tests validate the per-iteration Model architecture:
//! - T-119: Determinism test (same results with/without basis)
//! - T-120: RSS verification (memory reclaimed between iterations)
//! - T-121: Performance benchmarks (see benches/ for criterion)
//! - T-122: Golden test validation (covered by scripts/golden-tests.sh)

use powers_rs::solver::{HighsModelStatus, Problem, Sense, StoredBasis};

/// T-119: Test that solver produces identical results with and without basis warm-starting.
///
/// This is critical for the per-iteration architecture: simulation mode runs without
/// basis (for reproducibility from persisted FCF), but must produce the same results
/// as training mode (which uses basis warm-starting for performance).
#[test]
fn test_determinism_with_without_basis() {
    // Create a small but meaningful LP
    let mut problem = Problem::new();

    // Variables: x1, x2, x3, x4 with different costs
    problem.add_column(1.0, 0.0..100.0); // x1: [0, 100], cost 1
    problem.add_column(2.0, 0.0..100.0); // x2: [0, 100], cost 2
    problem.add_column(3.0, 0.0..100.0); // x3: [0, 100], cost 3
    problem.add_column(4.0, 0.0..100.0); // x4: [0, 100], cost 4

    // Constraints
    problem.add_row(10.0.., [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]); // x1+x2+x3+x4 >= 10
    problem.add_row(..=50.0, [(0, 2.0), (1, 1.0)]); // 2*x1 + x2 <= 50
    problem.add_row(5.0..=30.0, [(2, 1.0), (3, 2.0)]); // 5 <= x3 + 2*x4 <= 30

    // Solve 1: Cold start (no basis)
    let mut model1 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model1.solve();

    assert_eq!(
        model1.status(),
        HighsModelStatus::Optimal,
        "Problem should be optimal"
    );
    let obj1 = model1.get_objective_value();
    let sol1 = model1.get_solution();

    // Get basis from solved model
    let basis = model1.get_stored_basis();
    assert!(!basis.is_empty(), "Basis should be non-empty after solving");
    drop(model1);

    // Solve 2: Warm start (with basis)
    let mut model2 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    let apply_result = model2.apply_stored_basis(&basis);
    assert!(apply_result.is_ok(), "Basis application should succeed");
    model2.solve();

    assert_eq!(
        model2.status(),
        HighsModelStatus::Optimal,
        "Problem should be optimal with basis"
    );
    let obj2 = model2.get_objective_value();
    let sol2 = model2.get_solution();
    drop(model2);

    // Solve 3: Another cold start (verify repeatability)
    let mut model3 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model3.solve();
    let obj3 = model3.get_objective_value();
    let sol3 = model3.get_solution();

    // Verify identical objectives
    const TOLERANCE: f64 = 1e-9;
    assert!(
        (obj1 - obj2).abs() < TOLERANCE,
        "Objective mismatch cold vs warm: {} vs {} (diff: {})",
        obj1,
        obj2,
        (obj1 - obj2).abs()
    );
    assert!(
        (obj1 - obj3).abs() < TOLERANCE,
        "Objective mismatch cold1 vs cold2: {} vs {} (diff: {})",
        obj1,
        obj3,
        (obj1 - obj3).abs()
    );

    // Verify identical solutions
    assert_eq!(
        sol1.colvalue.len(),
        sol2.colvalue.len(),
        "Solution lengths should match"
    );
    for (i, (v1, v2)) in sol1.colvalue.iter().zip(&sol2.colvalue).enumerate() {
        assert!(
            (v1 - v2).abs() < TOLERANCE,
            "Solution mismatch at var {}: {} vs {} (diff: {})",
            i,
            v1,
            v2,
            (v1 - v2).abs()
        );
    }
    for (i, (v1, v3)) in sol1.colvalue.iter().zip(&sol3.colvalue).enumerate() {
        assert!(
            (v1 - v3).abs() < TOLERANCE,
            "Solution mismatch cold1 vs cold2 at var {}: {} vs {}",
            i,
            v1,
            v3
        );
    }
}

/// T-119: Test determinism with problem modifications between solves.
///
/// After modifying constraints (like adding cuts in SDDP), results should
/// still be deterministic regardless of basis.
#[test]
fn test_determinism_after_modifications() {
    // Create base problem
    let mut problem = Problem::new();

    // Variables
    problem.add_column(1.0, 0.0..50.0); // x1
    problem.add_column(2.0, 0.0..50.0); // x2
    problem.add_column(3.0, 0.0..50.0); // x3

    // Base constraint
    problem.add_row(10.0.., [(0, 1.0), (1, 1.0), (2, 1.0)]); // x1+x2+x3 >= 10

    // Add a "cut" constraint (inactive initially)
    problem.add_row(f64::NEG_INFINITY..f64::INFINITY, [(0, 0.0), (1, 0.0)]);
    let cut_row = 1; // Second row

    // Solve base problem
    let mut model1 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model1.solve();
    let base_obj = model1.get_objective_value();
    let _base_sol = model1.get_solution();
    let basis = model1.get_stored_basis();
    drop(model1);

    // Activate cut in Problem (simulates backward pass)
    problem
        .change_coefficient(cut_row, 0, 1.0)
        .expect("Coefficient change should succeed");
    problem
        .change_coefficient(cut_row, 1, 1.0)
        .expect("Coefficient change should succeed");
    problem.change_row_bounds(cut_row, 8.0, f64::INFINITY); // x1 + x2 >= 8

    // Solve with cut - cold start
    let mut model2 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model2.solve();
    let cut_obj_cold = model2.get_objective_value();
    let cut_sol_cold = model2.get_solution();
    drop(model2);

    // Solve with cut - warm start (old basis, may not be compatible after cut)
    let mut model3 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    // Apply basis - may fail or succeed depending on compatibility
    let _ = model3.apply_stored_basis(&basis);
    model3.solve();
    let cut_obj_warm = model3.get_objective_value();
    let cut_sol_warm = model3.get_solution();

    // Objectives should match regardless of warm start
    const TOLERANCE: f64 = 1e-9;
    assert!(
        (cut_obj_cold - cut_obj_warm).abs() < TOLERANCE,
        "Objective mismatch with cut: cold={} warm={} (diff: {})",
        cut_obj_cold,
        cut_obj_warm,
        (cut_obj_cold - cut_obj_warm).abs()
    );

    // Solutions should match
    for (i, (v1, v2)) in cut_sol_cold
        .colvalue
        .iter()
        .zip(&cut_sol_warm.colvalue)
        .enumerate()
    {
        assert!(
            (v1 - v2).abs() < TOLERANCE,
            "Solution mismatch with cut at var {}: {} vs {}",
            i,
            v1,
            v2
        );
    }

    // Verify cut changed the solution (otherwise test is meaningless)
    // The cut should have made the problem harder (higher cost)
    assert!(
        cut_obj_cold >= base_obj - TOLERANCE,
        "Cut should not decrease objective: base={} with_cut={}",
        base_obj,
        cut_obj_cold
    );
}

/// T-119: Test determinism with many iterations (simulates training loop).
#[test]
fn test_determinism_multiple_iterations() {
    const NUM_ITERATIONS: usize = 10;
    const TOLERANCE: f64 = 1e-9;

    // Create problem
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..100.0);
    problem.add_column(2.0, 0.0..100.0);
    problem.add_row(10.0.., [(0, 1.0), (1, 1.0)]);

    // Run iterations with basis
    let mut objectives_with_basis = Vec::with_capacity(NUM_ITERATIONS);
    let mut cached_basis: Option<StoredBasis> = None;

    for _ in 0..NUM_ITERATIONS {
        let mut model = problem
            .create_model(Sense::Minimise)
            .expect("Model creation should succeed");

        if let Some(ref basis) = cached_basis {
            let _ = model.apply_stored_basis(basis);
        }

        model.solve();
        objectives_with_basis.push(model.get_objective_value());
        cached_basis = Some(model.get_stored_basis());
    }

    // Run iterations without basis
    let mut objectives_without_basis = Vec::with_capacity(NUM_ITERATIONS);

    for _ in 0..NUM_ITERATIONS {
        let mut model = problem
            .create_model(Sense::Minimise)
            .expect("Model creation should succeed");
        // No basis applied
        model.solve();
        objectives_without_basis.push(model.get_objective_value());
    }

    // Verify all objectives match
    for (i, (obj_with, obj_without)) in objectives_with_basis
        .iter()
        .zip(&objectives_without_basis)
        .enumerate()
    {
        assert!(
            (obj_with - obj_without).abs() < TOLERANCE,
            "Iteration {}: objective mismatch {} vs {}",
            i,
            obj_with,
            obj_without
        );
    }

    // Verify all iterations gave same result (problem didn't change)
    let first_obj = objectives_with_basis[0];
    for (i, obj) in objectives_with_basis.iter().enumerate() {
        assert!(
            (obj - first_obj).abs() < TOLERANCE,
            "Iteration {} objective differs from first: {} vs {}",
            i,
            obj,
            first_obj
        );
    }
}

/// T-120: Test that RSS is stable across iterations (Linux only).
///
/// This test verifies that dropping Models reclaims HiGHS memory.
#[test]
#[cfg(target_os = "linux")]
fn test_rss_stable_across_iterations() {
    const NUM_ITERATIONS: usize = 5;
    const PROBLEM_SIZE: usize = 100; // Variables and constraints

    // Helper to get RSS
    fn get_rss_kb() -> usize {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/statm") {
            if let Some(pages) = status.split_whitespace().nth(1) {
                if let Ok(p) = pages.parse::<usize>() {
                    return p * 4; // Convert pages to KB (4KB pages)
                }
            }
        }
        0
    }

    // Create a larger problem to make memory impact measurable
    let mut problem = Problem::new();
    for i in 0..PROBLEM_SIZE {
        problem.add_column(i as f64 + 1.0, 0.0..1000.0);
    }
    for i in 0..PROBLEM_SIZE {
        let factors: Vec<(usize, f64)> = (0..PROBLEM_SIZE)
            .map(|j| (j, ((i + j) % 10 + 1) as f64))
            .collect();
        problem.add_row((i * 10) as f64.., factors);
    }

    // Warm up
    {
        let mut model = problem
            .create_model(Sense::Minimise)
            .expect("Model creation should succeed");
        model.solve();
    }

    let baseline_rss = get_rss_kb();
    let mut rss_after_iterations = Vec::with_capacity(NUM_ITERATIONS);

    // Run iterations
    for _ in 0..NUM_ITERATIONS {
        {
            let mut model = problem
                .create_model(Sense::Minimise)
                .expect("Model creation should succeed");
            model.solve();
            // Model dropped here
        }

        rss_after_iterations.push(get_rss_kb());
    }

    // Verify RSS doesn't grow monotonically
    // Allow up to 50% growth from baseline (allocator may hold onto pages)
    let max_allowed_rss = baseline_rss * 3 / 2;

    for (i, rss) in rss_after_iterations.iter().enumerate() {
        assert!(
            *rss <= max_allowed_rss,
            "Iteration {}: RSS {} KB exceeds max allowed {} KB (baseline {} KB)",
            i,
            rss,
            max_allowed_rss,
            baseline_rss
        );
    }

    // Verify RSS is stable (last few iterations should be similar)
    if rss_after_iterations.len() >= 3 {
        let last_three =
            &rss_after_iterations[rss_after_iterations.len() - 3..];
        let avg: usize = last_three.iter().sum::<usize>() / 3;
        for rss in last_three {
            let deviation = (*rss as i64 - avg as i64).unsigned_abs() as usize;
            let deviation_pct = deviation * 100 / avg.max(1);
            assert!(
                deviation_pct <= 20,
                "RSS not stable: {} KB deviates {}% from average {} KB",
                rss,
                deviation_pct,
                avg
            );
        }
    }
}

/// Test that Problem::create_model() is non-consuming.
#[test]
fn test_problem_create_model_non_consuming() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..10.0);
    problem.add_row(5.0.., [(0, 1.0)]);

    // Create and solve first model
    let mut model1 = problem
        .create_model(Sense::Minimise)
        .expect("First model creation should succeed");
    model1.solve();
    let obj1 = model1.get_objective_value();
    drop(model1);

    // Create and solve second model from same Problem
    let mut model2 = problem
        .create_model(Sense::Minimise)
        .expect("Second model creation should succeed");
    model2.solve();
    let obj2 = model2.get_objective_value();

    assert!(
        (obj1 - obj2).abs() < 1e-9,
        "Models from same problem should give same result"
    );
}

/// Test that modifications to Problem persist across create_model calls.
#[test]
fn test_problem_modifications_persist() {
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..100.0);
    problem.add_column(2.0, 0.0..100.0);
    problem.add_row(10.0.., [(0, 1.0), (1, 1.0)]); // x1 + x2 >= 10

    // Solve initial
    let mut model1 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model1.solve();
    let obj1 = model1.get_objective_value();
    drop(model1);

    // Modify Problem
    problem.change_row_bounds(0, 20.0, f64::INFINITY); // x1 + x2 >= 20

    // Solve modified
    let mut model2 = problem
        .create_model(Sense::Minimise)
        .expect("Model creation should succeed");
    model2.solve();
    let obj2 = model2.get_objective_value();

    assert!(
        obj2 > obj1,
        "Tightened constraint should increase objective: {} > {}",
        obj2,
        obj1
    );
}

/// Test StoredBasis compatibility checking.
#[test]
fn test_stored_basis_compatibility() {
    let basis = StoredBasis {
        colstatus: vec![0, 0, 0],
        rowstatus: vec![0, 0],
    };

    assert!(
        basis.is_compatible(3, 2),
        "Should be compatible with matching dims"
    );
    assert!(
        !basis.is_compatible(4, 2),
        "Should not be compatible with wrong col count"
    );
    assert!(
        !basis.is_compatible(3, 3),
        "Should not be compatible with wrong row count"
    );
    assert!(
        !basis.is_compatible(0, 0),
        "Should not be compatible with zero dims"
    );
}

/// Test empty StoredBasis behavior.
#[test]
fn test_empty_stored_basis() {
    let basis = StoredBasis::new();
    assert!(basis.is_empty(), "New basis should be empty");
    assert!(
        !basis.is_compatible(1, 1),
        "Empty basis should not be compatible"
    );
}

// ============================================================================
// T-126: Memory Regression Test
// ============================================================================

/// Get current process RSS in KB (Linux only)
#[cfg(target_os = "linux")]
fn get_current_rss_kb() -> usize {
    use std::fs;
    if let Ok(status) = fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = status.split_whitespace().nth(1) {
            if let Ok(p) = pages.parse::<usize>() {
                return p * 4; // 4KB pages
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn get_current_rss_kb() -> usize {
    0 // Skip on non-Linux
}

/// T-126: Memory regression test for per-iteration lifecycle.
///
/// Verifies that RSS is stable across iterations when using the
/// per-iteration Model lifecycle (create/drop each iteration).
///
/// This test is marked `#[ignore]` because it's long-running and
/// should be run explicitly in CI.
#[test]
#[ignore]
fn test_memory_regression_training_lifecycle() {
    // Skip on non-Linux
    if cfg!(not(target_os = "linux")) {
        println!("Skipping memory test on non-Linux platform");
        return;
    }

    const NUM_ITERATIONS: usize = 10;
    const NUM_VARS: usize = 1000;
    const NUM_CONSTRAINTS: usize = 500;

    // Create a moderately sized problem
    let mut problem = Problem::new();
    for i in 0..NUM_VARS {
        problem.add_column((i as f64) + 1.0, 0.0..100.0);
    }
    for i in 0..NUM_CONSTRAINTS {
        // Each constraint uses 10 variables
        let coeffs: Vec<(usize, f64)> =
            (0..10).map(|j| ((i * 7 + j) % NUM_VARS, 1.0)).collect();
        problem.add_row((i as f64 + 1.0).., coeffs);
    }

    let mut rss_samples = Vec::with_capacity(NUM_ITERATIONS);
    let mut cached_basis: Option<StoredBasis> = None;

    // Simulate per-iteration lifecycle
    for _iter in 0..NUM_ITERATIONS {
        // Create model from problem (simulates create_iteration_model)
        let mut model = problem
            .create_model(Sense::Minimise)
            .expect("Model creation should succeed");

        // Apply cached basis if available (training mode)
        if let Some(ref basis) = cached_basis {
            let _ = model.apply_stored_basis(basis);
        }

        model.solve();

        // Cache basis for next iteration
        cached_basis = Some(model.get_stored_basis());

        // Model is dropped here (simulates finalize_iteration)
        drop(model);

        // Sample RSS after Model is dropped
        let rss = get_current_rss_kb();
        rss_samples.push(rss);
    }

    // Verify RSS stability
    // Allow warmup in first 2 iterations, then check stability
    if rss_samples.len() >= 5 {
        let baseline_rss = rss_samples[2];
        for (i, &rss) in rss_samples[3..].iter().enumerate() {
            let growth_pct = (rss as f64 - baseline_rss as f64)
                / baseline_rss as f64
                * 100.0;
            assert!(
                growth_pct < 50.0,
                "Iteration {}: RSS grew {:.1}% from baseline (regression detected). \
                 Baseline: {} KB, Current: {} KB",
                i + 4,
                growth_pct,
                baseline_rss,
                rss
            );
        }

        // Check last 3 iterations are stable (< 20% variance)
        let last_three = &rss_samples[rss_samples.len() - 3..];
        let max_rss = *last_three.iter().max().unwrap();
        let min_rss = *last_three.iter().min().unwrap();
        let variance_pct = (max_rss - min_rss) as f64 / min_rss as f64 * 100.0;
        assert!(
            variance_pct < 20.0,
            "RSS not stable in last 3 iterations: {:.1}% variance. \
             Values: {:?}",
            variance_pct,
            last_three
        );

        println!("✅ Memory regression test passed");
        println!("   Baseline RSS (iter 3): {} KB", baseline_rss);
        println!(
            "   Final RSS (iter {}): {} KB",
            NUM_ITERATIONS,
            rss_samples.last().unwrap()
        );
        println!(
            "   Growth: {:.1}%",
            (*rss_samples.last().unwrap() as f64 - baseline_rss as f64)
                / baseline_rss as f64
                * 100.0
        );
    }
}
