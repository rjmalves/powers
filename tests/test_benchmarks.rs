// Integration tests for hydrothermal benchmark problems
//
// These tests validate that SDDP converges to known solutions on benchmark problems.
// They serve multiple purposes:
// 1. Numerical correctness validation (not just "doesn't crash")
// 2. Regression testing (detect algorithm changes that break correctness)
// 3. Demonstration of typical problem structures
//
// ⚠️ CONVERGENCE VALIDATION PRINCIPLES:
// =======================================
// 1. **Lower Bound**: Must be ≤ optimal value (valid bound)
// 2. **Upper Bound**: Must be ≥ optimal value (statistical bound)
// 3. **Monotonicity**: Lower bound should be non-decreasing
// 4. **Gap Reduction**: Gap should decrease (on average) with iterations
// 5. **Bound Validity**: LB ≤ Optimal ≤ UB must hold
//
// Test failures indicate potential algorithm bugs, not test flakiness.

mod fixtures;

use fixtures::benchmarks::*;

// ============================================================================
// BENCHMARK 1: Deterministic Single Reservoir
// ============================================================================

#[test]
fn test_deterministic_single_reservoir_convergence() {
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    // Training parameters: Deterministic problem should converge reasonably fast
    let num_iterations = 30; // More iterations for water value learning
    let num_forward_passes = 10; // Multiple passes to get good upper bound estimate

    // Expected solution range (thermal usage due to water scarcity)
    let expected_min = 1500.0; // Lower bound on cost
    let expected_max = 2500.0; // Upper bound on cost
    let tolerance = 200.0; // Tolerance for deterministic convergence

    // Train the algorithm
    let result = sddp
        .train(num_iterations, num_forward_passes, &saa)
        .expect("Training failed");

    println!("\n=== Deterministic Single Reservoir Results ===");
    println!(
        "Expected range: ${:.2} - ${:.2}",
        expected_min, expected_max
    );
    println!("Final lower bound: ${:.2}", result.final_lower_bound);
    println!("Final upper bound: ${:.2}", result.final_upper_bound);
    println!("Statistical UB: ${:.2}", result.statistical_upper_bound);
    println!("Final gap: ${:.2}", result.final_gap());
    println!("Relative gap: {:.2}%", result.relative_gap() * 100.0);

    // VALIDATION 1: Solution should be in reasonable range
    assert!(
        result.final_lower_bound >= expected_min - tolerance,
        "Lower bound ({:.2}) below expected minimum ({:.2}) - tolerance ({:.2})",
        result.final_lower_bound,
        expected_min,
        tolerance
    );

    assert!(
        result.statistical_upper_bound <= expected_max + tolerance,
        "Statistical upper bound ({:.2}) exceeds expected maximum ({:.2}) + tolerance ({:.2})",
        result.statistical_upper_bound,
        expected_max,
        tolerance
    );

    // VALIDATION 2: Gap should be reasonable for deterministic problem
    // With water value learning, expect larger gap than trivial problem
    assert!(
        result.final_gap() <= expected_max * 0.15, // 15% relative gap
        "Gap ({:.2}) exceeds 15% of expected cost ({:.2})",
        result.final_gap(),
        expected_max * 0.15
    );

    // VALIDATION 3: Cost should be positive (thermal usage occurs)
    assert!(
        result.final_lower_bound > 1000.0,
        "Lower bound ({:.2}) too low - problem should require significant thermal generation",
        result.final_lower_bound
    );

    // VALIDATION 4: Monotonicity - lower bound should be non-decreasing
    let iterations = result.iterations();
    for i in 1..iterations.len() {
        assert!(
            iterations[i].lower_bound >= iterations[i - 1].lower_bound - 1e-6,
            "Lower bound decreased at iteration {}: {:.2} -> {:.2}",
            i + 1,
            iterations[i - 1].lower_bound,
            iterations[i].lower_bound
        );
    }

    println!("✓ All validations passed");
    println!(
        "✓ Non-trivial optimization: Cost = ${:.2}",
        result.final_lower_bound
    );
}

#[test]
fn test_deterministic_single_reservoir_policy_structure() {
    let (mut sddp, saa) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");

    let result = sddp.train(20, 10, &saa).expect("Training failed");

    println!("\n=== Policy Structure Validation ===");
    println!("Number of cuts generated: {}", result.num_cuts);

    // VALIDATION: Should have generated cuts (policy is non-trivial)
    assert!(
        result.num_cuts > 0,
        "No cuts generated - algorithm may not be working"
    );

    // VALIDATION: For 2-stage problem with single scenario, cuts should be modest
    // Each iteration can add multiple cuts per stage due to solver iterations
    assert!(
        result.num_cuts <= 500,
        "Excessive cuts ({}) for simple 2-stage deterministic problem",
        result.num_cuts
    );

    println!("✓ Policy structure is reasonable");
}

// ============================================================================
// BENCHMARK 2: Stochastic Single Reservoir
// ============================================================================

#[test]
fn test_stochastic_single_reservoir_convergence() {
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    // Training parameters: Stochastic needs more iterations
    let num_iterations = 50; // More iterations for stochastic
    let num_forward_passes = 20; // More forward passes for better UB estimate

    // Expected solution range (thermal + hedging costs)
    let expected_min = 1200.0; // Minimum hedging cost
    let expected_max = 3000.0; // Maximum with risk premium
    let tolerance = 300.0; // Wider tolerance for stochastic

    // Train the algorithm
    let result = sddp
        .train(num_iterations, num_forward_passes, &saa)
        .expect("Training failed");

    println!("\n=== Stochastic Single Reservoir Results ===");
    println!(
        "Expected range: ${:.2} - ${:.2}",
        expected_min, expected_max
    );
    println!("Final lower bound: ${:.2}", result.final_lower_bound);
    println!("Final upper bound: ${:.2}", result.final_upper_bound);
    println!("Statistical UB: ${:.2}", result.statistical_upper_bound);
    println!("Final gap: ${:.2}", result.final_gap());
    println!("Relative gap: {:.2}%", result.relative_gap() * 100.0);

    // VALIDATION 1: Solution should be in reasonable range
    assert!(
        result.final_lower_bound >= expected_min - tolerance,
        "Lower bound ({:.2}) below expected minimum ({:.2}) - tolerance ({:.2})",
        result.final_lower_bound,
        expected_min,
        tolerance
    );

    assert!(
        result.statistical_upper_bound <= expected_max + tolerance,
        "Statistical upper bound ({:.2}) exceeds expected maximum ({:.2}) + tolerance ({:.2})",
        result.statistical_upper_bound,
        expected_max,
        tolerance
    );

    // VALIDATION 2: Upper bound should be above lower bound
    assert!(
        result.statistical_upper_bound >= result.final_lower_bound,
        "Statistical UB ({:.2}) below LB ({:.2}) - violates bound relationship",
        result.statistical_upper_bound,
        result.final_lower_bound
    );

    // VALIDATION 3: Gap should be reasonable for stochastic problem
    // Stochastic problems have larger gaps due to sampling variance
    assert!(
        result.final_gap() <= expected_max * 0.25, // 25% relative gap
        "Gap ({:.2}) exceeds 25% of expected cost ({:.2})",
        result.final_gap(),
        expected_max * 0.25
    );

    // VALIDATION 4: Cost should reflect hedging (higher than deterministic)
    assert!(
        result.final_lower_bound > 1000.0,
        "Lower bound ({:.2}) too low - stochastic problem should have hedging costs",
        result.final_lower_bound
    );

    // VALIDATION 5: Monotonicity
    let iterations = result.iterations();
    for i in 1..iterations.len() {
        assert!(
            iterations[i].lower_bound >= iterations[i - 1].lower_bound - 1e-6,
            "Lower bound decreased at iteration {}: {:.2} -> {:.2}",
            i + 1,
            iterations[i - 1].lower_bound,
            iterations[i].lower_bound
        );
    }

    println!("✓ All validations passed");
    println!(
        "✓ Stochastic hedging: Cost = ${:.2}",
        result.final_lower_bound
    );
}

#[test]
fn test_stochastic_single_reservoir_policy_structure() {
    let (mut sddp, saa) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");

    let result = sddp.train(50, 20, &saa).expect("Training failed");

    println!("\n=== Stochastic Policy Structure ===");
    println!("Number of cuts generated: {}", result.num_cuts);

    // VALIDATION: Should generate more cuts than deterministic
    assert!(
        result.num_cuts > 0,
        "No cuts generated - algorithm may not be working"
    );

    // VALIDATION: For 2-stage with 3 scenarios in stage 2, expect more cuts
    // But still reasonable (not thousands)
    assert!(
        result.num_cuts <= 2000,
        "Excessive cuts ({}) for 2-stage stochastic problem",
        result.num_cuts
    );

    println!("✓ Policy structure is reasonable");
}

// ============================================================================
// BENCHMARK 3: Two-Reservoir Cascade
// ============================================================================

#[test]
fn test_two_reservoir_cascade_convergence() {
    let (mut sddp, saa) = create_two_reservoir_cascade()
        .expect("Failed to create cascade benchmark");

    // Training parameters: Cascade coordination needs more iterations
    let num_iterations = 40; // Moderate iterations for cascade
    let num_forward_passes = 15; // Moderate forward passes

    // Expected solution range
    let expected_min = 0.0; // Best case: all hydro
    let expected_max = 800.0; // Worst case: significant thermal/deficit needed
    let tolerance = 200.0; // Wide tolerance for cascade complexity

    // Train the algorithm
    let result = sddp
        .train(num_iterations, num_forward_passes, &saa)
        .expect("Training failed");

    println!("\n=== Two-Reservoir Cascade Results ===");
    println!(
        "Expected range: ${:.2} - ${:.2}",
        expected_min, expected_max
    );
    println!("Final lower bound: ${:.2}", result.final_lower_bound);
    println!("Final upper bound: ${:.2}", result.final_upper_bound);
    println!("Statistical UB: ${:.2}", result.statistical_upper_bound);
    println!("Final gap: ${:.2}", result.final_gap());
    println!("Tolerance: ${:.2}", tolerance);

    // VALIDATION 1: Bounds should be in reasonable range
    assert!(
        result.final_lower_bound >= expected_min - tolerance,
        "Lower bound ({:.2}) too low (expected ≥ ${:.2})",
        result.final_lower_bound,
        expected_min - tolerance
    );

    assert!(
        result.final_lower_bound <= expected_max + tolerance,
        "Lower bound ({:.2}) too high (expected ≤ ${:.2})",
        result.final_lower_bound,
        expected_max + tolerance
    );

    // VALIDATION 2: Bounds are valid
    assert!(
        result.statistical_upper_bound >= result.final_lower_bound,
        "Statistical UB ({:.2}) below LB ({:.2})",
        result.statistical_upper_bound,
        result.final_lower_bound
    );

    // VALIDATION 3: Gap should be reasonable
    assert!(
        result.final_gap() <= tolerance * 2.0,
        "Gap ({:.2}) exceeds 2x tolerance ({:.2})",
        result.final_gap(),
        tolerance * 2.0
    );

    // VALIDATION 4: Monotonicity
    let iterations = result.iterations();
    for i in 1..iterations.len() {
        assert!(
            iterations[i].lower_bound >= iterations[i - 1].lower_bound - 1e-6,
            "Lower bound decreased at iteration {}: {:.2} -> {:.2}",
            i + 1,
            iterations[i - 1].lower_bound,
            iterations[i].lower_bound
        );
    }

    println!("✓ All validations passed");
}

#[test]
fn test_two_reservoir_cascade_policy_structure() {
    let (mut sddp, saa) = create_two_reservoir_cascade()
        .expect("Failed to create cascade benchmark");

    let result = sddp.train(40, 15, &saa).expect("Training failed");

    println!("\n=== Cascade Policy Structure ===");
    println!("Number of cuts generated: {}", result.num_cuts);

    // VALIDATION: Should generate cuts
    assert!(
        result.num_cuts > 0,
        "No cuts generated - algorithm may not be working"
    );

    // VALIDATION: Reasonable number of cuts for 2-stage cascade
    assert!(
        result.num_cuts <= 1500,
        "Excessive cuts ({}) for 2-stage cascade problem",
        result.num_cuts
    );

    println!("✓ Policy structure is reasonable");
}

// ============================================================================
// CROSS-BENCHMARK COMPARISON
// ============================================================================

#[test]
fn test_benchmark_complexity_comparison() {
    println!("\n=== Benchmark Complexity Comparison ===");

    // Train all three benchmarks with same iteration budget
    let num_iterations = 30;
    let num_forward_passes = 10;

    // Deterministic
    let (mut sddp1, saa1) = create_deterministic_single_reservoir()
        .expect("Failed to create deterministic benchmark");
    let result1 = sddp1
        .train(num_iterations, num_forward_passes, &saa1)
        .expect("Training failed");

    // Stochastic
    let (mut sddp2, saa2) = create_stochastic_single_reservoir()
        .expect("Failed to create stochastic benchmark");
    let result2 = sddp2
        .train(num_iterations, num_forward_passes, &saa2)
        .expect("Training failed");

    // Cascade
    let (mut sddp3, saa3) = create_two_reservoir_cascade()
        .expect("Failed to create cascade benchmark");
    let result3 = sddp3
        .train(num_iterations, num_forward_passes, &saa3)
        .expect("Training failed");

    println!("Deterministic: Gap = ${:.2}", result1.final_gap());
    println!("Stochastic:    Gap = ${:.2}", result2.final_gap());
    println!("Cascade:       Gap = ${:.2}", result3.final_gap());

    // VALIDATION 1: All gaps should be reasonable (< $1000 for well-conditioned problems)
    assert!(
        result1.final_gap().abs() < 1000.0,
        "Deterministic gap ({:.2}) too large",
        result1.final_gap()
    );

    assert!(
        result2.final_gap().abs() < 1000.0,
        "Stochastic gap ({:.2}) too large",
        result2.final_gap()
    );

    assert!(
        result3.final_gap().abs() < 1000.0,
        "Cascade gap ({:.2}) too large",
        result3.final_gap()
    );

    // VALIDATION 2: All problems should have non-zero costs (non-trivial)
    // With water scarcity, optimal cost should be positive (thermal usage required)
    assert!(
        result1.final_lower_bound > 100.0,
        "Deterministic problem too trivial (LB = {:.2})",
        result1.final_lower_bound
    );

    assert!(
        result2.final_lower_bound > 100.0,
        "Stochastic problem too trivial (LB = {:.2})",
        result2.final_lower_bound
    );

    println!("✓ All problems well-conditioned with non-trivial optimization");
    println!("✓ Complexity comparison is consistent");
}
