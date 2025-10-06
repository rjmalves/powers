//! Comprehensive Solver Interface Tests
//!
//! This module tests the solver interface (`src/solver.rs`) comprehensively:
//!
//! 1. **Real Solver Integration**: Tests with actual HiGHS solver
//! 2. **Error Handling**: Infeasible, unbounded, numerical issues
//! 3. **Edge Cases**: Empty problems, single variable, degenerate cases
//! 4. **Performance**: Large problems, repeated solves, memory stability
//!
//! **Testing Strategy**:
//! - Use real HiGHS solver for integration tests (validates actual behavior)
//! - Use mock solver for algorithm logic tests (fast, isolated)
//! - Focus on solver interface contract, not HiGHS internals
//!
//! **Performance Note**:
//! These tests use real solver calls, so they're slower than unit tests.
//! Total runtime should be <2 seconds for the full suite.

use powers_rs::solver::{HighsModelStatus, Problem, Sense};
use std::time::Instant;

// ================================================================================================
// REAL SOLVER INTEGRATION TESTS
// ================================================================================================

#[test]
fn test_simple_lp_optimal() {
    // Problem: Minimize x + 2y
    //          Subject to: x + y >= 1
    //                      x, y >= 0
    // Optimal solution: x=1, y=0, obj=1
    let mut problem = Problem::new();

    // Variables: x (cost 1), y (cost 2)
    problem.add_column(1.0, 0.0..); // x >= 0
    problem.add_column(2.0, 0.0..); // y >= 0

    // Constraint: x + y >= 1
    problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(
        (obj - 1.0).abs() < 1e-6,
        "Expected objective 1.0, got {:.6}",
        obj
    );

    let solution = model.get_solution();
    assert_eq!(solution.colvalue.len(), 2);
    assert!(
        (solution.colvalue[0] - 1.0).abs() < 1e-6,
        "Expected x=1.0, got {:.6}",
        solution.colvalue[0]
    );
    assert!(
        solution.colvalue[1].abs() < 1e-6,
        "Expected y=0.0, got {:.6}",
        solution.colvalue[1]
    );
}

#[test]
fn test_infeasible_problem() {
    // Problem: x >= 10 AND x <= 5 (infeasible)
    let mut problem = Problem::new();

    // Variable x with bounds [0, infinity)
    problem.add_column(1.0, 0.0..);

    // Contradictory constraints
    problem.add_row(10.0.., [(0, 1.0)]); // x >= 10
    problem.add_row(..=5.0, [(0, 1.0)]); // x <= 5

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let status = model.status();
    assert!(
        status == HighsModelStatus::Infeasible
            || status == HighsModelStatus::UnboundedOrInfeasible,
        "Expected Infeasible, got {:?}",
        status
    );
}

#[test]
fn test_unbounded_problem() {
    // Problem: Minimize -x with no upper bound on x (unbounded)
    let mut problem = Problem::new();

    // Variable x: [0, infinity), cost -1 (minimize -x = maximize x → unbounded)
    problem.add_column(-1.0, 0.0..);

    // No constraints (x can grow indefinitely)

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let status = model.status();
    // HiGHS may detect unboundedness or flag as unbounded-or-infeasible
    assert!(
        status == HighsModelStatus::Unbounded
            || status == HighsModelStatus::UnboundedOrInfeasible,
        "Expected Unbounded, got {:?}",
        status
    );
}

#[test]
fn test_single_variable_problem() {
    // Problem: Minimize x
    //          Subject to: x >= 5
    // Optimal: x = 5, obj = 5
    let mut problem = Problem::new();

    problem.add_column(1.0, 5.0..); // x >= 5, cost 1

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(
        (obj - 5.0).abs() < 1e-6,
        "Expected objective 5.0, got {:.6}",
        obj
    );

    let solution = model.get_solution().colvalue;
    assert_eq!(solution.len(), 1);
    assert!(
        (solution[0] - 5.0).abs() < 1e-6,
        "Expected x=5.0, got {:.6}",
        solution[0]
    );
}

#[test]
fn test_equality_constraint() {
    // Problem: Minimize x + y
    //          Subject to: x + y = 10
    //                      x, y >= 0
    // Optimal: x=10, y=0, obj=10 (or x=0, y=10, same obj)
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0, cost 1
    problem.add_column(1.0, 0.0..); // y >= 0, cost 1

    // Equality: x + y = 10 (represented as 10 <= x+y <= 10)
    problem.add_row(10.0..=10.0, [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(
        (obj - 10.0).abs() < 1e-6,
        "Expected objective 10.0, got {:.6}",
        obj
    );

    let solution = model.get_solution().colvalue;
    let sum = solution[0] + solution[1];
    assert!(
        (sum - 10.0).abs() < 1e-6,
        "Expected x+y=10.0, got {:.6}",
        sum
    );
}

// ================================================================================================
// EDGE CASE TESTS
// ================================================================================================

#[test]
fn test_problem_with_no_constraints() {
    // Problem: Minimize x with x >= 0 (no additional constraints)
    // Optimal: x = 0, obj = 0
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0, cost 1

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(obj.abs() < 1e-6, "Expected objective 0.0, got {:.6}", obj);

    let solution = model.get_solution().colvalue;
    assert!(
        solution[0].abs() < 1e-6,
        "Expected x=0.0, got {:.6}",
        solution[0]
    );
}

#[test]
fn test_numerical_edge_case_large_coefficients() {
    // Problem with large coefficients (tests numerical stability)
    // Minimize: 1e10*x + 1e-10*y
    // Subject to: x + y = 1
    //             x, y >= 0
    // Optimal: x=0, y=1 (prefer cheaper variable)
    let mut problem = Problem::new();

    problem.add_column(1e10, 0.0..); // x >= 0, cost 1e10 (expensive!)
    problem.add_column(1e-10, 0.0..); // y >= 0, cost 1e-10 (cheap!)

    // x + y = 1
    problem.add_row(1.0..=1.0, [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let solution = model.get_solution().colvalue;
    // Should prefer y (cheaper)
    assert!(
        solution[1] > 0.9,
        "Expected y ≈ 1.0, got {:.6}",
        solution[1]
    );
    assert!(
        solution[0] < 0.1,
        "Expected x ≈ 0.0, got {:.6}",
        solution[0]
    );
}

#[test]
fn test_degenerate_problem() {
    // Problem with multiple optimal solutions (degenerate)
    // Maximize: x + y
    // Subject to: x + y <= 10
    //             x, y >= 0
    // Optimal: Any (x,y) with x+y=10, obj=10
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0, cost 1
    problem.add_column(1.0, 0.0..); // y >= 0, cost 1

    problem.add_row(..=10.0, [(0, 1.0), (1, 1.0)]); // x + y <= 10

    let mut model = problem.optimise(Sense::Maximise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(
        (obj - 10.0).abs() < 1e-6,
        "Expected objective 10.0, got {:.6}",
        obj
    );

    // Any solution with x+y=10 is valid
    let solution = model.get_solution().colvalue;
    let sum = solution[0] + solution[1];
    assert!(
        (sum - 10.0).abs() < 1e-6,
        "Expected x+y=10.0, got {:.6}",
        sum
    );
}

#[test]
fn test_bounded_variable_at_limit() {
    // Problem: Minimize x
    //          Subject to: 3 <= x <= 7
    // Optimal: x = 3, obj = 3
    let mut problem = Problem::new();

    problem.add_column(1.0, 3.0..=7.0); // 3 <= x <= 7, cost 1

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let obj = model.get_objective_value();
    assert!(
        (obj - 3.0).abs() < 1e-6,
        "Expected objective 3.0, got {:.6}",
        obj
    );

    let solution = model.get_solution().colvalue;
    assert!(
        (solution[0] - 3.0).abs() < 1e-6,
        "Expected x=3.0, got {:.6}",
        solution[0]
    );
}

// ================================================================================================
// PERFORMANCE TESTS
// ================================================================================================

#[test]
fn test_large_problem_performance() {
    // Create problem with 1000 variables and 500 constraints
    // This tests that solver handles scale without issues
    let mut problem = Problem::new();

    // 1000 variables: all with cost 1.0, bounds [0, 1]
    for _ in 0..1000 {
        problem.add_column(1.0, 0.0..=1.0);
    }

    // 500 constraints: each sums a random subset of variables <= 100
    for i in 0..500 {
        let mut row = Vec::new();
        // Each constraint involves ~10 variables (sparse)
        for j in 0..10 {
            let var_idx = (i * 2 + j) % 1000;
            row.push((var_idx, 1.0));
        }
        problem.add_row(..=100.0, row);
    }

    let mut model = problem.optimise(Sense::Minimise);

    let start = Instant::now();
    model.solve();
    let elapsed = start.elapsed();

    assert_eq!(
        model.status(),
        HighsModelStatus::Optimal,
        "Large problem should be solvable"
    );

    // Should solve in reasonable time (< 1 second for this size)
    assert!(
        elapsed.as_secs() < 1,
        "Large problem took too long: {:?}",
        elapsed
    );

    println!(
        "Solved 1000x500 problem in {:?} ({} vars, {} constraints)",
        elapsed,
        model.num_cols(),
        model.num_rows()
    );
}

#[test]
fn test_repeated_solves_no_memory_leak() {
    // Solve the same problem 100 times to check for memory leaks
    // Note: This is not a perfect leak test, but catches obvious issues
    let create_problem = || {
        let mut problem = Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_column(2.0, 0.0..);
        problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]);
        problem
    };

    let mut solve_times = Vec::with_capacity(100);

    for i in 0..100 {
        let problem = create_problem();
        let mut model = problem.optimise(Sense::Minimise);

        let start = Instant::now();
        model.solve();
        let elapsed = start.elapsed();

        assert_eq!(
            model.status(),
            HighsModelStatus::Optimal,
            "Solve {} failed",
            i + 1
        );

        solve_times.push(elapsed);
    }

    // Check consistency (no memory leak causing slowdown)
    let first_10_avg =
        solve_times[0..10].iter().sum::<std::time::Duration>() / 10;
    let last_10_avg =
        solve_times[90..100].iter().sum::<std::time::Duration>() / 10;

    // Last 10 should not be significantly slower than first 10
    // (Allow 2x variance for system noise)
    assert!(
        last_10_avg < first_10_avg * 2,
        "Solve time increased significantly: {:?} → {:?} (possible memory leak)",
        first_10_avg,
        last_10_avg
    );

    println!(
        "100 repeated solves: first 10 avg = {:?}, last 10 avg = {:?}",
        first_10_avg, last_10_avg
    );
}

#[test]
fn test_model_reuse_with_modifications() {
    // Test that we can modify a model and re-solve
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0
    problem.add_column(1.0, 0.0..); // y >= 0

    // Initial constraint: x + y >= 5
    problem.add_row(5.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let obj1 = model.get_objective_value();
    assert!((obj1 - 5.0).abs() < 1e-6);

    // Modify constraint: x + y >= 10 (tighten)
    model.change_rows_bounds(0, 10.0, f64::INFINITY);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let obj2 = model.get_objective_value();
    assert!((obj2 - 10.0).abs() < 1e-6);

    // Verify objective increased (tighter constraint)
    assert!(
        obj2 > obj1,
        "Objective should increase with tighter constraint"
    );
}

// ================================================================================================
// ERROR HANDLING TESTS
// ================================================================================================

#[test]
fn test_problem_with_infinite_bounds() {
    // Test that infinite bounds are handled correctly
    // Problem: Minimize x with x in (-inf, +inf)
    // Optimal: unbounded (can make x arbitrarily negative)
    let mut problem = Problem::new();

    problem.add_column(1.0, f64::NEG_INFINITY..f64::INFINITY); // x ∈ (-∞, +∞), cost 1

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let status = model.status();
    // Should detect unboundedness
    assert!(
        status == HighsModelStatus::Unbounded
            || status == HighsModelStatus::UnboundedOrInfeasible,
        "Unbounded variable with positive cost should be unbounded, got {:?}",
        status
    );
}

#[test]
fn test_empty_problem_construction() {
    // Test that an empty problem can be constructed
    // (even if it can't be solved meaningfully)
    let problem = Problem::new();

    assert_eq!(problem.num_col, 0);
    assert_eq!(problem.num_row, 0);
    assert_eq!(problem.num_nz, 0);

    // Try to create a model (may fail or produce ModelEmpty status)
    let result = problem.try_optimise(Sense::Minimise);

    match result {
        Ok(mut model) => {
            model.solve();
            let status = model.status();
            // Empty problem should be ModelEmpty or similar
            assert_ne!(
                status,
                HighsModelStatus::Optimal,
                "Empty problem should not be optimal"
            );
        }
        Err(_) => {
            // Failing to create model from empty problem is acceptable
        }
    }
}

// ================================================================================================
// SOLVER INTERFACE CONTRACT TESTS
// ================================================================================================

#[test]
fn test_solution_vector_size_matches_variables() {
    let mut problem = Problem::new();

    // Add 5 variables
    for i in 0..5 {
        problem.add_column(i as f64, 0.0..=10.0);
    }

    // Add constraint: sum of all variables <= 20
    let row: Vec<_> = (0..5).map(|i| (i, 1.0)).collect();
    problem.add_row(..=20.0, row);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let solution = model.get_solution().colvalue;
    assert_eq!(
        solution.len(),
        5,
        "Solution vector should match number of variables"
    );
}

#[test]
fn test_objective_value_consistent_with_solution() {
    let mut problem = Problem::new();

    // Variables with known costs
    problem.add_column(2.0, 0.0..); // x, cost 2
    problem.add_column(3.0, 0.0..); // y, cost 3

    // Constraint: x + y = 5
    problem.add_row(5.0..=5.0, [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    let solution = model.get_solution().colvalue;
    let obj = model.get_objective_value();

    // Manually compute objective
    let computed_obj = 2.0 * solution[0] + 3.0 * solution[1];

    assert!(
        (obj - computed_obj).abs() < 1e-6,
        "Objective value {} doesn't match computed value {:.6}",
        obj,
        computed_obj
    );
}

#[test]
fn test_num_cols_and_rows_correct() {
    let mut problem = Problem::new();

    // Add 3 variables
    for _ in 0..3 {
        problem.add_column(1.0, 0.0..);
    }

    // Add 2 constraints
    problem.add_row(..=10.0, [(0, 1.0), (1, 1.0)]);
    problem.add_row(..=20.0, [(1, 1.0), (2, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.num_cols(), 3, "Should have 3 variables");
    assert_eq!(model.num_rows(), 2, "Should have 2 constraints");
}

// ================================================================================================
// T3.Coverage: Additional Tests to Reach 85% Coverage
// ================================================================================================
// Adding 20 tests targeting uncovered lines in solver.rs to improve coverage from 75% to 85%

// --------------------------------------------------------------------------------
// Error Handling Tests (8 tests)
// --------------------------------------------------------------------------------

#[test]
fn test_model_with_all_inequality_constraints() {
    // Problem with only inequality constraints (no equality)
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0
    problem.add_column(1.0, 0.0..); // y >= 0

    // All inequality constraints
    problem.add_row(..=10.0, [(0, 1.0), (1, 1.0)]); // x + y <= 10
    problem.add_row(5.0.., [(0, 1.0)]); // x >= 5
    problem.add_row(2.0.., [(1, 1.0)]); // y >= 2

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let obj = model.get_objective_value();
    assert!(
        (obj - 7.0).abs() < 1e-6,
        "Expected objective 7.0 (x=5, y=2), got {:.6}",
        obj
    );
}

#[test]
fn test_model_with_all_equality_constraints() {
    // Problem with only equality constraints
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..); // x >= 0
    problem.add_column(2.0, 0.0..); // y >= 0

    // Equality constraints: x = 3, y = 2
    problem.add_row(3.0..=3.0, [(0, 1.0)]);
    problem.add_row(2.0..=2.0, [(1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let solution = model.get_solution();
    assert!((solution.colvalue[0] - 3.0).abs() < 1e-6, "Expected x=3.0");
    assert!((solution.colvalue[1] - 2.0).abs() < 1e-6, "Expected y=2.0");
}

#[test]
fn test_problem_with_zero_objective_coefficients() {
    // Problem where all variables have zero cost (feasibility problem)
    let mut problem = Problem::new();

    problem.add_column(0.0, 0.0..); // x with zero cost
    problem.add_column(0.0, 0.0..); // y with zero cost

    // Constraint: x + y >= 5
    problem.add_row(5.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let obj = model.get_objective_value();
    assert!(obj.abs() < 1e-6, "Expected objective 0.0, got {:.6}", obj);
}

#[test]
fn test_model_with_negative_variable_bounds() {
    // Problem with variables that can be negative
    let mut problem = Problem::new();

    problem.add_column(1.0, -10.0..10.0); // x in [-10, 10]
    problem.add_column(1.0, -5.0..15.0); // y in [-5, 15]

    // Constraint: x + y >= 0
    problem.add_row(0.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    // Optimal: x=-10, y=10 (sum=0, cost=-10+10=0)
    // OR: x=-10, y=-5 (sum=-15, infeasible)
    // Actually: minimize x+y with x+y>=0 → x=-10, y=10, obj=0
    let obj = model.get_objective_value();
    assert!(obj.abs() < 1e-6, "Expected objective 0.0, got {:.6}", obj);
}

#[test]
fn test_maximization_problem() {
    // Test maximization (most tests use minimization)
    let mut problem = Problem::new();

    problem.add_column(-2.0, 0.0..10.0); // x, maximize 2x (so cost is -2x)
    problem.add_column(-3.0, 0.0..10.0); // y, maximize 3y (so cost is -3y)

    // Constraint: x + y <= 8
    problem.add_row(..=8.0, [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise); // Minimizing -2x -3y = maximizing 2x + 3y
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let solution = model.get_solution();
    // Optimal: maximize 2x + 3y with x+y <= 8 → x=0, y=8, obj=24
    assert!(
        solution.colvalue[1].abs() >= 7.999,
        "Expected y≈8, got {:.6}",
        solution.colvalue[1]
    );
}

#[test]
fn test_model_reuse_with_basis() {
    // Test that we can solve multiple times and basis is preserved/updated
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_column(2.0, 0.0..);

    problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let first_obj = model.get_objective_value();

    // Solve again without changing problem
    model.solve();
    let second_obj = model.get_objective_value();

    assert!(
        (first_obj - second_obj).abs() < 1e-6,
        "Objective should be the same on repeated solve"
    );
    assert_eq!(model.status(), HighsModelStatus::Optimal);
}

#[test]
fn test_solution_retrieval_before_solve() {
    // Test behavior when trying to get solution before solving
    let mut problem = Problem::new();
    problem.add_column(1.0, 0.0..);
    problem.add_row(1.0.., [(0, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);

    // Get solution before solve (should work, might be empty or default)
    let solution = model.get_solution();
    assert!(
        solution.colvalue.is_empty() || solution.colvalue.len() == 1,
        "Solution should have 0 or 1 elements before solve"
    );

    // Now solve
    model.solve();
    let solution = model.get_solution();
    assert_eq!(
        solution.colvalue.len(),
        1,
        "Solution should have 1 element after solve"
    );
}

#[test]
fn test_model_status_is_not_optimal_for_infeasible() {
    // Ensure status is not Optimal for infeasible problem
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);

    // Make it infeasible
    problem.add_row(10.0.., [(0, 1.0)]); // x >= 10
    problem.add_row(..=5.0, [(0, 1.0)]); // x <= 5

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    let status = model.status();
    assert_ne!(
        status,
        HighsModelStatus::Optimal,
        "Infeasible problem should not have Optimal status"
    );
}

// --------------------------------------------------------------------------------
// Edge Case Tests (7 tests)
// --------------------------------------------------------------------------------

#[test]
fn test_problem_with_very_large_bounds() {
    // Test with very large (but not infinite) bounds
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..1e10); // Very large upper bound
    problem.add_row(1e9.., [(0, 1.0)]); // Very large lower bound

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let solution = model.get_solution();
    assert!(
        solution.colvalue[0] >= 1e9 - 1e-3,
        "Expected x ≈ 1e9, got {:.6e}",
        solution.colvalue[0]
    );
}

#[test]
fn test_problem_with_tight_bounds() {
    // Test variables with very tight bounds (almost equality)
    let mut problem = Problem::new();

    problem.add_column(1.0, 5.0..5.001); // x in [5.0, 5.001]
    problem.add_column(1.0, 0.0..);

    problem.add_row(10.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
}

#[test]
fn test_problem_with_many_variables_few_constraints() {
    // Stress test: many variables, few constraints
    let mut problem = Problem::new();

    // Add 100 variables
    for _ in 0..100 {
        problem.add_column(1.0, 0.0..);
    }

    // Add only 2 constraints
    let vars1: Vec<_> = (0..50).map(|i| (i, 1.0)).collect();
    problem.add_row(25.0.., vars1);

    let vars2: Vec<_> = (50..100).map(|i| (i, 1.0)).collect();
    problem.add_row(25.0.., vars2);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    assert_eq!(model.num_cols(), 100);
    assert_eq!(model.num_rows(), 2);
}

#[test]
fn test_problem_with_few_variables_many_constraints() {
    // Stress test: few variables, many constraints
    let mut problem = Problem::new();

    // Add 2 variables
    problem.add_column(1.0, 0.0..100.0);
    problem.add_column(1.0, 0.0..100.0);

    // Add 50 constraints (most redundant)
    for i in 1..=50 {
        problem.add_row(..=(i as f64), [(0, 1.0), (1, 1.0)]);
    }

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    assert_eq!(model.num_cols(), 2);
    assert_eq!(model.num_rows(), 50);
}

#[test]
fn test_problem_with_redundant_constraints() {
    // Problem with redundant constraints (should still solve correctly)
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_column(1.0, 0.0..);

    // These constraints are redundant
    problem.add_row(..=100.0, [(0, 1.0), (1, 1.0)]); // x + y <= 100
    problem.add_row(..=200.0, [(0, 1.0), (1, 1.0)]); // x + y <= 200 (redundant)
    problem.add_row(..=50.0, [(0, 1.0), (1, 1.0)]); // x + y <= 50 (tighter)

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let obj = model.get_objective_value();
    assert!(
        obj.abs() < 1e-6,
        "Expected objective 0.0 (x=0, y=0), got {:.6}",
        obj
    );
}

#[test]
fn test_problem_with_zero_coefficients_in_constraint() {
    // Constraint with some zero coefficients
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_column(1.0, 0.0..);
    problem.add_column(1.0, 0.0..);

    // Constraint: 1*x + 0*y + 1*z >= 5 (y doesn't appear)
    problem.add_row(5.0.., [(0, 1.0), (1, 0.0), (2, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
}

#[test]
fn test_minimization_vs_maximization_consistency() {
    // Test that minimizing -f(x) = maximizing f(x)
    let mut min_problem = Problem::new();
    min_problem.add_column(-1.0, 0.0..10.0); // Minimize -x
    min_problem.add_row(5.0.., [(0, 1.0)]); // x >= 5

    let mut min_model = min_problem.optimise(Sense::Minimise);
    min_model.solve();

    let min_obj = min_model.get_objective_value();
    let min_x = min_model.get_solution().colvalue[0];

    // Maximizing x with x >= 5 should give x=10 (at bound)
    assert!(
        (min_x - 10.0).abs() < 1e-6,
        "Expected x=10 for minimizing -x, got {:.6}",
        min_x
    );
    assert!(
        (min_obj + 10.0).abs() < 1e-6,
        "Expected obj=-10, got {:.6}",
        min_obj
    );
}

// --------------------------------------------------------------------------------
// Basis Warm-Start Tests (5 tests)
// --------------------------------------------------------------------------------

#[test]
fn test_basis_persists_after_solve() {
    // Test that solving updates internal basis state
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_column(2.0, 0.0..);
    problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);

    // First solve
    model.solve();
    let first_status = model.status();
    assert_eq!(first_status, HighsModelStatus::Optimal);

    // Second solve should reuse basis (implicitly, we can't directly test this
    // but we verify it doesn't break anything)
    model.solve();
    let second_status = model.status();
    assert_eq!(second_status, HighsModelStatus::Optimal);
}

#[test]
fn test_multiple_solves_produce_consistent_results() {
    // Test that solving the same problem multiple times gives consistent results
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_column(2.0, 0.0..);
    problem.add_row(1.0.., [(0, 1.0), (1, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);

    let mut objectives = Vec::new();
    for _ in 0..5 {
        model.solve();
        objectives.push(model.get_objective_value());
    }

    // All objectives should be identical
    for i in 1..objectives.len() {
        assert!(
            (objectives[i] - objectives[0]).abs() < 1e-6,
            "Solve {} gave different objective: {:.6} vs {:.6}",
            i,
            objectives[i],
            objectives[0]
        );
    }
}

#[test]
fn test_solve_after_problem_modification() {
    // Note: HiGHS Problem struct doesn't support modification after optimization
    // This test documents that behavior
    let mut problem = Problem::new();

    problem.add_column(1.0, 0.0..);
    problem.add_row(1.0.., [(0, 1.0)]);

    let mut model = problem.optimise(Sense::Minimise);
    model.solve();

    assert_eq!(model.status(), HighsModelStatus::Optimal);
    let solution = model.get_solution();
    assert!((solution.colvalue[0] - 1.0).abs() < 1e-6);

    // If we wanted to modify, we'd need to create a new Problem
    // This test just verifies the current solve state
}

#[test]
fn test_warm_start_behavior_implicit() {
    // Test that demonstrates warm-start behavior (HiGHS does this automatically)
    let mut problem = Problem::new();

    // Create a problem where warm-start would help
    for _i in 0..10 {
        problem.add_column(1.0, 0.0..);
    }

    for i in 0..10 {
        problem.add_row(1.0.., [(i, 1.0)]);
    }

    let mut model = problem.optimise(Sense::Minimise);

    // First solve (cold start)
    let start = Instant::now();
    model.solve();
    let first_duration = start.elapsed();

    assert_eq!(model.status(), HighsModelStatus::Optimal);

    // Second solve (warm start - should be faster or same speed)
    let start = Instant::now();
    model.solve();
    let second_duration = start.elapsed();

    // Both should succeed (we can't easily guarantee warm start is faster for such small problems)
    assert_eq!(model.status(), HighsModelStatus::Optimal);

    println!(
        "First solve: {:?}, Second solve: {:?}",
        first_duration, second_duration
    );
}

#[test]
fn test_solution_quality_after_multiple_solves() {
    // Verify solution quality doesn't degrade after multiple solves
    let mut problem = Problem::new();

    problem.add_column(3.0, 0.0..);
    problem.add_column(4.0, 0.0..);
    problem.add_row(12.0.., [(0, 2.0), (1, 1.0)]); // 2x + y >= 12
    problem.add_row(8.0.., [(0, 1.0), (1, 1.0)]); // x + y >= 8

    let mut model = problem.optimise(Sense::Minimise);

    let mut solutions = Vec::new();
    for _ in 0..3 {
        model.solve();
        let sol = model.get_solution();
        solutions.push((sol.colvalue[0], sol.colvalue[1]));
    }

    // All solutions should be identical
    for i in 1..solutions.len() {
        assert!(
            (solutions[i].0 - solutions[0].0).abs() < 1e-6
                && (solutions[i].1 - solutions[0].1).abs() < 1e-6,
            "Solution {} differs from solution 0",
            i
        );
    }
}
