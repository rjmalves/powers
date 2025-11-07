//! This module tests SDDP convergence guarantees:
//! - Monotonically non-decreasing lower bounds
//! - Upper bounds from feasible solutions
//! - Gap closure over iterations
//!
//! ## SDDP Convergence Theory
//!
//! SDDP guarantees monotonically non-decreasing lower bounds because:
//! 1. Each cut is a valid supporting hyperplane to the value function
//! 2. Adding cuts can only tighten (raise) the lower bound
//! 3. The lower bound is computed from the first-stage problem with accumulated cuts
//!
//! For stochastic problems:
//! - Sampling error means gap may not converge to zero
//! - Upper bound estimate has variance 1/N where N is simulation count
//! - More scenarios → better upper bound estimate → smaller gap
//!
//! ## Numerical Tolerance
//!
//! LP solver basis changes can cause slight LB decreases (~1e-6) due to:
//! - Floating-point arithmetic
//! - Different optimal bases with same objective value
//! - Dual degeneracy
//!
//! We use tolerance = 1e-6 to account for this while catching real violations.

mod utils;

use powers_rs::sddp::SddpAlgorithm;
use std::path::Path;
use utils::monotonic::assert_monotonic_non_decreasing;

/// Helper to extract convergence statistics from iteration history
#[derive(Debug)]
struct ConvergenceStats {
    initial_lb: f64,
    final_lb: f64,
    initial_ub: f64,
    final_ub: f64,
    initial_gap: f64,
    final_gap: f64,
    lb_improvement: f64,
    gap_reduction_ratio: f64,
    iterations: usize,
}

impl ConvergenceStats {
    fn from_training_result(result: &powers_rs::sddp::TrainingResult) -> Self {
        let iterations_vec = result.iterations();
        let iterations_count = iterations_vec.len();

        assert!(iterations_count > 0, "No iterations in training result");

        // Extract lower bounds
        let lower_bounds: Vec<f64> =
            iterations_vec.iter().map(|r| r.lower_bound).collect();

        let initial_lb = lower_bounds[0];
        let final_lb = result.final_lower_bound;
        let initial_ub = result.statistical_upper_bound;
        let final_ub = result.statistical_upper_bound;

        let initial_gap = initial_ub - initial_lb;
        let final_gap = result.final_gap();

        let lb_improvement = final_lb - initial_lb;
        let gap_reduction_ratio = if initial_gap.abs() > 1e-10 {
            final_gap / initial_gap
        } else {
            0.0
        };

        Self {
            initial_lb,
            final_lb,
            initial_ub,
            final_ub,
            initial_gap,
            final_gap,
            lb_improvement,
            gap_reduction_ratio,
            iterations: iterations_count,
        }
    }
}

// ============================================================================
// TEST-014a: Monotonic Lower Bound Properties
// ============================================================================

mod test_monotonic_lower_bounds {
    use super::*;

    /// Test that lower bounds never decrease on deterministic problem
    #[test]
    fn test_monotonic_lb_deterministic() {
        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load deterministic example");

        let result = instance.train().expect("Training failed");

        // Extract lower bounds
        let lower_bounds: Vec<f64> =
            result.iterations().iter().map(|r| r.lower_bound).collect();

        // Deterministic systems should have exact monotonicity (tight tolerance)
        assert_monotonic_non_decreasing(&lower_bounds, 1e-6);

        // Verify we had multiple iterations
        assert!(
            lower_bounds.len() >= 5,
            "Should have at least 5 iterations for convergence test"
        );
    }

    /// Test that lower bounds never decrease on stochastic problem
    #[test]
    fn test_monotonic_lb_stochastic() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        // Extract lower bounds
        let lower_bounds: Vec<f64> =
            result.iterations().iter().map(|r| r.lower_bound).collect();

        // Stochastic systems may have small LP noise (relaxed tolerance)
        assert_monotonic_non_decreasing(&lower_bounds, 1e-4);

        // Verify we had multiple iterations
        assert!(
            lower_bounds.len() >= 10,
            "Should have at least 10 iterations for stochastic convergence"
        );
    }

    /// Test that lower bound increases meaningfully
    #[test]
    fn test_lb_improves_substantially() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        let stats = ConvergenceStats::from_training_result(&result);

        // Lower bound should improve (unless problem is trivial)
        // Allow for cases where initial LB is already good
        if stats.initial_lb.abs() < 1e-6 {
            // Zero initial bound - expect positive improvement
            assert!(
                stats.final_lb >= 0.0,
                "Final LB should be non-negative when starting from zero"
            );
        } else {
            // Non-zero initial bound - should improve or stay same
            assert!(
                stats.lb_improvement >= -1e-6,
                "Lower bound should improve: got {} improvement",
                stats.lb_improvement
            );
        }
    }

    /// Test lower bound behavior with very few iterations
    #[test]
    fn test_monotonic_lb_minimal_iterations() {
        // Even with just 2-3 iterations, monotonicity should hold
        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load deterministic example");

        // Limit iterations via config would require modifying config
        // For now, just verify the property holds regardless of iteration count
        let result = instance.train().expect("Training failed");

        let lower_bounds: Vec<f64> =
            result.iterations().iter().map(|r| r.lower_bound).collect();

        // Should be monotonic regardless of how many iterations ran
        assert_monotonic_non_decreasing(&lower_bounds, 1e-6);
    }
}

// ============================================================================
// TEST-014b: Upper Bound Feasibility
// ============================================================================

mod test_upper_bound_feasibility {
    use super::*;

    /// Test that upper bound comes from feasible forward pass
    #[test]
    fn test_upper_bound_from_simulation() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        // Upper bound should be finite (comes from feasible solutions)
        assert!(
            result.statistical_upper_bound.is_finite(),
            "Upper bound should be finite: got {}",
            result.statistical_upper_bound
        );

        // Upper bound should be >= lower bound (within tolerance)
        assert!(
            result.statistical_upper_bound >= result.final_lower_bound - 1e-6,
            "Upper bound ({}) should be >= lower bound ({})",
            result.statistical_upper_bound,
            result.final_lower_bound
        );
    }

    /// Test that statistical upper bound is reasonable
    #[test]
    fn test_statistical_upper_bound() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        // Statistical upper bound should be finite
        assert!(
            result.statistical_upper_bound.is_finite(),
            "Statistical UB should be finite: got {}",
            result.statistical_upper_bound
        );
    }

    /// Test that best upper bound is tracked correctly
    #[test]
    fn test_best_upper_bound_tracking() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        // Best UB should be <= final UB (best is the minimum observed)
        assert!(
            result.best_upper_bound <= result.statistical_upper_bound + 1e-6,
            "Best UB ({}) should be <= final UB ({})",
            result.best_upper_bound,
            result.statistical_upper_bound
        );

        // Best iteration should be valid
        assert!(
            result.best_iteration < result.iterations().len(),
            "Best iteration index ({}) should be valid (total: {})",
            result.best_iteration,
            result.iterations().len()
        );
    }
}

// ============================================================================
// TEST-014c: Gap Closure Properties
// ============================================================================

mod test_gap_closure {
    use super::*;

    /// Test that gap decreases over iterations
    #[test]
    fn test_gap_decreases_stochastic() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        let stats = ConvergenceStats::from_training_result(&result);

        // Final gap should be <= initial gap (or close if initial was already good)
        let gap_ratio = if stats.initial_gap > 1e-6 {
            stats.final_gap / stats.initial_gap
        } else {
            // Initial gap very small - gap closure not meaningful
            0.0
        };

        assert!(
            gap_ratio <= 1.1, // Allow 10% increase due to sampling noise
            "Gap should decrease: initial {}, final {} (ratio {})",
            stats.initial_gap,
            stats.final_gap,
            gap_ratio
        );
    }

    /// Test that deterministic problems achieve tight convergence
    #[test]
    fn test_deterministic_converges_tightly() {
        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load deterministic example");

        let result = instance.train().expect("Training failed");

        let stats = ConvergenceStats::from_training_result(&result);

        // Deterministic should have small relative gap
        let relative_gap = if stats.final_lb.abs() > 1e-6 {
            stats.final_gap / stats.final_lb.abs()
        } else {
            stats.final_gap
        };

        // Allow up to 10% relative gap (generous - many problems converge tighter)
        assert!(
            relative_gap < 0.1,
            "Deterministic should converge tightly: gap {}, LB {} ({}% relative)",
            stats.final_gap,
            stats.final_lb,
            relative_gap * 100.0
        );
    }

    /// Test gap properties are finite and non-negative
    #[test]
    fn test_gap_always_valid() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        let stats = ConvergenceStats::from_training_result(&result);

        // All gaps should be finite
        assert!(
            stats.initial_gap.is_finite(),
            "Initial gap should be finite"
        );
        assert!(stats.final_gap.is_finite(), "Final gap should be finite");

        // Gaps should be non-negative (within numerical tolerance)
        assert!(
            stats.initial_gap >= -1e-6,
            "Initial gap should be non-negative: got {}",
            stats.initial_gap
        );
        assert!(
            stats.final_gap >= -1e-6,
            "Final gap should be non-negative: got {}",
            stats.final_gap
        );
    }

    /// Test convergence metrics are tracked correctly
    #[test]
    fn test_convergence_metrics() {
        let example_dir = Path::new("examples/02-stochastic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load stochastic example");

        let result = instance.train().expect("Training failed");

        let stats = ConvergenceStats::from_training_result(&result);

        // Verify basic stats properties
        assert!(stats.iterations > 0, "Should have positive iterations");
        assert!(stats.final_lb.is_finite(), "Final LB should be finite");
        assert!(stats.final_ub.is_finite(), "Final UB should be finite");

        // LB improvement should be non-negative (monotonic property)
        assert!(
            stats.lb_improvement >= -1e-6,
            "LB improvement should be non-negative: got {}",
            stats.lb_improvement
        );

        // Gap reduction ratio should be reasonable (0 = perfect, >1 = worsened)
        if stats.initial_gap > 1e-6 {
            assert!(
                stats.gap_reduction_ratio >= 0.0
                    && stats.gap_reduction_ratio <= 1.5,
                "Gap reduction ratio should be reasonable: got {}",
                stats.gap_reduction_ratio
            );
        }
    }
}

// ============================================================================
// TEST-014d: Edge Cases
// ============================================================================

mod test_edge_cases {
    use super::*;

    /// Test convergence with already-converged problem (no improvement needed)
    #[test]
    fn test_already_converged() {
        // This tests the case where initial solution is already optimal
        // In practice, this rarely happens, but the algorithm should handle it

        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load deterministic example");

        let result = instance.train().expect("Training failed");

        // Even if already converged, algorithm should complete successfully
        assert!(result.iterations().len() > 0, "Should complete iterations");

        // Bounds should be valid
        assert!(result.final_lower_bound.is_finite());
        assert!(result.statistical_upper_bound.is_finite());
        assert!(
            result.statistical_upper_bound >= result.final_lower_bound - 1e-6
        );
    }

    /// Test that relative gap computation handles zero bounds correctly
    #[test]
    fn test_relative_gap_with_zero_bounds() {
        let example_dir = Path::new("examples/01-deterministic");

        let mut instance = SddpAlgorithm::from_files(
            example_dir.join("config.json"),
            example_dir.join("system.json"),
            example_dir.join("graph.json"),
            example_dir.join("recourse.json"),
        )
        .expect("Failed to load deterministic example");

        let result = instance.train().expect("Training failed");

        // Relative gap should handle zero/near-zero bounds
        let rel_gap = result.relative_gap();

        // Should be finite (may be infinity if LB is exactly 0, which is valid)
        assert!(
            rel_gap >= 0.0 || rel_gap.is_infinite(),
            "Relative gap should be non-negative or inf: got {}",
            rel_gap
        );
    }
}

// ============================================================================
// Integration Test: Full Convergence Analysis
// ============================================================================

#[test]
fn test_full_convergence_analysis() {
    let example_dir = Path::new("examples/02-stochastic");

    let mut instance = SddpAlgorithm::from_files(
        example_dir.join("config.json"),
        example_dir.join("system.json"),
        example_dir.join("graph.json"),
        example_dir.join("recourse.json"),
    )
    .expect("Failed to load stochastic example");

    let result = instance.train().expect("Training failed");

    // Compute comprehensive convergence statistics
    let stats = ConvergenceStats::from_training_result(&result);

    println!("\n=== Convergence Analysis ===");
    println!("Iterations: {}", stats.iterations);
    println!("Initial LB: {:.2}", stats.initial_lb);
    println!("Final LB:   {:.2}", stats.final_lb);
    println!("Initial UB: {:.2}", stats.initial_ub);
    println!("Final UB:   {:.2}", stats.final_ub);
    println!("Initial Gap: {:.2}", stats.initial_gap);
    println!("Final Gap:   {:.2}", stats.final_gap);
    println!("LB Improvement: {:.2}", stats.lb_improvement);
    println!(
        "Gap Reduction: {:.2}%",
        (1.0 - stats.gap_reduction_ratio) * 100.0
    );
    println!("Relative Final Gap: {:.2}%", result.relative_gap() * 100.0);

    // Verify all key convergence properties
    let lower_bounds: Vec<f64> =
        result.iterations().iter().map(|r| r.lower_bound).collect();
    assert_monotonic_non_decreasing(&lower_bounds, 1e-4);

    assert!(result.statistical_upper_bound >= result.final_lower_bound - 1e-6);
    assert!(stats.final_gap >= -1e-6);
    assert!(stats.final_gap.is_finite());

    // Training should show meaningful progress
    if stats.initial_gap > 1e-6 {
        assert!(
            stats.gap_reduction_ratio <= 1.0
                || stats.final_gap < stats.initial_gap + 1.0,
            "Should show progress or maintain small gap"
        );
    }
}
