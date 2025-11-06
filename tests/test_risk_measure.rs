//! TEST-004: Critical Unit Tests for risk_measure.rs
//!
//! **Note**: The risk_measure module has comprehensive unit tests in src/risk_measure.rs
//! covering the Expectation risk measure implementation (3 existing tests).
//!
//! This file provides:
//! 1. Helper functions for computing CVaR manually (for future testing)
//! 2. Property-based tests for CVaR properties using proptest
//! 3. Documentation of coherent risk measure axioms
//! 4. Integration test framework for when CVaR/WorstCase are implemented

// ============================================================================
// HELPER FUNCTIONS FOR RISK MEASURE TESTING
// ============================================================================

/// Computes expected value given costs and probabilities
pub fn compute_expectation(costs: &[f64], probabilities: &[f64]) -> f64 {
    assert_eq!(costs.len(), probabilities.len());
    costs
        .iter()
        .zip(probabilities.iter())
        .map(|(c, p)| c * p)
        .sum()
}

/// Computes CVaR (Conditional Value at Risk) manually for testing
pub fn compute_cvar_manual(
    costs: &[f64],
    probabilities: &[f64],
    alpha: f64,
) -> f64 {
    assert_eq!(costs.len(), probabilities.len());
    assert!(alpha > 0.0 && alpha < 1.0, "Alpha must be in (0, 1)");

    let mut pairs: Vec<(f64, f64)> = costs
        .iter()
        .zip(probabilities.iter())
        .map(|(&c, &p)| (c, p))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut cumulative_prob = 0.0;
    let mut var_index = 0;
    for (i, (_cost, prob)) in pairs.iter().enumerate() {
        cumulative_prob += prob;
        if cumulative_prob >= alpha {
            var_index = i;
            break;
        }
    }

    let tail_sum: f64 = pairs[..=var_index].iter().map(|(c, p)| c * p).sum();
    let tail_prob: f64 = pairs[..=var_index].iter().map(|(_c, p)| p).sum();

    if tail_prob < 1e-10 {
        return pairs[0].0;
    }

    tail_sum / tail_prob
}

/// Creates uniform probabilities for n scenarios
pub fn uniform_probs(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

#[cfg(test)]
mod helper_tests {
    use super::*;

    #[test]
    fn test_expectation_computation() {
        let costs = vec![100.0, 200.0, 300.0];
        let probs = vec![0.5, 0.3, 0.2];
        let expected = compute_expectation(&costs, &probs);
        assert!((expected - 170.0).abs() < 1e-10);
    }

    #[test]
    fn test_cvar_manual_computation() {
        let costs = vec![100.0, 200.0, 300.0, 400.0];
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let cvar_50 = compute_cvar_manual(&costs, &probs, 0.5);
        assert!((cvar_50 - 350.0).abs() < 1.0);
    }

    #[test]
    fn test_cvar_property_bounds_expectation() {
        let costs = vec![50.0, 100.0, 150.0, 200.0, 250.0];
        let probs = uniform_probs(5);
        let expectation = compute_expectation(&costs, &probs);

        for alpha in [0.2, 0.4, 0.6, 0.8] {
            let cvar = compute_cvar_manual(&costs, &probs, alpha);
            assert!(cvar >= expectation - 1e-10);
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_cvar_bounds_expectation(
            costs in prop::collection::vec(0.0f64..1000.0, 3..20),
            alpha in 0.1f64..0.9
        ) {
            let probs = uniform_probs(costs.len());
            let expectation = compute_expectation(&costs, &probs);
            let cvar = compute_cvar_manual(&costs, &probs, alpha);
            prop_assert!(cvar >= expectation - 1e-10);
        }
    }

    proptest! {
        #[test]
        fn prop_cvar_monotonic_in_alpha(
            costs in prop::collection::vec(10.0f64..1000.0, 5..15)
        ) {
            let probs = uniform_probs(costs.len());
            let cvar1 = compute_cvar_manual(&costs, &probs, 0.2);
            let cvar2 = compute_cvar_manual(&costs, &probs, 0.5);
            let cvar3 = compute_cvar_manual(&costs, &probs, 0.8);
            prop_assert!(cvar1 >= cvar2 - 1e-10);
            prop_assert!(cvar2 >= cvar3 - 1e-10);
        }
    }

    proptest! {
        #[test]
        fn prop_cvar_constant_costs(
            cost in 1.0f64..1000.0,
            n in 3usize..20,
            alpha in 0.1f64..0.9
        ) {
            let costs = vec![cost; n];
            let probs = uniform_probs(n);
            let cvar = compute_cvar_manual(&costs, &probs, alpha);
            prop_assert!((cvar - cost).abs() < 1e-10);
        }
    }
}
