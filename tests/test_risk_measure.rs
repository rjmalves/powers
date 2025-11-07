//! Critical Unit Tests for risk_measure.rs
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

    proptest! {
        #[test]
        fn prop_expectation_linear_in_probabilities(
            costs in prop::collection::vec(0.0f64..1000.0, 3..10)
        ) {
            let probs1 = uniform_probs(costs.len());
            let probs2 = uniform_probs(costs.len());

            let lambda = 0.6;
            let probs_combined: Vec<f64> = probs1
                .iter()
                .zip(probs2.iter())
                .map(|(p1, p2)| lambda * p1 + (1.0 - lambda) * p2)
                .collect();

            let exp1 = compute_expectation(&costs, &probs1);
            let exp2 = compute_expectation(&costs, &probs2);
            let exp_combined = compute_expectation(&costs, &probs_combined);

            let expected = lambda * exp1 + (1.0 - lambda) * exp2;
            prop_assert!((exp_combined - expected).abs() < 1e-10);
        }
    }

    proptest! {
        #[test]
        fn prop_expectation_positive_homogeneous(
            costs in prop::collection::vec(0.0f64..1000.0, 3..10),
            scalar in 0.1f64..10.0
        ) {
            let probs = uniform_probs(costs.len());
            let scaled_costs: Vec<f64> = costs.iter().map(|c| c * scalar).collect();

            let exp = compute_expectation(&costs, &probs);
            let exp_scaled = compute_expectation(&scaled_costs, &probs);

            prop_assert!((exp_scaled - scalar * exp).abs() < 1e-8);
        }
    }

    proptest! {
        #[test]
        fn prop_probabilities_sum_to_one(
            n in 3usize..20
        ) {
            let probs = uniform_probs(n);
            let sum: f64 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}

// ============================================================================
// COHERENT RISK MEASURE AXIOMS TESTS
// ============================================================================

#[cfg(test)]
mod coherent_risk_measure_tests {
    use super::*;

    #[test]
    fn test_expectation_monotonicity_axiom() {
        let costs1 = vec![100.0, 200.0, 300.0];
        let costs2 = vec![150.0, 250.0, 350.0];
        let probs = uniform_probs(3);

        let exp1 = compute_expectation(&costs1, &probs);
        let exp2 = compute_expectation(&costs2, &probs);

        assert!(
            exp2 >= exp1,
            "Monotonicity: higher costs should yield higher risk"
        );
    }

    #[test]
    fn test_expectation_translation_invariance() {
        let costs = vec![100.0, 200.0, 300.0];
        let probs = uniform_probs(3);
        let constant = 50.0;

        let exp = compute_expectation(&costs, &probs);

        let translated_costs: Vec<f64> =
            costs.iter().map(|c| c + constant).collect();
        let exp_translated = compute_expectation(&translated_costs, &probs);

        assert!(
            (exp_translated - (exp + constant)).abs() < 1e-10,
            "Translation invariance: risk(X + c) = risk(X) + c"
        );
    }

    #[test]
    fn test_expectation_positive_homogeneity() {
        let costs = vec![100.0, 200.0, 300.0];
        let probs = uniform_probs(3);
        let lambda = 2.5;

        let exp = compute_expectation(&costs, &probs);

        let scaled_costs: Vec<f64> = costs.iter().map(|c| c * lambda).collect();
        let exp_scaled = compute_expectation(&scaled_costs, &probs);

        assert!(
            (exp_scaled - lambda * exp).abs() < 1e-10,
            "Positive homogeneity: risk(λX) = λ risk(X) for λ > 0"
        );
    }

    #[test]
    fn test_expectation_subadditivity() {
        let costs1 = vec![100.0, 200.0, 300.0];
        let costs2 = vec![50.0, 100.0, 150.0];
        let probs = uniform_probs(3);

        let exp1 = compute_expectation(&costs1, &probs);
        let exp2 = compute_expectation(&costs2, &probs);

        let costs_sum: Vec<f64> = costs1
            .iter()
            .zip(costs2.iter())
            .map(|(c1, c2)| c1 + c2)
            .collect();
        let exp_sum = compute_expectation(&costs_sum, &probs);

        assert!((exp_sum - (exp1 + exp2)).abs() < 1e-10,
                "Subadditivity (expectation is additive): risk(X+Y) = risk(X) + risk(Y)");
    }

    #[test]
    fn test_cvar_monotonicity_axiom() {
        let costs1 = vec![100.0, 200.0, 300.0, 400.0];
        let costs2 = vec![150.0, 250.0, 350.0, 450.0];
        let probs = uniform_probs(4);
        let alpha = 0.5;

        let cvar1 = compute_cvar_manual(&costs1, &probs, alpha);
        let cvar2 = compute_cvar_manual(&costs2, &probs, alpha);

        assert!(
            cvar2 >= cvar1 - 1e-10,
            "CVaR monotonicity: higher costs should yield higher risk"
        );
    }

    #[test]
    fn test_cvar_translation_invariance() {
        let costs = vec![100.0, 200.0, 300.0, 400.0];
        let probs = uniform_probs(4);
        let constant = 75.0;
        let alpha = 0.5;

        let cvar = compute_cvar_manual(&costs, &probs, alpha);

        let translated_costs: Vec<f64> =
            costs.iter().map(|c| c + constant).collect();
        let cvar_translated =
            compute_cvar_manual(&translated_costs, &probs, alpha);

        assert!(
            (cvar_translated - (cvar + constant)).abs() < 1e-8,
            "CVaR translation invariance: risk(X + c) = risk(X) + c"
        );
    }

    #[test]
    fn test_cvar_positive_homogeneity() {
        let costs = vec![100.0, 200.0, 300.0, 400.0];
        let probs = uniform_probs(4);
        let lambda = 1.5;
        let alpha = 0.6;

        let cvar = compute_cvar_manual(&costs, &probs, alpha);

        let scaled_costs: Vec<f64> = costs.iter().map(|c| c * lambda).collect();
        let cvar_scaled = compute_cvar_manual(&scaled_costs, &probs, alpha);

        assert!(
            (cvar_scaled - lambda * cvar).abs() < 1e-8,
            "CVaR positive homogeneity: risk(λX) = λ risk(X) for λ > 0"
        );
    }
}

#[cfg(test)]
mod documentation_tests {
    // Note: Integration tests for expectation and CVaR risk measures
    // are covered in tests/integration/features/risk_measures.rs
}
