//! Tests for scenario generation and SAA construction
//!
//! This module verifies the behavior of ScenarioGenerator and validates
//! that the legacy par_states lag buffer does not affect SDDP execution.

use powers_rs::initial_condition::InitialCondition;
use powers_rs::input::{MarginalDistribution, UncertaintyType};
use powers_rs::scenario_generator::ScenarioGenerator;
use powers_rs::temporal_model::TemporalModel;

/// Create a simple PAR(1) model for testing
fn create_test_par_model() -> TemporalModel {
    TemporalModel::from_par(
        UncertaintyType::Inflow,
        0,
        1,
        vec![100.0],
        vec![10.0],
        vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 10.0,
        }],
        vec![1],
        vec![vec![0.5]],
    )
    .unwrap()
}

/// Test that innovations are deterministic given the same RNG seed
///
/// This validates the claim in SCENARIO_GENERATION_ANALYSIS.md that
/// innovations (what goes to SAA) are deterministic.
#[test]
fn test_par_states_independence() {
    let model = create_test_par_model();
    let initial_condition = InitialCondition::new(vec![], vec![vec![]]);

    let seed = 12345u64;

    let mut generator1 =
        ScenarioGenerator::new(vec![model.clone()], &initial_condition, None)
            .expect("Failed to create generator 1");

    let mut generator2 =
        ScenarioGenerator::new(vec![model], &initial_condition, None)
            .expect("Failed to create generator 2");

    use rand::SeedableRng;
    let mut rng1 = rand::rngs::StdRng::seed_from_u64(seed);
    let mut rng2 = rand::rngs::StdRng::seed_from_u64(seed);

    let scenarios1 = generator1.generate_stage_scenarios(0, 5, &mut rng1);
    let scenarios2 = generator2.generate_stage_scenarios(0, 5, &mut rng2);

    // Verify innovations are identical (this is what goes to SAA)
    for (scen1, scen2) in
        scenarios1.scenarios.iter().zip(scenarios2.scenarios.iter())
    {
        let diff = (scen1.innovations[0] - scen2.innovations[0]).abs();
        assert!(
            diff < 1e-10,
            "Innovations must be identical for same seed: {} vs {}",
            scen1.innovations[0],
            scen2.innovations[0]
        );
    }
}

/// Test that scenario generation produces reasonable statistical properties
///
/// After SG-003, PAR models only store innovations. Values are placeholders (0.0).
/// Innovations are transformed through the marginal distribution's inverse CDF,
/// so they're in the scale of the target distribution (not standardized).
#[test]
fn test_par_scenario_generation_sanity() {
    let model = create_test_par_model();
    let initial_condition = InitialCondition::new(vec![], vec![vec![]]);

    let mut generator =
        ScenarioGenerator::new(vec![model], &initial_condition, None)
            .expect("Failed to create generator");

    let mut rng = rand::rng();
    let scenarios = generator.generate_stage_scenarios(0, 100, &mut rng);

    let mut innovation_sum = 0.0;

    for scenario in &scenarios.scenarios {
        let innovation = scenario.innovations[0];
        let value = scenario.values[0];

        // Innovations are transformed through inverse CDF: N(100, 10)
        // So they should be roughly in range [60, 140] (mean ± 4σ)
        // Using 4σ to account for statistical outliers with 100 samples
        assert!(
            (60.0..=140.0).contains(&innovation),
            "Innovation should be in reasonable range for N(100,10): got {}",
            innovation
        );
        assert_eq!(
            value, 0.0,
            "Value should be placeholder (0.0) for PAR models after SG-003"
        );

        innovation_sum += innovation;
    }

    let innovation_mean = innovation_sum / 100.0;

    // Mean should be close to 100.0 (the distribution mean)
    assert!(
        (95.0..=105.0).contains(&innovation_mean),
        "Innovation mean should be ~100: got {}",
        innovation_mean
    );
}

/// Test that scenario structure is populated correctly
///
/// After SG-004, residuals field is removed. Only values and innovations remain.
#[test]
fn test_scenario_structure_populated() {
    let model = create_test_par_model();
    let initial_condition = InitialCondition::new(vec![], vec![vec![]]);
    let mut generator =
        ScenarioGenerator::new(vec![model], &initial_condition, None)
            .expect("Failed to create generator");

    let mut rng = rand::rng();
    let scenarios = generator.generate_stage_scenarios(0, 10, &mut rng);

    // After SG-004: Residuals field removed, only values and innovations remain
    for scenario in &scenarios.scenarios {
        assert_eq!(scenario.values.len(), 1);
        assert_eq!(scenario.innovations.len(), 1);

        // Only innovations are meaningful for PAR models after SG-003
        assert!(scenario.innovations[0].is_finite());
        assert_eq!(
            scenario.values[0], 0.0,
            "Values are placeholders for PAR models"
        );
    }
}
