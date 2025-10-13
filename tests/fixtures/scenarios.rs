#![allow(deprecated)]
// Test scenario fixtures
//
// Provides pre-configured scenario trees for testing stochastic optimization.
// Scenarios range from deterministic (single path) to stochastic (multiple branches).

use powers_rs::scenario::NoiseGenerator;
use rand_distr::{LogNormal, Normal};

/// Creates a deterministic scenario tree (single scenario per stage)
///
/// This is useful for testing algorithm mechanics without stochasticity.
/// All stages have exactly one scenario with probability 1.0.
///
/// # Parameters
/// - `num_stages`: Number of stages in the tree
/// - `num_load_entities`: Number of load entities (e.g., buses)
/// - `num_inflow_entities`: Number of inflow entities (e.g., hydro plants)
///
/// # Returns
/// A NoiseGenerator configured for deterministic scenarios
///
/// # Use Cases
/// - Testing basic SDDP mechanics
/// - Validating state transitions
/// - Debugging without randomness
/// - Baseline benchmarks
///
/// # Performance Note
/// Minimal memory footprint - single scenario per stage
pub fn deterministic_scenario(
    num_stages: usize,
    num_load_entities: usize,
    num_inflow_entities: usize,
) -> NoiseGenerator<Normal<f64>, LogNormal<f64>> {
    let mut generator = NoiseGenerator::new();

    // Create deterministic distributions (zero variance)
    // Using Normal(mean, 0.0001) for loads and LogNormal(mean, 0.0001) for inflows
    // Note: std_dev = 0 makes these degenerate distributions
    for _stage in 0..num_stages {
        let load_distributions = vec![
            Normal::new(1.0, 0.0001).unwrap(); // Near-zero variance
            num_load_entities
        ];
        let inflow_distributions = vec![
            LogNormal::new(1.0, 0.0001).unwrap(); // Near-zero variance
            num_inflow_entities
        ];

        generator.add_node_generator(
            load_distributions,
            inflow_distributions,
            1, // Single branching (deterministic)
        );
    }

    generator
}

/// Creates a simple stochastic scenario tree with uniform branching
///
/// Each stage has the same number of scenarios with equal probability.
/// Uses Normal distribution for loads and LogNormal for inflows.
///
/// # Parameters
/// - `num_stages`: Number of stages in the tree
/// - `num_scenarios_per_stage`: Number of scenarios at each stage
/// - `num_load_entities`: Number of load entities
/// - `num_inflow_entities`: Number of inflow entities
/// - `load_mean`: Mean for load normal distribution
/// - `load_std`: Standard deviation for load distribution
/// - `inflow_mean`: Mean for inflow log-normal distribution (log-scale)
/// - `inflow_std`: Standard deviation for inflow distribution (log-scale)
///
/// # Returns
/// A NoiseGenerator configured for stochastic scenarios
///
/// # Use Cases
/// - Testing stochastic optimization
/// - Testing scenario sampling
/// - Integration tests with randomness
/// - Convergence analysis
///
/// # Performance Note
/// Memory grows as O(scenarios^stages). Keep scenarios_per_stage small for tests.
#[allow(clippy::too_many_arguments)]
pub fn simple_stochastic_scenario(
    num_stages: usize,
    num_scenarios_per_stage: usize,
    num_load_entities: usize,
    num_inflow_entities: usize,
    load_mean: f64,
    load_std: f64,
    inflow_mean: f64,
    inflow_std: f64,
) -> NoiseGenerator<Normal<f64>, LogNormal<f64>> {
    let mut generator = NoiseGenerator::new();

    for _stage in 0..num_stages {
        let load_distributions =
            vec![Normal::new(load_mean, load_std).unwrap(); num_load_entities];
        let inflow_distributions = vec![
            LogNormal::new(inflow_mean, inflow_std)
                .unwrap();
            num_inflow_entities
        ];

        generator.add_node_generator(
            load_distributions,
            inflow_distributions,
            num_scenarios_per_stage,
        );
    }

    generator
}

/// Creates a fan scenario tree (many scenarios in first stage, single in others)
///
/// This is useful for testing scenario reduction and convergence with heavy sampling
/// at the first decision point.
///
/// # Parameters
/// - `num_stages`: Number of stages in the tree
/// - `first_stage_scenarios`: Number of scenarios in the first stage
/// - `num_load_entities`: Number of load entities
/// - `num_inflow_entities`: Number of inflow entities
///
/// # Returns
/// A NoiseGenerator with fan structure
///
/// # Use Cases
/// - Testing scenario reduction
/// - Testing convergence with detailed first-stage sampling
/// - Analyzing first-stage vs later-stage importance
///
/// # Performance Note
/// Concentrated sampling in first stage - memory efficient for multi-stage problems
pub fn fan_scenario(
    num_stages: usize,
    first_stage_scenarios: usize,
    num_load_entities: usize,
    num_inflow_entities: usize,
) -> NoiseGenerator<Normal<f64>, LogNormal<f64>> {
    let mut generator = NoiseGenerator::new();

    for stage in 0..num_stages {
        let load_distributions =
            vec![Normal::new(1.0, 0.2).unwrap(); num_load_entities];
        let inflow_distributions =
            vec![LogNormal::new(0.0, 0.3).unwrap(); num_inflow_entities];

        let num_scenarios = if stage == 0 {
            first_stage_scenarios
        } else {
            1 // Deterministic after first stage
        };

        generator.add_node_generator(
            load_distributions,
            inflow_distributions,
            num_scenarios,
        );
    }

    generator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_scenario() {
        let num_stages = 3;
        let num_load = 2;
        let num_inflow = 2;

        let mut generator =
            deterministic_scenario(num_stages, num_load, num_inflow);

        // Generate SAA
        let saa = generator.generate(42);

        // Each stage should have 1 scenario
        for stage in 0..num_stages {
            let node_gen = generator.get_node_generator(stage).unwrap();
            assert_eq!(
                node_gen.num_branchings, 1,
                "Stage {} should have 1 branching",
                stage
            );
            assert_eq!(
                node_gen.num_load_entities, num_load,
                "Stage {} should have {} load entities",
                stage, num_load
            );
            assert_eq!(
                node_gen.num_inflow_entities, num_inflow,
                "Stage {} should have {} inflow entities",
                stage, num_inflow
            );
        }

        // Validate SAA structure
        let noises = saa.get_noises_by_stage_and_branching(0, 0).unwrap();
        assert_eq!(noises.num_load_entities, num_load);
        assert_eq!(noises.num_inflow_entities, num_inflow);
    }

    #[test]
    fn test_simple_stochastic_scenario() {
        let num_stages = 2;
        let scenarios_per_stage = 3;
        let num_load = 2;
        let num_inflow = 1;

        let mut generator = simple_stochastic_scenario(
            num_stages,
            scenarios_per_stage,
            num_load,
            num_inflow,
            1.0, // load_mean
            0.2, // load_std
            0.0, // inflow_mean (log-scale)
            0.3, // inflow_std (log-scale)
        );

        // Each stage should have specified number of scenarios
        for stage in 0..num_stages {
            let node_gen = generator.get_node_generator(stage).unwrap();
            assert_eq!(
                node_gen.num_branchings, scenarios_per_stage,
                "Stage {} should have {} scenarios",
                stage, scenarios_per_stage
            );
        }

        // Generate and validate SAA
        let saa = generator.generate(123);
        for branching in 0..scenarios_per_stage {
            let noises =
                saa.get_noises_by_stage_and_branching(0, branching).unwrap();
            assert_eq!(noises.num_load_entities, num_load);
            assert_eq!(noises.num_inflow_entities, num_inflow);
        }
    }

    #[test]
    fn test_fan_scenario() {
        let num_stages = 4;
        let first_stage_scenarios = 10;
        let num_load = 3;
        let num_inflow = 2;

        let mut generator = fan_scenario(
            num_stages,
            first_stage_scenarios,
            num_load,
            num_inflow,
        );

        // First stage should have many scenarios
        let first_gen = generator.get_node_generator(0).unwrap();
        assert_eq!(first_gen.num_branchings, first_stage_scenarios);

        // Subsequent stages should have 1 scenario
        for stage in 1..num_stages {
            let node_gen = generator.get_node_generator(stage).unwrap();
            assert_eq!(
                node_gen.num_branchings, 1,
                "Stage {} should have 1 scenario",
                stage
            );
        }
    }

    #[test]
    fn test_scenario_reproducibility() {
        // Same seed should produce same scenarios
        let generator1 =
            simple_stochastic_scenario(2, 3, 1, 1, 1.0, 0.1, 0.0, 0.1);
        let generator2 =
            simple_stochastic_scenario(2, 3, 1, 1, 1.0, 0.1, 0.0, 0.1);

        let saa1 = generator1.generate(42);
        let saa2 = generator2.generate(42);

        // Compare first scenario values
        let noises1 = saa1.get_noises_by_stage_and_branching(0, 0).unwrap();
        let noises2 = saa2.get_noises_by_stage_and_branching(0, 0).unwrap();

        // Should be identical
        assert_eq!(
            noises1.get_load_noises(),
            noises2.get_load_noises(),
            "Same seed should produce same load noises"
        );
        assert_eq!(
            noises1.get_inflow_noises(),
            noises2.get_inflow_noises(),
            "Same seed should produce same inflow noises"
        );
    }

    #[test]
    fn test_scenario_size_for_performance() {
        // Verify fixtures have reasonable sizes for testing
        let mut det = deterministic_scenario(5, 2, 2);
        let mut simple =
            simple_stochastic_scenario(3, 3, 2, 2, 1.0, 0.1, 0.0, 0.1);

        // Deterministic should be minimal
        assert_eq!(det.get_node_generator(0).unwrap().num_branchings, 1);

        // Simple stochastic should be small enough for fast tests
        assert!(
            simple.get_node_generator(0).unwrap().num_branchings <= 10,
            "Stochastic scenarios should be compact for tests"
        );
    }
}
