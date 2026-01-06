// Simple 2-stage reservoir problem for integration testing
//
// PROBLEM DESCRIPTION:
// -------------------
// A single reservoir system over 2 stages with stochastic inflows in stage 2.
// The goal is to maximize expected hydropower revenue over both stages.
//
// SYSTEM:
// - 1 Hydro: 100 MWh storage capacity, 50 MW/h turbining capacity, productivity 1.0
// - 1 Bus with 100 $/MWh deficit cost
// - 1 Thermal: 30 MW capacity, 20 $/MWh cost (expensive backup)
// - Load: Fixed at 25 MW/h in both stages
//
// STAGES:
// - Stage 0 (PreStudy): Initial condition with 50 MWh storage, deterministic
// - Stage 1: First decision, deterministic inflow = 20 MWh
// - Stage 2: Second decision, stochastic inflow with 3 scenarios:
//     * Scenario 1 (Dry):    10 MWh inflow (prob 1/3)
//     * Scenario 2 (Average): 20 MWh inflow (prob 1/3)
//     * Scenario 3 (Wet):    30 MWh inflow (prob 1/3)
//
// DECISION TRADE-OFF:
// Stage 1 must balance:
// - Using water now (immediate hydropower revenue, avoid thermal cost)
// - Saving water for stage 2 (hedge against dry scenario)
//
// EXPECTED SOLUTION (APPROXIMATE):
// Using backward induction:
// - Stage 2 policy: Turbine all available water (terminal stage)
// - Stage 1 policy: Release ~20-25 MWh (balance current vs future value)
// - Expected cost: ~0-100 $ (mostly hydro, minimal thermal/deficit)
//
// This problem tests:
// 1. Forward pass with state transitions
// 2. Backward pass with cut generation
// 3. Convergence of upper/lower bounds
// 4. Correct handling of stochastic scenarios
// 5. Terminal condition handling

use powers_rs::initial_condition::InitialCondition;
use powers_rs::scenario::{NoiseGenerator, ScenarioTree};
use powers_rs::system::System;
use rand_distr::Normal;

/// Create JSON representation of a simple 2-stage hydrothermal system
///
/// System characteristics (inspired by examples/03-multistage):
/// - 1 bus with deficit cost (50 $/MWh)
/// - 2 thermal plants (5 & 10 $/MWh, 15 MW each = 30 MW total)
/// - 1 hydro plant (60 MW turbining capacity)
/// - Load: 75 MW (exceeds hydro capacity, forces thermal dispatch)
///
/// This system creates a meaningful water value optimization problem:
/// - Load (75 MW) > Hydro (60 MW) → Must use at least 15 MW thermal
/// - Two thermal options create economic dispatch problem
/// - Water value trade-off: use hydro now vs save for later
/// - Expected cost: ~1500-2000 $ over 2 stages (primarily thermal costs)
pub fn create_simple_2stage_system_json() -> String {
    r#"{
    "buses": [
        {
            "id": 0,
            "deficit_cost": 50.0
        }
    ],
    "lines": [],
    "thermals": [
        {
            "id": 0,
            "bus_id": 0,
            "cost": 5.0,
            "min_generation": 0.0,
            "max_generation": 15.0
        },
        {
            "id": 1,
            "bus_id": 0,
            "cost": 10.0,
            "min_generation": 0.0,
            "max_generation": 15.0
        }
    ],
    "hydros": [
        {
            "id": 0,
            "downstream_hydro_id": null,
            "bus_id": 0,
            "productivity": 1.0,
            "min_storage": 0.0,
            "max_storage": 100.0,
            "min_turbined_flow": 0.0,
            "max_turbined_flow": 60.0,
            "spillage_penalty": 0.01
        }
    ]
}"#
    .to_string()
}

/// Parse and build the System from JSON
pub fn create_simple_2stage_system() -> System {
    let json = create_simple_2stage_system_json();
    let input: powers_rs::input::SystemInput =
        serde_json::from_str(&json).expect("Failed to parse test system JSON");
    input.build_sddp_system().expect("Failed to build system")
}

/// Create scenario generator for 2-stage problem
///
/// Based on examples/03-multistage with adaptations for 2-stage testing.
///
/// IMPORTANT: The SAA must have entries for ALL nodes in the graph, including PreStudy!
///
/// Node 0 (PreStudy): Deterministic (not really used, but must exist for indexing)
/// Node 1 (Stage 1): Deterministic
/// - Load: 75 MW (exceeds hydro capacity of 60 MW)
/// - Inflow: 40 MWh (deterministic via zero variance)
///
/// Node 2 (Stage 2): Stochastic with 5 scenarios
/// - Load: 75 MW (fixed)
/// - Inflow: Mean 40 MWh, Std dev 20 MWh (creates variability)
///
/// Key feature: Load > Hydro capacity forces thermal dispatch and creates
/// meaningful water value decisions.
///
/// ARCHITECTURE NOTE: The SAA is indexed by node_id in the graph, so it must
/// have the same number of entries as nodes in the graph (3 in our case).
pub fn create_2stage_scenario_generator(
) -> NoiseGenerator<Normal<f64>, Normal<f64>> {
    let mut generator = NoiseGenerator::new();

    // Node 0 (PreStudy): Deterministic, 1 branching
    let prestudy_load = vec![Normal::new(75.0, 0.0).unwrap()];
    let prestudy_inflow = vec![Normal::new(0.0, 0.0).unwrap()]; // No inflow for prestudy
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Node 1 (Stage 1): Deterministic (1 branching, zero variance)
    // Load: 75 MW (forces 15 MW thermal), Inflow: 40 MWh
    let stage1_load = vec![Normal::new(75.0, 0.0).unwrap()]; // 1 bus
    let stage1_inflow = vec![Normal::new(40.0, 0.0).unwrap()]; // 1 hydro
    generator.add_node_generator(stage1_load, stage1_inflow, 1);

    // Node 2 (Stage 2): Stochastic (5 branchings for better representation)
    // Load: 75 MW (fixed), Inflow: Mean 40, Std 20 (20-60 MWh range approximately)
    let stage2_load = vec![Normal::new(75.0, 0.0).unwrap()]; // Fixed load
    let stage2_inflow = vec![Normal::new(40.0, 20.0).unwrap()]; // Significant variance
    generator.add_node_generator(stage2_load, stage2_inflow, 5); // 5 scenarios

    generator
}

/// Generate the ScenarioTree (Sample Average Approximation) for the 2-stage problem
///
/// This creates the scenario tree with:
/// - Stage 1: 1 deterministic scenario
/// - Stage 2: 3 stochastic scenarios
///
/// PERFORMANCE NOTE: ScenarioTree generation happens once per test, not performance-critical.
pub fn generate_2stage_saa(seed: u64) -> ScenarioTree {
    let generator = create_2stage_scenario_generator();
    generator.generate(seed)
}

/// Create initial condition for 2-stage test
///
/// Initial state (REDUCED from examples/03-multistage to create water scarcity):
/// - Storage: 40 MWh (was 100.0 in examples/03-multistage)
///
/// This creates a TRUE water value problem:
/// - NOT enough water to turbine 60 MW for both stages
/// - Stage 1 + Stage 2: Need ~120 MWh total for full hydro dispatch
/// - Have: 40 (storage) + 40 (S1 inflow) + 40 (S2 inflow) = 120 MWh (tight!)
/// - Dry scenarios in Stage 2: Must decide whether to save water or use thermal
/// - Creates meaningful water value learning opportunity for SDDP
pub fn create_simple_2stage_initial_condition() -> InitialCondition {
    InitialCondition::new(
        vec![40.0], // REDUCED initial storage to create scarcity
        vec![],     // No inflow lags
    )
}

/// Helper to compute expected solution analytically
///
/// SYSTEM CHARACTERISTICS (adapted from examples/03-multistage for water scarcity):
/// - Load: 75 MW (constant)
/// - Hydro: 60 MW max turbining, 40 MWh initial storage (REDUCED to create scarcity)
/// - Thermal 0: 15 MW @ 5 $/MWh (cheap baseload)
/// - Thermal 1: 15 MW @ 10 $/MWh (expensive peaker)
/// - Inflows: Mean 40 MWh per stage, Std dev 20 MWh in Stage 2
///
/// WATER SCARCITY ANALYSIS:
/// - Ideal hydro dispatch: 60 MWh per stage × 2 stages = 120 MWh needed
/// - Available: 40 (initial) + 40 (S1) + 40 (S2 mean) = 120 MWh (exactly!)
/// - Dry Stage 2 scenarios: Only ~20 MWh inflow → Water shortage!
/// - Wet Stage 2 scenarios: ~60 MWh inflow → Water surplus
///
/// OPTIMAL STRATEGY (what SDDP should learn):
/// - Stage 1: Save some water for potential dry Stage 2
/// - If use all 60 MWh hydro in S1: Risk running out in dry S2
/// - If save too much: Pay unnecessary thermal cost in S1
/// - Water value: How much is 1 MWh of storage worth?
///
/// EXPECTED COSTS:
/// - Best case (wet S2): ~150-200 $ (mostly hydro, minimal thermal)
/// - Worst case (dry S2): ~300-400 $ (more thermal due to water shortage)
/// - Expected total: ~200-300 $ with proper water management
///
/// CONVERGENCE EXPECTATION:
/// - Initial lower bound: ~100-150 $ (underestimates water value)
/// - Improving lower bound: Should increase to ~180-220 $ as water value is learned
/// - Upper bound: ~200-300 $ from simulations
/// - Gap should close as cuts refine the water value function
///
/// # Solution Estimate
///
/// Rough estimate based on problem structure:
///
/// **Cost Structure:**
/// - Hydro generation: nearly free (just spillage penalty)
/// - Thermal: 20 $/MWh
/// - Deficit: 100 $/MWh (should never occur with available capacities)
///
/// **Resource Balance:**
/// - Total load: 25 MW × 2h = 50 MWh
/// - Initial storage: 50 MWh
/// - Expected inflow: Stage 1 (20) + Stage 2 (avg 20) = 40 MWh
/// - Total available: ~90 MWh (sufficient for 50 MWh load)
///
/// **Expected Cost Range:**
/// - Best case: 0 $ (all hydro, no spillage)
/// - Typical case: 10-50 $ (minimal thermal in dry scenarios)
/// - Worst case: 100 $ (some thermal use)
///
/// Returns conservative validation bounds: (0.0, 150.0)
#[allow(dead_code)]
pub fn expected_solution_bounds() -> (f64, f64) {
    (0.0, 150.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let system = create_simple_2stage_system();

        assert_eq!(system.meta.buses_count, 1);
        assert_eq!(system.meta.hydros_count, 1);
        assert_eq!(system.meta.thermals_count, 2); // Two thermal plants now
        assert_eq!(system.meta.lines_count, 0);

        // Verify hydro parameters (60 MW max turbining)
        assert_eq!(system.hydros[0].max_turbined_flow, 60.0);
        assert_eq!(system.hydros[0].max_storage, 100.0);
    }

    #[test]
    fn test_scenario_generator() {
        let generator = create_2stage_scenario_generator();

        assert_eq!(generator.node_generators.len(), 3); // PreStudy + 2 study stages
        assert_eq!(generator.node_generators[0].num_branchings, 1); // PreStudy deterministic
        assert_eq!(generator.node_generators[1].num_branchings, 1); // Stage 1 deterministic
        assert_eq!(generator.node_generators[2].num_branchings, 5); // Stage 2 stochastic (5 scenarios)
    }

    #[test]
    fn test_saa_generation() {
        let saa = generate_2stage_saa(42);

        // Verify structure - should have 3 nodes (PreStudy + 2 study stages)
        assert_eq!(saa.stage_scenarios.len(), 3);
        assert_eq!(saa.get_branching_count_at_stage(0).unwrap(), 1); // PreStudy
        assert_eq!(saa.get_branching_count_at_stage(1).unwrap(), 1); // Stage 1
        assert_eq!(saa.get_branching_count_at_stage(2).unwrap(), 5); // Stage 2 (5 scenarios)

        // Verify noises are accessible
        let prestudy_noises =
            saa.get_noises_by_stage_and_branching(0, 0).unwrap();
        assert_eq!(prestudy_noises.num_load_entities, 1);
        assert_eq!(prestudy_noises.num_inflow_entities, 1);

        let stage1_noises =
            saa.get_noises_by_stage_and_branching(1, 0).unwrap();
        assert_eq!(stage1_noises.num_load_entities, 1);
        assert_eq!(stage1_noises.num_inflow_entities, 1);

        let stage2_noises =
            saa.get_noises_by_stage_and_branching(2, 0).unwrap();
        assert_eq!(stage2_noises.num_load_entities, 1);
        assert_eq!(stage2_noises.num_inflow_entities, 1);
    }

    #[test]
    fn test_initial_condition() {
        let ic = create_simple_2stage_initial_condition();

        assert_eq!(ic.get_storage().len(), 1);
        assert_eq!(ic.get_storage()[0], 40.0); // Reduced initial storage for water scarcity
    }
}
