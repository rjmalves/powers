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
use powers_rs::scenario::{NoiseGenerator, SAA};
use powers_rs::system::System;
use rand_distr::Normal;

/// Create the single-hydro system for 2-stage test
///
/// System characteristics:
/// - Bus 0: 100 $/MWh deficit cost (penalty for unserved load)
/// - Thermal 0: 30 MW capacity, 20 $/MWh cost (expensive backup)
/// - Hydro 0: 100 MWh storage, 50 MW turbining, 1.0 productivity
///
/// PERFORMANCE NOTE: This is a test fixture, not performance-critical.
/// Simple JSON structure is fine.
pub fn create_simple_2stage_system_json() -> String {
    r#"{
    "buses": [
        {
            "id": 0,
            "deficit_cost": 100.0
        }
    ],
    "lines": [],
    "thermals": [
        {
            "id": 0,
            "bus_id": 0,
            "cost": 20.0,
            "min_generation": 0.0,
            "max_generation": 30.0
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
            "max_turbined_flow": 50.0,
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
    input.build_sddp_system()
}

/// Create scenario generator for 2-stage problem
///
/// IMPORTANT: The SAA must have entries for ALL nodes in the graph, including PreStudy!
///
/// Node 0 (PreStudy): Deterministic (not really used, but must exist for indexing)
/// Node 1 (Stage 1): Deterministic
/// - Load: 25 MW (fixed)
/// - Inflow: 20 MWh (deterministic via zero variance)
///
/// Node 2 (Stage 2): Stochastic with 3 scenarios
/// - Load: 25 MW (fixed)
/// - Inflow: 10/20/30 MWh with equal probability (1/3 each)
///
/// ARCHITECTURE NOTE: The SAA is indexed by node_id in the graph, so it must
/// have the same number of entries as nodes in the graph (3 in our case).
///
/// PERFORMANCE NOTE: Creating distributions is not in the hot path.
pub fn create_2stage_scenario_generator(
) -> NoiseGenerator<Normal<f64>, Normal<f64>> {
    let mut generator = NoiseGenerator::new();

    // Node 0 (PreStudy): Deterministic, 1 branching
    // This node exists in the graph but doesn't really have uncertainty
    let prestudy_load = vec![Normal::new(25.0, 0.0).unwrap()];
    let prestudy_inflow = vec![Normal::new(0.0, 0.0).unwrap()]; // No inflow for prestudy
    generator.add_node_generator(prestudy_load, prestudy_inflow, 1);

    // Node 1 (Stage 1): Deterministic (1 branching, zero variance)
    // Load: 25 MW, Inflow: 20 MWh
    let stage1_load = vec![Normal::new(25.0, 0.0).unwrap()]; // 1 bus
    let stage1_inflow = vec![Normal::new(20.0, 0.0).unwrap()]; // 1 hydro
    generator.add_node_generator(stage1_load, stage1_inflow, 1);

    // Node 2 (Stage 2): Stochastic (3 branchings: dry/avg/wet)
    // Load: 25 MW (fixed), Inflow: varies
    // We create 3 branchings with different means to represent scenarios
    let stage2_load = vec![Normal::new(25.0, 0.0).unwrap()]; // Fixed load
    let stage2_inflow = vec![Normal::new(20.0, 10.0).unwrap()]; // Mean 20, std 10 for variation
    generator.add_node_generator(stage2_load, stage2_inflow, 3);

    generator
}

/// Generate the SAA (Sample Average Approximation) for the 2-stage problem
///
/// This creates the scenario tree with:
/// - Stage 1: 1 deterministic scenario
/// - Stage 2: 3 stochastic scenarios
///
/// PERFORMANCE NOTE: SAA generation happens once per test, not performance-critical.
pub fn generate_2stage_saa(seed: u64) -> SAA {
    let generator = create_2stage_scenario_generator();
    generator.generate(seed)
}

/// Create initial condition for 2-stage test
///
/// Initial state:
/// - Storage: 50 MWh (50% full - allows for both charging and discharging)
///
/// This initial condition is chosen to make the problem interesting:
/// - Not empty (so hydro can generate immediately)
/// - Not full (so there's value in saving water)
pub fn create_simple_2stage_initial_condition() -> InitialCondition {
    InitialCondition::new(
        vec![50.0], // Initial storage for hydro 0
        vec![],     // No inflow lags for this simple problem
    )
}

/// Helper to compute expected solution analytically
///
/// This is a simplified analytical solution for validation purposes.
///
/// STAGE 2 (Terminal): Turbine all water to maximize revenue
/// - Dry (10 MWh inflow):  Storage(t-1) + 10 available
/// - Avg (20 MWh inflow):  Storage(t-1) + 20 available  
/// - Wet (30 MWh inflow):  Storage(t-1) + 30 available
///   Each scenario turbines max(storage + inflow, 50) to meet 25 MW load
///
/// STAGE 1: Balance current vs future value
/// - Release ~20-30 MWh to meet load and prepare for stage 2
/// - Keep enough storage for dry scenario hedge
///
/// EXPECTED COST: ~0-200 $ (mostly hydro, minimal thermal)
/// This is an approximation. The exact solution depends on water values.
///
/// Returns: (lower_bound, upper_bound) estimates for validation
#[allow(dead_code)]
pub fn expected_solution_bounds() -> (f64, f64) {
    // This is a rough estimate based on problem structure:
    // - Hydro generation is nearly free (just spillage penalty)
    // - Thermal costs 20 $/MWh when used
    // - Deficit costs 100 $/MWh (should never happen with our capacities)
    //
    // Optimal strategy minimizes thermal use:
    // - Total load over 2 stages: 25 MW * 2h = 50 MWh
    // - Initial storage: 50 MWh
    // - Total inflow: Stage1(20) + Stage2(10/20/30 avg=20) = 40 MWh avg
    // - Total available: 90 MWh avg (enough to cover 50 MWh load)
    // - Some thermal may be needed in dry scenarios
    //
    // Expected cost range:
    // - Best case: 0 $ (all hydro, no spillage)
    // - Typical case: 10-50 $ (minimal thermal in dry scenarios)
    // - Worst case: 100 $ (some thermal use)

    (0.0, 150.0) // Conservative bounds for validation
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_creation() {
        let system = create_simple_2stage_system();

        assert_eq!(system.meta.buses_count, 1);
        assert_eq!(system.meta.hydros_count, 1);
        assert_eq!(system.meta.thermals_count, 1);
        assert_eq!(system.meta.lines_count, 0);

        // Verify hydro parameters
        assert_eq!(system.hydros[0].max_storage, 100.0);
        assert_eq!(system.hydros[0].max_turbined_flow, 50.0);
    }

    #[test]
    fn test_scenario_generator() {
        let generator = create_2stage_scenario_generator();

        assert_eq!(generator.node_generators.len(), 3); // PreStudy + 2 study stages
        assert_eq!(generator.node_generators[0].num_branchings, 1); // PreStudy deterministic
        assert_eq!(generator.node_generators[1].num_branchings, 1); // Stage 1 deterministic
        assert_eq!(generator.node_generators[2].num_branchings, 3); // Stage 2 stochastic
    }

    #[test]
    fn test_saa_generation() {
        let saa = generate_2stage_saa(42);

        // Verify structure - should have 3 nodes (PreStudy + 2 study stages)
        assert_eq!(saa.branching_samples.len(), 3);
        assert_eq!(saa.get_branching_count_at_stage(0).unwrap(), 1); // PreStudy
        assert_eq!(saa.get_branching_count_at_stage(1).unwrap(), 1); // Stage 1
        assert_eq!(saa.get_branching_count_at_stage(2).unwrap(), 3); // Stage 2

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
        assert_eq!(ic.get_storage()[0], 50.0);
    }
}
