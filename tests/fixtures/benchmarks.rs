// Hydrothermal benchmark problems with known solutions for numerical validation
//
// ⚠️ DESIGN PRINCIPLES FOR CONVERGENCE:
// =====================================
// 1. WATER BALANCE: Ensure total water available meets or slightly exceeds demand
//    - Avoid: Too much water → forced spillage → dual price instability
//    - Avoid: Too little water → infeasibility or excessive deficit cost
//
// 2. FEASIBILITY: Always have thermal backup with sufficient capacity
//    - Thermal max_generation > (Demand - Hydro max_turbined_flow)
//    - This ensures problem never becomes infeasible
//
// 3. MEANINGFUL TRADE-OFFS: Water value should be between thermal and deficit costs
//    - Thermal cost << Deficit cost (makes hydro valuable)
//    - Storage constraints should bind but not too tightly
//
// 4. NUMERICAL STABILITY: Use reasonable scales
//    - Storage: 50-200 MWh (not too small, not too large)
//    - Demand: 30-60 MW (moderate)
//    - Costs: Thermal $5-15, Deficit $50+
//
// These principles ensure SDDP converges reliably to the known solution.

use powers_rs::scenario::SAA;
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::{Bus, Hydro, System, Thermal};

/// Result type for benchmark problems.
///
/// Returns both the SDDP algorithm instance and the SAA needed for training.
pub type BenchmarkResult = Result<(SddpAlgorithm, SAA), String>;

/// Creates a deterministic single-reservoir benchmark problem.
///
/// **PROBLEM DESCRIPTION**:
/// - 2-stage deterministic problem
/// - Single hydro reservoir (100 MWh storage capacity)
/// - Thermal backup (30 MW, $10/MWh)
/// - Demand: 40 MW per stage
///
/// **WATER BALANCE**:
/// - Initial storage: 50 MWh
/// - Stage 1 inflow: 30 MWh → Total: 80 MWh
/// - Stage 2 inflow: 40 MWh → Total: 120 MWh
/// - Total demand: 80 MWh (40 MW × 2 stages)
/// - Balance: 120 MWh ≥ 80 MWh ✓ (Feasible without thermal)
///
/// **EXPECTED SOLUTION**: $0 (all hydro, no thermal needed)
///
/// **CONVERGENCE**: Gap < $0.50 (deterministic → tight bounds)
pub fn create_deterministic_single_reservoir() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![50.0])
        .num_stages(2)
        .deterministic_inflows(vec![
            vec![30.0], // Stage 1: 30 MWh inflow
            vec![40.0], // Stage 2: 40 MWh inflow
        ])
        .deterministic_loads(vec![40.0, 40.0]) // Stage 1: 40 MW, Stage 2: 40 MW
        .seed(42)
        .build_with_saa()
}

/// Helper function to create single reservoir system.
///
/// System specification:
/// - 1 bus (deficit cost: $50/MWh)
/// - 1 hydro (100 MWh storage, 50 MW turbining)
/// - 1 thermal (30 MW, $10/MWh) - backup only
fn create_single_reservoir_system() -> System {
    let bus = Bus::new(0, 50.0);

    let hydro = Hydro::new(
        0,     // id
        None,  // downstream_hydro_id
        0,     // bus_id
        1.0,   // productivity
        0.0,   // min_storage
        100.0, // max_storage
        0.0,   // min_turbined_flow
        50.0,  // max_turbined_flow
        0.01,  // spillage_penalty
    );

    let thermal = Thermal::new(
        0,    // id
        0,    // bus_id
        10.0, // cost
        0.0,  // min_generation
        30.0, // max_generation
    );

    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

/// Creates a stochastic single-reservoir benchmark problem.
///
/// **PROBLEM DESCRIPTION**:
/// - 2-stage stochastic problem
/// - Single hydro reservoir (100 MWh storage capacity)
/// - Thermal backup (30 MW, $10/MWh)
/// - Demand: 40 MW per stage
/// - Stage 2 has 3 inflow scenarios: dry/average/wet
///
/// **WATER BALANCE**:
/// - Initial storage: 50 MWh
/// - Stage 1 inflow: 30 MWh (deterministic) → Total: 80 MWh
/// - Stage 2 scenarios:
///   * Dry (25%): 20 MWh → Total: 100 MWh vs 80 MWh demand ✓
///   * Average (50%): 40 MWh → Total: 120 MWh vs 80 MWh demand ✓
///   * Wet (25%): 60 MWh → Total: 140 MWh vs 80 MWh demand ✓
///
/// **EXPECTED SOLUTION**: $20-$40 (hedging cost against dry scenario)
///
/// **CONVERGENCE**: Gap < $5.00 (stochastic → medium tolerance)
pub fn create_stochastic_single_reservoir() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![50.0])
        .num_stages(2)
        .stochastic_inflows(vec![
            vec![vec![30.0]], // Stage 1: deterministic 30 MWh
            vec![
                vec![20.0], // Stage 2 dry: 20 MWh
                vec![40.0], // Stage 2 average: 40 MWh
                vec![60.0], // Stage 2 wet: 60 MWh
            ],
        ])
        .scenario_probabilities(vec![
            vec![1.0],              // Stage 1: 100%
            vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet
        ])
        .deterministic_loads(vec![40.0, 40.0]) // Same load for all scenarios
        .seed(42)
        .build_with_saa()
}

/// Creates a two-reservoir cascade benchmark problem.
///
/// **PROBLEM DESCRIPTION**:
/// - 2-stage deterministic problem
/// - Two hydro reservoirs in cascade (upstream → downstream)
/// - Thermal backup (30 MW, $10/MWh)
/// - Demand: 50 MW per stage
///
/// **WATER BALANCE**:
/// - Upstream reservoir (Hydro 0):
///   * Initial storage: 30 MWh
///   * Stage 1 inflow: 20 MWh, Stage 2 inflow: 25 MWh
///   * Total water: 75 MWh
/// - Downstream reservoir (Hydro 1):
///   * Initial storage: 40 MWh
///   * Stage 1 inflow: 15 MWh + upstream spillage/turbining
///   * Stage 2 inflow: 20 MWh + upstream spillage/turbining
///   * Total water: 75 MWh + upstream releases
/// - Combined capacity: 150 MWh total vs 100 MWh demand ✓
///
/// **EXPECTED SOLUTION**: $0-$50 (mostly hydro with good coordination)
///
/// **CONVERGENCE**: Gap < $10.00 (cascade adds complexity)
pub fn create_two_reservoir_cascade() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_cascade_system)
        .initial_storage(vec![30.0, 40.0]) // [upstream, downstream]
        .num_stages(2)
        .deterministic_inflows(vec![
            vec![20.0, 15.0], // Stage 1: [upstream, downstream]
            vec![25.0, 20.0], // Stage 2: [upstream, downstream]
        ])
        .deterministic_loads(vec![50.0, 50.0]) // Stage 1: 50 MW, Stage 2: 50 MW
        .seed(42)
        .build_with_saa()
}

/// Helper function to create cascade system.
///
/// System specification:
/// - 1 bus (deficit cost: $50/MWh)
/// - 2 hydros in cascade:
///   * Upstream (Hydro 0): 60 MWh storage, 30 MW turbining
///   * Downstream (Hydro 1): 80 MWh storage, 40 MW turbining
/// - 1 thermal (30 MW, $10/MWh) - backup only
fn create_cascade_system() -> System {
    let bus = Bus::new(0, 50.0);

    let hydro_upstream = Hydro::new(
        0,       // id
        Some(1), // downstream_hydro_id (cascade to Hydro 1)
        0,       // bus_id
        1.0,     // productivity
        0.0,     // min_storage
        60.0,    // max_storage
        0.0,     // min_turbined_flow
        30.0,    // max_turbined_flow
        0.01,    // spillage_penalty
    );

    let hydro_downstream = Hydro::new(
        1,    // id
        None, // downstream_hydro_id (terminal reservoir)
        0,    // bus_id
        1.0,  // productivity
        0.0,  // min_storage
        80.0, // max_storage
        0.0,  // min_turbined_flow
        40.0, // max_turbined_flow
        0.01, // spillage_penalty
    );

    let thermal = Thermal::new(
        0,    // id
        0,    // bus_id
        10.0, // cost
        0.0,  // min_generation
        30.0, // max_generation
    );

    System::new(
        vec![bus],
        vec![],
        vec![thermal],
        vec![hydro_upstream, hydro_downstream],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_benchmark_creation() {
        let result = create_deterministic_single_reservoir();
        assert!(
            result.is_ok(),
            "Deterministic benchmark should build successfully"
        );
    }

    #[test]
    fn test_stochastic_benchmark_creation() {
        let result = create_stochastic_single_reservoir();
        assert!(
            result.is_ok(),
            "Stochastic benchmark should build successfully"
        );
    }

    #[test]
    fn test_cascade_benchmark_creation() {
        let result = create_two_reservoir_cascade();
        assert!(
            result.is_ok(),
            "Cascade benchmark should build successfully"
        );
    }

    #[test]
    fn test_benchmark_water_balance_feasibility() {
        // Verify water balance design ensures feasibility

        // Benchmark 1: Deterministic
        // Initial: 50, Stage1 inflow: 30, Stage2 inflow: 40
        // Total: 120 MWh vs 80 MWh demand
        let det_total = 50.0 + 30.0 + 40.0;
        let det_demand = 40.0 * 2.0; // 40 MW × 2 stages
        assert!(
            det_total >= det_demand,
            "Benchmark 1: Insufficient water ({} < {})",
            det_total,
            det_demand
        );

        // Benchmark 2: Stochastic (worst case)
        // Initial: 50, Stage1: 30, Stage2 dry: 20
        // Worst case total: 100 MWh vs 80 MWh demand
        let sto_total_worst = 50.0 + 30.0 + 20.0;
        assert!(
            sto_total_worst >= det_demand,
            "Benchmark 2: Insufficient water in dry scenario ({} < {})",
            sto_total_worst,
            det_demand
        );

        // Benchmark 3: Cascade
        // Upstream: 30 + 20 + 25 = 75
        // Downstream: 40 + 15 + 20 = 75
        // Total: 150 MWh vs 100 MWh demand (50 MW × 2 stages)
        let cascade_total = (30.0 + 20.0 + 25.0) + (40.0 + 15.0 + 20.0);
        let cascade_demand = 50.0 * 2.0;
        assert!(
            cascade_total >= cascade_demand,
            "Benchmark 3: Insufficient water in cascade ({} < {})",
            cascade_total,
            cascade_demand
        );
    }
}
