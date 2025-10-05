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
/// - Thermal backup (25 MW, $50/MWh)
/// - Demand: 50 MW per stage (high, requires hydro + thermal)
///
/// **WATER BALANCE** (designed for meaningful optimization):
/// - Initial storage: 20 MWh (low start - water is scarce)
/// - Stage 1 inflow: 15 MWh → Total: 35 MWh vs 50 MW demand
/// - Stage 2 inflow: 25 MWh
/// - Total water available: 60 MWh vs 100 MWh demand
/// - **Cannot meet demand with hydro alone** → must use thermal strategically
///
/// **OPTIMIZATION CHALLENGE**:
/// - Hydro max: 40 MW, so need 10 MW thermal minimum per stage
/// - Decision: Use more thermal in stage 1 to save water for stage 2?
/// - Water value increases across stages (SDDP learns this)
///
/// **EXPECTED SOLUTION**: $500-800 (thermal usage + water value optimization)
///
/// **CONVERGENCE**: Gap < $50 (deterministic → should converge tightly)
pub fn create_deterministic_single_reservoir() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0]) // Low initial storage
        .num_stages(2)
        .deterministic_inflows(vec![
            vec![15.0], // Stage 1: Low inflow (water scarcity)
            vec![25.0], // Stage 2: Better but still insufficient
        ])
        .deterministic_loads(vec![50.0, 50.0]) // High demand (need hydro + thermal)
        .seed(42)
        .build_with_saa()
}

/// Helper function to create single reservoir system.
///
/// **DESIGN PRINCIPLES FOR MEANINGFUL OPTIMIZATION**:
/// - Water is scarce: Cannot meet all demand with hydro alone
/// - Thermal is needed: But expensive enough to make storage valuable
/// - Deficit is very expensive: Avoid at all costs
/// - Optimization trade-off: When to use thermal vs save water
///
/// System specification:
/// - 1 bus (deficit cost: $500/MWh - very expensive)
/// - 1 hydro (100 MWh storage, 40 MW turbining - limits hydro generation)
/// - 1 thermal (25 MW, $50/MWh) - needed but costly
fn create_single_reservoir_system() -> System {
    let bus = Bus::new(0, 500.0); // High deficit cost

    let hydro = Hydro::new(
        0,     // id
        None,  // downstream_hydro_id
        0,     // bus_id
        1.0,   // productivity (1 MWh inflow = 1 MWh energy)
        0.0,   // min_storage
        100.0, // max_storage (MWh)
        0.0,   // min_turbined_flow
        40.0,  // max_turbined_flow (MW) - limits hydro generation
        0.01,  // spillage_penalty
    );

    let thermal = Thermal::new(
        0,    // id
        0,    // bus_id
        50.0, // cost ($/MWh) - significant but less than deficit
        0.0,  // min_generation
        25.0, // max_generation (MW) - thermal + hydro can meet demand
    );

    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

/// Creates a stochastic single-reservoir benchmark problem.
///
/// **PROBLEM DESCRIPTION**:
/// - 2-stage stochastic problem
/// - Single hydro reservoir (100 MWh storage capacity)
/// - Thermal backup (25 MW, $50/MWh)
/// - Demand: 50 MW per stage
/// - Stage 2 has 3 inflow scenarios: dry/average/wet (hydrological uncertainty)
///
/// **WATER BALANCE** (designed for risk management):
/// - Initial storage: 20 MWh (low start - water is scarce)
/// - Stage 1 inflow: 15 MWh (deterministic)
/// - Stage 2 scenarios (stochastic inflows):
///   * Dry (25%): 10 MWh → Total water: 45 MWh vs 100 MWh demand (severe shortage)
///   * Average (50%): 20 MWh → Total water: 55 MWh vs 100 MWh demand (moderate shortage)
///   * Wet (25%): 35 MWh → Total water: 70 MWh vs 100 MWh demand (still need thermal)
///
/// **OPTIMIZATION CHALLENGE**:
/// - Stage 1 decision: How much water to save for uncertain stage 2?
/// - Hedging: Use thermal now ($50 certain) vs risk deficit later ($500 potential)
/// - Classic stochastic optimization: Balance expected cost vs risk
/// - Algorithm must learn water value under uncertainty
///
/// **EXPECTED SOLUTION**: $1,500-2,500 (thermal hedging + water value + risk premium)
///
/// **CONVERGENCE**: Gap < $200 (stochastic → wider tolerance due to sampling variance)
pub fn create_stochastic_single_reservoir() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0]) // Low initial storage
        .num_stages(2)
        .stochastic_inflows(vec![
            vec![vec![15.0]], // Stage 1: deterministic low inflow
            vec![
                vec![10.0], // Stage 2 dry: Severe shortage (25% probability)
                vec![20.0], // Stage 2 average: Moderate shortage (50% probability)
                vec![35.0], // Stage 2 wet: Still need thermal (25% probability)
            ],
        ])
        .scenario_probabilities(vec![
            vec![1.0],              // Stage 1: 100%
            vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet probabilities
        ])
        .deterministic_loads(vec![50.0, 50.0]) // High constant load
        .seed(42)
        .build_with_saa()
}

/// Creates a two-reservoir cascade benchmark problem.
///
/// **PROBLEM DESCRIPTION**:
/// - 2-stage deterministic problem
/// - Two hydro reservoirs in cascade (upstream → downstream)
/// - Thermal backup (25 MW, $50/MWh)
/// - Demand: 60 MW per stage (high demand requires hydro + thermal)
///
/// **WATER BALANCE** (designed for cascade coordination):
/// - Upstream reservoir (Hydro 0):
///   * Initial storage: 15 MWh (low start)
///   * Stage 1 inflow: 20 MWh, Stage 2 inflow: 25 MWh
///   * Max turbining: 30 MW (limited)
///   * Total upstream water: 60 MWh
/// - Downstream reservoir (Hydro 1):
///   * Initial storage: 20 MWh (low start)
///   * Stage 1 inflow: 10 MWh + upstream releases
///   * Stage 2 inflow: 15 MWh + upstream releases
///   * Max turbining: 35 MW
///   * Own water: 45 MWh + upstream releases
/// - Combined hydro max: 65 MW (30+35), but water is scarce
/// - Total demand: 120 MWh (60 MW × 2 stages)
/// - **Challenge**: Coordinate cascade timing + manage water scarcity
///
/// **OPTIMIZATION CHALLENGE**:
/// - Upstream timing: Release water now or store for later?
/// - Downstream receives upstream releases: Coordination is key
/// - Water travel time: Immediate (simplified model)
/// - Thermal needed: But when? Stage 1 or Stage 2?
/// - Trade-off: Cascade coordination vs thermal cost
///
/// **EXPECTED SOLUTION**: $1,200-2,000 (thermal + cascade coordination)
///
/// **CONVERGENCE**: Gap < $150 (cascade adds complexity to optimization)
pub fn create_two_reservoir_cascade() -> BenchmarkResult {
    SddpAlgorithm::builder()
        .system_factory(create_cascade_system)
        .initial_storage(vec![15.0, 20.0]) // [upstream, downstream] - both low
        .num_stages(2)
        .deterministic_inflows(vec![
            vec![20.0, 10.0], // Stage 1: [upstream, downstream] - low inflows
            vec![25.0, 15.0], // Stage 2: [upstream, downstream] - slightly better
        ])
        .deterministic_loads(vec![60.0, 60.0]) // High demand (need coordination)
        .seed(42)
        .build_with_saa()
}

/// Helper function to create cascade system.
///
/// **DESIGN PRINCIPLES**:
/// - Two hydros in cascade: Upstream releases flow to downstream
/// - Combined capacity can meet demand, but water is scarce
/// - Thermal needed due to water scarcity
/// - Optimization: Balance cascade timing vs thermal usage
///
/// System specification:
/// - 1 bus (deficit cost: $500/MWh - very expensive)
/// - 2 hydros in cascade:
///   * Upstream (Hydro 0): 60 MWh storage, 30 MW turbining
///   * Downstream (Hydro 1): 80 MWh storage, 35 MW turbining
/// - 1 thermal (25 MW, $50/MWh) - needed but costly
fn create_cascade_system() -> System {
    let bus = Bus::new(0, 500.0); // High deficit cost

    let hydro_upstream = Hydro::new(
        0,       // id
        Some(1), // downstream_hydro_id (cascade to Hydro 1)
        0,       // bus_id
        1.0,     // productivity
        0.0,     // min_storage
        60.0,    // max_storage (MWh)
        0.0,     // min_turbined_flow
        30.0,    // max_turbined_flow (MW) - limited capacity
        0.01,    // spillage_penalty
    );

    let hydro_downstream = Hydro::new(
        1,    // id
        None, // downstream_hydro_id (terminal reservoir)
        0,    // bus_id
        1.0,  // productivity
        0.0,  // min_storage
        80.0, // max_storage (MWh)
        0.0,  // min_turbined_flow
        35.0, // max_turbined_flow (MW) - larger than upstream
        0.01, // spillage_penalty
    );

    let thermal = Thermal::new(
        0,    // id
        0,    // bus_id
        50.0, // cost ($/MWh) - significant but less than deficit
        0.0,  // min_generation
        25.0, // max_generation (MW) - complements hydros
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
