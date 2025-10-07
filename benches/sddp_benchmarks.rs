//! Performance benchmarks for SDDP algorithm components
//!
//! This benchmark suite measures the performance of critical operations
//! to detect performance regressions and track improvements over time.
//!
//! Run with: `cargo bench --bench sddp_benchmarks`
//! View results: `open target/criterion/report/index.html`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion,
};
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::{Bus, Hydro, System, Thermal};

/// Create a minimal single-reservoir system for benchmarking
///
/// **DESIGN PRINCIPLES FOR MEANINGFUL OPTIMIZATION**:
/// - Water is scarce: Cannot meet all demand with hydro alone
/// - Storage decisions matter: Must learn water value through SDDP
/// - Thermal is attractive: Cheaper than deficit but more expensive than hydro
/// - Non-trivial solution: Forces algorithm to balance hydro vs thermal usage
///
/// System specification:
/// - Bus deficit cost: $500/MWh (expensive, avoid at all costs)
/// - Hydro: 100 MWh storage, 100 MW max turbining, productivity 1.0
/// - Thermal: 100 MW capacity, $50/MWh (10x cheaper than deficit, but costly)
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
        100.0, // max_turbined_flow (MW) - limits hydro generation
        0.01,  // spillage_penalty (small, avoid waste)
    );

    let thermal = Thermal::new(
        0,     // id
        0,     // bus_id
        50.0,  // cost ($/MWh) - significant but less than deficit
        0.0,   // min_generation
        100.0, // max_generation (MW) - thermal + hydro can meet demand
    );

    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

/// Create a 2-stage deterministic problem with meaningful water value trade-offs
///
/// **WATER BALANCE DESIGN**:
/// - Initial storage: 20 MWh (starting low)
/// - Stage 1: 15 MWh inflow → 35 MWh available vs 50 MW demand
/// - Stage 2: 25 MWh inflow → need to save water from stage 1
/// - Total water: 60 MWh vs 100 MWh total demand
/// - **Cannot meet demand with hydro alone** → must use thermal
///
/// **OPTIMIZATION CHALLENGE**:
/// - Stage 1: Use thermal to save water for stage 2? Or deplete now?
/// - Stage 2: Will saved water + inflow be enough?
/// - **Key insight**: Thermal cost ($50) vs future water value
///
/// **EXPECTED BEHAVIOR**:
/// - Algorithm learns water is valuable in later stages
/// - Optimal policy: Use thermal in stage 1, save water for stage 2
/// - Non-trivial solution with thermal usage: Cost ~$500-800
fn create_2stage_problem() -> (SddpAlgorithm, powers_rs::scenario::SAA) {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0]) // Low initial storage
        .num_stages(2)
        .deterministic_inflows(vec![
            vec![15.0], // Stage 1: Low inflow (scarcity)
            vec![25.0], // Stage 2: Better inflow but not enough
        ])
        .deterministic_loads(vec![50.0, 50.0]) // High demand (need hydro + thermal)
        .seed(42)
        .build_with_saa()
        .expect("Failed to create 2-stage problem")
}

/// Create a 12-stage deterministic problem with seasonal pattern
///
/// **WATER BALANCE DESIGN**:
/// - Initial storage: 30 MWh (moderate start)
/// - Inflows: Seasonal pattern (15 MWh early, 25 MWh late) - avg 18.3 MWh/stage
/// - Load: 50 MW per stage (constant high demand)
/// - Total water: 30 + 12×18.3 = 250 MWh vs 600 MWh demand
/// - **Severe water shortage** → significant thermal usage required
///
/// **OPTIMIZATION CHALLENGE**:
/// - Early stages: Low inflows (15 MWh) vs 50 MW demand → need thermal
/// - Late stages: Better inflows (25 MWh) but still insufficient
/// - Must learn to manage storage across 12 stages
/// - Storage decisions have long-term consequences
///
/// **EXPECTED BEHAVIOR**:
/// - Heavy thermal usage throughout (water scarcity)
/// - Water value increases in later stages (steeper cuts)
/// - Cost: ~$10,000-15,000 (significant thermal generation)
fn create_12stage_problem() -> (SddpAlgorithm, powers_rs::scenario::SAA) {
    // Seasonal inflow pattern: dry early, wetter late
    let inflows: Vec<Vec<f64>> = vec![
        vec![15.0],
        vec![15.0],
        vec![15.0],
        vec![15.0], // Dry season
        vec![18.0],
        vec![18.0],
        vec![20.0],
        vec![20.0], // Transition
        vec![25.0],
        vec![25.0],
        vec![25.0],
        vec![25.0], // Wet season
    ];
    let loads: Vec<f64> = (0..12).map(|_| 50.0).collect(); // Constant high load

    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![30.0]) // Moderate initial storage
        .num_stages(12)
        .deterministic_inflows(inflows)
        .deterministic_loads(loads)
        .seed(42)
        .build_with_saa()
        .expect("Failed to create 12-stage problem")
}

/// Create a 2-stage stochastic problem with uncertainty in inflows
///
/// **WATER BALANCE DESIGN**:
/// - Initial storage: 20 MWh (low start)
/// - Stage 1: 15 MWh deterministic inflow
/// - Stage 2: Stochastic inflows modeling hydrological uncertainty
///   * Dry scenario (25%): 10 MWh - severe water shortage
///   * Average scenario (50%): 20 MWh - moderate shortage
///   * Wet scenario (25%): 35 MWh - adequate water
/// - Load: 50 MW per stage
/// - Total water varies: 45-70 MWh vs 100 MWh demand
///
/// **OPTIMIZATION CHALLENGE**:
/// - Stage 1 decision: How much water to save for stage 2?
/// - Uncertainty: Don't know if stage 2 will be dry/average/wet
/// - Trade-off: Use thermal now (certain $50) vs risk deficit later (potential $500)
/// - **Classic risk management problem**
///
/// **EXPECTED BEHAVIOR**:
/// - Algorithm learns to hedge against dry scenario
/// - Conservative storage policy (save water)
/// - Increased thermal usage in stage 1
/// - Cost: ~$1,500-2,500 (hedging premium + thermal)
fn create_stochastic_problem() -> (SddpAlgorithm, powers_rs::scenario::SAA) {
    SddpAlgorithm::builder()
        .system_factory(create_single_reservoir_system)
        .initial_storage(vec![20.0]) // Low initial storage
        .num_stages(2)
        .stochastic_inflows(vec![
            vec![vec![15.0]], // Stage 1: deterministic low inflow
            vec![
                vec![10.0], // Stage 2 dry: Severe shortage
                vec![20.0], // Stage 2 average: Moderate shortage
                vec![35.0], // Stage 2 wet: Adequate water
            ],
        ])
        .scenario_probabilities(vec![
            vec![1.0],              // Stage 1: 100%
            vec![0.25, 0.50, 0.25], // Stage 2: dry/avg/wet
        ])
        .deterministic_loads(vec![50.0, 50.0]) // High constant load
        .seed(42)
        .build_with_saa()
        .expect("Failed to create stochastic problem")
}

// ============================================================================
// BENCHMARK 1: Full Training Iteration
// ============================================================================

/// Benchmark a single full iteration of SDDP training
fn bench_full_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_iteration");
    group.sample_size(50); // Medium samples for moderate duration operations
    group.measurement_time(std::time::Duration::from_secs(10));

    // 2-stage deterministic (fast baseline)
    group.bench_function("2_stage_deterministic", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_2stage_problem();
            black_box(sddp.train(1, 5, &saa).expect("Training failed"));
        });
    });

    // 2-stage stochastic (more complex)
    group.bench_function("2_stage_stochastic", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_stochastic_problem();
            black_box(sddp.train(1, 5, &saa).expect("Training failed"));
        });
    });

    // 12-stage deterministic (real-world size)
    group.bench_function("12_stage_deterministic", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_12stage_problem();
            black_box(sddp.train(1, 5, &saa).expect("Training failed"));
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 2: Multi-Iteration Convergence
// ============================================================================

/// Benchmark convergence behavior over multiple iterations
fn bench_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence");
    group.sample_size(20); // Fewer samples for long operations
    group.measurement_time(std::time::Duration::from_secs(15));

    // 2-stage: Should converge quickly
    group.bench_function("2_stage_10iter", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_2stage_problem();
            black_box(sddp.train(10, 10, &saa).expect("Training failed"));
        });
    });

    // 2-stage stochastic: Slower convergence
    group.bench_function("2_stage_stochastic_20iter", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_stochastic_problem();
            black_box(sddp.train(20, 10, &saa).expect("Training failed"));
        });
    });

    // 12-stage: Real-world convergence
    group.bench_function("12_stage_20iter", |b| {
        b.iter(|| {
            let (mut sddp, saa) = create_12stage_problem();
            black_box(sddp.train(20, 10, &saa).expect("Training failed"));
        });
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Simulation
// ============================================================================

/// Benchmark policy simulation (out-of-sample evaluation)
fn bench_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulation");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    // Simulate with trained policy
    group.bench_with_input(
        BenchmarkId::new("2_stage", "100_scenarios"),
        &100,
        |b, &num_scenarios| {
            let (mut sddp, saa) = create_2stage_problem();
            sddp.train(10, 5, &saa).expect("Training failed");

            b.iter(|| {
                black_box(
                    sddp.simulate(num_scenarios, &saa)
                        .expect("Simulation failed"),
                );
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("12_stage", "100_scenarios"),
        &100,
        |b, &num_scenarios| {
            let (mut sddp, saa) = create_12stage_problem();
            sddp.train(10, 5, &saa).expect("Training failed");

            b.iter(|| {
                black_box(
                    sddp.simulate(num_scenarios, &saa)
                        .expect("Simulation failed"),
                );
            });
        },
    );

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Problem Scaling
// ============================================================================

/// Benchmark how performance scales with problem size
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(10));

    // Scale by number of stages
    for num_stages in [2, 6, 12].iter() {
        group.bench_with_input(
            BenchmarkId::new("stages", num_stages),
            num_stages,
            |b, &stages| {
                let inflows: Vec<Vec<f64>> =
                    (0..stages).map(|_| vec![30.0]).collect();
                let loads: Vec<f64> = (0..stages).map(|_| 40.0).collect();

                b.iter(|| {
                    let (mut sddp, saa) = SddpAlgorithm::builder()
                        .system_factory(create_single_reservoir_system)
                        .initial_storage(vec![50.0])
                        .num_stages(stages)
                        .deterministic_inflows(inflows.clone())
                        .deterministic_loads(loads.clone())
                        .seed(42)
                        .build_with_saa()
                        .expect("Failed to create problem");

                    black_box(sddp.train(5, 5, &saa).expect("Training failed"));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_full_iteration,
    bench_convergence,
    bench_simulation,
    bench_scaling
);
criterion_main!(benches);
