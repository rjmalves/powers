// Debug test to trace inflow values through the system
use powers_rs::sddp::SddpAlgorithm;
use powers_rs::system::{Bus, Hydro, System, Thermal};

fn create_simple_system() -> System {
    let bus = Bus::new(0, 500.0);
    let hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 40.0, 0.01);
    let thermal = Thermal::new(0, 0, 50.0, 0.0, 25.0); // cost=50, min=0, max=25
    System::new(vec![bus], vec![], vec![thermal], vec![hydro])
}

#[test]
fn test_debug_inflow_values() {
    eprintln!("\n=== INFLOW DEBUG TEST ===");
    eprintln!("Expected inflows: Stage 0 = 15.0, Stage 1 = 25.0\n");

    // Create SDDP problem
    let (mut sddp, saa) = SddpAlgorithm::builder()
        .system_factory(create_simple_system)
        .initial_storage(vec![20.0])
        .num_stages(2)
        .deterministic_inflows(vec![vec![15.0], vec![25.0]])
        .deterministic_loads(vec![vec![50.0], vec![50.0]])
        .seed(42)
        .build_with_saa()
        .expect("Failed to build");

    eprintln!("\n=== Checking SAA ===");
    for stage in 0..3 {
        // Check prestudy (0) and study stages (1, 2)
        if let Some(noises) = saa.get_noises_by_stage_and_branching(stage, 0) {
            let innovations = noises.get_inflow_innovations();
            eprintln!("SAA stage {}: innovations = {:?}", stage, innovations);
        } else {
            eprintln!("SAA stage {}: no noises", stage);
        }
    }

    eprintln!("\n=== Training (1 iteration) ===");
    sddp.train(1, 1, false, &saa).expect("Training failed");

    eprintln!("\n=== Simulation ===");
    let trajectories = sddp.simulate(1, &saa).expect("Simulation failed");

    eprintln!("\n=== RESULTS ===");
    let trajectory = &trajectories[0];
    for stage_idx in 0..2 {
        let realization = &trajectory.realizations[stage_idx];
        let inflow = realization.inflow[0];
        eprintln!(
            "Stage {}: inflow = {:.6} (expected: {})",
            stage_idx,
            inflow,
            if stage_idx == 0 { 15.0 } else { 25.0 }
        );
    }

    eprintln!("\n=== VERDICT ===");
    let stage0_inflow = trajectory.realizations[0].inflow[0];
    let stage1_inflow = trajectory.realizations[1].inflow[0];

    if (stage0_inflow - 15.0).abs() < 0.1 && (stage1_inflow - 25.0).abs() < 0.1
    {
        eprintln!("✓ CORRECT: Inflows match expected values");
    } else {
        eprintln!("✗ BUG: Inflows don't match!");
        eprintln!("  Expected: 15.0, 25.0");
        eprintln!("  Got: {:.6}, {:.6}", stage0_inflow, stage1_inflow);
        panic!("Inflow values are incorrect!");
    }
}
