//! Benchmark to measure allocation overhead in backward pass
//! 
//! This benchmark measures the baseline allocation behavior before TICKET-006b
//! optimization. We'll compare before/after to validate the optimization impact.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use powers_rs::sddp::SDDP;
use std::sync::Arc;

fn create_small_problem() -> (
    powers_rs::system::System,
    powers_rs::graph::DirectedGraph<powers_rs::sddp::NodeData>,
    powers_rs::input::Config,
) {
    // Create minimal problem for benchmarking
    let system = powers_rs::system::System::new(
        vec![powers_rs::system::Bus::new(0, 1000.0)],
        vec![],
        vec![],
        vec![
            powers_rs::system::Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0),
            powers_rs::system::Hydro::new(1, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0),
            powers_rs::system::Hydro::new(2, None, 0, 1.0, 0.0, 100.0, 0.0, 100.0, 10.0),
        ],
    );

    let mut graph = powers_rs::graph::DirectedGraph::new();
    
    // Create 3 stages
    for stage_id in 0..3 {
        let node_system = system.clone();
        let node = powers_rs::sddp::NodeData {
            id: stage_id,
            stage_id,
            season_id: 0,
            start_date: chrono::Utc::now(),
            end_date: chrono::Utc::now(),
            kind: powers_rs::subproblem::StudyPeriodKind::PreStudy,
            system: node_system,
            risk_measure: Box::new(powers_rs::risk_measure::Expectation::new()),
            uncertainty_models: Arc::new(vec![]),
            state_choice: "storage".to_string(),
            num_scenarios: 4,
        };
        graph.add_node(node).unwrap();
        
        if stage_id > 0 {
            graph.add_edge(stage_id - 1, stage_id, 1.0).unwrap();
        }
    }

    let config = powers_rs::input::Config {
        general: powers_rs::input::GeneralConfig {
            seed: 42,
            num_threads: Some(1),  // Single thread for consistent measurement
        },
        training: powers_rs::input::TrainingConfig {
            num_iterations: 2,
            num_forward_passes: 4,
            enable_cut_selection: false,
        },
        simulation: powers_rs::input::SimulationConfig {
            num_scenarios: Some(10),
        },
        output: powers_rs::input::OutputConfig::default(),
        logging: powers_rs::logging::LoggingConfig::default(),
    };

    (system, graph, config)
}

fn benchmark_backward_pass_baseline(c: &mut Criterion) {
    let (system, graph, config) = create_small_problem();
    
    c.bench_function("backward_pass_baseline_3hydros_4fps", |b| {
        b.iter(|| {
            let mut sddp = SDDP::new(
                black_box(system.clone()),
                black_box(graph.clone()),
                black_box(config.clone()),
                black_box(powers_rs::scenario::ScenarioTree::new()),
            ).unwrap();
            
            // Run training (includes backward passes)
            let _result = sddp.train();
        });
    });
}

criterion_group!(benches, benchmark_backward_pass_baseline);
criterion_main!(benches);
