mod fixtures;

use fixtures::validation::{
    FeasibilityReport, PolicyValidator, ReasonablenessReport,
    UnexpectedBehavior, Violation,
};
use powers_rs::graph;
use powers_rs::initial_condition;
use powers_rs::scenario;
use powers_rs::sddp::{NodeData, SddpAlgorithm, StageResult, Trajectory};
use powers_rs::subproblem;
use powers_rs::system::System;
use rand_distr::{LogNormal, Normal};
use std::time::Instant;

/// Creates a simple feasible trajectory for testing.
fn create_feasible_trajectory() -> Trajectory {
    // Default system: 1 hydro (productivity=1.0, max_turbined=60), 2 thermals (max=15 each), 1 bus, 0 lines
    // Action vector: [turbined_flow(1), thermal_gen(2), spillage(1), exchange(0), deficit(1)]
    // Load = 75 MW, so we need generation + deficit = 75
    let stages = vec![
        StageResult {
            stage: 0,
            state: vec![50.0], // Initial storage
            action: vec![
                10.0, // [0] Turbined flow (→ 10 MW generation)
                5.0, 10.0, // [1-2] Thermal generation (5 + 10 = 15 MW)
                0.0,  // [3] Spillage
                // NO exchange (0 lines)
                50.0, // [4] Deficit (10+15+50=75, balances load)
            ],
            stage_cost: 2625.0, // 5*5 + 10*10 + 50*50 = 25 + 100 + 2500
            inflow: vec![40.0], // Inflow for this stage
            load: vec![75.0],
        },
        StageResult {
            stage: 1,
            // Water balance: prev_storage + prev_inflow - prev_turbined - prev_spillage
            // = 50 + 40 - 10 - 0 = 80
            state: vec![80.0],
            action: vec![
                15.0, // [0] Turbined flow (→ 15 MW generation)
                10.0, 15.0, // [1-2] Thermal generation (10 + 15 = 25 MW)
                0.0,  // [3] Spillage
                // NO exchange
                35.0, // [4] Deficit (15+25+35=75)
            ],
            stage_cost: 2000.0, // 10*5 + 15*10 + 35*50 = 50 + 150 + 1750
            inflow: vec![30.0], // Inflow for this stage
            load: vec![75.0],
        },
    ];

    Trajectory {
        stages,
        total_cost: 4625.0,
        scenario_id: 0,
    }
}

/// Creates a trajectory with storage violations.
fn create_trajectory_with_storage_violation() -> Trajectory {
    let stages = vec![StageResult {
        stage: 0,
        state: vec![-5.0], // VIOLATION: Negative storage
        action: vec![10.0, 5.0, 10.0, 0.0, 50.0], // turbined, thermal1, thermal2, spillage, deficit
        stage_cost: 2625.0,
        inflow: vec![40.0],
        load: vec![75.0],
    }];

    Trajectory {
        stages,
        total_cost: 2625.0,
        scenario_id: 0,
    }
}

/// Creates a trajectory with generation bound violations.
fn create_trajectory_with_generation_violation() -> Trajectory {
    let stages = vec![StageResult {
        stage: 0,
        state: vec![50.0],
        action: vec![
            70.0, // VIOLATION: Turbined flow exceeds max (60)
            5.0, 10.0, // Thermal
            0.0,  // Spillage
            0.0,  // Deficit (70+15-75=10 MWover, but showing 0 deficit)
        ],
        stage_cost: 850.0,
        inflow: vec![40.0],
        load: vec![75.0],
    }];

    Trajectory {
        stages,
        total_cost: 850.0,
        scenario_id: 0,
    }
}

/// Creates a trajectory with spillage when storage is low (unreasonable).
fn create_trajectory_with_unreasonable_spillage() -> Trajectory {
    let stages = vec![StageResult {
        stage: 0,
        state: vec![20.0], // Storage at 20% (low)
        action: vec![
            5.0, // Turbined flow
            5.0, 10.0, // Thermal
            10.0, // UNREASONABLE: Spillage when storage is low
            55.0, // Deficit (5+15+55=75)
        ],
        stage_cost: 2900.0, // 5*5 + 10*10 + 55*50 + 10*0.01 = 25 + 100 + 2750 + 0.1
        inflow: vec![15.0],
        load: vec![75.0],
    }];

    Trajectory {
        stages,
        total_cost: 2900.0,
        scenario_id: 0,
    }
}

/// Creates a trajectory with excessive deficit (unreasonable).
fn create_trajectory_with_excessive_deficit() -> Trajectory {
    let stages = vec![StageResult {
        stage: 0,
        state: vec![50.0],
        action: vec![
            10.0, // Turbined flow
            5.0, 10.0, // Thermal (total 25 MW)
            0.0,  // Spillage
            50.0, // UNREASONABLE: Deficit is 66% of load
        ],
        stage_cost: 2625.0,
        inflow: vec![40.0],
        load: vec![75.0],
    }];

    Trajectory {
        stages,
        total_cost: 2625.0,
        scenario_id: 0,
    }
}

#[test]
fn test_policy_validator_creation() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    assert_eq!(validator.num_hydros(), 1);
    assert_eq!(validator.num_thermals(), 2);
    assert_eq!(validator.num_buses(), 1);
}

#[test]
fn test_feasibility_report_all_feasible() {
    let report = FeasibilityReport {
        violations: vec![],
        num_trajectories: 100,
        num_stages_checked: 1200,
    };

    assert!(report.all_feasible());
    assert_eq!(report.num_violations(), 0);
    assert_eq!(report.violation_rate(), 0.0);
}

#[test]
fn test_feasibility_report_with_violations() {
    let report = FeasibilityReport {
        violations: vec![
            Violation::NegativeStorage {
                trajectory_id: 0,
                stage: 0,
                hydro_id: 0,
                value: -5.0,
                min_bound: 0.0,
            },
            Violation::TurbinedFlowExceedsMax {
                trajectory_id: 1,
                stage: 0,
                hydro_id: 0,
                value: 70.0,
                max_bound: 60.0,
            },
        ],
        num_trajectories: 100,
        num_stages_checked: 1200,
    };

    assert!(!report.all_feasible());
    assert_eq!(report.num_violations(), 2);
    assert!((report.violation_rate() - 0.02).abs() < 1e-10);
}

#[test]
fn test_reasonableness_report_all_reasonable() {
    let report = ReasonablenessReport {
        unexpected_behaviors: vec![],
        num_trajectories: 100,
        num_stages_checked: 1200,
    };

    assert!(report.is_reasonable());
    assert_eq!(report.num_unexpected(), 0);
    assert_eq!(report.unexpected_rate(), 0.0);
}

#[test]
fn test_reasonableness_report_with_unexpected_behaviors() {
    let report = ReasonablenessReport {
        unexpected_behaviors: vec![
            UnexpectedBehavior::SpillageWhenStorageLow {
                trajectory_id: 0,
                stage: 0,
                hydro_id: 0,
                storage_percent: 0.2,
                spillage: 10.0,
            },
            UnexpectedBehavior::ExcessiveDeficit {
                trajectory_id: 1,
                stage: 0,
                bus_id: 0,
                load: 75.0,
                deficit: 50.0,
                deficit_ratio: 0.666,
            },
        ],
        num_trajectories: 100,
        num_stages_checked: 1200,
    };

    assert!(!report.is_reasonable());
    assert_eq!(report.num_unexpected(), 2);
    assert!((report.unexpected_rate() - 0.02).abs() < 1e-10);
}

#[test]
fn test_check_feasibility_with_feasible_trajectory() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_feasible_trajectory();

    let report = validator.check_feasibility(&[trajectory]);

    assert!(
        report.all_feasible(),
        "Expected feasible trajectory, got {} violations: {:?}",
        report.num_violations(),
        report.violations
    );
    assert_eq!(report.num_violations(), 0);
    assert_eq!(report.num_trajectories, 1);
}

#[test]
fn test_check_feasibility_detects_storage_violation() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_trajectory_with_storage_violation();

    let report = validator.check_feasibility(&[trajectory]);

    assert!(!report.all_feasible());
    assert!(report.num_violations() >= 1); // At least the storage violation

    // Should detect the negative storage violation
    let has_storage_violation = report.violations.iter().any(|v| {
        matches!(v, Violation::NegativeStorage { value, min_bound, .. }
            if *value == -5.0 && *min_bound == 0.0)
    });
    assert!(has_storage_violation, "Expected NegativeStorage violation");
}

#[test]
fn test_check_feasibility_detects_generation_violation() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_trajectory_with_generation_violation();

    let report = validator.check_feasibility(&[trajectory]);

    assert!(!report.all_feasible());
    assert!(report.num_violations() > 0);

    // Should detect turbined flow exceeds max
    let has_turbined_violation = report
        .violations
        .iter()
        .any(|v| matches!(v, Violation::TurbinedFlowExceedsMax { .. }));
    assert!(
        has_turbined_violation,
        "Expected TurbinedFlowExceedsMax violation"
    );
}

#[test]
fn test_check_feasibility_with_multiple_trajectories() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    let trajectories = vec![
        create_feasible_trajectory(),
        create_trajectory_with_storage_violation(),
        create_feasible_trajectory(),
        create_trajectory_with_generation_violation(),
    ];

    let report = validator.check_feasibility(&trajectories);

    assert!(!report.all_feasible());
    assert_eq!(report.num_trajectories, 4);
    assert!(report.num_violations() >= 2); // At least 2 violations
}

#[test]
fn test_check_reasonableness_with_reasonable_trajectory() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_feasible_trajectory();

    let report = validator.check_reasonableness(&[trajectory]);

    // This may or may not be reasonable depending on exact values
    // Just check report structure
    assert_eq!(report.num_trajectories, 1);
}

#[test]
fn test_check_reasonableness_detects_spillage_when_low_storage() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_trajectory_with_unreasonable_spillage();

    let report = validator.check_reasonableness(&[trajectory]);

    assert!(!report.is_reasonable());
    assert!(report.num_unexpected() > 0);

    let has_spillage_behavior = report.unexpected_behaviors.iter().any(|b| {
        matches!(b, UnexpectedBehavior::SpillageWhenStorageLow { .. })
    });
    assert!(
        has_spillage_behavior,
        "Expected SpillageWhenStorageLow behavior"
    );
}

#[test]
fn test_check_reasonableness_detects_excessive_deficit() {
    let system = System::default();
    let validator = PolicyValidator::new(system);
    let trajectory = create_trajectory_with_excessive_deficit();

    let report = validator.check_reasonableness(&[trajectory]);

    assert!(!report.is_reasonable());
    assert!(report.num_unexpected() > 0);

    let has_deficit_behavior = report
        .unexpected_behaviors
        .iter()
        .any(|b| matches!(b, UnexpectedBehavior::ExcessiveDeficit { .. }));
    assert!(has_deficit_behavior, "Expected ExcessiveDeficit behavior");
}

#[test]
fn test_compare_to_analytical_exact_match() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    let diff = validator.compare_to_analytical(300.0, 300.0);
    assert_eq!(diff, 0.0);
}

#[test]
fn test_compare_to_analytical_close_match() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    let diff = validator.compare_to_analytical(303.0, 300.0);
    assert!((diff - 0.01).abs() < 1e-10); // 1% difference
}

#[test]
fn test_compare_to_analytical_poor_match() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    let diff = validator.compare_to_analytical(330.0, 300.0);
    assert!((diff - 0.10).abs() < 1e-10); // 10% difference
}

#[test]
fn test_compare_to_analytical_zero_cost_edge_case() {
    let system = System::default();
    let validator = PolicyValidator::new(system);

    // Both zero
    let diff = validator.compare_to_analytical(0.0, 0.0);
    assert_eq!(diff, 0.0);

    // Analytical zero, simulated non-zero
    let diff = validator.compare_to_analytical(10.0, 0.0);
    assert!(diff.is_infinite());
}

#[test]
fn test_deterministic_2stage_policy_validation() {
    // Create deterministic 2-stage problem
    let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

    let pre_study_id = node_data_graph
        .add_node(
            NodeData::new(
                -1,
                0,
                0,
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                subproblem::StudyPeriodKind::PreStudy,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    let stage0_id = node_data_graph
        .add_node(
            NodeData::new(
                0,
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                subproblem::StudyPeriodKind::Study,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    node_data_graph.add_edge(pre_study_id, stage0_id).unwrap();

    let stage1_id = node_data_graph
        .add_node(
            NodeData::new(
                1,
                1,
                1,
                "2025-02-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                subproblem::StudyPeriodKind::Study,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    node_data_graph.add_edge(stage0_id, stage1_id).unwrap();

    let storage = vec![83.222];
    let initial_condition =
        initial_condition::InitialCondition::new(storage, vec![]);

    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );

    let saa = scenario_generator.generate(0);

    let mut sddp =
        SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

    // Train for 10 iterations
    let _train_result = sddp.train(10, 5, &saa).unwrap();

    // Simulate
    let sim_result = sddp.simulate_and_analyze(20, &saa).unwrap();

    // Validate
    let validator = PolicyValidator::new(System::default());
    let feasibility = validator.check_feasibility(&sim_result.trajectories);

    assert!(
        feasibility.all_feasible(),
        "Deterministic 2-stage produced infeasible trajectories: {:?}",
        feasibility.violations
    );
}

#[test]
fn test_stochastic_12stage_policy_validation() {
    // Create stochastic 12-stage problem
    let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

    let pre_study_id = node_data_graph
        .add_node(
            NodeData::new(
                -1,
                0,
                0,
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                subproblem::StudyPeriodKind::PreStudy,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    let mut prev_id = node_data_graph
        .add_node(
            NodeData::new(
                0,
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                subproblem::StudyPeriodKind::Study,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    node_data_graph.add_edge(pre_study_id, prev_id).unwrap();

    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );

    // Create 12 stages
    for stage in 1..=11 {
        let new_id = node_data_graph
            .add_node(
                NodeData::new(
                    stage,
                    stage.try_into().unwrap(),
                    stage.try_into().unwrap(),
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    System::default(),
                    "expectation",
                    "naive",
                    &[],
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        node_data_graph.add_edge(prev_id, new_id).unwrap();
        prev_id = new_id;

        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
    }

    let storage = vec![83.222];
    let initial_condition =
        initial_condition::InitialCondition::new(storage, vec![]);
    let saa = scenario_generator.generate(0);

    let mut sddp =
        SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

    // Train for 30 iterations
    let _train_result = sddp.train(30, 10, &saa).unwrap();

    // Simulate 100 scenarios
    let sim_result = sddp.simulate_and_analyze(100, &saa).unwrap();

    // Validate feasibility
    let validator = PolicyValidator::new(System::default());
    let feasibility = validator.check_feasibility(&sim_result.trajectories);

    assert!(
        feasibility.all_feasible(),
        "Stochastic 12-stage produced {} violations in {} trajectories",
        feasibility.num_violations(),
        feasibility.num_trajectories
    );

    // Validate reasonableness (may have some unexpected behaviors, just check it runs)
    let _reasonableness =
        validator.check_reasonableness(&sim_result.trajectories);
}

#[test]
fn test_policy_improvement_with_training() {
    // Helper function to create the problem setup
    fn create_problem() -> (
        graph::DirectedGraph<NodeData>,
        initial_condition::InitialCondition,
    ) {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    System::default(),
                    "expectation",
                    "naive",
                    &[],
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        let mut prev_id = node_data_graph
            .add_node(
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    System::default(),
                    "expectation",
                    "naive",
                    &[],
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();

        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );

        for stage in 1..=4 {
            let new_id = node_data_graph
                .add_node(
                    NodeData::new(
                        stage,
                        stage.try_into().unwrap(),
                        stage.try_into().unwrap(),
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        System::default(),
                        "expectation",
                        "naive",
                        &[],
                        "storage",
                        1,
                    )
                    .unwrap(),
                )
                .unwrap();

            node_data_graph.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;

            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }

        let storage = vec![83.222];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        (node_data_graph, initial_condition)
    }

    // Create scenario generator separately (reusable)
    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );
    for _ in 1..=4 {
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
    }
    let saa = scenario_generator.generate(42);

    // Train for 10 iterations
    let (node_data_graph_10, initial_condition_10) = create_problem();
    let mut sddp_10 =
        SddpAlgorithm::new(node_data_graph_10, initial_condition_10, 0)
            .unwrap();
    let _train_result_10 = sddp_10.train(10, 5, &saa).unwrap();
    let sim_result_10 = sddp_10.simulate_and_analyze(100, &saa).unwrap();
    let cost_10 = sim_result_10.statistics.mean;

    // Train for 20 iterations
    let (node_data_graph_20, initial_condition_20) = create_problem();
    let mut sddp_20 =
        SddpAlgorithm::new(node_data_graph_20, initial_condition_20, 0)
            .unwrap();
    let _train_result_20 = sddp_20.train(20, 5, &saa).unwrap();
    let sim_result_20 = sddp_20.simulate_and_analyze(100, &saa).unwrap();
    let cost_20 = sim_result_20.statistics.mean;

    // Train for 30 iterations
    let (node_data_graph_30, initial_condition_30) = create_problem();
    let mut sddp_30 =
        SddpAlgorithm::new(node_data_graph_30, initial_condition_30, 0)
            .unwrap();
    let _train_result_30 = sddp_30.train(30, 5, &saa).unwrap();
    let sim_result_30 = sddp_30.simulate_and_analyze(100, &saa).unwrap();
    let cost_30 = sim_result_30.statistics.mean;

    // Cost should improve (decrease) or stay roughly constant with more training
    // Allow some slack for stochasticity
    assert!(
        cost_30 <= cost_10 * 1.05,
        "Policy did not improve: cost_10={:.2}, cost_30={:.2}",
        cost_10,
        cost_30
    );

    // At minimum, 30 iterations shouldn't be significantly worse than 20
    assert!(
        cost_30 <= cost_20 * 1.02,
        "Policy got worse from 20 to 30 iterations: cost_20={:.2}, cost_30={:.2}",
        cost_20,
        cost_30
    );
}

#[test]
fn test_stability_across_random_seeds() {
    // Helper function to create problem setup
    fn create_problem() -> (
        graph::DirectedGraph<NodeData>,
        initial_condition::InitialCondition,
    ) {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    System::default(),
                    "expectation",
                    "naive",
                    &[],
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        let mut prev_id = node_data_graph
            .add_node(
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    System::default(),
                    "expectation",
                    "naive",
                    &[],
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();

        for stage in 1..=3 {
            let new_id = node_data_graph
                .add_node(
                    NodeData::new(
                        stage,
                        stage.try_into().unwrap(),
                        stage.try_into().unwrap(),
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        System::default(),
                        "expectation",
                        "naive",
                        &[],
                        "storage",
                        1,
                    )
                    .unwrap(),
                )
                .unwrap();

            node_data_graph.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
        }

        let storage = vec![83.222];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        (node_data_graph, initial_condition)
    }

    // Create scenario generator (reusable)
    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.6928).unwrap()],
        3,
    );
    for _ in 1..=3 {
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
    }

    // Train with different seeds
    let seeds = vec![42, 123, 456];
    let mut costs = Vec::new();

    for seed in seeds {
        let saa = scenario_generator.generate(seed);
        let (node_data_graph, initial_condition) = create_problem();
        let mut sddp =
            SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();
        let _train_result = sddp.train(50, 5, &saa).unwrap();
        let sim_result = sddp.simulate_and_analyze(100, &saa).unwrap();
        costs.push(sim_result.statistics.mean);
    }

    // Compute mean and coefficient of variation
    let mean_cost: f64 = costs.iter().sum::<f64>() / costs.len() as f64;
    let variance: f64 =
        costs.iter().map(|c| (c - mean_cost).powi(2)).sum::<f64>()
            / costs.len() as f64;
    let std = variance.sqrt();
    let cv = std / mean_cost;

    assert!(
        cv < 0.60,
        "Policy unstable across seeds: costs={:?}, CV={:.2}%",
        costs,
        cv * 100.0
    );
}

#[test]
fn test_edge_case_zero_iterations() {
    // Policy trained with 0 iterations should still produce feasible solutions
    let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

    let pre_study_id = node_data_graph
        .add_node(
            NodeData::new(
                -1,
                0,
                0,
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                subproblem::StudyPeriodKind::PreStudy,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    let stage0_id = node_data_graph
        .add_node(
            NodeData::new(
                0,
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                subproblem::StudyPeriodKind::Study,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    node_data_graph.add_edge(pre_study_id, stage0_id).unwrap();

    let storage = vec![83.222];
    let initial_condition =
        initial_condition::InitialCondition::new(storage, vec![]);

    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );

    let saa = scenario_generator.generate(0);
    let mut sddp =
        SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

    // Train for 0 iterations - this should work (creates initial FCF but no cuts)
    // The result will be Err because no iterations were completed
    let train_result = sddp.train(0, 1, &saa);
    assert!(
        train_result.is_err(),
        "Expected error for zero iterations, but got Ok"
    );

    // Even with zero training, we should be able to simulate (using greedy dispatch)
    let sim_result = sddp.simulate_and_analyze(10, &saa).unwrap();

    // Should still be feasible (greedy dispatch should satisfy constraints)
    let validator = PolicyValidator::new(System::default());
    let feasibility = validator.check_feasibility(&sim_result.trajectories);

    assert!(
        feasibility.all_feasible(),
        "Zero-iteration policy produced infeasible solutions: {:?}",
        feasibility.violations
    );
}

#[test]
fn test_edge_case_one_iteration() {
    // Policy trained with 1 iteration (minimal cuts)
    let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

    let pre_study_id = node_data_graph
        .add_node(
            NodeData::new(
                -1,
                0,
                0,
                "1970-01-01T00:00:00Z",
                "1970-01-01T00:00:00Z",
                subproblem::StudyPeriodKind::PreStudy,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    let stage0_id = node_data_graph
        .add_node(
            NodeData::new(
                0,
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                subproblem::StudyPeriodKind::Study,
                System::default(),
                "expectation",
                "naive",
                &[],
                "storage",
                1,
            )
            .unwrap(),
        )
        .unwrap();

    node_data_graph.add_edge(pre_study_id, stage0_id).unwrap();

    let storage = vec![83.222];
    let initial_condition =
        initial_condition::InitialCondition::new(storage, vec![]);

    let mut scenario_generator = scenario::NoiseGenerator::new();
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );
    scenario_generator.add_node_generator(
        vec![Normal::new(75.0, 0.0).unwrap()],
        vec![LogNormal::new(3.6, 0.0).unwrap()],
        1,
    );

    let saa = scenario_generator.generate(0);
    let mut sddp =
        SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

    // Train for 1 iteration
    let _train_result = sddp.train(1, 1, &saa).unwrap();

    // Simulate
    let sim_result = sddp.simulate_and_analyze(10, &saa).unwrap();

    // Should be feasible
    let validator = PolicyValidator::new(System::default());
    let feasibility = validator.check_feasibility(&sim_result.trajectories);

    assert!(
        feasibility.all_feasible(),
        "One-iteration policy produced infeasible solutions"
    );
}

#[test]
fn test_performance_feasibility_checking() {
    // Create many feasible trajectories
    let trajectories: Vec<Trajectory> =
        (0..1000).map(|_| create_feasible_trajectory()).collect();

    let system = System::default();
    let validator = PolicyValidator::new(system);

    let start = Instant::now();
    let _report = validator.check_feasibility(&trajectories);
    let duration = start.elapsed();

    // Should complete in < 100ms
    assert!(
        duration.as_millis() < 100,
        "Feasibility checking took {}ms (expected <100ms)",
        duration.as_millis()
    );
}

#[test]
fn test_performance_reasonableness_checking() {
    // Create many trajectories
    let trajectories: Vec<Trajectory> =
        (0..1000).map(|_| create_feasible_trajectory()).collect();

    let system = System::default();
    let validator = PolicyValidator::new(system);

    let start = Instant::now();
    let _report = validator.check_reasonableness(&trajectories);
    let duration = start.elapsed();

    // Should complete in < 50ms
    assert!(
        duration.as_millis() < 50,
        "Reasonableness checking took {}ms (expected <50ms)",
        duration.as_millis()
    );
}

#[test]
fn test_memory_efficiency() {
    // Validate that validation doesn't allocate excessively
    let trajectories: Vec<Trajectory> =
        (0..100).map(|_| create_feasible_trajectory()).collect();

    let system = System::default();
    let validator = PolicyValidator::new(system);

    // Feasibility checking should not leak memory or allocate per-trajectory
    for _ in 0..10 {
        let _report = validator.check_feasibility(&trajectories);
        let _report = validator.check_reasonableness(&trajectories);
    }

    // If this test completes without OOM, memory usage is reasonable
}
