//! Integration tests for PAR scenario generation
//!
//! Tests the full CEPEL pipeline: noise → correlation → residual transform → PAR dynamics

use powers_rs::{
    initial_condition::InitialCondition,
    input::{
        CorrelationBlock, CorrelationMethod, CorrelationSpecification,
        EntityReference, InitialConditionInput, MarginalDistribution,
        NoiseModel, PastInflow, Recourse, TemporalModel, UncertaintyType,
    },
    scenario::ScenarioGenerator,
};

/// Helper to create empty InitialConditionInput
fn empty_initial_condition_input() -> InitialConditionInput {
    InitialConditionInput {
        storage: vec![],
        inflow: vec![],
    }
}

/// Helper to create InitialCondition with no lags
fn empty_initial_condition() -> InitialCondition {
    InitialCondition::new(vec![], vec![])
}

/// Test PAR(1) model with 2 periods generates valid scenarios
#[test]
fn test_par_simple_2period() {
    // Create simple PAR(1) model with 2 periods
    let noise_models = vec![NoiseModel {
        uncertainty_type: UncertaintyType::Inflow,
        entity_id: 0,
        season_id: 0,
        marginal_distribution: MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        },
        innovation_distribution: None,
        temporal_model: TemporalModel::PeriodicAutoregressive {
            period: 2,
            ar_orders: vec![1, 1],
            ar_coefficients: vec![vec![0.7], vec![0.6]],
            seasonal_means: vec![100.0, 120.0],
            seasonal_stds: vec![20.0, 25.0],
        },
        residual_distribution: None,
    }];

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: empty_initial_condition_input(),
    };

    let initial_condition = empty_initial_condition();
    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    // Generate 4 stages with 10 scenarios each
    let saa = generator.generate_saa(4, &[10, 10, 10, 10]);

    // Verify structure
    for stage_id in 0..4 {
        assert_eq!(
            saa.get_branching_count_at_stage(stage_id),
            Some(10),
            "Stage {} should have 10 branchings",
            stage_id
        );

        // Check all scenarios in this stage
        for scenario_id in 0..10 {
            let noises = saa
                .get_noises_by_stage_and_branching(stage_id, scenario_id)
                .expect("Should have noises");

            assert_eq!(
                noises.num_inflow_entities, 1,
                "Should have 1 inflow entity"
            );
            assert_eq!(
                noises.num_load_entities, 0,
                "Should have 0 load entities"
            );

            let inflow = noises.get_inflow_noises()[0];
            // All values should be finite (no NaN, Inf)
            assert!(
                inflow.is_finite(),
                "Inflow should be finite, got {}",
                inflow
            );
            // Values should be reasonable for Normal(0,1) residuals with mean~100-120, std~20-25
            // Expect roughly mean ± 3*std range: [40, 200]
            assert!(
                inflow > 20.0 && inflow < 250.0,
                "Inflow {} should be in reasonable range [20, 250]",
                inflow
            );
        }
    }
}

/// Test PAR with initial lags (warm start)
#[test]
fn test_par_with_initial_lags() {
    // PAR(2) model with initial lags
    let noise_models = vec![NoiseModel {
        uncertainty_type: UncertaintyType::Inflow,
        entity_id: 0,
        season_id: 0,
        marginal_distribution: MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        },
        innovation_distribution: None,
        temporal_model: TemporalModel::PeriodicAutoregressive {
            period: 1, // Single period for simplicity
            ar_orders: vec![2],
            ar_coefficients: vec![vec![0.5, 0.3]],
            seasonal_means: vec![100.0],
            seasonal_stds: vec![20.0],
        },
        residual_distribution: None,
    }];

    // Provide initial lags
    let initial_condition = InitialCondition::new(
        vec![],
        vec![vec![1.0, 0.5]], // Two past residuals for AR(2) for hydro 0
    );

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![
                PastInflow {
                    hydro_id: 0,
                    lag: 1,
                    value: 1.0,
                },
                PastInflow {
                    hydro_id: 0,
                    lag: 2,
                    value: 0.5,
                },
            ],
        },
    };

    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    let saa = generator.generate_saa(2, &[5, 5]);

    // Verify scenarios generated successfully
    assert_eq!(saa.get_branching_count_at_stage(0), Some(5));
    assert_eq!(saa.get_branching_count_at_stage(1), Some(5));
}

/// Test mixing PAR and Independent models
#[test]
fn test_par_mixed_with_independent() {
    let noise_models = vec![
        // Entity 0: PAR model
        NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 0,
            marginal_distribution: MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            innovation_distribution: None,
            temporal_model: TemporalModel::PeriodicAutoregressive {
                period: 2,
                ar_orders: vec![1, 1],
                ar_coefficients: vec![vec![0.7], vec![0.6]],
                seasonal_means: vec![100.0, 120.0],
                seasonal_stds: vec![20.0, 25.0],
            },
            residual_distribution: None,
        },
        // Entity 1: Independent model
        NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 1,
            season_id: 0,
            marginal_distribution: MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            },
            innovation_distribution: None,
            temporal_model: TemporalModel::Independent,
            residual_distribution: None,
        },
    ];

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: empty_initial_condition_input(),
    };

    let initial_condition = empty_initial_condition();
    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    let saa = generator.generate_saa(3, &[8, 8, 8]);

    // Verify both entities have scenarios
    for stage_id in 0..3 {
        for scenario_id in 0..8 {
            let noises = saa
                .get_noises_by_stage_and_branching(stage_id, scenario_id)
                .unwrap();

            assert_eq!(noises.num_inflow_entities, 2);
            let inflows = noises.get_inflow_noises();

            // Entity 0 (PAR): should be reasonable for Normal(0,1) residuals
            assert!(
                inflows[0] > 20.0 && inflows[0] < 250.0,
                "PAR inflow {} out of range",
                inflows[0]
            );
            // Entity 1 (Independent): should be around mean ± 3σ
            assert!(
                inflows[1] > 20.0 && inflows[1] < 80.0,
                "Independent inflow {} out of range",
                inflows[1]
            );
        }
    }
}

/// Test PAR with correlation
#[test]
fn test_par_with_correlation() {
    let noise_models = vec![
        NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 0,
            season_id: 0,
            marginal_distribution: MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            innovation_distribution: None,
            temporal_model: TemporalModel::PeriodicAutoregressive {
                period: 1,
                ar_orders: vec![1],
                ar_coefficients: vec![vec![0.7]],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![20.0],
            },
            residual_distribution: None,
        },
        NoiseModel {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id: 1,
            season_id: 0,
            marginal_distribution: MarginalDistribution::Normal {
                mean: 120.0,
                std_dev: 25.0,
            },
            innovation_distribution: None,
            temporal_model: TemporalModel::PeriodicAutoregressive {
                period: 1,
                ar_orders: vec![1],
                ar_coefficients: vec![vec![0.6]],
                seasonal_means: vec![120.0],
                seasonal_stds: vec![25.0],
            },
            residual_distribution: None,
        },
    ];

    // Add correlation block with strong positive correlation
    let correlation = Some(CorrelationSpecification {
        method: CorrelationMethod::Cholesky,
        blocks: vec![CorrelationBlock {
            name: "hydro".to_string(),
            entities: vec![
                EntityReference {
                    uncertainty_type: UncertaintyType::Inflow,
                    entity_id: 0,
                },
                EntityReference {
                    uncertainty_type: UncertaintyType::Inflow,
                    entity_id: 1,
                },
            ],
            correlation_matrix: vec![vec![1.0, 0.8], vec![0.8, 1.0]],
        }],
    });

    let recourse = Recourse {
        noise_models,
        correlation,
        initial_condition: empty_initial_condition_input(),
    };

    let initial_condition = empty_initial_condition();
    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    let saa = generator.generate_saa(5, &[20, 20, 20, 20, 20]);

    // Verify correlation is present by checking that both entities
    // vary together (not a statistical test, just sanity check)
    let mut correlated_count = 0;
    let mut total_comparisons = 0;

    for stage_id in 0..5 {
        for scenario_id in 0..19 {
            // Compare consecutive scenarios
            let noises1 = saa
                .get_noises_by_stage_and_branching(stage_id, scenario_id)
                .unwrap();
            let noises2 = saa
                .get_noises_by_stage_and_branching(stage_id, scenario_id + 1)
                .unwrap();

            let inflows1 = noises1.get_inflow_noises();
            let inflows2 = noises2.get_inflow_noises();

            // Check if both entities move in same direction
            let delta0 = inflows2[0] - inflows1[0];
            let delta1 = inflows2[1] - inflows1[1];

            if delta0.abs() > 0.1 && delta1.abs() > 0.1 {
                total_comparisons += 1;
                if delta0.signum() == delta1.signum() {
                    correlated_count += 1;
                }
            }
        }
    }

    // With 0.8 correlation, we expect more than 50% to move together
    if total_comparisons > 0 {
        let correlation_ratio =
            correlated_count as f64 / total_comparisons as f64;
        println!(
            "Correlation ratio: {:.2} ({}/{})",
            correlation_ratio, correlated_count, total_comparisons
        );
        assert!(
            correlation_ratio > 0.5,
            "Expected positive correlation, got ratio {}",
            correlation_ratio
        );
    }
}

/// Test PAR with varying AR orders across periods
#[test]
fn test_par_varying_orders() {
    let noise_models = vec![NoiseModel {
        uncertainty_type: UncertaintyType::Inflow,
        entity_id: 0,
        season_id: 0,
        marginal_distribution: MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        },
        innovation_distribution: None,
        temporal_model: TemporalModel::PeriodicAutoregressive {
            period: 3,
            ar_orders: vec![1, 2, 1], // Varying orders
            ar_coefficients: vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
            seasonal_means: vec![100.0, 120.0, 110.0],
            seasonal_stds: vec![20.0, 25.0, 22.0],
        },
        residual_distribution: None,
    }];

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: empty_initial_condition_input(),
    };

    let initial_condition = empty_initial_condition();
    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    // Generate 6 stages (2 full periods)
    let saa = generator.generate_saa(6, &[10; 6]);

    // Verify all stages generated successfully
    for stage_id in 0..6 {
        assert_eq!(saa.get_branching_count_at_stage(stage_id), Some(10));

        for scenario_id in 0..10 {
            let noises = saa
                .get_noises_by_stage_and_branching(stage_id, scenario_id)
                .unwrap();
            let inflow = noises.get_inflow_noises()[0];

            assert!(inflow.is_finite(), "Inflow should be finite");
        }
    }
}

/// Regression test: Verify existing Independent/AR fixtures still work
#[test]
fn test_regression_independent_model() {
    // Pure Independent model (no PAR)
    let noise_models = vec![NoiseModel {
        uncertainty_type: UncertaintyType::Inflow,
        entity_id: 0,
        season_id: 0,
        marginal_distribution: MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        },
        innovation_distribution: None,
        temporal_model: TemporalModel::Independent,
        residual_distribution: None,
    }];

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: empty_initial_condition_input(),
    };

    let initial_condition = empty_initial_condition();
    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    let saa = generator.generate_saa(3, &[10, 10, 10]);

    // Verify scenarios generated successfully
    for stage_id in 0..3 {
        assert_eq!(saa.get_branching_count_at_stage(stage_id), Some(10));
    }
}

/// Regression test: Verify existing AR fixtures still work
#[test]
fn test_regression_ar_model() {
    // Pure AR model (no PAR)
    let noise_models = vec![NoiseModel {
        uncertainty_type: UncertaintyType::Inflow,
        entity_id: 0,
        season_id: 0,
        marginal_distribution: MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        },
        innovation_distribution: None,
        temporal_model: TemporalModel::Autoregressive {
            lag_order: 2,
            coefficients: vec![0.5, 0.3],
        },
        residual_distribution: None,
    }];

    let initial_condition = InitialCondition::new(
        vec![],
        vec![vec![100.0, 95.0]], // Initial lags for AR(2) for hydro 0
    );

    let recourse = Recourse {
        noise_models,
        correlation: None,
        initial_condition: InitialConditionInput {
            storage: vec![],
            inflow: vec![
                PastInflow {
                    hydro_id: 0,
                    lag: 1,
                    value: 100.0,
                },
                PastInflow {
                    hydro_id: 0,
                    lag: 2,
                    value: 95.0,
                },
            ],
        },
    };

    let generator = ScenarioGenerator::from_recourse_input(
        &recourse,
        &initial_condition,
        42,
    )
    .expect("Failed to create generator");

    let saa = generator.generate_saa(3, &[10, 10, 10]);

    // Verify scenarios generated successfully
    for stage_id in 0..3 {
        assert_eq!(saa.get_branching_count_at_stage(stage_id), Some(10));
    }
}
