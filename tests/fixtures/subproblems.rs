// Test fixtures for subproblem construction testing (T2.9)
//
// Provides helper functions to create test systems and subproblems
// for comprehensive testing of subproblem construction,
// constraint generation, and solver integration.
//
// ARCHITECTURE NOTE: Subproblem construction is complex with many
// components: variables, constraints, state interaction, cut integration.
// These fixtures provide known-good configurations for testing.

#![allow(deprecated)]

use powers_rs::input::{MarginalDistribution, UncertaintyType};
use powers_rs::subproblem::{Realization, Subproblem};
use powers_rs::system::System;
use powers_rs::temporal_model::TemporalModel;

/// Creates a minimal single-bus, single-hydro system for basic testing
///
/// System characteristics:
/// - 1 bus with deficit cost of 1000.0
/// - 1 hydro plant (productivity 1.0, no downstream)
/// - No thermal plants
/// - No transmission lines
///
/// # Use Cases
/// - Testing basic variable/constraint structure
/// - Validating minimal subproblem construction
/// - Edge case: system with no thermals, no lines
pub fn create_minimal_system() -> System {
    System::default() // Default system has exactly this configuration
}

/// Creates a simple two-bus system with cascade hydro plants
///
/// System characteristics:
/// - 2 buses (deficit costs: 1000.0, 900.0)
/// - 2 hydro plants in cascade (hydro 0 upstream of hydro 1)
/// - 1 transmission line (50 MW capacity each direction)
/// - No thermal plants
///
/// # Use Cases
/// - Testing hydro cascade constraint generation
/// - Testing transmission constraints
/// - Integration tests with meaningful structure
pub fn create_cascade_system() -> System {
    // Use the JSON loading mechanism for consistency
    let json = r#"{
        "buses": [
            {"id": 0, "deficit_cost": 1000.0},
            {"id": 1, "deficit_cost": 900.0}
        ],
        "hydros": [
            {
                "id": 0,
                "bus_id": 0,
                "downstream_hydro_id": 1,
                "productivity": 1.0,
                "min_storage": 0.0,
                "max_storage": 100.0,
                "min_turbined_flow": 0.0,
                "max_turbined_flow": 50.0,
                "spillage_penalty": 10.0
            },
            {
                "id": 1,
                "bus_id": 1,
                "downstream_hydro_id": null,
                "productivity": 0.95,
                "min_storage": 0.0,
                "max_storage": 80.0,
                "min_turbined_flow": 0.0,
                "max_turbined_flow": 40.0,
                "spillage_penalty": 5.0
            }
        ],
        "thermals": [],
        "lines": [
            {
                "id": 0,
                "source_bus_id": 0,
                "target_bus_id": 1,
                "direct_capacity": 50.0,
                "reverse_capacity": 50.0,
                "exchange_penalty": 1.0
            }
        ]
    }"#;

    let input: powers_rs::input::SystemInput = serde_json::from_str(json)
        .expect("Failed to parse cascade system JSON");
    input.build_sddp_system()
}

/// Creates a system with thermal plants for mixed generation testing
///
/// System characteristics:
/// - 1 bus with deficit cost of 1000.0
/// - 1 hydro plant (productivity 1.0)
/// - 2 thermal plants (costs: 50.0, 100.0)
/// - No transmission lines
///
/// # Use Cases
/// - Testing thermal generation variables
/// - Testing mixed hydro-thermal dispatch
/// - Validating objective function with thermal costs
pub fn create_mixed_system() -> System {
    let json = r#"{
        "buses": [
            {"id": 0, "deficit_cost": 1000.0}
        ],
        "hydros": [
            {
                "id": 0,
                "bus_id": 0,
                "downstream_hydro_id": null,
                "productivity": 1.0,
                "min_storage": 0.0,
                "max_storage": 100.0,
                "min_turbined_flow": 0.0,
                "max_turbined_flow": 50.0,
                "spillage_penalty": 0.0
            }
        ],
        "thermals": [
            {
                "id": 0,
                "bus_id": 0,
                "cost": 50.0,
                "min_generation": 0.0,
                "max_generation": 30.0
            },
            {
                "id": 1,
                "bus_id": 0,
                "cost": 100.0,
                "min_generation": 0.0,
                "max_generation": 20.0
            }
        ],
        "lines": []
    }"#;

    let input: powers_rs::input::SystemInput =
        serde_json::from_str(json).expect("Failed to parse mixed system JSON");
    input.build_sddp_system()
}

/// Creates a subproblem with the minimal system configuration
///
/// Uses "storage" state choice and independent uncertainty models.
///
/// # Returns
/// - Subproblem with model already constructed and ready to solve
///
/// # Use Cases
/// - Testing basic subproblem construction
/// - Validating variable/constraint counts
/// - Edge case testing with minimal structure
pub fn create_minimal_subproblem() -> Subproblem {
    let system = create_minimal_system();

    // Create minimal temporal model (Independent model with default params)
    let temporal_models = vec![TemporalModel::from_independent(
        UncertaintyType::Inflow,
        0,
        vec![100.0],
        vec![20.0],
        vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }],
    )
    .unwrap()];

    Subproblem::new_from_temporal_models(
        &system,
        "storage",
        &temporal_models,
        0,
    )
}

/// Creates a subproblem with the cascade system configuration
///
/// Uses "storage" state choice and independent uncertainty models.
///
/// # Returns
/// - Subproblem with cascade hydro constraints
///
/// # Use Cases
/// - Testing cascade constraint generation
/// - Testing upstream/downstream relationships
/// - Integration tests with multiple hydros
pub fn create_cascade_subproblem() -> Subproblem {
    let system = create_cascade_system();

    // Create temporal models for both hydros in the cascade
    let temporal_models = vec![
        TemporalModel::from_independent(
            UncertaintyType::Inflow,
            0,
            vec![100.0],
            vec![20.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }],
        )
        .unwrap(),
        TemporalModel::from_independent(
            UncertaintyType::Inflow,
            1,
            vec![100.0],
            vec![20.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            }],
        )
        .unwrap(),
    ];

    Subproblem::new_from_temporal_models(
        &system,
        "storage",
        &temporal_models,
        0,
    )
}

/// Creates a mixed subproblem (hydro + thermal generation).
///
/// System: 1 bus, 1 hydro, 2 thermals, 0 lines
#[allow(dead_code)]
pub fn create_mixed_subproblem() -> Subproblem {
    let system = create_mixed_system();

    let temporal_models = vec![TemporalModel::from_independent(
        UncertaintyType::Inflow,
        0,
        vec![100.0],
        vec![20.0],
        vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }],
    )
    .unwrap()];

    Subproblem::new_from_temporal_models(
        &system,
        "storage",
        &temporal_models,
        0,
    )
}

/// Creates a realization container with the given system dimensions
///
/// All fields initialized to zero except the specified ones.
///
/// # Arguments
/// - `system`: System to get dimensions from
/// - `final_storage`: Optional final storage values (defaults to zeros)
///
/// # Returns
/// - Realization container suitable for realize_uncertainties
pub fn create_test_realization(
    system: &System,
    final_storage: Option<Vec<f64>>,
) -> Realization {
    let storage =
        final_storage.unwrap_or_else(|| vec![0.0; system.meta.hydros_count]);

    Realization::new(
        vec![0.0; system.meta.buses_count],    // loads
        vec![0.0; system.meta.buses_count],    // deficit
        vec![0.0; system.meta.lines_count],    // exchange
        vec![0.0; system.meta.hydros_count],   // inflow
        vec![0.0; system.meta.hydros_count],   // turbined_flow
        vec![0.0; system.meta.hydros_count],   // spillage
        vec![0.0; system.meta.thermals_count], // thermal_generation
        vec![0.0; system.meta.hydros_count],   // water_value
        vec![0.0; system.meta.buses_count],    // marginal_cost
        0.0,                                   // current_stage_objective
        0.0,                                   // total_stage_objective
        storage,                               // final_storage
        powers_rs::solver::Basis::new(),       // basis
    )
}

/// Creates a minimal realization container for test purposes.
///
/// # Arguments
/// * `initial_storage` - Initial storage value (defaults to 50.0 if None)
#[allow(dead_code)]
pub fn create_minimal_realization(initial_storage: Option<f64>) -> Realization {
    let system = create_minimal_system();
    let storage = vec![initial_storage.unwrap_or(50.0)];
    create_test_realization(&system, Some(storage))
}

/// Creates a realization container for the cascade system
///
/// # Arguments
/// - `upstream_storage`: Storage for upstream hydro (default: 50.0)
/// - `downstream_storage`: Storage for downstream hydro (default: 40.0)
///
/// # Returns
/// - Realization container with cascade system dimensions
#[allow(dead_code)]
pub fn create_cascade_realization(
    upstream_storage: Option<f64>,
    downstream_storage: Option<f64>,
) -> Realization {
    let system = create_cascade_system();
    let storage = vec![
        upstream_storage.unwrap_or(50.0),
        downstream_storage.unwrap_or(40.0),
    ];
    create_test_realization(&system, Some(storage))
}

/// Helper to create naive stochastic processes (deterministic)
///
/// Returns a tuple of (load_sp, inflow_sp) both using "naive" strategy.
///
/// # Use Cases
/// - Deterministic testing
/// - Baseline subproblem tests without stochasticity
#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_system_structure() {
        let system = create_minimal_system();
        assert_eq!(system.meta.buses_count, 1);
        assert_eq!(system.meta.hydros_count, 1);
        assert_eq!(system.meta.thermals_count, 2); // Default has 2 thermals
        assert_eq!(system.meta.lines_count, 0);
    }

    #[test]
    fn test_cascade_system_structure() {
        let system = create_cascade_system();
        assert_eq!(system.meta.buses_count, 2);
        assert_eq!(system.meta.hydros_count, 2);
        assert_eq!(system.meta.thermals_count, 0);
        assert_eq!(system.meta.lines_count, 1);

        // Verify cascade relationship
        assert_eq!(system.hydros[0].downstream_hydro_id, Some(1));
        assert_eq!(system.hydros[1].downstream_hydro_id, None);
    }

    #[test]
    fn test_mixed_system_structure() {
        let system = create_mixed_system();
        assert_eq!(system.meta.buses_count, 1);
        assert_eq!(system.meta.hydros_count, 1);
        assert_eq!(system.meta.thermals_count, 2);
        assert_eq!(system.meta.lines_count, 0);
    }

    #[test]
    fn test_minimal_subproblem_creation() {
        let subproblem = create_minimal_subproblem();
        assert!(subproblem.model.is_some());
        assert_eq!(subproblem.variables.deficit.len(), 1);
        assert_eq!(subproblem.variables.stored_volume.len(), 1);
    }

    #[test]
    fn test_cascade_subproblem_creation() {
        let subproblem = create_cascade_subproblem();
        assert!(subproblem.model.is_some());
        assert_eq!(subproblem.variables.deficit.len(), 2);
        assert_eq!(subproblem.variables.stored_volume.len(), 2);
        assert_eq!(subproblem.variables.direct_exchange.len(), 1);
    }

    #[test]
    fn test_realization_dimensions() {
        let system = create_cascade_system();
        let realization = create_test_realization(&system, None);

        assert_eq!(realization.loads.len(), 2);
        assert_eq!(realization.deficit.len(), 2);
        assert_eq!(realization.exchange.len(), 1);
        assert_eq!(realization.inflow.len(), 2);
        assert_eq!(realization.final_storage.len(), 2);
    }
}

// ============================================================================
// AR Model Test Fixtures for TICKET-005
// ============================================================================

/// Creates a system with mixed entity types and heterogeneous AR orders
///
/// System characteristics:
/// - 2 buses: Bus 0 AR(1), Bus 1 AR(0)
/// - 3 hydros: Hydro 0 AR(2), Hydro 1 AR(0), Hydro 2 AR(1)
///
/// Use case: Testing correct separation and indexing of load/inflow lag duals
#[allow(dead_code)]
pub fn mixed_ar_system(
) -> (System, Vec<powers_rs::temporal_model::TemporalModel>) {
    use powers_rs::input::MarginalDistribution;
    use powers_rs::system::{Bus, Hydro};
    use powers_rs::temporal_model::TemporalModel;

    // Create system with 2 buses and 3 hydros
    let buses = vec![
        Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![0, 1],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
        Bus {
            id: 1,
            deficit_cost: 1000.0,
            hydro_ids: vec![2],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
    ];

    let hydros = vec![
        Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 1,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 2,
            downstream_hydro_id: None,
            bus_id: 1,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
    ];

    let system = System::new(buses, vec![], vec![], hydros);

    // Create temporal models
    let temporal_models = vec![
        // Bus 0: AR(1) load
        TemporalModel::from_par(
            UncertaintyType::Load,
            0,
            1,
            vec![50.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            }],
            vec![1],
            vec![vec![0.5]],
        )
        .unwrap(),
        // Bus 1: AR(0) load
        TemporalModel::from_par(
            UncertaintyType::Load,
            1,
            1,
            vec![40.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 40.0,
                std_dev: 10.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        // Hydro 0: AR(2) inflow
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![2],
            vec![vec![0.6, 0.3]],
        )
        .unwrap(),
        // Hydro 1: AR(0) inflow
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            1,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        // Hydro 2: AR(1) inflow
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            2,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![1],
            vec![vec![0.7]],
        )
        .unwrap(),
    ];

    (system, temporal_models)
}

/// Creates a system where only inflows have AR dynamics
#[allow(dead_code)]
pub fn inflow_only_ar_system(
) -> (System, Vec<powers_rs::temporal_model::TemporalModel>) {
    use powers_rs::input::MarginalDistribution;
    use powers_rs::system::{Bus, Hydro};
    use powers_rs::temporal_model::TemporalModel;

    let buses = vec![
        Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
        Bus {
            id: 1,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
    ];

    let hydros = vec![
        Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 1,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 2,
            downstream_hydro_id: None,
            bus_id: 1,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
    ];

    let system = System::new(buses, vec![], vec![], hydros);

    let temporal_models = vec![
        TemporalModel::from_par(
            UncertaintyType::Load,
            0,
            1,
            vec![50.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Load,
            1,
            1,
            vec![40.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 40.0,
                std_dev: 10.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![1],
            vec![vec![0.6]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            1,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![2],
            vec![vec![0.5, 0.3]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            2,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![3],
            vec![vec![0.4, 0.3, 0.2]],
        )
        .unwrap(),
    ];

    (system, temporal_models)
}

/// Creates a system where only loads have AR dynamics
#[allow(dead_code)]
pub fn load_only_ar_system(
) -> (System, Vec<powers_rs::temporal_model::TemporalModel>) {
    use powers_rs::input::MarginalDistribution;
    use powers_rs::system::{Bus, Hydro};
    use powers_rs::temporal_model::TemporalModel;

    let buses = vec![
        Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
        Bus {
            id: 1,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
    ];

    let hydros = vec![
        Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 1,
            downstream_hydro_id: None,
            bus_id: 1,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
    ];

    let system = System::new(buses, vec![], vec![], hydros);

    let temporal_models = vec![
        TemporalModel::from_par(
            UncertaintyType::Load,
            0,
            1,
            vec![50.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            }],
            vec![1],
            vec![vec![0.5]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Load,
            1,
            1,
            vec![40.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 40.0,
                std_dev: 10.0,
            }],
            vec![2],
            vec![vec![0.6, 0.3]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            1,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
    ];

    (system, temporal_models)
}

/// Creates a system with no AR dynamics
#[allow(dead_code)]
pub fn no_ar_system() -> (System, Vec<powers_rs::temporal_model::TemporalModel>)
{
    use powers_rs::input::MarginalDistribution;
    use powers_rs::system::{Bus, Hydro};
    use powers_rs::temporal_model::TemporalModel;

    let buses = vec![
        Bus {
            id: 0,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
        Bus {
            id: 1,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        },
    ];

    let hydros = vec![
        Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 1,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
        Hydro {
            id: 2,
            downstream_hydro_id: None,
            bus_id: 1,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        },
    ];

    let system = System::new(buses, vec![], vec![], hydros);

    let temporal_models = vec![
        TemporalModel::from_par(
            UncertaintyType::Load,
            0,
            1,
            vec![50.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Load,
            1,
            1,
            vec![40.0],
            vec![10.0],
            vec![MarginalDistribution::Normal {
                mean: 40.0,
                std_dev: 10.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            0,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            1,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
        TemporalModel::from_par(
            UncertaintyType::Inflow,
            2,
            1,
            vec![100.0],
            vec![15.0],
            vec![MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 15.0,
            }],
            vec![0],
            vec![vec![]],
        )
        .unwrap(),
    ];

    (system, temporal_models)
}

/// Creates a large system with heterogeneous AR orders
#[allow(dead_code)]
pub fn large_heterogeneous_system(
) -> (System, Vec<powers_rs::temporal_model::TemporalModel>) {
    use powers_rs::input::MarginalDistribution;
    use powers_rs::system::{Bus, Hydro};
    use powers_rs::temporal_model::TemporalModel;

    let buses: Vec<Bus> = (0..5)
        .map(|i| Bus {
            id: i,
            deficit_cost: 1000.0,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        })
        .collect();
    let hydros: Vec<Hydro> = (0..10)
        .map(|i| Hydro {
            id: i,
            downstream_hydro_id: None,
            bus_id: i % 5,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_turbined_flow: 0.0,
            max_turbined_flow: 50.0,
            spillage_penalty: 0.0,
            upstream_hydro_ids: vec![],
        })
        .collect();

    let system = System::new(buses, vec![], vec![], hydros);

    let load_ar_orders = vec![0, 1, 2, 0, 1];
    let inflow_ar_orders = vec![0, 1, 2, 0, 2, 1, 3, 0, 1, 2];

    let mut temporal_models = Vec::new();

    for (bus_id, &ar_order) in load_ar_orders.iter().enumerate() {
        let phi: Vec<f64> =
            (0..ar_order).map(|i| 0.5 - 0.1 * i as f64).collect();
        let seasonal_mean = 50.0 + bus_id as f64 * 5.0;
        temporal_models.push(
            TemporalModel::from_par(
                UncertaintyType::Load,
                bus_id,
                1,
                vec![seasonal_mean],
                vec![10.0],
                vec![MarginalDistribution::Normal {
                    mean: seasonal_mean,
                    std_dev: 10.0,
                }],
                vec![ar_order],
                vec![phi],
            )
            .unwrap(),
        );
    }

    for (hydro_id, &ar_order) in inflow_ar_orders.iter().enumerate() {
        let phi: Vec<f64> =
            (0..ar_order).map(|i| 0.6 - 0.1 * i as f64).collect();
        let seasonal_mean = 100.0 + hydro_id as f64 * 5.0;
        temporal_models.push(
            TemporalModel::from_par(
                UncertaintyType::Inflow,
                hydro_id,
                1,
                vec![seasonal_mean],
                vec![15.0],
                vec![MarginalDistribution::Normal {
                    mean: seasonal_mean,
                    std_dev: 15.0,
                }],
                vec![ar_order],
                vec![phi],
            )
            .unwrap(),
        );
    }

    (system, temporal_models)
}
