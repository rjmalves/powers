// T4.2 Coverage Completion: Subproblem Error Paths and Edge Cases
// Target: Increase subproblem.rs coverage from 75.27% to 90%+
//
// This test file focuses on PUBLIC API testing of uncovered error paths:
// 1. Realization::default() and with_capacity() edge cases
// 2. StudyPeriodKind variants
// 3. System::Hydro::add_upstream_hydro
// 4. Edge cases reachable via public API

use powers_rs::subproblem::{Realization, StudyPeriodKind};
use powers_rs::system::{Bus, Hydro, Line, System, Thermal};

// ============================================================================
// Test 1: Realization::default()
// ============================================================================
// Tests lines 1067-1084 in subproblem.rs
// Tests the Default trait implementation for Realization

#[test]
fn test_realization_default() {
    let realization = Realization::default();

    assert_eq!(realization.kind, StudyPeriodKind::Study);
    assert_eq!(realization.loads.len(), 0);
    assert_eq!(realization.deficit.len(), 0);
    assert_eq!(realization.exchange.len(), 0);
    assert_eq!(realization.inflow.len(), 0);
    assert_eq!(realization.turbined_flow.len(), 0);
    assert_eq!(realization.spillage.len(), 0);
    assert_eq!(realization.thermal_generation.len(), 0);
    assert_eq!(realization.water_value.len(), 0);
    assert_eq!(realization.marginal_cost.len(), 0);
    assert_eq!(realization.current_stage_objective, 0.0);
    assert_eq!(realization.total_stage_objective, 0.0);
    assert_eq!(realization.final_storage.len(), 0);
}

// ============================================================================
// Test 2: Realization with Different StudyPeriodKind
// ============================================================================
// Tests Realization with PreStudy and PostStudy variants

#[test]
fn test_realization_with_prestudy_kind() {
    let system = System::default();
    let realization =
        Realization::with_capacity(&StudyPeriodKind::PreStudy, &system);

    assert_eq!(realization.kind, StudyPeriodKind::PreStudy);
    assert_eq!(realization.loads.len(), system.meta.buses_count);
    assert_eq!(realization.deficit.len(), system.meta.buses_count);
    assert_eq!(realization.inflow.len(), system.meta.hydros_count);
}

#[test]
fn test_realization_with_poststudy_kind() {
    let system = System::default();
    let realization =
        Realization::with_capacity(&StudyPeriodKind::PostStudy, &system);

    assert_eq!(realization.kind, StudyPeriodKind::PostStudy);
    assert_eq!(realization.loads.len(), system.meta.buses_count);
    assert_eq!(realization.exchange.len(), system.meta.lines_count);
}

#[test]
fn test_realization_with_study_kind() {
    let system = System::default();
    let realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    assert_eq!(realization.kind, StudyPeriodKind::Study);
    assert_eq!(
        realization.thermal_generation.len(),
        system.meta.thermals_count
    );
}

// ============================================================================
// Test 3: Hydro add_upstream_hydro
// ============================================================================
// Tests line 138-140 in system.rs

#[test]
fn test_hydro_add_upstream_hydro() {
    let mut hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01);

    // Initially no upstream hydros
    assert_eq!(hydro.upstream_hydro_ids.len(), 0);

    // Add upstream hydros
    hydro.add_upstream_hydro(1);
    hydro.add_upstream_hydro(2);

    assert_eq!(hydro.upstream_hydro_ids.len(), 2);
    assert_eq!(hydro.upstream_hydro_ids[0], 1);
    assert_eq!(hydro.upstream_hydro_ids[1], 2);
}

#[test]
fn test_hydro_add_upstream_hydro_multiple_calls() {
    let mut hydro = Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01);

    for i in 0..10 {
        hydro.add_upstream_hydro(i);
    }

    assert_eq!(hydro.upstream_hydro_ids.len(), 10);
    assert_eq!(hydro.upstream_hydro_ids[5], 5);
}

// ============================================================================
// Test 4: System Construction with Different Configurations
// ============================================================================
// Tests various system construction paths

#[test]
fn test_system_with_no_lines() {
    let buses = vec![Bus::new(0, 50.0)];
    let lines: Vec<Line> = vec![];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];
    let hydros = vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, lines, thermals, hydros);

    assert_eq!(system.meta.lines_count, 0);
    assert_eq!(system.buses[0].source_line_ids.len(), 0);
    assert_eq!(system.buses[0].target_line_ids.len(), 0);
}

#[test]
fn test_system_with_no_thermals() {
    let buses = vec![Bus::new(0, 50.0)];
    let lines: Vec<Line> = vec![];
    let thermals: Vec<Thermal> = vec![];
    let hydros = vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, lines, thermals, hydros);

    assert_eq!(system.meta.thermals_count, 0);
    assert_eq!(system.buses[0].thermal_ids.len(), 0);
}

#[test]
fn test_system_with_multiple_buses_and_lines() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 60.0)];
    let lines = vec![Line::new(0, 0, 1, 100.0, 80.0, 1.0)];
    let thermals = vec![
        Thermal::new(0, 0, 5.0, 0.0, 15.0),
        Thermal::new(1, 1, 7.0, 0.0, 20.0),
    ];
    let hydros = vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, lines, thermals, hydros);

    assert_eq!(system.meta.buses_count, 2);
    assert_eq!(system.meta.lines_count, 1);
    assert_eq!(system.meta.thermals_count, 2);
    assert_eq!(system.meta.hydros_count, 1);

    // Verify line connections were added to buses
    assert_eq!(system.buses[0].source_line_ids.len(), 1);
    assert_eq!(system.buses[1].target_line_ids.len(), 1);

    // Verify thermal connections
    assert_eq!(system.buses[0].thermal_ids.len(), 1);
    assert_eq!(system.buses[1].thermal_ids.len(), 1);

    // Verify hydro connections
    assert_eq!(system.buses[0].hydro_ids.len(), 1);
}

#[test]
fn test_system_with_cascade_hydros() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 60.0)];
    let lines: Vec<Line> = vec![];
    let thermals: Vec<Thermal> = vec![];
    let hydros = vec![
        Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(1, None, 1, 0.95, 0.0, 80.0, 0.0, 50.0, 0.01),
    ];

    let system = System::new(buses, lines, thermals, hydros);

    assert_eq!(system.meta.hydros_count, 2);
    assert_eq!(system.hydros[0].downstream_hydro_id, Some(1));
    assert_eq!(system.hydros[1].downstream_hydro_id, None);
}

// ============================================================================
// Test 5: Realization with Large System
// ============================================================================
// Tests memory allocation for large systems

#[test]
fn test_realization_with_large_system() {
    let buses: Vec<Bus> = (0..100).map(|i| Bus::new(i, 50.0)).collect();
    let lines: Vec<Line> = vec![];
    let thermals: Vec<Thermal> = (0..50)
        .map(|i| Thermal::new(i, i % 100, 5.0, 0.0, 15.0))
        .collect();
    let hydros: Vec<Hydro> = (0..30)
        .map(|i| Hydro::new(i, None, i % 100, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01))
        .collect();

    let system = System::new(buses, lines, thermals, hydros);
    let realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    assert_eq!(realization.loads.len(), 100);
    assert_eq!(realization.thermal_generation.len(), 50);
    assert_eq!(realization.inflow.len(), 30);
}

// ============================================================================
// Test 6: Realization::new with All Parameters
// ============================================================================
// Tests lines 1010-1041 in subproblem.rs

#[test]
fn test_realization_new_with_all_parameters() {
    let loads = vec![10.0];
    let deficit = vec![0.0];
    let exchange = vec![];
    let inflow = vec![5.0];
    let turbined_flow = vec![4.0];
    let spillage = vec![1.0];
    let thermal_generation = vec![6.0, 8.0];
    let water_value = vec![25.0];
    let marginal_cost = vec![100.0];
    let current_stage_objective = 150.0;
    let total_stage_objective = 200.0;
    let final_storage = vec![50.0];
    let basis = powers_rs::solver::Basis::new();

    let realization = Realization::new(
        loads.clone(),
        deficit.clone(),
        exchange.clone(),
        inflow.clone(),
        turbined_flow.clone(),
        spillage.clone(),
        thermal_generation.clone(),
        water_value.clone(),
        marginal_cost.clone(),
        current_stage_objective,
        total_stage_objective,
        final_storage.clone(),
        basis.clone(),
    );

    assert_eq!(realization.kind, StudyPeriodKind::Study);
    assert_eq!(realization.loads, loads);
    assert_eq!(realization.deficit, deficit);
    assert_eq!(realization.inflow, inflow);
    assert_eq!(realization.thermal_generation, thermal_generation);
    assert_eq!(realization.current_stage_objective, 150.0);
    assert_eq!(realization.total_stage_objective, 200.0);
}

// ============================================================================
// Test 7: StudyPeriodKind Variants
// ============================================================================
// Tests all three variants of StudyPeriodKind

#[test]
fn test_study_period_kind_variants() {
    let study = StudyPeriodKind::Study;
    let pre = StudyPeriodKind::PreStudy;
    let post = StudyPeriodKind::PostStudy;

    // Test that they're different
    assert!(study != pre);
    assert!(study != post);
    assert!(pre != post);

    // Test Clone
    let study_clone = study.clone();
    assert_eq!(study, study_clone);
}

// ============================================================================
// Test 8: Bus Operations
// ============================================================================
// Tests Bus helper methods

#[test]
fn test_bus_add_multiple_hydros() {
    let mut bus = Bus::new(0, 50.0);

    bus.add_hydro(0);
    bus.add_hydro(1);
    bus.add_hydro(2);

    assert_eq!(bus.hydro_ids.len(), 3);
    assert_eq!(bus.hydro_ids[0], 0);
    assert_eq!(bus.hydro_ids[2], 2);
}

#[test]
fn test_bus_add_multiple_thermals() {
    let mut bus = Bus::new(0, 50.0);

    bus.add_thermal(0);
    bus.add_thermal(1);

    assert_eq!(bus.thermal_ids.len(), 2);
}

#[test]
fn test_bus_add_multiple_lines() {
    let mut bus = Bus::new(0, 50.0);

    bus.add_source_line(0);
    bus.add_source_line(1);
    bus.add_target_line(2);
    bus.add_target_line(3);

    assert_eq!(bus.source_line_ids.len(), 2);
    assert_eq!(bus.target_line_ids.len(), 2);
}
