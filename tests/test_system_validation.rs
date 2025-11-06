use powers::system::{Bus, Hydro, Line, System, Thermal};

#[test]
fn test_default_system_is_valid() {
    let system = System::default();
    assert!(system.validate().is_ok());
}

#[test]
fn test_system_validation_duplicate_bus_ids() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(0, 60.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Duplicate bus ID: 0")));
}

#[test]
fn test_system_validation_duplicate_line_ids() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 50.0)];
    let lines = vec![
        Line::new(0, 0, 1, 100.0, 100.0, 0.0),
        Line::new(0, 1, 0, 100.0, 100.0, 0.0),
    ];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, lines, thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Duplicate line ID: 0")));
}

#[test]
fn test_system_validation_duplicate_thermal_ids() {
    let buses = vec![Bus::new(0, 50.0)];
    let thermals = vec![
        Thermal::new(0, 0, 5.0, 0.0, 15.0),
        Thermal::new(0, 0, 10.0, 0.0, 20.0),
    ];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Duplicate thermal ID: 0")));
}

#[test]
fn test_system_validation_duplicate_hydro_ids() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![
        Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
    ];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Duplicate hydro ID: 0")));
}

#[test]
fn test_system_validation_line_invalid_source_bus() {
    let buses = vec![Bus::new(0, 50.0)];
    let lines = vec![Line::new(0, 99, 0, 100.0, 100.0, 0.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, lines, thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("Line 0 references non-existent source bus 99")));
}

#[test]
fn test_system_validation_line_invalid_target_bus() {
    let buses = vec![Bus::new(0, 50.0)];
    let lines = vec![Line::new(0, 0, 99, 100.0, 100.0, 0.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, lines, thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("Line 0 references non-existent target bus 99")));
}

#[test]
fn test_system_validation_thermal_invalid_bus() {
    let buses = vec![Bus::new(0, 50.0)];
    let thermals = vec![Thermal::new(0, 99, 5.0, 0.0, 15.0)];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("Thermal 0 references non-existent bus 99")));
}

#[test]
fn test_system_validation_hydro_invalid_bus() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(0, None, 99, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("Hydro 0 references non-existent bus 99")));
}

#[test]
fn test_system_validation_hydro_invalid_downstream() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(
        0,
        Some(99),
        0,
        1.0,
        0.0,
        100.0,
        0.0,
        60.0,
        0.01,
    )];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("Hydro 0 references non-existent downstream hydro 99")));
}

#[test]
fn test_hydro_cascade_linear_is_valid() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 50.0)];
    let hydros = vec![
        Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(1, None, 1, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
    ];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_ok());
}

#[test]
fn test_hydro_cascade_branching_is_valid() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 50.0), Bus::new(2, 50.0)];
    let hydros = vec![
        Hydro::new(0, Some(2), 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(1, Some(2), 1, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(2, None, 2, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
    ];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_ok());
}

#[test]
fn test_hydro_cascade_cycle_detected() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 50.0)];
    let hydros = vec![
        Hydro::new(0, Some(1), 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
        Hydro::new(1, Some(0), 1, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
    ];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Cycle detected")));
}

#[test]
fn test_hydro_cascade_self_cycle_detected() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(
        0,
        Some(0),
        0,
        1.0,
        0.0,
        100.0,
        0.0,
        60.0,
        0.01,
    )];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("Cycle detected")));
}

#[test]
fn test_empty_buses_rejected() {
    let system = System::new(vec![], vec![], vec![], vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("System must have at least one bus")));
}

#[test]
fn test_no_generation_units_rejected() {
    let buses = vec![Bus::new(0, 50.0)];
    let system = System::new(buses, vec![], vec![], vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("System must have at least one hydro or thermal unit")));
}

#[test]
fn test_hydro_storage_capacity_constraint() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(0, None, 0, 1.0, 100.0, 50.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("min_storage") && e.contains("max_storage")));
}

#[test]
fn test_hydro_turbining_capacity_constraint() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 60.0, 30.0, 0.01)];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("min_turbined_flow") && e.contains("max_turbined_flow")));
}

#[test]
fn test_thermal_generation_capacity_constraint() {
    let buses = vec![Bus::new(0, 50.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 50.0, 10.0)];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("min_generation") && e.contains("max_generation")));
}

#[test]
fn test_negative_deficit_cost_rejected() {
    let buses = vec![Bus::new(0, -10.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("deficit_cost must be non-negative")));
}

#[test]
fn test_negative_thermal_cost_rejected() {
    let buses = vec![Bus::new(0, 50.0)];
    let thermals = vec![Thermal::new(0, 0, -5.0, 0.0, 15.0)];

    let system = System::new(buses, vec![], thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("cost must be non-negative")));
}

#[test]
fn test_negative_line_capacity_rejected() {
    let buses = vec![Bus::new(0, 50.0), Bus::new(1, 50.0)];
    let lines = vec![Line::new(0, 0, 1, -100.0, 100.0, 0.0)];
    let thermals = vec![Thermal::new(0, 0, 5.0, 0.0, 15.0)];

    let system = System::new(buses, lines, thermals, vec![]);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("direct_capacity must be non-negative")));
}

#[test]
fn test_zero_hydro_productivity_rejected() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(0, None, 0, 0.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("productivity must be positive")));
}

#[test]
fn test_negative_hydro_productivity_rejected() {
    let buses = vec![Bus::new(0, 50.0)];
    let hydros = vec![Hydro::new(0, None, 0, -1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors
        .iter()
        .any(|e| e.contains("productivity must be positive")));
}

#[test]
fn test_multiple_validation_errors_reported() {
    let buses = vec![Bus::new(0, -50.0), Bus::new(0, 60.0)];
    let hydros = vec![
        Hydro::new(0, None, 99, -1.0, 100.0, 50.0, 0.0, 60.0, 0.01),
        Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01),
    ];

    let system = System::new(buses, vec![], vec![], hydros);
    let result = system.validate();

    assert!(result.is_err());
    let errors = result.unwrap_err();
    
    // Should have multiple errors
    assert!(errors.len() >= 4);
    assert!(errors.iter().any(|e| e.contains("Duplicate bus ID")));
    assert!(errors.iter().any(|e| e.contains("deficit_cost")));
    assert!(errors.iter().any(|e| e.contains("Duplicate hydro ID")));
    assert!(errors.iter().any(|e| e.contains("non-existent bus")));
}

#[test]
fn test_validation_performance() {
    // Create a moderately sized system
    let mut buses = Vec::new();
    let mut hydros = Vec::new();
    let mut thermals = Vec::new();

    for i in 0..50 {
        buses.push(Bus::new(i, 50.0));
    }

    for i in 0..30 {
        hydros.push(Hydro::new(
            i,
            if i > 0 { Some(i - 1) } else { None },
            i % 50,
            1.0,
            0.0,
            100.0,
            0.0,
            60.0,
            0.01,
        ));
    }

    for i in 0..20 {
        thermals.push(Thermal::new(i, i % 50, 5.0, 0.0, 15.0));
    }

    let system = System::new(buses, vec![], thermals, hydros);

    let start = std::time::Instant::now();
    let result = system.validate();
    let duration = start.elapsed();

    assert!(result.is_ok());
    assert!(
        duration.as_millis() < 10,
        "Validation took {}ms, expected <10ms",
        duration.as_millis()
    );
}
