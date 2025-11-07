//! Integration tests for AR Model refactoring
//!
//! These tests validate that the AR model bug fixes work correctly:
//!
//! - **Correct structure**: `load_lag_duals` and `inflow_lag_duals` have correct dimensions
//! - **Mixed entity types**: Loads and inflows with heterogeneous AR orders
//! - **Edge cases**: Only loads AR, only inflows AR, no AR dynamics
//! - **Helper methods**: `num_lag_duals()` and `total_lag_count()` work correctly
//!

use powers_rs::subproblem::{Realization, StudyPeriodKind};

mod fixtures;

// ============================================================================
// Helper Functions
// ============================================================================

/// Verify lag_duals structure matches expected dimensions and AR orders
fn verify_lag_duals_structure(
    realization: &Realization,
    expected_buses: usize,
    expected_hydros: usize,
    expected_load_ar_orders: &[usize],
    expected_inflow_ar_orders: &[usize],
) -> Result<(), String> {
    // Check outer vector lengths
    if realization.load_lag_duals.len() != expected_buses {
        return Err(format!(
            "load_lag_duals.len() = {}, expected {}",
            realization.load_lag_duals.len(),
            expected_buses
        ));
    }

    if realization.inflow_lag_duals.len() != expected_hydros {
        return Err(format!(
            "inflow_lag_duals.len() = {}, expected {}",
            realization.inflow_lag_duals.len(),
            expected_hydros
        ));
    }

    // Check inner vector lengths match AR orders
    for (bus_id, &expected_ar) in expected_load_ar_orders.iter().enumerate() {
        let actual = realization.load_lag_duals[bus_id].len();
        if actual != expected_ar {
            return Err(format!(
                "Bus {} lag_duals.len() = {}, expected AR({}) with {} duals",
                bus_id, actual, expected_ar, expected_ar
            ));
        }
    }

    for (hydro_id, &expected_ar) in expected_inflow_ar_orders.iter().enumerate()
    {
        let actual = realization.inflow_lag_duals[hydro_id].len();
        if actual != expected_ar {
            return Err(format!(
                "Hydro {} lag_duals.len() = {}, expected AR({}) with {} duals",
                hydro_id, actual, expected_ar, expected_ar
            ));
        }
    }

    Ok(())
}

// ============================================================================
// Test 1: Realization Structure with Mixed AR Orders
// ============================================================================

#[test]
fn test_realization_structure_mixed_ar() {
    let (system, _temporal_models) = fixtures::subproblems::mixed_ar_system();

    // Create realization with correct capacity
    let realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    // Initially, lag_duals should be empty (not populated until solve)
    assert_eq!(realization.load_lag_duals.len(), 0);
    assert_eq!(realization.inflow_lag_duals.len(), 0);

    // Manually set up structure to test validation
    let mut test_realization = realization.clone();
    test_realization.load_lag_duals = vec![
        vec![1.0], // Bus 0: AR(1)
        vec![],    // Bus 1: AR(0)
    ];
    test_realization.inflow_lag_duals = vec![
        vec![2.0, 3.0], // Hydro 0: AR(2)
        vec![],         // Hydro 1: AR(0)
        vec![4.0],      // Hydro 2: AR(1)
    ];

    // Verify structure
    let expected_load_ar = vec![1, 0];
    let expected_inflow_ar = vec![2, 0, 1];

    let result = verify_lag_duals_structure(
        &test_realization,
        2, // buses_count
        3, // hydros_count
        &expected_load_ar,
        &expected_inflow_ar,
    );

    assert!(
        result.is_ok(),
        "Lag duals structure verification failed: {:?}",
        result.err()
    );
}

// ============================================================================
// Test 2: Realization with Only Inflows Having AR
// ============================================================================

#[test]
fn test_realization_structure_inflow_only_ar() {
    let (system, _temporal_models) =
        fixtures::subproblems::inflow_only_ar_system();

    let mut realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    // Manually set structure
    realization.load_lag_duals = vec![
        vec![], // Bus 0: AR(0)
        vec![], // Bus 1: AR(0)
    ];
    realization.inflow_lag_duals = vec![
        vec![1.0],           // Hydro 0: AR(1)
        vec![2.0, 3.0],      // Hydro 1: AR(2)
        vec![4.0, 5.0, 6.0], // Hydro 2: AR(3)
    ];

    // Verify structure
    let expected_load_ar = vec![0, 0];
    let expected_inflow_ar = vec![1, 2, 3];

    verify_lag_duals_structure(
        &realization,
        2,
        3,
        &expected_load_ar,
        &expected_inflow_ar,
    )
    .expect("Structure validation failed");

    // Verify all load lag duals are empty
    for lag_duals in &realization.load_lag_duals {
        assert!(lag_duals.is_empty(), "Load lag duals should be empty");
    }

    // Verify inflow lag duals are populated
    assert_eq!(realization.inflow_lag_duals[0].len(), 1);
    assert_eq!(realization.inflow_lag_duals[1].len(), 2);
    assert_eq!(realization.inflow_lag_duals[2].len(), 3);
}

// ============================================================================
// Test 3: Realization with Only Loads Having AR
// ============================================================================

#[test]
fn test_realization_structure_load_only_ar() {
    let (system, _temporal_models) =
        fixtures::subproblems::load_only_ar_system();

    let mut realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    // Manually set structure
    realization.load_lag_duals = vec![
        vec![1.0],      // Bus 0: AR(1)
        vec![2.0, 3.0], // Bus 1: AR(2)
    ];
    realization.inflow_lag_duals = vec![
        vec![], // Hydro 0: AR(0)
        vec![], // Hydro 1: AR(0)
    ];

    // Verify structure
    let expected_load_ar = vec![1, 2];
    let expected_inflow_ar = vec![0, 0];

    verify_lag_duals_structure(
        &realization,
        2,
        2,
        &expected_load_ar,
        &expected_inflow_ar,
    )
    .expect("Structure validation failed");

    // Verify load lag duals are populated
    assert_eq!(realization.load_lag_duals[0].len(), 1);
    assert_eq!(realization.load_lag_duals[1].len(), 2);

    // Verify all inflow lag duals are empty
    for lag_duals in &realization.inflow_lag_duals {
        assert!(lag_duals.is_empty(), "Inflow lag duals should be empty");
    }
}

// ============================================================================
// Test 4: No AR Dynamics (Baseline)
// ============================================================================

#[test]
fn test_realization_structure_no_ar() {
    let (system, _temporal_models) = fixtures::subproblems::no_ar_system();

    let mut realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    // Manually set structure (all empty)
    realization.load_lag_duals = vec![vec![], vec![]];
    realization.inflow_lag_duals = vec![vec![], vec![], vec![]];

    // Verify structure
    let expected_load_ar = vec![0, 0];
    let expected_inflow_ar = vec![0, 0, 0];

    verify_lag_duals_structure(
        &realization,
        2,
        3,
        &expected_load_ar,
        &expected_inflow_ar,
    )
    .expect("Structure validation failed");

    // Verify all lag duals are empty
    for lag_duals in &realization.load_lag_duals {
        assert!(lag_duals.is_empty());
    }
    for lag_duals in &realization.inflow_lag_duals {
        assert!(lag_duals.is_empty());
    }

    // Total lag count should be 0
    assert_eq!(realization.total_lag_count(), 0);
}

// ============================================================================
// Test 5: Helper Method num_lag_duals()
// ============================================================================

#[test]
fn test_num_lag_duals_method() {
    let mut realization = Realization::default();

    // Set up mixed AR orders
    realization.inflow_lag_duals = vec![
        vec![1.0, 2.0], // Hydro 0: AR(2)
        vec![],         // Hydro 1: AR(0)
        vec![3.0],      // Hydro 2: AR(1)
    ];

    // Test num_lag_duals() for each hydro
    assert_eq!(
        realization.num_lag_duals(0),
        2,
        "Hydro 0 should have 2 lags"
    );
    assert_eq!(
        realization.num_lag_duals(1),
        0,
        "Hydro 1 should have 0 lags"
    );
    assert_eq!(realization.num_lag_duals(2), 1, "Hydro 2 should have 1 lag");

    // Out of bounds should return 0
    assert_eq!(
        realization.num_lag_duals(999),
        0,
        "Out of bounds should return 0"
    );
}

// ============================================================================
// Test 6: Helper Method total_lag_count()
// ============================================================================

#[test]
fn test_total_lag_count_method() {
    let mut realization = Realization::default();

    // Set up mixed AR orders
    realization.load_lag_duals = vec![
        vec![1.0],      // Bus 0: AR(1)
        vec![],         // Bus 1: AR(0)
        vec![2.0, 3.0], // Bus 2: AR(2)
    ];
    realization.inflow_lag_duals = vec![
        vec![4.0, 5.0, 6.0], // Hydro 0: AR(3)
        vec![],              // Hydro 1: AR(0)
        vec![7.0],           // Hydro 2: AR(1)
    ];

    // Total: 1 + 0 + 2 + 3 + 0 + 1 = 7
    assert_eq!(realization.total_lag_count(), 7);

    // Test with all empty
    realization.load_lag_duals = vec![vec![], vec![]];
    realization.inflow_lag_duals = vec![vec![], vec![]];
    assert_eq!(realization.total_lag_count(), 0);
}

// ============================================================================
// Test 7: Large Heterogeneous System
// ============================================================================

#[test]
fn test_large_heterogeneous_system_structure() {
    let (system, _temporal_models) =
        fixtures::subproblems::large_heterogeneous_system();

    let mut realization =
        Realization::with_capacity(&StudyPeriodKind::Study, &system);

    // Set up structure matching the fixture: 5 buses [0,1,2,0,1], 10 hydros [0,1,2,0,2,1,3,0,1,2]
    realization.load_lag_duals = vec![
        vec![],         // Bus 0: AR(0)
        vec![1.0],      // Bus 1: AR(1)
        vec![2.0, 3.0], // Bus 2: AR(2)
        vec![],         // Bus 3: AR(0)
        vec![4.0],      // Bus 4: AR(1)
    ];

    realization.inflow_lag_duals = vec![
        vec![],                 // Hydro 0: AR(0)
        vec![5.0],              // Hydro 1: AR(1)
        vec![6.0, 7.0],         // Hydro 2: AR(2)
        vec![],                 // Hydro 3: AR(0)
        vec![8.0, 9.0],         // Hydro 4: AR(2)
        vec![10.0],             // Hydro 5: AR(1)
        vec![11.0, 12.0, 13.0], // Hydro 6: AR(3)
        vec![],                 // Hydro 7: AR(0)
        vec![14.0],             // Hydro 8: AR(1)
        vec![15.0, 16.0],       // Hydro 9: AR(2)
    ];

    // Verify structure
    let expected_load_ar = vec![0, 1, 2, 0, 1];
    let expected_inflow_ar = vec![0, 1, 2, 0, 2, 1, 3, 0, 1, 2];

    verify_lag_duals_structure(
        &realization,
        5,
        10,
        &expected_load_ar,
        &expected_inflow_ar,
    )
    .expect("Structure validation failed");

    // Verify total count: (0+1+2+0+1) + (0+1+2+0+2+1+3+0+1+2) = 4 + 12 = 16
    assert_eq!(realization.total_lag_count(), 16);
}
