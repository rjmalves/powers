// Integration tests for the test utility library
//
// This file tests the test utilities to ensure they work correctly

mod utils;

use utils::{
    are_storage_coefficients_negative, assert_cut_validity,
    assert_monotonic_non_decreasing, assert_physical_bounds,
    assert_power_balance, assert_water_balance,
};

#[test]
fn test_monotonic_utilities() {
    // Test monotonic assertion
    let values = vec![1.0, 2.0, 2.5, 3.0];
    assert_monotonic_non_decreasing(&values, 1e-6);

    // Test with empty vector
    assert_monotonic_non_decreasing(&[], 1e-6);
}

#[test]
#[should_panic(expected = "Monotonicity violated")]
fn test_monotonic_violation() {
    let values = vec![1.0, 3.0, 2.0]; // Decreases
    assert_monotonic_non_decreasing(&values, 1e-10);
}

#[test]
fn test_water_balance_utilities() {
    // Test balanced case
    assert_water_balance(
        100.0, // initial
        95.0,  // final
        10.0,  // inflow
        12.0,  // turbining
        3.0,   // spillage
        1e-6,
    );
}

#[test]
#[should_panic(expected = "Water balance violated")]
fn test_water_balance_violation() {
    assert_water_balance(
        100.0, // initial
        100.0, // final - but should be 95.0
        10.0,  // inflow
        12.0,  // turbining
        3.0,   // spillage
        1e-6,
    );
}

#[test]
fn test_power_balance_utilities() {
    // Test balanced case
    assert_power_balance(
        150.0, // generation
        140.0, // demand
        0.0,   // net transmission
        10.0,  // deficit
        1e-6,
    );
}

#[test]
fn test_physical_bounds_utilities() {
    // Test within bounds
    assert_physical_bounds(50.0, 0.0, 100.0, 1e-6, "storage");
    assert_physical_bounds(0.0, 0.0, 100.0, 1e-6, "lower boundary");
    assert_physical_bounds(100.0, 0.0, 100.0, 1e-6, "upper boundary");
}

#[test]
fn test_cut_validation_utilities() {
    use powers_rs::cut::BendersCut;

    // Create a simple cut
    let cut = BendersCut::new(
        0,          // id
        vec![-2.0], // coefficients
        100.0,      // rhs
        1,          // iteration
        0,          // forward_pass_idx
    );

    // At state s=10: height = 100 + (-2 * 10) = 80
    let state = vec![10.0];
    let objective = 80.0;

    // This should pass
    assert_cut_validity(&cut, &state, objective, 1e-4);

    // Storage coefficients should be non-positive
    assert!(are_storage_coefficients_negative(&vec![-2.0], 1e-8));
    assert!(!are_storage_coefficients_negative(&vec![2.0], 1e-8));
}

#[test]
fn test_all_utilities_integration() {
    // This test ensures all utilities work together

    // 1. Test monotonic lower bounds (typical SDDP scenario)
    let lower_bounds = vec![100.0, 105.3, 108.1, 110.2];
    assert_monotonic_non_decreasing(&lower_bounds, 1e-6);

    // 2. Test water balance in a hydro system
    assert_water_balance(50.0, 48.0, 5.0, 6.0, 1.0, 1e-6);

    // 3. Test power balance
    assert_power_balance(100.0, 90.0, 0.0, 10.0, 1e-6);

    // 4. Test physical bounds
    assert_physical_bounds(75.0, 0.0, 100.0, 1e-6, "storage");
}
