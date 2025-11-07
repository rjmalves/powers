// Physical validation utilities for hydro-thermal systems
//
// Validates conservation laws and physical constraints:
// - Water balance (mass conservation)
// - Power balance (Kirchhoff's law)
// - Physical bounds and feasibility

/// Assert water balance for a single hydro plant
///
/// Validates mass conservation: storage change = inflow - outflow
/// where outflow = turbining + spillage
///
/// # Arguments
/// - `initial_storage`: Storage at start of period
/// - `final_storage`: Storage at end of period
/// - `inflow`: Natural inflow during period
/// - `turbining`: Water turbined for generation
/// - `spillage`: Water spilled (not used for generation)
/// - `tolerance`: Tolerance for numerical errors (typically 1e-6)
///
/// # Panics
/// If water balance violated beyond tolerance
///
/// # Conservation Law
/// ```text
/// final_storage = initial_storage + inflow - turbining - spillage
/// ```
///
/// # Example
/// ```
/// assert_water_balance(
///     50.0,  // initial storage
///     45.0,  // final storage
///     10.0,  // inflow
///     12.0,  // turbining
///     3.0,   // spillage
///     1e-6   // tolerance
/// );
/// // Check: 45 = 50 + 10 - 12 - 3 ✓
/// ```
#[track_caller]
pub fn assert_water_balance(
    initial_storage: f64,
    final_storage: f64,
    inflow: f64,
    turbining: f64,
    spillage: f64,
    tolerance: f64,
) {
    let expected_final = initial_storage + inflow - turbining - spillage;
    let error = (final_storage - expected_final).abs();

    if error > tolerance {
        panic!(
            "Water balance violated!\n  \
             Initial storage: {:.6}\n  \
             Inflow:          {:.6}\n  \
             Turbining:       {:.6}\n  \
             Spillage:        {:.6}\n  \
             Expected final:  {:.6}\n  \
             Actual final:    {:.6}\n  \
             Error:           {:.6} (tolerance: {:.6})",
            initial_storage,
            inflow,
            turbining,
            spillage,
            expected_final,
            final_storage,
            error,
            tolerance
        );
    }
}

/// Assert water balance for a cascade of hydro plants
///
/// Validates that upstream turbining and spillage become downstream inflow.
/// Accounts for natural inflow at each plant.
///
/// # Arguments
/// - `hydros`: Vector of hydro data (initial, final, inflow, turbining, spillage)
/// - `downstream_connections`: Map of (plant_idx -> downstream_plant_idx)
/// - `tolerance`: Tolerance for numerical errors
///
/// # Panics
/// If any plant's water balance is violated
///
/// # Example
/// ```
/// // Two-plant cascade: plant 0 flows into plant 1
/// let hydros = vec![
///     (100.0, 95.0, 10.0, 12.0, 3.0),  // upstream
///     (80.0, 90.0, 5.0, 10.0, 0.0),     // downstream
/// ];
/// let connections = vec![(0, 1)];  // 0 → 1
/// assert_cascade_water_balance(&hydros, &connections, 1e-6);
/// // Downstream inflow should be natural (5) + upstream outflow (15) = 20
/// ```
#[track_caller]
pub fn assert_cascade_water_balance(
    hydros: &[(f64, f64, f64, f64, f64)], // (init, final, inflow, turb, spill)
    downstream_connections: &[(usize, usize)],
    tolerance: f64,
) {
    // First check individual balances
    for &(init, final_s, inflow, turb, spill) in hydros.iter() {
        assert_water_balance(init, final_s, inflow, turb, spill, tolerance);
    }

    // Then check cascade connections
    for &(upstream_idx, downstream_idx) in downstream_connections {
        let (_, _, _, up_turb, up_spill) = hydros[upstream_idx];
        let (_, _, down_inflow, _, _) = hydros[downstream_idx];

        let upstream_outflow = up_turb + up_spill;

        // Downstream inflow should include upstream outflow
        // (Note: This is a simplified model; real systems may have travel time)
        if down_inflow < upstream_outflow - tolerance {
            panic!(
                "Cascade flow inconsistency!\n  \
                 Upstream plant {} releases {:.6} (turb + spill)\n  \
                 Downstream plant {} receives {:.6} inflow\n  \
                 Missing water: {:.6}",
                upstream_idx,
                upstream_outflow,
                downstream_idx,
                down_inflow,
                upstream_outflow - down_inflow
            );
        }
    }
}

/// Assert power balance at a bus
///
/// Validates Kirchhoff's law: generation + imports = demand + exports + deficit
///
/// # Arguments
/// - `generation`: Total generation at bus (hydro + thermal)
/// - `demand`: Load demand at bus
/// - `net_transmission`: Net power flow (imports - exports)
/// - `deficit`: Unmet demand (penalty generation)
/// - `tolerance`: Tolerance for numerical errors
///
/// # Panics
/// If power balance violated beyond tolerance
///
/// # Conservation Law
/// ```text
/// generation + net_transmission = demand + deficit
/// ```
///
/// # Example
/// ```
/// assert_power_balance(
///     150.0,  // generation (hydro + thermal)
///     140.0,  // demand
///     0.0,    // no transmission
///     10.0,   // deficit (generation shortfall)
///     1e-6
/// );
/// ```
#[track_caller]
pub fn assert_power_balance(
    generation: f64,
    demand: f64,
    net_transmission: f64,
    deficit: f64,
    tolerance: f64,
) {
    let supply = generation + net_transmission;
    let required = demand + deficit;
    let error = (supply - required).abs();

    if error > tolerance {
        panic!(
            "Power balance violated!\n  \
             Generation:       {:.6}\n  \
             Net transmission: {:.6}\n  \
             Total supply:     {:.6}\n  \
             Demand:           {:.6}\n  \
             Deficit:          {:.6}\n  \
             Total required:   {:.6}\n  \
             Imbalance:        {:.6} (tolerance: {:.6})",
            generation,
            net_transmission,
            supply,
            demand,
            deficit,
            required,
            error,
            tolerance
        );
    }
}

/// Assert that a value is within physical bounds
///
/// Checks that a physical quantity (storage, generation, etc.) respects its limits.
///
/// # Arguments
/// - `value`: Value to check
/// - `min_bound`: Minimum physical limit
/// - `max_bound`: Maximum physical limit  
/// - `tolerance`: Tolerance for constraint violations
/// - `name`: Name for error messages
///
/// # Panics
/// If value violates bounds beyond tolerance
#[track_caller]
pub fn assert_physical_bounds(
    value: f64,
    min_bound: f64,
    max_bound: f64,
    tolerance: f64,
    name: &str,
) {
    if value < min_bound - tolerance {
        panic!(
            "{} = {:.6} violates lower bound {:.6} (violation: {:.6}, tolerance: {:.6})",
            name,
            value,
            min_bound,
            min_bound - value,
            tolerance
        );
    }

    if value > max_bound + tolerance {
        panic!(
            "{} = {:.6} violates upper bound {:.6} (violation: {:.6}, tolerance: {:.6})",
            name,
            value,
            max_bound,
            value - max_bound,
            tolerance
        );
    }
}

/// Check if water balance holds (without panicking)
///
/// Returns true if balance is satisfied within tolerance.
///
/// # Returns
/// `true` if balanced, `false` otherwise
pub fn is_water_balanced(
    initial_storage: f64,
    final_storage: f64,
    inflow: f64,
    turbining: f64,
    spillage: f64,
    tolerance: f64,
) -> bool {
    let expected_final = initial_storage + inflow - turbining - spillage;
    (final_storage - expected_final).abs() <= tolerance
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_water_balance_pass() {
        // 100 + 20 - 15 - 5 = 100
        assert_water_balance(100.0, 100.0, 20.0, 15.0, 5.0, 1e-6);
    }

    #[test]
    fn test_water_balance_with_storage_change() {
        // 100 + 20 - 10 - 5 = 105 (storage increases)
        assert_water_balance(100.0, 105.0, 20.0, 10.0, 5.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "Water balance violated")]
    fn test_water_balance_fail() {
        // 100 + 20 - 15 - 5 = 100, but claiming final is 95
        assert_water_balance(100.0, 95.0, 20.0, 15.0, 5.0, 1e-6);
    }

    #[test]
    fn test_cascade_water_balance() {
        // Upstream: 100 + 10 - 12 - 3 = 95, releases 15
        // Downstream: 80 + (5 natural + 15 from upstream) - 10 - 0 = 90
        let hydros = vec![
            (100.0, 95.0, 10.0, 12.0, 3.0), // upstream
            (80.0, 90.0, 20.0, 10.0, 0.0), // downstream (20 = 5 natural + 15 from upstream)
        ];
        let connections = vec![(0, 1)];
        assert_cascade_water_balance(&hydros, &connections, 1e-6);
    }

    #[test]
    fn test_power_balance_pass() {
        // 150 generation + 0 transmission = 140 demand + 10 deficit
        assert_power_balance(150.0, 140.0, 0.0, 10.0, 1e-6);
    }

    #[test]
    fn test_power_balance_with_transmission() {
        // 100 gen + 50 import = 140 demand + 10 deficit
        assert_power_balance(100.0, 140.0, 50.0, 10.0, 1e-6);
    }

    #[test]
    #[should_panic(expected = "Power balance violated")]
    fn test_power_balance_fail() {
        // 100 + 0 ≠ 150 + 0
        assert_power_balance(100.0, 150.0, 0.0, 0.0, 1e-6);
    }

    #[test]
    fn test_physical_bounds_pass() {
        assert_physical_bounds(50.0, 0.0, 100.0, 1e-6, "storage");
        assert_physical_bounds(0.0, 0.0, 100.0, 1e-6, "boundary");
        assert_physical_bounds(100.0, 0.0, 100.0, 1e-6, "boundary");
    }

    #[test]
    #[should_panic(expected = "violates lower bound")]
    fn test_physical_bounds_below_fail() {
        assert_physical_bounds(-1.0, 0.0, 100.0, 1e-6, "storage");
    }

    #[test]
    #[should_panic(expected = "violates upper bound")]
    fn test_physical_bounds_above_fail() {
        assert_physical_bounds(101.0, 0.0, 100.0, 1e-6, "storage");
    }

    #[test]
    fn test_is_water_balanced_check() {
        assert!(is_water_balanced(100.0, 100.0, 20.0, 15.0, 5.0, 1e-6));
        assert!(!is_water_balanced(100.0, 95.0, 20.0, 15.0, 5.0, 1e-6));
    }
}
