//! Hydro balance (water balance) constraints.
//!
//! Water balance at each hydro plant:
//! ```text
//! stored_volume + turbined_flow + spillage = inflow + Σ(upstream_outflows)
//! ```
//!
//! Where:
//! - `stored_volume` = reservoir level at end of period
//! - `turbined_flow` = water passing through turbines
//! - `spillage` = water spilled (not used for generation)
//! - `inflow` = natural inflow observation
//! - `upstream_outflows` = turbined_flow + spillage from upstream hydros

use crate::solver::Problem;
use crate::subproblem::Variables;
use crate::system::System;

/// Build hydro balance (water balance) constraints.
///
/// Creates one constraint per hydro plant with the form:
/// ```text
/// stored_volume + turbined_flow + spillage - inflow - Σ(upstream_outflows) = 0
/// ```
///
/// The RHS is initially 0. The initial storage contribution is added during
/// state preparation via the state coefficients.
///
/// # Arguments
///
/// * `pb` - The LP problem builder
/// * `variables` - Variable indices for the LP model
/// * `system` - System definition with hydro topology (cascade structure)
///
/// # Returns
///
/// Vector of constraint indices, one per hydro (indexed by `hydro_id`).
pub fn build_hydro_balance_constraints(
    pb: &mut Problem,
    variables: &Variables,
    system: &System,
) -> Vec<usize> {
    let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];

    for hydro in system.hydros.iter() {
        let mut factors: Vec<(usize, f64)> = vec![
            (variables.stored_volume[hydro.id], 1.0),
            (variables.turbined_flow[hydro.id], 1.0),
            (variables.spillage[hydro.id], 1.0),
        ];

        // Add inflow variable if this hydro has inflow modeled
        if hydro.id < variables.inflow.len() {
            factors.push((variables.inflow[hydro.id], -1.0));
        }

        // Add upstream contributions (negative because they add to available water)
        for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
            factors.push((variables.turbined_flow[*upstream_hydro_id], -1.0));
            factors.push((variables.spillage[*upstream_hydro_id], -1.0));
        }

        hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
    }

    hydro_balance
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Placeholder to ensure module compiles correctly
        assert!(true);
    }
}
