//! Bus balance (load balance) constraints.
//!
//! Power balance at each network bus:
//! ```text
//! deficit + Σ(thermal_gen) + Σ(hydro_gen) + net_imports = load
//! ```
//!
//! Where:
//! - `deficit` = unmet load at the bus
//! - `thermal_gen` = generation from thermal plants at the bus
//! - `hydro_gen` = productivity × turbined_flow for hydros at the bus
//! - `net_imports` = imports - exports via transmission lines

use crate::solver::Problem;
use crate::subproblem::Variables;
use crate::system::System;

/// Build bus balance (load balance) constraints.
///
/// Creates one constraint per bus with the form:
/// ```text
/// deficit + Σ(thermal_gen) + Σ(hydro_gen × productivity) + net_exchange - load = 0
/// ```
///
/// # Arguments
///
/// * `pb` - The LP problem builder
/// * `variables` - Variable indices for the LP model
/// * `system` - System definition with bus/generator topology
///
/// # Returns
///
/// Vector of constraint indices, one per bus (indexed by `bus_id`).
pub fn build_bus_balance_constraints(
    pb: &mut Problem,
    variables: &Variables,
    system: &System,
) -> Vec<usize> {
    let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];

    for bus in system.buses.iter() {
        let mut factors = vec![
            (variables.deficit[bus.id], 1.0),
            (variables.load[bus.id], -1.0),
        ];

        // Add thermal generators at this bus
        for thermal_id in bus.thermal_ids.iter() {
            factors.push((variables.thermal_gen[*thermal_id], 1.0));
        }

        // Add hydro generators at this bus (with productivity factor)
        for hydro_id in bus.hydro_ids.iter() {
            factors.push((
                variables.turbined_flow[*hydro_id],
                system.hydros.get(*hydro_id).unwrap().productivity,
            ));
        }

        // Add transmission lines where this bus is the source (outgoing)
        for line_id in bus.source_line_ids.iter() {
            factors.push((variables.reverse_exchange[*line_id], 1.0));
            factors.push((variables.direct_exchange[*line_id], -1.0));
        }

        // Add transmission lines where this bus is the target (incoming)
        for line_id in bus.target_line_ids.iter() {
            factors.push((variables.direct_exchange[*line_id], 1.0));
            factors.push((variables.reverse_exchange[*line_id], -1.0));
        }

        load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
    }

    load_balance
}

#[cfg(test)]
mod tests {
    // Integration tests would require mock System and Problem objects.
    // Unit tests are limited without access to actual types.

    #[test]
    fn test_module_compiles() {
        // Placeholder to ensure module compiles correctly
        assert!(true);
    }
}
