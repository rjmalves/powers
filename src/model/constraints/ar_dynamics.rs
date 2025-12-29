//! AR dynamics constraints for uncertainty modeling.
//!
//! This module provides constraint builders for auto-regressive (AR) dynamics:
//!
//! 1. **Observation constraints**: `Y[t] - Σ(ψ_k · Y[t-k]) = base + σ·η`
//! 2. **Lag fixing constraints**: `Y[t-k] = value` (RHS updated per scenario)
//!
//! These constraints implement the PAR (Periodic Auto-Regressive) model
//! used for modeling uncertain inflows and loads.

use crate::input::UncertaintyType;
use crate::solver::Problem;
use crate::subproblem::{InflowLagConstraints, LoadLagConstraints, Variables};
use crate::temporal_model::TemporalModel;

/// Build uncertainty observation constraints.
///
/// Creates one constraint per uncertain entity (load or inflow) with the form:
/// ```text
/// Y[t] - Σ(ψ_k · Y[t-k]) = deterministic_base + σ·η
/// ```
///
/// Initially created with RHS=0. The actual RHS is computed and updated
/// during `realize_uncertainties` using the innovation values from SAA.
///
/// # Arguments
///
/// * `pb` - The LP problem builder
/// * `variables` - Variable indices for the LP model
/// * `temporal_models` - Temporal models for all uncertain entities
/// * `season_id` - Current season (for seasonal AR coefficients)
///
/// # Returns
///
/// Vector of constraint indices (one per entity, following `temporal_models` order).
pub fn build_uncertainty_observation_constraints(
    pb: &mut Problem,
    variables: &Variables,
    temporal_models: &[TemporalModel],
    season_id: usize,
) -> Vec<usize> {
    let mut constraint_indices = Vec::new();
    let mut load_idx = 0;
    let mut inflow_idx = 0;

    for (global_idx, model) in temporal_models.iter().enumerate() {
        // Get the observation variable for this entity
        let observation_var = match model.entity_type {
            UncertaintyType::Load => {
                let var = variables.load[load_idx];
                load_idx += 1;
                var
            }
            UncertaintyType::Inflow => {
                let var = variables.inflow[inflow_idx];
                inflow_idx += 1;
                var
            }
        };

        let mut factors = vec![(observation_var, 1.0)];

        // Add lag variables to the constraint with negative psi coefficients
        // Constraint: Y[t] - Σ(ψ_k · Y[t-k]) = deterministic_base + σ·η
        if let Some(ref lag_vars) = variables.lagged_state {
            let entity_lag_vars = &lag_vars[global_idx];
            let psi_coeffs = &model.psi_coefficients[season_id];

            for (lag_idx, &lag_var) in entity_lag_vars.iter().enumerate() {
                if lag_idx < psi_coeffs.len() {
                    let psi = psi_coeffs[lag_idx];
                    factors.push((lag_var, -psi));
                }
            }
        }

        // Initially RHS=0, will be updated in realize_uncertainties
        let row = pb.add_row(0.0..=0.0, &factors);
        constraint_indices.push(row);
    }

    constraint_indices
}

/// Build lag-fixing constraints for AR state variables.
///
/// Creates constraints that fix lag variables to their historical values:
/// ```text
/// Y[t-k] = value
/// ```
///
/// The RHS is initially 0 and updated during `realize_uncertainties`
/// with actual historical observation values.
///
/// # Arguments
///
/// * `pb` - The LP problem builder
/// * `variables` - Variable indices (must have `lagged_state`)
/// * `temporal_models` - Temporal models for entity type routing
/// * `buses_count` - Number of buses in the system
/// * `hydros_count` - Number of hydros in the system
///
/// # Returns
///
/// Tuple of (load_lag_constraints, inflow_lag_constraints), each `Option`.
/// Returns `None` if no constraints of that type exist.
pub fn build_lag_fixing_constraints(
    pb: &mut Problem,
    variables: &Variables,
    temporal_models: &[TemporalModel],
    buses_count: usize,
    hydros_count: usize,
) -> (Option<LoadLagConstraints>, Option<InflowLagConstraints>) {
    let lag_vars = match &variables.lagged_state {
        Some(lv) => lv,
        None => return (None, None),
    };

    let mut load_constraints = LoadLagConstraints::new(buses_count);
    let mut inflow_constraints = InflowLagConstraints::new(hydros_count);

    for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
        let mut entity_constraints = Vec::new();

        for &var in entity_lags {
            // Constraint: Y[t-k] = 0.0 (RHS updated in realize_uncertainties)
            let constraint = pb.add_row(0.0..=0.0, vec![(var, 1.0)]);
            entity_constraints.push(constraint);
        }

        // Route to appropriate structure based on entity type
        let model = &temporal_models[entity_idx];
        match model.entity_type {
            UncertaintyType::Load => {
                let bus_id = model.entity_id;
                load_constraints.constraints_by_bus[bus_id] =
                    entity_constraints;
            }
            UncertaintyType::Inflow => {
                let hydro_id = model.entity_id;
                inflow_constraints.constraints_by_hydro[hydro_id] =
                    entity_constraints;
            }
        }
    }

    // Convert to Option (None if empty)
    let load_opt = if load_constraints.total_constraint_count() > 0 {
        Some(load_constraints)
    } else {
        None
    };

    let inflow_opt = if inflow_constraints.total_constraint_count() > 0 {
        Some(inflow_constraints)
    } else {
        None
    };

    (load_opt, inflow_opt)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_compiles() {
        // Placeholder to ensure module compiles correctly
        assert!(true);
    }
}
