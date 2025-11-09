//! Arrow schemas for SDDP output types.
//!
//! Defines the columnar structure for each output file type.
//! Schemas optimize for:
//! - Memory efficiency (appropriate types for value ranges)
//! - Compression (similar values in columns compress better)
//! - Query performance (columnar access patterns)

use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Schema for training convergence data (training.csv equivalent).
///
/// Contains per-iteration metrics: bounds, costs, gaps, timings.
pub fn training_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("lower_bound", DataType::Float64, false),
        Field::new("policy_cost", DataType::Float64, false),
        Field::new("forward_time_ms", DataType::UInt64, false),
        Field::new("gap", DataType::Float64, false),
        Field::new("forward_passes", DataType::UInt16, false),
        Field::new("forward_scenarios", DataType::UInt32, false),
        Field::new("forward_solver_ms", DataType::UInt64, false),
        Field::new("forward_avg_objective", DataType::Float64, false),
        Field::new("forward_std_objective", DataType::Float64, false),
        Field::new("backward_time_ms", DataType::UInt64, false),
        Field::new("backward_stages", DataType::UInt16, false),
        Field::new("backward_states", DataType::UInt32, false),
        Field::new("backward_solver_ms", DataType::UInt64, false),
        Field::new("backward_cuts_added", DataType::UInt32, false),
        Field::new("backward_cuts_total", DataType::UInt32, false),
    ]))
}

/// Schema for forward pass detail data (forward_detail.csv equivalent).
///
/// Contains state, decision, and uncertainty realizations.
/// Optimized for indexed format (variable_index instead of variable_name).
pub fn forward_detail_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("forward_pass_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::Int16, false), // Signed for compatibility
        Field::new("variable_index", DataType::UInt16, false),
        Field::new("entity_id", DataType::UInt16, true), // Nullable for scalar variables
        Field::new("lag_index", DataType::UInt8, true), // Nullable for non-lag variables
        Field::new("value", DataType::Float64, false),
    ]))
}

/// Schema for backward pass detail data (backward_detail.csv equivalent).
///
/// Contains branching realizations from backward pass.
pub fn backward_detail_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("iteration", DataType::UInt32, false),
        Field::new("forward_pass_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::Int16, false),
        Field::new("training_state_id", DataType::UInt32, false),
        Field::new("branching_idx", DataType::UInt16, false),
        Field::new("variable_index", DataType::UInt16, false),
        Field::new("entity_id", DataType::UInt16, true),
        Field::new("lag_index", DataType::UInt8, true),
        Field::new("value", DataType::Float64, false),
    ]))
}

/// Schema for Benders cuts (cuts.csv equivalent).
///
/// Contains cut coefficients in indexed format.
pub fn cuts_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("stage_index", DataType::UInt16, false),
        Field::new("stage_cut_id", DataType::UInt32, false),
        Field::new("iteration", DataType::UInt32, false),
        Field::new("forward_pass_idx", DataType::UInt32, false),
        Field::new("active", DataType::Boolean, false),
        Field::new("coefficient_index", DataType::UInt16, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

/// Schema for visited states (states.csv equivalent).
///
/// Contains state information with dominating cuts.
pub fn states_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("stage_index", DataType::UInt16, false),
        Field::new("dominating_cut_id", DataType::UInt32, false),
        Field::new("iteration", DataType::UInt32, false),
        Field::new("forward_pass_idx", DataType::UInt32, false),
        Field::new("state_component_index", DataType::UInt16, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

/// Schema for simulation results - hydros (simulation_hydros.csv equivalent).
pub fn simulation_hydros_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("scenario_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::UInt16, false),
        Field::new("hydro_id", DataType::UInt16, false),
        Field::new("initial_storage", DataType::Float64, false),
        Field::new("final_storage", DataType::Float64, false),
        Field::new("inflow", DataType::Float64, false),
        Field::new("turbined_flow", DataType::Float64, false),
        Field::new("spillage", DataType::Float64, false),
        Field::new("water_value", DataType::Float64, false),
    ]))
}

/// Schema for simulation results - buses (simulation_buses.csv equivalent).
pub fn simulation_buses_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("scenario_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::UInt16, false),
        Field::new("bus_id", DataType::UInt16, false),
        Field::new("load", DataType::Float64, false),
        Field::new("deficit", DataType::Float64, false),
        Field::new("marginal_cost", DataType::Float64, false),
    ]))
}

/// Schema for simulation results - thermals (simulation_thermals.csv equivalent).
pub fn simulation_thermals_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("scenario_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::UInt16, false),
        Field::new("thermal_id", DataType::UInt16, false),
        Field::new("generation", DataType::Float64, false),
    ]))
}

/// Schema for simulation results - lines (simulation_lines.csv equivalent).
pub fn simulation_lines_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("scenario_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::UInt16, false),
        Field::new("line_id", DataType::UInt16, false),
        Field::new("flow", DataType::Float64, false),
    ]))
}

/// Schema for sampled noises from SAA (sampled_noises.csv equivalent).
pub fn sampled_noises_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("scenario_idx", DataType::UInt32, false),
        Field::new("stage_id", DataType::UInt16, false),
        Field::new("uncertainty_type", DataType::Utf8, false),
        Field::new("entity_id", DataType::UInt16, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_schema_fields() {
        let schema = training_schema();
        assert_eq!(schema.fields().len(), 16);
        assert_eq!(schema.field(0).name(), "iteration");
        assert_eq!(schema.field(0).data_type(), &DataType::UInt32);
    }

    #[test]
    fn test_forward_detail_schema_fields() {
        let schema = forward_detail_schema();
        assert_eq!(schema.fields().len(), 7);
        assert_eq!(schema.field(3).name(), "variable_index");
        assert!(schema.field(4).is_nullable()); // entity_id is nullable
    }

    #[test]
    fn test_cuts_schema_has_coefficient_index() {
        let schema = cuts_schema();
        assert!(schema
            .fields()
            .iter()
            .any(|f| f.name() == "coefficient_index"));
    }

    #[test]
    fn test_all_schemas_created_successfully() {
        // Just verify they don't panic
        training_schema();
        forward_detail_schema();
        backward_detail_schema();
        cuts_schema();
        states_schema();
        simulation_hydros_schema();
        simulation_buses_schema();
        simulation_thermals_schema();
        simulation_lines_schema();
        sampled_noises_schema();
    }
}
