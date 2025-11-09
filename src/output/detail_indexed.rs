//! Indexed forward and backward pass detail output module.
//!
//! This module handles writing detailed training pass data in indexed format,
//! using variable_index instead of variable_name to reduce file size by 20-30%.

use crate::output::dictionary::OutputVariable;
use crate::sddp;

use csv::Writer;
use std::error::Error;

/// Writes forward pass details to CSV using indexed format.
///
/// Exports complete forward pass trajectories with integer indices for variables.
/// Requires variable_dictionary.csv for decoding variable names.
///
/// Schema: iteration, forward_pass_idx, stage_id, variable_index, entity_id, lag_index, value
///
/// # Arguments
///
/// * `forward_details` - Forward pass details collected during training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None` or details are empty)
pub(super) fn write_forward_detail_indexed(
    forward_details: &[sddp::ForwardPassDetail],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    if forward_details.is_empty() {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/forward_detail.csv"))?;

    wtr.write_record([
        "iteration",
        "forward_pass_idx",
        "stage_id",
        "variable_index",
        "entity_id",
        "lag_index",
        "value",
    ])?;

    for detail in forward_details {
        let r = &detail.realization;
        let iteration = detail.iteration.to_string();
        let fp_idx = detail.forward_pass_idx.to_string();
        let stage_id = detail.stage_id.to_string();

        // Initial storage
        for (hydro_id, &value) in r.initial_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::InitialStorage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lags
        for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
            for (lag_idx, &value) in lags.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &(OutputVariable::InflowLag as usize).to_string(),
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Sampled loads
        for (bus_id, &value) in r.loads.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::SampledLoad as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Sampled inflows
        for (hydro_id, &value) in r.inflow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::SampledInflow as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Final storage
        for (hydro_id, &value) in r.final_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::FinalStorage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Turbined flow
        for (hydro_id, &value) in r.turbined_flow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::TurbinedFlow as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Spillage
        for (hydro_id, &value) in r.spillage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::Spillage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Water value
        for (hydro_id, &value) in r.water_value.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::WaterValue as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Thermal generation
        for (thermal_id, &value) in r.thermal_generation.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::ThermalGeneration as usize).to_string(),
                &thermal_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Deficit
        for (bus_id, &value) in r.deficit.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::Deficit as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Exchange
        for (line_id, &value) in r.exchange.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::Exchange as usize).to_string(),
                &line_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Marginal cost
        for (bus_id, &value) in r.marginal_cost.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &(OutputVariable::MarginalCost as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lag duals
        for (hydro_id, duals) in r.inflow_lag_duals.iter().enumerate() {
            for (lag_idx, &value) in duals.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &(OutputVariable::InflowLagDual as usize).to_string(),
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Current stage objective
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &(OutputVariable::CurrentStageObjective as usize).to_string(),
            "",
            "",
            &r.current_stage_objective.to_string(),
        ])?;

        // Total stage objective
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &(OutputVariable::TotalStageObjective as usize).to_string(),
            "",
            "",
            &r.total_stage_objective.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}

/// Writes backward pass details to CSV using indexed format.
///
/// Exports complete backward pass branching realizations with integer indices.
/// Requires variable_dictionary.csv for decoding variable names.
///
/// Schema: iteration, forward_pass_idx, stage_id, training_state_id, branching_idx,
///         variable_index, entity_id, lag_index, value
///
/// # Arguments
///
/// * `backward_details` - Backward pass details collected during training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None` or details are empty)
pub(super) fn write_backward_detail_indexed(
    backward_details: &[sddp::BackwardPassDetail],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    if backward_details.is_empty() {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/backward_detail.csv"))?;

    wtr.write_record([
        "iteration",
        "forward_pass_idx",
        "stage_id",
        "training_state_id",
        "branching_idx",
        "variable_index",
        "entity_id",
        "lag_index",
        "value",
    ])?;

    for detail in backward_details {
        let r = &detail.realization;
        let iteration = detail.iteration.to_string();
        let fp_idx = detail.forward_pass_idx.to_string();
        let stage_id = detail.stage_id.to_string();
        let ts_id = detail.training_state_id.to_string();
        let br_idx = detail.branching_idx.to_string();

        // Initial storage
        for (hydro_id, &value) in r.initial_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::InitialStorage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lags
        for (hydro_id, lags) in r.inflow_lags.iter().enumerate() {
            for (lag_idx, &value) in lags.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &ts_id,
                    &br_idx,
                    &(OutputVariable::InflowLag as usize).to_string(),
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Sampled loads
        for (bus_id, &value) in r.loads.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::SampledLoad as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Sampled inflows
        for (hydro_id, &value) in r.inflow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::SampledInflow as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Final storage
        for (hydro_id, &value) in r.final_storage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::FinalStorage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Turbined flow
        for (hydro_id, &value) in r.turbined_flow.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::TurbinedFlow as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Spillage
        for (hydro_id, &value) in r.spillage.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::Spillage as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Water value
        for (hydro_id, &value) in r.water_value.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::WaterValue as usize).to_string(),
                &hydro_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Thermal generation
        for (thermal_id, &value) in r.thermal_generation.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::ThermalGeneration as usize).to_string(),
                &thermal_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Deficit
        for (bus_id, &value) in r.deficit.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::Deficit as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Exchange
        for (line_id, &value) in r.exchange.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::Exchange as usize).to_string(),
                &line_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Marginal cost
        for (bus_id, &value) in r.marginal_cost.iter().enumerate() {
            wtr.write_record([
                &iteration,
                &fp_idx,
                &stage_id,
                &ts_id,
                &br_idx,
                &(OutputVariable::MarginalCost as usize).to_string(),
                &bus_id.to_string(),
                "",
                &value.to_string(),
            ])?;
        }

        // Inflow lag duals
        for (hydro_id, duals) in r.inflow_lag_duals.iter().enumerate() {
            for (lag_idx, &value) in duals.iter().enumerate() {
                wtr.write_record([
                    &iteration,
                    &fp_idx,
                    &stage_id,
                    &ts_id,
                    &br_idx,
                    &(OutputVariable::InflowLagDual as usize).to_string(),
                    &hydro_id.to_string(),
                    &(lag_idx + 1).to_string(),
                    &value.to_string(),
                ])?;
            }
        }

        // Current stage objective
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &ts_id,
            &br_idx,
            &(OutputVariable::CurrentStageObjective as usize).to_string(),
            "",
            "",
            &r.current_stage_objective.to_string(),
        ])?;

        // Total stage objective
        wtr.write_record([
            &iteration,
            &fp_idx,
            &stage_id,
            &ts_id,
            &br_idx,
            &(OutputVariable::TotalStageObjective as usize).to_string(),
            "",
            "",
            &r.total_stage_objective.to_string(),
        ])?;
    }

    wtr.flush()?;
    Ok(())
}
