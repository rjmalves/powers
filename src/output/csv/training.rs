//! Training results and noise output module.
//!
//! This module handles writing training iteration results and sampled noises
//! from the SDDP algorithm to CSV files.

use crate::output::dictionary::OutputVariable;
use crate::scenario;
use crate::sddp;
use crate::system;

use csv::Writer;
use std::error::Error;

#[derive(serde::Serialize)]
struct SampledNoiseOutput {
    stage_index: usize,
    branching_index: usize,
    variable_index: usize,
    entity_id: usize,
    value: f64,
}

/// Writes sampled noises from ScenarioTree to CSV file using indexed format.
///
/// # Arguments
///
/// * `tree` - Reference to the ScenarioTree containing sampled scenarios
/// * `system` - Power system configuration for variable mapping
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # CSV Format
///
/// Columns: `stage_index`, `branching_index`, `variable_index`, `entity_id`, `value`
/// - Uses OutputVariable enum indices for variable_index
/// - Consistent with other indexed outputs
pub(super) fn write_sampled_noises_indexed(
    tree: &scenario::ScenarioTree,
    _system: &system::System,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr = Writer::from_path(
        &(output_dir.to_owned() + "/training_sampled_noises.csv"),
    )?;

    for (stage_index, stage_branchings) in
        tree.stage_scenarios.iter().enumerate()
    {
        for (branching_index, branching_noises) in
            stage_branchings.branching_noises.iter().enumerate()
        {
            for (entity_id, &noise_value) in
                branching_noises.load_innovations.iter().enumerate()
            {
                wtr.serialize(SampledNoiseOutput {
                    stage_index,
                    branching_index,
                    variable_index: OutputVariable::SampledLoad as usize,
                    entity_id,
                    value: noise_value,
                })?;
            }

            for (entity_id, &noise_value) in
                branching_noises.inflow_innovations.iter().enumerate()
            {
                wtr.serialize(SampledNoiseOutput {
                    stage_index,
                    branching_index,
                    variable_index: OutputVariable::SampledInflow as usize,
                    entity_id,
                    value: noise_value,
                })?;
            }
        }
    }

    wtr.flush()?;
    Ok(())
}
#[derive(serde::Serialize)]
struct TrainingOutput {
    // Convergence metrics
    iteration: usize,
    lower_bound: f64,
    policy_cost: f64,
    policy_std: f64,
    gap_percent: f64,

    // Forward pass timing (milliseconds)
    forward_saa_sampling_ms: u64,
    forward_model_preprocessing_ms: u64,
    forward_solver_ms: u64,
    forward_model_postprocessing_ms: u64,
    forward_postprocessing_ms: u64,
    forward_total_ms: u64,

    // Backward pass timing (milliseconds)
    backward_preprocessing_ms: u64,
    backward_model_preprocessing_ms: u64,
    backward_solver_ms: u64,
    backward_model_postprocessing_ms: u64,
    backward_cut_selection_ms: u64,
    backward_fcf_state_update_ms: u64,
    backward_cut_cloning_ms: u64,
    backward_handler_application_ms: u64,
    backward_total_ms: u64,
}

/// Writes training iteration results to CSV file.
///
/// Exports iteration-level convergence tracking with detailed timing breakdowns.
///
/// # Arguments
///
/// * `results` - Iteration results from SDDP training
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Schema
///
/// **Convergence metrics**:
/// - `iteration`: Iteration number (1-indexed)
/// - `lower_bound`: Lower bound after backward pass ($)
/// - `policy_cost`: Mean cost of all forward passes in iteration ($)
/// - `policy_std`: Std dev of all forward pass costs in iteration ($)
/// - `gap_percent`: Relative gap = 100 × (policy_cost - lower_bound) / |lower_bound|
///
/// **Timing fields**: All durations in milliseconds
pub(super) fn write_training_results(
    results: &[sddp::IterationResult],
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/training.csv"))?;

    for result in results {
        let num_forward_passes = result.forward_costs.len();
        if num_forward_passes == 0 {
            continue;
        }

        let mean_cost = result.forward_costs.iter().sum::<f64>()
            / num_forward_passes as f64;

        let variance = result
            .forward_costs
            .iter()
            .map(|&cost| (cost - mean_cost).powi(2))
            .sum::<f64>()
            / num_forward_passes as f64;
        let policy_std = variance.sqrt();

        let gap_percent = if result.lower_bound.abs() > 1e-10 {
            100.0 * (mean_cost - result.lower_bound) / result.lower_bound.abs()
        } else {
            f64::INFINITY
        };

        wtr.serialize(TrainingOutput {
            iteration: result.iteration,
            lower_bound: result.lower_bound,
            policy_cost: mean_cost,
            policy_std,
            gap_percent,

            // Forward pass timing - map from new structure
            forward_saa_sampling_ms: result.timing.forward.saa_sampling.as_millis() as u64,
            forward_model_preprocessing_ms: result.timing.forward.model_preprocessing.as_millis() as u64,
            forward_solver_ms: result.timing.forward.solver.as_millis() as u64,
            forward_model_postprocessing_ms: result.timing.forward.model_postprocessing.as_millis() as u64,
            forward_postprocessing_ms: result.timing.forward.postprocessing.as_millis() as u64,
            forward_total_ms: result.timing.forward.total.as_millis() as u64,

            // Backward pass timing - map from new structure
            // Note: backward_preprocessing_ms removed (was always zero in new schema)
            backward_preprocessing_ms: 0,  // Removed field, set to zero for backward compatibility
            backward_model_preprocessing_ms: result.timing.backward.model_preprocessing.as_millis() as u64,
            backward_solver_ms: result.timing.backward.solver.as_millis() as u64,
            backward_model_postprocessing_ms: result.timing.backward.model_postprocessing.as_millis() as u64,
            backward_cut_selection_ms: result.timing.backward.cut_selection.as_millis() as u64,
            // Phase 3 fields now combined into problem_update - split evenly for backward compatibility
            backward_fcf_state_update_ms: result.timing.backward.problem_update.as_millis() as u64 / 3,
            backward_cut_cloning_ms: result.timing.backward.problem_update.as_millis() as u64 / 3,
            backward_handler_application_ms: result.timing.backward.problem_update.as_millis() as u64 / 3,
            backward_total_ms: result.timing.backward.total.as_millis() as u64,
        })?;
    }

    wtr.flush()?;
    Ok(())
}
