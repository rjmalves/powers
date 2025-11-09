//! Normalized simulation results output module.
//!
//! This module writes simulation trajectory results in a normalized indexed format
//! that matches the detail output structure for consistency.

use crate::output::dictionary::OutputVariable;
use crate::sddp;
use crate::system;
use csv::Writer;
use std::error::Error;

/// Output record for simulation results (indexed format)
#[derive(serde::Serialize)]
struct SimulationOutput {
    stage: usize,
    series: usize,
    variable_index: usize,
    entity_id: Option<usize>,
    value: f64,
}

/// Writes simulation results in normalized indexed format.
///
/// # Arguments
///
/// * `simulation_trajectories` - Lightweight trajectories with simulation results
/// * `system` - Power system configuration for entity counting
/// * `path` - Optional output directory path. If `None`, no file is written (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # CSV Schema
///
/// ```text
/// stage,series,variable_index,entity_id,value
/// 0,0,2,0,150.5
/// 0,0,9,0,0.0
/// ...
/// ```
pub(super) fn write_simulation_normalized(
    simulation_trajectories: &[sddp::SimulationTrajectory],
    _system: &system::System,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    let mut wtr =
        Writer::from_path(&(output_dir.to_owned() + "/simulation.csv"))?;

    for (series_index, trajectory) in simulation_trajectories.iter().enumerate()
    {
        for (stage_index, realization) in
            trajectory.realizations.iter().enumerate()
        {
            // Bus variables
            for (entity_id, &value) in realization.loads.iter().enumerate() {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::SampledLoad as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in realization.deficit.iter().enumerate() {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::Deficit as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in
                realization.marginal_cost.iter().enumerate()
            {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::MarginalCost as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            // Line variables
            for (entity_id, &value) in realization.exchange.iter().enumerate() {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::Exchange as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            // Thermal variables
            for (entity_id, &value) in
                realization.thermal_generation.iter().enumerate()
            {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::ThermalGeneration as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            // Hydro variables
            for (entity_id, &value) in
                realization.final_storage.iter().enumerate()
            {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::FinalStorage as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in realization.inflow.iter().enumerate() {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::SampledInflow as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in
                realization.turbined_flow.iter().enumerate()
            {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::TurbinedFlow as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in realization.spillage.iter().enumerate() {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::Spillage as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            for (entity_id, &value) in
                realization.water_value.iter().enumerate()
            {
                wtr.serialize(SimulationOutput {
                    stage: stage_index,
                    series: series_index,
                    variable_index: OutputVariable::WaterValue as usize,
                    entity_id: Some(entity_id),
                    value,
                })?;
            }

            // Stage objectives
            wtr.serialize(SimulationOutput {
                stage: stage_index,
                series: series_index,
                variable_index: OutputVariable::CurrentStageObjective as usize,
                entity_id: None,
                value: realization.current_stage_objective,
            })?;
        }
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_simulation_normalized_empty() {
        let result =
            write_simulation_normalized(&[], &system::System::default(), None);
        assert!(result.is_ok());
    }
}
