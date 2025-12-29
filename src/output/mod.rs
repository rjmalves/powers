//! Output generation for SDDP algorithm results.
//!
//! This module handles writing various output files from the SDDP algorithm,
//! including training results, simulation trajectories, cuts, and states.
//!
//! All outputs use indexed/normalized format for consistency and efficiency.

pub mod coefficient_dictionary;
pub mod csv;
pub mod dictionary;
pub mod factory;
pub mod parquet;
pub mod writer;

use crate::fcf;
use crate::graph;
use crate::scenario;
use crate::sddp;
use crate::system;

use std::error::Error;

/// Generates all output files from SDDP training and simulation results.
///
/// Supports multiple output formats through the OutputWriter trait.
/// All outputs use indexed/normalized format for consistency and space efficiency:
/// - Variables referenced by integer indices (requires variable_dictionary.csv)
/// - Cut/state coefficients use integer indices (requires coefficient_dictionary.csv + state_component_dictionary.csv)
/// - Simulation uses normalized long format (one row per value)
/// - Sampled noises use indexed format
///
/// # Arguments
///
/// * `future_cost_function_graph` - Graph with future cost functions and cuts
/// * `simulation_trajectories` - Lightweight trajectories with simulation output data
/// * `training_results` - Iteration-level convergence tracking
/// * `forward_details` - Forward pass trajectory details (if collected)
/// * `backward_details` - Backward pass branching details (if collected)
/// * `saa` - Sample Average Approximation containing sampled scenarios
/// * `system` - Power system configuration for dictionary generation
/// * `max_ar_order` - Maximum AR order across all temporal models (0 if none)
/// * `hydro_ar_orders` - AR order for each hydro (for coefficient dictionary)
/// * `output_config` - Complete output configuration including format and export flags
/// * `path` - Optional output directory path. If `None`, all output is skipped (no-op).
///
/// # Returns
///
/// `Ok(())` if successful or skipped (when `path` is `None`)
///
/// # Performance
///
/// When `path` is `None`, returns immediately with no I/O overhead.
/// Dictionary generation adds < 100ms overhead.
#[allow(clippy::too_many_arguments)]
pub fn generate_outputs(
    future_cost_function_graph: &graph::DirectedGraph<fcf::FutureCostFunction>,
    simulation_trajectories: &[sddp::SimulationTrajectory],
    training_results: &[sddp::IterationResult],
    forward_details: &[sddp::ForwardPassDetail],
    backward_details: &[sddp::BackwardPassDetail],
    saa: &scenario::ScenarioTree,
    system: &system::System,
    max_ar_order: usize,
    hydro_ar_orders: &[usize],
    output_config: &crate::input::OutputConfig,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let Some(output_dir) = path else {
        return Ok(());
    };

    // Generate all dictionaries first (provides metadata for indexed outputs)
    generate_dictionaries(
        output_dir,
        system,
        max_ar_order,
        hydro_ar_orders,
        output_config,
    )?;

    // Create writer based on configured format
    let mut writer = factory::create_writer(
        output_config.format,
        std::path::Path::new(output_dir),
    )?;

    // Write outputs using trait methods based on configuration flags
    if output_config.export_training {
        writer.write_training(training_results)?;
    }

    if output_config.export_simulation && !simulation_trajectories.is_empty() {
        writer.write_simulation(simulation_trajectories, system)?;
    }

    if output_config.export_forward_detail && !forward_details.is_empty() {
        writer.write_forward_detail(forward_details)?;
    }

    if output_config.export_backward_detail && !backward_details.is_empty() {
        writer.write_backward_detail(backward_details)?;
    }

    if output_config.export_cuts {
        writer.write_cuts(future_cost_function_graph)?;
    }

    if output_config.export_states {
        writer.write_states(future_cost_function_graph)?;
    }

    if output_config.export_training_noises {
        writer.write_noises(saa, system)?;
    }

    writer.flush()?;
    Ok(())
}

/// Generate dictionary files (format-agnostic).
///
/// Dictionaries are always written as CSV regardless of output format,
/// since they provide metadata for interpreting indexed outputs.
///
/// # Arguments
///
/// * `output_dir` - Directory where dictionaries will be written
/// * `system` - Power system configuration
/// * `max_ar_order` - Maximum AR order across temporal models
/// * `hydro_ar_orders` - AR order for each hydro unit
/// * `output_config` - Output configuration to determine which dictionaries to write
fn generate_dictionaries(
    output_dir: &str,
    system: &system::System,
    max_ar_order: usize,
    hydro_ar_orders: &[usize],
    output_config: &crate::input::OutputConfig,
) -> Result<(), Box<dyn Error>> {
    // Variable dictionary for detail and simulation outputs
    // Always write if any output might need it
    let var_dict =
        dictionary::VariableDictionary::generate(system, max_ar_order);
    var_dict.write_csv(output_dir)?;

    // Coefficient dictionary only needed when exporting cuts
    if output_config.export_cuts {
        let coef_dict = coefficient_dictionary::CoefficientDictionary::generate(
            system,
            hydro_ar_orders,
        );
        coef_dict.write_csv(output_dir)?;
    }

    // State component dictionary only needed when exporting states
    if output_config.export_states {
        let coef_dict = coefficient_dictionary::CoefficientDictionary::generate(
            system,
            hydro_ar_orders,
        );
        coef_dict.write_csv_as_state_components(output_dir)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::OutputFormat;

    #[test]
    fn test_generate_dictionaries() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let system = system::System::new_empty();
        let output_config = crate::input::OutputConfig::default();
        let result =
            generate_dictionaries(path, &system, 0, &[], &output_config);

        // Should succeed even with empty system
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_outputs_with_no_path() {
        let fcf_graph = graph::DirectedGraph::new();
        let system = system::System::new_empty();
        let saa = scenario::ScenarioTree::new_empty();
        let output_config = crate::input::OutputConfig::default();

        // Should succeed immediately with no path
        let result = generate_outputs(
            &fcf_graph,
            &[],
            &[],
            &[],
            &[],
            &saa,
            &system,
            0,
            &[],
            &output_config,
            None,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_outputs_csv_format() {
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let fcf_graph = graph::DirectedGraph::new();
        let system = system::System::new_empty();
        let saa = scenario::ScenarioTree::new_empty();
        let output_config = crate::input::OutputConfig {
            format: OutputFormat::CSV,
            ..Default::default()
        };

        let result = generate_outputs(
            &fcf_graph,
            &[],
            &[],
            &[],
            &[],
            &saa,
            &system,
            0,
            &[],
            &output_config,
            Some(path),
        );

        assert!(result.is_ok());
    }
}
