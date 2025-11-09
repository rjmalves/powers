//! Output generation for SDDP algorithm results.
//!
//! This module handles writing various output files from the SDDP algorithm,
//! including training results, simulation trajectories, cuts, and states.
//!
//! All outputs use indexed/normalized format for consistency and efficiency.

pub mod coefficient_dictionary;
mod cuts_states_indexed;
mod detail_indexed;
pub mod dictionary;

#[cfg(feature = "parquet-output")]
pub mod parquet;

mod simulation_normalized;
mod training;
pub mod writer;

use crate::fcf;
use crate::graph;
use crate::scenario;
use crate::sddp;
use crate::system;

use std::error::Error;
use std::sync::{Arc, Mutex};

use cuts_states_indexed::{
    write_benders_cuts_indexed, write_visited_states_indexed,
};
use detail_indexed::{
    write_backward_detail_indexed, write_forward_detail_indexed,
};
use simulation_normalized::write_simulation_normalized;
use training::{write_sampled_noises_indexed, write_training_results};

/// Generates all CSV output files from SDDP training and simulation results.
///
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
/// * `export_training_noises` - Whether to export sampled noises to CSV
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
    future_cost_function_graph: &graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    simulation_trajectories: &[sddp::SimulationTrajectory],
    training_results: &[sddp::IterationResult],
    forward_details: &[sddp::ForwardPassDetail],
    backward_details: &[sddp::BackwardPassDetail],
    saa: &scenario::ScenarioTree,
    system: &system::System,
    max_ar_order: usize,
    hydro_ar_orders: &[usize],
    export_training_noises: bool,
    path: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    // Generate all dictionaries first (provides metadata for indexed outputs)
    if let Some(output_dir) = path {
        // Variable dictionary for detail and simulation outputs
        let var_dict =
            dictionary::VariableDictionary::generate(system, max_ar_order);
        var_dict.write_csv(output_dir)?;

        // Coefficient dictionary for cuts output
        let coef_dict = coefficient_dictionary::CoefficientDictionary::generate(
            system,
            hydro_ar_orders,
        );
        coef_dict.write_csv(output_dir)?;

        // State component dictionary (same structure as coefficient dictionary)
        // States also use: [objective, storage_0, storage_1, ..., lag_0_1, lag_1_1, ...]
        coef_dict.write_csv_as_state_components(output_dir)?;
    }

    write_training_results(training_results, path)?;

    // All outputs use indexed/normalized format
    write_forward_detail_indexed(forward_details, path)?;
    write_backward_detail_indexed(backward_details, path)?;
    write_benders_cuts_indexed(future_cost_function_graph, path)?;
    write_visited_states_indexed(future_cost_function_graph, path)?;
    write_simulation_normalized(simulation_trajectories, system, path)?;

    if export_training_noises {
        write_sampled_noises_indexed(saa, system, path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_mock_iteration_result(
        iteration: usize,
        lower_bound: f64,
        forward_costs: Vec<f64>,
    ) -> sddp::IterationResult {
        sddp::IterationResult {
            iteration,
            lower_bound,
            forward_costs,
            iteration_time: Duration::from_millis(100),
            forward_timing: sddp::ForwardPassTiming {
                saa_sampling_time: Duration::from_millis(1),
                model_preprocessing_time: Duration::from_millis(2),
                solver_time: Duration::from_millis(50),
                model_postprocessing_time: Duration::from_millis(3),
                forward_postprocessing_time: Duration::from_millis(4),
                total_time: Duration::from_millis(60),
            },
            backward_timing: sddp::BackwardPassTiming {
                backward_preprocessing_time: Duration::from_millis(1),
                model_preprocessing_time: Duration::from_millis(2),
                solver_time: Duration::from_millis(30),
                model_postprocessing_time: Duration::from_millis(3),
                cut_selection_time: Duration::from_millis(1),
                fcf_state_update_time: Duration::from_millis(1),
                cut_cloning_time: Duration::from_millis(1),
                handler_application_time: Duration::from_millis(1),
                total_time: Duration::from_millis(40),
            },
            num_solver_calls: 10,
            num_cuts_added: 5,
            num_cuts_removed: 0,
            num_cuts_returned: 0,
            num_active_cuts: 5,
        }
    }

    #[test]
    fn test_training_csv_no_output_path() {
        let results =
            vec![create_mock_iteration_result(1, 100.0, vec![105.0, 103.0])];
        let result = write_training_results(&results, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_training_csv_row_count() {
        let results = vec![
            create_mock_iteration_result(1, 100.0, vec![105.0, 103.0, 104.0]),
            create_mock_iteration_result(2, 102.0, vec![106.0, 104.0]),
        ];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 3);

        assert!(lines[0].contains("iteration"));
        assert!(!lines[0].contains("forward_pass_idx"));
        assert!(lines[0].contains("lower_bound"));
        assert!(lines[0].contains("policy_cost"));

        let row1_fields: Vec<&str> = lines[1].split(',').collect();
        let policy_cost1 = row1_fields[2].parse::<f64>().unwrap();
        assert!((policy_cost1 - 104.0).abs() < 1e-10);

        let row2_fields: Vec<&str> = lines[2].split(',').collect();
        let policy_cost2 = row2_fields[2].parse::<f64>().unwrap();
        assert!((policy_cost2 - 105.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_csv_gap_computation() {
        let results = vec![
            create_mock_iteration_result(1, 100.0, vec![105.0, 103.0]),
            create_mock_iteration_result(2, 1e-12, vec![105.0]),
        ];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert_eq!(lines.len(), 3);

        let row1 = lines[1];
        let fields: Vec<&str> = row1.split(',').collect();
        let gap = fields[4].parse::<f64>().unwrap();
        assert!((gap - 4.0).abs() < 0.1);

        let row2 = lines[2];
        let fields: Vec<&str> = row2.split(',').collect();
        let gap_str = fields[4];
        assert!(
            gap_str == "inf" || gap_str.to_lowercase().contains("inf"),
            "Expected infinity, got: {}",
            gap_str
        );
    }

    #[test]
    fn test_training_csv_timing_conversion() {
        let results = vec![create_mock_iteration_result(1, 100.0, vec![105.0])];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        write_training_results(&results, Some(path)).unwrap();

        let csv_path = format!("{}/training.csv", path);
        let content = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        let row = lines[1];
        let fields: Vec<&str> = row.split(',').collect();

        let forward_solver_ms = fields[7].parse::<u64>().unwrap();
        let backward_solver_ms = fields[13].parse::<u64>().unwrap();

        assert_eq!(forward_solver_ms, 50);
        assert_eq!(backward_solver_ms, 30);
    }

    #[test]
    fn test_training_csv_empty_forward_costs() {
        let results = vec![create_mock_iteration_result(1, 100.0, vec![])];

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().to_str().unwrap();

        let result = write_training_results(&results, Some(path));
        assert!(result.is_ok());

        let csv_path = format!("{}/training.csv", path);

        assert!(std::path::Path::new(&csv_path).exists());

        let content = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(
            content.len(),
            0,
            "Empty forward_costs should produce empty CSV"
        );
    }
}
