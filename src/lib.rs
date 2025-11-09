pub mod cli;
pub mod correlation_applicator;
pub mod cut;
pub mod error;
pub mod fcf;
pub mod solver;
pub mod state;
pub mod subproblem;
pub mod system;

pub mod graph;
pub mod initial_condition;
pub mod input;
pub mod input_validation;
mod log;
pub mod output;
mod risk_measure;
pub mod scenario;
pub mod scenario_generator;
pub mod sddp;
pub mod temporal_model;

pub mod utils;

use std::error::Error;
use std::path::Path;
use std::time::Instant;

/// Main entry point for SDDP algorithm execution (run subcommand).
pub fn run(input_path: &Path) -> Result<(), Box<dyn Error>> {
    log::show_greeting();

    let begin = Instant::now();

    let path_str = input_path.display().to_string();
    log::input_reading_line(&path_str);

    let mut sddp = sddp::SddpAlgorithm::from_files(
        input_path.join("config.json"),
        input_path.join("system.json"),
        input_path.join("graph.json"),
        input_path.join("recourse.json"),
    )
    .map_err(|e| -> Box<dyn Error> { e.into() })?;

    let training_result =
        sddp.train().map_err(|e| -> Box<dyn Error> { e.into() })?;

    let simulation_trajectories = match sddp.config().num_simulation_scenarios {
        Some(_) => sddp
            .simulate()
            .map_err(|e| -> Box<dyn Error> { e.into() })?,
        None => {
            log::simulation_skipped();
            Vec::new()
        }
    };

    // Validate output configuration before writing
    sddp.config().output.validate().map_err(|e| e.to_string())?;

    // Resolve output path: if relative, make it relative to input directory
    let resolved_output_path = match &sddp.config().output.path {
        Some(path) => {
            let path_obj = Path::new(path);
            if path_obj.is_absolute() {
                // Absolute path: use as-is
                Some(path.clone())
            } else {
                // Relative path: resolve relative to input directory
                Some(input_path.join(path).display().to_string())
            }
        }
        None => None,
    };

    // Log the actual output path that will be used
    if let Some(ref output_path) = resolved_output_path {
        log::output_generation_line(output_path);
    }

    output::generate_outputs(
        &sddp.algorithm().future_cost_function_graph,
        &simulation_trajectories,
        training_result.iterations(),
        &training_result.forward_details,
        &training_result.backward_details,
        sddp.saa(),
        sddp.system(),
        sddp.max_ar_order(),
        &sddp.hydro_ar_orders(),
        &sddp.config().output,
        resolved_output_path.as_deref(),
    )?;

    log::show_farewell(begin.elapsed());

    Ok(())
}
