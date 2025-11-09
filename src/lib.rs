pub mod cli;
pub mod correlation_applicator;
pub mod cut;
pub mod error;
pub mod fcf;
pub mod logging;
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
    // Load config first
    let config = input::read_config_input(
        &input_path.join("config.json").display().to_string(),
    );

    // Initialize logging
    crate::logging::init(&config.logging)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    // Application greeting
    ::log::info!("");
    ::log::info!(
        "POWE.RS - Power Optimization for the World of Energy - in pure RuSt"
    );
    ::log::info!(
        "--------------------------------------------------------------------"
    );

    let begin = Instant::now();

    let path_str = input_path.display().to_string();
    ::log::info!("");
    ::log::info!("Reading input files from '{}'", path_str);

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
            ::log::info!("");
            ::log::info!("# Simulation");
            ::log::info!(
                "Simulation skipped (num_simulation_scenarios not configured)"
            );
            ::log::info!("");
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
        ::log::info!("");
        ::log::info!("Writing outputs to '{}'", output_path);
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

    // Application farewell with timing
    let duration = begin.elapsed();
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    let millis = duration.subsec_millis();
    ::log::info!("");
    ::log::info!(
        "Total running time: {:02}:{:02}:{:02}.{:03}",
        hours,
        minutes,
        seconds,
        millis
    );

    Ok(())
}
