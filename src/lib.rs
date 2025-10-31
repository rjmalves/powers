pub mod base_noise;
pub mod cli;
pub mod correlation_applicator;
pub mod cut;
pub mod error;
pub mod fcf;
pub mod lognormal3;
pub mod marginal_transformer;
pub mod solver;
pub mod state;
pub mod stochastic_process;
pub mod subproblem;
pub mod system;

pub mod graph;
pub mod initial_condition;
pub mod input;
pub mod input_validation;
mod log;
pub mod noise_model_cache;
pub mod output;
pub mod par_generator;
mod risk_measure;
pub mod scenario;
pub mod sddp;
pub mod seasonal_params;
pub mod unified_inflow_model;
pub mod unified_noise_spec;
pub mod utils;

use std::error::Error;
use std::path::Path;
use std::time::Instant;

/// Main entry point for SDDP algorithm execution (run subcommand).
///
/// This function uses the **Factory API** (`SddpAlgorithm::from_files()`)
///
/// For simpler use cases (unit tests with explicit scenarios), consider using the
/// **Builder API** via `sddp::SddpAlgorithm::builder()` instead.
///
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

    let _training_result =
        sddp.train().map_err(|e| -> Box<dyn Error> { e.into() })?;

    if sddp.config().num_simulation_scenarios.is_some() {
        let simulation_trajectories = sddp
            .simulate()
            .map_err(|e| -> Box<dyn Error> { e.into() })?;

        log::output_generation_line(&path_str);
        output::generate_outputs(
            &sddp.algorithm().future_cost_function_graph,
            &simulation_trajectories,
            sddp.config().output_path.as_deref(),
        )?;
    } else {
        log::simulation_skipped();
    }

    log::show_farewell(begin.elapsed());

    Ok(())
}
