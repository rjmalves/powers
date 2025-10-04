// Modules made public for testing purposes (T1.2, T1.3, T1.4, T1.6)
// In production builds, these are only accessible internally
pub mod cut;
pub mod fcf;
pub mod solver;
pub mod state;
pub mod stochastic_process;
pub mod subproblem;
pub mod system;

pub mod graph;
pub mod initial_condition; // Made public for T1.6 integration tests
pub mod input;
mod log;
pub mod output;
mod risk_measure;
pub mod scenario;
pub mod sddp;
pub mod utils;
use input::Input;
use std::error::Error;
use std::time::Instant;

/// Main entry point for production use with full JSON-based configuration.
///
/// This function uses the **low-level API** (manual graph construction) which supports:
/// - Complex seasonal structures with distribution-based uncertainty
/// - Markovian graph structures (not just linear paths)
/// - Per-node stochastic process configuration
/// - Normal/LogNormal distributions for loads and inflows
///
/// For simpler use cases (testing, benchmarks), consider using the
/// **Builder API** via `sddp::SddpAlgorithm::builder()` instead.
///
/// # Performance Notes
/// - This is the production entry point; performance is critical
/// - Uses pre-allocated structures where possible
/// - Leverages Rayon parallelism in train() and simulate()
/// - Optional CSV output (controlled by config.output_path)
pub fn run(input_args: &InputArgs) -> Result<(), Box<dyn Error>> {
    log::show_greeting();

    let begin = Instant::now();
    let input = Input::build(&input_args.path);
    let config = &input.config;
    let recourse = &input.recourse;
    let graph_input = &input.graph;

    log::input_reading_line(&input_args.path);

    let seed = config.seed;

    // Low-level API: Build graph from JSON configuration
    // This supports complex seasonal structures and distribution-based uncertainty
    let node_data_graph = graph_input.build_sddp_graph(&input.system)?;
    let initial_condition = recourse.build_sddp_initial_condition();

    // Generate SAA scenarios from distributions
    let saa = recourse.generate_sddp_noises(&node_data_graph, seed);

    // Create SDDP algorithm with low-level API
    // Note: Builder API (sddp::SddpAlgorithm::builder()) is available for simpler cases
    let mut sddp_algo =
        sddp::SddpAlgorithm::new(node_data_graph, initial_condition, seed)
            .unwrap();

    let _training_result = sddp_algo.train(
        config.num_iterations,
        config.num_forward_passes,
        &saa,
    )?;

    let simulation_handlers =
        sddp_algo.simulate(config.num_simulation_scenarios, &saa)?;

    log::output_generation_line(&input_args.path);
    output::generate_outputs(
        &sddp_algo.future_cost_function_graph,
        &simulation_handlers,
        &sddp_algo.study_period_ids,
        config.output_path.as_deref(),
    )?;

    log::show_farewell(begin.elapsed());

    Ok(())
}

pub struct InputArgs {
    pub path: String,
}

impl InputArgs {
    pub fn build(args: &[String]) -> Result<Self, &'static str> {
        if args.len() < 2 {
            return Err("Not enough arguments [PATH]");
        }

        let path = args[1].clone();

        Ok(Self { path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_args_build_success() {
        let args = vec!["program_name".to_string(), "some/path".to_string()];
        let input_args = InputArgs::build(&args).unwrap();
        assert_eq!(input_args.path, "some/path");
    }

    #[test]
    fn test_input_args_build_fail() {
        let args = vec!["program_name".to_string()];
        let result = InputArgs::build(&args);
        assert!(result.is_err());
        assert_eq!(result.err().unwrap(), "Not enough arguments [PATH]");
    }
}
