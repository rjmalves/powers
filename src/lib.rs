mod cut;
mod fcf;
pub mod graph;
mod initial_condition;
pub mod input;
mod log;
pub mod output;
mod risk_measure;
pub mod scenario;
pub mod sddp;
mod solver;
mod state;
mod stochastic_process;
pub mod subproblem;
mod system;
pub mod utils;
use input::Input;
use std::error::Error;
use std::time::Instant;

pub fn run(input_args: &InputArgs) -> Result<(), Box<dyn Error>> {
    log::show_greeting();

    let begin = Instant::now();
    let input = Input::build(&input_args.path);
    let config = &input.config;
    let recourse = &input.recourse;
    let graph_input = &input.graph;

    log::input_reading_line(&input_args.path);

    let seed = config.seed;

    let node_data_graph = graph_input.build_sddp_graph(&input.system)?;
    let initial_condition = recourse.build_sddp_initial_condition();

    let saa = recourse.generate_sddp_noises(&node_data_graph, seed);

    let mut sddp_algo =
        sddp::SddpAlgorithm::new(node_data_graph, initial_condition, seed)
            .unwrap();

    sddp_algo.train(config.num_iterations, config.num_forward_passes, &saa)?;

    let simulation_handlers =
        sddp_algo.simulate(config.num_simulation_scenarios, &saa)?;

    log::output_generation_line(&input_args.path);
    output::generate_outputs(
        &sddp_algo.future_cost_function_graph,
        &simulation_handlers,
        &sddp_algo.study_period_ids,
        &input_args.path,
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