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
mod subproblem;
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

    let mut g = graph_input.build_sddp_graph(&input.system);

    let initial_condition = recourse.build_sddp_initial_condition();

    // sddp::set_initial_condition(&mut g, initial_condition);

    let saa = recourse.generate_sddp_noises(&g, seed);
    sddp::train(&mut g, config.num_iterations, &initial_condition, &saa);
    let trajectories = sddp::simulate(
        &mut g,
        config.num_simulation_scenarios,
        &initial_condition,
        &saa,
    );

    log::output_generation_line(&input_args.path);
    output::generate_outputs(&g, &trajectories, &input_args.path)?;

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
