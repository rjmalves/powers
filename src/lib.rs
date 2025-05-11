mod cut;
mod fcf;
pub mod graph;
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
use std::time::{Duration, Instant};

fn show_greeting() {
    println!(
        "\nPOWE.RS - Power Optimization for the World of Energy - in pure RuSt"
    );
    println!(
        "--------------------------------------------------------------------"
    );
}

fn input_reading_line(input_path: &str) {
    println!("\nReading input files from '{}'", input_path);
}

fn output_generation_line(input_path: &str) {
    println!("\nWriting outputs to '{}'", input_path);
}

fn show_farewell(time: Duration) {
    println!(
        "\nTotal running time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}

pub fn run(input_args: &InputArgs) -> Result<(), Box<dyn Error>> {
    show_greeting();

    let begin = Instant::now();
    let input = Input::build(&input_args.path);
    let config = &input.config;
    let recourse = &input.recourse;
    let graph_input = &input.graph;

    input_reading_line(&input_args.path);

    let seed = 0;

    let mut g = graph_input.build_sddp_graph(&input.system);

    g.get_node_mut(0).unwrap().data.state = recourse.build_sddp_initial_state();

    let load_saa = recourse.generate_sddp_load_noises(&g, seed);
    let inflow_saa = recourse.generate_sddp_inflow_noises(&g, seed);
    sddp::train(&mut g, config.num_iterations, &load_saa, &inflow_saa);
    let trajectories = sddp::simulate(
        &mut g,
        config.num_simulation_scenarios,
        &load_saa,
        &inflow_saa,
    );

    output_generation_line(&input_args.path);
    output::generate_outputs(&g, &trajectories, &input_args.path)?;

    show_farewell(begin.elapsed());

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
