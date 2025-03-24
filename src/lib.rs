pub mod input;
pub mod sddp;
mod solver;
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
    let root = sddp::Node::new(0, input.system.build_sddp_system());

    input_reading_line(&input_args.path);

    let mut graph = sddp::Graph::new(root);

    for n in 1..config.num_stages {
        let node = sddp::Node::new(n, input.system.build_sddp_system());
        graph.append(node);
    }
    let hydros_initial_storage = recourse.build_sddp_initial_storages();
    let bus_loads = recourse.build_sddp_loads(config.num_stages);
    let scenario_generator =
        recourse.build_sddp_scenario_generator(config.num_stages);
    sddp::train(
        &mut graph,
        config.num_iterations,
        config.num_branchings,
        &bus_loads,
        &hydros_initial_storage,
        &scenario_generator,
    );
    sddp::simulate(
        &mut graph,
        config.num_simulation_scenarios,
        &bus_loads,
        &hydros_initial_storage,
        &scenario_generator,
    );

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
