mod myhighs;
pub mod sddp;
use rand_distr::LogNormal;
use std::error::Error;
use std::time::{Duration, Instant};

fn show_greeting() {
    println!(
        "\nPOWE.RS - Power Optimization for the World of Energy - in pure RuSt"
    );
    println!(
        "-----------------------------------------------------------------"
    );
}

fn show_farewell(time: Duration) {
    println!(
        "\nTotal running time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}

pub fn run(
    num_stages: usize,
    num_iterations: usize,
    num_branchings: usize,
) -> Result<(), Box<dyn Error>> {
    let begin = Instant::now();

    show_greeting();

    let mu = 3.6;
    let sigma = 0.6928;
    let load = 75.0;
    let x0 = 83.222;

    let root = sddp::Node::new(0, sddp::System::default());
    let mut graph = sddp::Graph::new(root);
    let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
        vec![vec![LogNormal::new(mu, sigma).unwrap()]];
    let mut bus_loads = vec![vec![load]];
    for n in 1..num_stages {
        let node = sddp::Node::new(n, sddp::System::default());
        graph.append(node);
        scenario_generator.push(vec![LogNormal::new(mu, sigma).unwrap()]);
        bus_loads.push(vec![load]);
    }
    let hydros_initial_storage = vec![x0];
    sddp::train(
        &mut graph,
        num_iterations,
        num_branchings,
        &bus_loads,
        &hydros_initial_storage,
        &scenario_generator,
    );

    show_farewell(begin.elapsed());

    Ok(())
}

pub struct Config {
    pub num_stages: usize,
    pub num_iterations: usize,
    pub num_branchings: usize,
}

impl Config {
    pub fn build(args: &[String]) -> Result<Self, &'static str> {
        if args.len() < 4 {
            return Err(
                "Not enough arguments [N_STAGES, N_ITERATIONS, N_BRANCHINGS]",
            );
        }

        let num_stages: usize = args[1].clone().parse::<usize>().unwrap_or(4);
        let num_iterations: usize =
            args[2].clone().parse::<usize>().unwrap_or(32);
        let num_branchings: usize =
            args[3].clone().parse::<usize>().unwrap_or(10);

        Ok(Self {
            num_stages,
            num_iterations,
            num_branchings,
        })
    }
}
