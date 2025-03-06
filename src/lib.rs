mod sddp;
use rand_distr::LogNormal;
use std::error::Error;

pub fn run(
    num_stages: usize,
    num_iterations: usize,
    num_branchings: usize,
) -> Result<(), Box<dyn Error>> {
    let node0 = sddp::Node::new(0, sddp::System::default());
    let mut graph = sddp::Graph::new(node0);
    let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
        vec![vec![LogNormal::new(3.6, 0.6928).unwrap()]];
    let mut bus_loads = vec![vec![75.0]];
    for n in 1..num_stages {
        let node = sddp::Node::new(n, sddp::System::default());
        graph.append(node);
        scenario_generator.push(vec![LogNormal::new(3.6, 0.6928).unwrap()]);
        bus_loads.push(vec![75.0]);
    }
    let hydros_initial_storage = vec![83.222];
    sddp::train(
        &mut graph,
        num_iterations,
        num_branchings,
        &bus_loads,
        &hydros_initial_storage,
        &scenario_generator,
    );
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
