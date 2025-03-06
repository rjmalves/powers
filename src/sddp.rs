use highs;
use rand::prelude::*;
use rand_distr::{LogNormal, Uniform};
use rand_xoshiro::Xoshiro256Plus;
use std::ops::Range;
use std::time::{Duration, Instant};

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

fn set_solver_options(model: &mut highs::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
    model.set_option("time_limit", 300);
}

#[derive(Debug)]
struct BendersCut {
    pub coefficients: Vec<f64>,
    pub rhs: f64,
}

impl BendersCut {
    pub fn new(average_water_value: Vec<f64>, average_cost: f64) -> Self {
        Self {
            coefficients: average_water_value,
            rhs: average_cost,
        }
    }
}

pub struct Node {
    pub index: usize,
    pub system: System,
    subproblem: Subproblem,
}

impl Node {
    pub fn new(index: usize, system: System) -> Self {
        let subproblem = Subproblem::new(&system);
        Self {
            index,
            system,
            subproblem,
        }
    }
}

pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    pub fn new(node: Node) -> Self {
        Self { nodes: vec![node] }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn append(&mut self, node: Node) {
        self.nodes.push(node);
    }
}

fn forward_step<'a>(
    node: &Node,
    bus_loads: &'a Vec<f64>, // loads for stage 'index' ordered by id
    initial_storage: &Vec<f64>,
    hydros_inflow: &'a Vec<f64>, // inflows for stage 'index' ordered by id
) -> (highs::Solution, ResourceRealization<'a>) {
    let mut iter_subproblem = node.subproblem.clone();
    for (id, load) in bus_loads.iter().enumerate() {
        iter_subproblem.add_load_balance(&node.system, id, load);
    }
    for (id, storage) in initial_storage.iter().enumerate() {
        iter_subproblem.add_hydro_balance(
            &node.system,
            id,
            &hydros_inflow[id],
            storage,
        );
    }

    let mut model = iter_subproblem
        .template_problem
        .optimise(highs::Sense::Minimise);
    set_solver_options(&mut model);
    let solved = model.solve();
    match solved.status() {
        highs::HighsModelStatus::Optimal => (
            solved.get_solution(),
            ResourceRealization::new(
                bus_loads,
                initial_storage.clone(),
                hydros_inflow,
            ),
        ),
        _ => panic!("Error while solving forward model"),
    }
}

fn forward<'a>(
    nodes: &Vec<Node>,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>, // indexed by stage | hydro
) -> Trajectory<'a> {
    let mut realizations = Vec::<ResourceRealization>::new();
    let mut solutions = Vec::<highs::Solution>::new();
    let mut cost = 0.0;

    for (index, node) in nodes.iter().enumerate() {
        let node_initial_storage = if node.index == 0 {
            hydros_initial_storage.clone()
        } else {
            solutions
                .get(node.index - 1)
                .unwrap()
                .columns()
                .get(node.subproblem.raw_accessors.stored_volume.clone())
                .unwrap()
                .to_vec()
        };
        let (solution, realization) = forward_step(
            node,
            &bus_loads[index], // loads for stage 'index' ordered by id
            &node_initial_storage,
            &hydros_inflow[index], // inflows for stage 'index' ordered by id
        );
        cost += node.subproblem.get_stage_solution_cost(&solution);

        solutions.push(solution);
        realizations.push(realization);
    }
    Trajectory::new(realizations, solutions, cost)
}

fn backward(
    nodes: &mut Vec<Node>,
    trajectory: &Trajectory,
    saa: &Vec<Vec<Vec<f64>>>, // indexed by stage | branching | hydro
) -> f64 {
    let mut node_iter = nodes.iter_mut().rev().peekable();
    loop {
        let node = node_iter.next().unwrap();
        match node_iter.peek() {
            Some(_) => {
                let index = node.index;
                // println!("Beginning node {index} -------------------");
                let forward_realization =
                    trajectory.realizations.get(index).unwrap();
                let forward_initial_storages =
                    &forward_realization.hydros_initial_storage;
                // solve each branching and store solutions
                let mut solutions = Vec::<highs::Solution>::new();

                for hydros_stage_inflow in saa.get(index).unwrap().iter() {
                    let mut iter_subproblem = node.subproblem.clone();

                    for (id, load) in
                        forward_realization.bus_loads.iter().enumerate()
                    {
                        iter_subproblem.add_load_balance(
                            &node.system,
                            id,
                            load,
                        );
                    }
                    for (id, storage) in
                        forward_initial_storages.iter().enumerate()
                    {
                        iter_subproblem.add_hydro_balance(
                            &node.system,
                            id,
                            &hydros_stage_inflow.get(id).unwrap(),
                            storage,
                        );
                    }

                    let mut model = iter_subproblem
                        .template_problem
                        .optimise(highs::Sense::Minimise);
                    set_solver_options(&mut model);
                    // warm start from forward solution - does not work because
                    // of the added cut.
                    // model.set_solution(
                    //     Some(forward_solution.columns()),
                    //     Some(forward_solution.rows()),
                    //     Some(forward_solution.dual_columns()),
                    //     Some(forward_solution.dual_rows()),
                    // );
                    let solved = model.solve();

                    match solved.status() {
                        highs::HighsModelStatus::Optimal => {
                            solutions.push(solved.get_solution())
                        }
                        _ => panic!("Error while solving backward model"),
                    }
                }
                // adds all branching results
                let n_buses = node.system.buses.len();
                let n_hydros = node.system.hydros.len();
                let n_cuts = node.subproblem.num_cuts;
                let hydro_balance_accessor = (n_hydros + n_cuts + n_buses)
                    ..(n_cuts + n_buses + 2 * n_hydros);
                let n_branchings = solutions.len();
                let mut average_water_values = vec![0.0; n_hydros];
                let mut average_solution_cost = 0.0;
                for solution in solutions.iter() {
                    let water_values = solution
                        .dual_rows()
                        .get(hydro_balance_accessor.clone())
                        .unwrap();
                    for hydro_id in 0..n_hydros {
                        average_water_values[hydro_id] += water_values[hydro_id]
                    }
                    average_solution_cost +=
                        node.subproblem.get_total_solution_cost(solution);
                }

                // evaluate average cut
                average_solution_cost =
                    average_solution_cost / (n_branchings as f64);
                for hydro_id in 0..n_hydros {
                    average_water_values[hydro_id] =
                        average_water_values[hydro_id] / (n_branchings as f64);
                }

                let cut_rhs = average_solution_cost
                    - dot_product(
                        &average_water_values,
                        forward_initial_storages,
                    );
                let next_node = node_iter.peek_mut().unwrap();
                let cut =
                    BendersCut::new(average_water_values.clone(), cut_rhs);
                // println!("Node {} {:?}", next_node.index, cut);
                next_node.subproblem.add_cut(&cut);
            }
            None => {
                let index = node.index;
                // println!("Beginning node {index} -------------------");
                let forward_realization =
                    trajectory.realizations.get(index).unwrap();
                let forward_initial_storages =
                    &forward_realization.hydros_initial_storage;
                // solve each branching and store solutions
                let mut solutions = Vec::<highs::Solution>::new();

                for hydros_stage_inflow in saa.get(index).unwrap().iter() {
                    let mut iter_subproblem = node.subproblem.clone();

                    for (id, load) in
                        forward_realization.bus_loads.iter().enumerate()
                    {
                        iter_subproblem.add_load_balance(
                            &node.system,
                            id,
                            load,
                        );
                    }
                    for (id, storage) in
                        forward_initial_storages.iter().enumerate()
                    {
                        iter_subproblem.add_hydro_balance(
                            &node.system,
                            id,
                            &hydros_stage_inflow.get(id).unwrap(),
                            storage,
                        );
                    }

                    let mut model = iter_subproblem
                        .template_problem
                        .optimise(highs::Sense::Minimise);
                    set_solver_options(&mut model);
                    // warm start from forward solution - does not work because
                    // of the added cut.
                    // model.set_solution(
                    //     Some(forward_solution.columns()),
                    //     Some(forward_solution.rows()),
                    //     Some(forward_solution.dual_columns()),
                    //     Some(forward_solution.dual_rows()),
                    // );
                    let solved = model.solve();

                    match solved.status() {
                        highs::HighsModelStatus::Optimal => {
                            solutions.push(solved.get_solution())
                        }
                        _ => panic!("Error while solving backward model"),
                    }
                }
                // adds all branching results
                let n_branchings = solutions.len();
                let mut average_solution_cost = 0.0;
                for solution in solutions.iter() {
                    average_solution_cost +=
                        node.subproblem.get_total_solution_cost(solution);
                }
                // runs first-stage cost estimate using new added cuts
                return average_solution_cost / (n_branchings as f64);
            }
        }
    }
}

fn iterate<'a>(
    graph: &mut Graph,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>,
    saa: &Vec<Vec<Vec<f64>>>,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory = forward(
        &graph.nodes,
        bus_loads,
        hydros_initial_storage,
        hydros_inflow,
    );

    let trajectory_cost = trajectory.cost;
    let first_stage_cost_estimate =
        backward(&mut graph.nodes, &trajectory, saa);

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_cost_estimate, iteration_time);
}

/// Generates an SAA indexed by stage | branching | hydro
fn generate_saa<'a>(
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
    num_branchings: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let mut saa: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![]; num_branchings]; scenario_generator.len()];
    for (stage_index, stage_generator) in scenario_generator.iter().enumerate()
    {
        let hydro_inflows: Vec<Vec<f64>> = stage_generator
            .iter()
            .map(|hydro_generator| {
                hydro_generator
                    .sample_iter(&mut rng)
                    .take(num_branchings)
                    .collect()
            })
            .collect();
        for branching_index in 0..num_branchings {
            for inflows in hydro_inflows.iter() {
                saa[stage_index][branching_index]
                    .push(inflows[branching_index]);
            }
        }
    }
    saa
}

fn training_table_header() {
    println!(
        "{0: ^10} | {1: ^15} | {2: ^14} | {3: ^12}",
        "iteration", "lower bound ($)", "simulation ($)", "time (s)"
    )
}

fn training_table_divider() {
    println!("------------------------------------------------------------")
}

fn training_table_row(
    iteration: usize,
    lower_bound: f64,
    simulation: f64,
    time: Duration,
) {
    println!(
        "{0: >10} | {1: >15.4} | {2: >14.4} | {3: >12.2}",
        iteration,
        lower_bound,
        simulation,
        time.as_millis() as f64 / 1000.0
    )
}

pub fn train<'a>(
    graph: &mut Graph,
    num_iterations: usize,
    num_branchings: usize,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
) {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);
    let forward_indices_dist =
        Uniform::<usize>::try_from(0..num_branchings).unwrap();

    let saa = generate_saa(scenario_generator, num_branchings);

    training_table_header();
    training_table_divider();

    for index in 0..num_iterations {
        // Generates indices for sampling inflows, indexed by stage
        let forward_branching_indices: Vec<usize> = forward_indices_dist
            .sample_iter(&mut rng)
            .take(scenario_generator.len())
            .collect();

        let hydros_inflow = saa
            .iter()
            .enumerate()
            .map(|(index, stage_inflows)| {
                &stage_inflows[forward_branching_indices[index]]
            })
            .collect();

        let (simulation, lower_bound, time) = iterate(
            graph,
            bus_loads,
            hydros_initial_storage,
            &hydros_inflow,
            &saa,
        );

        training_table_row(index + 1, lower_bound, simulation, time);
    }

    training_table_divider();
}

#[derive(Copy, Clone)]
struct Bus {
    pub id: usize,
    pub deficit_cost: f64,
}

#[derive(Copy, Clone)]
struct Line {
    pub id: usize,
    pub source_bus_id: usize,
    pub target_bus_id: usize,
    pub capacity: f64,
    pub exchange_penalty: f64,
}

#[derive(Copy, Clone)]
struct Thermal {
    pub id: usize,
    pub bus_id: usize,
    pub cost: f64,
    pub min_generation: f64,
    pub max_generation: f64,
}

#[derive(Copy, Clone)]
struct Hydro {
    pub id: usize,
    pub downstream_hydro_id: Option<usize>,
    pub bus_id: usize,
    pub productivity: f64,
    pub min_storage: f64,
    pub max_storage: f64,
    pub min_generation: f64,
    pub max_generation: f64,
    pub spillage_penalty: f64,
}

#[derive(Clone)]
pub struct System {
    pub buses: Vec<Bus>,
    pub lines: Vec<Line>,
    pub thermals: Vec<Thermal>,
    pub hydros: Vec<Hydro>,
}

impl System {
    pub fn default() -> Self {
        let buses = vec![Bus {
            id: 0,
            deficit_cost: 50.0,
        }];
        let lines = Vec::<Line>::new();

        let thermals = vec![
            Thermal {
                id: 0,
                bus_id: 0,
                cost: 5.0,
                min_generation: 0.0,
                max_generation: 15.0,
            },
            Thermal {
                id: 1,
                bus_id: 0,
                cost: 10.0,
                min_generation: 0.0,
                max_generation: 15.0,
            },
        ];
        let hydros = vec![Hydro {
            id: 0,
            downstream_hydro_id: None,
            bus_id: 0,
            productivity: 1.0,
            min_storage: 0.0,
            max_storage: 100.0,
            min_generation: 0.0,
            max_generation: 60.0,
            spillage_penalty: 0.01,
        }];

        Self {
            buses,
            lines,
            thermals,
            hydros,
        }
    }
}

#[derive(Clone)]
struct HighsAccessors {
    pub deficit: Vec<highs::Col>,
    pub direct_exchange: Vec<highs::Col>,
    pub reverse_exchange: Vec<highs::Col>,
    pub thermal_gen: Vec<highs::Col>,
    pub turbined_flow: Vec<highs::Col>,
    pub spillage: Vec<highs::Col>,
    pub stored_volume: Vec<highs::Col>,
    pub min_generation_slack: Vec<highs::Col>,
    pub alpha: highs::Col,
}

#[derive(Clone)]
struct RawAccessors {
    pub deficit: Range<usize>,
    pub direct_exchange: Range<usize>,
    pub reverse_exchange: Range<usize>,
    pub thermal_gen: Range<usize>,
    pub turbined_flow: Range<usize>,
    pub spillage: Range<usize>,
    pub stored_volume: Range<usize>,
    pub min_generation_slack: Range<usize>,
}

#[derive(Clone)]
struct Subproblem {
    template_problem: highs::RowProblem,
    highs_accessors: HighsAccessors,
    raw_accessors: RawAccessors,
    cost_vector: Vec<f64>,
    num_cuts: usize,
}

impl Subproblem {
    pub fn new(system: &System) -> Self {
        let mut pb = highs::RowProblem::new();

        const MIN_GENERATION_PENALTY: f64 = 50.050;

        // VARIABLES
        let deficit: Vec<highs::Col> = system
            .buses
            .iter()
            .map(|bus| pb.add_column(bus.deficit_cost, 0..))
            .collect();
        let direct_exchange = Vec::<highs::Col>::new();
        let reverse_exchange = Vec::<highs::Col>::new();
        let thermal_gen: Vec<highs::Col> = system
            .thermals
            .iter()
            .map(|thermal| {
                pb.add_column(
                    thermal.cost,
                    thermal.min_generation..thermal.max_generation,
                )
            })
            .collect();
        let turbined_flow: Vec<highs::Col> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(
                    0.0,
                    (hydro.min_generation / hydro.productivity)
                        ..(hydro.max_generation / hydro.productivity),
                )
            })
            .collect();
        let spillage = system
            .hydros
            .iter()
            .map(|hydro| pb.add_column(hydro.spillage_penalty, 0.0..))
            .collect();
        let stored_volume = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(0.0, hydro.min_storage..hydro.max_storage)
            })
            .collect();
        let min_generation_slack: Vec<highs::Col> = system
            .hydros
            .iter()
            .map(|_hydro| pb.add_column(MIN_GENERATION_PENALTY, 0.0..))
            .collect();

        let alpha = pb.add_column(1.0, 0.0..);

        // RAW ACCESSORS BY INDEX - TODO: obtain this in a better way from
        // defining the variables above
        let n_buses = system.buses.len();
        let n_lines = system.lines.len();
        let n_thermals = system.thermals.len();
        let n_hydros = system.hydros.len();
        let deficit_raw = 0..n_buses;
        let direct_exchange_raw = deficit_raw.end..(deficit_raw.end + n_lines);
        let reverse_exchange_raw =
            direct_exchange_raw.end..(direct_exchange_raw.end + n_lines);
        let thermal_gen_raw =
            reverse_exchange_raw.end..(reverse_exchange_raw.end + n_thermals);
        let turbined_flow_raw =
            thermal_gen_raw.end..(thermal_gen_raw.end + n_hydros);
        let spillage_raw =
            turbined_flow_raw.end..(turbined_flow_raw.end + n_hydros);
        let stored_volume_raw = spillage_raw.end..(spillage_raw.end + n_hydros);
        let min_generation_slack_raw =
            stored_volume_raw.end..(stored_volume_raw.end + n_hydros);

        // COST VECTOR BY INDEX - TODO: obtain this in a better way from
        // defining the variables above
        let mut cost_vector = Vec::<f64>::new();
        for bus in system.buses.iter() {
            cost_vector.push(bus.deficit_cost);
        }
        for line in system.lines.iter() {
            cost_vector.push(line.exchange_penalty);
        }
        for line in system.lines.iter() {
            cost_vector.push(line.exchange_penalty);
        }
        for thermal in system.thermals.iter() {
            cost_vector.push(thermal.cost);
        }
        for _hydro in system.hydros.iter() {
            cost_vector.push(0.0);
        }
        for hydro in system.hydros.iter() {
            cost_vector.push(hydro.spillage_penalty);
        }
        for _hydro in system.hydros.iter() {
            cost_vector.push(0.0);
        }
        for _hydro in system.hydros.iter() {
            cost_vector.push(MIN_GENERATION_PENALTY);
        }
        cost_vector.push(1.0);

        for hydro in system.hydros.iter() {
            pb.add_row(
                hydro.min_generation..,
                &[
                    (turbined_flow[hydro.id], 1.0),
                    (min_generation_slack[hydro.id], 1.0),
                ],
            );
        }

        Subproblem {
            template_problem: pb,
            highs_accessors: HighsAccessors {
                deficit,
                direct_exchange,
                reverse_exchange,
                thermal_gen,
                turbined_flow,
                spillage,
                stored_volume,
                min_generation_slack,
                alpha,
            },
            raw_accessors: RawAccessors {
                deficit: deficit_raw,
                direct_exchange: direct_exchange_raw,
                reverse_exchange: reverse_exchange_raw,
                thermal_gen: thermal_gen_raw,
                turbined_flow: turbined_flow_raw,
                spillage: spillage_raw,
                stored_volume: stored_volume_raw,
                min_generation_slack: min_generation_slack_raw,
            },
            cost_vector,
            num_cuts: 0,
        }
    }

    pub fn add_load_balance(
        &mut self,
        system: &System,
        bus_id: usize,
        load: &f64,
    ) {
        let bus_thermals: Vec<&Thermal> = system
            .thermals
            .iter()
            .filter(|thermal| thermal.bus_id == bus_id)
            .collect();
        let bus_hydros: Vec<&Hydro> = system
            .hydros
            .iter()
            .filter(|hydro| hydro.bus_id == bus_id)
            .collect();
        // TODO - add exchange
        let mut factors = vec![(self.highs_accessors.deficit[bus_id], 1.0)];
        for thermal in bus_thermals {
            factors.push((self.highs_accessors.thermal_gen[thermal.id], 1.0));
        }
        for hydro in bus_hydros {
            factors.push((
                self.highs_accessors.turbined_flow[hydro.id],
                hydro.productivity,
            ));
        }
        self.template_problem.add_row(*load..*load, &factors);
    }

    pub fn add_hydro_balance(
        &mut self,
        system: &System,
        hydro_id: usize,
        inflow: &f64,
        initial_storage: &f64,
    ) {
        let upstream_hydros: Vec<&Hydro> = system
            .hydros
            .iter()
            .filter(|hydro| hydro.downstream_hydro_id == Some(hydro_id))
            .collect();

        let mut factors: Vec<(highs::Col, f64)> = vec![
            (self.highs_accessors.stored_volume[hydro_id], 1.0),
            (self.highs_accessors.turbined_flow[hydro_id], 1.0),
            (self.highs_accessors.spillage[hydro_id], 1.0),
        ];
        for hydro in upstream_hydros {
            factors.push((self.highs_accessors.turbined_flow[hydro.id], -1.0));
            factors.push((self.highs_accessors.spillage[hydro.id], -1.0));
        }
        let rhs = inflow + initial_storage;
        self.template_problem.add_row(rhs..rhs, &factors);
    }

    pub fn get_total_solution_cost(&self, solution: &highs::Solution) -> f64 {
        let cols = solution.columns();
        dot_product(&self.cost_vector, &cols)
    }
    pub fn get_stage_solution_cost(&self, solution: &highs::Solution) -> f64 {
        let cols = solution.columns();
        dot_product(
            &self.cost_vector[..cols.len() - 1],
            &cols[..cols.len() - 1],
        )
    }

    pub fn add_cut(&mut self, cut: &BendersCut) {
        let mut factors: Vec<(highs::Col, f64)> =
            vec![(self.highs_accessors.alpha, 1.0)];
        for (hydro_id, stored_volume) in
            self.highs_accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        self.template_problem.add_row(cut.rhs.., &factors);
        self.num_cuts += 1;
    }
}

#[derive(Clone, Debug)]
struct ResourceRealization<'a> {
    pub bus_loads: &'a Vec<f64>,
    pub hydros_initial_storage: Vec<f64>,
    pub hydros_inflow: &'a Vec<f64>,
}
impl<'a> ResourceRealization<'a> {
    pub fn new(
        bus_loads: &'a Vec<f64>,
        hydros_initial_storage: Vec<f64>,
        hydros_inflow: &'a Vec<f64>,
    ) -> Self {
        Self {
            bus_loads,
            hydros_initial_storage,
            hydros_inflow,
        }
    }
}

#[derive(Clone, Debug)]
struct Trajectory<'a> {
    pub realizations: Vec<ResourceRealization<'a>>,
    pub solutions: Vec<highs::Solution>,
    pub cost: f64,
}
impl<'a> Trajectory<'a> {
    pub fn new(
        realizations: Vec<ResourceRealization<'a>>,
        solutions: Vec<highs::Solution>,
        cost: f64,
    ) -> Self {
        Self {
            realizations,
            solutions,
            cost,
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_default_system() {
        let system = System::default();
        assert_eq!(system.buses.len(), 1);
        assert_eq!(system.lines.len(), 0);
        assert_eq!(system.thermals.len(), 2);
        assert_eq!(system.hydros.len(), 1);
    }

    #[test]
    fn test_create_subproblem_with_default_system() {
        let system = System::default();
        let subproblem = Subproblem::new(&system);
        assert_eq!(subproblem.highs_accessors.deficit.len(), 1);
        assert_eq!(subproblem.highs_accessors.direct_exchange.len(), 0);
        assert_eq!(subproblem.highs_accessors.reverse_exchange.len(), 0);
        assert_eq!(subproblem.highs_accessors.thermal_gen.len(), 2);
        assert_eq!(subproblem.highs_accessors.turbined_flow.len(), 1);
        assert_eq!(subproblem.highs_accessors.spillage.len(), 1);
        assert_eq!(subproblem.highs_accessors.stored_volume.len(), 1);
        assert_eq!(subproblem.highs_accessors.min_generation_slack.len(), 1);
    }

    #[test]
    fn test_solve_subproblem_with_default_system() {
        let system = System::default();
        let mut subproblem = Subproblem::new(&system);
        let inflow = 0.0;
        let initial_storage = 83.333;
        let load = 50.0;
        subproblem.add_load_balance(&system, 0, &load);
        subproblem.add_hydro_balance(&system, 0, &inflow, &initial_storage);

        let mut model =
            subproblem.template_problem.optimise(highs::Sense::Minimise);
        set_solver_options(&mut model);
        let solved = model.solve();
        assert_eq!(solved.status(), highs::HighsModelStatus::Optimal);
    }

    #[test]
    fn test_get_solution_cost_with_default_system() {
        let system = System::default();
        let mut subproblem = Subproblem::new(&system);
        let inflow = 0.0;
        let initial_storage = 23.333;
        let load = 50.0;
        subproblem.add_load_balance(&system, 0, &load);
        subproblem.add_hydro_balance(&system, 0, &inflow, &initial_storage);

        let mut model = subproblem
            .clone()
            .template_problem
            .optimise(highs::Sense::Minimise);
        set_solver_options(&mut model);
        let solved = model.solve();
        let solution = solved.get_solution();
        assert_eq!(
            subproblem.get_stage_solution_cost(&solution),
            191.67000000000002
        );
    }

    #[test]
    fn test_create_node() {
        let node = Node::new(0, System::default());
        assert_eq!(node.index, 0);
    }

    #[test]
    fn test_create_graph() {
        let node0 = Node::new(0, System::default());
        let graph = Graph::new(node0);
        assert_eq!(graph.len(), 1);
    }

    #[test]
    fn test_append_node_to_graph() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_forward_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        forward(
            &graph.nodes,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
        );
    }

    #[test]
    fn test_backward_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let trajectory = forward(
            &graph.nodes,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
        );
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        backward(&mut graph.nodes, &trajectory, &branchings);
    }

    #[test]
    fn test_iterate_with_default_system() {
        let node0 = Node::new(0, System::default());
        let node1 = Node::new(1, System::default());
        let node2 = Node::new(2, System::default());
        let mut graph = Graph::new(node0);
        graph.append(node1);
        graph.append(node2);
        let bus_loads = vec![vec![75.0], vec![75.0], vec![75.0]];
        let hydros_initial_storage = vec![83.222];
        let example_inflow = vec![10.0];
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let branchings = vec![
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
            vec![vec![5.0], vec![10.0], vec![15.0]],
        ];
        iterate(
            &mut graph,
            &bus_loads,
            &hydros_initial_storage,
            &hydros_inflow,
            &branchings,
        );
    }

    #[test]
    fn test_train_with_default_system() {
        let node0 = Node::new(0, System::default());
        let mut graph = Graph::new(node0);
        let mut scenario_generator: Vec<Vec<LogNormal<f64>>> =
            vec![vec![LogNormal::new(3.6, 0.6928).unwrap()]];
        let mut bus_loads = vec![vec![75.0]];
        for n in 1..4 {
            let node = Node::new(n, System::default());
            graph.append(node);
            scenario_generator.push(vec![LogNormal::new(3.6, 0.6928).unwrap()]);
            bus_loads.push(vec![75.0]);
        }
        let hydros_initial_storage = vec![83.222];
        train(
            &mut graph,
            128,
            10,
            &bus_loads,
            &hydros_initial_storage,
            &scenario_generator,
        );
    }
}
