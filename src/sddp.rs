use crate::myhighs;
use rand::prelude::*;
use rand_distr::{LogNormal, Uniform};
use rand_xoshiro::Xoshiro256Plus;
use std::ops::Range;
use std::time::{Duration, Instant};

/// Helper function for evaluating the dot product between two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut product = 0.0;
    for i in 0..a.len() {
        product += a[i] * b[i];
    }
    product
}

/// Helper function for setting the same solver options on
/// every solved problem.
fn set_solver_options(model: &mut myhighs::Model) {
    model.set_option("presolve", "off");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
    model.set_option("time_limit", 300);
}

/// Helper function that solves a problem using the HiGHS solver with
/// some predefined options and returns the solved problem.
fn solve(pb: myhighs::Problem) -> myhighs::SolvedModel {
    let mut model = pb.optimise(myhighs::Sense::Minimise);
    set_solver_options(&mut model);
    model.solve()
}

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

    fn get_final_stored_volume_from_solutions(
        &self,
        solutions: &Vec<myhighs::Solution>,
    ) -> Vec<f64> {
        let volume_indices = &self.subproblem.accessors.stored_volume;
        let first = volume_indices.first().unwrap();
        let last = volume_indices.last().unwrap() + 1;
        solutions.last().unwrap().columns()[*first..last].to_vec()
    }

    fn hydro_balance_accessor(&self) -> Range<usize> {
        let n_buses = self.system.buses_count;
        let n_hydros = self.system.hydros_count;
        let n_cuts = self.subproblem.num_cuts;
        (n_hydros + n_cuts + n_buses)..(n_cuts + n_buses + 2 * n_hydros)
    }

    fn subproblem_with_uncertainties<'a>(
        &self,
        bus_loads: &'a Vec<f64>,
        initial_storage: &Vec<f64>,
        hydros_inflow: &'a Vec<f64>,
    ) -> Subproblem {
        let mut sp = self.subproblem.clone();
        for (id, load) in bus_loads.iter().enumerate() {
            sp.add_load_balance(&self.system, id, load);
        }
        for (id, storage) in initial_storage.iter().enumerate() {
            sp.add_hydro_balance(&self.system, id, &hydros_inflow[id], storage);
        }
        sp
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

#[derive(Clone)]
struct Bus {
    id: usize,
    deficit_cost: f64,
    hydro_ids: Vec<usize>,
    thermal_ids: Vec<usize>,
    source_line_ids: Vec<usize>,
    target_line_ids: Vec<usize>,
}

impl Bus {
    pub fn new(id: usize, deficit_cost: f64) -> Self {
        Self {
            id,
            deficit_cost,
            hydro_ids: vec![],
            thermal_ids: vec![],
            source_line_ids: vec![],
            target_line_ids: vec![],
        }
    }

    pub fn add_hydro(&mut self, hydro_id: usize) {
        self.hydro_ids.push(hydro_id);
    }

    pub fn add_thermal(&mut self, thermal_id: usize) {
        self.thermal_ids.push(thermal_id);
    }

    pub fn add_source_line(&mut self, line_id: usize) {
        self.source_line_ids.push(line_id);
    }

    pub fn add_target_line(&mut self, line_id: usize) {
        self.target_line_ids.push(line_id);
    }
}

#[derive(Clone)]
struct Line {
    pub id: usize,
    pub source_bus_id: usize,
    pub target_bus_id: usize,
    pub direct_capacity: f64,
    pub reverse_capacity: f64,
    pub exchange_penalty: f64,
}

impl Line {
    pub fn new(
        id: usize,
        source_bus_id: usize,
        target_bus_id: usize,
        direct_capacity: f64,
        reverse_capacity: f64,
        exchange_penalty: f64,
    ) -> Self {
        Self {
            id,
            source_bus_id,
            target_bus_id,
            direct_capacity,
            reverse_capacity,
            exchange_penalty,
        }
    }
}

#[derive(Clone)]
struct Thermal {
    pub id: usize,
    pub bus_id: usize,
    pub cost: f64,
    pub min_generation: f64,
    pub max_generation: f64,
}

impl Thermal {
    pub fn new(
        id: usize,
        bus_id: usize,
        cost: f64,
        min_generation: f64,
        max_generation: f64,
    ) -> Self {
        Self {
            id,
            bus_id,
            cost,
            min_generation,
            max_generation,
        }
    }
}

#[derive(Clone)]
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
    pub upstream_hydro_ids: Vec<usize>,
}

impl Hydro {
    pub fn new(
        id: usize,
        downstream_hydro_id: Option<usize>,
        bus_id: usize,
        productivity: f64,
        min_storage: f64,
        max_storage: f64,
        min_generation: f64,
        max_generation: f64,
        spillage_penalty: f64,
    ) -> Self {
        Self {
            id,
            downstream_hydro_id,
            bus_id,
            productivity,
            min_storage,
            max_storage,
            min_generation,
            max_generation,
            spillage_penalty,
            upstream_hydro_ids: vec![],
        }
    }

    pub fn add_upstream_hydro(&mut self, hydro_id: usize) {
        self.upstream_hydro_ids.push(hydro_id);
    }
}

#[derive(Clone)]
pub struct System {
    buses: Vec<Bus>,
    lines: Vec<Line>,
    thermals: Vec<Thermal>,
    hydros: Vec<Hydro>,
    buses_count: usize,
    lines_count: usize,
    thermals_count: usize,
    hydros_count: usize,
}

impl System {
    pub fn default() -> Self {
        let mut buses = vec![Bus::new(0, 50.0)];
        let lines: Vec<Line> = vec![];

        let thermals = vec![
            Thermal::new(0, 0, 5.0, 0.0, 15.0),
            Thermal::new(1, 0, 10.0, 0.0, 15.0),
        ];
        for t in thermals.iter() {
            buses.get_mut(0).unwrap().thermal_ids.push(t.id);
        }

        let hydros =
            vec![Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01)];
        for h in hydros.iter() {
            buses.get_mut(0).unwrap().hydro_ids.push(h.id);
        }

        Self {
            buses,
            lines,
            thermals,
            hydros,
            buses_count: 1,
            lines_count: 0,
            thermals_count: 2,
            hydros_count: 1,
        }
    }
}

#[derive(Clone, Debug)]
struct Accessors {
    pub deficit: Vec<usize>,
    pub direct_exchange: Vec<usize>,
    pub reverse_exchange: Vec<usize>,
    pub thermal_gen: Vec<usize>,
    pub turbined_flow: Vec<usize>,
    pub spillage: Vec<usize>,
    pub stored_volume: Vec<usize>,
    pub min_generation_slack: Vec<usize>,
    pub alpha: usize,
}

#[derive(Clone)]
struct Subproblem {
    template_problem: myhighs::Problem,
    accessors: Accessors,
    cost_vector: Vec<f64>,
    num_cuts: usize,
}

impl Subproblem {
    pub fn new(system: &System) -> Self {
        let mut pb = myhighs::Problem::new();

        const MIN_GENERATION_PENALTY: f64 = 50.050;

        // VARIABLES
        let deficit: Vec<usize> = system
            .buses
            .iter()
            .map(|bus| pb.add_column(bus.deficit_cost, 0.0..))
            .collect();
        let direct_exchange: Vec<usize> = system
            .lines
            .iter()
            .map(|line| {
                pb.add_column(line.exchange_penalty, 0.0..line.direct_capacity)
            })
            .collect();
        let reverse_exchange: Vec<usize> = system
            .lines
            .iter()
            .map(|line| {
                pb.add_column(line.exchange_penalty, 0.0..line.reverse_capacity)
            })
            .collect();
        let thermal_gen: Vec<usize> = system
            .thermals
            .iter()
            .map(|thermal| {
                pb.add_column(
                    thermal.cost,
                    thermal.min_generation..thermal.max_generation,
                )
            })
            .collect();
        let turbined_flow: Vec<usize> = system
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
        let spillage: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| pb.add_column(hydro.spillage_penalty, 0.0..))
            .collect();
        let stored_volume: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(0.0, hydro.min_storage..hydro.max_storage)
            })
            .collect();
        let min_generation_slack: Vec<usize> = system
            .hydros
            .iter()
            .map(|_hydro| pb.add_column(MIN_GENERATION_PENALTY, 0.0..))
            .collect();

        let alpha = pb.add_column(1.0, 0.0..);

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
            accessors: Accessors {
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
        let bus = system.buses.get(bus_id).unwrap();

        let mut factors = vec![(self.accessors.deficit[bus_id], 1.0)];
        for thermal_id in bus.thermal_ids.iter() {
            factors.push((self.accessors.thermal_gen[*thermal_id], 1.0));
        }
        for hydro_id in bus.hydro_ids.iter() {
            factors.push((
                self.accessors.turbined_flow[*hydro_id],
                system.hydros.get(*hydro_id).unwrap().productivity,
            ));
        }
        for line_id in bus.source_line_ids.iter() {
            factors.push((self.accessors.reverse_exchange[*line_id], 1.0));
            factors.push((self.accessors.direct_exchange[*line_id], -1.0));
        }
        for line_id in bus.target_line_ids.iter() {
            factors.push((self.accessors.direct_exchange[*line_id], 1.0));
            factors.push((self.accessors.reverse_exchange[*line_id], -1.0));
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
        let hydro = system.hydros.get(hydro_id).unwrap();
        let mut factors: Vec<(usize, f64)> = vec![
            (self.accessors.stored_volume[hydro_id], 1.0),
            (self.accessors.turbined_flow[hydro_id], 1.0),
            (self.accessors.spillage[hydro_id], 1.0),
        ];
        for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
            factors
                .push((self.accessors.turbined_flow[*upstream_hydro_id], -1.0));
            factors.push((self.accessors.spillage[*upstream_hydro_id], -1.0));
        }
        let rhs = inflow + initial_storage;
        self.template_problem.add_row(rhs..rhs, &factors);
    }

    pub fn get_total_solution_cost(&self, solution: &myhighs::Solution) -> f64 {
        let cols = solution.columns();
        dot_product(&self.cost_vector, &cols)
    }

    pub fn get_stage_solution_cost(&self, solution: &myhighs::Solution) -> f64 {
        let cols = solution.columns();
        let cols_without_alpha = cols.len() - 1;
        dot_product(
            &self.cost_vector[..cols_without_alpha],
            &cols[..cols_without_alpha],
        )
    }

    pub fn add_cut(&mut self, cut: &BendersCut) {
        let mut factors: Vec<(usize, f64)> = vec![(self.accessors.alpha, 1.0)];
        for (hydro_id, stored_volume) in
            self.accessors.stored_volume.iter().enumerate()
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
}

impl<'a> ResourceRealization<'a> {
    pub fn new(
        bus_loads: &'a Vec<f64>,
        hydros_initial_storage: Vec<f64>,
    ) -> Self {
        Self {
            bus_loads,
            hydros_initial_storage,
        }
    }
}

#[derive(Clone, Debug)]
struct Trajectory<'a> {
    pub realizations: Vec<ResourceRealization<'a>>,
    pub solutions: Vec<myhighs::Solution>,
    pub cost: f64,
}

impl<'a> Trajectory<'a> {
    pub fn new(
        realizations: Vec<ResourceRealization<'a>>,
        solutions: Vec<myhighs::Solution>,
        cost: f64,
    ) -> Self {
        Self {
            realizations,
            solutions,
            cost,
        }
    }
}

/// Runs a single step of the forward pass, solving a node's subproblem
/// for some sampled uncertainty realization.
///
/// Returns both the solution and the realization.
fn forward_step<'a>(
    node: &Node,
    bus_loads: &'a Vec<f64>, // loads for stage 'index' ordered by id
    initial_storage: &Vec<f64>,
    hydros_inflow: &'a Vec<f64>, // inflows for stage 'index' ordered by id
) -> (myhighs::Solution, ResourceRealization<'a>) {
    let sp = node.subproblem_with_uncertainties(
        bus_loads,
        initial_storage,
        hydros_inflow,
    );
    let solved = solve(sp.template_problem);
    match solved.status() {
        myhighs::HighsModelStatus::Optimal => (
            solved.get_solution(),
            ResourceRealization::new(bus_loads, initial_storage.clone()),
        ),
        _ => panic!("Error while solving forward model"),
    }
}

/// Runs a forward pass of the SDDP algorithm, obtaining a viable
/// trajectory of states to be used in the backward pass.
///
/// Returns the sampled trajectory.
fn forward<'a>(
    nodes: &Vec<Node>,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>, // indexed by stage | hydro
) -> Trajectory<'a> {
    let mut realizations = Vec::<ResourceRealization>::new();
    let mut solutions = Vec::<myhighs::Solution>::new();
    let mut cost = 0.0;

    for (index, node) in nodes.iter().enumerate() {
        let node_initial_storage = if node.index == 0 {
            hydros_initial_storage.clone()
        } else {
            node.get_final_stored_volume_from_solutions(&solutions)
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

/// Solves a node's subproblem for all it's branchings and
/// returns the solutions.
fn solve_all_branchings(
    node: &Node,
    node_forward_realization: &ResourceRealization,
    node_saa: &Vec<Vec<f64>>, // indexed by stage | branching | hydro
) -> Vec<myhighs::Solution> {
    let forward_initial_storages =
        &node_forward_realization.hydros_initial_storage;

    let mut solutions = Vec::<myhighs::Solution>::new();
    for hydros_inflow in node_saa.iter() {
        let sp = node.subproblem_with_uncertainties(
            node_forward_realization.bus_loads,
            forward_initial_storages,
            hydros_inflow,
        );
        let solved = solve(sp.template_problem);

        match solved.status() {
            myhighs::HighsModelStatus::Optimal => {
                solutions.push(solved.get_solution())
            }
            _ => panic!("Error while solving backward model"),
        }
    }

    solutions
}

/// Evaluates and returns the new cut to be added to a node from the
/// solutions of the node's subproblem for all branchings.
fn eval_average_cut(
    node: &Node,
    num_branchings: usize,
    solutions: &Vec<myhighs::Solution>,
    node_forward_realization: &ResourceRealization,
) -> BendersCut {
    let num_hydros = node.system.hydros_count;
    let forward_initial_storages =
        &node_forward_realization.hydros_initial_storage;
    let mut average_water_values = vec![0.0; num_hydros];
    let mut average_solution_cost = 0.0;
    for solution in solutions.iter() {
        let water_values = solution
            .dual_rows()
            .get(node.hydro_balance_accessor())
            .unwrap();
        for hydro_id in 0..num_hydros {
            average_water_values[hydro_id] += water_values[hydro_id]
        }
        average_solution_cost +=
            node.subproblem.get_total_solution_cost(solution);
    }

    // evaluate average cut
    average_solution_cost = average_solution_cost / (num_branchings as f64);
    for hydro_id in 0..num_hydros {
        average_water_values[hydro_id] =
            average_water_values[hydro_id] / (num_branchings as f64);
    }

    let cut_rhs = average_solution_cost
        - dot_product(&average_water_values, forward_initial_storages);
    BendersCut::new(average_water_values.clone(), cut_rhs)
}

/// Evaluates and returns the lower bound from the solutions
/// of the first stage problem for all branchings.
fn eval_first_stage_bound(
    node: &Node,
    solutions: &Vec<myhighs::Solution>,
    num_branchings: usize,
) -> f64 {
    let mut average_solution_cost = 0.0;
    for solution in solutions.iter() {
        average_solution_cost +=
            node.subproblem.get_total_solution_cost(solution);
    }
    return average_solution_cost / (num_branchings as f64);
}

/// Runs a backward pass of the SDDP algorithm, adding a new cut for
/// each node in the graph, except the first stage node, which is used
/// on estimating the lower bound of the current iteration.
///
/// Returns the current estimated lower bound.
fn backward(
    nodes: &mut Vec<Node>,
    trajectory: &Trajectory,
    saa: &Vec<Vec<Vec<f64>>>, // indexed by stage | branching | hydro
    num_branchings: usize,
) -> f64 {
    let mut node_iter = nodes.iter_mut().rev().peekable();
    loop {
        let node = node_iter.next().unwrap();
        let index = node.index;
        let node_forward_realization =
            trajectory.realizations.get(index).unwrap();
        let node_saa = saa.get(index).unwrap();

        match node_iter.peek() {
            Some(_) => {
                let solutions = solve_all_branchings(
                    &node,
                    node_forward_realization,
                    node_saa,
                );

                let cut = eval_average_cut(
                    &node,
                    num_branchings,
                    &solutions,
                    node_forward_realization,
                );
                node_iter.peek_mut().unwrap().subproblem.add_cut(&cut);
            }
            None => {
                let solutions = solve_all_branchings(
                    &node,
                    node_forward_realization,
                    node_saa,
                );
                return eval_first_stage_bound(
                    &node,
                    &solutions,
                    num_branchings,
                );
            }
        }
    }
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate<'a>(
    graph: &mut Graph,
    bus_loads: &'a Vec<Vec<f64>>,
    hydros_initial_storage: &'a Vec<f64>,
    hydros_inflow: &'a Vec<&'a Vec<f64>>,
    saa: &Vec<Vec<Vec<f64>>>,
    num_branchings: usize,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory = forward(
        &graph.nodes,
        bus_loads,
        hydros_initial_storage,
        hydros_inflow,
    );

    let trajectory_cost = trajectory.cost;
    let first_stage_bound =
        backward(&mut graph.nodes, &trajectory, saa, num_branchings);

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_bound, iteration_time);
}

/// Generates an inflow SAA indexed by stage | branching | hydro.
///
/// `scenario_generator` must be indexed by stage | hydro
///
/// `num_stages` is the number of nodes in the graph
///
/// `num_branchings` is the number of branchings (same for all nodes)
///
/// ## Example
///
/// ```
/// let mu = 3.6;
/// let sigma = 0.6928;
/// let num_hydros = 2;
/// let scenario_generator: Vec<Vec<rand_distr::LogNormal<f64>>> =
///     vec![vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_hydros]];
/// let num_stages = 1;
/// let num_branchings = 10;
///
/// let saa = powers::sddp::generate_saa(&scenario_generator, num_stages, num_branchings);
/// assert_eq!(saa.len(), num_stages);
/// assert_eq!(saa[0].len(), num_branchings);
/// assert_eq!(saa[0][0].len(), num_hydros);
///
/// ```
pub fn generate_saa<'a>(
    scenario_generator: &'a Vec<Vec<LogNormal<f64>>>,
    num_stages: usize,
    num_branchings: usize,
) -> Vec<Vec<Vec<f64>>> {
    let mut rng = Xoshiro256Plus::seed_from_u64(0);

    let mut saa: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![]; num_branchings]; num_stages];
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

/// Helper function for displaying the greeting data for the training
fn training_greeting(
    num_iterations: usize,
    num_stages: usize,
    num_branchings: usize,
) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Stages: {num_stages}");
    println!("- Branchings: {num_branchings}\n");
}

/// Helper function for displaying the training table header
fn training_table_header() {
    println!(
        "{0: ^10} | {1: ^15} | {2: ^14} | {3: ^12}",
        "iteration", "lower bound ($)", "simulation ($)", "time (s)"
    )
}

/// Helper function for displaying a divider for the training table
fn training_table_divider() {
    println!("------------------------------------------------------------")
}

/// Helper function for displaying a row of iteration results for
/// the training table
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

/// Runs a training step of the SDDP algorithm over a graph.
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

    let num_stages = graph.len();
    let saa = generate_saa(scenario_generator, num_stages, num_branchings);

    training_greeting(num_iterations, graph.len(), num_branchings);
    training_table_divider();
    training_table_header();
    training_table_divider();

    for index in 0..num_iterations {
        // Generates indices for sampling inflows, indexed by stage
        let forward_branching_indices: Vec<usize> = forward_indices_dist
            .sample_iter(&mut rng)
            .take(num_stages)
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
            num_branchings,
        );

        training_table_row(index + 1, lower_bound, simulation, time);
    }

    training_table_divider();
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
        assert_eq!(subproblem.accessors.deficit.len(), 1);
        assert_eq!(subproblem.accessors.direct_exchange.len(), 0);
        assert_eq!(subproblem.accessors.reverse_exchange.len(), 0);
        assert_eq!(subproblem.accessors.thermal_gen.len(), 2);
        assert_eq!(subproblem.accessors.turbined_flow.len(), 1);
        assert_eq!(subproblem.accessors.spillage.len(), 1);
        assert_eq!(subproblem.accessors.stored_volume.len(), 1);
        assert_eq!(subproblem.accessors.min_generation_slack.len(), 1);
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

        let mut model = subproblem
            .template_problem
            .optimise(myhighs::Sense::Minimise);
        set_solver_options(&mut model);
        let solved = model.solve();
        assert_eq!(solved.status(), myhighs::HighsModelStatus::Optimal);
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
            .optimise(myhighs::Sense::Minimise);
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
        backward(&mut graph.nodes, &trajectory, &branchings, 3);
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
            3,
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
            12,
            3,
            &bus_loads,
            &hydros_initial_storage,
            &scenario_generator,
        );
    }
}
