//! Implementation of the Stochastic Dual Dynamic Programming (SDDP)
//! algorithm for the hydrothermal dispatch problem. In exchange for
//! the simplified power system and state definition, some "smart"
//! optimizations and features are already considered in this code.
//!
//! The underlying power system is modeled with only four entities:
//! - Buses
//! - Lines
//! - Thermals
//! - Hydros
//!
//! Some considerations about the implementation:
//!
//! 1. Only hydro storages are considered as state variables.
//! 2. No memory management was made ready for parallelism (no locks and mutexes)
//! 3. Only risk-neutral policy evaluation is supported (no risk-aversion)
//! 4. An exact cut selection strategy (inspired in SDDP.jl) is implemented
//! 5. Only the "single-cut" (average cut) variant of the algorithm is supported.
//!
//! The only external dependencies are:
//!
//! 1. Random number generation and distribution sampling from rand* crates
//! 2. Low-level C-bindings from the highs-sys crate
//! 3. JSON and CSV serializers from the serde, serde_json and csv crates

use crate::graph;
use crate::risk_measure;
use crate::risk_measure::RiskMeasure;
use crate::scenario;
use crate::solver;
use crate::state;
use crate::stochastic_process;
use crate::subproblem;
use crate::system;
use crate::utils;
use rand::prelude::*;

use rand_xoshiro::Xoshiro256Plus;
use std::f64;
use std::sync::Arc;
use std::time::{Duration, Instant};

// TODO - general optimizations
// 1. Pre-allocate everywhere when the total size of the containers
// is known, in repacement to calling push! (or init vectors with allocated capacity)
// 2. Better handle cut and state storage:
//     - currently allocating twice the memory for cuts (BendersCut and Model row)
//     - currently allocating twice the memory for states of the same iteration (VisitedState and Realization)
// Expected memory cost for allocating 2200 state variables as f64 for 120 stages: 2MB

/// Helper function for removing the future cost term from the stage objective,
/// a.k.a the `alpha` term, or the epigraphical variable, assuming the objective
/// function is:
///
/// c^T x + `alpha`
pub fn get_current_stage_objective(
    total_stage_objective: f64,
    solution: &solver::Solution,
) -> f64 {
    let future_objective = solution.colvalue.last().unwrap();
    total_stage_objective - future_objective
}

/// Helper function for setting the same default solver options on
/// every solved problem.
fn set_default_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "off");
    model.set_option("solver", "simplex");
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
    model.set_option("time_limit", 300);
}

/// Helper function for setting the solver options when retrying a solve
fn set_first_retry_solver_options(model: &mut solver::Model) {
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
}

/// Helper function for setting the solver options when retrying a solve
fn set_second_retry_solver_options(model: &mut solver::Model) {
    model.set_option("primal_feasibility_tolerance", 1e-5);
    model.set_option("dual_feasibility_tolerance", 1e-5);
}

/// Helper function for setting the solver options when retrying a solve
fn set_third_retry_solver_options(model: &mut solver::Model) {
    model.set_option("simplex_strategy", 4);
}

/// Helper function for setting the solver options when retrying a solve
fn set_final_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "ipm");
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
}

/// Helper function for setting the solver options when retrying a solve
fn set_retry_solver_options(model: &mut solver::Model, retry: usize) {
    match retry {
        1 => set_first_retry_solver_options(model),
        2 => set_second_retry_solver_options(model),
        3 => set_third_retry_solver_options(model),
        _ => set_final_retry_solver_options(model),
    }
}

#[derive(Debug)]
pub struct VisitedState {
    pub state: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
}

impl VisitedState {
    pub fn new(
        state: Vec<f64>,
        dominating_objective: f64,
        dominating_cut_id: usize,
    ) -> Self {
        Self {
            state,
            dominating_objective,
            dominating_cut_id,
        }
    }
}

#[derive(Debug)]
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: bool,
    pub non_dominated_state_count: isize,
}

impl BendersCut {
    pub fn new(id: usize, coefficients: Vec<f64>, rhs: f64) -> Self {
        Self {
            id,
            coefficients,
            rhs,
            active: true,
            non_dominated_state_count: 1,
        }
    }

    pub fn eval_height_at_state(&self, state: &[f64]) -> f64 {
        self.rhs + utils::dot_product(&self.coefficients, state)
    }
}

#[derive(Debug)]
pub struct NodeData {
    pub system: system::System,
    pub subproblem: subproblem::Subproblem,
    pub state: Arc<state::State>,
    pub load_stochastic_process: stochastic_process::StochasticProcess,
    pub inflow_stochastic_process: stochastic_process::StochasticProcess,
    pub risk_measure: risk_measure::RiskMeasure,
    pub total_cut_count: usize,
    pub active_cut_ids: Vec<usize>,
    pub visited_state_pool: Vec<VisitedState>,
    pub benders_cut_pool: Vec<BendersCut>,
}

impl NodeData {
    pub fn new(system: system::System) -> Self {
        let subproblem = subproblem::Subproblem::new(&system);
        let num_buses = system.buses.len();
        let num_hydros = system.hydros.len();
        Self {
            system,
            subproblem,
            state: Arc::new(state::State::new(num_hydros)),
            load_stochastic_process: stochastic_process::StochasticProcess::new(
                num_buses,
            ),
            inflow_stochastic_process:
                stochastic_process::StochasticProcess::new(num_hydros),
            risk_measure: RiskMeasure::new(),
            total_cut_count: 0,
            active_cut_ids: Vec::<usize>::new(),
            visited_state_pool: Vec::<VisitedState>::new(),
            benders_cut_pool: Vec::<BendersCut>::new(),
        }
    }

    fn eval_new_cut_domination(&mut self, cut: &mut BendersCut) {
        // Tests the new cut on every previously visited state. If this cut dominates,
        // decrements the previous dominating cut counter and updates this.
        for state in self.visited_state_pool.iter_mut() {
            let height = cut.eval_height_at_state(&state.state);
            if height > state.dominating_objective {
                self.benders_cut_pool[state.dominating_cut_id]
                    .non_dominated_state_count -= 1;
                cut.non_dominated_state_count += 1;
                state.dominating_cut_id = cut.id;
                state.dominating_objective = height;
            }
        }
    }

    fn update_old_cuts_domination(
        &mut self,
        current_state: &mut VisitedState,
    ) -> Vec<usize> {
        // Tests the cuts that are not in the model for the new state. If any of these cuts
        // dominate the new state, increment their counter and puts them back inside the model
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();
        for old_cut in self.benders_cut_pool.iter_mut() {
            match old_cut.active {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(&current_state.state);
                    if height > current_state.dominating_objective {
                        cut_non_dominated_decrement_ids
                            .push(current_state.dominating_cut_id);

                        old_cut.non_dominated_state_count += 1;
                        current_state.dominating_cut_id = old_cut.id;
                        current_state.dominating_objective = height;
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }
        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.benders_cut_pool[*cut_id].non_dominated_state_count -= 1;
        }

        cut_ids_to_return_to_model
    }

    fn return_and_remove_cuts_from_model(
        &mut self,
        cut_ids_to_return_to_model: &[usize],
    ) {
        // Add cuts back to model
        for cut_id in cut_ids_to_return_to_model.iter() {
            self.return_cut_to_model(*cut_id);
        }

        // Iterate over all the cuts, deleting from the model the cuts that should be deleted
        let mut cut_ids_to_remove_from_model = Vec::<usize>::new();
        for cut in self.benders_cut_pool.iter_mut() {
            if (cut.non_dominated_state_count <= 0) && cut.active {
                cut_ids_to_remove_from_model.push(cut.id);
            }
        }

        for cut_id in cut_ids_to_remove_from_model.iter() {
            self.remove_cut_from_model(*cut_id);
        }
    }

    pub fn add_cut_to_model(&mut self, cut: &mut BendersCut) {
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.state.get_dimension() + 1);
        factors.push((self.subproblem.accessors.alpha, 1.0));
        for (hydro_id, stored_volume) in
            self.subproblem.accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        self.subproblem.model.add_row(cut.rhs.., factors);
        self.active_cut_ids.push(cut.id);
        self.total_cut_count += 1;
    }

    pub fn return_cut_to_model(&mut self, cut_id: usize) {
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.state.get_dimension() + 1);
        let cut = self.benders_cut_pool.get_mut(cut_id).unwrap();
        factors.push((self.subproblem.accessors.alpha, 1.0));
        for (hydro_id, stored_volume) in
            self.subproblem.accessors.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -1.0 * cut.coefficients[hydro_id]));
        }
        self.subproblem.model.add_row(cut.rhs.., factors);
        self.active_cut_ids.push(cut_id);
        self.benders_cut_pool[cut_id].active = true;
    }

    pub fn remove_cut_from_model(&mut self, cut_id: usize) {
        let cut_index = self
            .active_cut_ids
            .iter()
            .position(|&x| x == cut_id)
            .unwrap();
        let row_index =
            *self.subproblem.accessors.hydro_balance.last().unwrap()
                + 1
                + cut_index;
        self.subproblem.model.delete_row(row_index).unwrap();
        self.active_cut_ids.remove(cut_index);
        self.benders_cut_pool[cut_id].active = false;
    }
}

pub struct Trajectory {
    pub realizations: Vec<subproblem::Realization>,
    pub cost: f64,
}

impl Trajectory {
    pub fn new(realizations: Vec<subproblem::Realization>, cost: f64) -> Self {
        Self { realizations, cost }
    }
}

/// Runs a single step of the forward pass / backward branching,
/// solving a node's subproblem for some sampled uncertainty realization.
///
/// Returns the realization with relevant data.
fn realize_uncertainties<'a>(
    node: &mut graph::Node<NodeData>,
    initial_state: Arc<state::State>,
    buses_load_noises: &scenario::SampledBranchingNoises, // loads for stage 'index' ordered by id
    hydros_inflow_noises: &scenario::SampledBranchingNoises, // inflows for stage 'index' ordered by id
) -> subproblem::Realization {
    let initial_storage = initial_state.get_hydro_storages();
    let load = node
        .data
        .load_stochastic_process
        .realize(buses_load_noises.get_noises());
    let inflow = node
        .data
        .inflow_stochastic_process
        .realize(hydros_inflow_noises.get_noises());

    node.data
        .subproblem
        .realize_uncertainties(initial_storage, load, inflow)
}

/// Runs a forward pass of the SDDP algorithm, obtaining a viable
/// trajectory of states to be used in the backward pass.
///
/// Returns the sampled trajectory.
fn forward<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    initial_state: Arc<state::State>,
    buses_load_noises: Vec<&scenario::SampledBranchingNoises>,
    hydros_inflow_noises: Vec<&scenario::SampledBranchingNoises>,
) -> Trajectory {
    let mut realizations =
        Vec::<subproblem::Realization>::with_capacity(g.node_count());
    let mut cost = 0.0;

    for id in 0..g.node_count() {
        let state = if g.is_root(id) {
            Arc::clone(&initial_state)
        } else {
            Arc::clone(&realizations.last().unwrap().final_state)
        };
        let node = g.get_node_mut(id).unwrap();
        let realization = realize_uncertainties(
            node,
            state,
            buses_load_noises.get(id).unwrap(),
            hydros_inflow_noises.get(id).unwrap(),
        );
        cost += realization.current_stage_objective;
        realizations.push(realization);
    }
    Trajectory::new(realizations, cost)
}

// fn hot_start_with_forward_solution<'a>(
//     node: &mut Node,
//     node_forward_realization: &'a Realization,
// ) {
//     let num_model_rows = node.subproblem.model.num_rows();
//     let mut forward_rows = node_forward_realization.solution.rowvalue.to_vec();
//     let mut forward_dual_rows =
//         node_forward_realization.solution.rowdual.to_vec();
//     let num_forward_rows = forward_rows.len();

//     // checks if should add zeros to the rows (new cuts added)
//     if num_forward_rows < num_model_rows {
//         let row_diff = num_model_rows - num_forward_rows;
//         forward_rows.append(&mut vec![0.0; row_diff]);
//         forward_dual_rows.append(&mut vec![0.0; row_diff]);
//     }

//     node.subproblem.model.set_solution(
//         Some(&node_forward_realization.solution.colvalue),
//         Some(&forward_rows),
//         Some(&node_forward_realization.solution.coldual),
//         Some(&forward_dual_rows),
//     );
// }

fn reuse_forward_basis<'a>(
    node: &mut graph::Node<NodeData>,
    node_forward_realization: &'a subproblem::Realization,
) {
    let num_model_rows = node.data.subproblem.model.num_rows();
    let mut forward_rows = node_forward_realization.basis.rows().to_vec();
    let num_forward_rows = forward_rows.len();

    // checks if should add zeros to the rows (new cuts added)
    if num_forward_rows < num_model_rows {
        let row_diff = num_model_rows - num_forward_rows;
        forward_rows.append(&mut vec![0; row_diff]);
    } else if num_forward_rows > num_model_rows {
        forward_rows.truncate(num_model_rows);
    }

    node.data.subproblem.model.set_basis(
        Some(node_forward_realization.basis.columns()),
        Some(&forward_rows),
    );
}

/// Solves a node's subproblem for all it's branchings and
/// returns the solutions.
fn solve_all_branchings<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    node_id: usize,
    num_load_branchings: usize,
    num_inflow_branchings: usize,
    node_forward_realization: &'a subproblem::Realization,
    load_saa: &'a scenario::SAA, // indexed by stage | branching | bus
    inflow_saa: &'a scenario::SAA, // indexed by stage | branching | hydro
) -> Vec<subproblem::Realization> {
    let mut realizations =
        Vec::<subproblem::Realization>::with_capacity(num_inflow_branchings);
    let node = g.get_node_mut(node_id).unwrap();
    for inflow_branching_id in 0..num_inflow_branchings {
        let load_branching_id = inflow_branching_id % num_load_branchings;
        reuse_forward_basis(node, node_forward_realization);
        // hot_start_with_forward_solution(node, node_forward_realization);
        let realization = realize_uncertainties(
            node,
            Arc::clone(&node_forward_realization.initial_storage),
            load_saa
                .get_noises_by_stage_and_branching(node_id, load_branching_id)
                .unwrap(),
            inflow_saa
                .get_noises_by_stage_and_branching(node_id, inflow_branching_id)
                .unwrap(),
        );
        realizations.push(realization);
    }

    realizations
}

/// Evaluates and returns the new cut to be added to a node from the
/// solutions of the node's subproblem for all branchings.
fn eval_cut(
    node: &graph::Node<NodeData>,
    cut_id: usize,
    forward_realization: &Realization,
    branching_realizations: &Vec<Realization>,
) -> BendersCut {
    node.data.state.compute_cut(
        cut_id,
        node,
        forward_realization,
        branching_realizations,
    )
}

fn update_future_cost_function(
    g: &mut graph::DirectedGraph<NodeData>,
    parent_id: usize,
    child_id: usize,
    forward_realization: &Realization,
    branchings_realizations: &Vec<Realization>,
) {
    let child_node = g.get_node(child_id).unwrap();
    let new_cut_id = g.get_node(parent_id).unwrap().data.total_cut_count;
    let mut cut = eval_cut(
        &child_node,
        new_cut_id,
        forward_realization,
        branchings_realizations,
    );
    let mut state = VisitedState::new(
        forward_realization
            .final_state
            .get_hydro_storages()
            .to_vec(),
        cut.eval_height_at_state(
            &forward_realization.final_state.get_hydro_storages(),
        ),
        cut.id,
    );

    let parent_node: &mut graph::Node<NodeData> =
        g.get_node_mut(parent_id).unwrap();
    // Adds cuts to model and applies exact cut selection
    parent_node.data.add_cut_to_model(&mut cut);
    parent_node.data.eval_new_cut_domination(&mut cut);
    parent_node.data.benders_cut_pool.push(cut);
    let cut_ids_to_return_to_model =
        parent_node.data.update_old_cuts_domination(&mut state);
    parent_node
        .data
        .return_and_remove_cuts_from_model(&cut_ids_to_return_to_model);
    parent_node.data.visited_state_pool.push(state);
}

/// Evaluates and returns the lower bound from the solutions
/// of the first stage problem for all branchings.
fn eval_first_stage_bound(
    num_branchings: usize,
    branchings_realizations: &Vec<Realization>,
) -> f64 {
    let mut average_solution_cost = 0.0;
    for realization in branchings_realizations.iter() {
        average_solution_cost += realization.total_stage_objective
    }
    return average_solution_cost / (num_branchings as f64);
}

/// Runs a backward pass of the SDDP algorithm, adding a new cut for
/// each node in the graph, except the first stage node, which is used
/// on estimating the lower bound of the current iteration.
///
/// Returns the current estimated lower bound.
fn backward(
    g: &mut graph::DirectedGraph<NodeData>,
    trajectory: &Trajectory,
    load_saa: &scenario::SAA, // indexed by stage | branching | bus
    inflow_saa: &scenario::SAA, // indexed by stage | branching | hydro
) -> f64 {
    for id in (0..g.node_count()).rev() {
        let node_forward_realization = trajectory.realizations.get(id).unwrap();
        let num_load_branchings =
            load_saa.get_branching_count_at_stage(id).unwrap();
        let num_inflow_branchings =
            inflow_saa.get_branching_count_at_stage(id).unwrap();
        let realizations = solve_all_branchings(
            g,
            id,
            num_load_branchings,
            num_inflow_branchings,
            node_forward_realization,
            load_saa,
            inflow_saa,
        );
        if !g.is_root(id) {
            let parent_id = g.get_parents(id).unwrap()[0];
            update_future_cost_function(
                g,
                parent_id,
                id,
                node_forward_realization,
                &realizations,
            );
        } else {
            return eval_first_stage_bound(
                num_inflow_branchings,
                &realizations,
            );
        }
    }
    // TODO - better handle this edge case by returning a Result<>
    return 0.0;
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    initial_state: Arc<state::State>,
    buses_load_noises: Vec<&scenario::SampledBranchingNoises>,
    hydros_inflow_noises: Vec<&scenario::SampledBranchingNoises>,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory = forward(
        g,
        Arc::clone(&initial_state),
        buses_load_noises,
        hydros_inflow_noises,
    );

    let trajectory_cost = trajectory.cost;
    let first_stage_bound = backward(g, &trajectory, load_saa, inflow_saa);

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_bound, iteration_time);
}

/// Helper function for displaying the greeting data for the training
fn training_greeting(num_iterations: usize, num_stages: usize) {
    println!("\n# Training");
    println!("- Iterations: {num_iterations}");
    println!("- Stages: {num_stages}");
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

fn training_duration(time: Duration) {
    println!("\nTraining time: {:.2} s", time.as_millis() as f64 / 1000.0)
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
    g: &mut graph::DirectedGraph<NodeData>,
    num_iterations: usize,
    initial_state: Arc<state::State>,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) {
    let begin = Instant::now();

    let seed = 0;

    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    training_greeting(num_iterations, g.node_count());
    training_table_divider();
    training_table_header();
    training_table_divider();

    for index in 0..num_iterations {
        // Samples the SAA
        let buses_load_noise = load_saa.sample_scenario(&mut rng);
        let hydros_inflow_noise = inflow_saa.sample_scenario(&mut rng);

        let (simulation, lower_bound, time) = iterate(
            g,
            Arc::clone(&initial_state),
            buses_load_noise,
            hydros_inflow_noise,
            &load_saa,
            &inflow_saa,
        );

        training_table_row(index + 1, lower_bound, simulation, time);
    }

    training_table_divider();
    let duration = begin.elapsed();
    training_duration(duration);
}

/// Helper function for displaying the greeting data for the simulation
fn simulation_greeting(num_simulation_scenarios: usize) {
    println!("\n# Simulating");
    println!("- Scenarios: {num_simulation_scenarios}\n");
}

fn simulation_stats(mean: f64, std: f64) {
    println!("Expected cost ($): {:.2} +- {:.2}", mean, std);
}

fn simulation_duration(time: Duration) {
    println!(
        "\nSimulation time: {:.2} s",
        time.as_millis() as f64 / 1000.0
    )
}

/// Runs a simulation using the policy obtained by the SDDP algorithm.
pub fn simulate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    num_simulation_scenarios: usize,
    initial_state: Arc<state::State>,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) -> Vec<Trajectory> {
    let begin = Instant::now();

    let seed = 0;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    simulation_greeting(num_simulation_scenarios);

    let mut trajectories =
        Vec::<Trajectory>::with_capacity(num_simulation_scenarios);

    for _ in 0..num_simulation_scenarios {
        // Samples the SAA
        let bus_loads = load_saa.sample_scenario(&mut rng);
        let hydros_inflow = inflow_saa.sample_scenario(&mut rng);

        let trajectory =
            forward(g, Arc::clone(&initial_state), bus_loads, hydros_inflow);
        trajectories.push(trajectory);
    }

    let simulation_costs: Vec<f64> =
        trajectories.iter().map(|t| t.cost).collect();

    let total_cost: f64 = simulation_costs.iter().sum();
    let mean_cost = total_cost / (num_simulation_scenarios as f64);
    let cost_deviations: Vec<f64> = simulation_costs
        .iter()
        .map(|c| (c - mean_cost) * (c - mean_cost))
        .collect();
    let cost_total_deviation: f64 = cost_deviations.iter().sum();
    let std_cost =
        f64::sqrt(cost_total_deviation / (num_simulation_scenarios as f64));

    simulation_stats(mean_cost, std_cost);
    let duration = begin.elapsed();
    simulation_duration(duration);

    trajectories
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand_distr::{LogNormal, Normal};

    #[test]
    fn test_forward_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(system::System::default()));
        let id1 = g.add_node(NodeData::new(system::System::default()));
        let id2 = g.add_node(NodeData::new(system::System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let mut initial_state = state::State::new(1);
        initial_state.set_hydro_storages(&[83.222]);

        let example_load = scenario::SampledBranchingNoises {
            noises: vec![75.0],
            num_entities: 1,
        };
        let bus_loads = vec![&example_load, &example_load, &example_load];
        let example_inflow = scenario::SampledBranchingNoises {
            noises: vec![10.0],
            num_entities: 1,
        };
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        forward(&mut g, Arc::new(initial_state), bus_loads, hydros_inflow);
    }

    fn generate_load_saa() -> scenario::SAA {
        scenario::SAA {
            branching_samples: vec![
                scenario::SampledStageBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        noises: vec![75.0],
                        num_entities: 1,
                    }],
                },
                scenario::SampledStageBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        noises: vec![75.0],
                        num_entities: 1,
                    }],
                },
                scenario::SampledStageBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        noises: vec![75.0],
                        num_entities: 1,
                    }],
                },
            ],
            index_samplers: vec![],
        }
    }

    fn generate_inflow_saa() -> scenario::SAA {
        scenario::SAA {
            branching_samples: vec![
                scenario::SampledStageBranchings {
                    num_branchings: 3,
                    branching_noises: vec![
                        scenario::SampledBranchingNoises {
                            noises: vec![5.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![10.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![15.0],
                            num_entities: 1,
                        },
                    ],
                },
                scenario::SampledStageBranchings {
                    num_branchings: 3,
                    branching_noises: vec![
                        scenario::SampledBranchingNoises {
                            noises: vec![5.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![10.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![15.0],
                            num_entities: 1,
                        },
                    ],
                },
                scenario::SampledStageBranchings {
                    num_branchings: 3,
                    branching_noises: vec![
                        scenario::SampledBranchingNoises {
                            noises: vec![5.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![10.0],
                            num_entities: 1,
                        },
                        scenario::SampledBranchingNoises {
                            noises: vec![15.0],
                            num_entities: 1,
                        },
                    ],
                },
            ],
            index_samplers: vec![],
        }
    }

    #[test]
    fn test_backward_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(system::System::default()));
        let id1 = g.add_node(NodeData::new(system::System::default()));
        let id2 = g.add_node(NodeData::new(system::System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let mut initial_state = state::State::new(1);
        initial_state.set_hydro_storages(&[83.222]);

        let example_load = scenario::SampledBranchingNoises {
            noises: vec![75.0],
            num_entities: 1,
        };
        let bus_loads = vec![&example_load, &example_load, &example_load];

        let example_inflow = scenario::SampledBranchingNoises {
            noises: vec![10.0],
            num_entities: 1,
        };
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let trajectory =
            forward(&mut g, Arc::new(initial_state), bus_loads, hydros_inflow);
        let load_saa = generate_load_saa();
        let inflow_saa = generate_inflow_saa();

        backward(&mut g, &trajectory, &load_saa, &inflow_saa);
    }

    #[test]
    fn test_iterate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let id0 = g.add_node(NodeData::new(system::System::default()));
        let id1 = g.add_node(NodeData::new(system::System::default()));
        let id2 = g.add_node(NodeData::new(system::System::default()));
        g.add_edge(id0, id1).unwrap();
        g.add_edge(id1, id2).unwrap();
        let mut initial_state = state::State::new(1);
        initial_state.set_hydro_storages(&[83.222]);

        let example_load = scenario::SampledBranchingNoises {
            noises: vec![75.0],
            num_entities: 1,
        };
        let bus_loads = vec![&example_load, &example_load, &example_load];

        let example_inflow = scenario::SampledBranchingNoises {
            noises: vec![10.0],
            num_entities: 1,
        };
        let hydros_inflow =
            vec![&example_inflow, &example_inflow, &example_inflow];
        let load_saa = generate_load_saa();
        let inflow_saa = generate_inflow_saa();
        iterate(
            &mut g,
            Arc::new(initial_state),
            bus_loads,
            hydros_inflow,
            &load_saa,
            &inflow_saa,
        );
    }

    #[test]
    fn test_train_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = g.add_node(NodeData::new(system::System::default()));
        let mut load_scenario_generator = scenario::ScenarioGenerator::new();
        let mut inflow_scenario_generator = scenario::ScenarioGenerator::new();
        load_scenario_generator
            .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
        inflow_scenario_generator
            .add_stage_generator(vec![LogNormal::new(3.6, 0.6928).unwrap()], 3);
        for _ in 1..4 {
            let new_id = g.add_node(NodeData::new(system::System::default()));
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            load_scenario_generator
                .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
            inflow_scenario_generator.add_stage_generator(
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }

        let mut initial_state = state::State::new(1);
        initial_state.set_hydro_storages(&[83.222]);

        let load_saa = load_scenario_generator.generate(0);
        let inflow_saa = inflow_scenario_generator.generate(0);
        train(&mut g, 24, Arc::new(initial_state), &load_saa, &inflow_saa);
    }

    #[test]
    fn test_simulate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = g.add_node(NodeData::new(system::System::default()));
        let mut load_scenario_generator = scenario::ScenarioGenerator::new();
        let mut inflow_scenario_generator = scenario::ScenarioGenerator::new();
        load_scenario_generator
            .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
        inflow_scenario_generator
            .add_stage_generator(vec![LogNormal::new(3.6, 0.6928).unwrap()], 3);
        for _ in 1..4 {
            let new_id = g.add_node(NodeData::new(system::System::default()));
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            load_scenario_generator
                .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
            inflow_scenario_generator.add_stage_generator(
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }
        let mut initial_state = state::State::new(1);
        initial_state.set_hydro_storages(&[83.222]);
        let state = Arc::new(initial_state);

        let load_saa = load_scenario_generator.generate(0);
        let inflow_saa = inflow_scenario_generator.generate(0);
        train(&mut g, 24, Arc::clone(&state), &load_saa, &inflow_saa);
        simulate(&mut g, 100, Arc::clone(&state), &load_saa, &inflow_saa);
    }
}
