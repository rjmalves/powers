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

use crate::fcf;
use crate::graph;
use crate::initial_condition;
use crate::log;
use crate::risk_measure;
use crate::scenario;
use crate::stochastic_process;
use crate::subproblem;
use crate::system;
use crate::utils;
use chrono::prelude::*;
use rand::prelude::*;

use rand_xoshiro::Xoshiro256Plus;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// TODO - general optimizations
// 1. Pre-allocate everywhere when the total size of the containers
// is known, in repacement to calling push! (or init vectors with allocated capacity)
// 2. Better handle cut and state storage:
//     - currently allocating twice the memory for cuts (BendersCut and Model row)
//     - currently allocating twice the memory for states of the same iteration (VisitedState and Realization)
// Expected memory cost for allocating 2200 state variables as f64 for 120 stages: 2MB

pub struct NodeData {
    // these fields are common for all computing threads - read only
    pub id: isize,
    pub stage_id: usize,
    pub season_id: usize,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub kind: subproblem::StudyPeriodKind,
    pub system: system::System,
    pub risk_measure: Box<dyn risk_measure::RiskMeasure>,
    pub load_stochastic_process: Box<dyn stochastic_process::StochasticProcess>,
    pub inflow_stochastic_process:
        Box<dyn stochastic_process::StochasticProcess>,
    pub state_choice: String,
}

impl NodeData {
    pub fn new(
        node_id: isize,
        stage_id: usize,
        season_id: usize,
        start_date_str: &str,
        end_date_str: &str,
        kind: subproblem::StudyPeriodKind,
        system: system::System,
        risk_measure_str: &str,
        load_stochastic_process_str: &str,
        inflow_stochastic_process_str: &str,
        state_str: &str,
    ) -> Self {
        let load_stochastic_process =
            stochastic_process::factory(load_stochastic_process_str);
        let inflow_stochastic_process =
            stochastic_process::factory(inflow_stochastic_process_str);

        Self {
            id: node_id,
            stage_id,
            season_id,
            start_date: start_date_str.parse::<DateTime<Utc>>().unwrap(),
            end_date: end_date_str.parse::<DateTime<Utc>>().unwrap(),
            kind,
            system,
            risk_measure: risk_measure::factory(risk_measure_str),
            load_stochastic_process,
            inflow_stochastic_process,
            state_choice: state_str.to_string(),
        }
    }
}

/// Runs a single step of the forward pass / backward branching,
/// solving a node's subproblem for some sampled uncertainty realization.
///
/// Returns the realization with relevant data.
fn step(
    data_node: &graph::Node<NodeData>,
    subproblem_node: &mut graph::Node<subproblem::Subproblem>,
    noises: &scenario::SampledBranchingNoises,
) -> subproblem::Realization {
    subproblem_node.data.realize_uncertainties(
        noises,
        &data_node.data.load_stochastic_process,
        &data_node.data.inflow_stochastic_process,
    )
}

/// Runs a forward pass of the SDDP algorithm, obtaining a viable
/// trajectory of states to be used in the backward pass.
///
/// Returns the sampled trajectory.
fn forward(
    node_data_graph: &graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: &mut graph::DirectedGraph<subproblem::Realization>,
    sampled_noises: Vec<&scenario::SampledBranchingNoises>,
) {
    for id in 0..node_data_graph.node_count() {
        let data_node = node_data_graph.get_node(id).unwrap();
        let subproblem_node = subproblem_graph.get_node_mut(id).unwrap();
        // TODO - Extract a path (list) of realizations and pass to the update
        // function
        let bfs_nodes = realization_graph.get_bfs(id, true);
        subproblem_node
            .data
            .update_with_current_trajectory(&realization_graph);

        let realization =
            step(data_node, subproblem_node, sampled_noises.get(id).unwrap());

        subproblem_node
            .data
            .update_with_current_realization(&realization);

        t.add_step(realization);
    }
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

fn reuse_forward_basis(
    subproblem_node: &mut graph::Node<subproblem::Subproblem>,
    node_forward_realization: &subproblem::Realization,
) {
    if let Some(model) = subproblem_node.data.model.as_mut() {
        let num_model_rows = model.num_rows();
        let mut forward_rows = node_forward_realization.basis.rows().to_vec();
        let num_forward_rows = forward_rows.len();

        // checks if should add zeros to the rows (new cuts added)
        if num_forward_rows < num_model_rows {
            let row_diff = num_model_rows - num_forward_rows;
            forward_rows.append(&mut vec![0; row_diff]);
        } else if num_forward_rows > num_model_rows {
            forward_rows.truncate(num_model_rows);
        }

        model.set_basis(
            Some(node_forward_realization.basis.columns()),
            Some(&forward_rows),
        );
    }
}

/// Solves a node's subproblem for all it's branchings and
/// returns the solutions.
fn solve_all_branchings(
    node_data_graph: &graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    node_id: usize,
    num_branchings: usize,
    node_forward_trajectory: &[subproblem::Realization],
    saa: &scenario::SAA,
) -> Vec<subproblem::Realization> {
    let mut realizations =
        Vec::<subproblem::Realization>::with_capacity(num_branchings);
    let is_root_node = node_data_graph.is_root(node_id);
    let data_node = node_data_graph.get_node(node_id).unwrap();
    let subproblem_node = subproblem_graph.get_node_mut(node_id).unwrap();
    for branching_id in 0..num_branchings {
        let node_forward_realization = node_forward_trajectory.last().unwrap();
        if !is_root_node {
            reuse_forward_basis(subproblem_node, node_forward_realization);
        }
        let realization = step(
            data_node,
            subproblem_node,
            saa.get_noises_by_stage_and_branching(node_id, branching_id)
                .unwrap(),
        );
        realizations.push(realization);
    }

    realizations
}

/// Updates the future cost function (Benders Cut and State pools) stored inside each node.
/// The function is built by states sampled by the child node and is and stored in the parent node,
/// since managing the active cuts and editing the solver model is easier if the pools are in the
/// same node that manages the active cut constraints.
fn update_future_cost_function(
    iteration: usize,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    future_cost_function_graph: &mut graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    parent_id: usize,
    child_id: usize,
    forward_trajectory: &[subproblem::Realization],
    branching_realizations: &Vec<subproblem::Realization>,
) {
    // Evals cut with the state sampled by the child node, which will represent the
    // future cost function of that node, for the parent one.
    let new_cut_id = iteration;
    let child_data_node = node_data_graph.get_node(child_id).unwrap();
    let child_subproblem_node = subproblem_graph.get_node(child_id).unwrap();
    let cut_state_pair = child_subproblem_node.data.compute_new_cut(
        new_cut_id,
        forward_trajectory,
        branching_realizations,
        &child_data_node.data.risk_measure,
    );

    // Adds cut to the pools in the parent node, applying cut selection
    let parent_subproblem_node: &mut graph::Node<subproblem::Subproblem> =
        subproblem_graph.get_node_mut(parent_id).unwrap();
    let parent_fcf_node: &mut graph::Node<Arc<Mutex<fcf::FutureCostFunction>>> =
        future_cost_function_graph.get_node_mut(parent_id).unwrap();

    parent_subproblem_node
        .data
        .add_cut_and_evaluate_cut_selection(
            cut_state_pair,
            Arc::clone(&parent_fcf_node.data),
        );
}

/// Evaluates and returns the lower bound from the solutions
/// of the first stage problem for all branchings.
fn eval_first_stage_bound(
    branching_realizations: &Vec<subproblem::Realization>,
    risk_measure: &Box<dyn risk_measure::RiskMeasure>,
) -> f64 {
    // TODO - use first stage risk measure instead of average
    let costs: Vec<f64> = branching_realizations
        .iter()
        .map(|r| r.total_stage_objective)
        .collect();
    let num_branchings = costs.len();
    let probabilities = utils::uniform_prob_by_count(num_branchings);
    let adjusted_probabilities =
        risk_measure.adjust_probabilities(&probabilities, &costs);
    let average_solution_cost =
        utils::dot_product(adjusted_probabilities, &costs);
    average_solution_cost
}

/// Runs a backward pass of the SDDP algorithm, adding a new cut for
/// each node in the graph, except the first stage node, which is used
/// on estimating the lower bound of the current iteration.
///
/// Returns the current estimated lower bound.
fn backward(
    iteration: usize,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    future_cost_function_graph: &mut graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    trajectory: &subproblem::Trajectory,
    saa: &scenario::SAA, // indexed by stage | branching | entity_id
) -> f64 {
    for id in (0..node_data_graph.node_count()).rev() {
        // TODO - use the graph to traverse itself instead of relying on
        // indices based on id
        let node_forward_trajectory = &trajectory.realizations[..id + 2];
        let num_branchings = saa.get_branching_count_at_stage(id).unwrap();
        let realizations = solve_all_branchings(
            node_data_graph,
            subproblem_graph,
            id,
            num_branchings,
            node_forward_trajectory,
            saa,
        );
        if !node_data_graph.is_root(id) {
            let parent_id = node_data_graph.get_parents(id).unwrap()[0];
            update_future_cost_function(
                iteration,
                node_data_graph,
                subproblem_graph,
                future_cost_function_graph,
                parent_id,
                id,
                node_forward_trajectory,
                &realizations,
            );
        } else {
            return eval_first_stage_bound(
                &realizations,
                &node_data_graph.get_node(id).unwrap().data.risk_measure,
            );
        }
    }
    // TODO - better handle this edge case by returning a Result<>
    return 0.0;
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate(
    iteration: usize,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: &mut graph::DirectedGraph<subproblem::Realization>,
    future_cost_function_graph: &mut graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    initial_condition: &initial_condition::InitialCondition,
    sampled_noises: Vec<&scenario::SampledBranchingNoises>,
    saa: &scenario::SAA,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let mut trajectory =
        subproblem::Trajectory::from_initial_condition(initial_condition);

    forward(
        node_data_graph,
        subproblem_graph,
        realization_graph,
        sampled_noises,
    );

    let trajectory_cost = trajectory.cost;
    let first_stage_bound = backward(
        iteration,
        node_data_graph,
        subproblem_graph,
        future_cost_function_graph,
        realization_graph,
        saa,
    );

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_bound, iteration_time);
}

/// Runs a training step of the SDDP algorithm over a graph.
pub fn train(
    node_data_graph: &mut graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    future_cost_function_graph: &mut graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    num_iterations: usize,
    initial_condition: &initial_condition::InitialCondition,
    saa: &scenario::SAA,
) {
    let begin = Instant::now();

    let seed = 0;

    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    log::training_greeting(num_iterations);
    log::training_table_divider();
    log::training_table_header();
    log::training_table_divider();

    let mut realization_graph =
        node_data_graph.map_topology_with(|node_data, _id| {
            subproblem::Realization::with_capacity(
                subproblem::RealizationPeriodKind::Study,
                &node_data.system,
            )
        });

    for index in 0..num_iterations {
        // Samples the SAA
        let sampled_noises = saa.sample_scenario(&mut rng);

        let (simulation, lower_bound, time) = iterate(
            index,
            node_data_graph,
            subproblem_graph,
            realization_graph,
            future_cost_function_graph,
            initial_condition,
            sampled_noises,
            &saa,
        );

        log::training_table_row(index + 1, lower_bound, simulation, time);
    }

    log::training_table_divider();
    let duration = begin.elapsed();
    log::training_duration(duration);
}

/// Runs a simulation using the policy obtained by the SDDP algorithm.
pub fn simulate(
    node_data_graph: &mut graph::DirectedGraph<NodeData>,
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    num_simulation_scenarios: usize,
    initial_condition: &initial_condition::InitialCondition,
    saa: &scenario::SAA,
) -> Vec<subproblem::Trajectory> {
    let begin = Instant::now();

    let seed = 0;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    log::simulation_greeting(num_simulation_scenarios);

    let mut trajectories =
        Vec::<subproblem::Trajectory>::with_capacity(num_simulation_scenarios);

    for _ in 0..num_simulation_scenarios {
        // Samples the SAA
        let sampled_noises = saa.sample_scenario(&mut rng);

        let mut trajectory =
            subproblem::Trajectory::from_initial_condition(initial_condition);

        forward(
            node_data_graph,
            subproblem_graph,
            &mut trajectory,
            sampled_noises,
        );
        trajectories.push(trajectory);
    }

    let simulation_costs: Vec<f64> =
        trajectories.iter().map(|t| t.cost).collect();
    let mean_cost = utils::mean(&simulation_costs);
    let std_cost = utils::standard_deviation(&simulation_costs);
    log::simulation_stats(mean_cost, std_cost);
    let duration = begin.elapsed();
    log::simulation_duration(duration);

    trajectories
}

#[cfg(test)]
mod tests {

    use super::*;
    use rand_distr::{LogNormal, Normal};

    #[test]
    fn test_forward_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        node_data_graph
            .add_node(
                0,
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                1,
                NodeData::new(
                    1,
                    1,
                    1,
                    "2025-02-01T00:00:00Z",
                    "2025-03-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                2,
                NodeData::new(
                    2,
                    2,
                    2,
                    "2025-03-01T00:00:00Z",
                    "2025-04-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph.add_edge(0, 1).unwrap();
        node_data_graph.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let example_noises = scenario::SampledBranchingNoises {
            load_noises: vec![75.0],
            inflow_noises: vec![10.0],
            num_load_entities: 1,
            num_inflow_entities: 1,
        };
        let sampled_noises =
            vec![&example_noises, &example_noises, &example_noises];

        let mut trajectory =
            subproblem::Trajectory::from_initial_condition(&initial_condition);

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        forward(
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut trajectory,
            sampled_noises,
        );
    }

    fn generate_test_saa() -> scenario::SAA {
        scenario::SAA {
            branching_samples: vec![
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        load_noises: vec![75.0],
                        inflow_noises: vec![5.0],
                        num_load_entities: 1,
                        num_inflow_entities: 1,
                    }],
                },
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        load_noises: vec![75.0],
                        inflow_noises: vec![10.0],
                        num_load_entities: 1,
                        num_inflow_entities: 1,
                    }],
                },
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![scenario::SampledBranchingNoises {
                        load_noises: vec![75.0],
                        inflow_noises: vec![15.0],
                        num_load_entities: 1,
                        num_inflow_entities: 1,
                    }],
                },
            ],
            index_samplers: vec![],
        }
    }

    #[test]
    fn test_backward_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        node_data_graph
            .add_node(
                0,
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                1,
                NodeData::new(
                    1,
                    1,
                    1,
                    "2025-02-01T00:00:00Z",
                    "2025-03-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                2,
                NodeData::new(
                    2,
                    2,
                    2,
                    "2025-03-01T00:00:00Z",
                    "2025-04-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph.add_edge(0, 1).unwrap();
        node_data_graph.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let mut future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let example_noises = scenario::SampledBranchingNoises {
            load_noises: vec![75.0],
            inflow_noises: vec![10.0],
            num_load_entities: 1,
            num_inflow_entities: 1,
        };
        let sampled_noises =
            vec![&example_noises, &example_noises, &example_noises];

        let mut trajectory =
            subproblem::Trajectory::from_initial_condition(&initial_condition);

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });
        forward(
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut trajectory,
            sampled_noises,
        );
        let saa = generate_test_saa();

        backward(
            0,
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut future_cost_function_graph,
            &trajectory,
            &saa,
        );
    }

    #[test]
    fn test_iterate_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        node_data_graph
            .add_node(
                0,
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                1,
                NodeData::new(
                    1,
                    1,
                    1,
                    "2025-02-01T00:00:00Z",
                    "2025-03-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph
            .add_node(
                2,
                NodeData::new(
                    2,
                    2,
                    2,
                    "2025-03-01T00:00:00Z",
                    "2025-04-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        node_data_graph.add_edge(0, 1).unwrap();
        node_data_graph.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let mut future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        let example_noises = scenario::SampledBranchingNoises {
            load_noises: vec![75.0],
            inflow_noises: vec![10.0],
            num_load_entities: 1,
            num_inflow_entities: 1,
        };
        let sampled_noises =
            vec![&example_noises, &example_noises, &example_noises];

        let saa = generate_test_saa();
        iterate(
            0,
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut future_cost_function_graph,
            &initial_condition,
            sampled_noises,
            &saa,
        );
    }

    #[test]
    fn test_train_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = 0;
        node_data_graph
            .add_node(
                prev_id,
                NodeData::new(
                    prev_id,
                    prev_id,
                    prev_id,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );

        for new_id in 1..4 {
            node_data_graph
                .add_node(
                    new_id,
                    NodeData::new(
                        new_id,
                        new_id,
                        new_id,
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        system::System::default(),
                        "expectation",
                        "naive",
                        "naive",
                        "storage",
                    ),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }

        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let mut future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        let saa = scenario_generator.generate(0);
        train(
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut future_cost_function_graph,
            24,
            &initial_condition,
            &saa,
        );
    }

    #[test]
    fn test_simulate_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = 0;
        node_data_graph
            .add_node(
                prev_id,
                NodeData::new(
                    prev_id,
                    prev_id,
                    prev_id,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
        for new_id in 1..4 {
            node_data_graph
                .add_node(
                    new_id,
                    NodeData::new(
                        new_id,
                        new_id,
                        new_id,
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        system::System::default(),
                        "expectation",
                        "naive",
                        "naive",
                        "storage",
                    ),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let mut future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        let saa = scenario_generator.generate(0);
        train(
            &mut node_data_graph,
            &mut subproblem_graph,
            &mut future_cost_function_graph,
            24,
            &initial_condition,
            &saa,
        );
        simulate(
            &mut node_data_graph,
            &mut subproblem_graph,
            100,
            &initial_condition,
            &saa,
        );
    }
}
