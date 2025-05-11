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

use crate::cut;
use crate::fcf;
use crate::graph;
use crate::log;
use crate::risk_measure;
use crate::scenario;
use crate::state;
use crate::stochastic_process;
use crate::subproblem;
use crate::system;
use crate::utils;
use chrono::prelude::*;
use rand::prelude::*;

use rand_xoshiro::Xoshiro256Plus;
use std::f64;
use std::time::{Duration, Instant};

// TODO - general optimizations
// 1. Pre-allocate everywhere when the total size of the containers
// is known, in repacement to calling push! (or init vectors with allocated capacity)
// 2. Better handle cut and state storage:
//     - currently allocating twice the memory for cuts (BendersCut and Model row)
//     - currently allocating twice the memory for states of the same iteration (VisitedState and Realization)
// Expected memory cost for allocating 2200 state variables as f64 for 120 stages: 2MB

pub struct NodeData {
    // these fields are common for all computing threads
    pub id: usize,
    pub season_id: usize,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub system: system::System,
    pub risk_measure: Box<dyn risk_measure::RiskMeasure>,
    pub future_cost_function: fcf::FutureCostFunction,
    pub load_stochastic_process: Box<dyn stochastic_process::StochasticProcess>,
    pub inflow_stochastic_process:
        Box<dyn stochastic_process::StochasticProcess>,
    pub state: Box<dyn state::State>,
    // these fields will have to be allocated for each thread
    pub subproblem: subproblem::Subproblem,
}

impl NodeData {
    pub fn new(
        node_id: usize,
        season_id: usize,
        start_date_str: &str,
        end_date_str: &str,
        system: system::System,
        risk_measure_str: &str,
        load_stochastic_process_str: &str,
        inflow_stochastic_process_str: &str,
        state_str: &str,
    ) -> Self {
        let subproblem = subproblem::Subproblem::new(&system);
        let mut state = state::factory(state_str);
        state.set_dimension(system.meta.hydros_count);
        Self {
            id: node_id,
            season_id,
            start_date: start_date_str.parse::<DateTime<Utc>>().unwrap(),
            end_date: end_date_str.parse::<DateTime<Utc>>().unwrap(),
            system,
            subproblem,
            state,
            load_stochastic_process: stochastic_process::factory(
                load_stochastic_process_str,
            ),
            inflow_stochastic_process: stochastic_process::factory(
                inflow_stochastic_process_str,
            ),
            risk_measure: risk_measure::factory(risk_measure_str),
            future_cost_function: fcf::FutureCostFunction::new(),
        }
    }

    pub fn update_future_cost_function(
        &mut self,
        mut cut: cut::BendersCut,
        mut visited_state: Box<dyn state::State>,
    ) {
        self.add_cut_to_model(&mut cut);
        self.future_cost_function.eval_new_cut_domination(&mut cut);
        self.future_cost_function.add_cut(cut);

        let cut_ids_to_return_to_model = self
            .future_cost_function
            .update_old_cuts_domination(&mut visited_state);
        self.return_and_remove_cuts_from_model(&cut_ids_to_return_to_model);
        self.future_cost_function.add_state(visited_state);
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
        for cut in self.future_cost_function.cut_pool.pool.iter_mut() {
            if (cut.non_dominated_state_count <= 0) && cut.active {
                cut_ids_to_remove_from_model.push(cut.id);
            }
        }

        for cut_id in cut_ids_to_remove_from_model.iter() {
            self.remove_cut_from_model(*cut_id);
        }
    }

    pub fn add_cut_to_model(&mut self, cut: &mut cut::BendersCut) {
        self.state
            .add_cut_constraint_to_model(cut, &mut self.subproblem);
        self.future_cost_function.update_cut_pool_on_add(cut.id);
    }

    pub fn return_cut_to_model(&mut self, cut_id: usize) {
        let cut = self
            .future_cost_function
            .cut_pool
            .pool
            .get_mut(cut_id)
            .unwrap();
        self.state
            .add_cut_constraint_to_model(cut, &mut self.subproblem);
        self.future_cost_function.update_cut_pool_on_return(cut_id);
    }

    pub fn remove_cut_from_model(&mut self, cut_id: usize) {
        let cut_index =
            self.future_cost_function.get_active_cut_index_by_id(cut_id);
        let row_index = self.subproblem.first_cut_row_index() + cut_index;
        self.subproblem.model.delete_row(row_index).unwrap();
        self.future_cost_function
            .update_cut_pool_on_remove(cut_id, cut_index);
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
fn step<'a>(
    node: &mut graph::Node<NodeData>,
    buses_load_noises: &scenario::SampledBranchingNoises, // loads for stage 'index' ordered by id
    hydros_inflow_noises: &scenario::SampledBranchingNoises, // inflows for stage 'index' ordered by id
) -> subproblem::Realization {
    let initial_storage = node.data.state.get_initial_storage();
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
    buses_load_noises: Vec<&scenario::SampledBranchingNoises>,
    hydros_inflow_noises: Vec<&scenario::SampledBranchingNoises>,
) -> Trajectory {
    let mut realizations =
        Vec::<subproblem::Realization>::with_capacity(g.node_count());
    let mut cost = 0.0;

    for id in 0..g.node_count() {
        let node = g.get_node_mut(id).unwrap();
        if id != 0 {
            node.data.state.update_with_parent_node_realization(
                realizations.last().unwrap(),
            )
        };
        let realization = step(
            node,
            buses_load_noises.get(id).unwrap(),
            hydros_inflow_noises.get(id).unwrap(),
        );

        node.data
            .state
            .update_with_current_realization(&realization);

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
    num_branchings: usize,
    node_forward_realization: &'a subproblem::Realization,
    load_saa: &'a scenario::SAA, // indexed by stage | branching | bus
    inflow_saa: &'a scenario::SAA, // indexed by stage | branching | hydro
) -> Vec<subproblem::Realization> {
    let mut realizations =
        Vec::<subproblem::Realization>::with_capacity(num_branchings);
    let node = g.get_node_mut(node_id).unwrap();
    for branching_id in 0..num_branchings {
        reuse_forward_basis(node, node_forward_realization);
        // hot_start_with_forward_solution(node, node_forward_realization);
        let realization = step(
            node,
            load_saa
                .get_noises_by_stage_and_branching(node_id, branching_id)
                .unwrap(),
            inflow_saa
                .get_noises_by_stage_and_branching(node_id, branching_id)
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
    g: &mut graph::DirectedGraph<NodeData>,
    parent_id: usize,
    child_id: usize,
    forward_realization: &subproblem::Realization,
    branching_realizations: &Vec<subproblem::Realization>,
) {
    // Evals cut with the state sampled by the child node, which will represent the
    // future cost function of that node, for the parent one.
    let child_node = g.get_node(child_id).unwrap();
    let new_cut_id = g
        .get_node(parent_id)
        .unwrap()
        .data
        .future_cost_function
        .get_total_cut_count();

    // this only works when all nodes have the same state definition??
    let mut visited_state = child_node.data.state.clone();

    let cut = visited_state.compute_new_cut(
        new_cut_id,
        &child_node.data.risk_measure,
        forward_realization,
        branching_realizations,
    );

    // Adds cut to the pools in the parent node, applying cut selection
    let parent_node: &mut graph::Node<NodeData> =
        g.get_node_mut(parent_id).unwrap();

    parent_node
        .data
        .update_future_cost_function(cut, visited_state);
}

/// Evaluates and returns the lower bound from the solutions
/// of the first stage problem for all branchings.
fn eval_first_stage_bound(
    num_branchings: usize,
    branchings_realizations: &Vec<subproblem::Realization>,
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
        let num_branchings =
            inflow_saa.get_branching_count_at_stage(id).unwrap();
        let realizations = solve_all_branchings(
            g,
            id,
            num_branchings,
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
            return eval_first_stage_bound(num_branchings, &realizations);
        }
    }
    // TODO - better handle this edge case by returning a Result<>
    return 0.0;
}

/// Runs a single iteration, comprised of forward and backward passes,
/// of the SDDP algorithm.
fn iterate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    buses_load_noises: Vec<&scenario::SampledBranchingNoises>,
    hydros_inflow_noises: Vec<&scenario::SampledBranchingNoises>,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) -> (f64, f64, Duration) {
    let begin = Instant::now();

    let trajectory = forward(g, buses_load_noises, hydros_inflow_noises);

    let trajectory_cost = trajectory.cost;
    let first_stage_bound = backward(g, &trajectory, load_saa, inflow_saa);

    let iteration_time = begin.elapsed();
    return (trajectory_cost, first_stage_bound, iteration_time);
}

/// Runs a training step of the SDDP algorithm over a graph.
pub fn train<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    num_iterations: usize,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) {
    let begin = Instant::now();

    let seed = 0;

    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    log::training_greeting(num_iterations, g.node_count());
    log::training_table_divider();
    log::training_table_header();
    log::training_table_divider();

    for index in 0..num_iterations {
        // Samples the SAA
        let buses_load_noise = load_saa.sample_scenario(&mut rng);
        let hydros_inflow_noise = inflow_saa.sample_scenario(&mut rng);

        let (simulation, lower_bound, time) = iterate(
            g,
            buses_load_noise,
            hydros_inflow_noise,
            &load_saa,
            &inflow_saa,
        );

        log::training_table_row(index + 1, lower_bound, simulation, time);
    }

    log::training_table_divider();
    let duration = begin.elapsed();
    log::training_duration(duration);
}

/// Runs a simulation using the policy obtained by the SDDP algorithm.
pub fn simulate<'a>(
    g: &mut graph::DirectedGraph<NodeData>,
    num_simulation_scenarios: usize,
    load_saa: &'a scenario::SAA,
    inflow_saa: &'a scenario::SAA,
) -> Vec<Trajectory> {
    let begin = Instant::now();

    let seed = 0;
    let mut rng = Xoshiro256Plus::seed_from_u64(seed);

    log::simulation_greeting(num_simulation_scenarios);

    let mut trajectories =
        Vec::<Trajectory>::with_capacity(num_simulation_scenarios);

    for _ in 0..num_simulation_scenarios {
        // Samples the SAA
        let bus_loads = load_saa.sample_scenario(&mut rng);
        let hydros_inflow = inflow_saa.sample_scenario(&mut rng);

        let trajectory = forward(g, bus_loads, hydros_inflow);
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
        let mut g = graph::DirectedGraph::<NodeData>::new();
        g.add_node(
            0,
            NodeData::new(
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            1,
            NodeData::new(
                1,
                1,
                "2025-02-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            2,
            NodeData::new(
                2,
                2,
                "2025-03-01T00:00:00Z",
                "2025-04-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        g.get_node_mut(0)
            .unwrap()
            .data
            .state
            .set_initial_storage(storage);

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
        forward(&mut g, bus_loads, hydros_inflow);
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
        g.add_node(
            0,
            NodeData::new(
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            1,
            NodeData::new(
                1,
                1,
                "2025-02-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            2,
            NodeData::new(
                2,
                2,
                "2025-03-01T00:00:00Z",
                "2025-04-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        g.get_node_mut(0)
            .unwrap()
            .data
            .state
            .set_initial_storage(storage);

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
        let trajectory = forward(&mut g, bus_loads, hydros_inflow);
        let load_saa = generate_load_saa();
        let inflow_saa = generate_inflow_saa();

        backward(&mut g, &trajectory, &load_saa, &inflow_saa);
    }

    #[test]
    fn test_iterate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        g.add_node(
            0,
            NodeData::new(
                0,
                0,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            1,
            NodeData::new(
                1,
                1,
                "2025-02-01T00:00:00Z",
                "2025-03-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_node(
            2,
            NodeData::new(
                2,
                2,
                "2025-03-01T00:00:00Z",
                "2025-04-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        g.add_edge(0, 1).unwrap();
        g.add_edge(1, 2).unwrap();
        let storage = vec![83.222];

        g.get_node_mut(0)
            .unwrap()
            .data
            .state
            .set_initial_storage(storage);

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
        iterate(&mut g, bus_loads, hydros_inflow, &load_saa, &inflow_saa);
    }

    #[test]
    fn test_train_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = 0;
        g.add_node(
            prev_id,
            NodeData::new(
                prev_id,
                prev_id,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        let mut load_scenario_generator = scenario::ScenarioGenerator::new();
        let mut inflow_scenario_generator = scenario::ScenarioGenerator::new();
        load_scenario_generator
            .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
        inflow_scenario_generator
            .add_stage_generator(vec![LogNormal::new(3.6, 0.6928).unwrap()], 3);
        for new_id in 1..4 {
            g.add_node(
                new_id,
                NodeData::new(
                    new_id,
                    new_id,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            load_scenario_generator
                .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
            inflow_scenario_generator.add_stage_generator(
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }

        let storage = vec![83.222];

        g.get_node_mut(0)
            .unwrap()
            .data
            .state
            .set_initial_storage(storage);

        let load_saa = load_scenario_generator.generate(0);
        let inflow_saa = inflow_scenario_generator.generate(0);
        train(&mut g, 24, &load_saa, &inflow_saa);
    }

    #[test]
    fn test_simulate_with_default_system() {
        let mut g = graph::DirectedGraph::<NodeData>::new();
        let mut prev_id = 0;
        g.add_node(
            prev_id,
            NodeData::new(
                prev_id,
                prev_id,
                "2025-01-01T00:00:00Z",
                "2025-02-01T00:00:00Z",
                system::System::default(),
                "expectation",
                "naive",
                "naive",
                "storage",
            ),
        )
        .unwrap();
        let mut load_scenario_generator = scenario::ScenarioGenerator::new();
        let mut inflow_scenario_generator = scenario::ScenarioGenerator::new();
        load_scenario_generator
            .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
        inflow_scenario_generator
            .add_stage_generator(vec![LogNormal::new(3.6, 0.6928).unwrap()], 3);
        for new_id in 1..4 {
            g.add_node(
                new_id,
                NodeData::new(
                    new_id,
                    new_id,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                ),
            )
            .unwrap();
            g.add_edge(prev_id, new_id).unwrap();
            prev_id = new_id;
            load_scenario_generator
                .add_stage_generator(vec![Normal::new(75.0, 0.0).unwrap()], 1);
            inflow_scenario_generator.add_stage_generator(
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }
        let storage = vec![83.222];

        g.get_node_mut(0)
            .unwrap()
            .data
            .state
            .set_initial_storage(storage);

        let load_saa = load_scenario_generator.generate(0);
        let inflow_saa = inflow_scenario_generator.generate(0);
        train(&mut g, 24, &load_saa, &inflow_saa);
        simulate(&mut g, 100, &load_saa, &inflow_saa);
    }
}
