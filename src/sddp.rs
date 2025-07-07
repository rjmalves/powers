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
use rayon::prelude::*;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct NodeData {
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
    ) -> Result<Self, String> {
        // Changed to return Result
        let load_stochastic_process =
            stochastic_process::factory(load_stochastic_process_str);
        let inflow_stochastic_process =
            stochastic_process::factory(inflow_stochastic_process_str);

        Ok(Self {
            id: node_id,
            stage_id,
            season_id,
            start_date: start_date_str.parse::<DateTime<Utc>>().map_err(
                |e| {
                    format!(
                        "Failed to parse start_date {}: {}",
                        start_date_str, e
                    )
                },
            )?,
            end_date: end_date_str.parse::<DateTime<Utc>>().map_err(|e| {
                format!("Failed to parse end_date {}: {}", end_date_str, e)
            })?,
            kind,
            system,
            risk_measure: risk_measure::factory(risk_measure_str),
            load_stochastic_process,
            inflow_stochastic_process,
            state_choice: state_str.to_string(),
        })
    }
}

pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,
}

impl SddpTrainHandler {
    pub fn new(
        pre_study_id: &usize,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        initial_condition: &initial_condition::InitialCondition,
        saa: &scenario::SAA,
    ) -> Result<Self, String> {
        // allocates graph with all required memory for forward solutions
        let mut realization_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Realization::with_capacity(
                    &node_data.kind,
                    &node_data.system,
                )
            });

        let subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        // add initial_condition to the PreStudy realization graph node
        realization_graph
            .get_node_mut(*pre_study_id)
            .ok_or_else(|| {
                "Failed to set initial condition to graph".to_string()
            })?
            .data
            .final_storage
            .clone_from_slice(initial_condition.get_storage());

        // allocates branching graph with all required memory for backward solutions
        let branching_graph =
            node_data_graph.map_topology_with(|node_data, id| {
                vec![
                    subproblem::Realization::with_capacity(
                        &node_data.kind,
                        &node_data.system,
                    );
                    saa.get_branching_count_at_stage(id).expect(&format!(
                        "Missing branching count for node {}",
                        id
                    ))
                ]
            });

        Ok(Self {
            subproblem_graph,
            realization_graph,
            branching_graph,
        })
    }

    pub fn forward(
        &mut self,
        sampled_noises: Vec<&scenario::SampledBranchingNoises>,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        graph_bfs_table: &Vec<Vec<usize>>,
        study_period_ids: &Vec<usize>,
    ) -> Result<f64, String> {
        for (idx, id) in study_period_ids.iter().enumerate() {
            let data_node = node_data_graph.get_node(*id).ok_or_else(|| {
                format!("Could not find data for node {}", id)
            })?;

            let subproblem_node =
                self.subproblem_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find subproblem for node {}", id)
                })?;

            let past_node_ids = graph_bfs_table.get(idx).ok_or_else(|| {
                format!("Could not find past node ids for node {}", id)
            })?;
            let past_realizations: Vec<&subproblem::Realization> = past_node_ids
                .iter()
                .map(|&past_id| {
                    self.realization_graph
                        .get_node(past_id)
                        .map(|node| &node.data)
                        .ok_or_else(|| {
                            format!("Could not find realization for past_node {} (current_id {})", past_id, id)
                        })
                    })
                .collect::<Result<_, _>>()?;

            subproblem_node
                .data
                .update_with_current_trajectory(past_realizations);

            let realization_node =
                self.realization_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find realization for node {}", id)
                })?;

            let current_stage_noises =
                sampled_noises.get(*id).ok_or_else(|| {
                    format!("Could not find noises for node {}", id)
                })?;

            step(
                data_node,
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;

            subproblem_node
                .data
                .update_with_current_realization(&realization_node.data);
        }

        let trajectory_cost: f64 = study_period_ids
            .iter()
            .map(|&id| {
                self.realization_graph
                    .get_node(id)
                    .map(|node| node.data.current_stage_objective)
                    .ok_or_else(|| {
                        format!(
                            "Could not find realization node {} in iterate",
                            id
                        )
                    })
            })
            .sum::<Result<f64, String>>()?;
        Ok(trajectory_cost)
    }

    pub fn backward_step_at_node(
        &mut self,
        id: usize,
        past_node_ids: &Vec<usize>,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::SAA,
        future_cost_function_graph: &graph::DirectedGraph<
            Arc<Mutex<fcf::FutureCostFunction>>,
        >,
    ) -> Result<(), String> {
        let node_forward_trajectory: Vec<&subproblem::Realization> =
                past_node_ids
                    .iter()
                    .map(|&past_id| {
                        self.realization_graph
                            .get_node(past_id)
                            .map(|node| &node.data)
                            .ok_or_else(|| {
                                format!("Could not find realization for past_node {} (current_id {})", past_id, id)
                            })
                    })
                    .collect::<Result<_, _>>()?;

        let num_branchings =
            saa.get_branching_count_at_stage(id).ok_or_else(|| {
                format!(
                    "Missing branching count for node {} in backward pass",
                    id
                )
            })?;

        solve_all_branchings(
            &mut self.subproblem_graph,
            &mut self.branching_graph,
            id,
            num_branchings,
            &node_forward_trajectory,
            node_data_graph,
            saa,
        )?;

        let branching_node_data = &self
            .branching_graph
            .get_node(id)
            .ok_or_else(|| {
                format!("Could not find branching realizations for node {}", id)
            })?
            .data;

        let parent_id = node_data_graph
            .get_parents(id)
            .and_then(|parents| parents.first().copied()) // Assumes a single parent for path graphs
            .ok_or_else(|| {
                format!("Could not find a unique parent for node {}", id)
            })?;

        update_future_cost_function(
            &mut self.subproblem_graph,
            future_cost_function_graph,
            parent_id,
            id,
            node_data_graph,
            &node_forward_trajectory,
            branching_node_data,
        )?;

        Ok(())
    }

    pub fn eval_first_stage_bound(
        &mut self,
        id: usize,
        past_node_ids: &Vec<usize>,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::SAA,
    ) -> Result<f64, String> {
        let node_forward_trajectory: Vec<&subproblem::Realization> =
                past_node_ids
                    .iter()
                    .map(|&past_id| {
                        self.realization_graph
                            .get_node(past_id)
                            .map(|node| &node.data)
                            .ok_or_else(|| {
                                format!("Could not find realization for past_node {} (current_id {})", past_id, id)
                            })
                    })
                    .collect::<Result<_, _>>()?;

        let num_branchings =
            saa.get_branching_count_at_stage(id).ok_or_else(|| {
                format!(
                    "Missing branching count for node {} in backward pass",
                    id
                )
            })?;

        solve_all_branchings(
            &mut self.subproblem_graph,
            &mut self.branching_graph,
            id,
            num_branchings,
            &node_forward_trajectory,
            node_data_graph,
            saa,
        )?;

        let branching_node_data = &self
            .branching_graph
            .get_node(id)
            .ok_or_else(|| {
                format!("Could not find branching realizations for node {}", id)
            })?
            .data;

        eval_first_stage_bound(
            branching_node_data,
            &node_data_graph
                .get_node(id)
                .ok_or_else(|| {
                    format!("Could not find node data for node {}", id)
                })?
                .data
                .risk_measure,
        )
    }
}

fn solve_all_branchings(
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    branching_graph: &mut graph::DirectedGraph<Vec<subproblem::Realization>>,
    node_id: usize,
    num_branchings: usize,
    node_forward_trajectory: &Vec<&subproblem::Realization>,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::SAA,
) -> Result<(), String> {
    let data_node = node_data_graph.get_node(node_id).ok_or_else(|| {
        format!("Could not find node data for node {}", node_id)
    })?;

    let subproblem_node =
        subproblem_graph.get_node_mut(node_id).ok_or_else(|| {
            format!("Could not find subproblem for node {}", node_id)
        })?;

    let node_forward_realization =
        node_forward_trajectory.last().ok_or_else(|| {
            format!("Could not find forward realization for node {}", node_id)
        })?;

    let current_branching_node =
        branching_graph.get_node_mut(node_id).ok_or_else(|| {
            format!(
                "Could not find branching realizations for node {}",
                node_id
            )
        })?;

    for branching_id in 0..num_branchings {
        reuse_forward_basis(
            &mut subproblem_node.data,
            node_forward_realization,
        )?;

        step(
            data_node,
            &mut subproblem_node.data,
            current_branching_node
                .data
                .get_mut(branching_id)
                .ok_or_else(|| {
                    format!(
                        "Could not find branching {} realization for node {}",
                        branching_id, node_id
                    )
                })?,
            saa.get_noises_by_stage_and_branching(node_id, branching_id)
                .ok_or_else(|| {
                    format!(
                        "Could not find noises for branching {}, node {}",
                        branching_id, node_id
                    )
                })?,
        )?;
    }
    Ok(())
}

fn update_future_cost_function(
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    future_cost_function_graph: &graph::DirectedGraph<
        Arc<Mutex<fcf::FutureCostFunction>>,
    >,
    parent_id: usize,
    child_id: usize,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    forward_trajectory: &Vec<&subproblem::Realization>,
    branching_realizations: &Vec<subproblem::Realization>,
) -> Result<(), String> {
    // evals cut with the state sampled by the child node, which will represent the
    // future cost function of that node, for the parent one.

    let child_data_node =
        node_data_graph.get_node(child_id).ok_or_else(|| {
            format!("Could not find node data for node {}", child_id)
        })?;
    let child_subproblem_node =
        subproblem_graph.get_node(child_id).ok_or_else(|| {
            format!("Could not find subproblem for node {}", child_id)
        })?;
    let cut_state_pair = child_subproblem_node.data.compute_new_cut(
        forward_trajectory,
        branching_realizations,
        &child_data_node.data.risk_measure,
    );

    // adds cut to the pools in the parent node, applying cut selection
    let parent_subproblem_node: &mut graph::Node<subproblem::Subproblem> =
        subproblem_graph.get_node_mut(parent_id).ok_or_else(|| {
            format!("Could not find subproblem for node {}", parent_id)
        })?;
    let parent_fcf_node: &graph::Node<Arc<Mutex<fcf::FutureCostFunction>>> =
        future_cost_function_graph
            .get_node(parent_id)
            .ok_or_else(|| {
                format!(
                    "Could not find future cost function for node {}",
                    parent_id
                )
            })?;

    parent_subproblem_node
        .data
        .add_cut_and_evaluate_cut_selection(
            cut_state_pair,
            Arc::clone(&parent_fcf_node.data),
        );
    Ok(())
}

pub struct SddpSimulationHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
}

impl SddpSimulationHandler {
    pub fn new(
        pre_study_id: &usize,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        initial_condition: &initial_condition::InitialCondition,
    ) -> Result<Self, String> {
        // allocates graph with all required memory for forward solutions
        let mut realization_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Realization::with_capacity(
                    &node_data.kind,
                    &node_data.system,
                )
            });

        let subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                subproblem::Subproblem::new(
                    &node_data.system,
                    &node_data.state_choice,
                    &node_data.load_stochastic_process,
                    &node_data.inflow_stochastic_process,
                )
            });

        // add initial_condition to the PreStudy realization graph node
        realization_graph
            .get_node_mut(*pre_study_id)
            .ok_or_else(|| {
                "Failed to set initial condition to graph".to_string()
            })?
            .data
            .final_storage
            .clone_from_slice(initial_condition.get_storage());

        Ok(Self {
            subproblem_graph,
            realization_graph,
        })
    }

    pub fn forward(
        &mut self,
        sampled_noises: Vec<&scenario::SampledBranchingNoises>,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        graph_bfs_table: &Vec<Vec<usize>>,
        study_period_ids: &Vec<usize>,
    ) -> Result<f64, String> {
        for (idx, id) in study_period_ids.iter().enumerate() {
            let data_node = node_data_graph.get_node(*id).ok_or_else(|| {
                format!("Could not find data for node {}", id)
            })?;

            let subproblem_node =
                self.subproblem_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find subproblem for node {}", id)
                })?;

            let past_node_ids = graph_bfs_table.get(idx).ok_or_else(|| {
                format!("Could not find past node ids for node {}", id)
            })?;
            let past_realizations: Vec<&subproblem::Realization> = past_node_ids
                .iter()
                .map(|&past_id| {
                    self.realization_graph
                        .get_node(past_id)
                        .map(|node| &node.data)
                        .ok_or_else(|| {
                            format!("Could not find realization for past_node {} (current_id {})", past_id, id)
                        })
                    })
                .collect::<Result<_, _>>()?;

            subproblem_node
                .data
                .update_with_current_trajectory(past_realizations);

            let realization_node =
                self.realization_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find realization for node {}", id)
                })?;

            let current_stage_noises =
                sampled_noises.get(*id).ok_or_else(|| {
                    format!("Could not find noises for node {}", id)
                })?;

            step(
                data_node,
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;

            subproblem_node
                .data
                .update_with_current_realization(&realization_node.data);
        }

        let trajectory_cost: f64 = study_period_ids
            .iter()
            .map(|&id| {
                self.realization_graph
                    .get_node(id)
                    .map(|node| node.data.current_stage_objective)
                    .ok_or_else(|| {
                        format!(
                            "Could not find realization node {} in iterate",
                            id
                        )
                    })
            })
            .sum::<Result<f64, String>>()?;
        Ok(trajectory_cost)
    }

    pub fn get_realization_at_node(
        &self,
        id: usize,
    ) -> Option<&graph::Node<subproblem::Realization>> {
        self.realization_graph.get_node(id)
    }
}

pub struct SddpAlgorithm {
    // core graphs and data
    node_data_graph: graph::DirectedGraph<NodeData>,
    future_cost_function_graph:
        graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,

    // initial state
    initial_condition: initial_condition::InitialCondition,

    // for rng reproducibility
    seed: u64,

    // helpers for traversing the graphs
    pre_study_id: usize,
    study_period_ids: Vec<usize>,
    graph_bfs_table: Vec<Vec<usize>>, // BFS table for study periods
}

impl SddpAlgorithm {
    pub fn new(
        node_data_graph: graph::DirectedGraph<NodeData>,
        initial_condition: initial_condition::InitialCondition,
        seed: u64,
    ) -> Result<Self, String> {
        let future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let pre_study_id = node_data_graph
            .get_node_id_with(|node| {
                node.kind == subproblem::StudyPeriodKind::PreStudy
            })
            .ok_or_else(|| {
                "Failed to find initial condition info in graph".to_string()
            })?;

        let study_period_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::Study
        });

        // TODO - for the path graph case, this is enough. But for markovian graphs
        // and cyclic graphs (infinite horizon) this might not be enough.
        let graph_bfs_table = study_period_ids
            .iter()
            .map(|id| node_data_graph.get_bfs(*id, true))
            .collect();

        Ok(Self {
            node_data_graph,
            future_cost_function_graph,
            initial_condition,
            seed,
            pre_study_id,
            study_period_ids,
            graph_bfs_table,
        })
    }

    pub fn train(
        &mut self,
        num_iterations: usize,
        num_forward_passes: usize,
        saa: &scenario::SAA,
    ) -> Result<(), String> {
        // rng is always created for reproducibility
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);

        let begin = Instant::now();

        log::training_greeting(num_iterations, num_forward_passes);
        log::training_table_divider();
        log::training_table_header();
        log::training_table_divider();

        // Create a pool of handlers for the parallel forward passes.
        // A new set of handlers is created for each iteration. This is necessary because
        // the forward pass modifies the handler's internal state, and each pass needs
        // a clean state to work with, apart from the shared cuts.
        let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
            .map(|_| {
                SddpTrainHandler::new(
                    &self.pre_study_id,
                    &self.node_data_graph,
                    &self.initial_condition,
                    saa,
                )
            })
            .collect::<Result<_, _>>()?;

        // Main training loop
        for index in 0..num_iterations {
            let iter_begin = Instant::now();

            // Sample noises for each forward pass
            let all_sampled_noises: Vec<_> = (0..num_forward_passes)
                .map(|_| saa.sample_scenario(&mut rng))
                .collect();

            // --- Parallel Forward Passes ---
            let forward_costs: Vec<f64> = train_handlers
                .par_iter_mut()
                .zip(all_sampled_noises.par_iter())
                .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
                .collect::<Result<Vec<f64>, String>>()?;

            let avg_forward_cost = utils::mean(&forward_costs);

            // --- Parallel Backward Pass with Stage-wise Synchronization ---
            let num_study_periods = self.study_period_ids.len();
            let mut lower_bound = 0.0;
            // Iterate backwards through study periods
            for rev_idx in 0..num_study_periods {
                let current_stage_original_idx =
                    num_study_periods - 1 - rev_idx;
                let id = self.study_period_ids[current_stage_original_idx];

                let past_node_ids = self
                .graph_bfs_table
                .get(current_stage_original_idx)
                .ok_or_else(||
                    format!("Could not find past node ids for node {} (original_idx {})", id, current_stage_original_idx)
                )?;
                // If it's not the very first stage of the study (i.e., has a parent stage)
                if current_stage_original_idx > 0 {
                    train_handlers
                        .par_iter_mut()
                        .map(|handler| {
                            handler.backward_step_at_node(
                                id,
                                past_node_ids,
                                &self.node_data_graph,
                                saa,
                                &self.future_cost_function_graph,
                            )
                        })
                        .collect::<Result<(), String>>()?;
                    // TODO - try serial cut selection instead of selecting while the FCF is locked on each thread
                } else {
                    lower_bound = train_handlers
                        .get_mut(0)
                        .unwrap()
                        .eval_first_stage_bound(
                            id,
                            past_node_ids,
                            &self.node_data_graph,
                            saa,
                        )
                        .unwrap();
                }
            }

            let iter_time = iter_begin.elapsed();
            log::training_table_row(
                index + 1,
                lower_bound,
                avg_forward_cost,
                iter_time,
            );
        }

        log::training_table_divider();
        let duration = begin.elapsed();
        log::training_duration(duration);
        log::policy_size(
            self.future_cost_function_graph
                .get_node(1)
                .ok_or_else(|| {
                    format!("Could not find node 1 for counting cuts")
                })?
                .data
                .lock()
                .unwrap()
                .cut_pool
                .total_cut_count,
        );
        Ok(())
    }

    pub fn forward(
        &self,
        sampled_noises: Vec<&scenario::SampledBranchingNoises>,
        handler: &mut SddpTrainHandler,
    ) -> Result<f64, String> {
        let trajectory_cost = handler.forward(
            sampled_noises,
            &self.node_data_graph,
            &self.graph_bfs_table,
            &self.study_period_ids,
        )?;
        Ok(trajectory_cost)
    }

    pub fn simulate(
        &mut self,
        num_simulation_scenarios: usize,
        saa: &scenario::SAA,
    ) -> Result<Vec<SddpSimulationHandler>, String> {
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);

        let begin = Instant::now();

        log::simulation_greeting(num_simulation_scenarios);

        let mut simulation_handlers = Vec::<SddpSimulationHandler>::new();

        for _ in 0..num_simulation_scenarios {
            let mut handler = SddpSimulationHandler::new(
                &self.pre_study_id,
                &self.node_data_graph,
                &self.initial_condition,
            )?;
            // samples the SAA
            let sampled_noises = saa.sample_scenario(&mut rng);

            handler.forward(
                sampled_noises,
                &self.node_data_graph,
                &self.graph_bfs_table,
                &self.study_period_ids,
            )?;
            simulation_handlers.push(handler);
        }

        let simulation_costs: Vec<f64> = simulation_handlers
            .iter()
            .map(|t| {
                Ok(self.study_period_ids
                    .iter()
                    .map(|&id| {
                        t.get_realization_at_node(id)
                            .map(|node| node.data.current_stage_objective)
                            .ok_or_else(|| format!("Could not find realization for node {} in simulation_costs", id))
                    })
                    .collect::<Result<Vec<f64>, String>>()?
                    .iter()
                    .sum())
            })
            .collect::<Result<Vec<f64>, String>>()?;
        let mean_cost = utils::mean(&simulation_costs);
        let std_cost = utils::standard_deviation(&simulation_costs);
        log::simulation_stats(mean_cost, std_cost);
        let duration = begin.elapsed();
        log::simulation_duration(duration);

        Ok(simulation_handlers)
    }
}

fn step(
    data_node: &graph::Node<NodeData>,
    subproblem: &mut subproblem::Subproblem,
    realization_container: &mut subproblem::Realization,
    noises: &scenario::SampledBranchingNoises,
) -> Result<(), String> {
    subproblem.realize_uncertainties(
        noises,
        &data_node.data.load_stochastic_process,
        &data_node.data.inflow_stochastic_process,
        realization_container,
    );
    Ok(())
}

fn reuse_forward_basis(
    subproblem: &mut subproblem::Subproblem,
    node_forward_realization: &subproblem::Realization,
) -> Result<(), String> {
    if node_forward_realization.basis.columns().len() > 0 {
        if let Some(model) = subproblem.model.as_mut() {
            let num_model_rows = model.num_rows();
            let mut forward_rows =
                node_forward_realization.basis.rows().to_vec();
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
    Ok(())
}

fn eval_first_stage_bound(
    branching_realizations: &Vec<subproblem::Realization>,
    risk_measure: &Box<dyn risk_measure::RiskMeasure>,
) -> Result<f64, String> {
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
    Ok(average_solution_cost)
}

// #[cfg(test)]
// mod tests {

//     use super::*;
//     use rand_distr::{LogNormal, Normal};

//     #[test]
//     fn test_forward_with_default_system() {
//         let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
//         node_data_graph
//             .add_node(
//                 0,
//                 NodeData::new(
//                     0,
//                     0,
//                     0,
//                     "2025-01-01T00:00:00Z",
//                     "2025-02-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(), // Assuming System::default() is cheap or test-only
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 1,
//                 NodeData::new(
//                     1,
//                     1,
//                     1,
//                     "2025-02-01T00:00:00Z",
//                     "2025-03-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 2,
//                 NodeData::new(
//                     2,
//                     2,
//                     2,
//                     "2025-03-01T00:00:00Z",
//                     "2025-04-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph.add_edge(0, 1).unwrap();
//         node_data_graph.add_edge(1, 2).unwrap();
//         let storage = vec![83.222];

//         let initial_condition =
//             initial_condition::InitialCondition::new(storage, vec![]);

//         let example_noises = scenario::SampledBranchingNoises {
//             load_noises: vec![75.0],
//             inflow_noises: vec![10.0],
//             num_load_entities: 1,
//             num_inflow_entities: 1,
//         };
//         let sampled_noises =
//             vec![&example_noises, &example_noises, &example_noises];

//         let mut trajectory =
//             subproblem::Trajectory::from_initial_condition(&initial_condition);

//         let mut subproblem_graph =
//             node_data_graph.map_topology_with(|node_data, _id| {
//                 subproblem::Subproblem::new(
//                     &node_data.system,
//                     &node_data.state_choice,
//                     &node_data.load_stochastic_process,
//                     &node_data.inflow_stochastic_process,
//                 )
//             });

//         forward(
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut trajectory,
//             sampled_noises,
//         );
//     }

//     fn generate_test_saa() -> scenario::SAA {
//         scenario::SAA {
//             branching_samples: vec![
//                 scenario::SampledNodeBranchings {
//                     num_branchings: 1,
//                     branching_noises: vec![scenario::SampledBranchingNoises {
//                         load_noises: vec![75.0],
//                         inflow_noises: vec![5.0],
//                         num_load_entities: 1,
//                         num_inflow_entities: 1,
//                     }],
//                 },
//                 scenario::SampledNodeBranchings {
//                     num_branchings: 1,
//                     branching_noises: vec![scenario::SampledBranchingNoises {
//                         load_noises: vec![75.0],
//                         inflow_noises: vec![10.0],
//                         num_load_entities: 1,
//                         num_inflow_entities: 1,
//                     }],
//                 },
//                 scenario::SampledNodeBranchings {
//                     num_branchings: 1,
//                     branching_noises: vec![scenario::SampledBranchingNoises {
//                         load_noises: vec![75.0],
//                         inflow_noises: vec![15.0],
//                         num_load_entities: 1,
//                         num_inflow_entities: 1,
//                     }],
//                 },
//             ],
//             index_samplers: vec![],
//         }
//     }

//     #[test]
//     fn test_backward_with_default_system() {
//         let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
//         node_data_graph
//             .add_node(
//                 0,
//                 NodeData::new(
//                     0,
//                     0,
//                     0,
//                     "2025-01-01T00:00:00Z",
//                     "2025-02-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 1,
//                 NodeData::new(
//                     1,
//                     1,
//                     1,
//                     "2025-02-01T00:00:00Z",
//                     "2025-03-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 2,
//                 NodeData::new(
//                     2,
//                     2,
//                     2,
//                     "2025-03-01T00:00:00Z",
//                     "2025-04-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph.add_edge(0, 1).unwrap();
//         node_data_graph.add_edge(1, 2).unwrap();
//         let storage = vec![83.222];

//         let initial_condition =
//             initial_condition::InitialCondition::new(storage, vec![]);

//         let mut future_cost_function_graph =
//             node_data_graph.map_topology_with(|_node_data, _id| {
//                 Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
//             });

//         let example_noises = scenario::SampledBranchingNoises {
//             load_noises: vec![75.0],
//             inflow_noises: vec![10.0],
//             num_load_entities: 1,
//             num_inflow_entities: 1,
//         };
//         let sampled_noises =
//             vec![&example_noises, &example_noises, &example_noises];

//         let mut trajectory =
//             subproblem::Trajectory::from_initial_condition(&initial_condition);

//         let mut subproblem_graph =
//             node_data_graph.map_topology_with(|node_data, _id| {
//                 subproblem::Subproblem::new(
//                     &node_data.system,
//                     &node_data.state_choice,
//                     &node_data.load_stochastic_process,
//                     &node_data.inflow_stochastic_process,
//                 )
//             });
//         forward(
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut trajectory,
//             sampled_noises,
//         );
//         let saa = generate_test_saa();

//         backward(
//             0,
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut future_cost_function_graph,
//             &trajectory,
//             &saa,
//         );
//     }

//     #[test]
//     fn test_iterate_with_default_system() {
//         let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
//         node_data_graph
//             .add_node(
//                 0,
//                 NodeData::new(
//                     0,
//                     0,
//                     0,
//                     "2025-01-01T00:00:00Z",
//                     "2025-02-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 1,
//                 NodeData::new(
//                     1,
//                     1,
//                     1,
//                     "2025-02-01T00:00:00Z",
//                     "2025-03-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph
//             .add_node(
//                 2,
//                 NodeData::new(
//                     2,
//                     2,
//                     2,
//                     "2025-03-01T00:00:00Z",
//                     "2025-04-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         node_data_graph.add_edge(0, 1).unwrap();
//         node_data_graph.add_edge(1, 2).unwrap();
//         let storage = vec![83.222];

//         let initial_condition =
//             initial_condition::InitialCondition::new(storage, vec![]);

//         let mut future_cost_function_graph =
//             node_data_graph.map_topology_with(|_node_data, _id| {
//                 Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
//             });

//         let mut subproblem_graph =
//             node_data_graph.map_topology_with(|node_data, _id| {
//                 subproblem::Subproblem::new(
//                     &node_data.system,
//                     &node_data.state_choice,
//                     &node_data.load_stochastic_process,
//                     &node_data.inflow_stochastic_process,
//                 )
//             });

//         let example_noises = scenario::SampledBranchingNoises {
//             load_noises: vec![75.0],
//             inflow_noises: vec![10.0],
//             num_load_entities: 1,
//             num_inflow_entities: 1,
//         };
//         let sampled_noises =
//             vec![&example_noises, &example_noises, &example_noises];

//         let saa = generate_test_saa();
//         iterate(
//             0,
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut future_cost_function_graph,
//             &initial_condition,
//             sampled_noises,
//             &saa,
//         );
//     }

//     #[test]
//     fn test_train_with_default_system() {
//         let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
//         let mut prev_id = 0;
//         node_data_graph
//             .add_node(
//                 prev_id,
//                 NodeData::new(
//                     prev_id,
//                     prev_id,
//                     prev_id,
//                     "2025-01-01T00:00:00Z",
//                     "2025-02-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         let mut scenario_generator = scenario::NoiseGenerator::new();
//         scenario_generator.add_node_generator(
//             vec![Normal::new(75.0, 0.0).unwrap()],
//             vec![LogNormal::new(3.6, 0.6928).unwrap()],
//             3,
//         );

//         for new_id in 1..4 {
//             node_data_graph
//                 .add_node(
//                     new_id,
//                     NodeData::new(
//                         new_id,
//                         new_id,
//                         new_id,
//                         "2025-01-01T00:00:00Z",
//                         "2025-02-01T00:00:00Z",
//                         subproblem::StudyPeriodKind::Study,
//                         system::System::default(),
//                         "expectation",
//                         "naive",
//                         "naive",
//                         "storage",
//                     ),
//                 )
//                 .unwrap();
//             node_data_graph.add_edge(prev_id, new_id).unwrap();
//             prev_id = new_id;
//             scenario_generator.add_node_generator(
//                 vec![Normal::new(75.0, 0.0).unwrap()],
//                 vec![LogNormal::new(3.6, 0.6928).unwrap()],
//                 3,
//             );
//         }

//         let storage = vec![83.222];

//         let initial_condition =
//             initial_condition::InitialCondition::new(storage, vec![]);

//         let mut future_cost_function_graph =
//             node_data_graph.map_topology_with(|_node_data, _id| {
//                 Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
//             });

//         let mut subproblem_graph =
//             node_data_graph.map_topology_with(|node_data, _id| {
//                 subproblem::Subproblem::new(
//                     &node_data.system,
//                     &node_data.state_choice,
//                     &node_data.load_stochastic_process,
//                     &node_data.inflow_stochastic_process,
//                 )
//             });

//         let saa = scenario_generator.generate(0);
//         train(
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut future_cost_function_graph,
//             24,
//             &initial_condition,
//             &saa,
//         );
//     }

//     #[test]
//     fn test_simulate_with_default_system() {
//         let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
//         let mut prev_id = 0;
//         node_data_graph
//             .add_node(
//                 prev_id,
//                 NodeData::new(
//                     prev_id,
//                     prev_id,
//                     prev_id,
//                     "2025-01-01T00:00:00Z",
//                     "2025-02-01T00:00:00Z",
//                     subproblem::StudyPeriodKind::Study,
//                     system::System::default(),
//                     "expectation",
//                     "naive",
//                     "naive",
//                     "storage",
//                 ),
//             )
//             .unwrap();
//         let mut scenario_generator = scenario::NoiseGenerator::new();
//         scenario_generator.add_node_generator(
//             vec![Normal::new(75.0, 0.0).unwrap()],
//             vec![LogNormal::new(3.6, 0.6928).unwrap()],
//             3,
//         );
//         for new_id in 1..4 {
//             node_data_graph
//                 .add_node(
//                     new_id,
//                     NodeData::new(
//                         new_id,
//                         new_id,
//                         new_id,
//                         "2025-01-01T00:00:00Z",
//                         "2025-02-01T00:00:00Z",
//                         subproblem::StudyPeriodKind::Study,
//                         system::System::default(),
//                         "expectation",
//                         "naive",
//                         "naive",
//                         "storage",
//                     ),
//                 )
//                 .unwrap();
//             node_data_graph.add_edge(prev_id, new_id).unwrap();
//             prev_id = new_id;
//             scenario_generator.add_node_generator(
//                 vec![Normal::new(75.0, 0.0).unwrap()],
//                 vec![LogNormal::new(3.6, 0.6928).unwrap()],
//                 3,
//             );
//         }
//         let storage = vec![83.222];

//         let initial_condition =
//             initial_condition::InitialCondition::new(storage, vec![]);

//         let mut future_cost_function_graph =
//             node_data_graph.map_topology_with(|_node_data, _id| {
//                 Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
//             });

//         let mut subproblem_graph =
//             node_data_graph.map_topology_with(|node_data, _id| {
//                 subproblem::Subproblem::new(
//                     &node_data.system,
//                     &node_data.state_choice,
//                     &node_data.load_stochastic_process,
//                     &node_data.inflow_stochastic_process,
//                 )
//             });

//         let saa = scenario_generator.generate(0);
//         train(
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             &mut future_cost_function_graph,
//             24,
//             &initial_condition,
//             &saa,
//         );
//         simulate(
//             &mut node_data_graph,
//             &mut subproblem_graph,
//             100,
//             &initial_condition,
//             &saa,
//         );
//     }
// }
