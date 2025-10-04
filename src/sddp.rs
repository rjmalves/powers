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
use std::time::{Duration, Instant};

/// Result of a single SDDP training iteration.
///
/// Contains all convergence information for one iteration, including bounds,
/// costs, gaps, and timing information. This data enables convergence analysis
/// and numerical validation in tests.
///
/// # Performance Notes
///
/// This struct is designed for minimal overhead:
/// - All fields are `Copy` except `forward_costs` (which is unavoidable)
/// - Stored in a `Vec<IterationResult>` in `TrainingResult` for cache-friendly access
/// - No heap allocations except for the `forward_costs` vector
///
/// # Example
///
/// ```rust,ignore
/// let iter_result = IterationResult {
///     iteration: 1,
///     lower_bound: 1000.0,
///     upper_bound: 1200.0,
///     forward_costs: vec![1150.0, 1250.0],
///     gap: 200.0,
///     relative_gap: 0.2,
///     iteration_time: Duration::from_secs(5),
/// };
///
/// println!("Iteration {} gap: {:.2}%", iter_result.iteration, iter_result.relative_gap * 100.0);
/// ```
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Iteration number (1-indexed).
    pub iteration: usize,

    /// Lower bound from backward pass.
    ///
    /// This is a valid lower bound on the optimal value of the problem.
    /// In SDDP, the lower bound is non-decreasing (monotonic).
    pub lower_bound: f64,

    /// Average cost across all forward passes (upper bound estimate).
    ///
    /// This provides an upper bound estimate on the optimal value.
    /// The true upper bound would require infinite forward passes.
    pub upper_bound: f64,

    /// Individual forward pass costs.
    ///
    /// Stored for potential variance analysis. The upper bound is the mean of these costs.
    /// Vector is typically small (10-100 elements), so heap allocation overhead is acceptable.
    pub forward_costs: Vec<f64>,

    /// Absolute gap between upper and lower bound.
    ///
    /// `gap = upper_bound - lower_bound`
    ///
    /// This should be non-negative (within numerical tolerance).
    pub gap: f64,

    /// Relative gap (gap / |lower_bound|).
    ///
    /// Provides scale-independent convergence metric. Set to `f64::INFINITY`
    /// if `lower_bound` is very close to zero (< 1e-10).
    pub relative_gap: f64,

    /// Time taken for this iteration (forward + backward pass).
    pub iteration_time: Duration,
}

/// Result of SDDP training containing convergence history and final statistics.
///
/// This struct captures all convergence data from the training process, enabling:
/// - Numerical validation in tests (checking bounds converge to expected values)
/// - Convergence analysis (monotonicity, gap reduction, stability)
/// - Debugging convergence issues
/// - Performance analysis (timing per iteration)
///
/// # Performance Considerations
///
/// - `iterations` vector is pre-allocated with `Vec::with_capacity(num_iterations)` to avoid reallocations
/// - All scalar fields are `Copy` types (zero-cost to access)
/// - No runtime overhead compared to not storing results (data already computed)
/// - `iterations` vector has good cache locality for sequential access
///
/// # Example
///
/// ```rust,ignore
/// // After training:
/// let result = sddp.train(100, 20, &saa)?;
///
/// // Check convergence
/// if result.converged(1e-3) {
///     println!("Converged! Final gap: {:.4}", result.final_gap());
/// }
///
/// // Analyze convergence history
/// for iter in result.iterations() {
///     println!("Iteration {}: LB={:.2}, UB={:.2}, Gap={:.2}",
///              iter.iteration, iter.lower_bound, iter.upper_bound, iter.gap);
/// }
///
/// // Extract bounds for plotting
/// let lower_bounds = result.lower_bounds();
/// let upper_bounds = result.upper_bounds();
/// ```
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Iteration-by-iteration convergence history.
    ///
    /// This vector is pre-allocated with capacity `num_iterations` for optimal performance.
    /// Access via `iterations()` method for clarity.
    iterations: Vec<IterationResult>,

    /// Final lower bound (from last iteration).
    pub final_lower_bound: f64,

    /// Final upper bound (from last iteration).
    pub final_upper_bound: f64,

    /// Best (lowest) upper bound observed during training.
    ///
    /// Since upper bounds can fluctuate (stochastic sampling), we track
    /// the best value seen. This provides a tighter upper bound estimate.
    pub best_upper_bound: f64,

    /// Iteration where best upper bound was achieved (1-indexed).
    pub best_iteration: usize,

    /// Total training time (all iterations).
    pub total_time: Duration,

    /// Number of cuts in final policy.
    ///
    /// Useful for understanding policy complexity and memory usage.
    pub num_cuts: usize,

    /// Reason training terminated.
    pub termination_reason: TerminationReason,
}

/// Reason why SDDP training terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// Completed all requested iterations.
    IterationLimit,

    /// Reached convergence tolerance (not yet implemented).
    #[allow(dead_code)]
    Converged { gap_tolerance_thousandths: u32 },

    /// Time limit reached (not yet implemented).
    #[allow(dead_code)]
    TimeLimit,
}

impl TrainingResult {
    /// Get the final absolute gap (upper_bound - lower_bound).
    ///
    /// This should be non-negative within numerical tolerance.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = sddp.train(100, 20, &saa)?;
    /// println!("Final gap: {:.2}", result.final_gap());
    /// ```
    #[inline]
    pub fn final_gap(&self) -> f64 {
        self.final_upper_bound - self.final_lower_bound
    }

    /// Get the final relative gap (gap / |lower_bound|).
    ///
    /// Returns `f64::INFINITY` if lower bound is very close to zero (< 1e-10)
    /// to avoid division by zero.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = sddp.train(100, 20, &saa)?;
    /// println!("Relative gap: {:.2}%", result.relative_gap() * 100.0);
    /// ```
    #[inline]
    pub fn relative_gap(&self) -> f64 {
        if self.final_lower_bound.abs() < 1e-10 {
            f64::INFINITY
        } else {
            self.final_gap() / self.final_lower_bound.abs()
        }
    }

    /// Check if algorithm converged within specified absolute gap tolerance.
    ///
    /// # Arguments
    ///
    /// * `gap_tolerance` - Maximum acceptable absolute gap
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let result = sddp.train(100, 20, &saa)?;
    /// if result.converged(100.0) {
    ///     println!("Converged within gap tolerance of 100.0");
    /// }
    /// ```
    #[inline]
    pub fn converged(&self, gap_tolerance: f64) -> bool {
        self.final_gap().abs() <= gap_tolerance
    }

    /// Get vector of all lower bounds across iterations.
    ///
    /// Useful for plotting convergence or checking monotonicity.
    ///
    /// # Performance
    ///
    /// Allocates a new vector and copies values. For frequent access,
    /// consider iterating over `iterations()` directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let lower_bounds = result.lower_bounds();
    /// for (i, lb) in lower_bounds.iter().enumerate() {
    ///     println!("Iteration {}: LB = {:.2}", i + 1, lb);
    /// }
    /// ```
    pub fn lower_bounds(&self) -> Vec<f64> {
        self.iterations.iter().map(|it| it.lower_bound).collect()
    }

    /// Get vector of all upper bounds across iterations.
    ///
    /// Useful for plotting convergence or analyzing upper bound variance.
    ///
    /// # Performance
    ///
    /// Allocates a new vector and copies values. For frequent access,
    /// consider iterating over `iterations()` directly.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let upper_bounds = result.upper_bounds();
    /// println!("Upper bound variance: {:.2}", variance(&upper_bounds));
    /// ```
    pub fn upper_bounds(&self) -> Vec<f64> {
        self.iterations.iter().map(|it| it.upper_bound).collect()
    }

    /// Access iteration results.
    ///
    /// Provides read-only access to the complete iteration history.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for iter in result.iterations() {
    ///     if iter.gap < 100.0 {
    ///         println!("Iteration {} has small gap: {:.2}", iter.iteration, iter.gap);
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn iterations(&self) -> &[IterationResult] {
        &self.iterations
    }
}

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
    #[allow(clippy::too_many_arguments)]
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
                    node_data.load_stochastic_process.as_ref(),
                    node_data.inflow_stochastic_process.as_ref(),
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
                    saa.get_branching_count_at_stage(id).unwrap_or_else(
                        || panic!("Missing branching count for node {}", id)
                    )
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
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
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
        past_node_ids: &[usize],
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
        past_node_ids: &[usize],
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
            node_data_graph
                .get_node(id)
                .ok_or_else(|| {
                    format!("Could not find node data for node {}", id)
                })?
                .data
                .risk_measure
                .as_ref(),
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
    branching_realizations: &[subproblem::Realization],
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
        child_data_node.data.risk_measure.as_ref(),
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
                    node_data.load_stochastic_process.as_ref(),
                    node_data.inflow_stochastic_process.as_ref(),
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
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
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
    pub future_cost_function_graph:
        graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,

    // initial state
    initial_condition: initial_condition::InitialCondition,

    // for rng reproducibility
    seed: u64,

    // helpers for traversing the graphs
    pre_study_id: usize,
    pub study_period_ids: Vec<usize>,
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
                    "Could not find node 1 for counting cuts".to_string()
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

        let all_sampled_noises: Vec<_> = (0..num_simulation_scenarios)
            .map(|_| saa.sample_scenario(&mut rng))
            .collect();

        let mut simulation_handlers: Vec<SddpSimulationHandler> = (0
            ..num_simulation_scenarios)
            .map(|_| {
                SddpSimulationHandler::new(
                    &self.pre_study_id,
                    &self.node_data_graph,
                    &self.initial_condition,
                )
            })
            .collect::<Result<_, _>>()?;

        let simulation_costs: Vec<f64> = simulation_handlers
            .par_iter_mut()
            .zip(all_sampled_noises.par_iter())
            .map(|(handler, noises)| {
                handler.forward(
                    noises.to_vec(),
                    &self.node_data_graph,
                    &self.graph_bfs_table,
                    &self.study_period_ids,
                )
            })
            .collect::<Result<Vec<f64>, String>>()?;

        let _simulation_costs: Vec<f64> = simulation_handlers
            .par_iter()
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
        data_node.data.load_stochastic_process.as_ref(),
        data_node.data.inflow_stochastic_process.as_ref(),
        realization_container,
    )?;
    Ok(())
}

fn reuse_forward_basis(
    subproblem: &mut subproblem::Subproblem,
    node_forward_realization: &subproblem::Realization,
) -> Result<(), String> {
    if !node_forward_realization.basis.columns().is_empty() {
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
    branching_realizations: &[subproblem::Realization],
    risk_measure: &dyn risk_measure::RiskMeasure,
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

#[cfg(test)]
mod tests {

    use super::*;
    use rand_distr::{LogNormal, Normal};

    #[test]
    fn test_forward_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                )
                .unwrap(),
            )
            .unwrap();
        let node_0_id = node_data_graph
            .add_node(
                NodeData::new(
                    0,
                    0,
                    0,
                    "2025-01-01T00:00:00Z",
                    "2025-02-01T00:00:00Z",
                    subproblem::StudyPeriodKind::Study,
                    system::System::default(), // Assuming System::default() is cheap or test-only
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                )
                .unwrap(),
            )
            .unwrap();
        let node_1_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        let node_2_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, node_0_id).unwrap();
        node_data_graph.add_edge(node_0_id, node_1_id).unwrap();
        node_data_graph.add_edge(node_1_id, node_2_id).unwrap();
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let example_noises = scenario::SampledBranchingNoises {
            load_noises: vec![75.0],
            inflow_noises: vec![10.0],
            num_load_entities: 1,
            num_inflow_entities: 1,
        };
        let sampled_noises = vec![
            &example_noises,
            &example_noises,
            &example_noises,
            &example_noises,
        ];

        let pre_study_id = node_data_graph
            .get_node_id_with(|node| {
                node.kind == subproblem::StudyPeriodKind::PreStudy
            })
            .unwrap_or_else(|| {
                node_data_graph
                    .add_node(
                        NodeData::new(
                            -1,
                            0,
                            0,
                            "1970-01-01T00:00:00Z",
                            "1970-01-01T00:00:00Z",
                            subproblem::StudyPeriodKind::PreStudy,
                            system::System::default(),
                            "expectation",
                            "naive",
                            "naive",
                            "storage",
                        )
                        .unwrap(),
                    )
                    .unwrap()
            });

        let study_period_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::Study
        });

        let graph_bfs_table: Vec<Vec<usize>> = study_period_ids
            .iter()
            .map(|id| node_data_graph.get_bfs(*id, true))
            .collect();

        let mut handler = SddpTrainHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
            &generate_test_saa_for_four_stages(),
        )
        .unwrap();

        handler
            .forward(
                sampled_noises,
                &node_data_graph,
                &graph_bfs_table,
                &study_period_ids,
            )
            .unwrap();
    }

    fn generate_test_saa_for_four_stages() -> scenario::SAA {
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
        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                )
                .unwrap(),
            )
            .unwrap();
        let node_0_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        let node_1_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        let node_2_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, node_0_id).unwrap();
        node_data_graph.add_edge(node_0_id, node_1_id).unwrap();
        node_data_graph.add_edge(node_1_id, node_2_id).unwrap();
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::new()))
            });

        let example_noises = scenario::SampledBranchingNoises {
            load_noises: vec![75.0],
            inflow_noises: vec![10.0],
            num_load_entities: 1,
            num_inflow_entities: 1,
        };
        let sampled_noises = vec![
            &example_noises,
            &example_noises,
            &example_noises,
            &example_noises,
        ];

        let pre_study_id = node_data_graph
            .get_node_id_with(|node| {
                node.kind == subproblem::StudyPeriodKind::PreStudy
            })
            .unwrap_or_else(|| {
                node_data_graph
                    .add_node(
                        NodeData::new(
                            -1,
                            0,
                            0,
                            "1970-01-01T00:00:00Z",
                            "1970-01-01T00:00:00Z",
                            subproblem::StudyPeriodKind::PreStudy,
                            system::System::default(),
                            "expectation",
                            "naive",
                            "naive",
                            "storage",
                        )
                        .unwrap(),
                    )
                    .unwrap()
            });

        let study_period_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::Study
        });

        let graph_bfs_table: Vec<Vec<usize>> = study_period_ids
            .iter()
            .map(|id| node_data_graph.get_bfs(*id, true))
            .collect();

        let saa = generate_test_saa_for_four_stages();

        let mut handler = SddpTrainHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
            &saa,
        )
        .unwrap();

        handler
            .forward(
                sampled_noises,
                &node_data_graph,
                &graph_bfs_table,
                &study_period_ids,
            )
            .unwrap();

        let current_stage_original_idx = 1; // Corresponds to node 1
        let id = study_period_ids[current_stage_original_idx];
        let past_node_ids =
            graph_bfs_table.get(current_stage_original_idx).unwrap();

        handler
            .backward_step_at_node(
                id,
                past_node_ids,
                &node_data_graph,
                &saa,
                &future_cost_function_graph,
            )
            .unwrap();
    }

    #[test]
    fn test_train_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                )
                .unwrap(),
            )
            .unwrap();
        let prev_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );

        for new_id_isize in 1..4 {
            let new_id = node_data_graph
                .add_node(
                    NodeData::new(
                        new_id_isize,
                        new_id_isize.try_into().unwrap(),
                        new_id_isize.try_into().unwrap(),
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        system::System::default(),
                        "expectation",
                        "naive",
                        "naive",
                        "storage",
                    )
                    .unwrap(),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }

        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let saa = scenario_generator.generate(0);

        let mut sddp_algo =
            SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

        sddp_algo.train(24, 1, &saa).unwrap();
    }

    #[test]
    fn test_simulate_with_default_system() {
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system::System::default(),
                    "expectation",
                    "naive",
                    "naive",
                    "storage",
                )
                .unwrap(),
            )
            .unwrap();
        let prev_id = node_data_graph
            .add_node(
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
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            3,
        );
        for new_id_isize in 1..4 {
            let new_id = node_data_graph
                .add_node(
                    NodeData::new(
                        new_id_isize,
                        new_id_isize.try_into().unwrap(),
                        new_id_isize.try_into().unwrap(),
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        subproblem::StudyPeriodKind::Study,
                        system::System::default(),
                        "expectation",
                        "naive",
                        "naive",
                        "storage",
                    )
                    .unwrap(),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                3,
            );
        }
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let saa = scenario_generator.generate(0);

        let mut sddp_algo =
            SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

        sddp_algo.train(24, 1, &saa).unwrap();

        sddp_algo.simulate(100, &saa).unwrap();
    }

    // ====================================================================
    // Unit tests for TrainingResult and IterationResult (T2.1)
    // ====================================================================

    /// Helper function to create a test TrainingResult with realistic data.
    fn create_test_training_result() -> TrainingResult {
        let iterations = vec![
            IterationResult {
                iteration: 1,
                lower_bound: 1000.0,
                upper_bound: 1500.0,
                forward_costs: vec![1400.0, 1600.0],
                gap: 500.0,
                relative_gap: 0.5,
                iteration_time: Duration::from_secs(1),
            },
            IterationResult {
                iteration: 2,
                lower_bound: 1200.0,
                upper_bound: 1350.0,
                forward_costs: vec![1300.0, 1400.0],
                gap: 150.0,
                relative_gap: 0.125,
                iteration_time: Duration::from_secs(1),
            },
            IterationResult {
                iteration: 3,
                lower_bound: 1250.0,
                upper_bound: 1300.0,
                forward_costs: vec![1280.0, 1320.0],
                gap: 50.0,
                relative_gap: 0.04,
                iteration_time: Duration::from_millis(950),
            },
        ];

        TrainingResult {
            iterations,
            final_lower_bound: 1250.0,
            final_upper_bound: 1300.0,
            best_upper_bound: 1300.0,
            best_iteration: 3,
            total_time: Duration::from_millis(2950),
            num_cuts: 15,
            termination_reason: TerminationReason::IterationLimit,
        }
    }

    #[test]
    fn test_training_result_final_gap() {
        let result = create_test_training_result();
        assert_eq!(result.final_gap(), 50.0);
    }

    #[test]
    fn test_training_result_relative_gap() {
        let result = create_test_training_result();
        let expected_relative_gap = 50.0 / 1250.0;
        assert!((result.relative_gap() - expected_relative_gap).abs() < 1e-10);
        assert!((result.relative_gap() - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_training_result_relative_gap_zero_lower_bound() {
        let mut result = create_test_training_result();
        result.final_lower_bound = 0.0;
        result.final_upper_bound = 100.0;

        // Should return infinity when lower bound is zero
        assert_eq!(result.relative_gap(), f64::INFINITY);
    }

    #[test]
    fn test_training_result_relative_gap_near_zero_lower_bound() {
        let mut result = create_test_training_result();
        result.final_lower_bound = 1e-11; // Below threshold
        result.final_upper_bound = 100.0;

        // Should return infinity when lower bound is very close to zero
        assert_eq!(result.relative_gap(), f64::INFINITY);
    }

    #[test]
    fn test_training_result_converged_within_tolerance() {
        let result = create_test_training_result();

        // Final gap is 50.0
        assert!(result.converged(50.0)); // Exactly at tolerance
        assert!(result.converged(100.0)); // Well within tolerance
        assert!(result.converged(50.1)); // Just within tolerance
    }

    #[test]
    fn test_training_result_not_converged() {
        let result = create_test_training_result();

        // Final gap is 50.0
        assert!(!result.converged(49.9)); // Just outside tolerance
        assert!(!result.converged(10.0)); // Well outside tolerance
        assert!(!result.converged(0.0)); // Zero tolerance
    }

    #[test]
    fn test_training_result_lower_bounds() {
        let result = create_test_training_result();
        let lower_bounds = result.lower_bounds();

        assert_eq!(lower_bounds.len(), 3);
        assert_eq!(lower_bounds[0], 1000.0);
        assert_eq!(lower_bounds[1], 1200.0);
        assert_eq!(lower_bounds[2], 1250.0);
    }

    #[test]
    fn test_training_result_upper_bounds() {
        let result = create_test_training_result();
        let upper_bounds = result.upper_bounds();

        assert_eq!(upper_bounds.len(), 3);
        assert_eq!(upper_bounds[0], 1500.0);
        assert_eq!(upper_bounds[1], 1350.0);
        assert_eq!(upper_bounds[2], 1300.0);
    }

    #[test]
    fn test_training_result_iterations_access() {
        let result = create_test_training_result();
        let iterations = result.iterations();

        assert_eq!(iterations.len(), 3);
        assert_eq!(iterations[0].iteration, 1);
        assert_eq!(iterations[1].iteration, 2);
        assert_eq!(iterations[2].iteration, 3);

        // Check that we can access fields
        assert_eq!(iterations[0].lower_bound, 1000.0);
        assert_eq!(iterations[0].upper_bound, 1500.0);
        assert_eq!(iterations[0].gap, 500.0);
    }

    #[test]
    fn test_iteration_result_forward_costs_access() {
        let iter_result = IterationResult {
            iteration: 1,
            lower_bound: 1000.0,
            upper_bound: 1200.0,
            forward_costs: vec![1150.0, 1200.0, 1250.0],
            gap: 200.0,
            relative_gap: 0.2,
            iteration_time: Duration::from_secs(1),
        };

        assert_eq!(iter_result.forward_costs.len(), 3);
        assert_eq!(iter_result.forward_costs[0], 1150.0);
        assert_eq!(iter_result.forward_costs[1], 1200.0);
        assert_eq!(iter_result.forward_costs[2], 1250.0);

        // Verify average equals upper bound
        let avg: f64 = iter_result.forward_costs.iter().sum::<f64>() / 3.0;
        assert!((avg - iter_result.upper_bound).abs() < 1e-10);
    }

    #[test]
    fn test_training_result_single_iteration() {
        let iterations = vec![IterationResult {
            iteration: 1,
            lower_bound: 1000.0,
            upper_bound: 1100.0,
            forward_costs: vec![1100.0],
            gap: 100.0,
            relative_gap: 0.1,
            iteration_time: Duration::from_secs(1),
        }];

        let result = TrainingResult {
            iterations,
            final_lower_bound: 1000.0,
            final_upper_bound: 1100.0,
            best_upper_bound: 1100.0,
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 5,
            termination_reason: TerminationReason::IterationLimit,
        };

        assert_eq!(result.final_gap(), 100.0);
        assert_eq!(result.iterations().len(), 1);
        assert_eq!(result.lower_bounds().len(), 1);
        assert_eq!(result.upper_bounds().len(), 1);
    }

    #[test]
    fn test_training_result_best_upper_bound_tracking() {
        let result = create_test_training_result();

        // Best upper bound should be 1300.0 (from iteration 3)
        assert_eq!(result.best_upper_bound, 1300.0);
        assert_eq!(result.best_iteration, 3);

        // Verify it's indeed the minimum
        let all_upper_bounds = result.upper_bounds();
        let min_upper_bound = all_upper_bounds
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert_eq!(result.best_upper_bound, min_upper_bound);
    }

    #[test]
    fn test_termination_reason_copy_semantics() {
        // TerminationReason should be Copy (zero-cost)
        let reason1 = TerminationReason::IterationLimit;
        let reason2 = reason1; // Should be copy, not move
        let _reason3 = reason1; // Should still be usable

        assert_eq!(reason1, reason2);
    }

    #[test]
    fn test_training_result_negative_gap_edge_case() {
        // In theory, gap should never be negative, but test handling
        let mut result = create_test_training_result();
        result.final_lower_bound = 1500.0;
        result.final_upper_bound = 1400.0;

        let gap = result.final_gap();
        assert_eq!(gap, -100.0);

        // converged() uses abs(), so should still work correctly
        assert!(result.converged(100.0));
        assert!(result.converged(150.0));
        assert!(!result.converged(50.0));
    }

    #[test]
    fn test_training_result_large_gaps() {
        let result = TrainingResult {
            iterations: vec![IterationResult {
                iteration: 1,
                lower_bound: 1e6,
                upper_bound: 1e9,
                forward_costs: vec![1e9],
                gap: 1e9 - 1e6,
                relative_gap: (1e9 - 1e6) / 1e6,
                iteration_time: Duration::from_secs(1),
            }],
            final_lower_bound: 1e6,
            final_upper_bound: 1e9,
            best_upper_bound: 1e9,
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 1,
            termination_reason: TerminationReason::IterationLimit,
        };

        // Should handle large numbers correctly
        assert!((result.final_gap() - (1e9 - 1e6)).abs() < 1e3);
        assert!(result.relative_gap() > 900.0); // Very large relative gap
        assert!(!result.converged(1e8));
    }
}
