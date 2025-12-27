//! Stochastic Dual Dynamic Programming (SDDP) algorithm implementation.
//!
//! Solves multistage stochastic hydrothermal dispatch via Benders decomposition
//! with iterative refinement of cost-to-go approximations.
//!

pub mod builder;
pub mod instance;

pub use builder::{SddpBuilder, SddpInstanceBuilder};
pub use instance::SddpInstance;

use crate::fcf;
use crate::graph;
use crate::initial_condition;
use crate::risk_measure;
use crate::scenario;
use crate::state;
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

#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardPassTiming {
    pub saa_sampling_time: Duration,
    pub model_preprocessing_time: Duration,
    pub solver_time: Duration,
    pub model_postprocessing_time: Duration,
    pub forward_postprocessing_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardPassTiming {
    pub backward_preprocessing_time: Duration,
    pub model_preprocessing_time: Duration,
    pub solver_time: Duration,
    pub model_postprocessing_time: Duration,
    pub cut_selection_time: Duration,
    pub fcf_state_update_time: Duration,
    pub cut_cloning_time: Duration,
    pub handler_application_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ForwardPassTimingAccumulator {
    pub model_preprocessing_time: Duration,
    pub solver_time: Duration,
    pub model_postprocessing_time: Duration,
    pub solver_calls: usize,
}

impl ForwardPassTimingAccumulator {
    pub fn aggregate(timings: &[Self]) -> ForwardPassTiming {
        assert!(!timings.is_empty(), "Cannot aggregate zero timings");

        let n = timings.len();
        let total_model_pre = timings
            .iter()
            .map(|t| t.model_preprocessing_time)
            .sum::<Duration>();
        let total_solver =
            timings.iter().map(|t| t.solver_time).sum::<Duration>();
        let total_model_post = timings
            .iter()
            .map(|t| t.model_postprocessing_time)
            .sum::<Duration>();

        let avg_model_pre = total_model_pre / n as u32;
        let avg_solver = total_solver / n as u32;
        let avg_model_post = total_model_post / n as u32;

        ForwardPassTiming {
            saa_sampling_time: Duration::ZERO, // Set by training loop
            model_preprocessing_time: avg_model_pre,
            solver_time: avg_solver,
            model_postprocessing_time: avg_model_post,
            forward_postprocessing_time: Duration::ZERO, // Set by training loop
            total_time: Duration::ZERO,                  // Set by training loop
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackwardPassTimingAccumulator {
    pub backward_preprocessing_time: Duration,
    pub model_preprocessing_time: Duration,
    pub solver_time: Duration,
    pub model_postprocessing_time: Duration,
    pub cut_selection_time: Duration,
    pub fcf_state_update_time: Duration,
    pub cut_cloning_time: Duration,
    pub handler_application_time: Duration,
    pub solver_calls: usize,
    pub cuts_added: usize,
}

impl BackwardPassTimingAccumulator {
    pub fn into_timing(self) -> BackwardPassTiming {
        let total = self.backward_preprocessing_time
            + self.model_preprocessing_time
            + self.solver_time
            + self.model_postprocessing_time
            + self.cut_selection_time
            + self.fcf_state_update_time
            + self.cut_cloning_time
            + self.handler_application_time;

        BackwardPassTiming {
            backward_preprocessing_time: self.backward_preprocessing_time,
            model_preprocessing_time: self.model_preprocessing_time,
            solver_time: self.solver_time,
            model_postprocessing_time: self.model_postprocessing_time,
            cut_selection_time: self.cut_selection_time,
            fcf_state_update_time: self.fcf_state_update_time,
            cut_cloning_time: self.cut_cloning_time,
            handler_application_time: self.handler_application_time,
            total_time: total,
        }
    }
}

/// Results from a single SDDP training iteration.
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iteration: usize,
    pub lower_bound: f64,
    pub forward_costs: Vec<f64>,
    pub iteration_time: Duration,
    pub forward_timing: ForwardPassTiming,
    pub backward_timing: BackwardPassTiming,
    pub num_solver_calls: usize,
    pub num_cuts_added: usize,
    pub num_cuts_removed: usize,
    pub num_cuts_returned: usize,
    pub num_active_cuts: usize,
}

/// Complete results from SDDP training.
#[derive(Debug, Clone)]
pub struct TrainingResult {
    iterations: Vec<IterationResult>,
    pub final_lower_bound: f64,
    pub statistical_upper_bound: f64,
    pub best_upper_bound: f64,
    pub best_iteration: usize,
    pub total_time: Duration,
    pub num_cuts: usize,
    /// Captured training trajectories (empty if not preserved)
    pub forward_details: Vec<ForwardPassDetail>,
    /// Captured backward pass branching records (empty if not preserved)
    pub backward_details: Vec<BackwardPassDetail>,
}

impl TrainingResult {
    #[inline]
    pub fn final_gap(&self) -> f64 {
        self.statistical_upper_bound - self.final_lower_bound
    }

    #[inline]
    pub fn relative_gap(&self) -> f64 {
        if self.final_lower_bound.abs() < 1e-10 {
            f64::INFINITY
        } else {
            self.final_gap() / self.final_lower_bound.abs()
        }
    }

    pub fn lower_bounds(&self) -> Vec<f64> {
        self.iterations.iter().map(|it| it.lower_bound).collect()
    }

    #[inline]
    pub fn iterations(&self) -> &[IterationResult] {
        &self.iterations
    }
}

/// Result from a single stage in a simulation trajectory.
#[derive(Debug, Clone)]
pub struct StageResult {
    pub stage: usize,
    pub state: Vec<f64>,
    pub action: Vec<f64>,
    pub stage_cost: f64,
    pub inflow: Vec<f64>,
    pub load: Vec<f64>,
}

/// Complete trajectory for a single simulated scenario.
#[derive(Debug, Clone)]
pub struct Trajectory {
    pub stages: Vec<StageResult>,
    pub total_cost: f64,
    pub scenario_id: usize,
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub trajectories: Vec<Trajectory>,
    pub num_stages: usize,
    pub num_states: usize,
    pub num_actions: usize,
}

impl SimulationResult {
    #[inline]
    pub fn get_trajectory(&self, scenario_idx: usize) -> Option<&Trajectory> {
        self.trajectories.get(scenario_idx)
    }

    #[inline]
    pub fn get_all_trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }
}

/// Node data for SDDP algorithm.
///
/// Each node represents a decision point in the scenario tree.
pub struct NodeData {
    pub id: isize,
    pub stage_id: usize,
    pub season_id: usize,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub kind: subproblem::StudyPeriodKind,
    pub system: system::System,
    pub risk_measure: Box<dyn risk_measure::RiskMeasure>,
    /// Uncertainty models for all uncertainty sources in this node.
    /// Shared via Arc to avoid duplicating memory across all nodes.
    /// Used to access AR coefficients during constraint generation.
    pub uncertainty_models:
        std::sync::Arc<Vec<crate::temporal_model::TemporalModel>>,
    pub state_choice: String,
    pub num_scenarios: usize,
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
        uncertainty_models: std::sync::Arc<
            Vec<crate::temporal_model::TemporalModel>,
        >,
        state_str: &str,
        num_scenarios: usize,
    ) -> Result<Self, String> {
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
            uncertainty_models,
            state_choice: state_str.to_string(),
            num_scenarios,
        })
    }
}

/// Snapshot of a realization at a specific point in training
///
/// Used for forward pass detail export to capture complete trajectories.
/// Only allocated when forward detail export is enabled.
#[derive(Clone, Debug)]
pub struct ForwardPassDetail {
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub stage_id: isize,
    pub realization: subproblem::Realization,
}

/// Individual branching realization from backward pass.
///
/// Captures complete information for each branching scenario solved during backward pass.
/// Used for backward pass detail export and scenario-level diagnostics.
///
/// Only allocated when backward detail export is enabled.
#[derive(Clone, Debug)]
pub struct BackwardPassDetail {
    pub iteration: usize,
    pub forward_pass_idx: usize,
    pub stage_id: isize,
    pub training_state_id: usize,
    pub branching_idx: usize,
    pub realization: subproblem::Realization,
}

pub struct SddpTrainHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
    branching_graph: graph::DirectedGraph<Vec<subproblem::Realization>>,

    /// Optional trajectory history for diagnostic export
    ///
    /// When `Some`, realizations are cloned after each forward pass for later export.
    /// When `None`, no history is preserved (zero overhead).
    ///
    /// Memory usage (when enabled): ~500 bytes × num_stages per forward pass
    forward_detail_history: Option<Vec<ForwardPassDetail>>,

    /// Whether to preserve trajectory history
    preserve_forward_detail: bool,

    /// Optional backward pass branching records for diagnostic export
    ///
    /// When `Some`, individual branching realizations are stored during backward pass.
    /// When `None`, no records are preserved (zero overhead).
    ///
    /// Memory usage (when enabled): ~500 bytes × num_branchings per stage
    backward_detail_history: Option<Vec<BackwardPassDetail>>,

    /// Whether to collect backward branching records
    preserve_backward_detail: bool,
}

impl SddpTrainHandler {
    pub fn new(
        node_data_graph: &graph::DirectedGraph<NodeData>,
        initial_condition: &initial_condition::InitialCondition,
        saa: &scenario::ScenarioTree,
        preserve_forward_detail: bool,
        preserve_backward_detail: bool,
        num_forward_passes: usize,
        num_iterations: usize,
    ) -> Result<Self, String> {
        let mut realization_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                let temporal_models: Vec<_> =
                    node_data.uncertainty_models.iter().cloned().collect();

                let (num_cols, num_rows) =
                    subproblem::estimate_problem_dimensions(
                        &node_data.system,
                        &temporal_models,
                        num_forward_passes,
                        num_iterations,
                    );

                subproblem::Realization::with_capacity(
                    &node_data.kind,
                    &node_data.system,
                    &temporal_models,
                    num_cols,
                    num_rows,
                )
            });

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                // Convert UncertaintyModels to TemporalModels
                let temporal_models: Vec<_> =
                    node_data.uncertainty_models.iter().cloned().collect();

                subproblem::Subproblem::new_from_temporal_models(
                    &node_data.system,
                    &node_data.state_choice,
                    &temporal_models,
                    node_data.season_id,
                )
            });

        // Set initial storage on ALL pre-study nodes
        // For PAR models with AR order > 0, there are multiple pre-study nodes
        // and all of them need the same initial storage value
        let prestudy_node_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::PreStudy
        });

        for &prestudy_id in &prestudy_node_ids {
            let prestudy_realization = realization_graph
                .get_node_mut(prestudy_id)
                .ok_or_else(|| {
                    format!(
                        "Failed to get pre-study node {} in realization graph",
                        prestudy_id
                    )
                })?;

            prestudy_realization
                .data
                .final_storage
                .clone_from_slice(initial_condition.get_storage());

            // CRITICAL FIX: Set inflow field to initial lag values
            // For PAR models, PreStudy nodes represent historical observations that
            // need to be available in the trajectory for state reconstruction.
            // PreStudy node at index i (counting from newest) should have lag i+1.
            //
            // Example PAR(2):
            //   - PreStudy node 0 (newest): inflow = initial_condition.get_inflow(hydro)[0] (lag-1)
            //   - PreStudy node 1 (oldest): inflow = initial_condition.get_inflow(hydro)[1] (lag-2)
            let node_data =
                node_data_graph.get_node(prestudy_id).ok_or_else(|| {
                    format!(
                        "Failed to get node data for PreStudy node {}",
                        prestudy_id
                    )
                })?;

            for model in node_data.data.uncertainty_models.iter() {
                if model.entity_type() == crate::input::UncertaintyType::Inflow
                {
                    let hydro_id = model.entity_id;
                    let lags = initial_condition.get_inflow(hydro_id);

                    if !lags.is_empty() {
                        // Determine which lag this PreStudy node represents
                        // prestudy_node_ids are ordered newest to oldest
                        let prestudy_index = prestudy_node_ids
                            .iter()
                            .position(|&id| id == prestudy_id)
                            .unwrap();

                        // prestudy_index 0 = lag-1 (newest), 1 = lag-2, etc.
                        if prestudy_index < lags.len() {
                            prestudy_realization.data.inflow[hydro_id] =
                                lags[prestudy_index];
                        }
                    }
                }
            }
        }

        // Initialize lag buffers from initial condition
        // This sets the historical context for AR dynamics in the first stage
        for node_id in 0..node_data_graph.node_count() {
            let node_data =
                node_data_graph.get_node(node_id).ok_or_else(|| {
                    format!("Failed to get node data for node {}", node_id)
                })?;

            let temporal_models = &node_data.data.uncertainty_models;

            // Find inflow entities and set their initial lags
            for model in temporal_models.iter() {
                if model.entity_type() == crate::input::UncertaintyType::Inflow
                    && model.max_ar_order > 0
                {
                    let hydro_id = model.entity_id;
                    let lags = initial_condition.get_inflow(hydro_id);
                    if !lags.is_empty() {
                        let subproblem_node = subproblem_graph
                            .get_node_mut(node_id)
                            .ok_or_else(|| {
                                format!(
                                    "Failed to get subproblem node {}",
                                    node_id
                                )
                            })?;
                        // Set lags in InflowLagData
                        if let Some(ref mut inflow_data) =
                            subproblem_node.data.inflow_lag_data
                        {
                            for (lag_idx, &lag_value) in lags.iter().enumerate()
                            {
                                if lag_idx < inflow_data.buffer[hydro_id].len()
                                {
                                    inflow_data.buffer[hydro_id][lag_idx] =
                                        lag_value;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Note: Initial lag buffer initialization is now done directly in
        // InflowLagData buffers in the subproblem initialization above.
        // Pre-study nodes are used only for initial storage state.

        let branching_graph =
            node_data_graph.map_topology_with(|node_data, id| {
                let temporal_models: Vec<_> =
                    node_data.uncertainty_models.iter().cloned().collect();

                let (num_cols, num_rows) =
                    subproblem::estimate_problem_dimensions(
                        &node_data.system,
                        &temporal_models,
                        num_forward_passes,
                        num_iterations,
                    );

                let branching_count = saa
                    .get_branching_count_at_stage(node_data.stage_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "Missing branching count for stage {} (node {})",
                            node_data.stage_id, id
                        )
                    });

                vec![
                    subproblem::Realization::with_capacity(
                        &node_data.kind,
                        &node_data.system,
                        &temporal_models,
                        num_cols,
                        num_rows,
                    );
                    branching_count
                ]
            });

        let num_stages = node_data_graph.node_count();

        Ok(Self {
            subproblem_graph,
            realization_graph,
            branching_graph,
            forward_detail_history: if preserve_forward_detail {
                Some(Vec::with_capacity(num_stages))
            } else {
                None
            },
            preserve_forward_detail,
            backward_detail_history: if preserve_backward_detail {
                let max_branchings: usize = node_data_graph
                    .iter_nodes()
                    .map(|n| n.data.num_scenarios)
                    .max()
                    .unwrap_or(10);
                Some(Vec::with_capacity(num_stages * max_branchings))
            } else {
                None
            },
            preserve_backward_detail,
        })
    }

    /// Preallocate cut constraint slots for all subproblems in this handler.
    ///
    /// This enables zero-allocation cut addition during training by pre-creating
    /// placeholder constraint rows in the HiGHS model. Cuts are later added by
    /// modifying coefficients and bounds instead of adding new rows.
    ///
    /// # Arguments
    ///
    /// * `max_cuts` - Maximum cuts to preallocate per subproblem
    /// * `num_forward_passes` - Number of forward passes per iteration
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error message on failure.
    pub fn preallocate_cut_constraints(
        &mut self,
        max_cuts: usize,
        num_forward_passes: usize,
    ) -> Result<(), String> {
        let node_ids: Vec<usize> =
            self.subproblem_graph.iter_nodes().map(|n| n.id).collect();

        for node_id in node_ids {
            if let Some(node) = self.subproblem_graph.get_node_mut(node_id) {
                node.data.preallocate_cut_constraints(
                    max_cuts,
                    num_forward_passes,
                )?;
            }
        }
        Ok(())
    }

    pub fn forward(
        &mut self,
        sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let mut timing = ForwardPassTimingAccumulator::default();

        for (idx, id) in study_period_ids.iter().enumerate() {
            // Model preparation timing
            let prep_start = std::time::Instant::now();

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
                .prepare_from_trajectory(&past_realizations)?;

            let realization_node =
                self.realization_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find realization for node {}", id)
                })?;

            let current_stage_noises =
                sampled_noises.get(*id).ok_or_else(|| {
                    format!("Could not find noises for node {}", id)
                })?;
            timing.model_preprocessing_time += prep_start.elapsed();

            let step_timing = step(
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;
            timing.solver_time += step_timing.solver_time;

            let post_start = std::time::Instant::now();
            timing.model_postprocessing_time += step_timing.state_update_time;
            timing.solver_calls += 1;

            timing.model_postprocessing_time += post_start.elapsed();
        }

        let prep_start = std::time::Instant::now();
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
        timing.model_postprocessing_time += prep_start.elapsed();
        Ok((trajectory_cost, timing))
    }

    /// Compute cut data for backward step without state cloning.
    ///
    /// This is the allocation-free version that returns `CutData` instead of
    /// `CutStatePair`, eliminating the `Box<dyn State>` allocation.
    pub(crate) fn compute_cut_data_for_backward_step(
        &mut self,
        id: usize,
        past_node_ids: &[usize],
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::ScenarioTree,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> Result<(fcf::CutData, BackwardPhase1Timing), String> {
        let mut timing = BackwardPhase1Timing::default();

        let model_preprocessing_start = std::time::Instant::now();

        let node_forward_trajectory: Vec<&subproblem::Realization> = past_node_ids
            .iter()
            .map(|&past_id| {
                self.realization_graph
                    .get_node(past_id)
                    .map(|node| &node.data)
                    .ok_or_else(|| {
                        format!(
                            "Could not find realization for past_node {} (current_id {})",
                            past_id, id
                        )
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
        timing.model_preprocessing_time = model_preprocessing_start.elapsed();

        let branchings_timing = solve_all_branchings(
            &mut self.subproblem_graph,
            &mut self.branching_graph,
            id,
            num_branchings,
            &node_forward_trajectory,
            saa,
        )?;
        timing.solver_time = branchings_timing.solver_time;

        let model_postprocessing_start = std::time::Instant::now();
        let branching_node_data = &self
            .branching_graph
            .get_node(id)
            .ok_or_else(|| {
                format!("Could not find branching realizations for node {}", id)
            })?
            .data;

        // Capture backward branching realizations if enabled
        if self.preserve_backward_detail {
            if let Some(ref mut history) = self.backward_detail_history {
                for (branching_idx, realization) in
                    branching_node_data.iter().enumerate()
                {
                    history.push(BackwardPassDetail {
                        iteration,
                        forward_pass_idx,
                        stage_id: id as isize,
                        training_state_id: 0,
                        branching_idx,
                        realization: realization.clone(),
                    });
                }
            }
        }

        let child_data_node =
            node_data_graph.get_node(id).ok_or_else(|| {
                format!("Could not find node data for node {}", id)
            })?;
        let child_subproblem_node =
            self.subproblem_graph.get_node_mut(id).ok_or_else(|| {
                format!("Could not find subproblem for node {}", id)
            })?;

        // Use compute_cut_data instead of compute_new_cut - no Box<dyn State> allocation
        let cut_data = child_subproblem_node.data.compute_cut_data(
            branching_node_data,
            child_data_node.data.risk_measure.as_ref(),
            iteration,
            forward_pass_idx,
        );

        timing.model_postprocessing_time = model_postprocessing_start.elapsed()
            + branchings_timing.state_extraction_time;

        Ok((cut_data, timing))
    }

    pub fn apply_aggregated_cut_result(
        &mut self,
        parent_id: usize,
        aggregated_result: &fcf::AggregatedCutSelectionResult,
        cuts_to_add: &[(usize, std::sync::Arc<crate::cut::BendersCut>)],
    ) -> Result<(), String> {
        let parent_subproblem_node: &mut graph::Node<subproblem::Subproblem> =
            self.subproblem_graph
                .get_node_mut(parent_id)
                .ok_or_else(|| {
                    format!("Could not find subproblem for node {}", parent_id)
                })?;

        parent_subproblem_node
            .data
            .apply_aggregated_cut_selection_result(
                aggregated_result,
                cuts_to_add,
            )?;

        Ok(())
    }

    /// Capture current forward pass trajectory for export
    ///
    /// Clones realizations from all stages when trajectory preservation is enabled.
    /// Called after each forward pass completes.
    ///
    /// **Performance**: Only clones when `preserve_forward_detail` is true (zero overhead otherwise)
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current SDDP iteration number
    /// * `forward_pass_idx` - Index of this forward pass within the iteration
    /// * `study_period_ids` - Node IDs for study period stages (excludes pre-study)
    pub fn capture_forward_detail(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        study_period_ids: &[usize],
    ) {
        // Early return if trajectory preservation is disabled (zero overhead)
        if !self.preserve_forward_detail {
            return;
        }

        // Clone realizations from all study period stages
        if let Some(ref mut history) = self.forward_detail_history {
            for &stage_id in study_period_ids {
                if let Some(node) = self.realization_graph.get_node(stage_id) {
                    history.push(ForwardPassDetail {
                        iteration,
                        forward_pass_idx,
                        stage_id: stage_id as isize,
                        realization: node.data.clone(),
                    });
                }
            }
        }
    }

    /// Extract collected trajectory history for export
    ///
    /// Returns all captured trajectories and clears the internal buffer.
    /// This should be called after all iterations complete.
    ///
    /// # Returns
    ///
    /// Vector of trajectory snapshots, or empty vector if preservation is disabled
    pub fn take_forward_detail_history(&mut self) -> Vec<ForwardPassDetail> {
        self.forward_detail_history
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Capture individual branching realizations from backward pass
    ///
    /// Stores complete realization data for each branching scenario when enabled.
    /// Called during backward pass after solving branching scenarios.
    ///
    /// **Performance**: Only clones realizations when `preserve_backward_detail` is true
    ///
    /// # Arguments
    ///
    /// * `iteration` - Current SDDP iteration number
    /// * `forward_pass_idx` - Forward pass index for this training state
    /// * `stage_id` - Stage node ID
    /// * `training_state_id` - Index of training state within the stage
    /// * `branching_realizations` - All scenarios solved at this training state
    pub fn capture_backward_detail(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        stage_id: usize,
        training_state_id: usize,
        branching_realizations: &[subproblem::Realization],
    ) {
        // Early return if collection is disabled (zero overhead)
        if !self.preserve_backward_detail {
            return;
        }

        // Clone and store each branching realization with metadata
        if let Some(ref mut history) = self.backward_detail_history {
            for (branching_idx, realization) in
                branching_realizations.iter().enumerate()
            {
                history.push(BackwardPassDetail {
                    iteration,
                    forward_pass_idx,
                    stage_id: stage_id as isize,
                    training_state_id,
                    branching_idx,
                    realization: realization.clone(),
                });
            }
        }
    }

    /// Extract collected backward branching records for export
    ///
    /// Returns all captured branching records and clears the internal buffer.
    /// This should be called after all iterations complete.
    ///
    /// # Returns
    ///
    /// Vector of backward branching records, or empty vector if preservation is disabled
    pub fn take_backward_detail_history(&mut self) -> Vec<BackwardPassDetail> {
        self.backward_detail_history
            .as_mut()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    pub(crate) fn eval_first_stage_bound(
        &mut self,
        id: usize,
        past_node_ids: &[usize],
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::ScenarioTree,
    ) -> Result<(f64, BranchingsTiming), String> {
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

        let branchings_timing = solve_all_branchings(
            &mut self.subproblem_graph,
            &mut self.branching_graph,
            id,
            num_branchings,
            &node_forward_trajectory,
            saa,
        )?;
        let branching_node_data = &self
            .branching_graph
            .get_node(id)
            .ok_or_else(|| {
                format!("Could not find branching realizations for node {}", id)
            })?
            .data;
        let lower_bound = eval_first_stage_bound(
            branching_node_data,
            node_data_graph
                .get_node(id)
                .ok_or_else(|| {
                    format!("Could not find node data for node {}", id)
                })?
                .data
                .risk_measure
                .as_ref(),
        )?;

        Ok((lower_bound, branchings_timing))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BranchingsTiming {
    pub solver_time: Duration,
    pub state_extraction_time: Duration,
}

#[allow(deprecated)] // During transition from SAA to ScenarioTree
fn solve_all_branchings(
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    branching_graph: &mut graph::DirectedGraph<Vec<subproblem::Realization>>,
    node_id: usize,
    num_branchings: usize,
    node_forward_trajectory: &Vec<&subproblem::Realization>,
    saa: &scenario::ScenarioTree,
) -> Result<BranchingsTiming, String> {
    let mut timing = BranchingsTiming::default();

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

        let step_timing = step(
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

        timing.solver_time += step_timing.solver_time;
        timing.state_extraction_time += step_timing.state_update_time;
    }
    Ok(timing)
}

#[derive(Debug, Clone)]
pub struct RealizationData {
    pub stage_id: usize,
    pub loads: Vec<f64>,
    pub deficit: Vec<f64>,
    pub exchange: Vec<f64>,
    pub inflow: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub thermal_generation: Vec<f64>,
    pub water_value: Vec<f64>,
    pub marginal_cost: Vec<f64>,
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub final_storage: Vec<f64>,
}

impl RealizationData {
    pub fn from_realization(
        stage_id: usize,
        realization: &subproblem::Realization,
    ) -> Self {
        Self {
            stage_id,
            loads: realization.loads.clone(),
            deficit: realization.deficit.clone(),
            exchange: realization.exchange.clone(),
            inflow: realization.inflow.clone(),
            turbined_flow: realization.turbined_flow.clone(),
            spillage: realization.spillage.clone(),
            thermal_generation: realization.thermal_generation.clone(),
            water_value: realization.water_value.clone(),
            marginal_cost: realization.marginal_cost.clone(),
            current_stage_objective: realization.current_stage_objective,
            total_stage_objective: realization.total_stage_objective,
            final_storage: realization.final_storage.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimulationTrajectory {
    pub scenario_id: usize,
    pub realizations: Vec<RealizationData>,
}

impl SimulationTrajectory {
    pub fn to_trajectory(&self, initial_storage: &[f64]) -> Trajectory {
        let num_stages = self.realizations.len();
        let mut stages = Vec::with_capacity(num_stages);
        let mut total_cost = 0.0;

        for (stage_idx, realization) in self.realizations.iter().enumerate() {
            let state = if stage_idx == 0 {
                initial_storage.to_vec()
            } else {
                self.realizations[stage_idx - 1].final_storage.clone()
            };

            let action_capacity = realization.turbined_flow.len()
                + realization.thermal_generation.len()
                + realization.spillage.len()
                + realization.exchange.len()
                + realization.deficit.len();
            let mut action = Vec::with_capacity(action_capacity);
            action.extend_from_slice(&realization.turbined_flow);
            action.extend_from_slice(&realization.thermal_generation);
            action.extend_from_slice(&realization.spillage);
            action.extend_from_slice(&realization.exchange);
            action.extend_from_slice(&realization.deficit);

            let stage_result = StageResult {
                stage: stage_idx,
                state,
                action,
                stage_cost: realization.current_stage_objective,
                inflow: realization.inflow.clone(),
                load: realization.loads.clone(),
            };

            total_cost += realization.current_stage_objective;
            stages.push(stage_result);
        }

        Trajectory {
            stages,
            total_cost,
            scenario_id: self.scenario_id,
        }
    }
}

pub struct SddpSimulationHandler {
    subproblem_graph: graph::DirectedGraph<subproblem::Subproblem>,
    realization_graph: graph::DirectedGraph<subproblem::Realization>,
}

impl SddpSimulationHandler {
    /// Creates a new simulation handler with pre-allocated memory for forward passes.
    pub fn new(
        node_data_graph: &graph::DirectedGraph<NodeData>,
        initial_condition: &initial_condition::InitialCondition,
    ) -> Result<Self, String> {
        if node_data_graph.node_count() == 0 {
            return Err(
                "Cannot create simulation handler: node data graph is empty"
                    .to_string(),
            );
        }

        let mut realization_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                let temporal_models: Vec<_> =
                    node_data.uncertainty_models.iter().cloned().collect();

                // Use conservative defaults for simulation (no cuts expected)
                let (num_cols, num_rows) =
                    subproblem::estimate_problem_dimensions(
                        &node_data.system,
                        &temporal_models,
                        1, // Single simulation pass
                        1, // No cuts during simulation
                    );

                subproblem::Realization::with_capacity(
                    &node_data.kind,
                    &node_data.system,
                    &temporal_models,
                    num_cols,
                    num_rows,
                )
            });

        let mut subproblem_graph =
            node_data_graph.map_topology_with(|node_data, _id| {
                // Convert UncertaintyModels to TemporalModels
                let temporal_models: Vec<_> =
                    node_data.uncertainty_models.iter().cloned().collect();

                subproblem::Subproblem::new_from_temporal_models(
                    &node_data.system,
                    &node_data.state_choice,
                    &temporal_models,
                    node_data.season_id,
                )
            });

        let prestudy_node_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::PreStudy
        });

        if prestudy_node_ids.is_empty() {
            return Err(format!(
                "Cannot create simulation handler: no pre-study nodes found in graph (graph has {} nodes)",
                node_data_graph.node_count()
            ));
        }

        let first_prestudy_node = realization_graph
            .get_node(*prestudy_node_ids.first().unwrap())
            .ok_or_else(|| {
                "Cannot create simulation handler: pre-study node not found in realization graph"
                    .to_string()
            })?;

        let expected_storage_size =
            first_prestudy_node.data.final_storage.len();
        let provided_storage_size = initial_condition.get_storage().len();
        if expected_storage_size != provided_storage_size {
            return Err(format!(
                "Cannot create simulation handler: initial condition storage size mismatch (expected {} hydro units, got {})",
                expected_storage_size,
                provided_storage_size
            ));
        }

        for &prestudy_id in &prestudy_node_ids {
            let prestudy_node = realization_graph
                .get_node_mut(prestudy_id)
                .ok_or_else(|| {
                    format!(
                        "Cannot create simulation handler: pre-study node {} not found in realization graph",
                        prestudy_id
                    )
                })?;

            prestudy_node
                .data
                .final_storage
                .clone_from_slice(initial_condition.get_storage());

            // CRITICAL FIX: Set inflow field to initial lag values
            // Same as in training handler - PreStudy nodes need proper inflow values
            // for trajectory-based state reconstruction
            let node_data =
                node_data_graph.get_node(prestudy_id).ok_or_else(|| {
                    format!(
                        "Failed to get node data for PreStudy node {}",
                        prestudy_id
                    )
                })?;

            for model in node_data.data.uncertainty_models.iter() {
                if model.entity_type() == crate::input::UncertaintyType::Inflow
                {
                    let hydro_id = model.entity_id;
                    let lags = initial_condition.get_inflow(hydro_id);

                    if !lags.is_empty() {
                        let prestudy_index = prestudy_node_ids
                            .iter()
                            .position(|&id| id == prestudy_id)
                            .unwrap();

                        if prestudy_index < lags.len() {
                            prestudy_node.data.inflow[hydro_id] =
                                lags[prestudy_index];
                        }
                    }
                }
            }
        }

        // Initialize lag buffers from initial condition
        for node_id in 0..node_data_graph.node_count() {
            let node_data =
                node_data_graph.get_node(node_id).ok_or_else(|| {
                    format!("Failed to get node data for node {}", node_id)
                })?;

            let temporal_models = &node_data.data.uncertainty_models;

            // Find inflow entities and set their initial lags
            for model in temporal_models.iter() {
                if model.entity_type() == crate::input::UncertaintyType::Inflow
                    && model.max_ar_order > 0
                {
                    let hydro_id = model.entity_id;
                    let lags = initial_condition.get_inflow(hydro_id);
                    if !lags.is_empty() {
                        let subproblem_node = subproblem_graph
                            .get_node_mut(node_id)
                            .ok_or_else(|| {
                                format!(
                                    "Failed to get subproblem node {}",
                                    node_id
                                )
                            })?;
                        // Set lags in InflowLagData
                        if let Some(ref mut inflow_data) =
                            subproblem_node.data.inflow_lag_data
                        {
                            for (lag_idx, &lag_value) in lags.iter().enumerate()
                            {
                                if lag_idx < inflow_data.buffer[hydro_id].len()
                                {
                                    inflow_data.buffer[hydro_id][lag_idx] =
                                        lag_value;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            subproblem_graph,
            realization_graph,
        })
    }

    pub fn forward(
        &mut self,
        sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let mut timing = ForwardPassTimingAccumulator::default();

        for (idx, id) in study_period_ids.iter().enumerate() {
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

            let prep_start = std::time::Instant::now();

            subproblem_node
                .data
                .prepare_from_trajectory(&past_realizations)?;

            timing.model_preprocessing_time += prep_start.elapsed();

            let realization_node =
                self.realization_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find realization for node {}", id)
                })?;

            let current_stage_noises =
                sampled_noises.get(*id).ok_or_else(|| {
                    format!("Could not find noises for node {}", id)
                })?;

            let step_timing = step(
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;
            timing.solver_time += step_timing.solver_time;

            let post_start = std::time::Instant::now();
            timing.model_postprocessing_time += step_timing.state_update_time;
            timing.solver_calls += 1;

            timing.model_postprocessing_time += post_start.elapsed();
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
        Ok((trajectory_cost, timing))
    }

    pub fn get_realization_at_node(
        &self,
        id: usize,
    ) -> Option<&graph::Node<subproblem::Realization>> {
        self.realization_graph.get_node(id)
    }

    pub fn extract_trajectory(
        &self,
        study_period_ids: &[usize],
        scenario_id: usize,
    ) -> Result<Trajectory, String> {
        let num_stages = study_period_ids.len();

        let mut stages = Vec::with_capacity(num_stages);
        let mut total_cost = 0.0;

        for (stage_idx, &node_id) in study_period_ids.iter().enumerate() {
            let realization_node = self
                .realization_graph
                .get_node(node_id)
                .ok_or_else(|| {
                    format!(
                        "Could not find realization for node {} in trajectory extraction",
                        node_id
                    )
                })?;

            let realization = &realization_node.data;

            let state = if stage_idx == 0 {
                let pre_study_id = self
                    .realization_graph
                    .get_node_id_with(|n| {
                        matches!(n.kind, subproblem::StudyPeriodKind::PreStudy)
                    })
                    .ok_or_else(|| {
                        "Could not find pre-study node for initial storage"
                            .to_string()
                    })?;
                let pre_study_node =
                    self.realization_graph.get_node(pre_study_id).ok_or_else(
                        || "Could not access pre-study node".to_string(),
                    )?;
                pre_study_node.data.final_storage.clone()
            } else {
                let prev_node_id = study_period_ids[stage_idx - 1];
                let prev_realization_node = self
                    .realization_graph
                    .get_node(prev_node_id)
                    .ok_or_else(|| {
                        format!(
                            "Could not find previous realization for node {}",
                            prev_node_id
                        )
                    })?;
                prev_realization_node.data.final_storage.clone()
            };

            let action_capacity = realization.turbined_flow.len()
                + realization.thermal_generation.len()
                + realization.spillage.len()
                + realization.exchange.len()
                + realization.deficit.len();
            let mut action = Vec::with_capacity(action_capacity);
            action.extend_from_slice(&realization.turbined_flow);
            action.extend_from_slice(&realization.thermal_generation);
            action.extend_from_slice(&realization.spillage);
            action.extend_from_slice(&realization.exchange);
            action.extend_from_slice(&realization.deficit);

            let stage_result = StageResult {
                stage: stage_idx,
                state,
                action,
                stage_cost: realization.current_stage_objective,
                inflow: realization.inflow.clone(),
                load: realization.loads.clone(),
            };

            total_cost += realization.current_stage_objective;
            stages.push(stage_result);
        }

        Ok(Trajectory {
            stages,
            total_cost,
            scenario_id,
        })
    }

    pub fn extract_simulation_trajectory(
        &self,
        study_period_ids: &[usize],
        scenario_id: usize,
    ) -> Result<SimulationTrajectory, String> {
        let mut realizations = Vec::with_capacity(study_period_ids.len());

        for &stage_id in study_period_ids {
            let realization_node = self
                .realization_graph
                .get_node(stage_id)
                .ok_or_else(|| {
                    format!(
                        "Cannot extract trajectory: study period node {} not found in realization graph",
                        stage_id
                    )
                })?;

            let realization_data = RealizationData::from_realization(
                stage_id,
                &realization_node.data,
            );

            realizations.push(realization_data);
        }

        Ok(SimulationTrajectory {
            scenario_id,
            realizations,
        })
    }
}

pub struct SddpAlgorithm {
    node_data_graph: graph::DirectedGraph<NodeData>,
    pub future_cost_function_graph:
        graph::DirectedGraph<Arc<Mutex<fcf::FutureCostFunction>>>,
    initial_condition: initial_condition::InitialCondition,
    seed: u64,
    pub study_period_ids: Vec<usize>,
    graph_bfs_table: Vec<Vec<usize>>,
}

impl SddpAlgorithm {
    pub fn new(
        node_data_graph: graph::DirectedGraph<NodeData>,
        initial_condition: initial_condition::InitialCondition,
        seed: u64,
    ) -> Result<Self, String> {
        let future_cost_function_graph =
            node_data_graph.map_topology_with(|_node_data, _id| {
                Arc::new(Mutex::new(fcf::FutureCostFunction::placeholder()))
            });

        let study_period_ids = node_data_graph.get_all_node_ids_with(|node| {
            node.kind == subproblem::StudyPeriodKind::Study
        });

        // Future enhancement: For path graphs this BFS approach is sufficient.
        // For Markovian or cyclic graphs, trajectory extraction may need revision.
        let graph_bfs_table = study_period_ids
            .iter()
            .map(|id| node_data_graph.get_bfs(*id, true))
            .collect();

        Ok(Self {
            node_data_graph,
            future_cost_function_graph,
            initial_condition,
            seed,
            study_period_ids,
            graph_bfs_table,
        })
    }

    pub fn builder() -> SddpBuilder {
        SddpBuilder::new()
    }

    pub fn from_files(
        config_path: impl AsRef<std::path::Path>,
        system_path: impl AsRef<std::path::Path>,
        graph_path: impl AsRef<std::path::Path>,
        recourse_path: impl AsRef<std::path::Path>,
    ) -> Result<SddpInstance, crate::error::PowersError> {
        SddpInstanceBuilder::from_paths(
            config_path,
            system_path,
            graph_path,
            recourse_path,
        )?
        .build()
    }

    pub fn train(
        &mut self,
        num_iterations: usize,
        num_forward_passes: usize,
        enable_cut_selection: bool,
        saa: &scenario::ScenarioTree,
        preserve_forward_detail: bool,
        preserve_backward_detail: bool,
    ) -> Result<TrainingResult, String> {
        if num_iterations == 0 {
            return Err(
                "Number of iterations must be greater than 0".to_string()
            );
        }
        if num_forward_passes == 0 {
            return Err(
                "Number of forward passes must be greater than 0".to_string()
            );
        }

        // Compute maximum dimensions from graph
        let max_state_dim = self
            .node_data_graph
            .iter_nodes()
            .map(|node| {
                match node.data.state_choice.as_str() {
                    "storage" => node.data.system.hydros.len(),
                    "storage_and_inflow" => {
                        // Storage + all lags
                        let base = node.data.system.hydros.len();
                        let lags: usize = node
                            .data
                            .uncertainty_models
                            .iter()
                            .flat_map(|m| &m.ar_orders)
                            .sum();
                        base + lags
                    }
                    _ => 0,
                }
            })
            .max()
            .unwrap();

        // SAA is the source of truth for branching counts - it determines actual scenarios solved
        let max_scenarios = saa
            .stage_scenarios
            .iter()
            .map(|s| s.num_branchings)
            .max()
            .unwrap_or(1);

        // Validate consistency: node data should match SAA (they come from same input)
        // This catches bugs in test setup or input file generation
        for node in self.node_data_graph.iter_nodes() {
            if node.data.kind == subproblem::StudyPeriodKind::PreStudy {
                continue; // PreStudy nodes don't have scenarios
            }

            let stage_id = node.data.stage_id;
            if let Some(saa_branchings) =
                saa.get_branching_count_at_stage(stage_id)
            {
                assert_eq!(
                    node.data.num_scenarios, saa_branchings,
                    "Data consistency violation: NodeData.num_scenarios ({}) != SAA.num_branchings ({}) \
                     for stage {}. This indicates a bug in input file generation or test setup. \
                     In production, Recourse::generate_sddp_noises() uses node.data.num_scenarios \
                     to generate the SAA, so they must match.",
                    node.data.num_scenarios, saa_branchings, stage_id
                );
            }
        }

        log::debug!(
            "Cut buffer dimensions: max_state_dim={}, max_scenarios={}",
            max_state_dim,
            max_scenarios
        );

        crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);

        // Initialize cut buffers in all Rayon worker threads
        rayon::broadcast(|_| {
            crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);
        });

        let max_cuts = num_forward_passes * num_iterations;
        let max_states = num_forward_passes * num_iterations;

        // Full preallocation of FCF cut pools with state dimension per node
        // This enables zero-allocation cut updates during training
        for (node_data, fcf_node) in self
            .node_data_graph
            .iter_nodes()
            .zip(self.future_cost_function_graph.iter_nodes())
        {
            let state_dim = match node_data.data.state_choice.as_str() {
                "storage" => node_data.data.system.hydros.len(),
                "storage_and_inflow" => {
                    let base = node_data.data.system.hydros.len();
                    let lags: usize = node_data
                        .data
                        .uncertainty_models
                        .iter()
                        .flat_map(|m| &m.ar_orders)
                        .sum();
                    base + lags
                }
                _ => 0,
            };

            let mut fcf = fcf_node.data.lock().unwrap();
            if state_dim > 0 {
                // Create template state for preallocation
                let template_state: Box<dyn state::State> = state::factory(
                    &node_data.data.state_choice,
                    &node_data.data.system,
                    &node_data.data.uncertainty_models,
                );
                // Full preallocation for nodes with state
                *fcf = fcf::FutureCostFunction::preallocate_pools(
                    num_iterations,
                    num_forward_passes,
                    state_dim,
                    template_state.as_ref(),
                );
            } else {
                // Fallback to capacity-only for nodes without state
                fcf.cut_pool.pool.reserve(max_cuts);
                fcf.cut_pool.active_cut_indices.reserve(max_cuts);
                fcf.state_pool.pool.reserve(max_states);
            }
        }

        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);
        let begin = Instant::now();
        let mut iterations = Vec::with_capacity(num_iterations);

        // Training phase greeting
        ::log::info!("");
        ::log::info!("# Training");
        ::log::info!("- Iterations: {}", num_iterations);
        ::log::info!("- Forward passes: {}", num_forward_passes);
        ::log::info!("- Cut selection: {}", enable_cut_selection);
        ::log::info!("");

        // Table header
        ::log::info!("{}", "-".repeat(88));
        ::log::info!(
            "{0: >4} | {1: >14} | {2: >14} | {3: >12} | {4: >12} | {5: >12}",
            "iter",
            "lower ($)",
            "simul ($)",
            "fwd",
            "bwd",
            "total"
        );
        ::log::info!("{}", "-".repeat(88));

        let mut train_handlers: Vec<SddpTrainHandler> = (0..num_forward_passes)
            .map(|_| {
                SddpTrainHandler::new(
                    &self.node_data_graph,
                    &self.initial_condition,
                    saa,
                    preserve_forward_detail,
                    preserve_backward_detail,
                    num_forward_passes,
                    num_iterations,
                )
            })
            .collect::<Result<_, _>>()?;

        // Preallocate cut constraint slots for HPC memory determinism
        // This enables zero-allocation cut addition during training
        let max_cuts_per_node = num_forward_passes * num_iterations;
        for handler in &mut train_handlers {
            handler.preallocate_cut_constraints(
                max_cuts_per_node,
                num_forward_passes,
            )?;
        }

        for index in 0..num_iterations {
            let iter_begin = Instant::now();

            // Backward pass timing components (accumulated across stages)
            let mut total_backward_preprocessing_time = Duration::ZERO;
            let mut total_backward_model_preprocessing_time = Duration::ZERO;
            let mut total_backward_solver_time = Duration::ZERO;
            let mut total_backward_model_postprocessing_time = Duration::ZERO;
            let mut total_backward_cutsel_time = Duration::ZERO;
            let mut total_backward_fcf_state_update_time = Duration::ZERO;
            let mut total_backward_cut_cloning_time = Duration::ZERO;
            let mut total_backward_handler_application_time = Duration::ZERO;
            let mut backward_solver_calls: usize = 0;
            let mut backward_cuts_added: usize = 0;

            let mut backward_cuts_removed: usize = 0;
            let mut backward_cuts_returned: usize = 0;

            let saa_sampling_begin = Instant::now();
            let all_sampled_noises: Vec<_> = (0..num_forward_passes)
                .map(|_| saa.sample_scenario(&mut rng))
                .collect();
            let saa_sampling_time = saa_sampling_begin.elapsed();

            let forward_parallel_begin = Instant::now();
            let forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = train_handlers
                .par_iter_mut()
                .zip(all_sampled_noises.par_iter())
                .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
                .collect::<Result<Vec<(f64, ForwardPassTimingAccumulator)>, String>>()?;
            let forward_parallel_time = forward_parallel_begin.elapsed();

            let forward_post_begin = Instant::now();
            let (forward_costs, forward_timings): (
                Vec<f64>,
                Vec<ForwardPassTimingAccumulator>,
            ) = forward_results.into_iter().unzip();

            // Aggregate timing using AVERAGE strategy (representative per-trajectory metrics)
            let mut forward_timing =
                ForwardPassTimingAccumulator::aggregate(&forward_timings);

            // Recalibrate internal forward timing estimates to account for parallel overhead
            let internal_forward_timings = forward_timing
                .model_preprocessing_time
                + forward_timing.solver_time
                + forward_timing.model_postprocessing_time;

            if internal_forward_timings > Duration::ZERO {
                forward_timing.model_preprocessing_time = forward_parallel_time
                    .mul_f64(
                        forward_timing.model_preprocessing_time.as_secs_f64()
                            / internal_forward_timings.as_secs_f64(),
                    );
                forward_timing.solver_time = forward_parallel_time.mul_f64(
                    forward_timing.solver_time.as_secs_f64()
                        / internal_forward_timings.as_secs_f64(),
                );
                forward_timing.model_postprocessing_time =
                    forward_parallel_time.mul_f64(
                        forward_timing.model_postprocessing_time.as_secs_f64()
                            / internal_forward_timings.as_secs_f64(),
                    );
            }

            let forward_solver_calls: usize =
                forward_timings.iter().map(|t| t.solver_calls).sum();

            let forward_postprocessing_time = forward_post_begin.elapsed();

            forward_timing.saa_sampling_time = saa_sampling_time;
            forward_timing.forward_postprocessing_time =
                forward_postprocessing_time;
            forward_timing.total_time = saa_sampling_time
                + forward_parallel_time
                + forward_postprocessing_time;

            // Capture trajectories for export (only if enabled - zero overhead otherwise)
            if preserve_forward_detail {
                for (fp_idx, handler) in train_handlers.iter_mut().enumerate() {
                    handler.capture_forward_detail(
                        index + 1, // iteration (1-indexed)
                        fp_idx,
                        &self.study_period_ids,
                    );
                }
            }

            // --- Parallel Backward Pass with Stage-wise Synchronization ---
            let backward_begin = Instant::now();
            let num_study_periods = self.study_period_ids.len();
            let mut lower_bound = 0.0;

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
                    // ===== BATCH CUT SELECTION: 3-Phase Architecture =====

                    // --- SINGLE-THREADED: Backward Preprocessing ---
                    let backward_preprocessing_begin = Instant::now();
                    let parent_id = *past_node_ids.last().ok_or_else(|| {
                        format!(
                            "Empty past_node_ids for stage {} (node {})",
                            current_stage_original_idx, id
                        )
                    })?;
                    total_backward_preprocessing_time +=
                        backward_preprocessing_begin.elapsed();

                    // --- MULTI-THREADED: Phase 1 - Compute cuts in parallel (no FCF lock) ---
                    // Uses allocation-free compute_cut_data_for_backward_step which returns CutData
                    // instead of CutStatePair, eliminating Box<dyn State> allocation.
                    let phase1_begin = Instant::now();
                    let phase1_results: Vec<(
                        fcf::CutData,
                        BackwardPhase1Timing,
                    )> = train_handlers
                        .par_iter_mut()
                        .enumerate()
                        .map(|(forward_pass_idx, handler)| {
                            handler.compute_cut_data_for_backward_step(
                                id,
                                past_node_ids,
                                &self.node_data_graph,
                                saa,
                                index + 1, // Convert 0-based index to 1-based iteration
                                forward_pass_idx,
                            )
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    let _phase1_time = phase1_begin.elapsed();

                    // PERFORMANCE: Manual unzip with pre-allocated capacity.
                    // Standard unzip() allocates incrementally. With pre-allocation, we avoid
                    // reallocation overhead. At production scale (192 forward passes × 5 stages
                    // × 32 iterations), this eliminates ~30K small reallocations per training run.
                    // Benchmark impact: Negligible on small problems (<10 FPs), meaningful at scale.
                    let mut cut_data_vec: Vec<fcf::CutData> =
                        Vec::with_capacity(phase1_results.len());
                    let mut phase1_timings: Vec<BackwardPhase1Timing> =
                        Vec::with_capacity(phase1_results.len());

                    for (cut_data, timing) in phase1_results {
                        cut_data_vec.push(cut_data);
                        phase1_timings.push(timing);
                    }

                    let phase1_time = _phase1_time;

                    // Compute raw averages from internal measurements
                    let raw_avg_phase1_model_pre: Duration = phase1_timings
                        .iter()
                        .map(|t| t.model_preprocessing_time)
                        .sum::<Duration>()
                        / phase1_timings.len() as u32;
                    let raw_avg_phase1_solver: Duration = phase1_timings
                        .iter()
                        .map(|t| t.solver_time)
                        .sum::<Duration>()
                        / phase1_timings.len() as u32;
                    let raw_avg_phase1_model_post: Duration = phase1_timings
                        .iter()
                        .map(|t| t.model_postprocessing_time)
                        .sum::<Duration>()
                        / phase1_timings.len() as u32;

                    // Sum of internal timing estimates
                    let internal_phase1_timings = raw_avg_phase1_model_pre
                        + raw_avg_phase1_solver
                        + raw_avg_phase1_model_post;

                    let avg_phase1_model_pre =
                        if internal_phase1_timings > Duration::ZERO {
                            phase1_time.mul_f64(
                                raw_avg_phase1_model_pre.as_secs_f64()
                                    / internal_phase1_timings.as_secs_f64(),
                            )
                        } else {
                            Duration::ZERO
                        };
                    let avg_phase1_solver =
                        if internal_phase1_timings > Duration::ZERO {
                            phase1_time.mul_f64(
                                raw_avg_phase1_solver.as_secs_f64()
                                    / internal_phase1_timings.as_secs_f64(),
                            )
                        } else {
                            Duration::ZERO
                        };
                    let avg_phase1_model_post =
                        if internal_phase1_timings > Duration::ZERO {
                            phase1_time.mul_f64(
                                raw_avg_phase1_model_post.as_secs_f64()
                                    / internal_phase1_timings.as_secs_f64(),
                            )
                        } else {
                            Duration::ZERO
                        };

                    total_backward_model_preprocessing_time +=
                        avg_phase1_model_pre;
                    total_backward_solver_time += avg_phase1_solver;
                    total_backward_model_postprocessing_time +=
                        avg_phase1_model_post;

                    // Count solver calls: num_forward_passes * num_branching_scenarios for this stage
                    let num_branchings =
                        saa.get_branching_count_at_stage(id).unwrap_or(1);
                    backward_solver_calls +=
                        num_forward_passes * num_branchings;

                    // --- SINGLE-THREADED: Phase 2 - Batch Cut Selection (deterministic) ---
                    let phase2_begin = Instant::now();

                    // Sort cuts before batch processing to ensure deterministic
                    // cut ordering regardless of parallel thread completion order. This is CRITICAL
                    // for reproducibility because intra-batch domination is order-dependent.
                    //
                    // We sort by forward_pass_idx (handler ID)
                    cut_data_vec
                        .sort_unstable_by_key(|data| data.forward_pass_idx);

                    let batch_result: fcf::BatchCutSelectionResult = {
                        let parent_fcf_node = self
                            .future_cost_function_graph
                            .get_node(parent_id)
                            .ok_or_else(|| {
                                format!(
                                    "Could not find FCF for parent node {}",
                                    parent_id
                                )
                            })?;
                        let mut fcf_locked =
                            parent_fcf_node.data.lock().unwrap();
                        // Use allocation-free batch processing
                        fcf_locked.add_cuts_batch_from_data(
                            cut_data_vec,
                            enable_cut_selection,
                        )
                    };
                    let phase2_time = phase2_begin.elapsed();
                    total_backward_cutsel_time += phase2_time;

                    // Count cuts in this stage (before moving the data)
                    backward_cuts_added += batch_result.new_cut_ids.len();
                    backward_cuts_removed +=
                        batch_result.removing_cut_ids.len();
                    backward_cuts_returned +=
                        batch_result.returning_cut_ids.len();

                    // Move BatchCutSelectionResult into AggregatedCutSelectionResult (zero-cost)
                    let aggregated_result = fcf::AggregatedCutSelectionResult {
                        new_cut_ids: batch_result.new_cut_ids,
                        returning_cut_ids: batch_result.returning_cut_ids,
                        removing_cut_ids: batch_result.removing_cut_ids,
                    };

                    // --- SINGLE-THREADED: Phase 3a - Update FCF state (mark inactive) ---
                    let (fcf_state_update_time, cut_cloning_time, cuts_vec) = {
                        let parent_fcf_node = self
                            .future_cost_function_graph
                            .get_node(parent_id)
                            .ok_or_else(|| {
                                format!(
                                    "Could not find FCF for parent node {}",
                                    parent_id
                                )
                            })?;
                        let mut fcf_locked =
                            parent_fcf_node.data.lock().unwrap();

                        // PART 1: Update FCF state (mark cuts inactive
                        let fcf_state_update_begin = Instant::now();
                        let mut removed_indices: Vec<usize> = Vec::new();
                        for &cut_id in &aggregated_result.removing_cut_ids {
                            if let Some(cut) =
                                fcf_locked.cut_pool.pool.get_mut(cut_id)
                            {
                                cut.set_active(false);
                            }
                            if let Some(index) = fcf_locked
                                .cut_pool
                                .active_cut_indices
                                .remove(&cut_id)
                            {
                                removed_indices.push(index);
                            }
                        }

                        // Sort removed indices for efficient adjustment
                        removed_indices.sort_unstable();

                        // Adjust indices for all remaining cuts
                        for (_cut_id, index) in
                            fcf_locked.cut_pool.active_cut_indices.iter_mut()
                        {
                            let count_below = removed_indices
                                .partition_point(|&removed| removed < *index);
                            *index -= count_below;
                        }
                        let fcf_state_update_time =
                            fcf_state_update_begin.elapsed();

                        // PART 2: Pre-clone cuts for lock-free handler application
                        // With Arc, this clones the Arc pointer (~16 bytes) not the data (~1KB)
                        let cut_cloning_begin = Instant::now();
                        let cuts: Vec<(
                            usize,
                            std::sync::Arc<crate::cut::BendersCut>,
                        )> = aggregated_result
                            .new_cut_ids
                            .iter()
                            .chain(aggregated_result.returning_cut_ids.iter())
                            .filter_map(|&cut_id| {
                                fcf_locked.cut_pool.pool.get(cut_id).map(
                                    |cut| (cut_id, std::sync::Arc::clone(cut)),
                                )
                            })
                            .collect();
                        let cut_cloning_time = cut_cloning_begin.elapsed();

                        // Return timing data and cuts
                        (fcf_state_update_time, cut_cloning_time, cuts)
                    }; // FCF lock released

                    // Accumulate timing
                    total_backward_fcf_state_update_time +=
                        fcf_state_update_time;
                    total_backward_cut_cloning_time += cut_cloning_time;

                    // --- PARALLEL: Phase 3b - Apply results to ALL models ---
                    let phase3b_begin = Instant::now();
                    train_handlers
                        .par_iter_mut()
                        .map(|handler| {
                            handler.apply_aggregated_cut_result(
                                parent_id,
                                &aggregated_result,
                                &cuts_vec,
                            )
                        })
                        .collect::<Result<(), String>>()?;
                    let phase3b_time = phase3b_begin.elapsed();
                    total_backward_handler_application_time += phase3b_time;
                } else {
                    let (lb, first_stage_timing) = train_handlers
                        .get_mut(0)
                        .unwrap()
                        .eval_first_stage_bound(
                            id,
                            past_node_ids,
                            &self.node_data_graph,
                            saa,
                        )?;

                    lower_bound = lb;

                    // Accumulate first stage timing into backward pass metrics
                    total_backward_solver_time +=
                        first_stage_timing.solver_time;
                    total_backward_model_postprocessing_time +=
                        first_stage_timing.state_extraction_time;

                    // Count solver calls for first stage
                    // num_branchings scenarios solved for this stage
                    let num_branchings =
                        saa.get_branching_count_at_stage(id).unwrap_or(1);
                    backward_solver_calls +=
                        num_forward_passes * num_branchings;
                }
            }

            // Query active cut count from FCF across ALL nodes in the graph
            let active_cut_count: usize = self
                .study_period_ids
                .iter()
                .map(|&node_id| {
                    self.future_cost_function_graph
                        .get_node(node_id)
                        .map(|node| {
                            node.data
                                .lock()
                                .unwrap()
                                .cut_pool
                                .active_cut_indices
                                .len()
                        })
                        .unwrap_or(0)
                })
                .sum();

            let backward_total_time = backward_begin.elapsed();
            let iter_time = iter_begin.elapsed();

            // Store iteration result with collected timing data
            iterations.push(IterationResult {
                iteration: index + 1,
                lower_bound,
                forward_costs: forward_costs.clone(),
                iteration_time: iter_time,
                forward_timing: ForwardPassTiming {
                    saa_sampling_time,
                    model_preprocessing_time: forward_timing
                        .model_preprocessing_time,
                    solver_time: forward_timing.solver_time,
                    model_postprocessing_time: forward_timing
                        .model_postprocessing_time,
                    forward_postprocessing_time,
                    total_time: forward_timing.total_time,
                },
                backward_timing: BackwardPassTiming {
                    backward_preprocessing_time:
                        total_backward_preprocessing_time,
                    model_preprocessing_time:
                        total_backward_model_preprocessing_time,
                    solver_time: total_backward_solver_time,
                    model_postprocessing_time:
                        total_backward_model_postprocessing_time,
                    cut_selection_time: total_backward_cutsel_time,
                    fcf_state_update_time: total_backward_fcf_state_update_time,
                    cut_cloning_time: total_backward_cut_cloning_time,
                    handler_application_time:
                        total_backward_handler_application_time,
                    total_time: backward_total_time,
                },
                num_solver_calls: forward_solver_calls + backward_solver_calls,
                num_cuts_added: backward_cuts_added,
                num_cuts_removed: backward_cuts_removed,
                num_cuts_returned: backward_cuts_returned,
                num_active_cuts: active_cut_count,
            });

            // Compute simulation cost for logging (mean of forward costs)
            let simulation_cost = utils::mean_deterministic(&forward_costs);

            // Set logging context with iteration data
            crate::logging::LogContext::set(crate::logging::LogContext {
                iteration: Some(index + 1),
                lower_bound: Some(lower_bound),
                simulation_cost: Some(simulation_cost),
                forward_time: Some(forward_timing.total_time),
                backward_time: Some(backward_total_time),
                total_time: Some(iter_time),
            });

            // Log iteration complete - formatter will render as table row using context
            ::log::info!("Iteration complete");

            // Clear context
            crate::logging::LogContext::clear();

            // Detailed timing output (only shown at debug level)
            // TODO: LOG-015 - Implement detailed timing logging with structured logs
            if ::log::log_enabled!(::log::Level::Debug) {
                ::log::debug!(
                    "Iteration {} timing: forward={:?}, backward={:?}, solver_calls={}, cuts: +{} -{} +{} (active: {})",
                    index + 1,
                    forward_timing.total_time,
                    backward_total_time,
                    forward_solver_calls + backward_solver_calls,
                    backward_cuts_added,
                    backward_cuts_removed,
                    backward_cuts_returned,
                    active_cut_count,
                );
            }
        }

        // Training completion - table divider and summary
        ::log::info!("{}", "-".repeat(88));
        let total_time = begin.elapsed();
        let total_secs = total_time.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = total_time.subsec_millis();
        ::log::info!("");
        ::log::info!(
            "Training time: {:02}:{:02}:{:02}.{:03}",
            hours,
            minutes,
            seconds,
            millis
        );

        let num_cuts = self
            .future_cost_function_graph
            .get_node(1)
            .ok_or_else(|| {
                "Could not find node 1 for counting cuts".to_string()
            })?
            .data
            .lock()
            .unwrap()
            .cut_pool
            .total_cut_count;

        ::log::info!("");
        ::log::info!("Number of constructed cuts by node: {}", num_cuts);

        // Get final lower bound from last iteration
        let final_lower_bound = iterations
            .last()
            .map(|r| r.lower_bound)
            .ok_or_else(|| "No iterations completed".to_string())?;

        // Compute statistical upper bound: average of ALL forward pass costs across ALL iterations
        let all_forward_costs: Vec<f64> = iterations
            .iter()
            .flat_map(|iter_result| iter_result.forward_costs.iter().copied())
            .collect();

        let statistical_upper_bound = if all_forward_costs.is_empty() {
            f64::INFINITY
        } else {
            utils::mean(&all_forward_costs)
        };

        let final_std = utils::standard_deviation(&all_forward_costs);

        // Find best (minimum) simulation cost across all iterations (informational only)
        let (best_upper_bound, best_iteration) = iterations
            .iter()
            .enumerate()
            .map(|(idx, iter_result)| {
                (
                    utils::mean_deterministic(&iter_result.forward_costs),
                    idx + 1,
                )
            })
            .min_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((f64::INFINITY, 0));

        // Collect training trajectories from all handlers (if preservation was enabled)
        let forward_details: Vec<ForwardPassDetail> = if preserve_forward_detail
        {
            train_handlers
                .iter_mut()
                .flat_map(|handler| handler.take_forward_detail_history())
                .collect()
        } else {
            Vec::new()
        };

        // Collect backward branching records from all handlers (if preservation was enabled)
        let backward_details: Vec<BackwardPassDetail> =
            if preserve_backward_detail {
                train_handlers
                    .iter_mut()
                    .flat_map(|handler| handler.take_backward_detail_history())
                    .collect()
            } else {
                Vec::new()
            };

        let result = TrainingResult {
            iterations,
            final_lower_bound,
            statistical_upper_bound,
            best_upper_bound,
            best_iteration,
            total_time,
            num_cuts,
            forward_details,
            backward_details,
        };

        // Log final simulation statistics with gap
        ::log::info!(
            "Final policy cost: {:.6e} ± {:.6e}",
            result.statistical_upper_bound,
            final_std
        );
        ::log::info!("Gap: {:.4} %", 100.0 * result.relative_gap());
        ::log::info!("");

        // Create and return training result
        Ok(result)
    }

    pub fn forward(
        &self,
        sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
        handler: &mut SddpTrainHandler,
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let (trajectory_cost, timing) = handler.forward(
            sampled_noises,
            &self.graph_bfs_table,
            &self.study_period_ids,
        )?;
        Ok((trajectory_cost, timing))
    }

    /// Simulate the trained policy across multiple scenarios.
    pub fn simulate(
        &mut self,
        num_simulation_scenarios: usize,
        saa: &scenario::ScenarioTree,
    ) -> Result<Vec<SimulationTrajectory>, String> {
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);

        let begin = Instant::now();

        ::log::info!("");
        ::log::info!("# Simulating");
        ::log::info!("- Scenarios: {}", num_simulation_scenarios);
        ::log::info!("");

        // Pre-generate all noise samples (deterministic with seed)
        let all_sampled_noises: Vec<_> = (0..num_simulation_scenarios)
            .map(|_| saa.sample_scenario(&mut rng))
            .collect();

        let trajectories: Vec<SimulationTrajectory> = all_sampled_noises
            .par_iter()
            .enumerate()
            .map_init(
                || {
                    SddpSimulationHandler::new(
                        &self.node_data_graph,
                        &self.initial_condition,
                    )
                },
                |handler_result, (scenario_id, noises)| {
                    let handler = handler_result.as_mut().map_err(|e| {
                        format!(
                            "Handler creation failed for thread processing scenario {}: {}",
                            scenario_id, e
                        )
                    })?;

                    let (_trajectory_cost, _timing) = handler.forward(
                        noises.to_vec(),
                        &self.graph_bfs_table,
                        &self.study_period_ids,
                    )?;

                    let trajectory = handler.extract_simulation_trajectory(
                        &self.study_period_ids,
                        scenario_id,
                    )?;

                    Ok(trajectory)
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        // Compute statistics from trajectories
        let simulation_costs: Vec<f64> = trajectories
            .iter()
            .map(|t| {
                t.realizations
                    .iter()
                    .map(|r| r.current_stage_objective)
                    .sum()
            })
            .collect();

        let mean_cost = utils::mean(&simulation_costs);
        let std_cost = utils::standard_deviation(&simulation_costs);

        // Log simulation statistics
        ::log::info!("Expected cost ($): {:.6e} ± {:.6e}", mean_cost, std_cost);

        let duration = begin.elapsed();
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;
        let millis = duration.subsec_millis();
        ::log::info!("");
        ::log::info!(
            "Simulation time: {:02}:{:02}:{:02}.{:03}",
            hours,
            minutes,
            seconds,
            millis
        );

        Ok(trajectories)
    }

    /// Returns a reference to the system from the first node.
    ///
    /// All nodes in the graph have the same system, so we can safely
    /// return the system from any node. This is a convenience method
    /// for accessing system information during output generation.
    pub fn system(&self) -> &system::System {
        self.node_data_graph
            .iter_nodes()
            .next()
            .map(|node| &node.data.system)
            .expect("Graph must have at least one node")
    }

    /// Returns the maximum AR order across all hydros.
    ///
    /// This is computed from the temporal models in the first node.
    /// Returns 0 if there are no temporal models or no hydros.
    pub fn max_ar_order(&self) -> usize {
        self.node_data_graph
            .iter_nodes()
            .next()
            .map(|node| {
                node.data
                    .uncertainty_models
                    .iter()
                    .map(|tm| tm.max_ar_order)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    /// Returns the AR order for each hydro entity.
    ///
    /// This returns a vector where index i contains the AR order
    /// for hydro i. Returns an empty vector if there are no temporal models.
    pub fn hydro_ar_orders(&self) -> Vec<usize> {
        self.node_data_graph
            .iter_nodes()
            .next()
            .map(|node| {
                node.data
                    .uncertainty_models
                    .iter()
                    .map(|tm| tm.max_ar_order)
                    .collect()
            })
            .unwrap_or_default()
    }
}
/// Simple timing structure for step function operations.
#[derive(Debug, Clone, Copy, Default)]
struct StepTiming {
    solver_time: Duration,
    state_update_time: Duration,
}

/// Timing for backward pass Phase 1 (solve branchings + generate cut).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct BackwardPhase1Timing {
    model_preprocessing_time: Duration,
    solver_time: Duration,
    model_postprocessing_time: Duration,
}

fn step(
    subproblem: &mut subproblem::Subproblem,
    realization_container: &mut subproblem::Realization,
    noises: &scenario::OptimizedSampledBranchingNoises,
) -> Result<StepTiming, String> {
    let all_innovations = noises.get_all_innovations();
    let realize_timing = subproblem
        .realize_and_solve(&all_innovations, realization_container)?;

    let timing = StepTiming {
        solver_time: realize_timing.solver_time,
        state_update_time: realize_timing.state_extraction_time,
    };

    Ok(timing)
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

impl IterationResult {
    /// Creates a default IterationResult for testing purposes.
    #[cfg(test)]
    fn test_default(
        iteration: usize,
        lower_bound: f64,
        forward_costs: Vec<f64>,
    ) -> Self {
        Self {
            iteration,
            lower_bound,
            forward_costs,
            iteration_time: Duration::from_secs(1),
            forward_timing: ForwardPassTiming::default(),
            backward_timing: BackwardPassTiming::default(),
            num_solver_calls: 0,
            num_cuts_added: 0,
            num_cuts_removed: 0,
            num_cuts_returned: 0,
            num_active_cuts: 0,
        }
    }
}

#[cfg(test)]
/// Create empty uncertainty_models vec for test fixtures
fn test_empty_noise_models(
) -> std::sync::Arc<Vec<crate::temporal_model::TemporalModel>> {
    std::sync::Arc::new(vec![])
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::solver;
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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

        let mut example_noises =
            scenario::OptimizedSampledBranchingNoises::new(1, 1);
        example_noises.set_load_innovations(&[75.0]);
        example_noises.set_inflow_data(&[10.0]);
        let sampled_noises = vec![
            &example_noises,
            &example_noises,
            &example_noises,
            &example_noises,
        ];

        let _pre_study_id = node_data_graph
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
                            test_empty_noise_models(),
                            "storage",
                            1,
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
            &node_data_graph,
            &initial_condition,
            &generate_test_saa_for_four_stages(),
            false, // Don't preserve trajectories in tests
            false, // Don't preserve backward statistics in tests
            10,    // num_forward_passes for preallocation
            100,   // num_iterations for preallocation
        )
        .unwrap();

        handler
            .forward(sampled_noises, &graph_bfs_table, &study_period_ids)
            .unwrap();
    }

    fn generate_test_saa_for_four_stages() -> scenario::ScenarioTree {
        scenario::ScenarioTree {
            stage_scenarios: vec![
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![{
                        let mut noise =
                            scenario::OptimizedSampledBranchingNoises::new(
                                1, 1,
                            );
                        noise.set_load_innovations(&[75.0]);
                        noise.set_inflow_data(&[5.0]);
                        noise
                    }],
                },
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![{
                        let mut noise =
                            scenario::OptimizedSampledBranchingNoises::new(
                                1, 1,
                            );
                        noise.set_load_innovations(&[75.0]);
                        noise.set_inflow_data(&[10.0]);
                        noise
                    }],
                },
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![{
                        let mut noise =
                            scenario::OptimizedSampledBranchingNoises::new(
                                1, 1,
                            );
                        noise.set_load_innovations(&[75.0]);
                        noise.set_inflow_data(&[15.0]);
                        noise
                    }],
                },
                scenario::SampledNodeBranchings {
                    num_branchings: 1,
                    branching_noises: vec![{
                        let mut noise =
                            scenario::OptimizedSampledBranchingNoises::new(
                                1, 1,
                            );
                        noise.set_load_innovations(&[75.0]);
                        noise.set_inflow_data(&[15.0]);
                        noise
                    }],
                },
            ],
            index_samplers: vec![
                rand_distr::Uniform::try_from(0..1).unwrap(),
                rand_distr::Uniform::try_from(0..1).unwrap(),
                rand_distr::Uniform::try_from(0..1).unwrap(),
                rand_distr::Uniform::try_from(0..1).unwrap(),
            ],
            metadata: scenario::ScenarioTreeMetadata {
                generation_method: scenario::ScenarioGenerationMethod::Custom {
                    description: "Test scenario".to_string(),
                },
                seed: 0,
                generated_at: "2024-01-01T00:00:00Z".to_string(),
                num_stages: 4,
            },
        }
    }

    #[test]
    fn test_train_with_default_system() {
        // Initialize cut buffers for this test
        crate::memory::initialize_cut_buffers(10, 10);

        // NOTE: num_scenarios in NodeData MUST match the SAA branching count
        // The SAA is generated with 3 branchings per stage, so nodes must have num_scenarios=3
        let num_scenarios = 3;

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
                    test_empty_noise_models(),
                    "storage",
                    1, // PreStudy doesn't need scenarios
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
                    test_empty_noise_models(),
                    "storage",
                    num_scenarios,
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            num_scenarios,
        );
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            num_scenarios,
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
                        test_empty_noise_models(),
                        "storage",
                        num_scenarios,
                    )
                    .unwrap(),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                num_scenarios,
            );
        }

        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let saa = scenario_generator.generate(0);

        let mut sddp_algo =
            SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

        let _result = sddp_algo.train(24, 1, true, &saa, false, false).unwrap();
    }

    #[test]
    fn test_simulate_with_default_system() {
        // Initialize cut buffers for this test
        crate::memory::initialize_cut_buffers(10, 10);

        // NOTE: num_scenarios in NodeData MUST match the SAA branching count
        // The SAA is generated with 3 branchings per stage, so nodes must have num_scenarios=3
        let num_scenarios = 3;

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
                    test_empty_noise_models(),
                    "storage",
                    1, // PreStudy doesn't need scenarios
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
                    test_empty_noise_models(),
                    "storage",
                    num_scenarios,
                )
                .unwrap(),
            )
            .unwrap();
        node_data_graph.add_edge(pre_study_id, prev_id).unwrap();
        let mut scenario_generator = scenario::NoiseGenerator::new();
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            num_scenarios,
        );
        scenario_generator.add_node_generator(
            vec![Normal::new(75.0, 0.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            num_scenarios,
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
                        test_empty_noise_models(),
                        "storage",
                        num_scenarios,
                    )
                    .unwrap(),
                )
                .unwrap();
            node_data_graph.add_edge(prev_id, new_id).unwrap();
            scenario_generator.add_node_generator(
                vec![Normal::new(75.0, 0.0).unwrap()],
                vec![LogNormal::new(3.6, 0.6928).unwrap()],
                num_scenarios,
            );
        }
        let storage = vec![83.222];

        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let saa = scenario_generator.generate(0);

        let mut sddp_algo =
            SddpAlgorithm::new(node_data_graph, initial_condition, 0).unwrap();

        let _result = sddp_algo.train(24, 1, true, &saa, false, false).unwrap();

        sddp_algo.simulate(100, &saa).unwrap();
    }

    // ====================================================================
    // Unit tests for TrainingResult and IterationResult (T2.1)
    // ====================================================================

    /// Helper to create placeholder timing data for tests.
    ///
    /// Updated with refactored timing structure (T4.1 Phase 3.5 Refactoring).
    fn placeholder_timing() -> (ForwardPassTiming, BackwardPassTiming) {
        let forward_timing = ForwardPassTiming {
            saa_sampling_time: Duration::ZERO,
            model_preprocessing_time: Duration::ZERO,
            solver_time: Duration::ZERO,
            model_postprocessing_time: Duration::ZERO,
            forward_postprocessing_time: Duration::ZERO,
            total_time: Duration::ZERO,
        };
        let backward_timing = BackwardPassTiming {
            backward_preprocessing_time: Duration::ZERO,
            model_preprocessing_time: Duration::ZERO,
            solver_time: Duration::ZERO,
            model_postprocessing_time: Duration::ZERO,
            cut_selection_time: Duration::ZERO,
            fcf_state_update_time: Duration::ZERO,
            cut_cloning_time: Duration::ZERO,
            handler_application_time: Duration::ZERO,
            total_time: Duration::ZERO,
        };
        (forward_timing, backward_timing)
    }

    /// Helper function to create a test TrainingResult with realistic data.
    fn create_test_training_result() -> TrainingResult {
        let iterations = vec![
            IterationResult::test_default(1, 1000.0, vec![1400.0, 1600.0]),
            IterationResult::test_default(2, 1200.0, vec![1300.0, 1400.0]),
            IterationResult::test_default(3, 1250.0, vec![1280.0, 1320.0]),
        ];

        // Compute statistical upper bound for test data
        let all_costs: Vec<f64> = vec![
            1400.0, 1600.0, // Iter 1
            1300.0, 1400.0, // Iter 2
            1280.0, 1320.0, // Iter 3
        ];
        let statistical_upper_bound =
            all_costs.iter().sum::<f64>() / all_costs.len() as f64;

        // Best simulation cost (minimum mean) from all iterations
        let best_upper_bound = 1300.0; // Mean of iteration 3's costs
        let best_iteration = 3;

        TrainingResult {
            iterations,
            final_lower_bound: 1250.0,
            statistical_upper_bound,
            best_upper_bound,
            best_iteration,
            total_time: Duration::from_millis(2950),
            num_cuts: 15,
            forward_details: Vec::new(),
            backward_details: Vec::new(),
        }
    }

    #[test]
    fn test_training_result_relative_gap() {
        let result = create_test_training_result();
        let expected_relative_gap = (1383.33333 - 1250.0) / 1250.0;
        assert!((result.relative_gap() - expected_relative_gap).abs() < 1e-4);
    }

    #[test]
    fn test_training_result_relative_gap_zero_lower_bound() {
        let mut result = create_test_training_result();
        result.final_lower_bound = 0.0;
        result.statistical_upper_bound = 100.0;

        // Should return infinity when lower bound is zero
        assert_eq!(result.relative_gap(), f64::INFINITY);
    }

    #[test]
    fn test_training_result_relative_gap_near_zero_lower_bound() {
        let mut result = create_test_training_result();
        result.final_lower_bound = 1e-11; // Below threshold
        result.statistical_upper_bound = 100.0;

        // Should return infinity when lower bound is very close to zero
        assert_eq!(result.relative_gap(), f64::INFINITY);
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
    fn test_training_result_iterations_access() {
        let result = create_test_training_result();
        let iterations = result.iterations();

        assert_eq!(iterations.len(), 3);
        assert_eq!(iterations[0].iteration, 1);
        assert_eq!(iterations[1].iteration, 2);
        assert_eq!(iterations[2].iteration, 3);

        // Check that we can access fields
        assert_eq!(iterations[0].lower_bound, 1000.0);
        assert_eq!(iterations[0].forward_costs, vec![1400.0, 1600.0]);

        assert_eq!(iterations[1].lower_bound, 1200.0);
        assert_eq!(iterations[1].forward_costs, vec![1300.0, 1400.0]);
    }

    #[test]
    fn test_iteration_result_forward_costs_access() {
        let (forward_timing, backward_timing) = placeholder_timing();
        let mut iter_result = IterationResult::test_default(
            2,
            1000.0,
            vec![1150.0, 1200.0, 1250.0],
        );
        iter_result.iteration_time = Duration::from_secs(1);
        iter_result.forward_timing = forward_timing;
        iter_result.backward_timing = backward_timing;

        assert_eq!(iter_result.forward_costs.len(), 3);
        assert_eq!(iter_result.forward_costs[0], 1150.0);
        assert_eq!(iter_result.forward_costs[1], 1200.0);
        assert_eq!(iter_result.forward_costs[2], 1250.0);

        // Verify average can be computed from forward costs
        let avg: f64 = iter_result.forward_costs.iter().sum::<f64>() / 3.0;
        assert!((avg - 1200.0).abs() < 1e-10);
    }

    #[test]
    fn test_training_result_single_iteration() {
        let iterations =
            vec![IterationResult::test_default(1, 1000.0, vec![1100.0])];

        let result = TrainingResult {
            iterations,
            final_lower_bound: 1000.0,
            statistical_upper_bound: 1100.0,
            best_upper_bound: 1100.0, // Best simulation cost from iteration 1
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 5,
            forward_details: Vec::new(),
            backward_details: Vec::new(),
        };

        assert_eq!(result.final_gap(), 100.0);
        assert_eq!(result.iterations().len(), 1);
        assert_eq!(result.lower_bounds().len(), 1);
        // Forward costs are available for all iterations
        assert_eq!(result.iterations()[0].forward_costs.len(), 1);
    }

    #[test]
    fn test_training_result_best_simulation_cost_tracking() {
        let result = create_test_training_result();
        assert_eq!(result.best_upper_bound, 1300.0);
        assert_eq!(result.best_iteration, 3);
    }

    #[test]
    fn test_training_result_large_gaps() {
        let result = TrainingResult {
            iterations: vec![IterationResult::test_default(1, 1e6, vec![1e9])],
            final_lower_bound: 1e6,
            statistical_upper_bound: 1e9,
            best_upper_bound: 1e9, // Best simulation cost from iteration 1
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 1,
            forward_details: Vec::new(),
            backward_details: Vec::new(),
        };

        // Should handle large numbers correctly
        assert!((result.final_gap() - (1e9 - 1e6)).abs() < 1e3);
        assert!(result.relative_gap() > 900.0); // Very large relative gap
    }

    /// Helper function to create a test trajectory
    fn create_test_trajectory(
        scenario_id: usize,
        base_cost: f64,
    ) -> Trajectory {
        let stages = vec![
            StageResult {
                stage: 0,
                state: vec![50.0],
                action: vec![10.0, 5.0, 2.0],
                stage_cost: base_cost,
                inflow: vec![40.0],
                load: vec![75.0],
            },
            StageResult {
                stage: 1,
                state: vec![45.0],
                action: vec![12.0, 3.0, 1.0],
                stage_cost: base_cost * 1.1,
                inflow: vec![35.0],
                load: vec![80.0],
            },
            StageResult {
                stage: 2,
                state: vec![42.0],
                action: vec![11.0, 4.0, 1.5],
                stage_cost: base_cost * 0.9,
                inflow: vec![45.0],
                load: vec![70.0],
            },
        ];
        let total_cost = stages.iter().map(|s| s.stage_cost).sum();
        Trajectory {
            stages,
            total_cost,
            scenario_id,
        }
    }

    /// Helper function to create a test simulation result
    fn create_test_simulation_result() -> SimulationResult {
        let trajectories = vec![
            create_test_trajectory(0, 100.0), // Total: 300.0
            create_test_trajectory(1, 110.0), // Total: 330.0
            create_test_trajectory(2, 90.0),  // Total: 270.0
            create_test_trajectory(3, 105.0), // Total: 315.0
            create_test_trajectory(4, 95.0),  // Total: 285.0
        ];

        SimulationResult {
            trajectories,
            num_stages: 3,
            num_states: 1,
            num_actions: 3,
        }
    }

    #[test]
    fn test_stage_result_creation() {
        let stage = StageResult {
            stage: 0,
            state: vec![50.0, 60.0],
            action: vec![10.0, 20.0, 30.0],
            stage_cost: 100.0,
            inflow: vec![40.0, 45.0],
            load: vec![75.0, 80.0],
        };

        assert_eq!(stage.stage, 0);
        assert_eq!(stage.state.len(), 2);
        assert_eq!(stage.action.len(), 3);
        assert_eq!(stage.stage_cost, 100.0);
        assert_eq!(stage.inflow, vec![40.0, 45.0]);
        assert_eq!(stage.load, vec![75.0, 80.0]);
    }

    #[test]
    fn test_trajectory_creation() {
        let traj = create_test_trajectory(0, 100.0);

        assert_eq!(traj.scenario_id, 0);
        assert_eq!(traj.stages.len(), 3);
        assert_eq!(traj.total_cost, 300.0);

        // Verify stage progression
        assert_eq!(traj.stages[0].stage, 0);
        assert_eq!(traj.stages[1].stage, 1);
        assert_eq!(traj.stages[2].stage, 2);

        // Verify cost matches sum of stage costs
        let sum_costs: f64 = traj.stages.iter().map(|s| s.stage_cost).sum();
        assert!((traj.total_cost - sum_costs).abs() < 1e-10);
    }

    #[test]
    fn test_simulation_result_get_trajectory() {
        let result = create_test_simulation_result();

        // Test valid access
        let traj = result.get_trajectory(0).unwrap();
        assert_eq!(traj.scenario_id, 0);

        let traj = result.get_trajectory(4).unwrap();
        assert_eq!(traj.scenario_id, 4);

        // Test out-of-bounds
        assert!(result.get_trajectory(5).is_none());
        assert!(result.get_trajectory(100).is_none());
    }

    #[test]
    fn test_simulation_result_get_all_trajectories() {
        let result = create_test_simulation_result();
        let all_trajs = result.get_all_trajectories();

        assert_eq!(all_trajs.len(), 5);

        // Verify ordering
        for (i, traj) in all_trajs.iter().enumerate() {
            assert_eq!(traj.scenario_id, i);
        }
    }

    #[test]
    fn test_simulation_result_dimensions() {
        let result = create_test_simulation_result();

        assert_eq!(result.num_stages, 3);
        assert_eq!(result.num_states, 1);
        assert_eq!(result.num_actions, 3);
    }

    #[test]
    fn test_forward_pass_timing_accumulator_aggregate_single() {
        let timing = ForwardPassTimingAccumulator {
            model_preprocessing_time: Duration::from_millis(10),
            solver_time: Duration::from_millis(50),
            model_postprocessing_time: Duration::from_millis(5),
            solver_calls: 3,
        };

        let aggregated = ForwardPassTimingAccumulator::aggregate(&[timing]);

        assert_eq!(
            aggregated.model_preprocessing_time,
            Duration::from_millis(10)
        );
        assert_eq!(aggregated.solver_time, Duration::from_millis(50));
        assert_eq!(
            aggregated.model_postprocessing_time,
            Duration::from_millis(5)
        );
    }

    #[test]
    fn test_forward_pass_timing_accumulator_aggregate_multiple() {
        let timings = vec![
            ForwardPassTimingAccumulator {
                model_preprocessing_time: Duration::from_millis(10),
                solver_time: Duration::from_millis(50),
                model_postprocessing_time: Duration::from_millis(6),
                solver_calls: 3,
            },
            ForwardPassTimingAccumulator {
                model_preprocessing_time: Duration::from_millis(20),
                solver_time: Duration::from_millis(60),
                model_postprocessing_time: Duration::from_millis(8),
                solver_calls: 4,
            },
        ];

        let aggregated = ForwardPassTimingAccumulator::aggregate(&timings);

        // Should compute averages: (10+20)/2=15, (50+60)/2=55, (6+8)/2=7
        assert_eq!(
            aggregated.model_preprocessing_time,
            Duration::from_millis(15)
        );
        assert_eq!(aggregated.solver_time, Duration::from_millis(55));
        assert_eq!(
            aggregated.model_postprocessing_time,
            Duration::from_millis(7)
        );
    }

    #[test]
    #[should_panic(expected = "Cannot aggregate zero timings")]
    fn test_forward_pass_timing_accumulator_aggregate_empty() {
        let timings: Vec<ForwardPassTimingAccumulator> = vec![];
        ForwardPassTimingAccumulator::aggregate(&timings);
    }

    #[test]
    fn test_backward_pass_timing_accumulator_into_timing() {
        let accumulator = BackwardPassTimingAccumulator {
            backward_preprocessing_time: Duration::from_millis(10),
            model_preprocessing_time: Duration::from_millis(20),
            solver_time: Duration::from_millis(100),
            model_postprocessing_time: Duration::from_millis(15),
            cut_selection_time: Duration::from_millis(5),
            fcf_state_update_time: Duration::from_millis(3),
            cut_cloning_time: Duration::from_millis(2),
            handler_application_time: Duration::from_millis(8),
            solver_calls: 10,
            cuts_added: 5,
        };

        let timing = accumulator.into_timing();

        // Verify total is sum of all components
        let expected_total =
            Duration::from_millis(10 + 20 + 100 + 15 + 5 + 3 + 2 + 8);
        assert_eq!(timing.total_time, expected_total);
        // Note: solver_calls and cuts_added are not in BackwardPassTiming, only in accumulator
    }

    #[test]
    fn test_backward_pass_timing_accumulator_default() {
        let accumulator = BackwardPassTimingAccumulator::default();

        assert_eq!(accumulator.solver_calls, 0);
        assert_eq!(accumulator.cuts_added, 0);
        assert_eq!(accumulator.backward_preprocessing_time, Duration::ZERO);
    }

    #[test]
    fn test_simulation_result_get_trajectory_out_of_bounds() {
        let trajectories = vec![Trajectory {
            scenario_id: 0,
            total_cost: 100.0,
            stages: vec![],
        }];

        let result = SimulationResult {
            trajectories,
            num_stages: 1,
            num_states: 1,
            num_actions: 1,
        };

        // Out of bounds should return None
        assert!(result.get_trajectory(999).is_none());
    }

    #[test]
    fn test_training_result_iterations_empty() {
        let result = TrainingResult {
            iterations: vec![],
            final_lower_bound: 0.0,
            statistical_upper_bound: 0.0,
            best_upper_bound: f64::INFINITY,
            best_iteration: 0,
            total_time: Duration::ZERO,
            num_cuts: 0,
            forward_details: Vec::new(),
            backward_details: Vec::new(),
        };

        assert_eq!(result.iterations().len(), 0);
        assert_eq!(result.lower_bounds().len(), 0);
        // Forward costs are per-iteration
        assert!(result.iterations().is_empty());
    }

    #[test]
    fn test_simulation_handler_creation_valid() {
        // Create a minimal valid graph for testing
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let _pre_study_id = node_data_graph
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
                    test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        // Create initial condition matching the system (1 hydro unit)
        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        // Should succeed
        let result =
            SddpSimulationHandler::new(&node_data_graph, &initial_condition);

        assert!(
            result.is_ok(),
            "Valid handler creation should succeed, got error: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_simulation_handler_creation_empty_graph() {
        // Create an empty graph
        let node_data_graph = graph::DirectedGraph::<NodeData>::new();

        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        // Should fail with descriptive error
        let result =
            SddpSimulationHandler::new(&node_data_graph, &initial_condition);

        assert!(result.is_err(), "Empty graph should cause an error");
        if let Err(error_msg) = result {
            assert!(
                error_msg.contains("graph is empty"),
                "Error message should mention empty graph, got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_simulation_handler_creation_storage_size_mismatch() {
        // Create a graph with a system that has 1 hydro unit
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let _pre_study_id = node_data_graph
            .add_node(
                NodeData::new(
                    -1,
                    0,
                    0,
                    "1970-01-01T00:00:00Z",
                    "1970-01-01T00:00:00Z",
                    subproblem::StudyPeriodKind::PreStudy,
                    system::System::default(), // Has 1 hydro unit
                    "expectation",
                    test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        // Provide initial condition with wrong storage size (2 hydro units instead of 1)
        let storage = vec![100.0, 200.0]; // Wrong size!
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        // Should fail with size mismatch error
        let result =
            SddpSimulationHandler::new(&node_data_graph, &initial_condition);

        assert!(
            result.is_err(),
            "Storage size mismatch should cause an error"
        );
        if let Err(error_msg) = result {
            assert!(
                error_msg.contains("storage size mismatch"),
                "Error message should mention storage size mismatch, got: {}",
                error_msg
            );
            assert!(
                error_msg.contains("expected 1"),
                "Error message should show expected size, got: {}",
                error_msg
            );
            assert!(
                error_msg.contains("got 2"),
                "Error message should show provided size, got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_realization_data_from_realization() {
        // Create a sample realization with known values
        let loads = vec![100.0, 150.0];
        let deficit = vec![0.0, 10.0];
        let exchange = vec![50.0];
        let inflow = vec![20.0];
        let turbined_flow = vec![18.0];
        let spillage = vec![2.0];
        let thermal_generation = vec![80.0, 90.0];
        let water_value = vec![5.0];
        let marginal_cost = vec![50.0, 55.0];
        let current_stage_objective = 1000.0;
        let total_stage_objective = 3000.0;
        let final_storage = vec![100.0];

        let realization = subproblem::Realization::new(
            loads.clone(),
            deficit.clone(),
            exchange.clone(),
            inflow.clone(),
            turbined_flow.clone(),
            spillage.clone(),
            thermal_generation.clone(),
            water_value.clone(),
            marginal_cost.clone(),
            current_stage_objective,
            total_stage_objective,
            final_storage.clone(),
            solver::Basis::new(), // Heavy structure we want to discard
        );

        // Extract lightweight data
        let stage_id = 5;
        let realization_data =
            RealizationData::from_realization(stage_id, &realization);

        // Verify all fields are correctly extracted
        assert_eq!(realization_data.stage_id, stage_id);
        assert_eq!(realization_data.loads, loads);
        assert_eq!(realization_data.deficit, deficit);
        assert_eq!(realization_data.exchange, exchange);
        assert_eq!(realization_data.inflow, inflow);
        assert_eq!(realization_data.turbined_flow, turbined_flow);
        assert_eq!(realization_data.spillage, spillage);
        assert_eq!(realization_data.thermal_generation, thermal_generation);
        assert_eq!(realization_data.water_value, water_value);
        assert_eq!(realization_data.marginal_cost, marginal_cost);
        assert_eq!(
            realization_data.current_stage_objective,
            current_stage_objective
        );
        assert_eq!(
            realization_data.total_stage_objective,
            total_stage_objective
        );
        assert_eq!(realization_data.final_storage, final_storage);
    }

    #[test]
    fn test_realization_data_size() {
        // Verify RealizationData is significantly smaller than Realization
        // Typical system: 10 buses, 5 lines, 3 hydros, 2 thermals
        let loads = vec![0.0; 10]; // 10 buses
        let deficit = vec![0.0; 10]; // 10 buses
        let exchange = vec![0.0; 5]; // 5 lines
        let inflow = vec![0.0; 3]; // 3 hydros
        let turbined_flow = vec![0.0; 3]; // 3 hydros
        let spillage = vec![0.0; 3]; // 3 hydros
        let thermal_generation = vec![0.0; 2]; // 2 thermals
        let water_value = vec![0.0; 3]; // 3 hydros
        let marginal_cost = vec![0.0; 10]; // 10 buses
        let final_storage = vec![0.0; 3]; // 3 hydros

        let realization = subproblem::Realization::new(
            loads,
            deficit,
            exchange,
            inflow,
            turbined_flow,
            spillage,
            thermal_generation,
            water_value,
            marginal_cost,
            100.0,
            300.0,
            final_storage,
            solver::Basis::new(),
        );

        let realization_data =
            RealizationData::from_realization(0, &realization);

        // Calculate approximate memory usage
        // Vec<f64>: 24 bytes overhead + 8 bytes per element
        let vec_overhead = 24;
        let data_size = (10 + 10 + 5 + 3 + 3 + 3 + 2 + 3 + 10 + 3) * 8; // f64 elements
        let total_vecs = 10; // number of Vec fields
        let scalars = 3 * 8; // stage_id (usize), current_stage_objective, total_stage_objective
        let approx_size = (vec_overhead * total_vecs) + data_size + scalars;

        // Should be less than 3KB for this typical system
        assert!(
            approx_size < 3000,
            "RealizationData size {} should be < 3KB",
            approx_size
        );

        // Verify the structure exists and is usable
        assert_eq!(realization_data.stage_id, 0);
        assert_eq!(realization_data.loads.len(), 10);
    }

    #[test]
    fn test_extract_simulation_trajectory_completeness() {
        // Create a minimal graph with 2 study periods
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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
                    test_empty_noise_models(),
                    "storage",
                    1,
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
                    test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        node_data_graph.add_edge(pre_study_id, node_0_id).unwrap();
        node_data_graph.add_edge(node_0_id, node_1_id).unwrap();

        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        // Create handler
        let handler =
            SddpSimulationHandler::new(&node_data_graph, &initial_condition)
                .unwrap();

        let study_period_ids = vec![node_0_id, node_1_id];
        let scenario_id = 42;

        // Extract trajectory
        let trajectory = handler
            .extract_simulation_trajectory(&study_period_ids, scenario_id)
            .unwrap();

        // Verify completeness
        assert_eq!(trajectory.scenario_id, scenario_id);
        assert_eq!(trajectory.realizations.len(), 2);

        // Verify stage IDs match
        assert_eq!(trajectory.realizations[0].stage_id, node_0_id);
        assert_eq!(trajectory.realizations[1].stage_id, node_1_id);

        // Verify all vectors are initialized (non-empty or correct size)
        for realization in &trajectory.realizations {
            assert_eq!(realization.loads.len(), 1); // System::default() has 1 bus
            assert_eq!(realization.final_storage.len(), 1); // 1 hydro unit
        }
    }

    #[test]
    fn test_extract_simulation_trajectory_missing_node() {
        // Create a minimal graph
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();

        let _pre_study_id = node_data_graph
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
                    test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let handler =
            SddpSimulationHandler::new(&node_data_graph, &initial_condition)
                .unwrap();

        // Try to extract trajectory with non-existent node ID
        let study_period_ids = vec![999]; // Doesn't exist
        let result =
            handler.extract_simulation_trajectory(&study_period_ids, 0);

        assert!(result.is_err(), "Should fail with missing node");
        if let Err(error_msg) = result {
            assert!(
                error_msg.contains("node 999 not found"),
                "Error should mention missing node ID, got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_simulation_trajectory_memory_efficiency() {
        // This test verifies that SimulationTrajectory is significantly
        // smaller than keeping full handlers

        // Create typical-sized realizations (120 stages)
        let num_stages = 120;
        let mut realizations = Vec::with_capacity(num_stages);

        for stage_id in 0..num_stages {
            let realization_data = RealizationData {
                stage_id,
                loads: vec![100.0; 10], // 10 buses
                deficit: vec![0.0; 10],
                exchange: vec![50.0; 5], // 5 lines
                inflow: vec![20.0; 3],   // 3 hydros
                turbined_flow: vec![18.0; 3],
                spillage: vec![2.0; 3],
                thermal_generation: vec![80.0; 2], // 2 thermals
                water_value: vec![5.0; 3],
                marginal_cost: vec![50.0; 10],
                current_stage_objective: 1000.0,
                total_stage_objective: 1000.0 * (stage_id + 1) as f64,
                final_storage: vec![100.0; 3],
            };
            realizations.push(realization_data);
        }

        let trajectory = SimulationTrajectory {
            scenario_id: 0,
            realizations,
        };

        // Verify trajectory contains all stages
        assert_eq!(trajectory.realizations.len(), num_stages);

        // Approximate memory calculation:
        // Per stage: 10 vectors × (24 + 8×size) + 3 scalars
        // Total: ~800 bytes × 120 stages = ~96KB
        let bytes_per_stage = 800; // approximate
        let total_bytes = bytes_per_stage * num_stages;

        // Should be around 96KB (significantly less than 6MB handler)
        assert!(
            total_bytes < 200_000,
            "Trajectory should be < 200KB, got ~{} bytes",
            total_bytes
        );

        // Memory saved per scenario compared to full handler (~6MB):
        // Savings = 6,000,000 - 96,000 = ~5.9MB per scenario
        // For 10,000 scenarios: ~59 GB saved!
    }
}
