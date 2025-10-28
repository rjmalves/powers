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
//! 1. Only risk-neutral policy evaluation is supported (no risk-aversion)
//! 2. An exact cut selection strategy (inspired in SDDP.jl) is implemented
//! 3. Only the "single-cut" (average cut) variant of the algorithm is supported.
//!
//! The only external dependencies are:
//!
//! 1. Random number generation and distribution sampling from rand* crates
//! 2. Low-level C-bindings from the highs-sys crate
//! 3. JSON and CSV serializers from the serde, serde_json and csv crates

pub mod builder;
pub mod instance;

pub use builder::{SddpBuilder, SddpInstanceBuilder};
pub use instance::SddpInstance;

use crate::fcf;
use crate::graph;
use crate::initial_condition;
use crate::input::UncertaintyType;
use crate::log;
use crate::risk_measure;
use crate::scenario;
use crate::stochastic_process;
use crate::subproblem;
use crate::system;
use crate::unified_noise_spec::{TemporalModelSpec, UnifiedNoiseSpec};
use crate::utils;
use chrono::prelude::*;
use rand::prelude::*;

use rand_xoshiro::Xoshiro256Plus;
use rayon::prelude::*;
use std::f64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct ForwardPassTiming {
    pub saa_sampling_time: Duration,
    pub model_preprocessing_time: Duration,
    pub solver_time: Duration,
    pub model_postprocessing_time: Duration,
    pub forward_postprocessing_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug, Clone, Copy)]
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
///
/// Each iteration performs:
/// 1. Forward pass: Samples scenarios and computes trajectories (stored in `forward_costs`)
/// 2. Backward pass: Adds cuts to improve policy (increases `lower_bound`)
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
    pub final_upper_bound: f64,
    pub statistical_upper_bound: f64,
    pub best_upper_bound: f64,
    pub best_iteration: usize,
    pub total_time: Duration,
    pub num_cuts: usize,
    pub termination_reason: TerminationReason,
    pub final_simulation_performed: bool,
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
    /// # Note
    ///
    /// Filters out `None` values (from iteration 1, which has no upper bound).
    /// The returned vector will have length `num_iterations - 1`.
    ///
    /// Access iteration results.
    ///
    /// Provides read-only access to the complete iteration history.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// for iter in result.iterations() {
    ///     let simul_cost: f64 = iter.forward_costs.iter().sum::<f64>() / iter.forward_costs.len() as f64;
    ///     println!("Iteration {}: LB={:.2}, Simul={:.2}", iter.iteration, iter.lower_bound, simul_cost);
    /// }
    /// ```
    #[inline]
    pub fn iterations(&self) -> &[IterationResult] {
        &self.iterations
    }
}

/// Result from a single stage in a simulation trajectory.
///
/// Contains all relevant information for one stage of a simulated scenario:
/// state variables, control actions, costs, and realized uncertainties.
///
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage number (0-indexed, where 0 is first study period).
    pub stage: usize,

    /// State variables at the beginning of this stage (before decisions).
    pub state: Vec<f64>,

    /// Control actions taken at this stage.
    ///
    /// For hydrothermal problems, this includes:
    /// - Hydro generation (turbined flow)
    /// - Thermal generation
    /// - Spillage
    /// - Line flows (exchange)
    /// - Deficit
    pub action: Vec<f64>,

    /// Objective cost for this stage only (not cumulative).
    pub stage_cost: f64,

    /// Realized inflow values for this stage.
    pub inflow: Vec<f64>,

    /// Realized load values for this stage.
    pub load: Vec<f64>,
}

/// Complete trajectory for a single simulated scenario.
///
/// A trajectory represents one complete path through the scenario tree,
/// containing the sequence of states, actions, and costs from initial
/// condition to final stage.
///
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Stage-by-stage results for this trajectory.
    pub stages: Vec<StageResult>,

    /// Total cost across all stages (sum of stage_cost values).
    pub total_cost: f64,

    /// Scenario identifier (0-indexed).
    pub scenario_id: usize,
}

/// Confidence interval for a statistic.
///
/// Computed using normal approximation (CLT) for mean estimation.
///
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    /// Lower bound of the confidence interval.
    pub lower: f64,

    /// Upper bound of the confidence interval.
    pub upper: f64,

    /// Confidence level (e.g., 0.95 for 95% confidence).
    pub confidence_level: f64,
}

/// Statistical summary of simulation results.
///
/// Contains all relevant statistics computed from trajectory costs:
/// mean, standard deviation, percentiles, and confidence intervals.
///
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    /// Mean (expected) cost across all trajectories.
    pub mean: f64,

    /// Standard deviation of costs across trajectories.
    pub std: f64,

    /// 5th percentile of cost distribution.
    pub p5: f64,

    /// 25th percentile (first quartile) of cost distribution.
    pub p25: f64,

    /// 50th percentile (median) of cost distribution.
    pub p50: f64,

    /// 75th percentile (third quartile) of cost distribution.
    pub p75: f64,

    /// 95th percentile of cost distribution.
    pub p95: f64,

    /// 95% confidence interval for the mean cost.
    pub ci_95: ConfidenceInterval,

    /// Number of trajectories used to compute these statistics.
    pub num_trajectories: usize,
}

/// Complete result from simulation analysis.
///
/// Contains all trajectory data and computed statistics for a set of
/// simulated scenarios under a trained SDDP policy.
///
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// All simulated trajectories.
    pub trajectories: Vec<Trajectory>,

    /// Statistical summary of trajectory costs.
    pub statistics: Statistics,

    /// Number of stages per trajectory (excludes pre-study period).
    pub num_stages: usize,

    /// Number of state variables in the system.
    pub num_states: usize,

    /// Number of action variables per stage.
    pub num_actions: usize,
}

impl SimulationResult {
    /// Get a specific trajectory by scenario index.
    ///
    /// # Arguments
    ///
    /// * `scenario_idx` - Zero-indexed scenario identifier
    ///
    /// # Returns
    ///
    /// Reference to the trajectory, or `None` if index out of bounds.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(traj) = result.get_trajectory(42) {
    ///     println!("Scenario 42 cost: {:.2}", traj.total_cost);
    /// }
    /// ```
    #[inline]
    pub fn get_trajectory(&self, scenario_idx: usize) -> Option<&Trajectory> {
        self.trajectories.get(scenario_idx)
    }

    /// Get all trajectories.
    ///
    /// Returns a slice to all trajectories for iteration or analysis.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let high_cost_scenarios: Vec<_> = result.get_all_trajectories()
    ///     .iter()
    ///     .filter(|t| t.total_cost > result.statistics.p95)
    ///     .collect();
    /// ```
    #[inline]
    pub fn get_all_trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Get the statistics summary.
    ///
    /// Returns a copy of the statistics (all fields are `Copy`).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = result.get_statistics();
    /// assert!(stats.mean >= stats.p5 && stats.mean <= stats.p95);
    /// ```
    #[inline]
    pub fn get_statistics(&self) -> Statistics {
        self.statistics
    }
}

/// Compute percentile value from a sorted vector.
///
/// Uses linear interpolation between values when percentile falls between indices.
///
/// # Arguments
///
/// * `sorted_values` - MUST be sorted in ascending order
/// * `percentile` - Value between 0.0 and 1.0
///
/// # Performance
///
/// O(1) - assumes input is already sorted
///
fn compute_percentile(sorted_values: &[f64], percentile: f64) -> f64 {
    assert!(
        !sorted_values.is_empty(),
        "Cannot compute percentile of empty vector"
    );
    assert!(
        (0.0..=1.0).contains(&percentile),
        "Percentile must be in [0, 1]"
    );

    let n = sorted_values.len();
    let index = percentile * (n - 1) as f64;
    let lower_idx = index.floor() as usize;
    let upper_idx = index.ceil() as usize;

    if lower_idx == upper_idx {
        sorted_values[lower_idx]
    } else {
        // Linear interpolation
        let weight = index - lower_idx as f64;
        sorted_values[lower_idx] * (1.0 - weight)
            + sorted_values[upper_idx] * weight
    }
}

/// Compute statistics from a collection of trajectories.
///
/// Computes mean, standard deviation, percentiles (5, 25, 50, 75, 95), and
/// 95% confidence interval for the mean cost across all trajectories.
///
/// # Arguments
///
/// * `trajectories` - Collection of simulation trajectories
///
/// # Returns
///
/// `Statistics` struct with all computed values
///
/// # Performance
///
/// - Time: O(n log n) due to sorting for percentiles
/// - Space: O(n) temporary vector for costs
/// - For n=1000 trajectories: ~0.1-1 ms on modern CPU
/// - Uses stable sort for determinism (negligible overhead vs unstable)
///
fn compute_statistics(trajectories: &[Trajectory]) -> Statistics {
    let n = trajectories.len();
    assert!(n > 0, "Cannot compute statistics for zero trajectories");

    // Extract costs into separate vector for sorting
    // This avoids sorting full trajectories (much cheaper)
    let mut costs: Vec<f64> =
        trajectories.iter().map(|t| t.total_cost).collect();

    // Compute mean and std using existing utils (tested and optimized)
    let mean = utils::mean(&costs);
    let std = utils::standard_deviation(&costs);

    // Use stable sort to ensure deterministic ordering when
    // costs are equal (common with similar scenarios).
    costs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute percentiles from sorted vector (O(1) each)
    let p5 = compute_percentile(&costs, 0.05);
    let p25 = compute_percentile(&costs, 0.25);
    let p50 = compute_percentile(&costs, 0.50);
    let p75 = compute_percentile(&costs, 0.75);
    let p95 = compute_percentile(&costs, 0.95);

    // Compute 95% confidence interval using normal approximation
    // CI = mean ± z * (std / √n), where z=1.96 for 95% confidence
    let standard_error = std / (n as f64).sqrt();
    let margin = 1.96 * standard_error;
    let ci_95 = ConfidenceInterval {
        lower: mean - margin,
        upper: mean + margin,
        confidence_level: 0.95,
    };

    Statistics {
        mean,
        std,
        p5,
        p25,
        p50,
        p75,
        p95,
        ci_95,
        num_trajectories: n,
    }
}

/// Node data for SDDP algorithm.
///
/// Each node represents a decision point in the scenario tree.
/// **Multi-Process Architecture**: Each hydro plant has its own stochastic process,
/// built from `noise_models` in `recourse.json` filtered by `entity_id`.
/// This enables heterogeneous hydrology (e.g., different PAR orders per hydro,
/// or mixing PAR and independent processes).
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
    /// Inflow stochastic processes - one per hydro plant.
    /// Length must equal `system.hydros.len()`.
    /// Process for hydro i is at index i, built from `noise_models` with `entity_id == i`.
    pub inflow_stochastic_processes:
        Vec<Box<dyn stochastic_process::StochasticProcess>>,
    /// Unified noise specifications for all uncertainty sources in this node.
    /// Used to access AR coefficients during constraint generation.
    pub unified_specs: Vec<UnifiedNoiseSpec>,
    /// Transformation cache for observation ↔ residual conversions (PAR models only)
    /// - Some(Arc<TransformCache>) if any entity has PAR model
    /// - None if all entities use Independent models
    pub transform_cache:
        Option<std::sync::Arc<crate::space_transform::TransformCache>>,
    pub state_choice: String,
    pub num_scenarios: usize,
}

/// Build a stochastic process from a unified noise specification
///
/// Maps `UnifiedNoiseSpec` temporal model to the appropriate `StochasticProcess` implementation:
/// - `Independent` → `NaiveProcess`
/// - `PeriodicAutoregressive` → `PARProcess` with seasonal parameters
///
/// # Arguments
///
/// * `spec` - Unified noise specification (internal representation)
///
/// # Returns
///
/// * `Ok(Box<dyn StochasticProcess>)` - Successfully created process
/// * `Err(String)` - Validation or construction error
fn build_process_from_unified_spec(
    spec: &UnifiedNoiseSpec,
) -> Result<Box<dyn stochastic_process::StochasticProcess>, String> {
    match &spec.temporal_model {
        TemporalModelSpec::Independent => {
            // Independent noise → Naive process
            Ok(stochastic_process::factory("naive"))
        }
        TemporalModelSpec::PeriodicAutoregressive {
            num_seasons,
            seasonal_ar_params,
        } => {
            // PAR model → Build PARProcess from seasonal parameters

            // Extract AR orders and coefficients for all seasons
            let mut ar_orders = Vec::with_capacity(*num_seasons);
            let mut ar_coefficients = Vec::with_capacity(*num_seasons);
            let mut seasonal_means = Vec::with_capacity(*num_seasons);
            let mut seasonal_stds = Vec::with_capacity(*num_seasons);

            for season in 0..*num_seasons {
                // Get AR parameters for this season
                let ar_params =
                    seasonal_ar_params.get(&season).ok_or_else(|| {
                        format!("Missing AR parameters for season {}", season)
                    })?;

                ar_orders.push(ar_params.ar_order);
                ar_coefficients.push(ar_params.ar_coefficients.clone());

                // Get seasonal noise parameters (mean, std_dev)
                let noise_params =
                    spec.seasonal_params.get(&season).ok_or_else(|| {
                        format!(
                            "Missing noise parameters for season {}",
                            season
                        )
                    })?;

                seasonal_means.push(noise_params.mean);
                seasonal_stds.push(noise_params.std_dev);
            }

            let params = crate::seasonal_params::SeasonalParams::new(
                *num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            )
            .map_err(|e| format!("Invalid PAR parameters: {}", e))?;

            let par_process = stochastic_process::PARProcess::new(params)
                .map_err(|e| format!("Failed to create PAR process: {}", e))?;

            Ok(Box::new(par_process))
        }
    }
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
        unified_specs: &[UnifiedNoiseSpec],
        state_str: &str,
        num_scenarios: usize,
    ) -> Result<Self, String> {
        let load_stochastic_process =
            stochastic_process::factory(load_stochastic_process_str);

        // Build per-hydro inflow processes from unified_specs
        let inflow_stochastic_processes: Vec<
            Box<dyn stochastic_process::StochasticProcess>,
        > = system
            .hydros
            .iter()
            .map(|hydro| {
                // Find unified spec for this hydro
                // For PAR models: One spec covers all seasons
                // For Independent models: Spec has seasonal_params for this season
                let hydro_spec = unified_specs.iter().find(|spec| {
                    spec.uncertainty_type == UncertaintyType::Inflow
                        && spec.entity_id == hydro.id
                });

                match hydro_spec {
                    Some(spec) => {
                        // Verify this spec covers the current season
                        if spec.seasonal_params.contains_key(&season_id) {
                            build_process_from_unified_spec(spec)
                        } else {
                            // Spec exists but doesn't cover this season → default to naive
                            Ok(stochastic_process::factory("naive"))
                        }
                    }
                    None => {
                        // No spec → default to naive
                        Ok(stochastic_process::factory("naive"))
                    }
                }
            })
            .collect::<Result<Vec<_>, String>>()?;

        // Validation: process count must match hydro count
        if inflow_stochastic_processes.len() != system.hydros.len() {
            return Err(format!(
                "Process count mismatch: {} hydros, {} processes",
                system.hydros.len(),
                inflow_stochastic_processes.len()
            ));
        }

        // Build transformation cache if any PAR models present
        // PERFORMANCE: Cache is built once per node, then shared (Arc) across subproblems
        let has_par_models = unified_specs.iter().any(|spec| {
            matches!(
                spec.temporal_model,
                TemporalModelSpec::PeriodicAutoregressive { .. }
            )
        });

        let transform_cache = if has_par_models {
            // Determine number of seasons from system metadata or unified specs
            let num_seasons = unified_specs
                .iter()
                .flat_map(|spec| spec.seasonal_params.keys())
                .max()
                .map(|max_season| max_season + 1)
                .unwrap_or(1);

            Some(std::sync::Arc::new(
                crate::space_transform::TransformCache::new(
                    unified_specs,
                    system.hydros.len(),
                    num_seasons,
                ),
            ))
        } else {
            None
        };

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
            inflow_stochastic_processes,
            unified_specs: unified_specs.to_vec(),
            transform_cache,
            state_choice: state_str.to_string(),
            num_scenarios,
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
                    &node_data.inflow_stochastic_processes,
                    &node_data.unified_specs,
                    node_data.season_id,
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
                    saa.get_branching_count_at_stage(node_data.stage_id)
                        .unwrap_or_else(|| panic!(
                            "Missing branching count for stage {} (node {})",
                            node_data.stage_id, id
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
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let mut timing = ForwardPassTimingAccumulator::default();

        for (idx, id) in study_period_ids.iter().enumerate() {
            // Model preparation timing
            let prep_start = std::time::Instant::now();

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
            timing.model_preprocessing_time += prep_start.elapsed();

            // Step includes solver + state extraction
            let step_timing = step(
                data_node,
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;
            timing.solver_time += step_timing.solver_time;

            // Model postprocessing: state transition to next stage
            let post_start = std::time::Instant::now();
            timing.model_postprocessing_time += step_timing.state_update_time;
            timing.solver_calls += 1;

            subproblem_node
                .data
                .update_with_current_realization(&realization_node.data);
            timing.model_postprocessing_time += post_start.elapsed();
        }

        // Final cost aggregation (model postprocessing)
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

    /// Compute cut for backward pass without adding to FCF (for batch processing)
    ///
    /// # Phase 1 of batch cut selection
    /// This computes the cut based on branching scenarios but doesn't lock
    /// or modify the shared FCF. Returns the CutStatePair and precise timing for later batch processing.
    ///
    pub(crate) fn compute_cut_for_backward_step(
        &mut self,
        id: usize,
        past_node_ids: &[usize],
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::SAA,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> Result<(fcf::CutStatePair, BackwardPhase1Timing), String> {
        let mut timing = BackwardPhase1Timing::default();

        let model_preprocessing_start = std::time::Instant::now();

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

        timing.model_preprocessing_time = model_preprocessing_start.elapsed();

        // Solver phase: Solve all branching subproblems
        let branchings_timing = solve_all_branchings(
            &mut self.subproblem_graph,
            &mut self.branching_graph,
            id,
            num_branchings,
            &node_forward_trajectory,
            node_data_graph,
            saa,
        )?;

        timing.solver_time = branchings_timing.solver_time;

        // Model postprocessing: Extract solutions, dual values, and compute cut
        let model_postprocessing_start = std::time::Instant::now();

        // State extraction from branchings is part of postprocessing
        // (already included in branchings_timing.state_extraction_time)
        let branching_node_data = &self
            .branching_graph
            .get_node(id)
            .ok_or_else(|| {
                format!("Could not find branching realizations for node {}", id)
            })?
            .data;

        let child_data_node =
            node_data_graph.get_node(id).ok_or_else(|| {
                format!("Could not find node data for node {}", id)
            })?;
        let child_subproblem_node =
            self.subproblem_graph.get_node(id).ok_or_else(|| {
                format!("Could not find subproblem for node {}", id)
            })?;

        // Cut generation (extract duals, compute coefficients)
        let cut_state_pair = child_subproblem_node.data.compute_new_cut(
            &node_forward_trajectory,
            branching_node_data,
            child_data_node.data.risk_measure.as_ref(),
            iteration,
            forward_pass_idx,
        );

        timing.model_postprocessing_time = model_postprocessing_start.elapsed()
            + branchings_timing.state_extraction_time;

        Ok((cut_state_pair, timing))
    }

    /// Apply aggregated cut results without FCF locking
    pub fn apply_aggregated_cut_result(
        &mut self,
        parent_id: usize,
        aggregated_result: &fcf::AggregatedCutSelectionResult,
        active_cut_indices_before: &std::collections::BTreeMap<usize, usize>,
        cuts_to_add: &[(usize, crate::cut::BendersCut)],
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
                active_cut_indices_before,
                cuts_to_add,
            )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_step_at_node(
        &mut self,
        id: usize,
        past_node_ids: &[usize],
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::SAA,
        future_cost_function_graph: &graph::DirectedGraph<
            Arc<Mutex<fcf::FutureCostFunction>>,
        >,
        iteration: usize,
        forward_pass_idx: usize,
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
            iteration,
            forward_pass_idx,
        )?;

        Ok(())
    }

    /// Evaluate first stage bound
    pub(crate) fn eval_first_stage_bound(
        &mut self,
        id: usize,
        past_node_ids: &[usize],
        node_data_graph: &graph::DirectedGraph<NodeData>,
        saa: &scenario::SAA,
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

        // solve_all_branchings returns timing - we must capture and return it
        let branchings_timing = solve_all_branchings(
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

fn solve_all_branchings(
    subproblem_graph: &mut graph::DirectedGraph<subproblem::Subproblem>,
    branching_graph: &mut graph::DirectedGraph<Vec<subproblem::Realization>>,
    node_id: usize,
    num_branchings: usize,
    node_forward_trajectory: &Vec<&subproblem::Realization>,
    node_data_graph: &graph::DirectedGraph<NodeData>,
    saa: &scenario::SAA,
) -> Result<BranchingsTiming, String> {
    let mut timing = BranchingsTiming::default();

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

        let step_timing = step(
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

        timing.solver_time += step_timing.solver_time;
        timing.state_extraction_time += step_timing.state_update_time;
    }
    Ok(timing)
}

#[allow(clippy::too_many_arguments)]
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
    iteration: usize,
    forward_pass_idx: usize,
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
        iteration,
        forward_pass_idx,
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

/// Lightweight data structure containing only output values from a simulation stage.
///
/// This structure is designed for memory-efficient storage of simulation results.
/// Unlike `Realization`, it excludes heavy components (solver basis, kind enum) that
/// are only needed during computation, not for output generation.
///
/// # Memory Efficiency
///
/// For a typical system with:
/// - 10 buses, 5 lines, 3 hydros, 2 thermals
/// - Each Vec<f64>: ~8 bytes per element + 24 bytes overhead
/// - Total per stage: ~800 bytes
///
/// Compare with full `Realization` (includes basis): ~6KB per stage
/// **Memory savings: ~87% per stage**
///
/// For 10,000 scenarios × 120 stages:
/// - Full handlers: 7.2 GB
/// - Trajectories only: 960 MB
///
/// **Total reduction: ~87%**
///
/// # Design Notes
///
/// - All fields are public for direct CSV export access
/// - stage_id included for ordering and verification
/// - No `basis` field (this is the key memory saving)
/// - No `kind` field (all stages are StudyPeriod in output)
/// - Derives Clone for flexibility, but intended to be moved/consumed
///
#[derive(Debug, Clone)]
pub struct RealizationData {
    /// Stage identifier (node ID in the graph)
    pub stage_id: usize,

    /// Load values at each bus [MW]
    pub loads: Vec<f64>,

    /// Deficit (unmet demand) at each bus [MW]
    pub deficit: Vec<f64>,

    /// Power flow on each transmission line [MW]
    pub exchange: Vec<f64>,

    /// Inflow to each hydro reservoir [m³/s or hm³]
    pub inflow: Vec<f64>,

    /// Water turbined at each hydro plant [m³/s or hm³]
    pub turbined_flow: Vec<f64>,

    /// Water spilled at each hydro plant [m³/s or hm³]
    pub spillage: Vec<f64>,

    /// Generation from each thermal plant [MW]
    pub thermal_generation: Vec<f64>,

    /// Marginal water value at each hydro reservoir [$/hm³]
    pub water_value: Vec<f64>,

    /// Marginal cost of electricity at each bus [$/MWh]
    pub marginal_cost: Vec<f64>,

    /// Objective function value for this stage only [currency units]
    pub current_stage_objective: f64,

    /// Cumulative objective from start to this stage [currency units]
    pub total_stage_objective: f64,

    /// Final storage level at each reservoir [hm³ or %]
    pub final_storage: Vec<f64>,
}

impl RealizationData {
    /// Extract realization data from a full Realization structure.
    ///
    /// Clones all Vec<f64> fields while discarding the heavy basis structure.
    /// This is intentionally a clone operation to keep the handler in a valid state.
    ///
    /// # Performance
    ///
    /// - Complexity: O(n) where n = sum of all vector lengths
    /// - Typical overhead: <0.1ms per stage
    /// - Memory: Allocates ~800 bytes per stage for typical systems
    ///
    /// The clone cost is negligible compared to solver time (~10-100ms per stage).
    ///
    /// # Arguments
    ///
    /// * `stage_id` - Stage identifier for tracking and ordering
    /// * `realization` - Reference to the source realization data
    ///
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

/// Lightweight trajectory containing output data for one complete simulation scenario.
///
/// This structure represents the memory-efficient output of a simulation run,
/// containing only the data needed for CSV export and analysis, without the
/// heavy computational structures (solver models, basis).
///
/// # Memory Model: Extract-and-Release Pattern
///
/// The lifecycle is:
/// 1. Create handler (expensive: allocates solver models, basis)
/// 2. Run forward pass (computation: uses handler resources)
/// 3. Extract trajectory (lightweight: clone only output data)
/// 4. Drop handler (release: frees solver models, basis)
/// 5. Export CSV (lightweight: iterate trajectories)
///
/// This pattern enables Rayon's `map_init` to create one handler per thread,
/// reuse it across scenarios, then release it when thread completes.
///
/// # Memory Efficiency Example
///
/// 10,000 scenarios, 120 stages, typical system:
/// - **Without extraction**: 10,000 handlers × 6MB = 60 GB
/// - **With extraction**: 10,000 trajectories × 96KB = 960 MB
/// - **Reduction**: 98.4% memory savings
///
/// # Design Notes
///
/// - `realizations` ordered by stage (corresponds to `study_period_ids`)
/// - `scenario_id` for tracking and debugging
/// - Entire structure is self-contained for easy serialization
/// - No references to graph structures (fully independent)
///
#[derive(Debug, Clone)]
pub struct SimulationTrajectory {
    /// Scenario identifier (0-indexed)
    pub scenario_id: usize,

    /// Stage-by-stage realization data, ordered by stage index
    pub realizations: Vec<RealizationData>,
}

impl SimulationTrajectory {
    /// Convert lightweight `SimulationTrajectory` to full `Trajectory` for output.
    ///
    /// This method converts the memory-efficient intermediate representation
    /// (used during simulation) to the full `Trajectory` format required by
    /// the output module and statistics computation.
    ///
    /// # Memory Note
    ///
    /// This conversion is performed AFTER simulation, when handlers have been
    /// released. It reconstructs the `StageResult` structures from the
    /// lightweight `RealizationData`.
    ///
    /// # Arguments
    ///
    /// * `initial_storage` - Initial storage for stage 0 (from pre-study node)
    ///
    /// # Returns
    ///
    /// Full `Trajectory` with `stages`, `total_cost`, and `scenario_id`
    ///
    pub fn to_trajectory(&self, initial_storage: &[f64]) -> Trajectory {
        let num_stages = self.realizations.len();
        let mut stages = Vec::with_capacity(num_stages);
        let mut total_cost = 0.0;

        for (stage_idx, realization) in self.realizations.iter().enumerate() {
            // State: initial storage for stage 0, previous final storage for subsequent stages
            let state = if stage_idx == 0 {
                initial_storage.to_vec()
            } else {
                self.realizations[stage_idx - 1].final_storage.clone()
            };

            // Action: aggregate all action variables
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
    ///
    /// This function allocates all required graph structures and initializes them with
    /// the provided initial condition. It's designed to be called once per simulation
    /// scenario, typically within a parallel context (e.g., Rayon's `map_init`).
    ///
    /// # Arguments
    ///
    /// * `pre_study_id` - The ID of the pre-study node in the graph
    /// * `node_data_graph` - Reference to the node data graph containing system configurations
    /// * `initial_condition` - Initial storage and inflow conditions for hydro units
    ///
    /// # Returns
    ///
    /// * `Ok(SddpSimulationHandler)` - Successfully created handler with initialized state
    /// * `Err(String)` - Descriptive error message if creation fails
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    ///
    /// * `pre_study_id` does not exist in the graph (invalid node ID)
    /// * Initial condition storage size doesn't match the system's hydro unit count
    /// * The graph is empty or malformed
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Successful creation
    /// let handler = SddpSimulationHandler::new(
    ///     &pre_study_id,
    ///     &node_data_graph,
    ///     &initial_condition,
    /// )?;
    ///
    /// // Error handling in parallel context (Rayon map_init)
    /// let trajectories: Vec<SimulationTrajectory> = (0..num_scenarios)
    ///     .into_par_iter()
    ///     .map_init(
    ///         || SddpSimulationHandler::new(&pre_study_id, &graph, &ic),
    ///         |handler_result, _scenario_idx| {
    ///             let handler = handler_result.as_mut().unwrap();
    ///             // ... use handler ...
    ///         }
    ///     )
    ///     .collect::<Result<Vec<_>, String>>()?;
    /// ```
    ///
    /// # Performance
    ///
    /// This function performs memory allocation proportional to:
    /// - Number of nodes in the graph (O(N))
    /// - System size (buses, lines, hydros, thermals) per node
    ///
    /// Memory is allocated once and reused across all forward passes for this scenario.
    pub fn new(
        pre_study_id: &usize,
        node_data_graph: &graph::DirectedGraph<NodeData>,
        initial_condition: &initial_condition::InitialCondition,
    ) -> Result<Self, String> {
        // Validate graph is not empty
        if node_data_graph.node_count() == 0 {
            return Err(
                "Cannot create simulation handler: node data graph is empty"
                    .to_string(),
            );
        }

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
                    &node_data.inflow_stochastic_processes,
                    &node_data.unified_specs,
                    node_data.season_id,
                )
            });

        // Get pre-study node and validate initial condition size
        let pre_study_node = realization_graph
            .get_node_mut(*pre_study_id)
            .ok_or_else(|| {
                format!(
                    "Cannot create simulation handler: pre-study node with ID {} not found in graph (graph has {} nodes)",
                    pre_study_id,
                    node_data_graph.node_count()
                )
            })?;

        // Validate storage size matches before attempting clone
        let expected_storage_size = pre_study_node.data.final_storage.len();
        let provided_storage_size = initial_condition.get_storage().len();
        if expected_storage_size != provided_storage_size {
            return Err(format!(
                "Cannot create simulation handler: initial condition storage size mismatch (expected {} hydro units, got {})",
                expected_storage_size,
                provided_storage_size
            ));
        }

        // Safe to clone now that sizes are validated
        pre_study_node
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
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let mut timing = ForwardPassTimingAccumulator::default();

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

            // Model preprocessing timing
            let prep_start = std::time::Instant::now();
            subproblem_node
                .data
                .update_with_current_trajectory(past_realizations);
            timing.model_preprocessing_time += prep_start.elapsed();

            let realization_node =
                self.realization_graph.get_node_mut(*id).ok_or_else(|| {
                    format!("Could not find realization for node {}", id)
                })?;

            let current_stage_noises =
                sampled_noises.get(*id).ok_or_else(|| {
                    format!("Could not find noises for node {}", id)
                })?;

            // Step includes solver + state extraction
            let step_timing = step(
                data_node,
                &mut subproblem_node.data,
                &mut realization_node.data,
                current_stage_noises,
            )?;
            timing.solver_time += step_timing.solver_time;

            // Model postprocessing: state transition to next stage
            let post_start = std::time::Instant::now();
            timing.model_postprocessing_time += step_timing.state_update_time;
            timing.solver_calls += 1;

            subproblem_node
                .data
                .update_with_current_realization(&realization_node.data);
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

    /// Extract complete trajectory data from this simulation handler.
    ///
    /// Constructs a `Trajectory` by iterating through all study period nodes
    /// and collecting state, action, cost, and uncertainty realization data.
    ///
    /// # Performance
    ///
    /// - Pre-allocates `stages` vector with capacity (zero reallocation)
    /// - Pre-allocates `action` vector for each stage
    /// - Clones required due to ownership (unavoidable for return value)
    /// - Total overhead: <5% of simulation time for typical problems
    /// - Dominated by vector clones, not iteration overhead
    ///
    /// # Returns
    ///
    /// `Trajectory` containing complete stage-by-stage data, or error if any
    /// study period node is missing from the realization graph.
    ///
    pub fn extract_trajectory(
        &self,
        study_period_ids: &[usize],
        scenario_id: usize,
    ) -> Result<Trajectory, String> {
        let num_stages = study_period_ids.len();

        // PERFORMANCE: Pre-allocate stages vector to avoid reallocation
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

            // Get previous stage storage (initial storage for stage 0)
            let state = if stage_idx == 0 {
                // For first stage, get from pre-study node
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
                // For subsequent stages, get final storage from previous stage
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

            // PERFORMANCE: Collect action variables into single vector with pre-allocation
            // Order: turbined_flow, thermal_generation, spillage, exchange, deficit
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

    /// Extract lightweight simulation trajectory from this handler.
    ///
    /// This method extracts only the output data (Vec<f64> fields and scalars)
    /// without the heavy computational structures (solver basis, models).
    /// It's designed for the Extract-and-Release memory optimization pattern.
    ///
    /// # Memory Optimization Strategy
    ///
    /// The key insight: handlers are expensive (~6MB each with solver models),
    /// but we only need lightweight output (~96KB per trajectory). This method
    /// enables:
    ///
    /// 1. **Thread-local handlers**: Rayon's `map_init` creates one handler per thread
    /// 2. **Reuse**: Same handler processes multiple scenarios on that thread
    /// 3. **Extract**: After each simulation, extract lightweight trajectory
    /// 4. **Release**: Drop handler when thread finishes, not per-scenario
    ///
    /// Result: O(threads) memory instead of O(scenarios) memory.
    ///
    /// # Performance
    ///
    /// - Complexity: O(stages × system_size)
    /// - Typical overhead: <1% of simulation time
    /// - Memory allocated: ~96KB for 120 stages, typical system
    ///
    /// The clone cost (~0.1ms per stage) is negligible compared to solver time
    /// (~10-100ms per stage).
    ///
    /// # Arguments
    ///
    /// * `study_period_ids` - IDs of study period nodes to extract (in order)
    /// * `scenario_id` - Identifier for this scenario (for tracking)
    ///
    /// # Returns
    ///
    /// * `Ok(SimulationTrajectory)` - Extracted lightweight trajectory
    /// * `Err(String)` - Error if any study period node is missing
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In Rayon map_init pattern (SIM-OPT-005)
    /// let trajectories: Vec<SimulationTrajectory> = (0..num_scenarios)
    ///     .into_par_iter()
    ///     .map_init(
    ///         || SddpSimulationHandler::new(&pre_study_id, &graph, &ic).unwrap(),
    ///         |handler, scenario_id| {
    ///             // Run simulation
    ///             handler.forward(noises, ...)?;
    ///             
    ///             // Extract lightweight data (handler stays alive for reuse)
    ///             handler.extract_simulation_trajectory(&study_period_ids, scenario_id)
    ///         }
    ///     )
    ///     .collect::<Result<Vec<_>, _>>()?;
    ///
    /// // handlers dropped here (one per thread, not per scenario)
    /// // trajectories contain all needed data for CSV export
    /// ```
    ///
    pub fn extract_simulation_trajectory(
        &self,
        study_period_ids: &[usize],
        scenario_id: usize,
    ) -> Result<SimulationTrajectory, String> {
        // PERFORMANCE: Pre-allocate realizations vector
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

            // Extract lightweight data (clones Vec<f64> fields, discards basis)
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

    /// Create a high-level builder for ergonomic SDDP construction.
    ///
    /// This is a convenience method that returns a `SddpBuilder`, which provides
    /// a fluent API for common SDDP construction patterns. Reduces typical test
    /// code from ~150 lines to ~8 lines.
    ///
    /// # Returns
    ///
    /// A fresh `SddpBuilder` instance with default values.
    ///
    pub fn builder() -> SddpBuilder {
        SddpBuilder::new()
    }

    /// Create SDDP algorithm from JSON input files (Factory API).
    ///
    /// This is the **recommended method** for testing, benchmarking, and production use.
    /// It encapsulates the complete construction pattern from `run()` in a single call.
    ///
    /// # Factory Pattern
    ///
    /// This method combines six construction steps into one:
    /// 1. Load and validate input files (config, system, graph, recourse)
    /// 2. Build the graph from JSON configuration
    /// 3. Create initial condition from recourse data
    /// 4. Generate SAA scenarios from stochastic processes
    /// 5. Construct SDDP algorithm with low-level API
    /// 6. Bundle everything into `SddpInstance` for ergonomic use
    ///
    /// The `SddpInstance` provides zero-argument `train()` and `simulate()` methods
    /// for maximum convenience.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if:
    /// - Any input file is missing or unreadable
    /// - JSON parsing fails (malformed JSON)
    /// - Validation fails (invalid constraints, see `InputValidator`)
    /// - Graph construction fails (connectivity, references)
    /// - Algorithm initialization fails
    ///
    /// Error messages are designed to be actionable, identifying:
    /// - Which file failed
    /// - What constraint was violated
    /// - What value was found vs. expected
    ///
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

    /// Train the SDDP algorithm using Sample Average Approximation.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of SDDP iterations to perform
    /// * `num_forward_passes` - Number of forward passes per iteration
    /// * `saa` - Sample Average Approximation for uncertainty realization
    ///
    /// # Returns
    ///
    /// Returns `Ok(TrainingResult)` containing complete convergence history including:
    /// - Iteration-by-iteration bounds and gaps
    /// - Best upper bound found and its iteration
    /// - Final lower and upper bounds
    /// - Total training time and cut count
    /// - Termination reason
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = sddp.train(100, 20, &saa)?;
    /// println!("Final gap: {:.4}", result.final_gap());
    /// println!("Converged: {}", result.converged(1e-3));
    /// ```
    pub fn train(
        &mut self,
        num_iterations: usize,
        num_forward_passes: usize,
        saa: &scenario::SAA,
    ) -> Result<TrainingResult, String> {
        // Validate parameters
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

        // rng is always created for reproducibility
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);

        let begin = Instant::now();

        // Pre-allocate iterations vector for zero-cost tracking
        let mut iterations = Vec::with_capacity(num_iterations);

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

            // Cut selection statistics for this iteration
            let mut backward_cuts_removed: usize = 0;
            let mut backward_cuts_returned: usize = 0;

            // --- SINGLE-THREADED: SAA Sampling ---
            let saa_sampling_begin = Instant::now();
            let all_sampled_noises: Vec<_> = (0..num_forward_passes)
                .map(|_| saa.sample_scenario(&mut rng))
                .collect();
            let saa_sampling_time = saa_sampling_begin.elapsed();

            // --- MULTI-THREADED: Parallel Forward Passes ---
            let forward_parallel_begin = Instant::now();
            let forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = train_handlers
                .par_iter_mut()
                .zip(all_sampled_noises.par_iter())
                .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
                .collect::<Result<Vec<(f64, ForwardPassTimingAccumulator)>, String>>()?;
            let forward_parallel_time = forward_parallel_begin.elapsed();

            // --- SINGLE-THREADED: Forward Postprocessing ---
            let forward_post_begin = Instant::now();

            // Unzip costs and timings
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
            // If internal_forward_timings is zero, components remain zero (edge case)

            // Count total solver calls across all trajectories
            let forward_solver_calls: usize =
                forward_timings.iter().map(|t| t.solver_calls).sum();

            let forward_postprocessing_time = forward_post_begin.elapsed();

            // Complete forward timing structure with single-threaded components
            forward_timing.saa_sampling_time = saa_sampling_time;
            forward_timing.forward_postprocessing_time =
                forward_postprocessing_time;
            forward_timing.total_time = saa_sampling_time
                + forward_parallel_time
                + forward_postprocessing_time;

            // --- Parallel Backward Pass with Stage-wise Synchronization ---
            let backward_begin = Instant::now();
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
                    let phase1_begin = Instant::now();
                    let phase1_results: Vec<(
                        fcf::CutStatePair,
                        BackwardPhase1Timing,
                    )> = train_handlers
                        .par_iter_mut()
                        .enumerate()
                        .map(|(forward_pass_idx, handler)| {
                            handler.compute_cut_for_backward_step(
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

                    // Unzip cuts and timings
                    let (mut cut_state_pairs, phase1_timings): (
                        Vec<fcf::CutStatePair>,
                        Vec<BackwardPhase1Timing>,
                    ) = phase1_results.into_iter().unzip();

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
                    let active_cut_indices_before: std::collections::BTreeMap<
                        usize,
                        usize,
                    > = {
                        let parent_fcf_node = self
                            .future_cost_function_graph
                            .get_node(parent_id)
                            .ok_or_else(|| {
                                format!(
                                    "Could not find FCF for parent node {}",
                                    parent_id
                                )
                            })?;
                        let fcf_locked = parent_fcf_node.data.lock().unwrap();
                        fcf_locked.cut_pool.active_cut_indices.clone()
                    };

                    // Sort cuts before batch processing to ensure deterministic
                    // cut ordering regardless of parallel thread completion order. This is CRITICAL
                    // for reproducibility because intra-batch domination is order-dependent.
                    //
                    // We sort by forward_pass_idx (handler ID)
                    cut_state_pairs
                        .sort_unstable_by_key(|pair| pair.forward_pass_idx);

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
                        fcf_locked.add_cuts_batch(cut_state_pairs)
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
                                cut.active = false;
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
                        let cut_cloning_begin = Instant::now();
                        let cuts: Vec<(usize, crate::cut::BendersCut)> =
                            aggregated_result
                                .new_cut_ids
                                .iter()
                                .chain(
                                    aggregated_result.returning_cut_ids.iter(),
                                )
                                .filter_map(|&cut_id| {
                                    fcf_locked
                                        .cut_pool
                                        .pool
                                        .get(cut_id)
                                        .map(|cut| (cut_id, cut.clone()))
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
                                &active_cut_indices_before,
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

            log::training_table_row(
                index + 1,
                lower_bound,
                simulation_cost,
                forward_timing.total_time,
                backward_total_time,
                iter_time,
            );

            if std::env::var("POWERS_TIMING_DETAIL").is_ok() {
                log::training_iteration_timing(
                    forward_timing.total_time,
                    saa_sampling_time,
                    forward_timing.model_preprocessing_time,
                    forward_timing.solver_time,
                    forward_timing.model_postprocessing_time,
                    forward_postprocessing_time,
                    backward_total_time,
                    total_backward_preprocessing_time,
                    total_backward_model_preprocessing_time,
                    total_backward_solver_time,
                    total_backward_model_postprocessing_time,
                    total_backward_cutsel_time,
                    total_backward_fcf_state_update_time,
                    total_backward_cut_cloning_time,
                    total_backward_handler_application_time,
                    forward_solver_calls + backward_solver_calls,
                    backward_cuts_added,
                    backward_cuts_removed,
                    backward_cuts_returned,
                    active_cut_count,
                );
            }
        }

        log::training_table_divider();
        let total_time = begin.elapsed();
        log::training_duration(total_time);

        // Count cuts in final policy
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

        log::policy_size(num_cuts);

        // === FINAL SIMULATION: Evaluate the trained policy (in-sample) ===
        log::final_simulation_greeting(num_forward_passes);

        let final_sampled_noises: Vec<_> = (0..num_forward_passes)
            .map(|_| saa.sample_scenario(&mut rng))
            .collect();

        let final_forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = train_handlers
            .par_iter_mut()
            .zip(final_sampled_noises.par_iter())
            .map(|(handler, noises)| self.forward(noises.to_vec(), handler))
            .collect::<Result<Vec<(f64, ForwardPassTimingAccumulator)>, String>>()?;

        let (final_forward_costs, _): (
            Vec<f64>,
            Vec<ForwardPassTimingAccumulator>,
        ) = final_forward_results.into_iter().unzip();

        let final_upper_bound = utils::mean_deterministic(&final_forward_costs);
        let final_std = utils::standard_deviation(&final_forward_costs);

        log::final_simulation_stats(final_upper_bound, final_std);

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

        // Create and return training result
        Ok(TrainingResult {
            iterations,
            final_lower_bound,
            final_upper_bound,
            statistical_upper_bound,
            best_upper_bound,
            best_iteration,
            total_time,
            num_cuts,
            termination_reason: TerminationReason::IterationLimit,
            final_simulation_performed: true,
        })
    }

    pub fn forward(
        &self,
        sampled_noises: Vec<&scenario::SampledBranchingNoises>,
        handler: &mut SddpTrainHandler,
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        let (trajectory_cost, timing) = handler.forward(
            sampled_noises,
            &self.node_data_graph,
            &self.graph_bfs_table,
            &self.study_period_ids,
        )?;
        Ok((trajectory_cost, timing))
    }

    /// Simulate the trained policy across multiple scenarios.
    ///
    /// This method implements the **Extract-and-Release memory optimization pattern**
    /// using Rayon's `map_init` to achieve O(threads) memory usage instead of O(scenarios).
    ///
    /// # Memory Optimization Strategy
    ///
    /// **Old approach** (before SIM-OPT-005):
    /// ```ignore
    /// // Allocate one handler per scenario upfront
    /// let handlers: Vec<SddpSimulationHandler> = (0..num_scenarios)
    ///     .map(|_| SddpSimulationHandler::new(...))  // 10K scenarios × 6MB = 60 GB!
    ///     .collect();
    /// ```
    ///
    /// **New approach** (Extract-and-Release with map_init):
    /// ```ignore
    /// // Lazy per-thread handler allocation (Rayon work-stealing)
    /// let trajectories = scenarios.par_iter().map_init(
    ///     || SddpSimulationHandler::new(...),  // 8 threads × 6MB = 48 MB
    ///     |handler, scenario| {
    ///         handler.forward(...)?;           // Computation (reuses handler)
    ///         handler.extract_trajectory(...)  // Extract lightweight data (96 KB)
    ///     }
    /// ).collect()?;
    /// // Handlers dropped here (per-thread, not per-scenario)
    /// // Total memory: 48 MB + (10K × 96 KB) = 2.5 GB instead of 60 GB!
    /// ```
    ///
    /// # Key Advantages
    ///
    /// - **Memory**: O(threads) handlers + O(scenarios) trajectories = ~96% reduction
    /// - **Performance**: No batch synchronization, pure work-stealing for load balancing
    /// - **Simplicity**: Rayon handles thread-local state automatically
    /// - **Correctness**: `forward()` overwrites all state, so handler reuse is safe
    ///
    /// # Performance Characteristics
    ///
    /// - Handler allocation: Lazy (only when thread needs one)
    /// - Parallelism: Full work-stealing (no synchronization overhead)
    /// - Extraction overhead: <1% (~0.1ms per stage vs 10-100ms solver)
    /// - Memory: O(threads × handler_size + scenarios × trajectory_size)
    ///
    /// For 10,000 scenarios, 120 stages, 8 threads:
    /// - Handlers: 8 × 6 MB = 48 MB
    /// - Trajectories: 10,000 × 240 KB = 2.4 GB
    /// - **Total: 2.45 GB vs 60 GB (96% reduction)**
    ///
    /// # Arguments
    ///
    /// * `num_simulation_scenarios` - Number of scenarios to simulate
    /// * `saa` - Sample average approximation for scenario generation
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<SimulationTrajectory>)` - Lightweight trajectories with output data
    /// * `Err(String)` - Error if handler creation or simulation fails
    ///
    /// # Errors
    ///
    /// - Handler creation fails (e.g., invalid initial condition, graph issues)
    /// - Forward pass fails (e.g., infeasible subproblem, solver error)
    /// - Trajectory extraction fails (e.g., missing study period node)
    ///
    pub fn simulate(
        &mut self,
        num_simulation_scenarios: usize,
        saa: &scenario::SAA,
    ) -> Result<Vec<SimulationTrajectory>, String> {
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);

        let begin = Instant::now();

        log::simulation_greeting(num_simulation_scenarios);

        // Pre-generate all noise samples (deterministic with seed)
        let all_sampled_noises: Vec<_> = (0..num_simulation_scenarios)
            .map(|_| saa.sample_scenario(&mut rng))
            .collect();

        // PERFORMANCE: Extract-and-Release pattern with map_init
        // - Init closure: Creates ONE handler per thread (lazy allocation)
        // - Map closure: Runs forward pass, extracts trajectory, returns lightweight data
        // - Handler is reused across scenarios on same thread
        // - Handler is automatically dropped when thread finishes
        //
        // Memory: O(threads) × 6MB + O(scenarios) × 96KB
        //   vs old O(scenarios) × 6MB
        //
        // Result: 96% memory reduction for large simulations
        let trajectories: Vec<SimulationTrajectory> = all_sampled_noises
            .par_iter()
            .enumerate()
            .map_init(
                || {
                    // Init closure: called once per thread (lazy)
                    // Returns Result for error propagation
                    SddpSimulationHandler::new(
                        &self.pre_study_id,
                        &self.node_data_graph,
                        &self.initial_condition,
                    )
                },
                |handler_result, (scenario_id, noises)| {
                    // Map closure: called for each scenario
                    // Handler is reused (forward() overwrites all state)

                    // Check handler creation succeeded
                    let handler = handler_result.as_mut().map_err(|e| {
                        format!(
                            "Handler creation failed for thread processing scenario {}: {}",
                            scenario_id, e
                        )
                    })?;

                    // Run forward pass (mutates handler state)
                    let (_trajectory_cost, _timing) = handler.forward(
                        noises.to_vec(),
                        &self.node_data_graph,
                        &self.graph_bfs_table,
                        &self.study_period_ids,
                    )?;

                    // Extract lightweight trajectory data
                    // (clones Vec<f64> fields, discards heavy basis)
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
        log::simulation_stats(mean_cost, std_cost);

        let duration = begin.elapsed();
        log::simulation_duration(duration);

        Ok(trajectories)
    }

    /// Simulate and analyze the trained policy with comprehensive statistics.
    ///
    /// This is a convenience method that:
    /// 1. Calls `simulate()` to run forward passes and extract lightweight trajectories
    /// 2. Converts lightweight `SimulationTrajectory` to full `Trajectory` for output
    /// 3. Computes statistics across all trajectories
    /// 4. Packages results into `SimulationResult` for CSV export
    ///
    /// # Memory Note
    ///
    /// After SIM-OPT-005, the memory flow is:
    /// - `simulate()`: O(threads) handlers + O(scenarios) lightweight trajectories
    /// - `to_trajectory()`: Converts lightweight to full format for output
    /// - Result: O(scenarios) full trajectories for CSV export
    ///
    /// The key optimization is that handlers are released during simulation,
    /// not kept until output. This saves ~96% memory for large simulations.
    ///
    /// # Arguments
    ///
    /// * `num_simulation_scenarios` - Number of scenarios to simulate
    /// * `saa` - Sample average approximation for scenario generation
    ///
    /// # Returns
    ///
    /// `SimulationResult` containing trajectories, statistics, and dimensions
    ///
    pub fn simulate_and_analyze(
        &mut self,
        num_simulation_scenarios: usize,
        saa: &scenario::SAA,
    ) -> Result<SimulationResult, String> {
        // Run simulation (returns lightweight trajectories, handlers already released)
        let sim_trajectories = self.simulate(num_simulation_scenarios, saa)?;

        // Get initial storage for trajectory conversion
        let initial_storage = self.initial_condition.get_storage();

        // Convert lightweight trajectories to full format for output
        let trajectories: Vec<Trajectory> = sim_trajectories
            .iter()
            .map(|sim_traj| sim_traj.to_trajectory(initial_storage))
            .collect();

        // Compute statistics across all trajectories
        let statistics = compute_statistics(&trajectories);

        // Get dimensions from first trajectory (all should be identical)
        let (num_stages, num_states, num_actions) =
            if let Some(first_traj) = trajectories.first() {
                let num_stages = first_traj.stages.len();
                let num_states = first_traj
                    .stages
                    .first()
                    .map(|s| s.state.len())
                    .unwrap_or(0);
                let num_actions = first_traj
                    .stages
                    .first()
                    .map(|s| s.action.len())
                    .unwrap_or(0);
                (num_stages, num_states, num_actions)
            } else {
                (0, 0, 0)
            };

        Ok(SimulationResult {
            trajectories,
            statistics,
            num_stages,
            num_states,
            num_actions,
        })
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
    data_node: &graph::Node<NodeData>,
    subproblem: &mut subproblem::Subproblem,
    realization_container: &mut subproblem::Realization,
    noises: &scenario::SampledBranchingNoises,
) -> Result<StepTiming, String> {
    // realize_uncertainties now returns precise timing
    let realize_timing = subproblem.realize_uncertainties(
        noises,
        data_node.data.load_stochastic_process.as_ref(),
        &data_node.data.inflow_stochastic_processes,
        realization_container,
    )?;

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

#[cfg(test)]
/// Create empty noise_models vec for test fixtures
///
/// Returns an empty slice that can be passed to NodeData::new() in tests
/// where we don't care about the specific noise models (using "naive" processes).
fn test_empty_noise_models() -> Vec<crate::unified_noise_spec::UnifiedNoiseSpec>
{
    vec![]
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                            &test_empty_noise_models(),
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
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
            .unwrap();

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
                1, // iteration = 1 for tests
                0, // forward_pass_idx = 0 for tests
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                        &test_empty_noise_models(),
                        "storage",
                        1,
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

        let _result = sddp_algo.train(24, 1, &saa).unwrap();
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                        &test_empty_noise_models(),
                        "storage",
                        1,
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

        let _result = sddp_algo.train(24, 1, &saa).unwrap();

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
        let (forward_timing, backward_timing) = placeholder_timing();
        let iterations = vec![
            IterationResult {
                iteration: 1,
                lower_bound: 1000.0,
                forward_costs: vec![1400.0, 1600.0],
                iteration_time: Duration::from_secs(1),
                forward_timing,
                backward_timing,
                num_solver_calls: 0,
                num_cuts_added: 0,
                num_cuts_removed: 0,
                num_cuts_returned: 0,
                num_active_cuts: 10,
            },
            IterationResult {
                iteration: 2,
                lower_bound: 1200.0,
                forward_costs: vec![1300.0, 1400.0],
                iteration_time: Duration::from_secs(1),
                forward_timing,
                backward_timing,
                num_solver_calls: 0,
                num_cuts_added: 0,
                num_cuts_removed: 0,
                num_cuts_returned: 0,
                num_active_cuts: 20,
            },
            IterationResult {
                iteration: 3,
                lower_bound: 1250.0,
                forward_costs: vec![1280.0, 1320.0],
                iteration_time: Duration::from_millis(950),
                forward_timing,
                backward_timing,
                num_solver_calls: 0,
                num_cuts_added: 0,
                num_cuts_removed: 0,
                num_cuts_returned: 0,
                num_active_cuts: 30,
            },
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
            final_upper_bound: 1300.0, // From final simulation
            statistical_upper_bound,
            best_upper_bound,
            best_iteration,
            total_time: Duration::from_millis(2950),
            num_cuts: 15,
            termination_reason: TerminationReason::IterationLimit,
            final_simulation_performed: true,
        }
    }

    #[test]
    fn test_training_result_final_gap() {
        let result = create_test_training_result();
        // Gap: 1300.0 (final_upper_bound) - 1250.0 (final_lower_bound) = 50.0
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
        let iter_result = IterationResult {
            iteration: 2,
            lower_bound: 1000.0,
            forward_costs: vec![1150.0, 1200.0, 1250.0],
            iteration_time: Duration::from_secs(1),
            forward_timing,
            backward_timing,
            num_solver_calls: 0,
            num_cuts_added: 0,
            num_cuts_removed: 0,
            num_cuts_returned: 0,
            num_active_cuts: 10,
        };

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
        let (forward_timing, backward_timing) = placeholder_timing();
        let iterations = vec![IterationResult {
            iteration: 1,
            lower_bound: 1000.0,
            forward_costs: vec![1100.0],
            iteration_time: Duration::from_secs(1),
            forward_timing,
            backward_timing,
            num_solver_calls: 0,
            num_cuts_added: 0,
            num_cuts_removed: 0,
            num_cuts_returned: 0,
            num_active_cuts: 10,
        }];

        let result = TrainingResult {
            iterations,
            final_lower_bound: 1000.0,
            final_upper_bound: 1100.0, // From final simulation
            statistical_upper_bound: 1100.0,
            best_upper_bound: 1100.0, // Best simulation cost from iteration 1
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 5,
            termination_reason: TerminationReason::IterationLimit,
            final_simulation_performed: true,
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
        let (forward_timing, backward_timing) = placeholder_timing();
        let result = TrainingResult {
            iterations: vec![IterationResult {
                iteration: 1,
                lower_bound: 1e6,
                forward_costs: vec![1e9],
                iteration_time: Duration::from_secs(1),
                forward_timing,
                backward_timing,
                num_solver_calls: 0,
                num_cuts_added: 0,
                num_cuts_removed: 0,
                num_cuts_returned: 0,
                num_active_cuts: 10,
            }],
            final_lower_bound: 1e6,
            final_upper_bound: 1e9, // From final simulation
            statistical_upper_bound: 1e9,
            best_upper_bound: 1e9, // Best simulation cost from iteration 1
            best_iteration: 1,
            total_time: Duration::from_secs(1),
            num_cuts: 1,
            termination_reason: TerminationReason::IterationLimit,
            final_simulation_performed: true,
        };

        // Should handle large numbers correctly
        assert!((result.final_gap() - (1e9 - 1e6)).abs() < 1e3);
        assert!(result.relative_gap() > 900.0); // Very large relative gap
        assert!(!result.converged(1e8));
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

        let statistics = compute_statistics(&trajectories);

        SimulationResult {
            trajectories,
            statistics,
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
    fn test_confidence_interval_creation() {
        let ci = ConfidenceInterval {
            lower: 90.0,
            upper: 110.0,
            confidence_level: 0.95,
        };

        assert_eq!(ci.lower, 90.0);
        assert_eq!(ci.upper, 110.0);
        assert_eq!(ci.confidence_level, 0.95);

        // Verify it's Copy
        let ci2 = ci;
        assert_eq!(ci.lower, ci2.lower);
    }

    #[test]
    fn test_compute_percentile_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(compute_percentile(&values, 0.0), 1.0);
        assert_eq!(compute_percentile(&values, 0.25), 2.0);
        assert_eq!(compute_percentile(&values, 0.5), 3.0);
        assert_eq!(compute_percentile(&values, 0.75), 4.0);
        assert_eq!(compute_percentile(&values, 1.0), 5.0);
    }

    #[test]
    fn test_compute_percentile_interpolation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // 20th percentile: between index 0 and 1
        // index = 0.2 * 4 = 0.8
        // result = 1.0 * 0.2 + 2.0 * 0.8 = 1.8
        let p20 = compute_percentile(&values, 0.2);
        assert!((p20 - 1.8).abs() < 1e-10);

        // 60th percentile: between index 2 and 3
        // index = 0.6 * 4 = 2.4
        // result = 3.0 * 0.6 + 4.0 * 0.4 = 3.4
        let p60 = compute_percentile(&values, 0.6);
        assert!((p60 - 3.4).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "Cannot compute percentile of empty vector")]
    fn test_compute_percentile_empty_panics() {
        let values: Vec<f64> = vec![];
        compute_percentile(&values, 0.5);
    }

    #[test]
    #[should_panic(expected = "Percentile must be in [0, 1]")]
    fn test_compute_percentile_invalid_percentile() {
        let values = vec![1.0, 2.0, 3.0];
        compute_percentile(&values, 1.5);
    }

    #[test]
    fn test_compute_statistics_basic() {
        // Create simple trajectories with known costs
        let trajectories = vec![
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 0,
            },
            Trajectory {
                stages: vec![],
                total_cost: 200.0,
                scenario_id: 1,
            },
            Trajectory {
                stages: vec![],
                total_cost: 300.0,
                scenario_id: 2,
            },
        ];

        let stats = compute_statistics(&trajectories);

        // Mean should be 200.0
        assert!((stats.mean - 200.0).abs() < 1e-10);

        // Check percentiles (after sorting: [100, 200, 300])
        // p5: index=0.05*2=0.1 -> 100*(1-0.1)+200*0.1 = 110
        assert!((stats.p5 - 110.0).abs() < 1e-10);
        assert_eq!(stats.p50, 200.0);
        // p95: index=0.95*2=1.9 -> 200*(1-0.9)+300*0.9 = 290
        assert!((stats.p95 - 290.0).abs() < 1e-10);

        assert_eq!(stats.num_trajectories, 3);
    }

    #[test]
    fn test_compute_statistics_standard_deviation() {
        let trajectories = vec![
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 0,
            },
            Trajectory {
                stages: vec![],
                total_cost: 200.0,
                scenario_id: 1,
            },
            Trajectory {
                stages: vec![],
                total_cost: 300.0,
                scenario_id: 2,
            },
        ];

        let stats = compute_statistics(&trajectories);

        // Manual calculation: std = sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 3)
        // = sqrt((10000 + 0 + 10000) / 3) = sqrt(20000/3) ≈ 81.65
        assert!((stats.std - 81.65).abs() < 0.01);
    }

    #[test]
    fn test_compute_statistics_confidence_interval() {
        let trajectories = vec![
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 0,
            },
            Trajectory {
                stages: vec![],
                total_cost: 200.0,
                scenario_id: 1,
            },
            Trajectory {
                stages: vec![],
                total_cost: 300.0,
                scenario_id: 2,
            },
        ];

        let stats = compute_statistics(&trajectories);

        // CI = mean ± 1.96 * (std / sqrt(n))
        // mean = 200, std ≈ 81.65, n = 3
        // margin = 1.96 * 81.65 / sqrt(3) ≈ 92.4
        let expected_margin = 1.96 * stats.std / (3.0_f64).sqrt();

        assert!((stats.ci_95.lower - (200.0 - expected_margin)).abs() < 0.1);
        assert!((stats.ci_95.upper - (200.0 + expected_margin)).abs() < 0.1);
        assert_eq!(stats.ci_95.confidence_level, 0.95);
    }

    #[test]
    fn test_compute_statistics_many_trajectories() {
        // Generate 100 trajectories with costs from 1 to 100
        let trajectories: Vec<Trajectory> = (1..=100)
            .map(|i| Trajectory {
                stages: vec![],
                total_cost: i as f64,
                scenario_id: i - 1,
            })
            .collect();

        let stats = compute_statistics(&trajectories);

        // Mean should be 50.5
        assert!((stats.mean - 50.5).abs() < 1e-10);

        // Check percentiles
        assert!((stats.p5 - 5.95).abs() < 0.1); // 5th percentile
        assert!((stats.p25 - 25.75).abs() < 0.1); // 25th percentile
        assert!((stats.p50 - 50.5).abs() < 0.1); // Median
        assert!((stats.p75 - 75.25).abs() < 0.1); // 75th percentile
        assert!((stats.p95 - 95.05).abs() < 0.1); // 95th percentile

        assert_eq!(stats.num_trajectories, 100);
    }

    #[test]
    fn test_compute_statistics_identical_costs() {
        // All trajectories have the same cost
        let trajectories = vec![
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 0,
            },
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 1,
            },
            Trajectory {
                stages: vec![],
                total_cost: 100.0,
                scenario_id: 2,
            },
        ];

        let stats = compute_statistics(&trajectories);

        assert_eq!(stats.mean, 100.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.p5, 100.0);
        assert_eq!(stats.p50, 100.0);
        assert_eq!(stats.p95, 100.0);

        // CI should be zero-width
        assert_eq!(stats.ci_95.lower, 100.0);
        assert_eq!(stats.ci_95.upper, 100.0);
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
    fn test_simulation_result_get_statistics() {
        let result = create_test_simulation_result();
        let stats = result.get_statistics();

        // Verify it's a copy (Statistics is Copy)
        let stats2 = result.get_statistics();
        assert_eq!(stats.mean, stats2.mean);

        // Verify statistics are reasonable
        assert!(stats.mean > 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.p5 <= stats.p50);
        assert!(stats.p50 <= stats.p95);
        assert_eq!(stats.num_trajectories, 5);
    }

    #[test]
    fn test_simulation_result_dimensions() {
        let result = create_test_simulation_result();

        assert_eq!(result.num_stages, 3);
        assert_eq!(result.num_states, 1);
        assert_eq!(result.num_actions, 3);
    }

    #[test]
    fn test_simulation_result_statistics_consistency() {
        let result = create_test_simulation_result();

        // Total costs: [300.0, 330.0, 270.0, 315.0, 285.0]
        // Mean = (300 + 330 + 270 + 315 + 285) / 5 = 1500 / 5 = 300.0

        let stats = result.statistics;
        assert!((stats.mean - 300.0).abs() < 1e-10);

        // Verify ordering of percentiles
        assert!(stats.p5 <= stats.p25);
        assert!(stats.p25 <= stats.p50);
        assert!(stats.p50 <= stats.p75);
        assert!(stats.p75 <= stats.p95);

        // All percentiles should be within the data range
        assert!(stats.p5 >= 270.0);
        assert!(stats.p95 <= 330.0);

        // Median should be middle value (300.0 when sorted: 270, 285, 300, 315, 330)
        assert_eq!(stats.p50, 300.0);
    }

    #[test]
    fn test_simulation_result_filtering_high_cost_scenarios() {
        let result = create_test_simulation_result();

        // Find scenarios above 95th percentile
        let high_cost: Vec<_> = result
            .get_all_trajectories()
            .iter()
            .filter(|t| t.total_cost > result.statistics.p95)
            .collect();

        // With 5 scenarios, we expect ~0-1 above p95
        assert!(high_cost.len() <= 1);
    }

    #[test]
    fn test_statistics_copy_semantics() {
        let stats = Statistics {
            mean: 100.0,
            std: 10.0,
            p5: 85.0,
            p25: 92.0,
            p50: 100.0,
            p75: 108.0,
            p95: 115.0,
            ci_95: ConfidenceInterval {
                lower: 95.0,
                upper: 105.0,
                confidence_level: 0.95,
            },
            num_trajectories: 100,
        };

        // Should be Copy
        let stats2 = stats;
        let _stats3 = stats; // Should still be usable

        assert_eq!(stats.mean, stats2.mean);
    }

    #[test]
    fn test_simulate_and_analyze_with_trained_policy() {
        // Integration test: Train a simple policy and run simulation analysis
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
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
                        &test_empty_noise_models(),
                        "storage",
                        1,
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

        // Train for a few iterations
        let _train_result = sddp_algo.train(5, 1, &saa).unwrap();

        // Now simulate and analyze
        let sim_result = sddp_algo.simulate_and_analyze(10, &saa).unwrap();

        // Verify result structure
        assert_eq!(sim_result.trajectories.len(), 10);
        assert_eq!(sim_result.num_stages, 4); // Stages 0, 1, 2, 3
        assert_eq!(sim_result.num_states, 1); // One hydro reservoir
        assert!(sim_result.num_actions > 0);

        // Verify statistics are reasonable
        assert!(sim_result.statistics.mean > 0.0);
        assert!(sim_result.statistics.std >= 0.0);
        assert!(sim_result.statistics.p5 <= sim_result.statistics.p95);
        assert_eq!(sim_result.statistics.num_trajectories, 10);

        // Verify each trajectory has correct structure
        for (i, traj) in sim_result.trajectories.iter().enumerate() {
            assert_eq!(traj.scenario_id, i);
            assert_eq!(traj.stages.len(), 4); // Stages 0, 1, 2, 3

            // Verify total cost matches sum of stage costs
            let sum_costs: f64 = traj.stages.iter().map(|s| s.stage_cost).sum();
            assert!((traj.total_cost - sum_costs).abs() < 1e-6);
        }
    }

    // ========================================================================
    // ADDITIONAL TESTS FOR PHASE 5c
    // ========================================================================

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
            statistics: Statistics {
                mean: 100.0,
                std: 0.0,
                p5: 100.0,
                p25: 100.0,
                p50: 100.0,
                p75: 100.0,
                p95: 100.0,
                ci_95: ConfidenceInterval {
                    lower: 100.0,
                    upper: 100.0,
                    confidence_level: 0.95,
                },
                num_trajectories: 1,
            },
        };

        // Out of bounds should return None
        assert!(result.get_trajectory(999).is_none());
    }

    #[test]
    fn test_training_result_iterations_empty() {
        let result = TrainingResult {
            iterations: vec![],
            final_lower_bound: 0.0,
            final_upper_bound: 0.0,
            statistical_upper_bound: 0.0,
            best_upper_bound: f64::INFINITY,
            best_iteration: 0,
            total_time: Duration::ZERO,
            num_cuts: 0,
            termination_reason: TerminationReason::IterationLimit,
            final_simulation_performed: false,
        };

        assert_eq!(result.iterations().len(), 0);
        assert_eq!(result.lower_bounds().len(), 0);
        // Forward costs are per-iteration
        assert!(result.iterations().is_empty());
    }

    #[test]
    fn test_confidence_interval_display() {
        let ci = ConfidenceInterval {
            lower: 95.5,
            upper: 104.5,
            confidence_level: 0.95,
        };

        let display_str = format!("{:?}", ci);
        assert!(display_str.contains("95.5"));
        assert!(display_str.contains("104.5"));
    }

    #[test]
    fn test_simulation_handler_creation_valid() {
        // Create a minimal valid graph for testing
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
                    &test_empty_noise_models(),
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
        let result = SddpSimulationHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
        );

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
        let result = SddpSimulationHandler::new(
            &0,
            &node_data_graph,
            &initial_condition,
        );

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
    fn test_simulation_handler_creation_invalid_pre_study_id() {
        // Create a graph with one node
        let mut node_data_graph = graph::DirectedGraph::<NodeData>::new();
        let _node_id = node_data_graph
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        // Try to use an invalid pre_study_id (999 doesn't exist)
        let result = SddpSimulationHandler::new(
            &999,
            &node_data_graph,
            &initial_condition,
        );

        assert!(
            result.is_err(),
            "Invalid pre_study_id should cause an error"
        );
        if let Err(error_msg) = result {
            assert!(
                error_msg.contains("node with ID 999 not found"),
                "Error message should mention the invalid node ID, got: {}",
                error_msg
            );
            assert!(
                error_msg.contains("graph has 1 nodes"),
                "Error message should show graph size, got: {}",
                error_msg
            );
        }
    }

    #[test]
    fn test_simulation_handler_creation_storage_size_mismatch() {
        // Create a graph with a system that has 1 hydro unit
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
                    system::System::default(), // Has 1 hydro unit
                    "expectation",
                    "naive",
                    &test_empty_noise_models(),
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
        let result = SddpSimulationHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
        );

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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
                    "naive",
                    &test_empty_noise_models(),
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
        let handler = SddpSimulationHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
        )
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
                    &test_empty_noise_models(),
                    "storage",
                    1,
                )
                .unwrap(),
            )
            .unwrap();

        let storage = vec![100.0];
        let initial_condition =
            initial_condition::InitialCondition::new(storage, vec![]);

        let handler = SddpSimulationHandler::new(
            &pre_study_id,
            &node_data_graph,
            &initial_condition,
        )
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
