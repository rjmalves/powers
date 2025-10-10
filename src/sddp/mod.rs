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

pub mod builder;
pub mod instance;

pub use builder::{SddpBuilder, SddpInstanceBuilder};
pub use instance::SddpInstance;

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

        // Note: saa_sampling, forward_postprocessing, and total_time are set by training loop
        // This only aggregates the per-trajectory parallel components
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

#[derive(Debug, Clone)]
pub struct IterationResult {
    pub iteration: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub forward_costs: Vec<f64>,
    pub gap: f64,
    pub relative_gap: f64,
    pub iteration_time: Duration,
    pub forward_timing: ForwardPassTiming,
    pub backward_timing: BackwardPassTiming,
    pub num_solver_calls: usize,
    pub num_cuts_added: usize,
    pub num_cuts_removed: usize,
    pub num_cuts_returned: usize,
    pub num_active_cuts: usize,
}

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

// ============================================================================
// Simulation Result Analysis Structures (T3.1)
// ============================================================================

/// Result from a single stage in a simulation trajectory.
///
/// Contains all relevant information for one stage of a simulated scenario:
/// state variables, control actions, costs, and realized uncertainties.
///
/// # Performance Notes
///
/// - Uses `Vec<f64>` for state/action/noise storage (heap-allocated)
/// - Designed for post-simulation analysis (not hot path)
/// - Vectors typically small (1-20 elements), acceptable overhead
/// - Memory: ~200-500 bytes per stage for typical problems
#[derive(Debug, Clone)]
pub struct StageResult {
    /// Stage number (0-indexed, where 0 is first study period).
    pub stage: usize,

    /// State variables at the beginning of this stage (before decisions).
    ///
    /// For hydrothermal problems, this is typically reservoir storage levels.
    /// Dimension matches the number of state variables in the system.
    pub state: Vec<f64>,

    /// Control actions taken at this stage.
    ///
    /// For hydrothermal problems, this includes:
    /// - Hydro generation (turbined flow)
    /// - Thermal generation
    /// - Spillage
    /// - Line flows (exchange)
    /// - Deficit
    ///
    /// Dimension matches total control variables in the system.
    pub action: Vec<f64>,

    /// Objective cost for this stage only (not cumulative).
    ///
    /// Includes generation costs, deficit costs, and exchange penalties.
    pub stage_cost: f64,

    /// Realized inflow values for this stage.
    ///
    /// One value per hydro reservoir. These are the actual sampled values
    /// from the stochastic process, not expectations or bounds.
    pub inflow: Vec<f64>,

    /// Realized load values for this stage.
    ///
    /// One value per bus. These are the actual sampled values from the
    /// stochastic process.
    pub load: Vec<f64>,
}

/// Complete trajectory for a single simulated scenario.
///
/// A trajectory represents one complete path through the scenario tree,
/// containing the sequence of states, actions, and costs from initial
/// condition to final stage.
///
/// # Performance Notes
///
/// - `stages` vector pre-allocated with `num_stages` capacity
/// - Total memory per trajectory: ~8 bytes/float × (state_dim + action_dim) × num_stages
/// - For 12-stage problem with 10 state vars + 20 actions: ~3 KB per trajectory
/// - 1000 trajectories: ~3 MB total (acceptable for analysis phase)
///
/// # Example
///
/// ```rust,ignore
/// for stage in &trajectory.stages {
///     println!("Stage {}: cost={:.2}, inflow={:?}",
///              stage.stage, stage.stage_cost, stage.inflow);
/// }
/// println!("Total cost: {:.2}", trajectory.total_cost);
/// ```
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Stage-by-stage results for this trajectory.
    ///
    /// Length equals number of study periods (excludes pre-study period).
    /// Ordered chronologically from stage 0 to final stage.
    pub stages: Vec<StageResult>,

    /// Total cost across all stages (sum of stage_cost values).
    ///
    /// This is the objective value for this simulated scenario.
    pub total_cost: f64,

    /// Scenario identifier (0-indexed).
    ///
    /// Useful for tracking which scenario generated this trajectory,
    /// especially when analyzing outliers or extreme scenarios.
    pub scenario_id: usize,
}

/// Confidence interval for a statistic.
///
/// Computed using normal approximation (CLT) for mean estimation.
///
/// # Performance Notes
///
/// - All fields are `Copy` types (zero-cost)
/// - 24 bytes total (3 × f64)
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
/// # Performance Notes
///
/// - All fields are `Copy` types for efficient access (80 bytes total)
/// - Statistics computed once during result construction
/// - No recomputation on repeated access (O(1) access)
/// - Cache-friendly: all data in single cache line
///
/// # Example
///
/// ```rust,ignore
/// let stats = result.statistics;
/// println!("Mean: {:.2} ± {:.2}",
///          stats.mean,
///          (stats.ci_95.upper - stats.ci_95.lower) / 2.0);
/// println!("Median: {:.2}, 95th percentile: {:.2}",
///          stats.p50, stats.p95);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Statistics {
    /// Mean (expected) cost across all trajectories.
    ///
    /// This is the primary estimate of policy performance.
    /// According to SDDP theory, this provides an unbiased estimate
    /// of the true optimal value (upper bound).
    pub mean: f64,

    /// Standard deviation of costs across trajectories.
    ///
    /// Measures the variability of policy performance across scenarios.
    /// High standard deviation indicates high sensitivity to uncertainty.
    pub std: f64,

    /// 5th percentile of cost distribution.
    ///
    /// 5% of scenarios have cost less than or equal to this value.
    /// Useful for understanding best-case scenarios.
    pub p5: f64,

    /// 25th percentile (first quartile) of cost distribution.
    pub p25: f64,

    /// 50th percentile (median) of cost distribution.
    ///
    /// More robust to outliers than the mean. If mean >> median,
    /// the distribution has a long right tail (high-cost scenarios).
    pub p50: f64,

    /// 75th percentile (third quartile) of cost distribution.
    pub p75: f64,

    /// 95th percentile of cost distribution.
    ///
    /// 95% of scenarios have cost less than or equal to this value.
    /// Useful for understanding worst-case scenarios and risk exposure.
    pub p95: f64,

    /// 95% confidence interval for the mean cost.
    ///
    /// With 95% confidence, the true expected cost lies within this interval.
    /// Computed using normal approximation: mean ± 1.96 * (std / √n).
    ///
    /// Valid for n ≥ 30 under mild conditions (Central Limit Theorem).
    pub ci_95: ConfidenceInterval,

    /// Number of trajectories used to compute these statistics.
    pub num_trajectories: usize,
}

/// Complete result from simulation analysis.
///
/// Contains all trajectory data and computed statistics for a set of
/// simulated scenarios under a trained SDDP policy.
///
/// # Performance Notes
///
/// - `trajectories` vector pre-allocated with `num_scenarios` capacity
/// - Statistics computed once during construction (not lazily)
/// - Total memory: ~3-5 MB for 1000 trajectories on typical problems
/// - Access to statistics is O(1) (no recomputation)
///
/// # Example
///
/// ```rust,ignore
/// // Train policy
/// let mut sddp = SddpAlgorithm::builder()
///     .system(system)
///     .initial_storage(vec![50.0])
///     .num_stages(12)
///     .deterministic_inflows(vec![40.0; 12])
///     .build()?;
/// sddp.train(30, 10, &saa)?;
///
/// // Simulate and analyze
/// let result = sddp.simulate_and_analyze(1000, &saa)?;
///
/// // Access statistics
/// println!("Mean cost: {:.2} ± {:.2}",
///          result.statistics.mean,
///          (result.statistics.ci_95.upper - result.statistics.ci_95.lower) / 2.0);
/// println!("95th percentile: {:.2}", result.statistics.p95);
///
/// // Analyze high-cost scenarios
/// let high_cost: Vec<_> = result.trajectories
///     .iter()
///     .filter(|t| t.total_cost > result.statistics.p95)
///     .collect();
/// println!("Found {} high-cost scenarios (>{:.2})",
///          high_cost.len(), result.statistics.p95);
/// ```
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// All simulated trajectories.
    ///
    /// Length equals `num_simulation_scenarios`. Each trajectory contains
    /// complete stage-by-stage information for one simulated scenario.
    ///
    /// # Performance
    ///
    /// Vector is pre-allocated with capacity to avoid reallocations.
    /// Total memory typically 3-5 MB for 1000 trajectories.
    pub trajectories: Vec<Trajectory>,

    /// Statistical summary of trajectory costs.
    ///
    /// Computed once during result construction. Accessing these statistics
    /// has zero overhead (no recomputation).
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

// ============================================================================
// Statistics Computation Functions (T3.1)
// ============================================================================

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
/// # Panics
///
/// Panics if `sorted_values` is empty or `percentile` is not in [0, 1].
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
/// # Confidence Interval Method
///
/// Uses normal approximation (CLT): mean ± 1.96 * (std / √n)
///
/// Valid for n ≥ 30 under mild conditions. For smaller samples or
/// heavy-tailed distributions, bootstrap would be more accurate but slower.
///
/// # Example
///
/// ```rust,ignore
/// let stats = compute_statistics(&trajectories);
/// println!("Mean: {:.2} ± {:.2}",
///          stats.mean,
///          (stats.ci_95.upper - stats.ci_95.lower) / 2.0);
/// println!("Median: {:.2}", stats.p50);
/// println!("95th percentile: {:.2}", stats.p95);
/// ```
fn compute_statistics(trajectories: &[Trajectory]) -> Statistics {
    let n = trajectories.len();
    assert!(n > 0, "Cannot compute statistics for zero trajectories");

    // PERFORMANCE: Extract costs into separate vector for sorting
    // This avoids sorting full trajectories (much cheaper)
    let mut costs: Vec<f64> =
        trajectories.iter().map(|t| t.total_cost).collect();

    // Compute mean and std using existing utils (tested and optimized)
    let mean = utils::mean(&costs);
    let std = utils::standard_deviation(&costs);

    // REPRODUCIBILITY: Use stable sort to ensure deterministic ordering when
    // costs are equal (common with similar scenarios). Unstable sort can
    // reorder equal elements arbitrarily, causing non-deterministic percentiles.
    // Sequential sort is sufficient here (sorting happens once per simulation,
    // typically <1000 elements, <0.1% of total runtime). See REPRO-006.
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
    /// # Arguments
    /// * `iteration` - 1-based iteration number for tracking cut generation
    /// * `forward_pass_idx` - Index identifying which forward pass (handler) generated this cut.
    ///   Used for deterministic sorting to ensure reproducible results.
    ///
    /// This is an internal method used only within the training loop for batch cut selection.
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

        // Model preprocessing: Currently minimal (realization graph lookups)
        // In future could include: model updates, constraint preparation
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

    /// Apply aggregated cut results without FCF locking (LOCK-FREE version)
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

    /// Evaluate first stage bound when no cuts exist yet.
    ///
    /// This is called for the first stage when there are no parent cuts to evaluate.
    /// It solves all branching scenarios and computes the expected value of the
    /// immediate cost plus future costs (which are zero when no cuts exist).
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

        // PERFORMANCE: solve_all_branchings returns timing - we must capture and return it
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
    /// # Arguments
    ///
    /// * `study_period_ids` - Node IDs for study periods (chronological order)
    /// * `scenario_id` - Scenario identifier for this trajectory
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
    /// # Example
    ///
    /// ```rust,ignore
    /// let trajectory = handler.extract_trajectory(&study_period_ids, 0)?;
    /// println!("Total cost: {:.2}", trajectory.total_cost);
    /// for stage in &trajectory.stages {
    ///     println!("Stage {}: cost={:.2}", stage.stage, stage.stage_cost);
    /// }
    /// ```
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
    /// # Example
    ///
    /// ```ignore
    /// let sddp = SddpAlgorithm::builder()
    ///     .system(my_system)
    ///     .initial_storage(vec![50.0])
    ///     .num_stages(2)
    ///     .deterministic_inflows(vec![vec![30.0], vec![40.0]])
    ///     .build()?;
    /// ```
    ///
    /// For maximum flexibility, use the low-level `new()` constructor instead.
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
    /// # Returns
    ///
    /// `Ok(SddpInstance)` containing:
    /// - The SDDP algorithm (fully initialized, ready to train)
    /// - Configuration (num_iterations, seed, output_path, etc.)
    /// - SAA scenarios (pre-sampled noise realizations)
    ///
    /// The `SddpInstance` provides zero-argument `train()` and `simulate()` methods
    /// for maximum convenience.
    ///
    /// # Arguments
    ///
    /// * `config_path` - Path to config.json (iterations, seed, output)
    /// * `system_path` - Path to system.json (buses, lines, thermals, hydros)
    /// * `graph_path` - Path to graph.json (nodes, edges, stochastic processes)
    /// * `recourse_path` - Path to recourse.json (initial storage, uncertainty specs)
    ///
    /// All paths can be relative or absolute. This method is more flexible than
    /// `Input::build()` which assumes all files are in the same directory.
    ///
    /// # Performance
    ///
    /// - **Zero overhead** compared to manual construction
    /// - Same performance as `run()` in `lib.rs`
    /// - Input validation adds <10μs (<0.002% of training time)
    /// - No allocations beyond what's needed for algorithm itself
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // One-line construction (replaces ~50 lines of boilerplate)
    /// let mut sddp = SddpAlgorithm::from_files(
    ///     "example/config.json",
    ///     "example/system.json",
    ///     "example/graph.json",
    ///     "example/recourse.json",
    /// )?;
    ///
    /// // Zero-argument training
    /// let result = sddp.train()?;
    ///
    /// // Zero-argument simulation
    /// let handlers = sddp.simulate()?;
    ///
    /// // Access components if needed
    /// let fcf_graph = sddp.algorithm().future_cost_function_graph();
    /// let config = sddp.config();
    /// ```
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
    /// # Comparison with Other APIs
    ///
    /// | API | Use Case | Boilerplate | Flexibility |
    /// |-----|----------|-------------|-------------|
    /// | **Factory** (`from_files()`) | Tests, Benchmarks, Production | ~5 lines | Distribution-based uncertainty |
    /// | **Builder** (`builder()`) | Simple tests | ~8 lines | Explicit scenarios only |
    /// | **Low-level** (`new()`) | Power users | ~50 lines | Full control |
    ///
    /// # Design Rationale
    ///
    /// This method exists because:
    /// 1. **Benchmarks need it**: `parallel_efficiency.rs` can't use Builder (no distribution support)
    /// 2. **Tests need it**: Integration tests duplicate 50+ lines of construction
    /// 3. **Production uses it**: This encapsulates the pattern from `run()`
    ///
    /// The factory returns `SddpInstance` (not raw `SddpAlgorithm`) because:
    /// - Training needs config (num_iterations, num_forward_passes)
    /// - Simulation needs config (num_simulation_scenarios) and SAA
    /// - Bundle avoids passing these separately (ergonomics)
    /// - Zero overhead (wrapper is optimized away)
    ///
    /// # See Also
    ///
    /// - `Input::from_paths()` - The underlying file loader
    /// - `SddpInstance` - The returned wrapper type
    /// - `builder()` - Alternative API for simple cases
    /// - `new()` - Low-level constructor for power users
    pub fn from_files(
        config_path: impl AsRef<std::path::Path>,
        system_path: impl AsRef<std::path::Path>,
        graph_path: impl AsRef<std::path::Path>,
        recourse_path: impl AsRef<std::path::Path>,
    ) -> Result<SddpInstance, crate::error::PowersError> {
        // Use builder pattern internally for backward compatibility
        // This is a zero-cost abstraction (inlined, no overhead)
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
        // PERFORMANCE: Avoids reallocation during training loop
        let mut iterations = Vec::with_capacity(num_iterations);

        // Track best upper bound across all iterations
        let mut best_upper_bound = f64::INFINITY;
        let mut best_iteration = 0;

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

            // Timing accumulators for this iteration (NEW STRUCTURE - T4.1 Phase 3.5)
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

            // Cut selection statistics for this iteration (T4.1 Phase 3.5 - Option 4)
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

            // REPRODUCIBILITY: Use deterministic mean to ensure consistent results
            // regardless of parallel thread completion order. Forward passes execute
            // in parallel via Rayon, and their costs arrive in non-deterministic order.
            // Kahan summation guarantees order-independent accumulation. See REPRO-004.
            let avg_forward_cost = utils::mean_deterministic(&forward_costs);

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

                    // REPRODUCIBILITY: Sort cuts before batch processing to ensure deterministic
                    // cut ordering regardless of parallel thread completion order. This is CRITICAL
                    // for reproducibility because intra-batch domination is order-dependent: later
                    // cuts are evaluated against states added by earlier cuts. Random processing
                    // order → random domination results → random active cut sets → diverging bounds.
                    // See REPRO-010 for detailed analysis.
                    //
                    // We sort by forward_pass_idx (handler ID), which provides:
                    // 1. Fast O(1) integer comparison vs O(m) float vector comparison
                    // 2. Robust: unaffected by numerical precision or state representation changes
                    // 3. Semantic: reflects the actual algorithm structure (which handler generated each cut)
                    // 4. Debuggable: can trace cuts back to their generating forward pass
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

                    // --- SINGLE-THREADED: Phase 3a - Update FCF state (mark inactive, adjust HashMap) ---
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

                        // PART 1: Update FCF state (mark cuts inactive, update HashMap)
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

                    // --- PARALLEL: Phase 3b - Apply results to ALL models (LOCK-FREE!) ---
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

            // Compute convergence metrics
            let gap = avg_forward_cost - lower_bound;
            let relative_gap = if lower_bound.abs() < 1e-10 {
                f64::INFINITY
            } else {
                gap / lower_bound.abs()
            };

            // Track best upper bound
            if avg_forward_cost < best_upper_bound {
                best_upper_bound = avg_forward_cost;
                best_iteration = index + 1;
            }

            let backward_total_time = backward_begin.elapsed();
            let iter_time = iter_begin.elapsed();

            // Store iteration result with collected timing data
            iterations.push(IterationResult {
                iteration: index + 1,
                lower_bound,
                upper_bound: avg_forward_cost,
                forward_costs,
                gap,
                relative_gap,
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

            log::training_table_row(
                index + 1,
                lower_bound,
                avg_forward_cost,
                relative_gap,
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

        // Get final bounds from last iteration
        let (final_lower_bound, final_upper_bound) = iterations
            .last()
            .map(|r| (r.lower_bound, r.upper_bound))
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

        let simulation_results: Vec<(f64, ForwardPassTimingAccumulator)> = simulation_handlers
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
            .collect::<Result<Vec<(f64, ForwardPassTimingAccumulator)>, String>>()?;

        // Unzip costs and timings
        let (simulation_costs, _simulation_timings): (
            Vec<f64>,
            Vec<ForwardPassTimingAccumulator>,
        ) = simulation_results.into_iter().unzip();

        // TODO: Add simulation timing logging (similar to training timing logging)

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

    /// Simulate policy and perform comprehensive analysis.
    ///
    /// Combines simulation with trajectory extraction and statistical analysis,
    /// returning a `SimulationResult` with complete trajectory data and statistics.
    ///
    /// This is the recommended method for policy evaluation and validation.
    /// Use `simulate()` if you only need raw simulation handlers.
    ///
    /// # Arguments
    ///
    /// * `num_simulation_scenarios` - Number of scenarios to simulate
    /// * `saa` - Sample Average Approximation for noise generation
    ///
    /// # Returns
    ///
    /// `SimulationResult` containing:
    /// - All simulated trajectories (state/action/cost per stage)
    /// - Statistical summary (mean, std, percentiles, confidence intervals)
    /// - Metadata (num_stages, num_states, num_actions)
    ///
    /// # Performance
    ///
    /// - Simulation: O(num_scenarios × num_stages × optimization_cost)
    /// - Trajectory extraction: O(num_scenarios × num_stages) - typically <5% overhead
    /// - Statistics computation: O(num_scenarios log num_scenarios) - negligible
    ///
    /// Total overhead vs. basic `simulate()`: ~5-10%
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Train policy
    /// let mut sddp = SddpAlgorithm::builder()
    ///     .system(system)
    ///     .initial_storage(vec![50.0])
    ///     .num_stages(12)
    ///     .deterministic_inflows(vec![40.0; 12])
    ///     .build()?;
    /// sddp.train(30, 10, &saa)?;
    ///
    /// // Simulate and analyze
    /// let result = sddp.simulate_and_analyze(1000, &saa)?;
    ///
    /// // Use results
    /// println!("Mean cost: {:.2} ± {:.2}",
    ///          result.statistics.mean,
    ///          (result.statistics.ci_95.upper - result.statistics.ci_95.lower) / 2.0);
    /// println!("95th percentile: {:.2}", result.statistics.p95);
    ///
    /// // Analyze high-cost scenarios
    /// let high_cost_scenarios: Vec<_> = result.trajectories
    ///     .iter()
    ///     .filter(|t| t.total_cost > result.statistics.p95)
    ///     .collect();
    /// println!("Found {} high-cost scenarios", high_cost_scenarios.len());
    /// ```
    pub fn simulate_and_analyze(
        &mut self,
        num_simulation_scenarios: usize,
        saa: &scenario::SAA,
    ) -> Result<SimulationResult, String> {
        // Run simulation (existing method)
        let simulation_handlers =
            self.simulate(num_simulation_scenarios, saa)?;

        // PERFORMANCE: Pre-allocate trajectories vector
        let mut trajectories = Vec::with_capacity(num_simulation_scenarios);

        // Extract trajectories from all handlers
        for (scenario_id, handler) in simulation_handlers.iter().enumerate() {
            let trajectory = handler
                .extract_trajectory(&self.study_period_ids, scenario_id)?;
            trajectories.push(trajectory);
        }

        // Compute statistics
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
/// Updated with refactored timing structure (T4.1 Phase 3.5 Refactoring).
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
        data_node.data.inflow_stochastic_process.as_ref(),
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
                upper_bound: 1500.0,
                forward_costs: vec![1400.0, 1600.0],
                gap: 500.0,
                relative_gap: 0.5,
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
                upper_bound: 1350.0,
                forward_costs: vec![1300.0, 1400.0],
                gap: 150.0,
                relative_gap: 0.125,
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
                upper_bound: 1300.0,
                forward_costs: vec![1280.0, 1320.0],
                gap: 50.0,
                relative_gap: 0.04,
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
            1150.0, 1250.0, // Iter 1
            1300.0, 1400.0, // Iter 2
            1280.0, 1320.0, // Iter 3
        ];
        let statistical_upper_bound =
            all_costs.iter().sum::<f64>() / all_costs.len() as f64;

        TrainingResult {
            iterations,
            final_lower_bound: 1250.0,
            final_upper_bound: 1300.0,
            statistical_upper_bound,
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
        let (forward_timing, backward_timing) = placeholder_timing();
        let iter_result = IterationResult {
            iteration: 1,
            lower_bound: 1000.0,
            upper_bound: 1200.0,
            forward_costs: vec![1150.0, 1200.0, 1250.0],
            gap: 200.0,
            relative_gap: 0.2,
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

        // Verify average equals upper bound
        let avg: f64 = iter_result.forward_costs.iter().sum::<f64>() / 3.0;
        assert!((avg - iter_result.upper_bound).abs() < 1e-10);
    }

    #[test]
    fn test_training_result_single_iteration() {
        let (forward_timing, backward_timing) = placeholder_timing();
        let iterations = vec![IterationResult {
            iteration: 1,
            lower_bound: 1000.0,
            upper_bound: 1100.0,
            forward_costs: vec![1100.0],
            gap: 100.0,
            relative_gap: 0.1,
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
            final_upper_bound: 1100.0,
            statistical_upper_bound: 1100.0, // Only one forward pass in this test
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
        let (forward_timing, backward_timing) = placeholder_timing();
        let result = TrainingResult {
            iterations: vec![IterationResult {
                iteration: 1,
                lower_bound: 1e6,
                upper_bound: 1e9,
                forward_costs: vec![1e9],
                gap: 1e9 - 1e6,
                relative_gap: (1e9 - 1e6) / 1e6,
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
            final_upper_bound: 1e9,
            statistical_upper_bound: 1e9, // Only one forward pass
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

    // ====================================================================
    // Unit tests for Simulation Result Analysis (T3.1)
    // ====================================================================

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
            best_upper_bound: 0.0,
            best_iteration: 0,
            total_time: Duration::ZERO,
            num_cuts: 0,
            termination_reason: TerminationReason::IterationLimit,
        };

        assert_eq!(result.iterations().len(), 0);
        assert_eq!(result.lower_bounds().len(), 0);
        assert_eq!(result.upper_bounds().len(), 0);
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
}
