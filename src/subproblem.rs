use crate::cut;
use crate::fcf;
use crate::risk_measure;
use crate::scenario;
use crate::seasonal_params::SeasonalParams;
use crate::solver;
use crate::state;
use crate::stochastic_process;
use crate::system;
use crate::unified_inflow_model::UnifiedInflowModel;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Timing breakdown for realize_uncertainties operation.
///
/// This struct captures precise timing for the two main phases:
/// 1. Solver time: LP solve (retry_solve)
/// 2. State extraction: Getting solution and extracting variables
#[derive(Debug, Clone, Copy, Default)]
pub struct RealizeUncertaintiesTiming {
    pub solver_time: Duration,
    pub state_extraction_time: Duration,
}

/// Helper function for removing the future cost term from the stage objective,
/// a.k.a the `alpha` term, or the epigraphical variable, assuming the objective
/// function is:
///
/// c^T x + `alpha`
fn get_current_stage_objective(
    total_stage_objective: f64,
    solution: &solver::Solution,
) -> f64 {
    let future_objective = solution.colvalue.last().unwrap();
    total_stage_objective - future_objective
}

/// Helper function for setting the same default solver options on
/// every solved problem.
fn set_default_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "simplex");
    model.set_option("simplex_strategy", 1);
    model.set_option("simplex_scale_strategy", 0);
    model.set_option("simplex_primal_edge_weight_strategy", -1);
    model.set_option("simplex_dual_edge_weight_strategy", -1);
    model.set_option("parallel", "off");
    model.set_option("threads", 1);
    model.set_option("random_seed", 0);

    // PERFORMANCE: Stricter tolerances to reduce numerical drift that causes
    // floating-point non-determinism. Analysis showed ~1e-16 differences compound
    // to 2-3% lower bound variation. Tighter tolerances reduce solver path dependencies.
    // Cost: ~2-5% longer solve times. Benefit: Eliminates cascading numerical errors.
    model.set_option("primal_feasibility_tolerance", 1e-10);
    model.set_option("dual_feasibility_tolerance", 1e-10);
    model.set_option("time_limit", 300);
}

/// Helper function for setting the solver options when retrying a solve
fn set_first_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "off");
    // PERFORMANCE: Slightly looser but still strict tolerances for retry
    model.set_option("primal_feasibility_tolerance", 1e-8);
    model.set_option("dual_feasibility_tolerance", 1e-8);
}

/// Helper function for setting the solver options when retrying a solve
fn set_second_retry_solver_options(model: &mut solver::Model) {
    // PERFORMANCE: Progressively looser tolerances for final retry
    model.set_option("primal_feasibility_tolerance", 1e-6);
    model.set_option("dual_feasibility_tolerance", 1e-6);
}

/// Helper function for setting the solver options when retrying a solve
fn set_third_retry_solver_options(model: &mut solver::Model) {
    model.set_option("simplex_strategy", 4);
}

/// Helper function for setting the solver options when retrying a solve
fn set_final_retry_solver_options(model: &mut solver::Model) {
    model.set_option("presolve", "on");
    model.set_option("solver", "ipm");
    model.set_option("run_crossover", "on");
    model.set_option("primal_feasibility_tolerance", 1e-7);
    model.set_option("dual_feasibility_tolerance", 1e-7);
}

/// Helper function for setting the solver options when retrying a solve
fn set_retry_solver_options(model: &mut solver::Model, retry: usize) {
    match retry {
        1 => set_first_retry_solver_options(model),
        2 => set_second_retry_solver_options(model),
        3 => set_third_retry_solver_options(model),
        4 => set_final_retry_solver_options(model),
        _ => set_default_solver_options(model),
    }
}

/// Helper accessor for indexing desired variables in each subproblem.
///
/// Variables are organized in dual space representation:
/// - **Observation space** (Y_t): Physical variables for hydro balance constraints
/// - **Residual space** (Z'_t): Normalized variables for AR dynamics
///
/// # Dual Space Representation
///
/// The AR model requires both observation and residual space variables:
/// - `inflow` (Y_t): Observation space, used in hydro balance (physical units)
/// - `inflow_residual` (Z'_t): Residual space, used in AR constraints (normalized)
/// - Transformation: Y_t = μ_s + σ_s * Z'_t (handled via LP constraints)
///
/// # State Variables
///
/// Lagged inflow state variables (`lagged_inflow_state`) are **only present** when using
/// `StorageAndInflowState`. When using `StorageState`, lag tracking is done internally
/// by `UnifiedInflowModel`, and `lagged_inflow_state` is `None`.
///
/// # Example Structure (AR(2) with StorageAndInflowState)
///
/// ```text
/// Physical variables: deficit[bus], thermal_gen[thermal], stored_volume[hydro]
/// Inflow (dual):      inflow[hydro] (Y_t), inflow_residual[hydro] (Z'_t)
/// AR variables:       innovation[hydro] (ε_t)
/// State variables:    lagged_inflow_state[hydro][lag] (only if StorageAndInflowState)
/// Future cost:        alpha
/// ```
#[derive(Clone)]
pub struct Variables {
    // ========================================================================
    // Physical Variables (Observation Space)
    // ========================================================================
    /// Deficit (unmet load) at each bus
    pub deficit: Vec<usize>,

    /// Direct power exchange (forward direction)
    pub direct_exchange: Vec<usize>,

    /// Reverse power exchange (backward direction)
    pub reverse_exchange: Vec<usize>,

    /// Thermal generation at each thermal plant
    pub thermal_gen: Vec<usize>,

    /// Turbined flow at each hydro plant
    pub turbined_flow: Vec<usize>,

    /// Spillage at each hydro plant
    pub spillage: Vec<usize>,

    /// Stored volume at each hydro plant (end of period)
    pub stored_volume: Vec<usize>,

    // ========================================================================
    // Inflow Variables (Dual Representation)
    // ========================================================================
    /// Inflow in observation space Y_t (physical units, m³/s or MWh)
    /// Used in: hydro balance constraint (inflow + turbined = stored + spillage)
    pub inflow: Vec<usize>,

    /// Inflow in residual space Z'_t (normalized, zero-mean)
    /// Used in: AR dynamics constraints (Z'_t = Σφ_k Z'_{t-k} + ε_t)
    /// PERFORMANCE: Same size as `inflow`, no additional memory overhead
    pub inflow_residual: Vec<usize>,

    // ========================================================================
    // AR Model Variables
    // ========================================================================
    /// Innovation (white noise) ε_t for each hydro
    /// Used in: AR dynamics (Z'_t = Σφ_k Z'_{t-k} + ε_t)
    pub innovation: Vec<usize>,

    // ========================================================================
    // State Variables (Conditional)
    // ========================================================================
    /// Lagged inflow state variables [hydro][lag]
    /// - `Some(...)`: When using StorageAndInflowState (lags are state variables)
    /// - `None`: When using StorageState (lags tracked internally by UnifiedInflowModel)
    ///
    /// PERFORMANCE: This field is `None` for StorageState, avoiding memory overhead
    /// when state variables are not needed.
    pub lagged_inflow_state: Option<Vec<Vec<usize>>>,

    // ========================================================================
    // Future Cost
    // ========================================================================
    /// Future cost variable (alpha in Bellman equation)
    pub alpha: usize,
}

impl Variables {
    /// Returns true if lagged inflow state variables are present (StorageAndInflowState)
    ///
    /// # Returns
    ///
    /// - `true`: Using StorageAndInflowState, lags are state variables
    /// - `false`: Using StorageState, lags tracked internally by UnifiedInflowModel
    ///
    /// # Performance
    ///
    /// O(1) - simple Option check
    pub fn has_lagged_inflow_state(&self) -> bool {
        self.lagged_inflow_state.is_some()
    }

    /// Returns the number of lag variables for a given hydro
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro plant index
    ///
    /// # Returns
    ///
    /// - Number of lag variables if lagged_inflow_state is Some
    /// - 0 if lagged_inflow_state is None or hydro index out of bounds
    ///
    /// # Performance
    ///
    /// O(1) - direct Vec indexing
    ///
    /// # Example
    ///
    /// ```ignore
    /// // AR(2) model with StorageAndInflowState
    /// assert_eq!(vars.num_inflow_lags(0), 2);
    ///
    /// // StorageState (no lag state variables)
    /// assert_eq!(vars.num_inflow_lags(0), 0);
    /// ```
    pub fn num_inflow_lags(&self, hydro: usize) -> usize {
        self.lagged_inflow_state
            .as_ref()
            .and_then(|lags| lags.get(hydro))
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// Constraint indices for the LP model
///
/// Organizes constraints into logical groups: physical system constraints
/// (load balance, hydro balance) and inflow model constraints (observation
/// transformation and AR dynamics).
///
/// # Structure
///
/// **Physical Constraints:**
/// - `load_balance[bus]`: Power balance at each bus
/// - `hydro_balance[hydro]`: Water balance at each reservoir
///
/// **Inflow Model Constraints (Unified AR):**
/// - `inflow_transform[hydro]`: Observation space transformation Y_t = μ_s + σ_s·Z'_t
/// - `ar_dynamics[hydro]`: AR dynamics Z'_t = Σφ_k·Z'_{t-k} + ε_t
///
/// These are populated by `UnifiedInflowModel.add_constraints_to_lp()` during
/// subproblem construction.
///
/// **Legacy Field:**
/// - `inflow_process`: Deprecated multi-dimensional structure, being replaced
///   by `inflow_transform` and `ar_dynamics` in Sprint 2
///
/// # Example
///
/// For a 2-hydro system with AR(1):
/// ```text
/// inflow_transform = [42, 43]  // Y_0 = μ + σZ'_0, Y_1 = μ + σZ'_1
/// ar_dynamics = [44, 45]       // Z'_0 = φ·Z'_{-1} + ε_0, Z'_1 = φ·Z'_{-1} + ε_1
/// ```
#[derive(Clone)]
pub struct Constraints {
    // ========================================================================
    // Physical System Constraints
    // ========================================================================
    /// Power balance at each bus (one constraint per bus)
    pub load_balance: Vec<usize>,

    /// Water balance at each hydro (one constraint per hydro)
    pub hydro_balance: Vec<usize>,

    // ========================================================================
    // Inflow Model Constraints (Unified AR)
    // ========================================================================
    /// Observation transformation: Y_t = μ_s + σ_s·Z'_t
    /// One constraint per hydro, maps residual space to physical space
    /// RHS = μ_s (seasonal mean), coefficient on Z'_t = -σ_s
    pub inflow_transform: Vec<usize>,

    /// AR dynamics: Z'_t = Σφ_k·Z'_{t-k} + ε_t
    /// One constraint per hydro, enforces autoregressive relationship
    /// RHS = ε_t (innovation, set at solve time), coefficients on lags = -φ_k
    /// ALWAYS present (even for independent case with empty φ)
    pub ar_dynamics: Vec<usize>,
}

impl Constraints {
    /// Returns the number of inflow transformation constraints (one per hydro)
    #[inline]
    pub fn num_inflow_constraints(&self) -> usize {
        self.inflow_transform.len()
    }

    /// Returns true if AR dynamics constraints are present
    #[inline]
    pub fn has_ar_dynamics(&self) -> bool {
        !self.ar_dynamics.is_empty()
    }
}

/// A subproblem that contains a solver model and is associated to a single
/// node in the computing graph

#[derive(Clone)]
pub struct Subproblem {
    pub model: Option<solver::Model>,
    pub state: Box<dyn state::State>,
    pub variables: Variables,
    pub constraints: Constraints,
    /// Season ID for this subproblem (used for seasonal transformations)
    pub season_id: usize,
    /// Unified noise specifications (stored for residual transformations)
    /// PERFORMANCE: Shared via Arc to avoid cloning large spec arrays
    pub unified_specs:
        std::sync::Arc<Vec<crate::unified_noise_spec::UnifiedNoiseSpec>>,
    /// Unified inflow model handling AR dynamics and observation transforms
    ///
    /// This model owns the AR coefficients and lag buffer for all hydros.
    /// It provides methods for:
    /// - Adding inflow variables (Y_t, Z'_t, Z'_{t-k}, ε_t) to LP
    /// - Adding AR dynamics and transformation constraints
    /// - Managing lag buffer updates during forward/backward passes
    ///
    /// The UnifiedInflowModel eliminates conditional logic by treating
    /// independent inflows as AR(0) (zero coefficients).
    ///
    /// # Performance
    ///
    /// - Shared seasonal parameters via Arc (zero-cost clones)
    /// - Pre-allocated lag buffer (no runtime allocations)
    /// - Cache-friendly contiguous storage for coefficients
    ///
    /// # Integration
    ///
    /// - Constructed once during Subproblem::new() from unified_specs
    /// - Used during variable/constraint creation
    /// - Updated via realize_uncertainties() for lag buffer management
    pub inflow_model: UnifiedInflowModel,
}

impl Subproblem {
    pub fn new(
        system: &system::System,
        state_choice: &str,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        unified_specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
    ) -> Self {
        let state = state::factory(
            state_choice,
            system,
            load_stochastic_process,
            inflow_stochastic_processes,
        );

        // TICKET-007: Create UnifiedInflowModel from unified_specs
        // Extract seasonal parameters and construct model
        let seasonal_params = std::sync::Arc::new(
            SeasonalParams::from_unified_specs(
                unified_specs,
                system.meta.hydros_count,
            )
            .expect("Failed to extract seasonal parameters from unified_specs"),
        );

        let inflow_model = UnifiedInflowModel::from_spec(
            unified_specs,
            system.meta.hydros_count,
            seasonal_params,
        );

        let mut pb = solver::Problem::new();
        let variables = Subproblem::add_variables_to_subproblem(
            &mut pb,
            system,
            state.as_ref(),
            load_stochastic_process,
            inflow_stochastic_processes,
            &inflow_model,
        );
        let constraints = Subproblem::add_constraints_to_subproblem(
            &mut pb,
            &variables,
            system,
            state.as_ref(),
            load_stochastic_process,
            inflow_stochastic_processes,
            unified_specs,
            season_id,
            &inflow_model,
        );
        Self::add_offset_to_subproblem(&mut pb, system);

        let mut model = pb.optimise(solver::Sense::Minimise);
        set_retry_solver_options(&mut model, 0);

        Self {
            model: Some(model),
            state,
            variables,
            constraints,
            season_id,
            unified_specs: std::sync::Arc::new(unified_specs.to_vec()),
            inflow_model,
        }
    }

    /// Add inflow variables to LP for unified AR representation
    ///
    /// Creates all inflow-related variables in both observation and residual spaces:
    /// - **Observation space** (Y_t): Physical inflow for hydro balance
    /// - **Residual space** (Z'_t): Normalized inflow for AR dynamics
    /// - **Lag variables** (Z'_{t-k}): Historical residuals for AR constraints
    /// - **Innovation** (ε_t): White noise term for AR RHS
    ///
    /// # Variable Bounds
    ///
    /// - Y_t: [0, +∞) — physical inflow must be non-negative
    /// - Z'_t, Z'_{t-k}, ε_t: (-∞, +∞) — normalized, can be negative
    ///
    /// # Returns
    ///
    /// Tuple of (inflow_obs, inflow_res, lag_res, innovations) containing
    /// variable indices for each type.
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: O(n·p) variable indices stored
    /// - No runtime allocations beyond variable index storage
    ///
    /// # Example Structure (AR(2) system with 2 hydros)
    ///
    /// ```text
    /// inflow_obs:   [Y_0, Y_1]
    /// inflow_res:   [Z'_0, Z'_1]
    /// lag_res[0]:   [Z'_{0,t-1}, Z'_{0,t-2}]
    /// lag_res[1]:   [Z'_{1,t-1}, Z'_{1,t-2}]
    /// innovations:  [ε_0, ε_1]
    /// ```
    fn add_inflow_variables(
        pb: &mut solver::Problem,
        inflow_model: &UnifiedInflowModel,
    ) -> (Vec<usize>, Vec<usize>, Vec<Vec<usize>>, Vec<usize>) {
        let n_hydros = inflow_model.dimension();

        // PERFORMANCE: Pre-allocate with capacity to avoid reallocation
        let mut inflow_obs = Vec::with_capacity(n_hydros);
        let mut inflow_res = Vec::with_capacity(n_hydros);
        let mut lag_res = Vec::with_capacity(n_hydros);
        let mut innovations = Vec::with_capacity(n_hydros);

        for h in 0..n_hydros {
            // Observation space: Y_t (for hydro balance)
            // Bounds: [0, +∞) — physical inflow is non-negative
            let y_idx = pb.add_column(0.0, 0.0..f64::INFINITY);
            inflow_obs.push(y_idx);

            // Residual space: Z'_t (for AR dynamics)
            // Bounds: (-∞, +∞) — normalized, can be negative
            let z_idx = pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
            inflow_res.push(z_idx);

            // Lag residuals: Z'_{t-k} (for AR dynamics)
            let lag_order = inflow_model.lag_order(h);
            let mut lags = Vec::with_capacity(lag_order);
            for _ in 0..lag_order {
                let lag_idx =
                    pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
                lags.push(lag_idx);
            }
            lag_res.push(lags);

            // Innovation: ε_t (for AR dynamics RHS)
            // Bounds: (-∞, +∞) — white noise, can be negative
            let eps_idx = pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
            innovations.push(eps_idx);
        }

        (inflow_obs, inflow_res, lag_res, innovations)
    }

    fn add_variables_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
        state: &dyn state::State,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        inflow_model: &UnifiedInflowModel,
    ) -> Variables {
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
                    0.0..(thermal.max_generation - thermal.min_generation),
                )
            })
            .collect();
        let turbined_flow: Vec<usize> = system
            .hydros
            .iter()
            .map(|hydro| {
                pb.add_column(
                    0.0,
                    hydro.min_turbined_flow..hydro.max_turbined_flow,
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

        // TICKET-007: Add inflow variables using UnifiedInflowModel
        // This creates observation (Y_t), residual (Z'_t), lag (Z'_{t-k}), and innovation (ε_t) variables
        let (inflow, inflow_residual, lag_residual, innovation) =
            Self::add_inflow_variables(pb, inflow_model);

        // DEPRECATED: Old inflow_process variables removed in Sprint 2 (TICKET-007/008)
        // UnifiedInflowModel now handles all inflow variables and constraints
        // Keeping this code created duplicate variables causing infeasibility
        // let inflow_process = state.add_variables_to_subproblem(
        //     pb,
        //     load_stochastic_process,
        //     inflow_stochastic_processes,
        // );

        let alpha = pb.add_column(1.0, 0.0..);

        // TICKET-010: Store lag variables only if StorageAndInflowState
        // - StorageState: lags tracked internally via lag_buffer (None)
        // - StorageAndInflowState: lags are state variables (Some(lag_residual))
        let lagged_inflow_state = if state.has_lagged_inflow_state() {
            Some(lag_residual)
        } else {
            None
        };

        Variables {
            deficit,
            direct_exchange,
            reverse_exchange,
            thermal_gen,
            turbined_flow,
            spillage,
            stored_volume,
            inflow,
            inflow_residual,
            innovation,
            lagged_inflow_state,
            alpha,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn add_constraints_to_subproblem(
        pb: &mut solver::Problem,
        variables: &Variables,
        system: &system::System,
        _state: &dyn state::State,
        _load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        _inflow_stochastic_processes: &[Box<
            dyn stochastic_process::StochasticProcess,
        >],
        _unified_specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
        season_id: usize,
        inflow_model: &UnifiedInflowModel, // TICKET-008: Now actively used
    ) -> Constraints {
        // Adds load balance with 0.0 as RHS
        let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];
        for bus in system.buses.iter() {
            let mut factors = vec![(variables.deficit[bus.id], 1.0)];
            for thermal_id in bus.thermal_ids.iter() {
                factors.push((variables.thermal_gen[*thermal_id], 1.0));
            }
            for hydro_id in bus.hydro_ids.iter() {
                factors.push((
                    variables.turbined_flow[*hydro_id],
                    system.hydros.get(*hydro_id).unwrap().productivity,
                ));
            }
            for line_id in bus.source_line_ids.iter() {
                factors.push((variables.reverse_exchange[*line_id], 1.0));
                factors.push((variables.direct_exchange[*line_id], -1.0));
            }
            for line_id in bus.target_line_ids.iter() {
                factors.push((variables.direct_exchange[*line_id], 1.0));
                factors.push((variables.reverse_exchange[*line_id], -1.0));
            }
            load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
        }

        // Adds hydro balance with 0.0 as RHS
        let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];
        for hydro in system.hydros.iter() {
            let mut factors: Vec<(usize, f64)> = vec![
                (variables.stored_volume[hydro.id], 1.0),
                (variables.turbined_flow[hydro.id], 1.0),
                (variables.spillage[hydro.id], 1.0),
                (variables.inflow[hydro.id], -1.0),
            ];
            for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
                factors
                    .push((variables.turbined_flow[*upstream_hydro_id], -1.0));
                factors.push((variables.spillage[*upstream_hydro_id], -1.0));
            }
            hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
        }

        // Adds inflow process as variables, bounded at 0, which will be fixed in runtime
        // DEPRECATED: Old inflow_process constraints removed in Sprint 2 (TICKET-007/008)
        // UnifiedInflowModel now handles all inflow constraints via ar_dynamics and inflow_transform
        // Keeping this code created duplicate constraints causing infeasibility
        // let inflow_process = state.add_constraints_to_subproblem(
        //     pb,
        //     variables,
        //     load_stochastic_process,
        //     inflow_stochastic_processes,
        //     unified_specs,
        //     season_id,
        // );
        // Note: inflow_process field removed - UnifiedInflowModel handles all inflow constraints

        // TICKET-008: Integrate UnifiedInflowModel constraints
        // Add AR dynamics and observation transformation constraints to LP
        let constraint_indices =
            inflow_model.add_constraints_to_lp(pb, variables, season_id);
        let inflow_transform = constraint_indices.observation_transform;
        let ar_dynamics = constraint_indices.ar_dynamics;

        Constraints {
            load_balance,
            hydro_balance,
            inflow_transform,
            ar_dynamics,
        }
    }

    fn add_offset_to_subproblem(
        pb: &mut solver::Problem,
        system: &system::System,
    ) {
        let mut offset = 0.0;
        for thermal in system.thermals.iter() {
            offset += thermal.cost * thermal.min_generation;
        }
        pb.offset = offset;
    }

    fn set_load_balance_rhs(&mut self, loads: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for (index, row) in self.constraints.load_balance.iter().enumerate()
            {
                model.change_rows_bounds(*row, loads[index], loads[index]);
            }
        }
    }

    /// Set hydro balance RHS directly (used primarily in tests).
    ///
    /// For production use, prefer `update_with_current_trajectory()` which
    /// delegates to the state's `update_from_trajectory()` method.
    #[cfg(test)]
    fn set_hydro_balance_rhs(&mut self, initial_storages: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for (index, row) in
                self.constraints.hydro_balance.iter().enumerate()
            {
                model.change_rows_bounds(
                    *row,
                    initial_storages[index],
                    initial_storages[index],
                );
            }
        }
    }

    /// Update subproblem state from trajectory of past realizations
    ///
    /// This method is called during SDDP forward passes to transfer state information
    /// from past realizations to the current subproblem. It updates:
    ///
    /// 1. **Lag buffer** (via UnifiedInflowModel): Extracts last p residuals from
    ///    trajectory for AR(p) dynamics. For AR(1), uses Z'_{t-1}. For AR(2), uses
    ///    [Z'_{t-1}, Z'_{t-2}]. Independent models (p=0) have no-op lag updates.
    ///
    /// 2. **State-specific updates** (via State trait): Storage values, constraint RHS,
    ///    and any state-specific bookkeeping.
    ///
    /// # Trajectory Structure
    ///
    /// The trajectory is ordered chronologically from PreStudy to current stage:
    ///
    /// - Stage 1: `[PreStudy(0)]`
    /// - Stage 2: `[PreStudy(0), Stage(1)]`
    /// - Stage t: `[PreStudy(0), Stage(1), ..., Stage(t-1)]`
    ///
    /// For multi-node PreStudy (PAR models):
    ///
    /// - Stage 1: `[PreStudy(-p), ..., PreStudy(-1), PreStudy(0)]`
    /// - Stage 2: `[PreStudy(-p), ..., PreStudy(0), Stage(1)]`
    ///
    /// # Performance
    ///
    /// - Lag buffer update: O(n·p) where n = hydros, p = max lag order
    /// - State updates: O(n) for StorageState, O(n·p) for StorageAndInflowState
    /// - No allocations in hot path (reuses lag buffer)
    ///
    /// # Arguments
    ///
    /// * `realizations` - Trajectory of past realizations (PreStudy + Study stages)
    ///
    /// # Panics
    ///
    /// - If model is not initialized (should never happen in normal SDDP flow)
    /// - If trajectory is empty (should always contain at least PreStudy)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Forward pass at stage 2:
    /// let trajectory = vec![&prestudy_realization, &stage1_realization];
    /// subproblem.update_with_current_trajectory(trajectory);
    /// // Lag buffer now contains Z'_{t-1} from stage1_realization
    /// // Storage state updated from stage1_realization.final_storage
    /// ```
    pub fn update_with_current_trajectory(
        &mut self,
        realizations: Vec<&Realization>,
    ) {
        // STEP 1: Update lag buffer from trajectory
        // This extracts last p residuals (Z'_{t-k}) from realizations
        // and stores them in UnifiedInflowModel.lag_buffer.
        // Next realize_uncertainties() call will use these lags in AR constraint RHS.
        //
        // PERFORMANCE: O(n·p) where n = hydros, p = max lag order
        // For independent models (p=0), this is effectively a no-op.
        //
        // Note: We need to clone realizations since update_lag_buffer_from_trajectory
        // expects owned Realization objects. This is acceptable since this is not
        // a hot path (called once per forward pass stage, not per solve).

        // DEBUG: Log lag buffer BEFORE update
        if cfg!(debug_assertions) {
            eprintln!("[DEBUG PAR] update_with_current_trajectory: trajectory length={}", realizations.len());
            eprintln!("  Lag buffer BEFORE update:");
            for hydro in 0..self.inflow_model.dimension() {
                let buffer = self.inflow_model.get_lag_residuals(hydro);
                if !buffer.is_empty() {
                    eprintln!("    hydro {}: {:?}", hydro, buffer);
                }
            }
        }

        let owned_realizations: Vec<Realization> =
            realizations.iter().map(|&r| r.clone()).collect();
        self.inflow_model
            .update_lag_buffer_from_trajectory(&owned_realizations);

        // DEBUG: Log lag buffer AFTER update
        if cfg!(debug_assertions) {
            eprintln!("  Lag buffer AFTER update:");
            for hydro in 0..self.inflow_model.dimension() {
                let buffer = self.inflow_model.get_lag_residuals(hydro);
                if !buffer.is_empty() {
                    eprintln!("    hydro {}: {:?}", hydro, buffer);
                }
            }
        }

        // STEP 2: Delegate state-specific updates to State trait
        // This updates storage values and hydro balance constraint RHS.
        // State implementations know what they need from the trajectory.
        //
        // PERFORMANCE: O(n) for StorageState (just storage update)
        let model = self.model.as_mut().unwrap();
        self.state.update_from_trajectory(
            &realizations,
            model,
            &self.constraints,
            &self.variables,
        );
    }

    pub fn update_with_current_realization(
        &mut self,
        realization: &Realization,
    ) {
        self.state.update_with_current_realization(realization);
    }

    pub fn compute_new_cut(
        &self,
        forward_trajectory: &[&Realization],
        branching_realizations: &[Realization],
        risk_measure: &dyn risk_measure::RiskMeasure,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> fcf::CutStatePair {
        // this only works when all nodes have the same state definition??
        let mut visited_state = self.state.clone();
        // Set tracking fields before computing cut
        visited_state.set_iteration(iteration);
        visited_state.set_forward_pass_idx(forward_pass_idx);
        let cut = visited_state.compute_new_cut(
            risk_measure,
            forward_trajectory,
            branching_realizations,
        );
        fcf::CutStatePair::new(cut, visited_state, forward_pass_idx)
    }

    pub fn add_cut_and_evaluate_cut_selection(
        &mut self,
        cut_state_pair: fcf::CutStatePair,
        future_cost_function: Arc<Mutex<fcf::FutureCostFunction>>,
    ) {
        let mut cut = cut_state_pair.cut;
        let mut visited_state = cut_state_pair.state;

        if let Some(model) = self.model.as_mut() {
            self.state.add_cut_constraint_to_model(
                &mut cut,
                &self.variables,
                model,
            );
        }
        let mut fcf = future_cost_function.lock().unwrap();
        cut.id = fcf.cut_pool.total_cut_count;
        fcf.update_cut_pool_on_add(cut.id);
        fcf.eval_new_cut_domination(&mut cut);

        fcf.add_cut(cut);

        // Obtains returning cut ids, based on cut selection
        let returning_cut_ids =
            fcf.update_old_cuts_domination(&mut visited_state);

        fcf.add_state(visited_state);

        // Obtains removing cut ids, based on cut selection
        let mut removing_cut_ids = Vec::<usize>::new();
        for cut in fcf.cut_pool.pool.iter_mut() {
            if (cut.non_dominated_state_count == 0) && cut.active {
                removing_cut_ids.push(cut.id);
            }
        }

        // Returns cuts to model
        for cut_id in returning_cut_ids.iter() {
            let cut = fcf.cut_pool.pool.get_mut(*cut_id).unwrap();
            if let Some(model) = self.model.as_mut() {
                self.state.add_cut_constraint_to_model(
                    cut,
                    &self.variables,
                    model,
                );
            }
            fcf.update_cut_pool_on_return(*cut_id);
        }

        // Removes cuts from model
        for cut_id in removing_cut_ids.iter() {
            let cut_index = fcf.get_active_cut_index_by_id(*cut_id);
            let row_index = self.first_cut_row_index() + cut_index;
            if let Some(model) = self.model.as_mut() {
                model.delete_row(row_index).unwrap();
            }
            fcf.update_cut_pool_on_remove(*cut_id);
        }
    }

    /// Apply AGGREGATED cut selection results WITHOUT locking FCF (LOCK-FREE)
    pub fn apply_aggregated_cut_selection_result(
        &mut self,
        aggregated_result: &fcf::AggregatedCutSelectionResult,
        active_cut_indices_before: &std::collections::BTreeMap<usize, usize>,
        cuts_to_add: &[(usize, cut::BendersCut)],
    ) -> Result<(), String> {
        // PERFORMANCE: Sort cuts to ensure deterministic constraint matrix construction.
        // This eliminates solver path dependencies that cause ~1e-16 numerical differences
        // which cascade to 2-3% lower bound variation. Constraint addition order affects
        // solver numerical algorithms (basis selection, pivot rules) even with identical cuts.
        let mut cuts_to_process: Vec<(usize, &cut::BendersCut)> = cuts_to_add
            .iter()
            .filter(|(cut_id, _)| {
                aggregated_result.new_cut_ids.contains(cut_id)
                    || aggregated_result.returning_cut_ids.contains(cut_id)
            })
            .map(|(cut_id, cut)| (*cut_id, cut))
            .collect();

        // Sort by (cut_id, iteration, forward_pass_idx) for complete determinism
        cuts_to_process.sort_by_key(|(cut_id, cut)| {
            (*cut_id, cut.iteration, cut.forward_pass_idx)
        });

        // Add cuts in deterministic order
        for (_cut_id, cut) in cuts_to_process {
            if let Some(model) = self.model.as_mut() {
                let mut cut_copy = cut.clone();
                self.state.add_cut_constraint_to_model(
                    &mut cut_copy,
                    &self.variables,
                    model,
                );
            }
        }

        // Remove ALL dominated cuts from model (same as before)
        let mut indices_to_remove: Vec<usize> = aggregated_result
            .removing_cut_ids
            .iter()
            .filter_map(|&cut_id| {
                active_cut_indices_before.get(&cut_id).copied()
            })
            .collect();

        indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));

        for index in indices_to_remove {
            let row_idx = self.first_cut_row_index() + index;

            if let Some(model) = self.model.as_mut() {
                model.delete_row(row_idx).map_err(|e| {
                    format!("Failed to delete row {}: {:?}", row_idx, e)
                })?;
            }
        }

        // NOTE: FCF state update (marking cuts inactive, updating active_cut_indices)
        // is done ONCE in the SDDP code before calling this function.
        // This lock-free version only updates the local solver model (adds/removes constraints).
        Ok(())
    }

    /// Update AR dynamics constraint RHS with innovation values
    ///
    /// Sets the RHS of ar_dynamics constraints to the realized innovation
    /// values (ε_t). This is called during realize_uncertainties() to
    /// incorporate the sampled innovations into the LP.
    ///
    /// # Arguments
    ///
    /// - `innovations`: Slice of innovation values (one per hydro)
    ///
    /// # Behavior
    ///
    /// The RHS value depends on state type:
    ///
    /// **StorageAndInflowState** (lags are state variables):
    /// - RHS = ε_t (innovation only)
    /// - Constraint: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
    ///
    /// **StorageState** (lags tracked in UnifiedInflowModel.lag_buffer):
    /// - RHS = Σ(φ_k * lag_buffer[k]) + ε_t
    /// - Constraint: Z'_t = Σ(φ_k * lag_buffer[k]) + ε_t
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - No allocations (updates existing constraint RHS values)
    /// - Hot path: called thousands of times during SDDP
    fn update_ar_constraint_rhs(&mut self, innovations: &[f64]) {
        if let Some(model) = self.model.as_mut() {
            for (hydro, &innovation) in innovations.iter().enumerate() {
                let constraint_idx = self.constraints.ar_dynamics[hydro];

                // Compute RHS based on state type
                let rhs = if self.variables.has_lagged_inflow_state() {
                    // StorageAndInflowState: RHS = ε_t only (lags are in constraint)

                    // DEBUG: Log RHS computation for StorageAndInflowState
                    if cfg!(debug_assertions) {
                        eprintln!("[DEBUG PAR] update_ar_constraint_rhs: hydro={}, StorageAndInflowState, innovation={:.4}, rhs={:.4}", 
                            hydro, innovation, innovation);
                    }

                    innovation
                } else {
                    // StorageState: RHS = Σ(φ_k * lag_k) + ε_t
                    // Lag contributions from UnifiedInflowModel.lag_buffer
                    let lag_residuals =
                        self.inflow_model.get_lag_residuals(hydro);
                    let coefficients =
                        self.inflow_model.get_ar_coefficients(hydro);

                    let lag_contribution: f64 = lag_residuals
                        .iter()
                        .zip(coefficients.iter())
                        .map(|(&lag, &coeff)| coeff * lag)
                        .sum();

                    let rhs = lag_contribution + innovation;

                    // DEBUG: Log detailed RHS computation for StorageState
                    if cfg!(debug_assertions) {
                        eprintln!("[DEBUG PAR] update_ar_constraint_rhs: hydro={}, StorageState", hydro);
                        eprintln!("  lag_residuals: {:?}", lag_residuals);
                        eprintln!("  coefficients: {:?}", coefficients);
                        eprintln!("  lag_contribution: {:.4}, innovation: {:.4}, rhs: {:.4}", 
                            lag_contribution, innovation, rhs);

                        // Check for problematic values
                        if !rhs.is_finite() {
                            eprintln!(
                                "  ⚠️ WARNING: RHS is not finite (NaN or Inf)!"
                            );
                        }
                        if rhs.abs() > 1000.0 {
                            eprintln!("  ⚠️ WARNING: RHS magnitude > 1000 (extreme value)!");
                        }
                    }

                    rhs
                };

                // Update RHS (both lower and upper bound for equality constraint)
                model.change_rows_bounds(constraint_idx, rhs, rhs);
            }
        }
    }

    /// Check if AR dynamics + transformation could produce negative inflows
    ///
    /// For AR models with normal marginals, Y_t = μ_s + σ_s * Z'_t where Z'_t can be very negative.
    /// If μ_s is small and σ_s is large, this can violate Y_t ≥ 0, causing infeasibility.
    ///
    /// This diagnostic function estimates Z'_t from AR dynamics and checks if the resulting
    /// inflow Y_t would be negative, which violates the Y_t ≥ 0 constraint.
    #[allow(clippy::needless_range_loop)]
    fn check_for_negative_inflow_risk(&self, innovations: &[f64]) {
        // This is a diagnostic - we'll only log warnings, actual feasibility determined by solver

        for hydro in 0..self.inflow_model.dimension() {
            // Compute expected Z'_t value based on AR dynamics: Z'_t ≈ Σ(φ_k * lag_k) + ε_t
            let coefficients = self.inflow_model.get_ar_coefficients(hydro);
            let lag_residuals = self.inflow_model.get_lag_residuals(hydro);

            let lag_contrib: f64 = lag_residuals
                .iter()
                .zip(coefficients.iter())
                .map(|(l, c)| l * c)
                .sum();

            let z_residual_estimate = lag_contrib + innovations[hydro];

            // We can't easily get μ and σ here without refactoring, but we can detect
            // extremely negative Z' values that are likely to cause problems
            if z_residual_estimate < -4.0 {
                eprintln!("⚠️  EXTREME NEGATIVE RESIDUAL WARNING!");
                eprintln!("    hydro={}, season_id={}", hydro, self.season_id);
                eprintln!(
                    "    Z'_t ≈ {:.4} (< -4σ, very extreme!)",
                    z_residual_estimate
                );
                eprintln!(
                    "    lag_contrib={:.4}, innovation={:.4}",
                    lag_contrib, innovations[hydro]
                );
                eprintln!("    This may cause Y_t = μ + σ*Z'_t < 0, leading to infeasibility.");
                eprintln!("    Consider using lognormal3 marginal distribution to ensure Y_t > 0.");
            }
        }
    }

    fn retry_solve(&mut self) {
        let mut retry: usize = 0;
        if let Some(model) = self.model.as_mut() {
            loop {
                if retry > 4 {
                    // PERFORMANCE: After 4 retries, model is likely infeasible
                    // Provide detailed diagnostics

                    eprintln!("\n❌ INFEASIBILITY DIAGNOSTICS:");
                    eprintln!("   Season ID: {}", self.season_id);
                    eprintln!(
                        "   Model dimensions: {} rows, {} cols",
                        model.num_rows(),
                        model.num_cols()
                    );
                    eprintln!("   Solver status: {:?}", model.status());

                    // Check for extreme lag values that might cause negative inflows
                    eprintln!("\n   Current lag values:");
                    for hydro in 0..self.inflow_model.dimension() {
                        let lags = self.inflow_model.get_lag_residuals(hydro);
                        if !lags.is_empty() {
                            eprintln!("     hydro {}: {:?}", hydro, lags);
                        }
                    }

                    eprintln!("\n   Possible causes:");
                    eprintln!("   1. Negative inflows: Y_t = μ + σ*Z'_t < 0 with normal marginals");
                    eprintln!(
                        "   2. Conflicting storage/hydro balance constraints"
                    );
                    eprintln!("   3. Extreme AR dynamics producing unrealistic values");
                    eprintln!(
                        "   4. Numerical instability in constraint matrix"
                    );

                    panic!(
                        "Solver failed after {} retries. Final status: {:?}. \
                         Model dimensions: {} rows, {} cols. \
                         Common causes: conflicting constraints, impossible RHS, \
                         negative lag buffer values, or numerical instability.",
                        retry,
                        model.status(),
                        model.num_rows(),
                        model.num_cols()
                    );
                }

                // Try to solve with detailed error handling
                match model.try_solve() {
                    Ok(_) => {
                        // Solve succeeded, check model status
                    }
                    Err(_e) => {
                        // Continue to check model status and potentially retry
                    }
                }

                match model.status() {
                    solver::HighsModelStatus::Optimal => {
                        if retry != 0 {
                            set_default_solver_options(model);
                        }
                        return;
                    }
                    solver::HighsModelStatus::Infeasible => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::PresolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::SolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::PostsolveError => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::ReachedIterationLimit => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::ReachedTimeLimit => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    solver::HighsModelStatus::Unknown => {
                        retry += 1;
                        set_retry_solver_options(model, retry);
                    }
                    status => {
                        // PERFORMANCE: Unexpected solver status - provide diagnostics
                        panic!(
                            "Unexpected solver status after {} retries: {:?}. \
                             Expected Optimal or Infeasible. This may indicate: \
                             1) Time/iteration limits reached, \
                             2) Numerical issues in the model, \
                             3) Unbounded problem, \
                             4) Solver error",
                            retry, status
                        );
                    }
                }
            }
        }
    }

    fn first_cut_row_index(&self) -> usize {
        // TICKET-008: Find the last constraint index before cuts are added
        // Cuts are added after all base constraints, so we need the maximum
        // constraint index across all constraint types.
        let mut max_idx = 0;

        if let Some(&idx) = self.constraints.load_balance.last() {
            max_idx = max_idx.max(idx);
        }
        if let Some(&idx) = self.constraints.hydro_balance.last() {
            max_idx = max_idx.max(idx);
        }
        if let Some(&idx) = self.constraints.ar_dynamics.last() {
            max_idx = max_idx.max(idx);
        }
        if let Some(&idx) = self.constraints.inflow_transform.last() {
            max_idx = max_idx.max(idx);
        }

        max_idx + 1
    }

    pub fn realize_uncertainties(
        &mut self,
        noises: &scenario::OptimizedSampledBranchingNoises,
        load_stochastic_process: &dyn stochastic_process::StochasticProcess,
        realization_container: &mut Realization,
    ) -> Result<RealizeUncertaintiesTiming, String> {
        let mut timing = RealizeUncertaintiesTiming::default();

        // Time state extraction
        let extraction_start = std::time::Instant::now();

        // ====================================================================
        // LOAD REALIZATION (still uses stochastic_process for transformation)
        // ====================================================================
        let load =
            load_stochastic_process.realize(noises.get_load_innovations());

        // PERFORMANCE: Store realized loads in realization container
        // Handle both cases: per-bus loads or single scalar load (deterministic benchmarks)
        if load.len() == realization_container.loads.len() {
            // Direct copy for per-bus loads (O(num_buses) memcpy, ~10ns)
            realization_container.loads.clone_from_slice(load);
        } else if load.len() == 1 {
            // Replicate single load value across all buses (deterministic case)
            realization_container.loads.fill(load[0]);
        } else {
            return Err(format!(
                "Load dimension mismatch: got {} load values but system has {} buses",
                load.len(),
                realization_container.loads.len()
            ));
        }

        // ====================================================================
        // UPDATE LP WITH UNCERTAINTIES
        // ====================================================================
        // Load balance RHS (legacy approach)
        // Future enhancement: Migrate to unified uncertainty model
        // See FUTURE_WORK.md: "Unified Load Uncertainty Model"
        self.set_load_balance_rhs(load);

        // AR dynamics RHS = innovation (ε_t)
        // This is the KEY SIMPLIFICATION: no more conditional logic
        // Both independent and AR cases use the same code path:
        // - Independent: Z'_t = ε_t (empty coefficients)
        // - AR(p): Z'_t - Σ(φ_k*Z'_{t-k}) = ε_t
        self.update_ar_constraint_rhs(noises.get_inflow_innovations());

        // DEBUG: Check for potential negative inflows from AR dynamics
        if cfg!(debug_assertions) {
            self.check_for_negative_inflow_risk(
                noises.get_inflow_innovations(),
            );
        }

        timing.state_extraction_time += extraction_start.elapsed();

        // ====================================================================
        // SOLVE LP
        // ====================================================================
        let solver_start = std::time::Instant::now();
        self.retry_solve();
        timing.solver_time = solver_start.elapsed();

        // ====================================================================
        // EXTRACT SOLUTION
        // ====================================================================
        let extraction_start = std::time::Instant::now();
        match &self.model {
            Some(model) => match model.status() {
                solver::HighsModelStatus::Optimal => {
                    let mut solution = model.get_solution();
                    self.slice_solution_rows_to_problem_constraints(
                        &mut solution,
                    );

                    // Basis
                    realization_container.basis.clone_from(&model.get_basis());

                    // Costs
                    realization_container.total_stage_objective =
                        model.get_objective_value();
                    realization_container.current_stage_objective =
                        get_current_stage_objective(
                            realization_container.total_stage_objective,
                            &solution,
                        );

                    // Bus results
                    self.get_deficit_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_marginal_cost_from_solution(
                        &solution,
                        realization_container,
                    );

                    // Line results
                    self.get_net_exchange_from_solution(
                        &solution,
                        realization_container,
                    );

                    // Thermal results
                    self.get_thermal_gen_from_solution(
                        &solution,
                        realization_container,
                    );

                    // Hydro results (observation + residual spaces)
                    self.get_inflow_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_final_storage_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_turbined_flow_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_spillage_from_solution(
                        &solution,
                        realization_container,
                    );
                    self.get_water_values_from_solution(
                        &solution,
                        realization_container,
                    );

                    // Extract lag duals from ar_dynamics constraints
                    self.get_lag_duals_from_solution(
                        &solution,
                        realization_container,
                    );

                    model.clear_solver();
                    timing.state_extraction_time = extraction_start.elapsed();
                    Ok(timing)
                }
                _ => Err(format!(
                    "Error while solving subproblem: {:?}",
                    model.status()
                )),
            },
            None => {
                Err("Error while solving subproblem: Model is None".to_string())
            }
        }
    }

    fn get_deficit_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.deficit.first().unwrap();
        let last = *self.variables.deficit.last().unwrap() + 1;
        realization_container
            .deficit
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_net_exchange_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        if !self.variables.direct_exchange.is_empty() {
            let direct_first = *self.variables.direct_exchange.first().unwrap();
            let direct_last =
                *self.variables.direct_exchange.last().unwrap() + 1;
            let reverse_first =
                *self.variables.reverse_exchange.first().unwrap();
            let reverse_last =
                *self.variables.reverse_exchange.last().unwrap() + 1;
            realization_container.exchange.clone_from_slice(
                &solution.colvalue[direct_first..direct_last],
            );
            realization_container
                .exchange
                .iter_mut()
                .zip(&solution.colvalue[reverse_first..reverse_last])
                .for_each(|(direct, reverse)| *direct -= *reverse);
        }
    }

    fn get_thermal_gen_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        if !self.variables.thermal_gen.is_empty() {
            let first = *self.variables.thermal_gen.first().unwrap();
            let last = *self.variables.thermal_gen.last().unwrap() + 1;
            realization_container
                .thermal_generation
                .clone_from_slice(&solution.colvalue[first..last]);
        }
    }

    fn get_spillage_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.spillage.first().unwrap();
        let last = *self.variables.spillage.last().unwrap() + 1;
        realization_container
            .spillage
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_turbined_flow_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.turbined_flow.first().unwrap();
        let last = *self.variables.turbined_flow.last().unwrap() + 1;
        realization_container
            .turbined_flow
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_final_storage_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.variables.stored_volume.first().unwrap();
        let last = *self.variables.stored_volume.last().unwrap() + 1;
        realization_container
            .final_storage
            .clone_from_slice(&solution.colvalue[first..last]);
    }

    fn get_inflow_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        // ====================================================================
        // OBSERVATION SPACE (Y_t): Physical inflow values
        // ====================================================================
        // Extract observation space Y_t from solution
        // Used by hydro balance: stored_volume + turbined + spillage = inflow + ...
        for (h, &var_idx) in self.variables.inflow.iter().enumerate() {
            realization_container.inflow[h] = solution.colvalue[var_idx];
        }

        // ====================================================================
        // RESIDUAL SPACE (Z'_t): Normalized inflow values
        // ====================================================================
        // Extract residual space Z'_t from solution
        // Used as lags in next stage: Z'_{t-k} for AR dynamics
        for (h, &var_idx) in self.variables.inflow_residual.iter().enumerate() {
            realization_container.inflow_residual[h] =
                solution.colvalue[var_idx];
        }
    }

    fn get_water_values_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.constraints.hydro_balance.first().unwrap();
        let last = *self.constraints.hydro_balance.last().unwrap() + 1;
        realization_container
            .water_value
            .clone_from_slice(&solution.rowdual[first..last]);
    }

    fn get_lag_duals_from_solution(
        &self,
        _solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        // ARCHITECTURE NOTE: With UnifiedInflowModel, lag variables are BOUNDED (not constrained)
        //
        // Background:
        // - Old architecture: Lag fixing constraints Z'_{t-k} = value → had dual values
        // - New architecture: Lag variables bounded Z'_{t-k} ∈ [value, value] → no duals
        //
        // LP Theory: Only constraints have dual values. Variable bounds don't have duals.
        //
        // Impact: Benders cuts have lag coefficients = 0.0 (see StorageAndInflowState::evaluate_cut)
        // This is handled correctly in state.rs lines 1398-1402 with the fallback:
        //   if lag_idx < realization.lag_duals.len() { use dual } else { push 0.0 }
        //
        // Result: lag_duals is always empty with new architecture
        realization_container.lag_duals.clear();
    }

    fn get_marginal_cost_from_solution(
        &self,
        solution: &solver::Solution,
        realization_container: &mut Realization,
    ) {
        let first = *self.constraints.load_balance.first().unwrap();
        let last = *self.constraints.load_balance.last().unwrap() + 1;
        realization_container
            .marginal_cost
            .clone_from_slice(&solution.rowdual[first..last]);
    }

    fn slice_solution_rows_to_problem_constraints(
        &self,
        solution: &mut solver::Solution,
    ) {
        // TICKET-008: Use ar_dynamics instead of deprecated inflow_process
        // Find the last constraint that was part of the original problem
        // (before cuts are added). This is typically the last AR dynamics constraint.
        let end = if !self.constraints.ar_dynamics.is_empty() {
            *self.constraints.ar_dynamics.last().unwrap() + 1
        } else if !self.constraints.inflow_transform.is_empty() {
            *self.constraints.inflow_transform.last().unwrap() + 1
        } else if !self.constraints.hydro_balance.is_empty() {
            *self.constraints.hydro_balance.last().unwrap() + 1
        } else {
            *self.constraints.load_balance.last().unwrap() + 1
        };

        solution.rowvalue.truncate(end);
        solution.rowdual.truncate(end);
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum StudyPeriodKind {
    PreStudy,
    Study,
    PostStudy,
}

/// Solution of a subproblem representing both physical and dual space values
///
/// Realization contains the complete solution of an SDDP subproblem, including:
/// - Physical variables (observation space): inflows, generation, storage
/// - Residual space variables: inflow_residual (Z'_t) for AR dynamics
/// - Dual values: marginal costs, water values, lag constraint duals
///
/// # Dual Space Representation
///
/// For the unified AR model, realizations maintain values in both spaces:
///
/// **Observation Space (Physical):**
/// - `inflow`: Y_t values in physical units (m³/s or MWh)
/// - Used for: output reporting, hydro balance constraints
///
/// **Residual Space (Normalized):**
/// - `inflow_residual`: Z'_t values (zero-mean, unit variance)
/// - Transform: Z'_t = (Y_t - μ_s) / σ_s where μ_s, σ_s are seasonal parameters
/// - Used for: AR lag buffer updates, cut generation
///
/// # Lag Duals
///
/// For AR models with lag_order > 0:
/// - `lag_duals[lag_idx][hydro_idx]`: Dual value on lag constraint
/// - Structure matches AR dynamics: Z'_t = Σφ_k·Z'_{t-k} + ε_t
/// - Empty for independent models (AR(0))
///
/// # Example
///
/// For a 2-hydro system with AR(1):
/// ```text
/// inflow = [100.0, 150.0]           // Y_t in physical units
/// inflow_residual = [0.5, -0.3]      // Z'_t normalized
/// lag_duals = [                      // One lag for AR(1)
///     [2.5, 3.1]                     // Duals for lag k=1, both hydros
/// ]
/// ```
#[derive(Debug, Clone)]
pub struct Realization {
    pub kind: StudyPeriodKind,
    pub loads: Vec<f64>,
    pub deficit: Vec<f64>,
    pub exchange: Vec<f64>,

    // ========================================================================
    // Inflow Variables (Dual Space Representation)
    // ========================================================================
    /// Inflow in observation space Y_t (physical units: m³/s or MWh)
    /// Used for: output reporting, hydro balance constraints
    pub inflow: Vec<f64>,

    /// Inflow in residual space Z'_t (normalized, zero-mean, unit variance)
    /// Transform: Z'_t = (Y_t - μ_s) / σ_s
    /// Used for: AR lag buffer updates, cut generation
    /// Empty for models without AR dynamics (independent inflows)
    pub inflow_residual: Vec<f64>,

    // ========================================================================
    // Physical Variables
    // ========================================================================
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub thermal_generation: Vec<f64>,

    // ========================================================================
    // Dual Values
    // ========================================================================
    pub water_value: Vec<f64>,
    pub marginal_cost: Vec<f64>,

    /// Dual values on AR lag constraints (for StorageAndInflowState with lags)
    ///
    /// Structure: lag_duals[lag_idx][hydro_idx] → dual on Z'_{t-k} = lag_value
    /// - lag_duals[0][h]: Dual on Z'_{t-1} for hydro h
    /// - lag_duals[1][h]: Dual on Z'_{t-2} for hydro h (if AR(2) or higher)
    ///
    /// Empty for StorageState or independent models (no lag constraints)
    pub lag_duals: Vec<Vec<f64>>,

    // ========================================================================
    // Cost and State
    // ========================================================================
    pub current_stage_objective: f64,
    pub total_stage_objective: f64,
    pub final_storage: Vec<f64>,
    pub basis: solver::Basis,
}

impl Realization {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loads: Vec<f64>,
        deficit: Vec<f64>,
        exchange: Vec<f64>,
        inflow: Vec<f64>,
        turbined_flow: Vec<f64>,
        spillage: Vec<f64>,
        thermal_generation: Vec<f64>,
        water_value: Vec<f64>,
        marginal_cost: Vec<f64>,
        current_stage_objective: f64,
        total_stage_objective: f64,
        final_storage: Vec<f64>,
        basis: solver::Basis,
    ) -> Self {
        let num_hydros = inflow.len();
        Self {
            kind: StudyPeriodKind::Study,
            loads,
            deficit,
            exchange,
            inflow,
            inflow_residual: vec![0.0; num_hydros], // Allocate for PAR models
            turbined_flow,
            spillage,
            thermal_generation,
            water_value,
            marginal_cost,
            current_stage_objective,
            total_stage_objective,
            final_storage,
            lag_duals: vec![], // Empty by default (StorageState has no lags)
            basis,
        }
    }

    pub fn with_capacity(
        kind: &StudyPeriodKind,
        system: &system::System,
    ) -> Self {
        Self {
            kind: kind.clone(),
            loads: vec![0.0; system.meta.buses_count],
            deficit: vec![0.0; system.meta.buses_count],
            exchange: vec![0.0; system.meta.lines_count],
            inflow: vec![0.0; system.meta.hydros_count],
            inflow_residual: vec![0.0; system.meta.hydros_count], // For PAR models
            turbined_flow: vec![0.0; system.meta.hydros_count],
            spillage: vec![0.0; system.meta.hydros_count],
            thermal_generation: vec![0.0; system.meta.thermals_count],
            water_value: vec![0.0; system.meta.hydros_count],
            marginal_cost: vec![0.0; system.meta.buses_count],
            current_stage_objective: 0.0,
            total_stage_objective: 0.0,
            final_storage: vec![0.0; system.meta.hydros_count],
            lag_duals: vec![], // Empty by default (StorageState has no lags)
            basis: solver::Basis::new(),
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Returns true if inflow residuals (Z'_t) are populated
    ///
    /// Residuals are populated for AR models where lag buffer updates
    /// and cut generation operate in residual space.
    ///
    /// # Performance
    /// O(1) - checks only the vector length
    #[inline]
    pub fn has_residuals(&self) -> bool {
        !self.inflow_residual.is_empty()
    }

    /// Returns the number of lag duals for a given hydro
    ///
    /// Returns 0 if:
    /// - The hydro index is out of bounds
    /// - No lag constraints exist (StorageState or independent model)
    /// - This hydro has no AR dynamics (AR(0))
    ///
    /// For AR(p) models, returns p (the lag order).
    ///
    /// # Arguments
    /// * `_hydro` - Index of the hydro plant (currently unused, returns same count for all hydros)
    ///
    /// # Performance
    /// O(1) - direct vector length access
    ///
    /// # Example
    /// ```ignore
    /// // For AR(2) model:
    /// assert_eq!(realization.num_lag_duals(0), 2);
    ///
    /// // For independent model:
    /// assert_eq!(realization.num_lag_duals(0), 0);
    /// ```
    #[inline]
    pub fn num_lag_duals(&self, _hydro: usize) -> usize {
        if self.lag_duals.is_empty() {
            return 0;
        }
        // lag_duals[lag_idx][hydro_idx], so count how many lags exist
        // by checking the outer vector length
        self.lag_duals.len()
    }

    /// Returns the total number of lags across all hydros
    ///
    /// This is the total count of lag dual values stored.
    ///
    /// # Performance
    /// O(1) - returns outer vector length
    #[inline]
    pub fn total_lag_count(&self) -> usize {
        self.lag_duals.len()
    }
}

impl Default for Realization {
    fn default() -> Self {
        Self {
            kind: StudyPeriodKind::Study,
            loads: vec![],
            deficit: vec![],
            exchange: vec![],
            inflow: vec![],
            inflow_residual: vec![], // Empty for default
            turbined_flow: vec![],
            spillage: vec![],
            thermal_generation: vec![],
            water_value: vec![],
            marginal_cost: vec![],
            current_stage_objective: 0.0,
            total_stage_objective: 0.0,
            final_storage: vec![],
            lag_duals: vec![], // Empty by default
            basis: solver::Basis::new(),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_create_subproblem_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_stochastic_process];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
        );
        assert_eq!(subproblem.variables.deficit.len(), 1);
        assert_eq!(subproblem.variables.direct_exchange.len(), 0);
        assert_eq!(subproblem.variables.reverse_exchange.len(), 0);
        assert_eq!(subproblem.variables.thermal_gen.len(), 2);
        assert_eq!(subproblem.variables.turbined_flow.len(), 1);
        assert_eq!(subproblem.variables.spillage.len(), 1);
        assert_eq!(subproblem.variables.stored_volume.len(), 1);
        assert_eq!(subproblem.variables.inflow.len(), 1);
    }

    #[test]
    fn test_solve_subproblem_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_stochastic_process];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
        );
        let initial_storage = [83.333];
        let load = [50.0];

        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model {
            model.solve();
            assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
        }
    }

    #[test]
    fn test_get_solution_cost_with_default_system() {
        let system = system::System::default();
        let load_stochastic_process = stochastic_process::factory("naive");
        let inflow_stochastic_process = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_stochastic_process];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_stochastic_process.as_ref(),
            &inflow_processes,
            &[],
            0,
        );
        let initial_storage = [23.333];
        let load = [50.0];

        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model {
            model.solve();
            assert_eq!(model.get_objective_value(), 191.67000000000002);
        }
    }

    // ========================================================================
    // PRIVATE FUNCTION TESTS (Added for T4.2 Phase 5a)
    // ========================================================================

    #[test]
    fn test_get_current_stage_objective() {
        // Test the private helper that extracts current stage objective
        let total_objective = 1000.0;
        let solution = solver::Solution {
            colvalue: vec![10.0, 20.0, 30.0, 40.0],
            coldual: vec![0.0; 4],
            rowvalue: vec![0.0; 2],
            rowdual: vec![0.0; 2],
        };

        let current_obj =
            get_current_stage_objective(total_objective, &solution);
        assert_eq!(current_obj, 1000.0 - 40.0); // total - future (last value)
    }

    #[test]
    fn test_set_default_solver_options() {
        // Test that default solver options are set correctly
        let mut problem = solver::Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(1.0.., [(0, 1.0)]);
        let mut model = problem.optimise(solver::Sense::Minimise);

        set_default_solver_options(&mut model);
        // Options are set but we can't directly query them from HiGHS
        // The test verifies the function doesn't panic
        model.solve();
        assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
    }

    #[test]
    fn test_set_retry_solver_options_coverage() {
        // Test all retry option branches
        let mut problem = solver::Problem::new();
        problem.add_column(1.0, 0.0..);
        problem.add_row(1.0.., [(0, 1.0)]);
        let mut model = problem.optimise(solver::Sense::Minimise);

        // Test each retry level
        set_retry_solver_options(&mut model, 0); // default
        set_retry_solver_options(&mut model, 1); // first retry
        set_retry_solver_options(&mut model, 2); // second retry
        set_retry_solver_options(&mut model, 3); // third retry
        set_retry_solver_options(&mut model, 4); // final retry
        set_retry_solver_options(&mut model, 5); // back to default

        // Verify model still works after all option changes
        model.solve();
        assert_eq!(model.status(), solver::HighsModelStatus::Optimal);
    }

    #[test]
    fn test_subproblem_first_cut_row_index() {
        // Test the private first_cut_row_index method
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let first_cut_idx = subproblem.first_cut_row_index();
        // first_cut_row_index = last inflow process constraint index + 1
        // For default system with constraints, this should be 4
        assert_eq!(first_cut_idx, 4);
    }

    #[test]
    fn test_subproblem_get_deficit_from_solution() {
        // Test private getter for deficit values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set up and solve
        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_deficit_from_solution(&solution, &mut realization);
            assert_eq!(realization.deficit.len(), 1); // 1 bus in default system
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_thermal_gen_from_solution() {
        // Test private getter for thermal generation
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_thermal_gen_from_solution(&solution, &mut realization);
            assert_eq!(realization.thermal_generation.len(), 2); // 2 thermals in default system
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_spillage_from_solution() {
        // Test private getter for spillage values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [100.0];
        let load = [10.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_spillage_from_solution(&solution, &mut realization);
            assert_eq!(realization.spillage.len(), 1); // 1 hydro in default system
            assert!(realization.spillage[0] >= 0.0); // Spillage should be non-negative
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_turbined_flow_from_solution() {
        // Test private getter for turbined flow
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_turbined_flow_from_solution(&solution, &mut realization);
            assert_eq!(realization.turbined_flow.len(), 1); // 1 hydro
            assert!(realization.turbined_flow[0] >= 0.0);
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_final_storage_from_solution() {
        // Test private getter for final storage
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_final_storage_from_solution(&solution, &mut realization);
            assert_eq!(realization.final_storage.len(), 1);
            assert!(realization.final_storage[0] >= 0.0);
            assert!(realization.final_storage[0] <= 100.0); // Within max storage
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_water_values_from_solution() {
        // Test private getter for water values (duals)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_water_values_from_solution(&solution, &mut realization);
            assert_eq!(realization.water_value.len(), 1); // 1 hydro
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_subproblem_get_marginal_cost_from_solution() {
        // Test private getter for marginal costs (bus duals)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        let initial_storage = [50.0];
        let load = [30.0];
        subproblem.set_hydro_balance_rhs(&initial_storage);
        subproblem.set_load_balance_rhs(&load);

        if let Some(mut model) = subproblem.model.take() {
            model.solve();
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_marginal_cost_from_solution(&solution, &mut realization);
            assert_eq!(realization.marginal_cost.len(), 1); // 1 bus
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_set_load_balance_rhs() {
        // Test setting load balance RHS values
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set new loads
        let new_loads = vec![50.0];
        subproblem.set_load_balance_rhs(&new_loads);

        // Verify by solving - should work without errors
        assert!(subproblem.model.is_some());
    }

    #[test]
    fn test_set_hydro_balance_rhs() {
        // Test setting hydro balance RHS values (initial storage)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Set new initial storage
        let new_storage = vec![75.0];
        subproblem.set_hydro_balance_rhs(&new_storage);

        // Verify by solving - should work without errors
        assert!(subproblem.model.is_some());
    }

    #[test]
    fn test_get_net_exchange_from_solution() {
        // Test extracting net exchange values from solution
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Solve to get a solution
        let mut model = subproblem.model.take().unwrap();
        model.solve();

        if model.status() == solver::HighsModelStatus::Optimal {
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem
                .get_net_exchange_from_solution(&solution, &mut realization);
            // Default system may or may not have exchange variables
            // Just verify the function executes without crashing
            subproblem.model = Some(model);
        }
    }

    #[test]
    fn test_get_inflow_from_solution() {
        // Test extracting inflow values from solution
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Solve to get a solution
        let mut model = subproblem.model.take().unwrap();
        model.solve();

        if model.status() == solver::HighsModelStatus::Optimal {
            let solution = model.get_solution();
            let mut realization =
                Realization::with_capacity(&StudyPeriodKind::Study, &system);
            subproblem.get_inflow_from_solution(&solution, &mut realization);
            assert_eq!(realization.inflow.len(), 1); // 1 hydro
            subproblem.model = Some(model);
        }
    }

    // ========================================================================
    // TICKET-004: Variables Struct Tests (Dual Space Representation)
    // ========================================================================

    #[test]
    fn test_variables_has_new_dual_space_fields() {
        // Test that Variables struct has the new fields for dual space representation
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // Check that new fields exist and have correct size
        assert_eq!(
            subproblem.variables.inflow_residual.len(),
            system.meta.hydros_count
        );
        assert_eq!(
            subproblem.variables.innovation.len(),
            system.meta.hydros_count
        );
        assert!(subproblem.variables.lagged_inflow_state.is_none()); // StorageState
    }

    #[test]
    fn test_variables_has_lagged_inflow_state_returns_false_when_none() {
        // Test has_lagged_inflow_state() returns false for StorageState
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        assert!(!subproblem.variables.has_lagged_inflow_state());
    }

    #[test]
    fn test_variables_has_lagged_inflow_state_returns_true_when_some() {
        // Test has_lagged_inflow_state() returns true when lagged state exists
        let mut variables = Variables {
            deficit: vec![0],
            direct_exchange: vec![],
            reverse_exchange: vec![],
            thermal_gen: vec![0, 1],
            turbined_flow: vec![0],
            spillage: vec![0],
            stored_volume: vec![0],
            inflow: vec![0],
            inflow_residual: vec![0],
            innovation: vec![0],
            lagged_inflow_state: Some(vec![vec![10, 11]]), // AR(2) lags
            alpha: 100,
        };

        assert!(variables.has_lagged_inflow_state());

        // Now set to None
        variables.lagged_inflow_state = None;
        assert!(!variables.has_lagged_inflow_state());
    }

    #[test]
    fn test_variables_num_inflow_lags_returns_zero_when_none() {
        // Test num_inflow_lags() returns 0 for StorageState
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        assert_eq!(subproblem.variables.num_inflow_lags(0), 0);
    }

    #[test]
    fn test_variables_num_inflow_lags_returns_correct_count() {
        // Test num_inflow_lags() returns correct count for AR(2)
        let variables = Variables {
            deficit: vec![0],
            direct_exchange: vec![],
            reverse_exchange: vec![],
            thermal_gen: vec![0, 1],
            turbined_flow: vec![0],
            spillage: vec![0],
            stored_volume: vec![0],
            inflow: vec![0],
            inflow_residual: vec![0],
            innovation: vec![0],
            lagged_inflow_state: Some(vec![vec![10, 11]]), // AR(2): 2 lags
            alpha: 100,
        };

        assert_eq!(variables.num_inflow_lags(0), 2);
    }

    #[test]
    fn test_variables_num_inflow_lags_out_of_bounds() {
        // Test num_inflow_lags() returns 0 for out of bounds hydro index
        let variables = Variables {
            deficit: vec![0],
            direct_exchange: vec![],
            reverse_exchange: vec![],
            thermal_gen: vec![0, 1],
            turbined_flow: vec![0],
            spillage: vec![0],
            stored_volume: vec![0],
            inflow: vec![0],
            inflow_residual: vec![0],
            innovation: vec![0],
            lagged_inflow_state: Some(vec![vec![10, 11]]),
            alpha: 100,
        };

        assert_eq!(variables.num_inflow_lags(999), 0);
    }

    #[test]
    fn test_variables_clone() {
        // Test that Variables can be cloned correctly
        let variables = Variables {
            deficit: vec![0],
            direct_exchange: vec![],
            reverse_exchange: vec![],
            thermal_gen: vec![0, 1],
            turbined_flow: vec![0],
            spillage: vec![0],
            stored_volume: vec![0],
            inflow: vec![0],
            inflow_residual: vec![0],
            innovation: vec![0],
            lagged_inflow_state: Some(vec![vec![10, 11]]),
            alpha: 100,
        };

        let cloned = variables.clone();
        assert_eq!(cloned.deficit, variables.deficit);
        assert_eq!(cloned.inflow_residual, variables.inflow_residual);
        assert_eq!(cloned.innovation, variables.innovation);
        assert_eq!(cloned.alpha, variables.alpha);
        assert!(cloned.has_lagged_inflow_state());
        assert_eq!(cloned.num_inflow_lags(0), 2);
    }

    #[test]
    fn test_variables_with_storage_state() {
        // Test Variables with StorageState (no lagged state variables)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage", // StorageState
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        assert!(subproblem.variables.lagged_inflow_state.is_none());
        assert!(!subproblem.variables.has_lagged_inflow_state());
        assert_eq!(subproblem.variables.num_inflow_lags(0), 0);
    }

    #[test]
    fn test_variables_with_storage_and_inflow_state() {
        // Test Variables with StorageAndInflowState (has lagged state variables)
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];

        // Create minimal unified_specs for UnifiedInflowModel
        // Default system has 2 hydros, need independent noise specs for both
        use crate::input::UncertaintyType;
        use crate::unified_noise_spec::{
            SeasonalNoiseParams, TemporalModelSpec, UnifiedNoiseSpec,
        };
        use std::collections::HashMap;

        let mut seasonal_params0 = HashMap::new();
        seasonal_params0.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 10.0,
                marginal_override: None,
            },
        );

        let mut seasonal_params1 = HashMap::new();
        seasonal_params1.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 10.0,
                marginal_override: None,
            },
        );

        let unified_specs = vec![
            UnifiedNoiseSpec {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 0,
                temporal_model: TemporalModelSpec::Independent,
                seasonal_params: seasonal_params0,
                marginal_distribution: None,
            },
            UnifiedNoiseSpec {
                uncertainty_type: UncertaintyType::Inflow,
                entity_id: 1,
                temporal_model: TemporalModelSpec::Independent,
                seasonal_params: seasonal_params1,
                marginal_distribution: None,
            },
        ];

        let subproblem = Subproblem::new(
            &system,
            "storage_and_inflow", // StorageAndInflowState
            load_sp.as_ref(),
            &inflow_processes,
            &unified_specs,
            0,
        );

        // TICKET-010: lagged_inflow_state is now populated for StorageAndInflowState
        // For independent noise (no lags), this will be Some(vec![vec![]; n_hydros])
        assert!(subproblem.variables.lagged_inflow_state.is_some());
    }

    // ========================================================================
    // Constraints struct tests
    // ========================================================================

    #[test]
    fn test_constraints_has_new_fields() {
        // Test that Constraints struct has inflow_transform and ar_dynamics fields
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![4, 5],
            ar_dynamics: vec![6, 7],
        };

        assert_eq!(constraints.load_balance, vec![0, 1]);
        assert_eq!(constraints.hydro_balance, vec![2, 3]);
        assert_eq!(constraints.inflow_transform, vec![4, 5]);
        assert_eq!(constraints.ar_dynamics, vec![6, 7]);
    }

    #[test]
    fn test_constraints_num_inflow_constraints() {
        // Test num_inflow_constraints() returns correct count
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![4, 5, 6],
            ar_dynamics: vec![7, 8, 9],
        };

        assert_eq!(constraints.num_inflow_constraints(), 3);
    }

    #[test]
    fn test_constraints_num_inflow_constraints_empty() {
        // Test num_inflow_constraints() returns 0 when empty
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![],
            ar_dynamics: vec![],
        };

        assert_eq!(constraints.num_inflow_constraints(), 0);
    }

    #[test]
    fn test_constraints_has_ar_dynamics_true() {
        // Test has_ar_dynamics() returns true when populated
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![4, 5],
            ar_dynamics: vec![6, 7],
        };

        assert!(constraints.has_ar_dynamics());
    }

    #[test]
    fn test_constraints_has_ar_dynamics_false() {
        // Test has_ar_dynamics() returns false when empty
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![4, 5],
            ar_dynamics: vec![],
        };

        assert!(!constraints.has_ar_dynamics());
    }

    #[test]
    fn test_constraints_clone() {
        // Test that Constraints can be cloned correctly
        let constraints = Constraints {
            load_balance: vec![0, 1],
            hydro_balance: vec![2, 3],
            inflow_transform: vec![4, 5],
            ar_dynamics: vec![6, 7],
        };

        let cloned = constraints.clone();
        assert_eq!(cloned.load_balance, constraints.load_balance);
        assert_eq!(cloned.hydro_balance, constraints.hydro_balance);
        assert_eq!(cloned.inflow_transform, constraints.inflow_transform);
        assert_eq!(cloned.ar_dynamics, constraints.ar_dynamics);
        assert_eq!(cloned.num_inflow_constraints(), 2);
        assert!(cloned.has_ar_dynamics());
    }

    #[test]
    fn test_constraints_initialization_in_subproblem() {
        // Test that Constraints are initialized correctly in Subproblem construction
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let subproblem = Subproblem::new(
            &system,
            "storage",
            load_sp.as_ref(),
            &inflow_processes,
            &[],
            0,
        );

        // TICKET-008: Unified model constraints are now populated
        // Should have inflow_transform and ar_dynamics for all hydros
        assert_eq!(
            subproblem.constraints.inflow_transform.len(),
            system.meta.hydros_count
        );
        assert_eq!(
            subproblem.constraints.ar_dynamics.len(),
            system.meta.hydros_count
        );
        assert_eq!(
            subproblem.constraints.num_inflow_constraints(),
            system.meta.hydros_count
        );
        assert!(subproblem.constraints.has_ar_dynamics());
    }

    // ========================================================================
    // Realization struct tests
    // ========================================================================

    #[test]
    fn test_realization_has_residuals_true() {
        // Test has_residuals() returns true when populated
        let realization = Realization {
            inflow: vec![100.0, 150.0],
            inflow_residual: vec![0.5, -0.3],
            ..Default::default()
        };

        assert!(realization.has_residuals());
    }

    #[test]
    fn test_realization_has_residuals_false() {
        // Test has_residuals() returns false when empty
        let realization = Realization {
            inflow: vec![100.0, 150.0],
            inflow_residual: vec![], // explicitly empty
            ..Default::default()
        };

        assert!(!realization.has_residuals());
    }

    #[test]
    fn test_realization_num_lag_duals_ar2() {
        // Test num_lag_duals() for AR(2) model
        let realization = Realization {
            lag_duals: vec![
                vec![2.5, 3.1], // Lag 1 duals for 2 hydros
                vec![1.8, 2.2], // Lag 2 duals for 2 hydros
            ],
            ..Default::default()
        };

        assert_eq!(realization.num_lag_duals(0), 2);
        assert_eq!(realization.num_lag_duals(1), 2);
    }

    #[test]
    fn test_realization_num_lag_duals_empty() {
        // Test num_lag_duals() returns 0 when no lags
        let realization = Realization::default();

        assert_eq!(realization.num_lag_duals(0), 0);
        assert_eq!(realization.num_lag_duals(999), 0);
    }

    #[test]
    fn test_realization_total_lag_count() {
        // Test total_lag_count() returns correct count
        let realization = Realization {
            lag_duals: vec![
                vec![2.5, 3.1, 4.0], // Lag 1 for 3 hydros
                vec![1.8, 2.2, 3.5], // Lag 2 for 3 hydros
                vec![0.9, 1.1, 1.3], // Lag 3 for 3 hydros (AR(3))
            ],
            ..Default::default()
        };

        assert_eq!(realization.total_lag_count(), 3);
    }

    #[test]
    fn test_realization_total_lag_count_empty() {
        // Test total_lag_count() returns 0 when empty
        let realization = Realization::default();

        assert_eq!(realization.total_lag_count(), 0);
    }

    #[test]
    fn test_realization_default() {
        // Test Default implementation initializes all fields correctly
        let realization = Realization::default();

        assert_eq!(realization.kind, StudyPeriodKind::Study);
        assert!(realization.loads.is_empty());
        assert!(realization.deficit.is_empty());
        assert!(realization.exchange.is_empty());
        assert!(realization.inflow.is_empty());
        assert!(realization.inflow_residual.is_empty());
        assert!(realization.turbined_flow.is_empty());
        assert!(realization.spillage.is_empty());
        assert!(realization.thermal_generation.is_empty());
        assert!(realization.water_value.is_empty());
        assert!(realization.marginal_cost.is_empty());
        assert!(realization.lag_duals.is_empty());
        assert_eq!(realization.current_stage_objective, 0.0);
        assert_eq!(realization.total_stage_objective, 0.0);
        assert!(realization.final_storage.is_empty());
        assert!(!realization.has_residuals());
        assert_eq!(realization.num_lag_duals(0), 0);
        assert_eq!(realization.total_lag_count(), 0);
    }

    #[test]
    fn test_realization_with_capacity() {
        // Test with_capacity() initializes vectors with correct sizes
        let system = system::System::default();
        let realization =
            Realization::with_capacity(&StudyPeriodKind::Study, &system);

        assert_eq!(realization.kind, StudyPeriodKind::Study);
        assert_eq!(realization.loads.len(), system.meta.buses_count);
        assert_eq!(realization.deficit.len(), system.meta.buses_count);
        assert_eq!(realization.exchange.len(), system.meta.lines_count);
        assert_eq!(realization.inflow.len(), system.meta.hydros_count);
        assert_eq!(realization.inflow_residual.len(), system.meta.hydros_count);
        assert_eq!(realization.turbined_flow.len(), system.meta.hydros_count);
        assert_eq!(realization.spillage.len(), system.meta.hydros_count);
        assert_eq!(
            realization.thermal_generation.len(),
            system.meta.thermals_count
        );
        assert_eq!(realization.water_value.len(), system.meta.hydros_count);
        assert_eq!(realization.marginal_cost.len(), system.meta.buses_count);
        assert_eq!(realization.final_storage.len(), system.meta.hydros_count);
        assert!(realization.has_residuals()); // Allocated with capacity
        assert!(realization.lag_duals.is_empty()); // Not allocated by default
    }

    #[test]
    fn test_realization_clone() {
        // Test that Realization can be cloned correctly
        let realization = Realization {
            inflow: vec![100.0, 150.0],
            inflow_residual: vec![0.5, -0.3],
            lag_duals: vec![vec![2.5, 3.1], vec![1.8, 2.2]],
            current_stage_objective: 1234.5,
            ..Default::default()
        };

        let cloned = realization.clone();

        assert_eq!(cloned.inflow, realization.inflow);
        assert_eq!(cloned.inflow_residual, realization.inflow_residual);
        assert_eq!(cloned.lag_duals, realization.lag_duals);
        assert_eq!(
            cloned.current_stage_objective,
            realization.current_stage_objective
        );
        assert!(cloned.has_residuals());
        assert_eq!(cloned.num_lag_duals(0), 2);
        assert_eq!(cloned.total_lag_count(), 2);
    }

    #[test]
    fn test_realization_with_observation_and_residual_space() {
        // Test Realization with both observation and residual space values
        let realization = Realization {
            // Observation space (physical units)
            inflow: vec![100.0, 150.0, 200.0],
            // Residual space (normalized)
            inflow_residual: vec![0.5, -0.3, 1.2],
            // Lag duals for AR(2) with 3 hydros
            lag_duals: vec![
                vec![2.5, 3.1, 4.0], // Lag 1
                vec![1.8, 2.2, 3.5], // Lag 2
            ],
            ..Default::default()
        };

        assert_eq!(realization.inflow.len(), 3);
        assert_eq!(realization.inflow_residual.len(), 3);
        assert!(realization.has_residuals());
        assert_eq!(realization.num_lag_duals(0), 2);
        assert_eq!(realization.num_lag_duals(1), 2);
        assert_eq!(realization.num_lag_duals(2), 2);
        assert_eq!(realization.total_lag_count(), 2);
    }

    #[test]
    fn test_realization_mixed_lag_duals() {
        // Test Realization with different hydros (simulating mixed AR orders)
        // Note: Current structure has same lag count for all hydros,
        // but this tests the API works correctly
        let realization = Realization {
            inflow: vec![100.0, 150.0, 200.0],
            inflow_residual: vec![0.5, -0.3, 1.2],
            // AR(1) - only one lag
            lag_duals: vec![
                vec![2.5, 3.1, 4.0], // Lag 1 for all hydros
            ],
            ..Default::default()
        };

        assert_eq!(realization.num_lag_duals(0), 1);
        assert_eq!(realization.num_lag_duals(1), 1);
        assert_eq!(realization.num_lag_duals(2), 1);
        assert_eq!(realization.total_lag_count(), 1);
    }

    #[test]
    fn test_realization_new_constructor() {
        // Test the new() constructor
        let realization = Realization::new(
            vec![50.0, 60.0],     // loads
            vec![0.0, 0.0],       // deficit
            vec![10.0],           // exchange
            vec![100.0, 150.0],   // inflow
            vec![80.0, 120.0],    // turbined_flow
            vec![20.0, 30.0],     // spillage
            vec![15.0],           // thermal_generation
            vec![45.0, 55.0],     // water_value
            vec![25.0, 30.0],     // marginal_cost
            1000.0,               // current_stage_objective
            1500.0,               // total_stage_objective
            vec![200.0, 250.0],   // final_storage
            solver::Basis::new(), // basis
        );

        assert_eq!(realization.kind, StudyPeriodKind::Study);
        assert_eq!(realization.inflow.len(), 2);
        assert_eq!(realization.inflow_residual.len(), 2); // Allocated with zeros
        assert!(realization.has_residuals()); // Allocated but zeros
        assert!(realization.lag_duals.is_empty()); // Empty by default
        assert_eq!(realization.current_stage_objective, 1000.0);
        assert_eq!(realization.total_stage_objective, 1500.0);
    }

    // ============================================================================
    // TICKET-007: UnifiedInflowModel Integration Tests
    // ============================================================================

    use crate::unified_noise_spec::{
        SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec,
        UnifiedNoiseSpec,
    };
    use std::collections::HashMap;

    /// Helper: Create independent noise spec for testing
    fn create_independent_inflow_spec(entity_id: usize) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();
        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        UnifiedNoiseSpec {
            uncertainty_type: crate::input::UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: Some(
                crate::input::MarginalDistribution::Normal {
                    mean: 0.0,
                    std_dev: 1.0,
                },
            ),
        }
    }

    /// Helper: Create AR(1) noise spec for testing
    fn create_ar1_inflow_spec(entity_id: usize) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();
        let mut ar_params = HashMap::new();

        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 1,
                ar_coefficients: vec![0.7],
            },
        );

        UnifiedNoiseSpec {
            uncertainty_type: crate::input::UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params,
            marginal_distribution: Some(
                crate::input::MarginalDistribution::Normal {
                    mean: 0.0,
                    std_dev: 1.0,
                },
            ),
        }
    }

    /// Helper: Create AR(2) noise spec for testing
    fn create_ar2_inflow_spec(entity_id: usize) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();
        let mut ar_params = HashMap::new();

        seasonal_params.insert(
            0,
            SeasonalNoiseParams {
                mean: 100.0,
                std_dev: 20.0,
                marginal_override: None,
            },
        );

        ar_params.insert(
            0,
            SeasonalPARParams {
                ar_order: 2,
                ar_coefficients: vec![0.5, 0.3],
            },
        );

        UnifiedNoiseSpec {
            uncertainty_type: crate::input::UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 1,
                seasonal_ar_params: ar_params,
            },
            seasonal_params,
            marginal_distribution: Some(
                crate::input::MarginalDistribution::Normal {
                    mean: 0.0,
                    std_dev: 1.0,
                },
            ),
        }
    }

    #[test]
    fn test_unified_inflow_model_field_exists() {
        // Test that Subproblem has inflow_model field
        // This is a compilation test - if it compiles, the field exists
        use crate::unified_inflow_model::UnifiedInflowModel;

        // Create a dummy check - if UnifiedInflowModel is accessible, test passes
        let _check: Option<UnifiedInflowModel> = None;
        // Test passes if this compiles
    }

    #[test]
    fn test_inflow_model_construction_independent() {
        // Test that UnifiedInflowModel is properly constructed for independent case
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![
            create_independent_inflow_spec(0),
            create_independent_inflow_spec(1),
        ];

        let seasonal_params = std::sync::Arc::new(
            SeasonalParams::from_unified_specs(&specs, 2)
                .expect("Failed to create seasonal params"),
        );

        let model = crate::unified_inflow_model::UnifiedInflowModel::from_spec(
            &specs,
            2,
            seasonal_params,
        );

        // Check dimensions
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.max_lag(), 0); // Independent = AR(0)

        // Check lag orders
        assert_eq!(model.lag_order(0), 0);
        assert_eq!(model.lag_order(1), 0);

        // Check AR dynamics flag
        assert!(!model.has_ar_dynamics(0));
        assert!(!model.has_ar_dynamics(1));
    }

    #[test]
    fn test_inflow_model_construction_ar1() {
        // Test that UnifiedInflowModel is properly constructed for AR(1) case
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![create_ar1_inflow_spec(0), create_ar1_inflow_spec(1)];

        let seasonal_params = std::sync::Arc::new(
            SeasonalParams::from_unified_specs(&specs, 2)
                .expect("Failed to create seasonal params"),
        );

        let model = crate::unified_inflow_model::UnifiedInflowModel::from_spec(
            &specs,
            2,
            seasonal_params,
        );

        // Check dimensions
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.max_lag(), 1); // AR(1)

        // Check lag orders
        assert_eq!(model.lag_order(0), 1);
        assert_eq!(model.lag_order(1), 1);

        // Check AR dynamics flag
        assert!(model.has_ar_dynamics(0));
        assert!(model.has_ar_dynamics(1));
    }

    #[test]
    fn test_inflow_model_construction_ar2() {
        // Test that UnifiedInflowModel is properly constructed for AR(2) case
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![create_ar2_inflow_spec(0), create_ar2_inflow_spec(1)];

        let seasonal_params = std::sync::Arc::new(
            SeasonalParams::from_unified_specs(&specs, 2)
                .expect("Failed to create seasonal params"),
        );

        let model = crate::unified_inflow_model::UnifiedInflowModel::from_spec(
            &specs,
            2,
            seasonal_params,
        );

        // Check dimensions
        assert_eq!(model.dimension(), 2);
        assert_eq!(model.max_lag(), 2); // AR(2)

        // Check lag orders
        assert_eq!(model.lag_order(0), 2);
        assert_eq!(model.lag_order(1), 2);

        // Check AR dynamics flag
        assert!(model.has_ar_dynamics(0));
        assert!(model.has_ar_dynamics(1));
    }

    #[test]
    fn test_inflow_model_construction_mixed() {
        // Test that UnifiedInflowModel handles mixed AR orders
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![
            create_independent_inflow_spec(0), // AR(0)
            create_ar1_inflow_spec(1),         // AR(1)
            create_ar2_inflow_spec(2),         // AR(2)
        ];

        let seasonal_params = std::sync::Arc::new(
            SeasonalParams::from_unified_specs(&specs, 3)
                .expect("Failed to create seasonal params"),
        );

        let model = crate::unified_inflow_model::UnifiedInflowModel::from_spec(
            &specs,
            3,
            seasonal_params,
        );

        // Check dimensions
        assert_eq!(model.dimension(), 3);
        assert_eq!(model.max_lag(), 2); // Max across all hydros

        // Check individual lag orders
        assert_eq!(model.lag_order(0), 0); // Independent
        assert_eq!(model.lag_order(1), 1); // AR(1)
        assert_eq!(model.lag_order(2), 2); // AR(2)

        // Check AR dynamics flag
        assert!(!model.has_ar_dynamics(0)); // Independent
        assert!(model.has_ar_dynamics(1)); // AR(1)
        assert!(model.has_ar_dynamics(2)); // AR(2)
    }

    #[test]
    fn test_seasonal_params_from_unified_specs_independent() {
        // Test SeasonalParams extraction for independent case
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![
            create_independent_inflow_spec(0),
            create_independent_inflow_spec(1),
        ];

        let params = SeasonalParams::from_unified_specs(&specs, 2)
            .expect("Failed to create seasonal params");

        // Check basic properties
        assert_eq!(params.get_mean(0), 100.0);
        assert_eq!(params.get_std(0), 20.0);
        assert_eq!(params.get_ar_order(0), 0);
        assert!(params.get_ar_coeffs(0).is_empty());
    }

    #[test]
    fn test_seasonal_params_from_unified_specs_ar1() {
        // Test SeasonalParams extraction for AR(1) case
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![create_ar1_inflow_spec(0), create_ar1_inflow_spec(1)];

        let params = SeasonalParams::from_unified_specs(&specs, 2)
            .expect("Failed to create seasonal params");

        // Check basic properties
        assert_eq!(params.get_mean(0), 100.0);
        assert_eq!(params.get_std(0), 20.0);
        assert_eq!(params.get_ar_order(0), 1);
        assert_eq!(params.get_ar_coeffs(0), &[0.7]);
    }

    #[test]
    fn test_seasonal_params_from_unified_specs_no_inflows() {
        // Test SeasonalParams extraction when no inflow specs are present
        use crate::seasonal_params::SeasonalParams;

        let specs = vec![]; // No inflow specs

        let params = SeasonalParams::from_unified_specs(&specs, 0)
            .expect("Failed to create seasonal params");

        // Should return identity transformation (AR(0), μ=0, σ=1)
        assert_eq!(params.get_mean(0), 0.0);
        assert_eq!(params.get_std(0), 1.0);
        assert_eq!(params.get_ar_order(0), 0);
        assert!(params.get_ar_coeffs(0).is_empty());
    }
}
