//! State representations for SDDP subproblems.
//!
//! # Architecture: Single Source of Truth
//!
//! The `State` trait provides a unified interface where `coefficients()` returns
//! the Markov state representation used in cut evaluation. This flat vector IS the
//! state - everything else is extraction logic to populate it from the trajectory.
//!
//! ## Core Principle
//!
//! **State = Coefficients + Extraction Logic**
//!
//! - The `state_coefficients` field is the single source of truth
//! - The `coefficients()` method returns a direct reference to it
//! - State updates extract data from the trajectory and rebuild coefficients
//! - Cut evaluation uses `coefficients()` directly: `rhs = objective - dot(cut_coeffs, state_coeffs)`
//!
//! ## State Implementations
//!
//! ### StorageState
//!
//! ```text
//! state_coefficients: [V₀, V₁, V₂, ..., Vₙ]
//! ```
//!
//! - **Purpose**: Simplest state representation where only reservoir storage levels matter
//! - **Extraction**: O(1) via `trajectory.last().final_storage`
//! - **Use case**: Systems without temporal correlation in inflows
//!
//! ### StorageAndInflowState
//!
//! ```text
//! state_coefficients: [V₀, ..., Vₙ, Y₀⁽¹⁾, Y₀⁽²⁾, ..., Yₙ⁽ᵖ⁾]
//!                      └─ Storage ┘  └──── Lagged Inflows ─────┘
//! ```
//!
//! - **Purpose**: Captures temporal correlation via lagged inflows as state variables
//! - **Extraction**: O(p) via `trajectory[len-p..len]` for each hydro
//! - **Use case**: PAR(p) inflow models where past inflows affect future
//!
//! **Heterogeneous AR Orders Example** (3 hydros: AR(0), AR(1), AR(2)):
//!
//! ```text
//! state_coefficients: [V₀, V₁, Y₁⁽¹⁾, V₂, Y₂⁽¹⁾, Y₂⁽²⁾]
//!                      │   │    │     │    │      │
//!                      │   │    │     │    │      └─ Hydro 2 lag 2
//!                      │   │    │     │    └──────── Hydro 2 lag 1
//!                      │   │    │     └───────────── Hydro 2 storage
//!                      │   │    └─────────────────── Hydro 1 lag 1
//!                      │   └──────────────────────── Hydro 1 storage
//!                      └──────────────────────────── Hydro 0 storage (no lags)
//! ```
//!

use crate::cut;
use crate::input::UncertaintyType;
use crate::risk_measure;
use crate::solver;
use crate::subproblem;
use crate::system;
use crate::utils;
use std::ops::Range;

pub trait State: Send + Sync {
    // Metadata methods
    fn set_dimension(&mut self, dimension: usize);

    /// Returns the Markov state as a flat coefficient vector.
    ///
    /// This is THE state representation for SDDP. The returned slice is used
    /// directly in cut evaluation:
    ///
    /// ```text
    /// cut_rhs = objective - dot_product(cut_coefficients, state_coefficients)
    /// ```
    ///
    /// # State Layout
    ///
    /// - `StorageState`: `[V₀, V₁, ..., Vₙ]` (storage only)
    /// - `StorageAndInflowState`: `[V₀, ..., Vₙ, Y₀⁽¹⁾, ..., Yₙ⁽ᵖ⁾]` (storage + lags)
    ///
    /// # Performance
    ///
    /// O(1) - returns a slice reference to the internal state vector, no allocation.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let coeffs = state.coefficients();  // &[f64]
    /// let cut_rhs = objective - dot_product(&cut.coefficients, coeffs);
    /// ```
    fn coefficients(&self) -> &[f64];

    fn get_dominating_objective(&self) -> f64;
    fn set_dominating_objective(&mut self, dominating_objective: f64);
    fn get_dominating_cut_id(&self) -> usize;
    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize);

    fn get_iteration(&self) -> usize;
    fn set_iteration(&mut self, iteration: usize);
    fn get_forward_pass_idx(&self) -> usize;
    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize);

    /// Returns true if this state type includes lagged observation state variables.
    ///
    /// This method supports unified lag tracking for all entity types (loads and inflows).
    ///
    /// - `StorageState`: false (only storage is state variable)
    /// - `StorageAndInflowState`: true (only tracks inflow lags for backward compat)
    /// - `StorageAndObservationState`: true (tracks all entity lags)
    fn has_lagged_observation_state(&self) -> bool {
        false
    }

    /// Get lagged observations for a specific entity (unified approach).
    ///
    /// # Arguments
    ///
    /// - `entity_idx`: Global entity index (0..num_entities), ordered as [loads..., inflows...]
    ///
    /// # Returns
    ///
    /// Slice of lagged observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] for this entity,
    /// or empty slice if entity has no lags (ar_order = 0) or state doesn't track lags.
    ///
    /// # Default Implementation
    ///
    /// Returns empty slice. Override in states that track lagged observations.
    fn get_lagged_observations(&self, _entity_idx: usize) -> &[f64] {
        &[]
    }

    /// Set lagged observations for a specific entity (unified approach).
    ///
    /// # Arguments
    ///
    /// - `entity_idx`: Global entity index (0..num_entities)
    /// - `observations`: New lag values [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///
    /// # Default Implementation
    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    );

    /// Update state and subproblem from trajectory of past realizations.
    ///
    /// This method is called during forward pass to transfer state information
    /// from previous stages to the current subproblem. It implements the
    /// **Extract → Rebuild** pattern:
    ///
    /// 1. **Extract** relevant data from trajectory (source of truth)
    /// 2. **Rebuild** `state_coefficients` from extracted data
    /// 3. **Update** LP bounds to reflect new state
    ///
    /// Each state implementation extracts what it needs:
    ///
    /// - `StorageState`: uses `.last()` for previous storage (O(1))
    /// - `StorageAndInflowState`: uses `[len-p..len]` for lags (O(p) per hydro)
    ///
    /// # Arguments
    ///
    /// - `past_realizations`: Trajectory of all previous realizations (source of truth)
    /// - `model`: Solver model to update with state-dependent bounds
    /// - `constraints`: Constraint indices for updating RHS
    /// - `variables`: Variable indices for updating bounds
    ///
    /// # Example Trajectory Structure
    ///
    /// ```text
    /// Stage 1: [PreStudy(0)]
    /// Stage 2: [PreStudy(0), Stage(1)]
    /// Stage 3: [PreStudy(0), Stage(1), Stage(2)]
    ///
    /// With multi-node pre-study (PAR):
    /// Stage 1: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0)]
    /// Stage 2: [PreStudy(-3), PreStudy(-2), PreStudy(-1), PreStudy(0), Stage(1)]
    /// ```
    ///
    /// # Performance
    ///
    /// - StorageState: O(n) to extract and update bounds
    /// - StorageAndInflowState: O(n+Σp) to extract storage, lags, and update bounds
    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        variables: &subproblem::Variables,
    );

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>>;

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    );

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut;

    // default implementations
    fn update_dominating_cut(&mut self, cut: &cut::BendersCut, height: f64) {
        self.set_dominating_cut_id(cut.id);
        self.set_dominating_objective(height);
    }

    fn compute_new_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        // NOTE: Don't call update_dominating_cut() here! The cut has id=0 at this point.
        // The FCF will handle domination properly after assigning the real cut ID.
        self.evaluate_cut(
            risk_measure,
            forward_trajectory,
            branching_realizations,
        )
    }
    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State>;
}

// trait for cloning boxes of State trait objects
impl Clone for Box<dyn State> {
    fn clone(&self) -> Self {
        self.as_ref().clone_dyn()
    }
}

pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}

impl Default for VisitedStatePool {
    fn default() -> Self {
        Self::new()
    }
}

impl VisitedStatePool {
    pub fn new() -> Self {
        Self { pool: vec![] }
    }
}

/// Extract maximum AR order for a hydro from TemporalModel
fn extract_max_ar_order_for_hydro(
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    hydro_id: usize,
    _season_id: usize,
) -> usize {
    use crate::input::UncertaintyType;

    uncertainty_models
        .iter()
        .filter(|model| {
            model.entity_type() == UncertaintyType::Inflow
                && model.entity_id() == hydro_id
        })
        .map(|model| model.max_ar_order)
        .max()
        .unwrap_or(0)
}

/// Calculate per-hydro state dimensions from TemporalModel
pub fn per_hydro_state_dims(
    system: &system::System,
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    season_id: usize,
) -> Vec<usize> {
    system
        .hydros
        .iter()
        .map(|hydro| {
            let max_order = extract_max_ar_order_for_hydro(
                uncertainty_models,
                hydro.id,
                season_id,
            );
            1 + max_order // storage + lags
        })
        .collect()
}

/// Calculate total state dimension from TemporalModel
pub fn total_state_dim(
    system: &system::System,
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    season_id: usize,
) -> usize {
    per_hydro_state_dims(system, uncertainty_models, season_id)
        .iter()
        .sum()
}

/// State layout for variable-length per-hydro state vectors.
///
/// Tracks offsets and dimensions for each hydro's state slice in the
/// flattened state vector. Enables O(1) state access for heterogeneous
/// AR orders.
///
/// # State Vector Layout
///
/// For hydros with different AR orders:
/// ```text
/// Hydro 0: AR(2) → [storage₀, lag₀₁, lag₀₂]       (dim=3)
/// Hydro 1: AR(1) → [storage₁, lag₁₁]              (dim=2)
/// Hydro 2: AR(0) → [storage₂]                     (dim=1)
///
/// Flattened: [storage₀, lag₀₁, lag₀₂, storage₁, lag₁₁, storage₂]
///            ←───── offset=0 ─────→  ←─ offset=3 ─→  ←offset=5→
/// ```
///
/// # Invariants
///
/// - `offsets.len() == num_hydros + 1`
/// - `offsets[i+1] - offsets[i] == per_hydro_dims[i]`
/// - `offsets.last() == total_dim`
/// - `per_hydro_dims.iter().sum() == total_dim`
///
/// # Performance
///
/// - Memory: ~16 bytes × num_hydros (Vec overhead)
/// - Offset lookup: O(1) array access
/// - Slice extraction: O(dim) memcpy
///
/// # Example
///
/// ```ignore
/// let layout = StateLayout::from_unified_specs(&system, &unified_specs, 0);
/// let hydro1_range = layout.hydro_slice(1);
/// let hydro1_state = &state[hydro1_range]; // [storage₁, lag₁₁]
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StateLayout {
    /// Per-hydro state dimensions [dim₀, dim₁, ..., dimₙ]
    ///
    /// For hydro i: dim[i] = 1 + max_ar_order[i]
    pub per_hydro_dims: Vec<usize>,

    /// Cumulative offsets [0, dim₀, dim₀+dim₁, ..., total]
    ///
    /// Length = num_hydros + 1
    /// offsets[i] = start index of hydro i's state
    /// offsets[i+1] = end index (exclusive) of hydro i's state
    pub offsets: Vec<usize>,

    /// Total state dimension (sum of all per_hydro_dims)
    pub total_dim: usize,
}

impl StateLayout {
    /// Get the slice range for a hydro's state.
    ///
    /// Returns `start..end` range for indexing into the flattened state vector.
    ///
    /// # Arguments
    ///
    /// * `hydro_id` - Hydro index (0-based)
    ///
    /// # Performance
    ///
    /// O(1) - simple array access
    ///
    /// # Example
    ///
    /// ```ignore
    /// let range = layout.hydro_slice(1);
    /// let hydro1_state = &state[range]; // Extract hydro 1's state
    /// ```
    #[inline]
    pub fn hydro_slice(&self, hydro_id: usize) -> Range<usize> {
        self.offsets[hydro_id]..self.offsets[hydro_id + 1]
    }

    /// Get the dimension for a specific hydro.
    ///
    /// # Performance
    ///
    /// O(1) - array access
    #[inline]
    pub fn hydro_dim(&self, hydro_id: usize) -> usize {
        self.per_hydro_dims[hydro_id]
    }

    /// Get the offset for a specific hydro's storage (first element).
    ///
    /// # Performance
    ///
    /// O(1) - array access
    #[inline]
    pub fn hydro_storage_offset(&self, hydro_id: usize) -> usize {
        self.offsets[hydro_id]
    }

    /// Get the number of lags for a specific hydro.
    ///
    /// Returns dimension - 1 (excluding storage).
    ///
    /// # Performance
    ///
    /// O(1) - array access and subtraction
    #[inline]
    pub fn hydro_lag_count(&self, hydro_id: usize) -> usize {
        self.per_hydro_dims[hydro_id].saturating_sub(1)
    }
}

/// State representation tracking only storage levels.
///
/// This is the simplest state representation where the Markov state consists
/// only of reservoir storage levels at the beginning of each stage.
///
/// # State Layout
///
/// ```text
/// state_coefficients: [V₀, V₁, V₂, ..., Vₙ]
/// ```
///
#[derive(Debug, Clone)]
pub struct StorageState {
    dimension: usize,
    /// The Markov state as a flat vector [V₀, V₁, ..., Vₙ] (storage levels).
    /// This is the single source of truth returned by `coefficients()`.
    state_coefficients: Vec<f64>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    /// DEBUGGING: Iteration number when this state was visited (1-based)
    iteration: usize,
    /// DEBUGGING: Forward pass index that visited this state (0-based handler ID)
    forward_pass_idx: usize,
}

impl StorageState {
    pub fn new(system: &system::System) -> Self {
        Self {
            dimension: system.meta.hydros_count,
            state_coefficients: vec![0.0; system.meta.hydros_count],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
}

impl State for StorageState {
    fn set_dimension(&mut self, dimension: usize) {
        self.dimension = dimension
    }

    fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.forward_pass_idx = forward_pass_idx;
    }

    fn coefficients(&self) -> &[f64] {
        &self.state_coefficients
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>> {
        let mut col_indices = vec![vec![0; 1]; self.dimension];
        for col in &mut col_indices {
            col[0] = pb.add_column(0.0, 0.0..);
        }
        col_indices
    }

    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        _variables: &subproblem::Variables,
    ) {
        // PERFORMANCE: O(1) access - get previous storage from last realization
        let prev_realization = past_realizations.last().unwrap();
        self.state_coefficients
            .clone_from_slice(&prev_realization.final_storage);

        // Update hydro balance RHS: V_{t-1} = state_coefficients
        for (index, row) in constraints.hydro_balance.iter().enumerate() {
            model.change_rows_bounds(
                *row,
                self.state_coefficients[index],
                self.state_coefficients[index],
            );
        }
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        self.state_coefficients
            .clone_from_slice(&realization.final_storage);
    }

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    ) {
        let mut factors =
            Vec::<(usize, f64)>::with_capacity(self.dimension + 1);
        factors.push((variables.alpha, 1.0));
        for (hydro_id, stored_volume) in
            variables.stored_volume.iter().enumerate()
        {
            factors.push((*stored_volume, -cut.coefficients[hydro_id]));
        }
        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        let mut cut_coefficients = vec![0.0; self.dimension];
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);

        // Collect all contributions before accumulating.
        // This ensures deterministic order for Kahan summation regardless
        // of parallel thread completion order in backward pass. Without this,
        // floating-point accumulation order varies across runs, causing cut
        // coefficient drift that compounds through iterations.
        //
        // Memory overhead: num_branchings × dimension f64s per cut
        let mut coef_contributions: Vec<Vec<f64>> =
            Vec::with_capacity(branching_realizations.len());
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];

            // Store contributions instead of accumulating immediately
            let contrib: Vec<f64> = realization
                .water_value
                .iter()
                .map(|&val| prob * val)
                .collect();
            coef_contributions.push(contrib);
            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        // Deterministic accumulation using Kahan summation
        for hydro_idx in 0..cut_coefficients.len() {
            let values: Vec<f64> = coef_contributions
                .iter()
                .map(|contrib| contrib[hydro_idx])
                .collect();
            cut_coefficients[hydro_idx] = utils::kahan_sum(&values);
        }
        let objective = utils::kahan_sum(&objective_contributions);

        let last_realization = forward_trajectory.last().unwrap();

        let cut_rhs = objective
            - utils::dot_product(
                &cut_coefficients,
                &last_realization.final_storage,
            );
        // Temporary sets cut id to 0 - will be updated when adding to pool
        // Use state's tracking information for iteration and forward_pass_idx
        cut::BendersCut::new(
            0,
            cut_coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

/// State representation including both storage volumes and lagged inflows.
///
/// # State Vector Definition
///
/// The full state vector is: `[V_t, Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]`
/// where:
/// - `V_t`: Final storage volumes at time t (dimension: n_hydros)
/// - `Y_{t-k}`: Lagged inflow realizations (dimension: n_hydros each)
/// - `p`: Lag order from the stochastic process
///
/// Total state dimension: `n + p×n` where n = number of hydros
///
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    dimension: usize,
    layout: StateLayout,
    state_coefficients: Vec<f64>,
    transformed_coefficients: Vec<Vec<f64>>,
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}

impl StorageAndInflowState {
    pub fn new(
        system: &system::System,
        uncertainty_models: &[crate::temporal_model::TemporalModel],
    ) -> Self {
        let dimension = system.meta.hydros_count;

        let per_hydro_dims: Vec<usize> =
            per_hydro_state_dims(system, uncertainty_models, 0);

        // Build cumulative offsets: [0, dim₀, dim₀+dim₁, ...]
        let mut offsets = Vec::with_capacity(dimension + 1);
        offsets.push(0);
        let mut cumsum = 0;
        for &dim in &per_hydro_dims {
            cumsum += dim;
            offsets.push(cumsum);
        }

        let layout = StateLayout {
            per_hydro_dims,
            offsets,
            total_dim: cumsum,
        };

        let transformed_coefficients = Self::extract_transformed_coefficients(
            system,
            uncertainty_models,
            0,
        );

        Self {
            dimension,
            layout,
            state_coefficients: vec![0.0; cumsum],
            transformed_coefficients,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    /// Extract transformed (observation-space) AR coefficients from uncertainty models
    ///
    /// Computes ψ_i = φ_i * (σ_t / σ_{t-i}) for each hydro and lag, where:
    /// - φ_i: residual-space AR coefficient from PAR statistical model
    /// - σ_t: standard deviation of current season
    /// - σ_{t-i}: standard deviation of lag season
    ///
    /// # Mathematical Foundation
    ///
    /// The PAR model operates in residual space: Z'_t = Σ φ_i * Z'_{t-i} + ε_t
    /// where Z'_t = (Y_t - μ_t) / σ_t
    ///
    /// Transforming to observation space gives: Y_t = Σ ψ_i * Y_{t-i} + η_t
    /// where ψ_i = φ_i * (σ_t / σ_{t-i}) (Equation 7 in par_derivation.pdf)
    ///
    /// The LP constraints use ψ coefficients, so Benders cuts must also use ψ
    /// for mathematical consistency. Using φ directly would be incorrect.
    ///
    fn extract_transformed_coefficients(
        system: &system::System,
        uncertainty_models: &[crate::temporal_model::TemporalModel],
        season_id: usize,
    ) -> Vec<Vec<f64>> {
        let mut coeffs = vec![Vec::new(); system.meta.hydros_count];

        for model in uncertainty_models.iter() {
            if model.entity_type() != UncertaintyType::Inflow {
                continue;
            }

            let hydro_id = model.entity_id();

            if !model.is_autoregressive() {
                // Independent model
                coeffs[hydro_id] = vec![];
            } else {
                // PAR model
                let phi = &model.ar_coefficients[season_id]; // φ_i
                let current_params = model.seasonal_params(season_id);
                let ar_order = phi.len();
                let num_seasons = model.num_seasons;

                // Compute ψ_i = φ_i * (σ_t / σ_{t-i})
                let mut psi = Vec::with_capacity(ar_order);
                for (i, &phi_coef) in phi.iter().enumerate() {
                    let lag_offset = i + 1;
                    let lag_season = if num_seasons == 1 {
                        0
                    } else if season_id >= lag_offset {
                        season_id - lag_offset
                    } else {
                        num_seasons - (lag_offset - season_id)
                    };

                    let lag_params = model.seasonal_params(lag_season);
                    let psi_i = phi_coef
                        * (current_params.std_dev / lag_params.std_dev);
                    psi.push(psi_i);
                }

                coeffs[hydro_id] = psi;
            }
        }

        coeffs
    }

    pub fn get_lag_order(&self) -> usize {
        self.layout
            .per_hydro_dims
            .iter()
            .map(|&dim| dim.saturating_sub(1)) // dim = 1 + lag_count
            .max()
            .unwrap_or(0)
    }

    pub fn get_total_dimension(&self) -> usize {
        self.layout.total_dim
    }

    /// Extract storage values from trajectory
    ///
    /// Gets storage from the last realization in the trajectory.
    /// This is O(n) due to the clone operation.
    fn extract_storage_from_trajectory(
        &self,
        trajectory: &[&subproblem::Realization],
    ) -> Vec<f64> {
        let prev = trajectory.last().unwrap();
        prev.final_storage.clone()
    }

    /// Extract lagged inflows from trajectory using O(p) window
    ///
    /// For each hydro, extracts the last `lag_count` inflows from the trajectory,
    /// building a vector of lagged values. Handles cases where trajectory is
    /// shorter than the required lag count by padding with zeros.
    fn extract_lags_from_trajectory(
        &self,
        trajectory: &[&subproblem::Realization],
    ) -> Vec<Vec<f64>> {
        let traj_len = trajectory.len();
        let mut lags = Vec::with_capacity(self.dimension);

        for hydro_id in 0..self.dimension {
            let lag_count = self.layout.hydro_lag_count(hydro_id);
            let mut hydro_lags = Vec::with_capacity(lag_count);

            for lag_idx in 0..lag_count {
                let hist_idx = traj_len.saturating_sub(1 + lag_idx);
                if hist_idx < traj_len {
                    hydro_lags.push(trajectory[hist_idx].inflow[hydro_id]);
                } else {
                    hydro_lags.push(0.0); // Fallback for insufficient history
                }
            }
            lags.push(hydro_lags);
        }

        lags
    }

    /// Rebuild state_coefficients from storage and lags
    ///
    /// Layout: [V₀, ..., Vₙ, Y₀⁽¹⁾, Y₀⁽²⁾, ..., Yₙ⁽ᵖ⁾]
    ///
    /// This method repopulates `state_coefficients` in-place from separate
    /// storage and lag vectors following the heterogeneous layout.
    fn rebuild_state_coefficients(
        &mut self,
        storage: &[f64],
        lags: &[Vec<f64>],
    ) {
        for hydro_id in 0..self.dimension {
            let offset = self.layout.offsets[hydro_id];

            // Storage
            self.state_coefficients[offset] = storage[hydro_id];

            // Lags
            let lag_count = self.layout.hydro_lag_count(hydro_id);
            if lag_count > 0 {
                let lag_start = offset + 1;
                let lag_end = lag_start + lag_count;
                self.state_coefficients[lag_start..lag_end]
                    .copy_from_slice(&lags[hydro_id]);
            }
        }
    }
}

impl State for StorageAndInflowState {
    fn set_dimension(&mut self, dimension: usize) {
        self.dimension = dimension;
    }

    fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.forward_pass_idx = forward_pass_idx;
    }

    fn has_lagged_observation_state(&self) -> bool {
        true
    }

    fn get_lagged_observations(&self, entity_idx: usize) -> &[f64] {
        // Extract lagged observations from state_coefficients for a specific hydro
        // entity_idx is treated as hydro_id for this legacy state
        if entity_idx >= self.dimension {
            return &[];
        }

        let offset = self.layout.offsets[entity_idx];
        let lag_count = self.layout.hydro_lag_count(entity_idx);

        if lag_count == 0 {
            &[]
        } else {
            let lag_start = offset + 1; // Skip storage
            let lag_end = lag_start + lag_count;
            &self.state_coefficients[lag_start..lag_end]
        }
    }

    fn coefficients(&self) -> &[f64] {
        &self.state_coefficients
    }

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>> {
        let max_lag_count = self
            .layout
            .per_hydro_dims
            .iter()
            .map(|&dim| dim.saturating_sub(1))
            .max()
            .unwrap_or(0);

        let num_var_types = if max_lag_count == 0 {
            1
        } else {
            1 + max_lag_count
        };

        let mut variable_indices = Vec::with_capacity(num_var_types);

        for var_idx in 0..num_var_types {
            let mut hydro_vars = Vec::with_capacity(self.dimension);

            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);

                if var_idx == 0 || (var_idx <= hydro_lag_count) {
                    let var = pb.add_column(0.0, 0.0..f64::INFINITY);
                    hydro_vars.push(var);
                } else {
                    hydro_vars.push(0);
                }
            }
            variable_indices.push(hydro_vars);
        }

        variable_indices
    }

    fn update_from_trajectory(
        &mut self,
        past_realizations: &[&subproblem::Realization],
        model: &mut solver::Model,
        constraints: &subproblem::Constraints,
        variables: &subproblem::Variables,
    ) {
        // Extract from trajectory (source of truth)
        let storage = self.extract_storage_from_trajectory(past_realizations);
        let lags = self.extract_lags_from_trajectory(past_realizations);

        // Rebuild state coefficients
        self.rebuild_state_coefficients(&storage, &lags);

        // Update LP bounds (storage)
        for (index, row) in constraints.hydro_balance.iter().enumerate() {
            model.change_rows_bounds(*row, storage[index], storage[index]);
        }

        // TICKET-005: Update lag values using variable bounds (legacy approach)
        // NOTE: When using the unified approach (new_from_temporal_models),
        // lag_fixing_constraints are managed entirely by subproblem's
        // update_lag_fixing_constraints() via uncertainty_manager.
        // The state object should only update variable bounds for backward compatibility.
        if let Some(lag_vars) = &variables.lagged_state {
            // Check if this is the OLD state-managed approach (no lag_fixing_constraints)
            // or if lag_fixing_constraints don't exist
            let should_update_bounds =
                constraints.lag_fixing_constraints.is_none();

            if should_update_bounds {
                // Bounds-based approach (legacy): Update variable bounds
                // Variable: lag_var with bounds [value, value]
                for hydro_id in 0..self.dimension {
                    let lag_count = self.layout.hydro_lag_count(hydro_id);
                    for lag_idx in 0..lag_count {
                        if lag_idx < lag_vars[hydro_id].len() {
                            let var = lag_vars[hydro_id][lag_idx];
                            let value = lags[hydro_id][lag_idx];
                            model.change_column_bounds(var, value, value);
                        }
                    }
                }
            }
        }
    }

    fn update_with_current_realization(
        &mut self,
        realization: &subproblem::Realization,
    ) {
        // Extract current storage
        let storage = realization.final_storage.clone();

        // Rotate lags: [Y_t, Y_{t-1}, ...] becomes [Y_{t+1}, Y_t, ...]
        // Current inflow becomes newest lag
        let mut lags = Vec::with_capacity(self.dimension);
        for hydro_id in 0..self.dimension {
            let lag_count = self.layout.hydro_lag_count(hydro_id);
            if lag_count > 0 {
                let offset = self.layout.offsets[hydro_id];
                let lag_start = offset + 1;
                let lag_end = lag_start + lag_count;

                let mut new_lags = vec![realization.inflow[hydro_id]]; // New lag
                new_lags.extend_from_slice(
                    &self.state_coefficients[lag_start..lag_end - 1], // Shift old lags
                );
                lags.push(new_lags);
            } else {
                lags.push(Vec::new());
            }
        }

        // Rebuild state coefficients
        self.rebuild_state_coefficients(&storage, &lags);
    }

    fn add_cut_constraint_to_model(
        &mut self,
        cut: &mut cut::BendersCut,
        variables: &subproblem::Variables,
        model: &mut solver::Model,
    ) {
        // Total vars = alpha (1) + storage (n) + all lags (per-hydro variable)
        let total_vars = 1 + self.layout.total_dim;
        let mut factors = Vec::<(usize, f64)>::with_capacity(total_vars);

        factors.push((variables.alpha, 1.0));

        // Storage coefficients (first n coefficients)
        for (hydro_id, &coef) in
            cut.coefficients[0..self.dimension].iter().enumerate()
        {
            factors.push((variables.stored_volume[hydro_id], -coef));
        }

        // Lag coefficients (per-hydro variable count)
        // NOTE: lag_vars is indexed by entity (loads + inflows), but cut coefficients
        // are indexed by hydro. We need to skip non-inflow entities.
        let mut coef_idx = self.dimension;
        if let Some(lag_vars) = &variables.lagged_state {
            let mut hydro_count = 0;
            eprintln!(
                "[DEBUG CUT] Building cut with {} lag_vars entities",
                lag_vars.len()
            );
            for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
                // Skip load entities (they have empty lag vectors or we skip them)
                // Only process inflow entities up to self.dimension (num_hydros)
                if hydro_count >= self.dimension {
                    break;
                }

                // Check if this entity has lags matching our expected hydro layout
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_count);
                eprintln!("[DEBUG CUT]   Entity {}: {} lags, expected {} for hydro {}", 
                         entity_idx, entity_lags.len(), hydro_lag_count, hydro_count);
                if entity_lags.len() == hydro_lag_count {
                    // This looks like a hydro entity
                    for lag_idx in 0..hydro_lag_count {
                        let lag_var = entity_lags[lag_idx];
                        eprintln!(
                            "[DEBUG CUT]     Adding lag var {} with coef {}",
                            lag_var, cut.coefficients[coef_idx]
                        );
                        factors.push((lag_var, -cut.coefficients[coef_idx]));
                        coef_idx += 1;
                    }
                    hydro_count += 1;
                } else if entity_lags.is_empty() {
                    // This is likely a load entity (no lags), skip it
                    eprintln!("[DEBUG CUT]     Skipping (empty, likely load)");
                    continue;
                } else {
                    // Unexpected lag count, skip
                    eprintln!("[DEBUG CUT]     Skipping (unexpected count)");
                    continue;
                }
            }
        }

        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        _forward_trajectory: &[&subproblem::Realization],
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        let costs: Vec<f64> = branching_realizations
            .iter()
            .map(|r| r.total_stage_objective)
            .collect();
        let num_branchings = costs.len();
        let probabilities = utils::uniform_prob_by_count(num_branchings);
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(&probabilities, &costs);

        // Total coefficients = storage (n) + all lags (per-hydro variable)
        let total_coefficients = self.layout.total_dim;
        let mut coef_contributions: Vec<Vec<f64>> =
            Vec::with_capacity(branching_realizations.len());
        let mut objective_contributions: Vec<f64> =
            Vec::with_capacity(branching_realizations.len());

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];
            let mut contrib = Vec::with_capacity(total_coefficients);

            // Storage coefficients (water values)
            contrib
                .extend(realization.water_value.iter().map(|&val| prob * val));

            // Lag coefficients computation (TICKET-007)
            //
            // Two approaches based on LP formulation:
            //
            // 1. **Explicit constraints** (use_explicit_lag_constraints=true):
            //    - LP has equality constraints: Y_{t-k} = lag_value
            //    - Duals extracted from these constraints directly give ∂FO/∂Y_{t-k}
            //    - Coefficient: prob * lag_dual (no transformation needed)
            //    - Detection: lag_duals[hydro_id].len() == hydro_lag_count
            //
            // 2. **Bounds-based** (legacy, use_explicit_lag_constraints=false):
            //    - LP fixes lags via variable bounds
            //    - Duals from AR observation constraints need chain rule
            //    - Coefficient: prob * water_value * ψ_j (transformation needed)
            //    - Detection: lag_duals[hydro_id].len() != hydro_lag_count
            //
            for hydro_id in 0..self.dimension {
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);

                if hydro_lag_count == 0 {
                    continue; // No lags for this hydro
                }

                // Detect which approach based on lag_duals structure
                let use_explicit_constraints = !realization
                    .lag_duals
                    .is_empty()
                    && hydro_id < realization.lag_duals.len()
                    && realization.lag_duals[hydro_id].len() == hydro_lag_count;

                if use_explicit_constraints {
                    // NEW: Direct from lag-fixing constraint duals
                    // Each lag has its own constraint: Y_{t-k} = value
                    // The dual of this constraint is directly ∂FO/∂Y_{t-k}
                    for lag_idx in 0..hydro_lag_count {
                        let lag_dual = realization.lag_duals[hydro_id][lag_idx];
                        contrib.push(prob * lag_dual);
                    }
                    eprintln!(
                        "[DEBUG CUT] Hydro {}: Using explicit lag duals: {:?}",
                        hydro_id, realization.lag_duals[hydro_id]
                    );
                } else {
                    // LEGACY: Chain rule with AR coefficients
                    // Lags are fixed via variable bounds, need transformation
                    let water_val = realization.water_value[hydro_id];

                    for lag_idx in 0..hydro_lag_count {
                        let psi_j =
                            self.transformed_coefficients[hydro_id][lag_idx];
                        let lag_coef = water_val * psi_j;
                        contrib.push(prob * lag_coef);
                    }
                }
            }

            coef_contributions.push(contrib);
            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        let mut cut_coefficients = vec![0.0; total_coefficients];
        for coef_idx in 0..total_coefficients {
            let values: Vec<f64> = coef_contributions
                .iter()
                .map(|contrib| contrib[coef_idx])
                .collect();
            cut_coefficients[coef_idx] = utils::kahan_sum(&values);
        }
        let objective = utils::kahan_sum(&objective_contributions);

        let state_coefficients = self.coefficients();

        let cut_rhs = objective
            - utils::dot_product(&cut_coefficients, state_coefficients);

        cut::BendersCut::new(
            0,
            cut_coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    fn clone_dyn(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }
}

/// Factory function to create state representations for SDDP subproblems.
pub fn factory(
    kind: &str,
    system: &system::System,
    uncertainty_models: &[crate::temporal_model::TemporalModel],
) -> Box<dyn State> {
    match kind {
        "storage" => Box::new(StorageState::new(system)),
        "storage_and_inflow" => Box::new(StorageAndInflowState::new(
            system,
            uncertainty_models,
        )),
        _ => panic!(
            "Unknown state_choice: '{}'. Valid options: 'storage', 'storage_and_inflow'",
            kind
        ),
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::input;
    use crate::system;
    use crate::uncertainty_model;

    // Helper to convert Vec<UncertaintyModel> to Vec<TemporalModel>
    fn convert_models(
        models: &[uncertainty_model::UncertaintyModel],
    ) -> Vec<crate::temporal_model::TemporalModel> {
        models.iter().map(|m| m.to_temporal_model()).collect()
    }

    #[test]
    fn test_new_storage_state() {
        let system = system::System::default();
        // StorageState::new() only takes system, no uncertainty models needed
        let state = StorageState::new(&system);
        assert_eq!(state.dimension, 1);
        assert_eq!(state.state_coefficients, vec![0.0]);
        assert_eq!(state.dominating_objective, 0.0);
        assert_eq!(state.dominating_cut_id, 0);
    }

    #[test]
    fn test_factory_storage_state() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let state =
            factory("storage", &system, &convert_models(&uncertainty_models));
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    fn test_factory_storage_and_inflow_state() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let state = factory(
            "storage_and_inflow",
            &system,
            &convert_models(&uncertainty_models),
        );

        // With Independent process (lag_order=0), dimension should be n(1+0) = n
        // Default system has 1 hydro, so dimension = 1
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    #[should_panic(
        expected = "Unknown state_choice: 'invalid'. Valid options: 'storage', 'storage_and_inflow'"
    )]
    fn test_factory_invalid_choice() {
        let system = system::System::default();
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];
        let _ =
            factory("invalid", &system, &convert_models(&uncertainty_models));
    }

    #[test]
    fn test_factory_preserves_system_dimension() {
        // Test with multi-hydro system
        let mut system = system::System::default();
        // Add 2 more hydros to have 3 total
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        // Create one Independent model per hydro (3 hydros)
        let uncertainty_models: Vec<_> = (0..3)
            .map(|_| uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 10.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            })
            .collect();

        let state_storage =
            factory("storage", &system, &convert_models(&uncertainty_models));
        assert_eq!(state_storage.coefficients().len(), 3);

        // Create fresh models for second test
        let uncertainty_models2: Vec<_> = (0..3)
            .map(|id| uncertainty_model::UncertaintyModel::Independent {
                entity_id: id,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 10.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            })
            .collect();

        let state_inflow = factory(
            "storage_and_inflow",
            &system,
            &convert_models(&uncertainty_models2),
        );
        // With lag_order=0, dimension is n(1+0) = 3
        assert_eq!(state_inflow.coefficients().len(), 3);
    }

    #[test]
    fn test_ar_coefficient_application_logic() {
        // Test the logic of how AR coefficients would be applied in constraints
        // This tests the algorithm without creating actual solver objects

        // Scenario: Hydro with AR(2) model, coefficients [0.6, 0.3]
        let ar_coeffs = [0.6, 0.3];
        let lag_count = 2;

        // Expected constraint terms: inflow_noise - 0.6*lag[0] - 0.3*lag[1] = white_noise
        // This means coefficients should be negated when added to constraint
        assert_eq!(ar_coeffs.len(), lag_count);

        // Verify that we have the right number of lag terms
        let expected_constraint_terms = 1 + lag_count; // inflow_noise + 2 lags
        assert_eq!(expected_constraint_terms, 3);

        // Verify coefficient signs in constraint (should be negative)
        for (idx, &coeff) in ar_coeffs.iter().enumerate() {
            assert!(coeff > 0.0, "AR coefficient {} should be positive", idx);
            // In constraint, it becomes: -coeff * lag_var
        }
    }

    #[test]
    fn test_heterogeneous_ar_orders_logic() {
        // Test the logic for systems with heterogeneous AR orders
        // Tests constraint structure without creating actual solver objects

        // 3 hydros: AR(2), AR(1), Independent
        let ar_specs = [
            vec![0.7, 0.2], // Hydro 0: AR(2)
            vec![0.8],      // Hydro 1: AR(1)
            vec![],         // Hydro 2: Independent
        ];

        // Expected constraint counts per hydro:
        // Hydro 0: AR RHS + equality + 2 lags = 4 constraints
        // Hydro 1: AR RHS + equality + 1 lag = 3 constraints
        // Hydro 2: simple RHS + equality + 0 lags = 2 constraints

        let expected_counts = [4, 3, 2];

        for (hydro_idx, ar_coeffs) in ar_specs.iter().enumerate() {
            let lag_count = ar_coeffs.len();
            let constraint_count = 2 + lag_count; // RHS + equality + lags
            assert_eq!(
                constraint_count, expected_counts[hydro_idx],
                "Hydro {} should have {} constraints",
                hydro_idx, expected_counts[hydro_idx]
            );
        }
    }

    // ========================================================================
    // Tests for [AR-PSI-001]: Transformed Coefficients (ψ from φ)
    // ========================================================================

    #[test]
    fn test_transformed_coefficients_uniform_sigma() {
        // When all σ are equal, ψ should equal φ
        let system = create_system_with_hydros(1);
        let phi = vec![0.8, 0.3];

        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(
            &system,
            &convert_models(&uncertainty_models),
        );
        let psi = &state.transformed_coefficients[0];

        // ψ = φ × (σ_t / σ_{t-i}) = φ × (10 / 10) = φ
        assert_eq!(psi.len(), phi.len());
        for i in 0..phi.len() {
            assert!(
                (psi[i] - phi[i]).abs() < 1e-12,
                "ψ[{}] = {} should equal φ[{}] = {} when σ is uniform",
                i,
                psi[i],
                i,
                phi[i]
            );
        }
    }

    #[test]
    fn test_transformed_coefficients_seasonal_variance() {
        // Test ψ = φ × (σ_t / σ_{t-i}) with seasonal variance
        let system = create_system_with_hydros(1);
        let phi = vec![0.7];

        // Season 0: σ = 50, Season 1: σ = 100
        // For season 1: ψ = 0.7 × (100 / 50) = 1.4
        let uncertainty_models =
            vec![create_par_model_seasonal_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(
            &system,
            &convert_models(&uncertainty_models),
        );
        let psi = &state.transformed_coefficients[0];

        assert_eq!(psi.len(), 1);
        // Using season_id = 0 in constructor, so:
        // ψ[0] = φ[0] × (σ_0 / σ_{11}) = 0.7 × (50 / 100) = 0.35
        let expected_psi = 0.7 * (50.0 / 100.0);
        assert!(
            (psi[0] - expected_psi).abs() < 1e-10,
            "ψ[0] = {} should be {} (φ × σ_t/σ_{{t-1}})",
            psi[0],
            expected_psi
        );
    }

    #[test]
    fn test_transformed_coefficients_ar2_seasonal() {
        // Test AR(2) with seasonal variance
        let system = create_system_with_hydros(1);
        let phi = vec![0.8, 0.3];

        let uncertainty_models =
            vec![create_par_model_seasonal_sigma(0, phi.clone())];

        let state = StorageAndInflowState::new(
            &system,
            &convert_models(&uncertainty_models),
        );
        let psi = &state.transformed_coefficients[0];

        assert_eq!(psi.len(), 2);

        // Season 0: σ_0 = 50 (even index)
        // Lag 0 (t-1): season 11, σ_{11} = 100 (odd index)
        // Lag 1 (t-2): season 10, σ_{10} = 50 (even index)
        // ψ[0] = φ[0] × (σ_0 / σ_{11}) = 0.8 × (50 / 100) = 0.4
        // ψ[1] = φ[1] × (σ_0 / σ_{10}) = 0.3 × (50 / 50) = 0.3

        let expected_psi_0 = 0.8 * (50.0 / 100.0);
        let expected_psi_1 = 0.3 * (50.0 / 50.0); // Same σ, so no transformation

        assert!(
            (psi[0] - expected_psi_0).abs() < 1e-10,
            "ψ[0] = {} should be {}",
            psi[0],
            expected_psi_0
        );
        assert!(
            (psi[1] - expected_psi_1).abs() < 1e-10,
            "ψ[1] = {} should be {}",
            psi[1],
            expected_psi_1
        );
    }

    #[test]
    fn test_transformed_coefficients_independent_model() {
        // Independent model should have empty transformed coefficients
        let system = create_system_with_hydros(1);
        let uncertainty_models =
            vec![uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            }];

        let state = StorageAndInflowState::new(
            &system,
            &convert_models(&uncertainty_models),
        );
        assert!(state.transformed_coefficients[0].is_empty());
    }

    #[test]
    fn test_transformed_coefficients_mixed_system() {
        // Test heterogeneous system with different AR orders
        let system = create_system_with_hydros(3);
        let phi1 = vec![0.8];
        let phi2 = vec![0.7, 0.2];

        let uncertainty_models = vec![
            // Hydro 0: Independent (no AR)
            uncertainty_model::UncertaintyModel::Independent {
                entity_id: 0,
                entity_type: input::UncertaintyType::Inflow,
                seasonal_params: vec![uncertainty_model::SeasonalParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    distribution: uncertainty_model::DistributionType::Normal,
                }],
            },
            // Hydro 1: AR(1)
            create_par_model_uniform_sigma(1, phi1.clone()),
            // Hydro 2: AR(2)
            create_par_model_uniform_sigma(2, phi2.clone()),
        ];

        let state = StorageAndInflowState::new(
            &system,
            &convert_models(&uncertainty_models),
        );

        // Hydro 0: empty
        assert!(state.transformed_coefficients[0].is_empty());

        // Hydro 1: AR(1), uniform σ → ψ = φ
        assert_eq!(state.transformed_coefficients[1].len(), 1);
        assert!((state.transformed_coefficients[1][0] - phi1[0]).abs() < 1e-12);

        // Hydro 2: AR(2), uniform σ → ψ = φ
        assert_eq!(state.transformed_coefficients[2].len(), 2);
        for (i, &phi) in phi2.iter().enumerate() {
            assert!((state.transformed_coefficients[2][i] - phi).abs() < 1e-12);
        }
    }

    // Helper function to create system with n hydros
    fn create_system_with_hydros(n: usize) -> system::System {
        let mut system = system::System::default();
        system.hydros.clear();
        for i in 0..n {
            system.hydros.push(system::Hydro::new(
                i, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
            ));
        }
        system.meta.hydros_count = n;
        system
    }

    // Helper to create PAR model with uniform σ (all seasons same std_dev)
    fn create_par_model_uniform_sigma(
        entity_id: usize,
        phi: Vec<f64>,
    ) -> uncertainty_model::UncertaintyModel {
        let ar_order = phi.len();
        uncertainty_model::UncertaintyModel::PeriodicAR {
            entity_id,
            entity_type: input::UncertaintyType::Inflow,
            par_params: uncertainty_model::PARParams {
                num_seasons: 1,
                ar_orders: vec![ar_order],
                ar_coefficients: vec![phi],
                seasonal_means: vec![100.0],
                seasonal_stds: vec![10.0], // Uniform σ
                seasonal_distributions: vec![
                    uncertainty_model::DistributionType::Normal,
                ],
                max_ar_order: ar_order,
            },
        }
    }

    // Helper to create PAR model with seasonal σ variance
    fn create_par_model_seasonal_sigma(
        entity_id: usize,
        phi: Vec<f64>,
    ) -> uncertainty_model::UncertaintyModel {
        let ar_order = phi.len();
        // Create 12 seasons with alternating σ: 50, 100, 50, 100, ...
        let seasonal_stds: Vec<f64> = (0..12)
            .map(|i| if i % 2 == 0 { 50.0 } else { 100.0 })
            .collect();

        uncertainty_model::UncertaintyModel::PeriodicAR {
            entity_id,
            entity_type: input::UncertaintyType::Inflow,
            par_params: uncertainty_model::PARParams {
                num_seasons: 12,
                ar_orders: vec![ar_order; 12],
                ar_coefficients: vec![phi; 12],
                seasonal_means: vec![100.0; 12],
                seasonal_stds,
                seasonal_distributions:
                    vec![uncertainty_model::DistributionType::Normal; 12],
                max_ar_order: ar_order,
            },
        }
    }

    // Helper to create Independent model (AR(0))
    fn create_independent_model(
        entity_id: usize,
    ) -> uncertainty_model::UncertaintyModel {
        uncertainty_model::UncertaintyModel::Independent {
            entity_id,
            entity_type: input::UncertaintyType::Inflow,
            seasonal_params: vec![uncertainty_model::SeasonalParams {
                mean: 100.0,
                std_dev: 10.0,
                distribution: uncertainty_model::DistributionType::Normal,
            }],
        }
    }

    // TICKET-007: Tests for simplified cut generation with explicit constraints

    /// Test that direct lag coefficients are used when lag_duals has correct structure
    #[test]
    fn test_evaluate_cut_uses_direct_lag_duals_when_available() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5, 0.3])];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create a realization with lag_duals structured for explicit constraints
        // For explicit constraints: lag_duals[hydro_id].len() == lag_count (2 in this case)
        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0];
        realization.lag_duals = vec![vec![2.0, 3.0]]; // Two lag duals for AR(2)
        realization.total_stage_objective = 100.0;
        realization.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&realization];
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // With explicit constraints and prob=1.0:
        // cut_coefficients = [water_value, lag_dual_0, lag_dual_1]
        //                  = [10.0, 2.0, 3.0]
        assert_eq!(cut.coefficients.len(), 3); // storage + 2 lags
        assert!(
            (cut.coefficients[0] - 10.0).abs() < 1e-10,
            "Storage coefficient should be water_value"
        );
        assert!(
            (cut.coefficients[1] - 2.0).abs() < 1e-10,
            "First lag coefficient should be lag_dual[0]"
        );
        assert!(
            (cut.coefficients[2] - 3.0).abs() < 1e-10,
            "Second lag coefficient should be lag_dual[1]"
        );
    }

    /// Test that chain rule is used when lag_duals has legacy structure
    #[test]
    fn test_evaluate_cut_uses_chain_rule_for_legacy_structure() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5, 0.3])];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create a realization with lag_duals structured for legacy bounds approach
        // For legacy: lag_duals[hydro_id].len() == 1 (only AR constraint dual)
        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0];
        realization.lag_duals = vec![vec![5.0]]; // Single AR dual (legacy)
        realization.total_stage_objective = 100.0;
        realization.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&realization];
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // With legacy approach and prob=1.0:
        // cut_coefficients = [water_value, water_value * psi_0, water_value * psi_1]
        // state.transformed_coefficients for AR(2) with phi=[0.5, 0.3] would be calculated
        assert_eq!(cut.coefficients.len(), 3); // storage + 2 lags
        assert!(
            (cut.coefficients[0] - 10.0).abs() < 1e-10,
            "Storage coefficient should be water_value"
        );

        // Lag coefficients should be water_value * transformed_coefficients[hydro_id][lag_idx]
        let expected_lag0 = 10.0 * state.transformed_coefficients[0][0];
        let expected_lag1 = 10.0 * state.transformed_coefficients[0][1];
        assert!(
            (cut.coefficients[1] - expected_lag0).abs() < 1e-10,
            "First lag coefficient should use chain rule"
        );
        assert!(
            (cut.coefficients[2] - expected_lag1).abs() < 1e-10,
            "Second lag coefficient should use chain rule"
        );
    }

    /// Test that cut evaluation handles empty lag_duals (independent model)
    #[test]
    fn test_evaluate_cut_handles_empty_lag_duals() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5])];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create a realization with empty lag_duals (independent model fallback)
        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0];
        realization.lag_duals = vec![]; // Empty (no AR dynamics)
        realization.total_stage_objective = 100.0;
        realization.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&realization];
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // Should fall back to chain rule with water_value only
        assert_eq!(cut.coefficients.len(), 2); // storage + 1 lag
        assert!((cut.coefficients[0] - 10.0).abs() < 1e-10);

        // Lag coefficient uses chain rule with only water_value
        let expected_lag = 10.0 * state.transformed_coefficients[0][0];
        assert!((cut.coefficients[1] - expected_lag).abs() < 1e-10);
    }

    /// Test cut generation with multiple branching realizations
    #[test]
    fn test_evaluate_cut_multiple_realizations_explicit_constraints() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5])];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create multiple realizations with explicit constraint structure
        let mut r1 = subproblem::Realization::default();
        r1.water_value = vec![10.0];
        r1.lag_duals = vec![vec![2.0]]; // Explicit constraint
        r1.total_stage_objective = 100.0;
        r1.final_storage = vec![50.0];

        let mut r2 = subproblem::Realization::default();
        r2.water_value = vec![12.0];
        r2.lag_duals = vec![vec![3.0]]; // Explicit constraint
        r2.total_stage_objective = 110.0;
        r2.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&r1];
        let branching_realizations = vec![r1.clone(), r2.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // With uniform probabilities (0.5 each):
        // storage_coef = 0.5 * 10.0 + 0.5 * 12.0 = 11.0
        // lag_coef = 0.5 * 2.0 + 0.5 * 3.0 = 2.5
        assert_eq!(cut.coefficients.len(), 2);
        assert!(
            (cut.coefficients[0] - 11.0).abs() < 1e-10,
            "Storage coefficient averaged"
        );
        assert!(
            (cut.coefficients[1] - 2.5).abs() < 1e-10,
            "Lag coefficient averaged"
        );
    }

    /// Test that cut RHS calculation is correct with lag coefficients
    #[test]
    fn test_evaluate_cut_rhs_calculation_with_lags() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let uncertainty_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5])];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Set state coefficients to known values
        state.state_coefficients = vec![50.0, 100.0]; // [storage, lag]

        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0];
        realization.lag_duals = vec![vec![2.0]]; // Explicit constraint
        realization.total_stage_objective = 1000.0;
        realization.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&realization];
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // cut_rhs = objective - dot_product(cut_coef, state_coef)
        //         = 1000.0 - (10.0 * 50.0 + 2.0 * 100.0)
        //         = 1000.0 - (500.0 + 200.0)
        //         = 1000.0 - 700.0 = 300.0
        assert!(
            (cut.rhs - 300.0).abs() < 1e-10,
            "Cut RHS should be correctly calculated"
        );
    }

    /// Test heterogeneous AR orders with explicit constraints
    #[test]
    fn test_evaluate_cut_heterogeneous_ar_orders() {
        use crate::risk_measure;
        use crate::subproblem;
        use crate::system;

        // Create system with 3 hydros
        let mut system = system::System::default();
        system.hydros.push(system::Hydro::new(
            1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.hydros.push(system::Hydro::new(
            2, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
        ));
        system.meta.hydros_count = 3;

        // AR(0), AR(1), AR(2) - heterogeneous orders
        let uncertainty_models = vec![
            create_independent_model(0), // Independent (no lags)
            create_par_model_uniform_sigma(1, vec![0.5]), // AR(1)
            create_par_model_uniform_sigma(2, vec![0.5, 0.3]), // AR(2)
        ];
        let temporal_models = convert_models(&uncertainty_models);

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create realization with explicit constraint structure
        // lag_duals[0] = [] (no lags)
        // lag_duals[1] = [dual1] (1 lag)
        // lag_duals[2] = [dual2_0, dual2_1] (2 lags)
        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0, 20.0, 30.0];
        realization.lag_duals = vec![
            vec![],         // AR(0) - no lags
            vec![2.0],      // AR(1) - 1 lag
            vec![3.0, 4.0], // AR(2) - 2 lags
        ];
        realization.total_stage_objective = 1000.0;
        realization.final_storage = vec![50.0, 60.0, 70.0];

        let risk_measure = risk_measure::Expectation {};
        let forward_trajectory = vec![&realization];
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(
            &risk_measure,
            &forward_trajectory,
            &branching_realizations,
        );

        // Total coefficients: 3 storage + 0 + 1 + 2 = 6
        assert_eq!(
            cut.coefficients.len(),
            6,
            "Should have correct total dimension"
        );

        // Storage coefficients
        assert!(
            (cut.coefficients[0] - 10.0).abs() < 1e-10,
            "Hydro 0 storage"
        );
        assert!(
            (cut.coefficients[1] - 20.0).abs() < 1e-10,
            "Hydro 1 storage"
        );
        assert!(
            (cut.coefficients[2] - 30.0).abs() < 1e-10,
            "Hydro 2 storage"
        );

        // Lag coefficients (direct from lag_duals with explicit constraints)
        assert!((cut.coefficients[3] - 2.0).abs() < 1e-10, "Hydro 1 lag 1");
        assert!((cut.coefficients[4] - 3.0).abs() < 1e-10, "Hydro 2 lag 1");
        assert!((cut.coefficients[5] - 4.0).abs() < 1e-10, "Hydro 2 lag 2");
    }
}
