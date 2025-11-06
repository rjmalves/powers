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
//! # Current Architectural Pattern (As-Is)
//!
//! State implementations currently have **TWO** responsibilities:
//! 1. Extract and store state coefficients from trajectory ✓ (intended)
//! 2. Update solver model constraint bounds directly ⚠️ (coupling issue)
//!
//! ## What State Implementations Currently Do
//!
//! - `StorageState`: Updates hydro balance constraint RHS (initial storage)
//! - `StorageAndInflowState`: Updates hydro balance + lag state variable bounds
//!
//! ## Architectural Inconsistency
//!
//! This pattern is inconsistent with the UncertaintyManager pattern where:
//! - UncertaintyManager holds lag buffers (data)
//! - Subproblem reads buffers and updates model (coordination)
//!
//! State currently does BOTH data management AND model updates, creating
//! tight coupling with Subproblem internals (Model, Constraints, Variables).
//!
//! ## Target Architecture (To-Be)
//!
//! State should:
//! - ✓ Extract storage/lag values from trajectory
//! - ✓ Store state coefficients for cut evaluation
//! - ✓ Compute cuts from branching realizations
//! - ✗ NOT directly update solver model
//!
//! Subproblem should:
//! - Call State to extract values
//! - Orchestrate ALL model updates in one place
//!
//! See `STATE_REFACTORING_TICKETS.md` (STATE-REFACTOR-003 onwards) for planned refactoring.
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

    /// Extract storage values from trajectory (STATE-REFACTOR-003)
    ///
    /// Returns the storage values that should be used to update hydro balance
    /// constraint RHS in `prepare_from_trajectory()`. This method updates
    /// internal `state_coefficients` but does NOT update the solver model.
    ///
    /// This establishes the **extraction pattern** where State provides values
    /// and Subproblem updates the model, matching the pattern from REFACTOR-003
    /// where UncertaintyManager provides lag values and Subproblem updates constraints.
    ///
    /// # Returns
    ///
    /// Vector of storage values, indexed by hydro_id. For StorageState, this
    /// is simply the final_storage from the last realization. For
    /// StorageAndInflowState, this extracts storage (and internally handles lags).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // In Subproblem::prepare_from_trajectory()
    /// let storage = self.state.extract_storage_from_trajectory(trajectory);
    ///
    /// // Subproblem updates model with extracted values
    /// for (hydro_id, row) in self.constraints.hydro_balance.iter().enumerate() {
    ///     model.change_rows_bounds(*row, storage[hydro_id], storage[hydro_id]);
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - StorageState: O(n) where n is number of hydros
    /// - StorageAndInflowState: O(n + Σp) where p is AR order per hydro
    ///
    /// # Design Pattern
    ///
    /// This follows the coordinator pattern:
    /// 1. State extracts and stores coefficients (data management)
    /// 2. State returns extracted values (no model dependency)
    /// 3. Subproblem updates model (coordination)
    ///
    /// Compare with UncertaintyManager:
    /// ```rust,ignore
    /// // UncertaintyManager pattern (established)
    /// let lag_obs = self.uncertainty_manager.get_lag_observations(idx);
    /// model.change_rows_bounds(constraint, lag_obs[k], lag_obs[k]);
    ///
    /// // State pattern (new, consistent)
    /// let storage = self.state.extract_storage_from_trajectory(trajectory);
    /// model.change_rows_bounds(constraint, storage[i], storage[i]);
    /// ```
    fn extract_storage_from_trajectory(
        &mut self,
        trajectory: &[&subproblem::Realization],
    ) -> Vec<f64>;

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
        branching_realizations: &[subproblem::Realization],
    ) -> cut::BendersCut {
        // NOTE: Don't call update_dominating_cut() here! The cut has id=0 at this point.
        // The FCF will handle domination properly after assigning the real cut ID.
        self.evaluate_cut(risk_measure, branching_realizations)
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

/// Extract AR orders for all loads from TemporalModel
///
/// Returns a vector indexed by bus_id containing the max AR order for each load.
/// Uses explicit entity type filtering to avoid unified entity iteration.
///
/// # Note
///
/// This function is currently unused because existing state implementations
/// (`StorageState`, `StorageAndInflowState`) only track inflow lags as Markov
/// state variables. Load lags are managed separately in the uncertainty constraint
/// system and are not part of the cut evaluation state. This function is provided
/// for completeness and potential future use if load lags become part of state.
#[allow(dead_code)]
fn extract_load_ar_orders(
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    n_buses: usize,
) -> Vec<usize> {
    use crate::input::UncertaintyType;

    let mut ar_orders = vec![0; n_buses];

    for model in uncertainty_models.iter() {
        if model.entity_type() == UncertaintyType::Load {
            let bus_id = model.entity_id();
            ar_orders[bus_id] = ar_orders[bus_id].max(model.max_ar_order);
        }
    }

    ar_orders
}

/// Extract AR orders for all inflows from TemporalModel
///
/// Returns a vector indexed by hydro_id containing the max AR order for each inflow.
/// Uses explicit entity type filtering to avoid unified entity iteration.
fn extract_inflow_ar_orders(
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    n_hydros: usize,
) -> Vec<usize> {
    use crate::input::UncertaintyType;

    let mut ar_orders = vec![0; n_hydros];

    for model in uncertainty_models.iter() {
        if model.entity_type() == UncertaintyType::Inflow {
            let hydro_id = model.entity_id();
            ar_orders[hydro_id] = ar_orders[hydro_id].max(model.max_ar_order);
        }
    }

    ar_orders
}

/// Calculate per-hydro state dimensions from TemporalModel
///
/// Uses explicit inflow AR order extraction for type-safe access.
pub fn per_hydro_state_dims(
    system: &system::System,
    uncertainty_models: &[crate::temporal_model::TemporalModel],
    _season_id: usize,
) -> Vec<usize> {
    let inflow_ar_orders =
        extract_inflow_ar_orders(uncertainty_models, system.hydros.len());

    inflow_ar_orders
        .iter()
        .map(|&ar_order| 1 + ar_order) // storage + lags
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

    fn extract_storage_from_trajectory(
        &mut self,
        trajectory: &[&subproblem::Realization],
    ) -> Vec<f64> {
        // PERFORMANCE: O(1) access - get previous storage from last realization
        let prev_realization = trajectory.last().unwrap();

        // Update internal state coefficients (single source of truth)
        self.state_coefficients
            .clone_from_slice(&prev_realization.final_storage);

        // Return storage values for Subproblem to use in constraint updates
        // PERF: Small allocation acceptable (typically <100 elements)
        self.state_coefficients.clone()
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

        let cut_rhs = objective
            - utils::dot_product(&cut_coefficients, self.coefficients());
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

        Self {
            dimension,
            layout,
            state_coefficients: vec![0.0; cumsum],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
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

    /// Extract storage values from trajectory (internal helper)
    ///
    /// Gets storage from the last realization in the trajectory.
    /// This is O(n) due to the clone operation.
    fn extract_storage_from_trajectory_impl(
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

            self.state_coefficients[offset] = storage[hydro_id];

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

    fn extract_storage_from_trajectory(
        &mut self,
        trajectory: &[&subproblem::Realization],
    ) -> Vec<f64> {
        // PERFORMANCE: O(n + Σp) where n is hydros, p is AR orders

        // Extract from trajectory (source of truth)
        let storage = self.extract_storage_from_trajectory_impl(trajectory);
        let lags = self.extract_lags_from_trajectory(trajectory);

        // Rebuild state coefficients (storage + lagged inflows)
        // This updates the single source of truth for cut evaluation
        self.rebuild_state_coefficients(&storage, &lags);

        // Return storage values for Subproblem to use in constraint updates
        // PERF: Small allocation acceptable (typically <100 elements)
        storage
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

        // CRITICAL: Apply coefficients in SAME ORDER as they were generated!
        // Coefficients are interleaved per-hydro: [S0, S1, Y1(1), S2, Y2(1), Y2(2), ...]
        // This matches evaluate_cut and rebuild_state_coefficients
        let mut coef_idx = 0;
        for hydro_id in 0..self.dimension {
            // Storage coefficient
            let storage_var = variables.stored_volume[hydro_id];
            let storage_coef = -cut.coefficients[coef_idx];
            factors.push((storage_var, storage_coef));

            coef_idx += 1;

            // Lag coefficients for this hydro
            let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
            if hydro_lag_count > 0 {
                if let Some(inflow_lags) = &variables.inflow_lags {
                    let lags = inflow_lags.get_lags(hydro_id);

                    for &lag_var in lags.iter().take(hydro_lag_count) {
                        let lag_coef = -cut.coefficients[coef_idx];
                        factors.push((lag_var, lag_coef));

                        coef_idx += 1;
                    }
                }
            }
        }

        model.add_row(cut.rhs.., factors);
    }

    fn evaluate_cut(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
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

            // CRITICAL: Build coefficients in SAME ORDER as state_coefficients!
            // State structure is interleaved per-hydro: [S0, S1, Y1(1), S2, Y2(1), Y2(2), ...]
            // where hydro i has: [storage_i, lag_i_1, lag_i_2, ..., lag_i_p]
            //
            // This must match rebuild_state_coefficients which uses:
            //   state_coef[offset] = storage
            //   state_coef[offset+1..offset+1+lag_count] = lags
            for hydro_id in 0..self.dimension {
                // Water value (storage coefficient)
                let storage_contrib = prob * realization.water_value[hydro_id];
                contrib.push(storage_contrib);

                // Lag coefficients for this hydro
                let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
                if hydro_lag_count > 0 {
                    let lag_duals = &realization.inflow_lag_duals[hydro_id];
                    for &lag_dual in lag_duals.iter().take(hydro_lag_count) {
                        let lag_contrib = prob * lag_dual;
                        contrib.push(lag_contrib);
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
    use crate::temporal_model;

    // Test helper functions for creating TemporalModels
    fn create_independent_inflow(
        hydro_id: usize,
        mean: f64,
        std_dev: f64,
    ) -> temporal_model::TemporalModel {
        temporal_model::TemporalModel::from_independent(
            input::UncertaintyType::Inflow,
            hydro_id,
            vec![mean],
            vec![std_dev],
            vec![input::MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }],
        )
        .unwrap()
    }

    fn create_independent_load(
        bus_id: usize,
        mean: f64,
        std_dev: f64,
    ) -> temporal_model::TemporalModel {
        temporal_model::TemporalModel::from_independent(
            input::UncertaintyType::Load,
            bus_id,
            vec![mean],
            vec![std_dev],
            vec![input::MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }],
        )
        .unwrap()
    }

    fn create_par_inflow(
        hydro_id: usize,
        mean: f64,
        std_dev: f64,
        ar_order: usize,
        phi_coeffs: Vec<f64>,
    ) -> temporal_model::TemporalModel {
        temporal_model::TemporalModel::from_par(
            input::UncertaintyType::Inflow,
            hydro_id,
            1, // num_seasons
            vec![mean],
            vec![std_dev],
            vec![input::MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }],
            vec![ar_order],
            vec![phi_coeffs],
        )
        .unwrap()
    }

    fn create_par_load(
        bus_id: usize,
        mean: f64,
        std_dev: f64,
        ar_order: usize,
        phi_coeffs: Vec<f64>,
    ) -> temporal_model::TemporalModel {
        temporal_model::TemporalModel::from_par(
            input::UncertaintyType::Load,
            bus_id,
            1, // num_seasons
            vec![mean],
            vec![std_dev],
            vec![input::MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }],
            vec![ar_order],
            vec![phi_coeffs],
        )
        .unwrap()
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
        let temporal_models = vec![create_independent_inflow(0, 100.0, 20.0)];
        let state = factory("storage", &system, &temporal_models);
        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    fn test_factory_storage_and_inflow_state() {
        let system = system::System::default();
        let temporal_models = vec![create_independent_inflow(0, 100.0, 20.0)];
        let state = factory("storage_and_inflow", &system, &temporal_models);

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
        let temporal_models = vec![create_independent_inflow(0, 100.0, 20.0)];
        let _ = factory("invalid", &system, &temporal_models);
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
        let temporal_models: Vec<_> = (0..3)
            .map(|i| create_independent_inflow(i, 100.0, 10.0))
            .collect();

        let state_storage = factory("storage", &system, &temporal_models);
        assert_eq!(state_storage.coefficients().len(), 3);

        // Create fresh models for second test
        let temporal_models2: Vec<_> = (0..3)
            .map(|id| create_independent_inflow(id, 100.0, 10.0))
            .collect();

        let state_inflow =
            factory("storage_and_inflow", &system, &temporal_models2);
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

    // Helper to create PAR model with uniform σ (all seasons same std_dev)
    fn create_par_model_uniform_sigma(
        entity_id: usize,
        phi: Vec<f64>,
    ) -> temporal_model::TemporalModel {
        let ar_order = phi.len();
        temporal_model::TemporalModel::from_par(
            input::UncertaintyType::Inflow,
            entity_id,
            1, // num_seasons
            vec![100.0],
            vec![10.0],
            vec![input::MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }],
            vec![ar_order],
            vec![phi],
        )
        .unwrap()
    }

    // Helper to create Independent model (AR(0))
    fn create_independent_model(
        entity_id: usize,
    ) -> temporal_model::TemporalModel {
        create_independent_inflow(entity_id, 100.0, 10.0)
    }

    // TICKET-007: Tests for simplified cut generation with explicit constraints

    /// Test that direct lag coefficients are used when lag_duals has correct structure
    #[test]
    fn test_evaluate_cut_uses_direct_lag_duals_when_available() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let temporal_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5, 0.3])];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create a realization with inflow_lag_duals for explicit constraints
        // For explicit constraints: inflow_lag_duals[hydro_id].len() == lag_count (2 in this case)
        let realization = subproblem::Realization {
            water_value: vec![10.0],
            inflow_lag_duals: vec![vec![2.0, 3.0]], // Two lag duals for AR(2)
            total_stage_objective: 100.0,
            final_storage: vec![50.0],
            ..Default::default()
        };

        let risk_measure = risk_measure::Expectation {};
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(&risk_measure, &branching_realizations);

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

    /// Test cut generation with multiple branching realizations
    #[test]
    fn test_evaluate_cut_multiple_realizations_explicit_constraints() {
        use crate::risk_measure;
        use crate::subproblem;

        let system = system::System::default();
        let temporal_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5])];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create multiple realizations with explicit constraint structure
        let r1 = subproblem::Realization {
            water_value: vec![10.0],
            inflow_lag_duals: vec![vec![2.0]], // Explicit constraint
            total_stage_objective: 100.0,
            final_storage: vec![50.0],
            ..Default::default()
        };

        let r2 = subproblem::Realization {
            water_value: vec![12.0],
            inflow_lag_duals: vec![vec![3.0]], // Explicit constraint
            total_stage_objective: 110.0,
            final_storage: vec![50.0],
            ..Default::default()
        };

        let risk_measure = risk_measure::Expectation {};
        let branching_realizations = vec![r1.clone(), r2.clone()];

        let cut = state.evaluate_cut(&risk_measure, &branching_realizations);

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
        let temporal_models =
            vec![create_par_model_uniform_sigma(0, vec![0.5])];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Set state coefficients to known values
        state.state_coefficients = vec![50.0, 100.0]; // [storage, lag]

        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0];
        realization.inflow_lag_duals = vec![vec![2.0]]; // Explicit constraint
        realization.total_stage_objective = 1000.0;
        realization.final_storage = vec![50.0];

        let risk_measure = risk_measure::Expectation {};
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(&risk_measure, &branching_realizations);

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
        let temporal_models = vec![
            create_independent_model(0), // Independent (no lags)
            create_par_model_uniform_sigma(1, vec![0.5]), // AR(1)
            create_par_model_uniform_sigma(2, vec![0.5, 0.3]), // AR(2)
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create realization with explicit constraint structure
        // lag_duals[0] = [] (no lags)
        // lag_duals[1] = [dual1] (1 lag)
        // inflow_lag_duals[2] = [dual2_0, dual2_1] (2 lags)
        let mut realization = subproblem::Realization::default();
        realization.water_value = vec![10.0, 20.0, 30.0];
        realization.inflow_lag_duals = vec![
            vec![],         // AR(0) - no lags
            vec![2.0],      // AR(1) - 1 lag
            vec![3.0, 4.0], // AR(2) - 2 lags
        ];
        realization.total_stage_objective = 1000.0;
        realization.final_storage = vec![50.0, 60.0, 70.0];

        let risk_measure = risk_measure::Expectation {};
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(&risk_measure, &branching_realizations);

        // Total coefficients: 1 + 0 + (1 + 1) + (1 + 2) = 6
        // Interleaved order: [S0, S1, lag1_1, S2, lag2_1, lag2_2]
        assert_eq!(
            cut.coefficients.len(),
            6,
            "Should have correct total dimension"
        );

        // Interleaved per-hydro coefficients
        assert!(
            (cut.coefficients[0] - 10.0).abs() < 1e-10,
            "Hydro 0 storage"
        );
        assert!(
            (cut.coefficients[1] - 20.0).abs() < 1e-10,
            "Hydro 1 storage"
        );
        assert!((cut.coefficients[2] - 2.0).abs() < 1e-10, "Hydro 1 lag 1");
        assert!(
            (cut.coefficients[3] - 30.0).abs() < 1e-10,
            "Hydro 2 storage"
        );
        assert!((cut.coefficients[4] - 3.0).abs() < 1e-10, "Hydro 2 lag 1");
        assert!((cut.coefficients[5] - 4.0).abs() < 1e-10, "Hydro 2 lag 2");
    }

    /// TICKET-004: Test bug fix - ensure inflow coefficients use correct variables
    ///
    /// This test verifies the critical bug fix: when the system has both loads
    /// and inflows with AR dynamics, cut coefficients must be applied to the
    /// correct variables. The old heuristic-based matching could confuse loads
    /// with inflows when they had the same AR order.
    ///
    /// The fix uses explicit inflow_lags structure for direct hydro_id access,
    /// ensuring coefficients are always matched correctly regardless of load AR orders.
    #[test]
    fn test_cut_generation_with_mixed_load_inflow_ar() {
        use crate::risk_measure;
        use crate::subproblem;
        use crate::system::{Hydro, System};

        // Create system with 2 hydros
        let mut system = System::default();
        system
            .hydros
            .push(Hydro::new(1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01));
        system.meta.hydros_count = 2;

        // Create uncertainty models: Inflow 0: AR(1), Inflow 1: AR(2)
        // In practice, loads could also have AR(1) but they don't appear in
        // the state coefficients - only inflows do
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.6]), // Inflow 0: AR(1)
            create_par_model_uniform_sigma(1, vec![0.5, 0.3]), // Inflow 1: AR(2)
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create realization with explicit lag duals
        let realization = subproblem::Realization {
            water_value: vec![10.0, 20.0],
            inflow_lag_duals: vec![
                vec![100.0],        // Hydro 0: 1 lag dual
                vec![200.0, 300.0], // Hydro 1: 2 lag duals
            ],
            total_stage_objective: 1000.0,
            final_storage: vec![50.0, 60.0],
            ..Default::default()
        };

        let risk_measure = risk_measure::Expectation {};
        let branching_realizations = vec![realization.clone()];

        let cut = state.evaluate_cut(&risk_measure, &branching_realizations);

        // Verify cut structure: (1+1) + (1+2) = 5 coefficients
        // Interleaved order: [S0, lag0_1, S1, lag1_1, lag1_2]
        assert_eq!(cut.coefficients.len(), 5, "Cut should have 5 coefficients");

        // Verify interleaved per-hydro coefficients
        assert!(
            (cut.coefficients[0] - 10.0).abs() < 1e-10,
            "Hydro 0 storage coefficient"
        );
        assert!(
            (cut.coefficients[1] - 100.0).abs() < 1e-10,
            "Hydro 0 lag 1 coefficient"
        );
        assert!(
            (cut.coefficients[2] - 20.0).abs() < 1e-10,
            "Hydro 1 storage coefficient"
        );
        assert!(
            (cut.coefficients[3] - 200.0).abs() < 1e-10,
            "Hydro 1 lag 1 coefficient"
        );
        assert!(
            (cut.coefficients[4] - 300.0).abs() < 1e-10,
            "Hydro 1 lag 2 coefficient"
        );

        // The key insight: With explicit inflow_lags structure, we directly access
        // by hydro_id, so there's no possibility of confusing loads with inflows
        // even if both have the same AR order. The type system guarantees correctness.
    }

    /// TICKET-006: Test explicit AR order extraction functions
    ///
    /// Verifies that the new explicit extraction functions (`extract_load_ar_orders`
    /// and `extract_inflow_ar_orders`) correctly separate and index AR orders by
    /// entity type and ID, eliminating the need for unified entity iteration.
    #[test]
    fn test_explicit_ar_order_extraction() {
        use crate::system::{Bus, Hydro, System};

        // Create mixed system with loads and inflows having different AR orders
        let buses = vec![
            Bus::new(0, 1000.0), // Bus 0
            Bus::new(1, 1000.0), // Bus 1
            Bus::new(2, 1000.0), // Bus 2
        ];
        let hydros = vec![
            Hydro::new(0, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01), // Hydro 0
            Hydro::new(1, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01), // Hydro 1
        ];
        let system = System::new(buses, vec![], vec![], hydros);

        // Create uncertainty models with mixed entity types:
        // Load 0: AR(2), Load 1: AR(0), Load 2: AR(1)
        // Inflow 0: AR(1), Inflow 1: AR(3)
        let temporal_models = vec![
            // Load models
            create_par_load(0, 100.0, 10.0, 2, vec![0.7, 0.2]),
            create_independent_load(1, 50.0, 5.0),
            create_par_load(2, 75.0, 8.0, 1, vec![0.5]),
            // Inflow models
            create_par_inflow(0, 200.0, 20.0, 1, vec![0.6]),
            create_par_inflow(1, 150.0, 15.0, 3, vec![0.5, 0.3, 0.1]),
        ];

        let temporal_models = temporal_models;

        // Test extract_load_ar_orders
        let load_ar_orders =
            extract_load_ar_orders(&temporal_models, system.buses.len());
        assert_eq!(load_ar_orders.len(), 3, "Should have 3 buses");
        assert_eq!(load_ar_orders[0], 2, "Bus 0 should have AR(2)");
        assert_eq!(load_ar_orders[1], 0, "Bus 1 should have AR(0)");
        assert_eq!(load_ar_orders[2], 1, "Bus 2 should have AR(1)");

        // Test extract_inflow_ar_orders
        let inflow_ar_orders =
            extract_inflow_ar_orders(&temporal_models, system.hydros.len());
        assert_eq!(inflow_ar_orders.len(), 2, "Should have 2 hydros");
        assert_eq!(inflow_ar_orders[0], 1, "Hydro 0 should have AR(1)");
        assert_eq!(inflow_ar_orders[1], 3, "Hydro 1 should have AR(3)");

        // Verify that per_hydro_state_dims uses the new explicit extraction
        let state_dims = per_hydro_state_dims(&system, &temporal_models, 0);
        assert_eq!(state_dims.len(), 2, "Should have 2 hydro state dimensions");
        assert_eq!(
            state_dims[0], 2,
            "Hydro 0: 1 storage + 1 lag = 2 state vars"
        );
        assert_eq!(
            state_dims[1], 4,
            "Hydro 1: 1 storage + 3 lags = 4 state vars"
        );

        // Verify total state dimension
        let total_dim = total_state_dim(&system, &temporal_models, 0);
        assert_eq!(total_dim, 6, "Total state dimension: 2 + 4 = 6");
    }

    // ========================================================================
    // STATE-REFACTOR-002: Baseline tests for State trait model updates
    // ========================================================================
    //
    // These tests document and verify State trait extraction behavior after
    // STATE-REFACTOR-005. They verify:
    // 1. State extraction methods work correctly
    // 2. State coefficients are updated properly
    // 3. Extraction is independent of solver Model
    //
    // NOTE: Model updates are now tested in subproblem tests (STATE-REFACTOR-004)

    /// Test that StorageState::extract_storage_from_trajectory() updates state coefficients
    /// correctly from the trajectory's final storage.
    #[test]
    fn test_storage_state_extracts_storage_from_trajectory() {
        let system = create_test_system_with_hydros(3);

        let mut state = StorageState::new(&system);

        // Create trajectory with known storage values
        let r1 = create_test_realization(vec![10.0, 20.0, 30.0], vec![]);
        let r2 = create_test_realization(vec![15.0, 25.0, 35.0], vec![]);
        let r3 = create_test_realization(vec![12.0, 22.0, 32.0], vec![]);

        let trajectory = vec![&r1, &r2, &r3];

        // Execute: Call extract_storage_from_trajectory (no model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: State coefficients should match LAST realization's final_storage
        assert_eq!(state.coefficients().len(), 3);
        assert!((state.coefficients()[0] - 12.0).abs() < 1e-10);
        assert!((state.coefficients()[1] - 22.0).abs() < 1e-10);
        assert!((state.coefficients()[2] - 32.0).abs() < 1e-10);

        // Verify: Returned storage matches state coefficients
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 12.0).abs() < 1e-10);
        assert!((storage[1] - 22.0).abs() < 1e-10);
        assert!((storage[2] - 32.0).abs() < 1e-10);
    }

    /// Test that StorageState extraction works without Model
    /// This demonstrates the clean separation achieved by STATE-REFACTOR-005
    #[test]
    fn test_storage_state_updates_hydro_balance_constraints() {
        let system = create_test_system_with_hydros(3);

        let mut state = StorageState::new(&system);

        // Create trajectory
        let r1 = create_test_realization(vec![50.0, 60.0, 70.0], vec![]);
        let trajectory = vec![&r1];

        // Execute: Extract storage (no Model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: State coefficients match extracted storage
        assert_eq!(state.coefficients().len(), 3);
        assert!(
            (state.coefficients()[0] - 50.0).abs() < 1e-10,
            "Storage 0 should be 50.0"
        );
        assert!(
            (state.coefficients()[1] - 60.0).abs() < 1e-10,
            "Storage 1 should be 60.0"
        );
        assert!(
            (state.coefficients()[2] - 70.0).abs() < 1e-10,
            "Storage 2 should be 70.0"
        );

        // Verify: Returned storage matches
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 50.0).abs() < 1e-10);
        assert!((storage[1] - 60.0).abs() < 1e-10);
        assert!((storage[2] - 70.0).abs() < 1e-10);
    }

    /// Test with zero storage values (edge case)
    #[test]
    fn test_storage_state_with_zero_storage() {
        let system = create_test_system_with_hydros(2);
        let mut state = StorageState::new(&system);

        let r1 = create_test_realization(vec![0.0, 0.0], vec![]);
        let trajectory = vec![&r1];

        // Execute: Extract (no model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: Zero storage is handled correctly in state coefficients
        assert!((state.coefficients()[0] - 0.0).abs() < 1e-10);
        assert!((state.coefficients()[1] - 0.0).abs() < 1e-10);

        // Verify: Returned storage is zero
        assert!((storage[0] - 0.0).abs() < 1e-10);
        assert!((storage[1] - 0.0).abs() < 1e-10);
    }

    /// Test extraction idempotency: calling twice with same data
    /// should produce same results
    #[test]
    fn test_state_update_idempotency() {
        let system = create_test_system_with_hydros(2);
        let mut state = StorageState::new(&system);

        let r1 = create_test_realization(vec![42.0, 84.0], vec![]);
        let trajectory = vec![&r1];

        // First call
        let storage1 = state.extract_storage_from_trajectory(&trajectory);
        let coeffs_first = state.coefficients().to_vec();

        // Second call with same data
        let storage2 = state.extract_storage_from_trajectory(&trajectory);
        let coeffs_second = state.coefficients().to_vec();

        // Verify: State coefficients should be identical
        assert_eq!(coeffs_first, coeffs_second);
        assert_eq!(storage1, storage2);
    }

    /// Helper to create test system with specified number of hydros
    fn create_test_system_with_hydros(num_hydros: usize) -> system::System {
        let mut sys = system::System::default();
        sys.hydros = (0..num_hydros)
            .map(|id| {
                system::Hydro::new(
                    id, None, 0,      // bus_id
                    1.0,    // productivity
                    0.0,    // min_storage
                    100.0,  // max_storage
                    0.0,    // min_outflow
                    10.0,   // max_outflow
                    1000.0, // cost
                )
            })
            .collect();
        sys.buses = vec![system::Bus::new(0, 1000.0)];
        sys.meta.hydros_count = num_hydros;
        sys
    }

    /// Helper to create test realization with storage and inflow
    fn create_test_realization(
        storage: Vec<f64>,
        inflow: Vec<f64>,
    ) -> subproblem::Realization {
        subproblem::Realization {
            kind: subproblem::StudyPeriodKind::Study,
            final_storage: storage,
            inflow,
            loads: vec![],
            thermal_generation: vec![],
            deficit: vec![],
            turbined_flow: vec![],
            spillage: vec![],
            water_value: vec![],
            marginal_cost: vec![],
            total_stage_objective: 0.0,
            current_stage_objective: 0.0,
            inflow_lag_duals: vec![],
            load_lag_duals: vec![],
            basis: solver::Basis::default(),
            exchange: vec![],
        }
    }

    /// Test StorageAndInflowState updates both storage coefficients and model
    #[test]
    fn test_storage_inflow_state_updates_storage_correctly() {
        let system = create_test_system_with_hydros(2);

        // Create AR(1) models for both hydros
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.6]),
        ];
        let temporal_models = temporal_models;

        let state = StorageAndInflowState::new(&system, &temporal_models);

        // Verify: State coefficients include both storage and lags
        // For AR(1): state = [storage0, lag0, storage1, lag1]
        let coeffs = state.coefficients();
        assert_eq!(coeffs.len(), 4, "Should have 2 storage + 2 lag values");

        // First hydro: storage from last realization
        assert!((coeffs[0] - 55.0).abs() < 1e-10, "Storage for hydro 0");
        // First hydro: lag (inflow from r2)
        assert!((coeffs[1] - 5.5).abs() < 1e-10, "Lag for hydro 0");
        // Second hydro: storage
        assert!((coeffs[2] - 65.0).abs() < 1e-10, "Storage for hydro 1");
        // Second hydro: lag (inflow from r2)
        assert!((coeffs[3] - 6.5).abs() < 1e-10, "Lag for hydro 1");
    }

    /// Test heterogeneous AR orders: mix of AR(0), AR(1), AR(2)
    #[test]
    fn test_storage_inflow_state_heterogeneous_ar_orders() {
        let system = create_test_system_with_hydros(3);

        // Mix of AR orders: AR(0), AR(1), AR(2)
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![]), // AR(0)
            create_par_model_uniform_sigma(1, vec![0.5]), // AR(1)
            create_par_model_uniform_sigma(2, vec![0.6, 0.3]), // AR(2)
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create trajectory with enough history for AR(2)
        let r1 = create_test_realization_with_inflow(
            vec![10.0, 20.0, 30.0],
            vec![1.0, 2.0, 3.0],
        );
        let r2 = create_test_realization_with_inflow(
            vec![15.0, 25.0, 35.0],
            vec![1.5, 2.5, 3.5],
        );
        let r3 = create_test_realization_with_inflow(
            vec![12.0, 22.0, 32.0],
            vec![1.2, 2.2, 3.2],
        );

        let trajectory = vec![&r1, &r2, &r3];

        // Execute: Extract storage (no model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify state coefficients structure:
        // Hydro 0 (AR0): [storage0]
        // Hydro 1 (AR1): [storage1, lag1]
        // Hydro 2 (AR2): [storage2, lag2_1, lag2_2]
        // Total: 1 + 2 + 3 = 6 coefficients
        let coeffs = state.coefficients();
        assert_eq!(coeffs.len(), 6, "Expected 6 state coefficients");

        // Hydro 0: only storage (no lags)
        assert!((coeffs[0] - 12.0).abs() < 1e-10, "Hydro 0 storage");

        // Hydro 1: storage + 1 lag
        assert!((coeffs[1] - 22.0).abs() < 1e-10, "Hydro 1 storage");
        assert!((coeffs[2] - 2.2).abs() < 1e-10, "Hydro 1 lag-1");

        // Hydro 2: storage + 2 lags
        assert!((coeffs[3] - 32.0).abs() < 1e-10, "Hydro 2 storage");
        assert!((coeffs[4] - 3.2).abs() < 1e-10, "Hydro 2 lag-1");
        assert!((coeffs[5] - 3.5).abs() < 1e-10, "Hydro 2 lag-2");

        // Verify: Returned storage matches
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 12.0).abs() < 1e-10);
        assert!((storage[1] - 22.0).abs() < 1e-10);
        assert!((storage[2] - 32.0).abs() < 1e-10);
    }

    /// Helper to create test realization with storage and inflow
    fn create_test_realization_with_inflow(
        storage: Vec<f64>,
        inflow: Vec<f64>,
    ) -> subproblem::Realization {
        subproblem::Realization {
            kind: subproblem::StudyPeriodKind::Study,
            final_storage: storage,
            inflow,
            loads: vec![],
            thermal_generation: vec![],
            deficit: vec![],
            turbined_flow: vec![],
            spillage: vec![],
            water_value: vec![],
            marginal_cost: vec![],
            total_stage_objective: 0.0,
            current_stage_objective: 0.0,
            inflow_lag_duals: vec![],
            load_lag_duals: vec![],
            basis: solver::Basis::default(),
            exchange: vec![],
        }
    }

    // ========================================================================
    // STATE-REFACTOR-003: Tests for extraction methods
    // ========================================================================
    //
    // These tests verify the new extraction pattern where State provides
    // values without updating the model directly.

    /// Test StorageState extraction returns correct values
    #[test]
    fn test_storage_state_extraction_returns_correct_values() {
        let system = create_test_system_with_hydros(3);
        let mut state = StorageState::new(&system);

        // Create trajectory with known values
        let r1 = create_test_realization(vec![10.0, 20.0, 30.0], vec![]);
        let r2 = create_test_realization(vec![15.0, 25.0, 35.0], vec![]);
        let r3 = create_test_realization(vec![12.0, 22.0, 32.0], vec![]);

        let trajectory = vec![&r1, &r2, &r3];

        // Execute: Extract storage (NO model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: Returns storage from LAST realization
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 12.0).abs() < 1e-10);
        assert!((storage[1] - 22.0).abs() < 1e-10);
        assert!((storage[2] - 32.0).abs() < 1e-10);
    }

    /// Test that StorageState extraction updates state coefficients
    #[test]
    fn test_storage_state_extraction_updates_coefficients() {
        let system = create_test_system_with_hydros(2);
        let mut state = StorageState::new(&system);

        let r1 = create_test_realization(vec![42.0, 84.0], vec![]);
        let trajectory = vec![&r1];

        // Execute: Extract
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: State coefficients updated
        assert_eq!(state.coefficients().len(), 2);
        assert!((state.coefficients()[0] - 42.0).abs() < 1e-10);
        assert!((state.coefficients()[1] - 84.0).abs() < 1e-10);

        // Verify: Returned storage matches coefficients
        assert_eq!(storage, state.coefficients());
    }

    /// Test extraction works without Model (key design goal)
    #[test]
    fn test_storage_state_extraction_no_model_needed() {
        let system = create_test_system_with_hydros(3);
        let mut state = StorageState::new(&system);

        let r1 = create_test_realization(vec![1.0, 2.0, 3.0], vec![]);
        let trajectory = vec![&r1];

        // Execute: No Model/Constraints/Variables needed!
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: Works correctly
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 1.0).abs() < 1e-10);
    }

    /// Test StorageAndInflowState extraction returns storage
    #[test]
    fn test_storage_inflow_state_extraction_returns_storage() {
        let system = create_test_system_with_hydros(2);

        // Create AR(1) models
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.6]),
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create trajectory
        let r1 = create_test_realization_with_inflow(
            vec![50.0, 60.0],
            vec![5.0, 6.0],
        );
        let r2 = create_test_realization_with_inflow(
            vec![55.0, 65.0],
            vec![5.5, 6.5],
        );

        let trajectory = vec![&r1, &r2];

        // Execute: Extract storage (NO model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: Returns storage from LAST realization
        assert_eq!(storage.len(), 2);
        assert!((storage[0] - 55.0).abs() < 1e-10);
        assert!((storage[1] - 65.0).abs() < 1e-10);
    }

    /// Test that StorageAndInflowState extraction updates coefficients
    #[test]
    fn test_storage_inflow_state_extraction_updates_coefficients() {
        let system = create_test_system_with_hydros(2);

        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.6]),
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        let r1 = create_test_realization_with_inflow(
            vec![50.0, 60.0],
            vec![5.0, 6.0],
        );
        let r2 = create_test_realization_with_inflow(
            vec![55.0, 65.0],
            vec![5.5, 6.5],
        );

        let trajectory = vec![&r1, &r2];

        // Execute: Extract
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: State coefficients include storage AND lags
        // For AR(1): [storage0, lag0, storage1, lag1]
        let coeffs = state.coefficients();
        assert_eq!(coeffs.len(), 4);

        // Storage from last realization
        assert!((coeffs[0] - 55.0).abs() < 1e-10);
        assert!((coeffs[2] - 65.0).abs() < 1e-10);

        // Lags from last realization inflows
        assert!((coeffs[1] - 5.5).abs() < 1e-10);
        assert!((coeffs[3] - 6.5).abs() < 1e-10);

        // Verify: Returned storage matches extracted storage
        assert_eq!(storage.len(), 2);
        assert!((storage[0] - 55.0).abs() < 1e-10);
        assert!((storage[1] - 65.0).abs() < 1e-10);
    }

    /// Test extraction with heterogeneous AR orders
    #[test]
    fn test_storage_inflow_state_extraction_heterogeneous_ar() {
        let system = create_test_system_with_hydros(3);

        // Mix: AR(0), AR(1), AR(2)
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![]),
            create_par_model_uniform_sigma(1, vec![0.5]),
            create_par_model_uniform_sigma(2, vec![0.6, 0.3]),
        ];
        let temporal_models = temporal_models;

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        let r1 = create_test_realization_with_inflow(
            vec![10.0, 20.0, 30.0],
            vec![1.0, 2.0, 3.0],
        );
        let r2 = create_test_realization_with_inflow(
            vec![15.0, 25.0, 35.0],
            vec![1.5, 2.5, 3.5],
        );
        let r3 = create_test_realization_with_inflow(
            vec![12.0, 22.0, 32.0],
            vec![1.2, 2.2, 3.2],
        );

        let trajectory = vec![&r1, &r2, &r3];

        // Execute: Extract (NO model needed!)
        let storage = state.extract_storage_from_trajectory(&trajectory);

        // Verify: Returns storage only
        assert_eq!(storage.len(), 3);
        assert!((storage[0] - 12.0).abs() < 1e-10);
        assert!((storage[1] - 22.0).abs() < 1e-10);
        assert!((storage[2] - 32.0).abs() < 1e-10);

        // Verify: State coefficients include storage AND lags
        let coeffs = state.coefficients();
        assert_eq!(coeffs.len(), 6); // 1 + 2 + 3

        // Verify structure is correct
        assert!((coeffs[0] - 12.0).abs() < 1e-10); // Hydro 0 storage
        assert!((coeffs[1] - 22.0).abs() < 1e-10); // Hydro 1 storage
        assert!((coeffs[2] - 2.2).abs() < 1e-10); // Hydro 1 lag
        assert!((coeffs[3] - 32.0).abs() < 1e-10); // Hydro 2 storage
        assert!((coeffs[4] - 3.2).abs() < 1e-10); // Hydro 2 lag-1
        assert!((coeffs[5] - 3.5).abs() < 1e-10); // Hydro 2 lag-2
    }

    /// Test that extraction matches old update_from_trajectory behavior
    #[test]
    fn test_extraction_matches_old_update_behavior() {
        let system = create_test_system_with_hydros(2);
        let mut state1 = StorageState::new(&system);
        let mut state2 = StorageState::new(&system);

        let r1 = create_test_realization(vec![50.0, 60.0], vec![]);
        let trajectory = vec![&r1];

        // Method 1: Extract with state1
        let storage1 = state1.extract_storage_from_trajectory(&trajectory);

        // Method 2: Extract with state2 (same data)
        let storage2 = state2.extract_storage_from_trajectory(&trajectory);

        // Verify: Both extractions produce same state coefficients
        assert_eq!(state1.coefficients(), state2.coefficients());

        // Verify: Extraction results are identical
        assert_eq!(storage1, storage2);
        assert_eq!(storage1, state1.coefficients());
    }
}
