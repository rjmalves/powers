//! State representations for SDDP subproblems.
//!
//! # Architecture: Single Source of Truth
//!
//! The `State` trait provides a unified interface where `coefficients()` returns
//! the Markov state representation used in cut evaluation. This flat vector IS the
//! state - everything else is extraction logic to populate it from the trajectory.
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
use std::borrow::Cow;
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

    /// Update coefficient values in place (no allocation).
    ///
    /// This method is used with preallocated states to avoid heap allocations
    /// during training. The input slice must have the same length as the
    /// internal coefficient vector.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - New values to copy into internal storage
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `coefficients.len() != self.dimension()`
    fn update_coefficients(&mut self, coefficients: &[f64]);

    /// Reset coefficients to zero while preserving capacity.
    ///
    /// Used for initializing preallocated states to a clean state.
    /// Preserves the allocated memory but fills with 0.0.
    fn reset_to_zero(&mut self);

    /// Get the state dimension (number of coefficients).
    ///
    /// This is the total number of state variables tracked, including
    /// storage and any lagged values.
    fn dimension(&self) -> usize;

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
    /// `Cow<[f64]>` of storage values, indexed by hydro_id. For StorageState,
    /// this borrows from internal state_coefficients (zero allocation). For
    /// StorageAndInflowState, this returns an owned Vec (storage is non-contiguous).
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
    /// - StorageState: O(n) copy to internal buffer, returns borrowed slice (no alloc)
    /// - StorageAndInflowState: O(n + Σp) where p is AR order per hydro (allocates)
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
    ) -> Cow<'_, [f64]>;

    fn add_variables_to_subproblem(
        &self,
        pb: &mut solver::Problem,
    ) -> Vec<Vec<usize>>;

    /// Returns the variable indices for cut constraint coefficients.
    ///
    /// This defines the sparsity pattern for preallocated cut constraints.
    /// Each implementation returns indices matching its coefficient structure.
    ///
    /// # Returns
    ///
    /// Vector of (variable_index, is_alpha) pairs in coefficient order.
    /// The `is_alpha` flag indicates whether this is the alpha variable.
    ///
    /// # Structure by Implementation
    ///
    /// - **StorageState**: `[(alpha, true), (S0, false), (S1, false), ...]`
    /// - **StorageAndInflowState**: `[(alpha, true), (S0, false), (Y0_lag1, false), ..., (S1, false), ...]`
    fn get_cut_variable_indices(
        &self,
        variables: &subproblem::Variables,
    ) -> Vec<usize>;

    /// Returns the number of coefficients in a cut for this state type.
    ///
    /// This is used for preallocating cut constraint slots with the correct
    /// sparsity pattern.
    fn get_cut_coefficient_count(&self) -> usize;

    /// Evaluate cut and return lightweight result with references.
    ///
    /// Unlike `evaluate_cut`, this does not allocate. The returned
    /// `CutEvalResult` holds references to the caller-provided computation buffers.
    ///
    /// # Arguments
    ///
    /// * `risk_measure` - Risk measure for probability adjustment
    /// * `branching_realizations` - Scenario realizations to compute cut from
    /// * `buffers` - Pre-allocated computation buffers (caller-provided, not thread-local)
    ///
    /// # Lifetime
    ///
    /// The returned result references `buffers.coefficients`, so it must not outlive
    /// the buffers.
    fn evaluate_cut_ref<'a>(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        buffers: &'a mut crate::memory::CutComputationBuffers,
    ) -> cut::CutEvalResult<'a>;

    // default implementations
    fn update_dominating_cut(&mut self, cut: &cut::BendersCut, height: f64) {
        self.set_dominating_cut_id(cut.id);
        self.set_dominating_objective(height);
    }

    /// Compute cut and write directly to preallocated pool slots.
    ///
    /// # Zero Allocation
    ///
    /// This method performs **no heap allocation**. Cut and state coefficients
    /// are copied directly from thread-local buffers to preallocated pool slots
    /// using `copy_from_slice`.
    ///
    /// # Arguments
    ///
    /// * `risk_measure` - Risk measure for probability adjustment
    /// * `branching_realizations` - Results from backward solve
    /// * `cut_pool` - Preallocated cut pool to update
    /// * `state_pool` - Preallocated state pool to update
    /// * `iteration` - Training iteration (1-based)
    /// * `forward_pass_idx` - Forward pass index (0-based)
    ///
    /// # Returns
    ///
    /// Slot index where cut and state were stored.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slot = state.compute_cut_into_slot(
    ///     risk_measure,
    ///     &realizations,
    ///     &mut fcf.cut_pool,
    ///     &mut fcf.state_pool,
    ///     iteration,
    ///     forward_pass_idx,
    /// );
    /// ```
    fn compute_cut_into_slot(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        cut_pool: &mut cut::BendersCutPool,
        state_pool: &mut VisitedStatePool,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> usize;

    // clone helper for storing visited states
    fn clone_dyn(&self) -> Box<dyn State>;
}

// trait for cloning boxes of State trait objects
impl Clone for Box<dyn State> {
    fn clone(&self) -> Self {
        self.as_ref().clone_dyn()
    }
}

/// Identifier for State implementation types.
///
/// Used by pools to manage heterogeneous state types and enable
/// pool-based allocation in Epic 5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateTypeId {
    /// Storage-only state (no lagged observations)
    Storage,
    /// Storage + lagged inflow observations
    StorageAndInflow,
}

/// Common state fields shared by all State implementations.
///
/// Contains domination tracking, iteration tracking, and coefficient storage.
/// State implementations embed this struct and delegate common methods.
///
/// # Architecture (Epic 4 - T-040)
///
/// This struct eliminates duplication between `StorageState` and
/// `StorageAndInflowState` by extracting shared fields into a reusable core.
///
/// # Fields
///
/// - `state_coefficients`: The Markov state as a flat coefficient vector
/// - `dominating_objective`: Best cut height for this state
/// - `dominating_cut_id`: ID of the dominating cut
/// - `iteration`: Training iteration when state was visited (1-based)
/// - `forward_pass_idx`: Forward pass index (0-based)
#[derive(Debug, Clone)]
pub struct StateCore {
    /// The Markov state as a flat coefficient vector.
    /// This is the single source of truth returned by `coefficients()`.
    pub state_coefficients: Vec<f64>,
    /// Best cut height observed at this state
    pub dominating_objective: f64,
    /// ID of the cut that achieves dominating_objective
    pub dominating_cut_id: usize,
    /// Training iteration when this state was visited (1-based)
    pub iteration: usize,
    /// Forward pass index that visited this state (0-based)
    pub forward_pass_idx: usize,
}

impl StateCore {
    /// Create a new StateCore with the specified dimension.
    ///
    /// Initializes coefficients to zero and all tracking fields to default values.
    #[inline]
    pub fn new(dimension: usize) -> Self {
        Self {
            state_coefficients: vec![0.0; dimension],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    /// Create a StateCore with the given coefficients.
    ///
    /// Takes ownership of the coefficient vector.
    #[inline]
    pub fn with_coefficients(state_coefficients: Vec<f64>) -> Self {
        Self {
            state_coefficients,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    /// Returns the state coefficients as a slice.
    #[inline]
    pub fn coefficients(&self) -> &[f64] {
        &self.state_coefficients
    }

    /// Returns the dimension (number of coefficients).
    #[inline]
    pub fn dimension(&self) -> usize {
        self.state_coefficients.len()
    }

    /// Update coefficient values in place (no allocation).
    #[inline]
    pub fn update_coefficients(&mut self, coefficients: &[f64]) {
        debug_assert_eq!(
            self.state_coefficients.len(),
            coefficients.len(),
            "coefficient dimension mismatch: expected {}, got {}",
            self.state_coefficients.len(),
            coefficients.len()
        );
        self.state_coefficients.copy_from_slice(coefficients);
    }

    /// Reset all fields to initial values while preserving capacity.
    pub fn reset_to_zero(&mut self) {
        self.state_coefficients.fill(0.0);
        self.dominating_objective = 0.0;
        self.dominating_cut_id = 0;
        self.iteration = 0;
        self.forward_pass_idx = 0;
    }
}

/// Pure state coefficient data without layout metadata.
///
/// # Architecture (Epic 5 - T-075)
///
/// Separates state **data** from state **metadata**. The layout information
/// is stored once at the pool level, not duplicated per state.
///
/// # Memory Layout
///
/// - Stack: 56 bytes (5 fields)
/// - Heap: 8 × dimension bytes (coefficients Vec)
///
/// This eliminates the need for per-state layout metadata storage,
/// as layout is now shared at the pool level.
#[derive(Debug, Clone)]
pub struct StateData {
    /// State coefficients (storage volumes + optional lag values)
    pub coefficients: Vec<f64>,
    /// Best cut height observed at this state
    pub dominating_objective: f64,
    /// ID of the cut that achieves dominating_objective
    pub dominating_cut_id: usize,
    /// Training iteration when this state was visited (1-based)
    pub iteration: usize,
    /// Forward pass index that visited this state (0-based)
    pub forward_pass_idx: usize,
}

impl StateData {
    /// Create a new StateData with the specified dimension.
    ///
    /// Initializes coefficients to zero and all tracking fields to default values.
    #[inline]
    pub fn new(dimension: usize) -> Self {
        Self {
            coefficients: vec![0.0; dimension],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    /// Create with existing coefficients.
    #[inline]
    pub fn with_coefficients(coefficients: Vec<f64>) -> Self {
        Self {
            coefficients,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }

    /// Get coefficients slice.
    #[inline]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Get dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }

    /// Update coefficients in place.
    #[inline]
    pub fn update_coefficients(&mut self, values: &[f64]) {
        debug_assert_eq!(
            self.coefficients.len(),
            values.len(),
            "coefficient dimension mismatch: expected {}, got {}",
            self.coefficients.len(),
            values.len()
        );
        self.coefficients.copy_from_slice(values);
    }

    /// Reset to zero values while preserving capacity.
    pub fn reset_to_zero(&mut self) {
        self.coefficients.fill(0.0);
        self.dominating_objective = 0.0;
        self.dominating_cut_id = 0;
        self.iteration = 0;
        self.forward_pass_idx = 0;
    }

    /// Clone data from another StateData.
    pub fn clone_from_data(&mut self, other: &StateData) {
        debug_assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "coefficient dimension mismatch: expected {}, got {}",
            self.coefficients.len(),
            other.coefficients.len()
        );
        self.coefficients.copy_from_slice(&other.coefficients);
        self.dominating_objective = other.dominating_objective;
        self.dominating_cut_id = other.dominating_cut_id;
        self.iteration = other.iteration;
        self.forward_pass_idx = other.forward_pass_idx;
    }

    /// Get the iteration.
    #[inline]
    pub fn get_iteration(&self) -> usize {
        self.iteration
    }

    /// Set the iteration.
    #[inline]
    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    /// Get the forward pass index.
    #[inline]
    pub fn get_forward_pass_idx(&self) -> usize {
        self.forward_pass_idx
    }

    /// Set the forward pass index.
    #[inline]
    pub fn set_forward_pass_idx(&mut self, idx: usize) {
        self.forward_pass_idx = idx;
    }

    /// Get the dominating cut ID.
    #[inline]
    pub fn get_dominating_cut_id(&self) -> usize {
        self.dominating_cut_id
    }

    /// Set the dominating cut ID.
    #[inline]
    pub fn set_dominating_cut_id(&mut self, id: usize) {
        self.dominating_cut_id = id;
    }

    /// Get the dominating objective value.
    #[inline]
    pub fn get_dominating_objective(&self) -> f64 {
        self.dominating_objective
    }

    /// Set the dominating objective value.
    #[inline]
    pub fn set_dominating_objective(&mut self, obj: f64) {
        self.dominating_objective = obj;
    }

    /// Update dominating cut (convenience method).
    #[inline]
    pub fn update_dominating_cut(
        &mut self,
        cut: &cut::BendersCut,
        height: f64,
    ) {
        self.dominating_cut_id = cut.id;
        self.dominating_objective = height;
    }
}

/// Configuration for creating state pool.
///
/// Replaces `&dyn State` template pattern for pool initialization.
/// This enables fully static dispatch with no Box allocation.
///
/// # Architecture (Epic 5 - T-076)
///
/// Used by `VisitedStatePool::preallocate` to create `StateData`
/// instances without requiring a template `Box<dyn State>`.
#[derive(Debug, Clone)]
pub enum StateConfig {
    /// Storage-only states (no AR lags)
    Storage {
        /// Number of hydro plants
        num_hydros: usize,
    },
    /// Storage + inflow lag states (for AR models)
    StorageAndInflow {
        /// Number of hydro plants
        num_hydros: usize,
        /// Per-hydro state dimensions (1 + AR order for each hydro)
        per_hydro_state_dims: Vec<usize>,
    },
}

impl StateConfig {
    /// Create a `StateData` from this configuration.
    #[inline]
    pub fn create_state_data(&self) -> StateData {
        StateData::new(self.dimension())
    }

    /// Get the state type identifier.
    #[inline]
    pub fn state_type(&self) -> StateTypeId {
        match self {
            Self::Storage { .. } => StateTypeId::Storage,
            Self::StorageAndInflow { .. } => StateTypeId::StorageAndInflow,
        }
    }

    /// Get the total state dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        match self {
            Self::Storage { num_hydros } => *num_hydros,
            Self::StorageAndInflow {
                per_hydro_state_dims,
                ..
            } => per_hydro_state_dims.iter().sum(),
        }
    }

    /// Create StateConfig from a dyn State reference (for migration).
    ///
    /// Detects the state type and extracts configuration.
    pub fn from_dyn(state: &dyn State) -> Self {
        if state.has_lagged_observation_state() {
            // StorageAndInflow - need to reconstruct per-hydro dims
            // This is a best-effort approximation
            let dimension = state.dimension();
            Self::StorageAndInflow {
                num_hydros: dimension, // Approximate
                per_hydro_state_dims: vec![1; dimension],
            }
        } else {
            Self::Storage {
                num_hydros: state.dimension(),
            }
        }
    }
}

pub struct VisitedStatePool {
    /// Pool of state data without layout duplication.
    ///
    /// # Architecture (Epic 5 - T-076)
    ///
    /// Uses `Vec<StateData>` for:
    /// - Elimination of layout duplication across states
    /// - Better cache locality (pure data, no metadata)
    /// - Reduced memory footprint (~160 bytes per state saved)
    pub pool: Vec<StateData>,

    /// Shared layout for all states (None for Storage type).
    ///
    /// Stored once in the pool instead of per-state, eliminating
    /// redundant allocations.
    pub layout: Option<StateLayout>,

    /// State type identifier for all states in this pool.
    pub state_type: StateTypeId,

    /// Number of hydros for this pool.
    pub num_hydros: usize,
}

impl VisitedStatePool {
    /// Create state pool with pre-allocated capacity.
    ///
    /// # Performance Optimization (TICKET-006d)
    ///
    /// Pre-allocates Vec to avoid reallocations during training.
    ///
    /// **State sizes** (approximate):
    /// - `Storage`: ~48 bytes (stack) + 8×num_hydros (heap for coefficients)
    /// - `StorageAndInflow`: ~80 bytes (stack) + 8×state_dim (heap for coefficients)
    ///   where state_dim = num_hydros + Σ(AR_orders)
    ///
    /// **Expected behavior** (200 states):
    /// - Without preallocation: ~8 Vec reallocations
    /// - With preallocation: 0 reallocations
    ///
    /// # Arguments
    ///
    /// * `num_states` - Expected number of states (num_forward_passes × num_iterations)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pool = VisitedStatePool::with_capacity(200);
    /// // pool has capacity for 200 StateData
    /// ```
    pub fn with_capacity(num_states: usize) -> Self {
        Self {
            pool: Vec::with_capacity(num_states),
            layout: None,
            state_type: StateTypeId::Storage,
            num_hydros: 0,
        }
    }

    /// Preallocate all states using a template dyn State (legacy API).
    ///
    /// Uses the template state to detect the state type and create
    /// StateData instances with the same structure.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `template_state` - Template with correct dimension
    ///
    /// # Performance
    ///
    /// Allocates `num_iterations * num_forward_passes` states upfront.
    /// Each state has preallocated coefficient vector.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template: Box<dyn State> = Box::new(StorageState::new(&system));
    /// let pool = VisitedStatePool::preallocate(8, 16, &*template);
    /// assert_eq!(pool.pool.len(), 128);
    /// ```
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &dyn State,
    ) -> Self {
        // Convert template to StateConfig
        let config = StateConfig::from_dyn(template_state);
        Self::preallocate_concrete(num_iterations, num_forward_passes, &config)
    }

    /// Preallocate all states using StateConfig (preferred API).
    ///
    /// Creates StateData instances without any Box allocation.
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `config` - State configuration describing the state type
    ///
    /// # Performance
    ///
    /// - Zero Box allocation
    /// - Zero vtable overhead
    /// - Better cache locality than Box<dyn State>
    /// - Layout stored once (not per state)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = StateConfig::Storage { num_hydros: 10 };
    /// let pool = VisitedStatePool::preallocate_concrete(8, 16, &config);
    /// assert_eq!(pool.pool.len(), 128);
    /// ```
    pub fn preallocate_concrete(
        num_iterations: usize,
        num_forward_passes: usize,
        config: &StateConfig,
    ) -> Self {
        let total_states = num_iterations * num_forward_passes;

        match config {
            StateConfig::Storage { num_hydros } => {
                let pool: Vec<StateData> = (0..total_states)
                    .map(|_| StateData::new(*num_hydros))
                    .collect();

                Self {
                    pool,
                    layout: None,
                    state_type: StateTypeId::Storage,
                    num_hydros: *num_hydros,
                }
            }
            StateConfig::StorageAndInflow {
                num_hydros,
                per_hydro_state_dims,
            } => {
                // Build layout ONCE (not per state)
                let mut offsets = Vec::with_capacity(*num_hydros + 1);
                offsets.push(0);
                let mut cumsum = 0;
                for &dim in per_hydro_state_dims {
                    cumsum += dim;
                    offsets.push(cumsum);
                }

                let layout = StateLayout {
                    per_hydro_dims: per_hydro_state_dims.clone(),
                    offsets,
                    total_dim: cumsum,
                };

                let pool: Vec<StateData> =
                    (0..total_states).map(|_| StateData::new(cumsum)).collect();

                Self {
                    pool,
                    layout: Some(layout),
                    state_type: StateTypeId::StorageAndInflow,
                    num_hydros: *num_hydros,
                }
            }
        }
    }

    /// Check if pool was created with preallocate().
    #[inline]
    pub fn is_preallocated(&self) -> bool {
        !self.pool.is_empty()
            && self.pool[0].get_iteration() == 0
            && self.pool[0].get_forward_pass_idx() == 0
    }

    /// Update state at the given slot index.
    ///
    /// No allocation - modifies preallocated state in place.
    ///
    /// # Arguments
    ///
    /// * `slot` - Slot index computed from (iteration, forward_pass_idx)
    /// * `coefficients` - New coefficient values
    /// * `iteration` - Current iteration number
    /// * `forward_pass_idx` - Forward pass index
    ///
    /// # Returns
    ///
    /// Mutable reference to the updated state.
    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut StateData {
        let state = &mut self.pool[slot];
        state.update_coefficients(coefficients);
        state.set_iteration(iteration);
        state.set_forward_pass_idx(forward_pass_idx);
        state
    }

    /// Get the shared layout (for StorageAndInflow states).
    #[inline]
    pub fn get_layout(&self) -> Option<&StateLayout> {
        self.layout.as_ref()
    }

    /// Check if this pool has a layout (StorageAndInflow type).
    #[inline]
    pub fn has_layout(&self) -> bool {
        self.layout.is_some()
    }

    /// Get state type.
    #[inline]
    pub fn state_type(&self) -> StateTypeId {
        self.state_type
    }

    /// Get number of hydros.
    #[inline]
    pub fn num_hydros(&self) -> usize {
        self.num_hydros
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
/// # Architecture (Epic 4 - T-041)
///
/// Uses `StateCore` composition to share common fields with other state types.
/// The `dimension` field is retained for API compatibility but delegates to core.
#[derive(Debug, Clone)]
pub struct StorageState {
    /// Common state fields (coefficients, tracking)
    core: StateCore,
    /// Number of hydros (equals core.dimension())
    dimension: usize,
}

impl StorageState {
    pub fn new(system: &system::System) -> Self {
        let dimension = system.meta.hydros_count;
        Self {
            core: StateCore::new(dimension),
            dimension,
        }
    }
}

impl State for StorageState {
    fn set_dimension(&mut self, dimension: usize) {
        self.dimension = dimension
    }

    fn get_dominating_objective(&self) -> f64 {
        self.core.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.core.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.core.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.core.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.core.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.core.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.core.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.core.forward_pass_idx = forward_pass_idx;
    }

    fn coefficients(&self) -> &[f64] {
        self.core.coefficients()
    }

    fn update_coefficients(&mut self, coefficients: &[f64]) {
        self.core.update_coefficients(coefficients)
    }

    fn reset_to_zero(&mut self) {
        self.core.reset_to_zero()
    }

    fn dimension(&self) -> usize {
        self.dimension
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
    ) -> Cow<'_, [f64]> {
        // PERFORMANCE: O(1) access - get previous storage from last realization
        let prev_realization = trajectory.last().unwrap();

        // Update internal state coefficients (single source of truth)
        self.core
            .state_coefficients
            .clone_from_slice(&prev_realization.final_storage);

        // Return borrowed reference to internal buffer (zero allocation)
        Cow::Borrowed(&self.core.state_coefficients)
    }

    fn get_cut_variable_indices(
        &self,
        variables: &subproblem::Variables,
    ) -> Vec<usize> {
        // StorageState: alpha + storage variables
        let mut indices = Vec::with_capacity(self.dimension + 1);
        indices.push(variables.alpha);
        for &stored_volume in &variables.stored_volume {
            indices.push(stored_volume);
        }
        indices
    }

    fn get_cut_coefficient_count(&self) -> usize {
        // alpha (1) + storage (dimension)
        1 + self.dimension
    }

    fn evaluate_cut_ref<'a>(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        buffers: &'a mut crate::memory::CutComputationBuffers,
    ) -> cut::CutEvalResult<'a> {
        // Reset buffers for this cut computation (preserves capacity)
        buffers.reset_for_cut(self.dimension, branching_realizations.len());

        // Reuse preallocated costs buffer (zero allocation)
        let costs = &mut buffers.costs;
        costs.extend(
            branching_realizations
                .iter()
                .map(|r| r.total_stage_objective),
        );
        // PERF: Use preallocated probabilities buffer (zero allocation)
        let probabilities = &buffers.probabilities;
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(probabilities, costs);

        // Collect all contributions before accumulating for deterministic order
        let coef_contributions = &mut buffers.contributions_outer;
        let objective_contributions = &mut buffers.objective_contributions;

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];

            // Reuse pre-allocated inner vector (zero allocations)
            let contrib = &mut coef_contributions[index];
            contrib.clear();
            contrib
                .extend(realization.water_value.iter().map(|&val| prob * val));

            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        // Deterministic accumulation using Kahan summation
        let cut_coefficients = &mut buffers.coefficients;
        let num_scenarios = branching_realizations.len();
        for hydro_idx in 0..cut_coefficients.len() {
            cut_coefficients[hydro_idx] = utils::kahan_sum_iter(
                coef_contributions
                    .iter()
                    .take(num_scenarios)
                    .map(|contrib| contrib[hydro_idx]),
            );
        }
        let objective = utils::kahan_sum(objective_contributions);

        let cut_rhs = objective
            - utils::dot_product(cut_coefficients, self.coefficients());

        // NO ALLOCATION: Return reference to buffer
        cut::CutEvalResult::new(
            &buffers.coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    fn compute_cut_into_slot(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        cut_pool: &mut cut::BendersCutPool,
        state_pool: &mut VisitedStatePool,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> usize {
        use crate::memory::with_cut_buffers;

        // Set tracking fields before computing cut
        self.set_iteration(iteration);
        self.set_forward_pass_idx(forward_pass_idx);

        with_cut_buffers(|buffers| {
            let eval_result = self.evaluate_cut_ref(
                risk_measure,
                branching_realizations,
                buffers,
            );

            // Zero allocation: copy directly from buffers to preallocated slots
            // eval_result.coefficients is a reference to buffers.coefficients
            cut_pool.update_cut_and_state_slots(
                eval_result.iteration,
                eval_result.forward_pass_idx,
                eval_result.coefficients,
                eval_result.rhs,
                self.coefficients(),
                state_pool,
            )
        })
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
/// # Architecture (Epic 4 - T-042)
///
/// Uses `StateCore` composition to share common fields with other state types.
/// Note: `dimension` here is `num_hydros`, while `core.dimension()` is `layout.total_dim`.
#[derive(Debug, Clone)]
pub struct StorageAndInflowState {
    /// Common state fields (coefficients, tracking)
    core: StateCore,
    /// Number of hydros (for iteration in evaluate_cut)
    dimension: usize,
    /// Per-hydro layout with offsets and dimensions
    layout: StateLayout,
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
            core: StateCore::new(cumsum),
            dimension,
            layout,
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

            self.core.state_coefficients[offset] = storage[hydro_id];

            let lag_count = self.layout.hydro_lag_count(hydro_id);
            if lag_count > 0 {
                let lag_start = offset + 1;
                let lag_end = lag_start + lag_count;
                self.core.state_coefficients[lag_start..lag_end]
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
        self.core.dominating_objective
    }

    fn set_dominating_objective(&mut self, dominating_objective: f64) {
        self.core.dominating_objective = dominating_objective;
    }

    fn get_dominating_cut_id(&self) -> usize {
        self.core.dominating_cut_id
    }

    fn set_dominating_cut_id(&mut self, dominating_cut_id: usize) {
        self.core.dominating_cut_id = dominating_cut_id;
    }

    fn get_iteration(&self) -> usize {
        self.core.iteration
    }

    fn set_iteration(&mut self, iteration: usize) {
        self.core.iteration = iteration;
    }

    fn get_forward_pass_idx(&self) -> usize {
        self.core.forward_pass_idx
    }

    fn set_forward_pass_idx(&mut self, forward_pass_idx: usize) {
        self.core.forward_pass_idx = forward_pass_idx;
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
            &self.core.state_coefficients[lag_start..lag_end]
        }
    }

    fn coefficients(&self) -> &[f64] {
        self.core.coefficients()
    }

    fn update_coefficients(&mut self, coefficients: &[f64]) {
        self.core.update_coefficients(coefficients)
    }

    fn reset_to_zero(&mut self) {
        self.core.reset_to_zero()
    }

    fn dimension(&self) -> usize {
        self.layout.total_dim
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
    ) -> Cow<'_, [f64]> {
        // PERFORMANCE: O(n + Σp) where n is hydros, p is AR orders

        // Extract from trajectory (source of truth)
        let storage = self.extract_storage_from_trajectory_impl(trajectory);
        let lags = self.extract_lags_from_trajectory(trajectory);

        // Rebuild state coefficients (storage + lagged inflows)
        // This updates the single source of truth for cut evaluation
        self.rebuild_state_coefficients(&storage, &lags);

        // Return owned storage values (non-contiguous in state_coefficients)
        // NOTE: Cannot return borrowed slice because storage is interleaved with lags
        Cow::Owned(storage)
    }

    fn get_cut_variable_indices(
        &self,
        variables: &subproblem::Variables,
    ) -> Vec<usize> {
        // StorageAndInflowState: alpha + interleaved (storage + lags per hydro)
        // Structure: [alpha, S0, Y0_lag1, Y0_lag2, ..., S1, Y1_lag1, ...]
        let total_vars = 1 + self.layout.total_dim;
        let mut indices = Vec::with_capacity(total_vars);

        indices.push(variables.alpha);

        for hydro_id in 0..self.dimension {
            // Storage variable
            indices.push(variables.stored_volume[hydro_id]);

            // Lag variables for this hydro
            let hydro_lag_count = self.layout.hydro_lag_count(hydro_id);
            if hydro_lag_count > 0 {
                if let Some(inflow_lags) = &variables.inflow_lags {
                    let lags = inflow_lags.get_lags(hydro_id);
                    for &lag_var in lags.iter().take(hydro_lag_count) {
                        indices.push(lag_var);
                    }
                }
            }
        }

        indices
    }

    fn get_cut_coefficient_count(&self) -> usize {
        // alpha (1) + total_dim (storage + all lags)
        1 + self.layout.total_dim
    }

    fn evaluate_cut_ref<'a>(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        buffers: &'a mut crate::memory::CutComputationBuffers,
    ) -> cut::CutEvalResult<'a> {
        let num_branchings = branching_realizations.len();
        let total_coefficients = self.layout.total_dim;

        // Reset buffers for this cut computation
        buffers.reset_for_cut(total_coefficients, num_branchings);

        let costs = &mut buffers.costs;
        costs.extend(
            branching_realizations
                .iter()
                .map(|r| r.total_stage_objective),
        );
        // PERF: Use preallocated probabilities buffer (zero allocation)
        let probabilities = &buffers.probabilities;
        let adjusted_probabilities =
            risk_measure.adjust_probabilities(probabilities, costs);

        let coef_contributions = &mut buffers.contributions_outer;
        let objective_contributions = &mut buffers.objective_contributions;

        for (index, realization) in branching_realizations.iter().enumerate() {
            let prob = adjusted_probabilities[index];

            // Reuse pre-allocated inner vector
            let contrib = &mut coef_contributions[index];
            contrib.clear();

            // Build coefficients in SAME ORDER as state_coefficients
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

            objective_contributions
                .push(prob * realization.total_stage_objective);
        }

        // Deterministic Kahan summation
        let cut_coefficients = &mut buffers.coefficients;
        let num_scenarios = branching_realizations.len();
        for coef_idx in 0..total_coefficients {
            cut_coefficients[coef_idx] = utils::kahan_sum_iter(
                coef_contributions
                    .iter()
                    .take(num_scenarios)
                    .map(|contrib| contrib[coef_idx]),
            );
        }
        let objective = utils::kahan_sum(objective_contributions);

        let state_coefficients = self.coefficients();

        let cut_rhs = objective
            - utils::dot_product(cut_coefficients, state_coefficients);

        // NO ALLOCATION: Return reference to buffer
        cut::CutEvalResult::new(
            &buffers.coefficients,
            cut_rhs,
            self.get_iteration(),
            self.get_forward_pass_idx(),
        )
    }

    fn compute_cut_into_slot(
        &mut self,
        risk_measure: &dyn risk_measure::RiskMeasure,
        branching_realizations: &[subproblem::Realization],
        cut_pool: &mut cut::BendersCutPool,
        state_pool: &mut VisitedStatePool,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> usize {
        use crate::memory::with_cut_buffers;

        // Set tracking fields before computing cut
        self.set_iteration(iteration);
        self.set_forward_pass_idx(forward_pass_idx);

        with_cut_buffers(|buffers| {
            let eval_result = self.evaluate_cut_ref(
                risk_measure,
                branching_realizations,
                buffers,
            );

            // Zero allocation: copy directly from buffers to preallocated slots
            // eval_result.coefficients is a reference to buffers.coefficients
            cut_pool.update_cut_and_state_slots(
                eval_result.iteration,
                eval_result.forward_pass_idx,
                eval_result.coefficients,
                eval_result.rhs,
                self.coefficients(),
                state_pool,
            )
        })
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
        assert_eq!(state.core.state_coefficients, vec![0.0]);
        assert_eq!(state.core.dominating_objective, 0.0);
        assert_eq!(state.core.dominating_cut_id, 0);
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

    // Tests for simplified cut generation with explicit constraints

    /// Test that direct lag coefficients are used when lag_duals has correct structure

    /// Test that evaluate_cut_ref produces same results as evaluate_cut

    /// Test compute_cut_into_slot updates preallocated pools correctly (T-051)

    /// Test that compute_cut_into_slot produces bit-for-bit identical results to evaluate_cut

    /// Test cut generation with multiple branching realizations

    /// Test that cut RHS calculation is correct with lag coefficients

    /// Test heterogeneous AR orders with explicit constraints

    /// Ensure inflow coefficients use correct variables

    /// Test explicit AR order extraction functions
    ///
    /// Verifies that the explicit extraction functions (`extract_load_ar_orders`
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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage1 = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();
        let coeffs_first = state.coefficients().to_vec();

        // Second call with same data
        let storage2 = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();
        let coeffs_second = state.coefficients().to_vec();

        // Verify: State coefficients should be identical
        assert_eq!(coeffs_first, coeffs_second);
        assert_eq!(storage1, storage2);
    }

    /// Test that StorageState::evaluate_cut_ref produces same results as evaluate_cut

    /// Helper to create test system with specified number of hydros
    fn create_test_system_with_hydros(num_hydros: usize) -> system::System {
        let hydros: Vec<system::Hydro> = (0..num_hydros)
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
        let buses = vec![system::Bus::new(0, 1000.0)];
        let meta = system::SystemMetadata {
            buses_count: buses.len(),
            lines_count: 0,
            thermals_count: 0,
            hydros_count: num_hydros,
        };
        system::System {
            buses,
            lines: vec![],
            thermals: vec![],
            hydros,
            meta,
        }
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
            initial_storage: vec![],
            inflow_lags: vec![],
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

        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Create trajectory with enough history for AR(1)
        let r1 = create_test_realization_with_inflow(
            vec![55.0, 65.0],
            vec![5.5, 6.5],
        );

        let trajectory = vec![&r1];

        state.extract_storage_from_trajectory(&trajectory);
        state.extract_lags_from_trajectory(&trajectory);

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
            initial_storage: vec![],
            inflow_lags: vec![],
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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

        // Verify: State coefficients updated
        assert_eq!(state.coefficients().len(), 2);
        assert!((state.coefficients()[0] - 42.0).abs() < 1e-10);
        assert!((state.coefficients()[1] - 84.0).abs() < 1e-10);

        // Verify: Returned storage matches coefficients
        assert_eq!(&storage, state.coefficients());
    }

    /// Test extraction works without Model (key design goal)
    #[test]
    fn test_storage_state_extraction_no_model_needed() {
        let system = create_test_system_with_hydros(3);
        let mut state = StorageState::new(&system);

        let r1 = create_test_realization(vec![1.0, 2.0, 3.0], vec![]);
        let trajectory = vec![&r1];

        // Execute: No Model/Constraints/Variables needed!
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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
        let storage = state
            .extract_storage_from_trajectory(&trajectory)
            .into_owned();

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

    // ========================================================================
    // TICKET-007, TICKET-008, TICKET-009: State trait in-place update methods
    // ========================================================================

    #[test]
    fn test_storage_state_update_coefficients() {
        let system = create_test_system_with_hydros(3);
        let mut state = StorageState::new(&system);

        // Initial coefficients are zero
        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);

        // Update coefficients
        state.update_coefficients(&[10.0, 20.0, 30.0]);
        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);

        // Update again
        state.update_coefficients(&[5.0, 15.0, 25.0]);
        assert_eq!(state.coefficients(), &[5.0, 15.0, 25.0]);
    }

    #[test]
    fn test_storage_state_reset_to_zero() {
        let system = create_test_system_with_hydros(3);
        let mut state = StorageState::new(&system);

        // Set some non-zero values
        state.update_coefficients(&[10.0, 20.0, 30.0]);
        state.set_dominating_objective(100.0);
        state.set_dominating_cut_id(42);
        state.set_iteration(5);
        state.set_forward_pass_idx(7);

        // Reset to zero
        state.reset_to_zero();

        // Verify all fields are reset
        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);
        assert_eq!(state.get_dominating_objective(), 0.0);
        assert_eq!(state.get_dominating_cut_id(), 0);
        assert_eq!(state.get_iteration(), 0);
        assert_eq!(state.get_forward_pass_idx(), 0);
    }

    #[test]
    fn test_storage_state_dimension() {
        let system = create_test_system_with_hydros(5);
        let state = StorageState::new(&system);
        assert_eq!(state.dimension(), 5);
    }

    #[test]
    fn test_storage_and_inflow_state_update_coefficients() {
        let system = create_test_system_with_hydros(2);
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.5]),
        ];
        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Initial coefficients are zero (4 total: 2 storage + 2 lags)
        assert_eq!(state.coefficients().len(), 4);
        assert!(state.coefficients().iter().all(|&c| c == 0.0));

        // Update coefficients
        state.update_coefficients(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_storage_and_inflow_state_reset_to_zero() {
        let system = create_test_system_with_hydros(2);
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.5]),
        ];
        let mut state = StorageAndInflowState::new(&system, &temporal_models);

        // Set some values
        state.update_coefficients(&[10.0, 20.0, 30.0, 40.0]);
        state.set_dominating_objective(100.0);
        state.set_dominating_cut_id(42);

        // Reset
        state.reset_to_zero();

        // Verify reset
        assert!(state.coefficients().iter().all(|&c| c == 0.0));
        assert_eq!(state.get_dominating_objective(), 0.0);
        assert_eq!(state.get_dominating_cut_id(), 0);
    }

    #[test]
    fn test_storage_and_inflow_state_dimension() {
        let system = create_test_system_with_hydros(3);
        // AR(0), AR(1), AR(2) -> total_dim = 1 + 2 + 3 = 6
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![]),
            create_par_model_uniform_sigma(1, vec![0.5]),
            create_par_model_uniform_sigma(2, vec![0.5, 0.3]),
        ];
        let state = StorageAndInflowState::new(&system, &temporal_models);
        assert_eq!(state.dimension(), 6);
    }

    #[test]
    fn test_update_coefficients_no_reallocation() {
        let system = create_test_system_with_hydros(100);
        let mut state = StorageState::new(&system);
        let original_capacity = state.core.state_coefficients.capacity();

        // Update multiple times
        for i in 0..10 {
            let coeffs: Vec<f64> = (0..100).map(|x| (x + i) as f64).collect();
            state.update_coefficients(&coeffs);
        }

        // Capacity should not change
        assert_eq!(state.core.state_coefficients.capacity(), original_capacity);
    }

    // ========================================================================
    // TICKET-010: VisitedStatePool::preallocate tests
    // ========================================================================

    #[test]
    fn test_state_pool_preallocate_storage() {
        let system = create_test_system_with_hydros(3);
        let template: Box<dyn State> = Box::new(StorageState::new(&system));

        let pool = VisitedStatePool::preallocate(8, 16, template.as_ref());

        assert_eq!(pool.pool.len(), 128); // 8 * 16
        assert_eq!(pool.pool[0].dimension(), 3);
        assert_eq!(pool.pool[0].coefficients(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_state_pool_preallocate_storage_and_inflow() {
        let system = create_test_system_with_hydros(2);
        let temporal_models = vec![
            create_par_model_uniform_sigma(0, vec![0.5]),
            create_par_model_uniform_sigma(1, vec![0.5, 0.3]),
        ];
        let template: Box<dyn State> =
            Box::new(StorageAndInflowState::new(&system, &temporal_models));

        let pool = VisitedStatePool::preallocate(4, 8, template.as_ref());

        assert_eq!(pool.pool.len(), 32); // 4 * 8
                                         // Dimension: (1+1) + (1+2) = 5
        assert_eq!(pool.pool[0].dimension(), 5);
        assert!(pool.pool[0].coefficients().iter().all(|&c| c == 0.0));
    }

    #[test]
    fn test_state_pool_preallocate_states_start_zeroed() {
        let system = create_test_system_with_hydros(3);
        let template: Box<dyn State> = Box::new(StorageState::new(&system));

        let pool = VisitedStatePool::preallocate(2, 4, template.as_ref());

        for state in &pool.pool {
            assert!(state.coefficients().iter().all(|&c| c == 0.0));
            assert_eq!(state.get_dominating_objective(), 0.0);
            assert_eq!(state.get_dominating_cut_id(), 0);
            assert_eq!(state.get_iteration(), 0);
            assert_eq!(state.get_forward_pass_idx(), 0);
        }
    }

    #[test]
    fn test_state_pool_update_state() {
        let system = create_test_system_with_hydros(3);
        let template: Box<dyn State> = Box::new(StorageState::new(&system));

        let mut pool = VisitedStatePool::preallocate(4, 8, template.as_ref());

        // Update state at slot 5
        let state = pool.update_state(5, &[10.0, 20.0, 30.0], 1, 5);
        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);
        assert_eq!(state.get_iteration(), 1);
        assert_eq!(state.get_forward_pass_idx(), 5);

        // Verify state is in pool
        assert_eq!(pool.pool[5].coefficients(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_state_pool_update_state_no_reallocation() {
        let system = create_test_system_with_hydros(50);
        let template: Box<dyn State> = Box::new(StorageState::new(&system));

        let mut pool = VisitedStatePool::preallocate(4, 8, template.as_ref());

        // Get original capacity
        let original_capacity = pool.pool.capacity();
        let original_len = pool.pool.len();

        // Update all states
        for slot in 0..32 {
            let coeffs: Vec<f64> = (0..50).map(|x| (x + slot) as f64).collect();
            pool.update_state(slot, &coeffs, slot / 8 + 1, slot % 8);
        }

        // Capacity and length should not change
        assert_eq!(pool.pool.capacity(), original_capacity);
        assert_eq!(pool.pool.len(), original_len);
    }

    // ========================================================================
    // TICKET-072: StateConfig and VisitedStatePool tests
    // ========================================================================

    #[test]
    fn test_state_config_storage() {
        let config = StateConfig::Storage { num_hydros: 5 };

        assert_eq!(config.state_type(), StateTypeId::Storage);
        assert_eq!(config.dimension(), 5);

        let state = config.create_state_data();
        assert_eq!(state.dimension(), 5);
    }

    #[test]
    fn test_state_config_storage_and_inflow() {
        let config = StateConfig::StorageAndInflow {
            num_hydros: 3,
            per_hydro_state_dims: vec![2, 3, 1], // AR(1), AR(2), AR(0) -> dims 2,3,1
        };

        assert_eq!(config.state_type(), StateTypeId::StorageAndInflow);
        assert_eq!(config.dimension(), 6); // 2 + 3 + 1

        let state = config.create_state_data();
        assert_eq!(state.dimension(), 6);
    }

    #[test]
    fn test_state_config_from_dyn_storage() {
        let system = create_test_system_with_hydros(4);
        let dyn_state: Box<dyn State> = Box::new(StorageState::new(&system));

        let config = StateConfig::from_dyn(dyn_state.as_ref());

        assert_eq!(config.state_type(), StateTypeId::Storage);
        assert_eq!(config.dimension(), 4);
    }

    #[test]
    fn test_state_pool_preallocate_concrete_storage() {
        let config = StateConfig::Storage { num_hydros: 5 };
        let pool = VisitedStatePool::preallocate_concrete(4, 8, &config);

        assert_eq!(pool.pool.len(), 32); // 4 * 8
        assert_eq!(pool.pool[0].dimension(), 5);
        assert_eq!(pool.state_type(), StateTypeId::Storage);
        assert!(pool.pool[0].coefficients().iter().all(|&c| c == 0.0));
    }

    #[test]
    fn test_state_pool_preallocate_concrete_storage_and_inflow() {
        let config = StateConfig::StorageAndInflow {
            num_hydros: 2,
            per_hydro_state_dims: vec![2, 3],
        };
        let pool = VisitedStatePool::preallocate_concrete(2, 4, &config);

        assert_eq!(pool.pool.len(), 8); // 2 * 4
        assert_eq!(pool.pool[0].dimension(), 5); // 2 + 3
        assert_eq!(pool.state_type(), StateTypeId::StorageAndInflow);
    }

    #[test]
    fn test_state_pool_update_state_returns_state_data() {
        let config = StateConfig::Storage { num_hydros: 3 };
        let mut pool = VisitedStatePool::preallocate_concrete(2, 4, &config);

        let state = pool.update_state(3, &[10.0, 20.0, 30.0], 1, 3);

        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);
        assert_eq!(state.get_iteration(), 1);
        assert_eq!(state.get_forward_pass_idx(), 3);
    }

    #[test]
    fn test_state_pool_is_preallocated_with_concrete() {
        let config = StateConfig::Storage { num_hydros: 3 };
        let pool = VisitedStatePool::preallocate_concrete(2, 4, &config);

        assert!(pool.is_preallocated());
    }

    #[test]
    fn test_state_pool_legacy_preallocate_creates_concrete_states() {
        // Test that the legacy preallocate API still works and creates StateData
        let system = create_test_system_with_hydros(3);
        let template: Box<dyn State> = Box::new(StorageState::new(&system));

        let pool = VisitedStatePool::preallocate(2, 4, template.as_ref());

        assert_eq!(pool.pool.len(), 8);
        assert_eq!(pool.pool[0].dimension(), 3);
        // The pool now contains StateData with shared metadata
        assert_eq!(pool.state_type(), StateTypeId::Storage);
    }

    // ========================================================================
    // TICKET-076: Shared Layout tests
    // ========================================================================

    #[test]
    fn test_state_pool_shared_layout_storage() {
        let config = StateConfig::Storage { num_hydros: 5 };
        let pool = VisitedStatePool::preallocate_concrete(4, 8, &config);

        assert_eq!(pool.pool.len(), 32);
        assert_eq!(pool.state_type(), StateTypeId::Storage);
        assert_eq!(pool.num_hydros(), 5);
        assert!(pool.get_layout().is_none()); // No layout for Storage
        assert!(!pool.has_layout());
    }

    #[test]
    fn test_state_pool_shared_layout_storage_and_inflow() {
        let config = StateConfig::StorageAndInflow {
            num_hydros: 2,
            per_hydro_state_dims: vec![2, 3], // AR(1), AR(2)
        };
        let pool = VisitedStatePool::preallocate_concrete(4, 8, &config);

        assert_eq!(pool.pool.len(), 32);
        assert_eq!(pool.state_type(), StateTypeId::StorageAndInflow);
        assert_eq!(pool.num_hydros(), 2);

        // Layout stored ONCE
        let layout = pool.get_layout().expect("should have layout");
        assert_eq!(layout.total_dim, 5);
        assert_eq!(layout.per_hydro_dims, vec![2, 3]);
        assert!(pool.has_layout());
    }

    #[test]
    fn test_state_pool_layout_not_duplicated() {
        let config = StateConfig::StorageAndInflow {
            num_hydros: 10,
            per_hydro_state_dims: vec![2; 10], // AR(1) for all
        };
        let pool = VisitedStatePool::preallocate_concrete(10, 50, &config);

        // 500 states, but only ONE layout
        assert_eq!(pool.pool.len(), 500);
        assert!(pool.has_layout());

        // Each state is just StateData (no layout embedded)
        assert_eq!(pool.pool[0].dimension(), 20); // 10 * 2
        assert_eq!(pool.pool[499].dimension(), 20);

        // Layout metadata is shared
        let layout = pool.get_layout().unwrap();
        assert_eq!(layout.total_dim, 20);
    }

    // ========================================================================
    // TICKET-075: StateData tests
    // ========================================================================

    #[test]
    fn test_state_data_new() {
        let data = StateData::new(5);
        assert_eq!(data.dimension(), 5);
        assert_eq!(data.coefficients(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(data.get_iteration(), 0);
        assert_eq!(data.get_forward_pass_idx(), 0);
        assert_eq!(data.get_dominating_cut_id(), 0);
        assert_eq!(data.get_dominating_objective(), 0.0);
    }

    #[test]
    fn test_state_data_with_coefficients() {
        let data = StateData::with_coefficients(vec![1.0, 2.0, 3.0]);
        assert_eq!(data.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(data.dimension(), 3);
    }

    #[test]
    fn test_state_data_update_coefficients() {
        let mut data = StateData::new(3);
        data.update_coefficients(&[1.0, 2.0, 3.0]);
        assert_eq!(data.coefficients(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_state_data_tracking() {
        let mut data = StateData::new(3);
        data.set_iteration(5);
        data.set_forward_pass_idx(10);
        data.set_dominating_cut_id(42);
        data.set_dominating_objective(123.456);

        assert_eq!(data.get_iteration(), 5);
        assert_eq!(data.get_forward_pass_idx(), 10);
        assert_eq!(data.get_dominating_cut_id(), 42);
        assert!((data.get_dominating_objective() - 123.456).abs() < 1e-10);
    }

    #[test]
    fn test_state_data_reset() {
        let mut data = StateData::new(3);
        data.update_coefficients(&[1.0, 2.0, 3.0]);
        data.set_iteration(5);
        data.set_forward_pass_idx(10);
        data.set_dominating_cut_id(42);
        data.set_dominating_objective(123.456);

        data.reset_to_zero();

        assert_eq!(data.coefficients(), &[0.0, 0.0, 0.0]);
        assert_eq!(data.get_iteration(), 0);
        assert_eq!(data.get_forward_pass_idx(), 0);
        assert_eq!(data.get_dominating_cut_id(), 0);
        assert_eq!(data.get_dominating_objective(), 0.0);
    }

    #[test]
    fn test_state_data_clone_from() {
        let mut target = StateData::new(3);
        let source = {
            let mut s = StateData::new(3);
            s.update_coefficients(&[1.0, 2.0, 3.0]);
            s.set_iteration(5);
            s.set_forward_pass_idx(10);
            s.set_dominating_cut_id(42);
            s.set_dominating_objective(123.456);
            s
        };

        target.clone_from_data(&source);

        assert_eq!(target.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(target.get_iteration(), 5);
        assert_eq!(target.get_forward_pass_idx(), 10);
        assert_eq!(target.get_dominating_cut_id(), 42);
        assert!((target.get_dominating_objective() - 123.456).abs() < 1e-10);
    }

    #[test]
    fn test_state_data_size_smaller_than_state_core_with_layout() {
        use std::mem::size_of;

        // StateData should be smaller than StateCore + StateLayout combined
        // StateData: Vec (24) + f64 (8) + 3*usize (24) = 56 bytes
        // StateCore + StateLayout would be ~80+ bytes
        let state_data_size = size_of::<StateData>();
        let state_core_size = size_of::<StateCore>();
        let state_layout_size = size_of::<StateLayout>();

        // StateData is comparable to StateCore (both hold similar data)
        assert!(state_data_size <= state_core_size + state_layout_size);
    }

    #[test]
    fn test_state_data_clone() {
        let original = {
            let mut s = StateData::new(3);
            s.update_coefficients(&[1.0, 2.0, 3.0]);
            s.set_iteration(5);
            s
        };

        let cloned = original.clone();

        assert_eq!(cloned.coefficients(), &[1.0, 2.0, 3.0]);
        assert_eq!(cloned.get_iteration(), 5);
    }
}
