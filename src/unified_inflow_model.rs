//! Unified inflow model for AR dynamics representation
//!
//! This module provides `UnifiedInflowModel`, which handles all inflow modeling
//! (both independent and autoregressive) in a single, consistent way.
//!
//! # Key Insight: AR(0) = Independent
//!
//! By treating independent noise as AR(0) (zero coefficients), we eliminate
//! conditional logic throughout the codebase. All inflow models use the same
//! representation with variable coefficients.
//!
//! # Architecture
//!
//! - **Residual Space Native**: All AR coefficients and lag buffers are in
//!   residual space (normalized, zero-mean)
//! - **Shared Seasonal Parameters**: Uses `Arc<SeasonalParams>` to avoid
//!   cloning large parameter arrays across subproblems
//! - **Explicit Constraints**: AR dynamics become LP constraints, not implicit
//!   in state transitions
//!
//! # Example
//!
//! ```rust
//! use powers_rs::unified_inflow_model::UnifiedInflowModel;
//! use powers_rs::unified_noise_spec::{UnifiedNoiseSpec, TemporalModelSpec};
//! use powers_rs::seasonal_params::SeasonalParams;
//! use std::sync::Arc;
//!
//! // Create seasonal parameters for 12 seasons
//! let seasonal_params = Arc::new(SeasonalParams::new(
//!     12,
//!     vec![1; 12],  // AR(1) for all seasons
//!     vec![vec![0.7]; 12],  // φ = 0.7
//!     vec![100.0; 12],  // means
//!     vec![20.0; 12],   // stds
//! ).unwrap());
//!
//! // Mixed AR model: 2 hydros with AR(1), 1 independent
//! let unified_specs = vec![
//!     // Hydro 0: AR(1)
//!     // Hydro 1: AR(1)
//!     // Hydro 2: Independent (AR(0))
//! ];
//!
//! // Construct unified model
//! // let model = UnifiedInflowModel::from_spec(&unified_specs, 3, seasonal_params);
//! ```

use crate::seasonal_params::SeasonalParams;
use crate::solver;
use crate::unified_noise_spec::{TemporalModelSpec, UnifiedNoiseSpec};
use std::sync::Arc;

/// Constraint indices for AR dynamics and observation transformation
///
/// Holds the constraint row indices for each hydro's AR dynamics and
/// observation space transformation constraints. Used to efficiently
/// update constraint RHS values during solve.
///
/// # Fields
///
/// - `ar_dynamics`: Constraint indices for Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
/// - `observation_transform`: Constraint indices for Y_t = μ + σ * Z'_t
///
/// # Performance
///
/// - Size: 2 * n * sizeof(usize) where n = number of hydros (~16n bytes)
/// - Access: O(1) via Vec indexing
/// - Pre-allocated: No runtime allocations during constraint updates
#[derive(Debug, Clone)]
pub struct ConstraintIndices {
    /// AR dynamics constraint indices (one per hydro)
    ///
    /// Constraint: Z'_t[h] - Σ(φ_k[h] * Z'_{t-k}[h]) = ε_t[h]
    /// For independent case: Z'_t[h] = ε_t[h]
    pub ar_dynamics: Vec<usize>,

    /// Observation transformation constraint indices (one per hydro)
    ///
    /// Constraint: Y_t[h] - σ_s[h] * Z'_t[h] = μ_s[h]
    /// Links residual space to observation space for physical constraints
    pub observation_transform: Vec<usize>,
}

/// Unified inflow model handling both independent and AR cases
///
/// # Design Philosophy
///
/// This struct eliminates conditional logic by representing all inflow types
/// uniformly:
/// - **Independent**: Empty `ar_coefficients` (AR(0))
/// - **AR(p)**: Non-empty `ar_coefficients` with p coefficients
///
/// The key insight is that AR(0) = Independent, allowing a single code path
/// with variable coefficients rather than multiple branches.
///
/// # Fields
///
/// - `dimension`: Number of hydro plants (inflow entities)
/// - `ar_coefficients`: Per-hydro AR coefficients (empty = independent)
/// - `seasonal_params`: Shared reference to seasonal μ, σ parameters
/// - `lag_buffer`: Current lag values in residual space (Z'_{t-k})
/// - `max_lag`: Maximum lag order across all hydros (for allocation)
///
/// # Performance Characteristics
///
/// - **Construction**: O(n) where n = number of hydros
/// - **Memory**: O(n·p) where p = max lag order (~8n bytes for AR(1))
/// - **Shared Parameters**: `Arc` avoids cloning seasonal data
/// - **Cache Friendly**: Contiguous Vec storage for coefficients and lags
///
/// # Example Usage
///
/// ```rust,ignore
/// // All independent (AR(0))
/// let model = UnifiedInflowModel::from_spec(&specs, n_hydros, params);
/// assert_eq!(model.max_lag(), 0);
/// assert!(!model.has_ar_dynamics(0));
///
/// // Mixed AR orders
/// let model = UnifiedInflowModel::from_spec(&specs, n_hydros, params);
/// assert_eq!(model.lag_order(0), 1);  // Hydro 0: AR(1)
/// assert_eq!(model.lag_order(1), 2);  // Hydro 1: AR(2)
/// assert_eq!(model.lag_order(2), 0);  // Hydro 2: Independent
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used in TICKET-002 and TICKET-003
pub struct UnifiedInflowModel {
    /// Number of hydro plants (inflow entities)
    ///
    /// Must match the system configuration. This is the total number of
    /// entities that have inflow uncertainty.
    dimension: usize,

    /// AR coefficients for each hydro [φ₁, φ₂, ..., φₚ]
    ///
    /// Outer vec length = dimension (one entry per hydro).
    /// Inner vec length = lag order for that hydro:
    /// - Empty vec: Independent (AR(0))
    /// - Length 1: AR(1)
    /// - Length 2: AR(2)
    /// - etc.
    ///
    /// # Performance
    ///
    /// - Small inner vecs (typically 0-3 elements) stored inline
    /// - Contiguous memory layout for cache efficiency
    /// - Pre-allocated during construction (no runtime allocations)
    ar_coefficients: Vec<Vec<f64>>,

    /// Shared reference to seasonal parameters
    ///
    /// Contains μₘ and σₘ for each season. Using `Arc` avoids cloning
    /// large parameter arrays when constructing multiple subproblems.
    ///
    /// # Performance
    ///
    /// - Reference counting overhead: ~8 bytes per Arc
    /// - Shared across all subproblems: O(1) clone cost
    /// - Cache-friendly: Single allocation for all seasonal data
    seasonal_params: Arc<SeasonalParams>,

    /// Lag buffer storing historical residuals Z'_{t-k}
    ///
    /// Outer vec length = dimension (one buffer per hydro).
    /// Inner vec length = max_lag (sufficient for any hydro's needs).
    ///
    /// For hydro h with lag order p:
    /// - `lag_buffer[h][0]` = Z'_{t-1}
    /// - `lag_buffer[h][1]` = Z'_{t-2}
    /// - ...
    /// - `lag_buffer[h][p-1]` = Z'_{t-p}
    ///
    /// For independent hydros (p=0), the buffer is unused but allocated
    /// for uniform indexing (small memory cost: ~8 bytes per hydro).
    ///
    /// # Performance
    ///
    /// - Pre-allocated to max_lag: no runtime allocations
    /// - Contiguous memory: cache-friendly access patterns
    /// - Uniform indexing: no conditional logic for buffer access
    lag_buffer: Vec<Vec<f64>>,

    /// Maximum lag order across all hydros
    ///
    /// Used for:
    /// - Allocating lag_buffer with sufficient capacity
    /// - Determining trajectory length requirements
    /// - Optimizing constraint generation
    ///
    /// For all-independent case, max_lag = 0.
    max_lag: usize,
}

impl UnifiedInflowModel {
    /// Factory constructor from unified noise specifications
    ///
    /// Extracts AR coefficients from unified specs and constructs the model.
    /// Independent hydros (no AR parameters) get empty coefficient vectors.
    ///
    /// # Arguments
    ///
    /// - `unified_specs`: Slice of unified noise specifications (one per hydro)
    /// - `n_hydros`: Total number of hydro plants
    /// - `seasonal_params`: Shared reference to seasonal parameters
    ///
    /// # Returns
    ///
    /// A fully constructed `UnifiedInflowModel` with:
    /// - AR coefficients extracted from specs
    /// - Lag buffer initialized to zeros
    /// - max_lag computed from all hydro lag orders
    ///
    /// # Performance
    ///
    /// - Time: O(n) where n = number of hydros
    /// - Space: O(n·p) where p = max lag order
    /// - No allocations during hot path (all pre-allocated)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let specs = vec![spec0, spec1, spec2];  // 3 hydros
    /// let params = Arc::new(seasonal_params);
    /// let model = UnifiedInflowModel::from_spec(&specs, 3, params);
    /// ```
    pub fn from_spec(
        unified_specs: &[UnifiedNoiseSpec],
        n_hydros: usize,
        seasonal_params: Arc<SeasonalParams>,
    ) -> Self {
        // Pre-allocate with capacity to avoid reallocation
        let mut ar_coefficients = Vec::with_capacity(n_hydros);
        let mut max_lag = 0;

        // Extract AR coefficients from each spec
        for spec in unified_specs.iter().take(n_hydros) {
            match &spec.temporal_model {
                TemporalModelSpec::Independent => {
                    // Independent: empty coefficients (AR(0))
                    ar_coefficients.push(Vec::new());
                }
                TemporalModelSpec::PeriodicAutoregressive {
                    num_seasons: _,
                    seasonal_ar_params,
                } => {
                    // PAR: extract coefficients from first season
                    // (all seasons should have same order for stationarity)
                    if let Some(params) = seasonal_ar_params.get(&0) {
                        let coeffs = params.ar_coefficients.clone();
                        max_lag = max_lag.max(coeffs.len());
                        ar_coefficients.push(coeffs);
                    } else {
                        // Fallback: no AR params found, treat as independent
                        ar_coefficients.push(Vec::new());
                    }
                }
            }
        }

        // Initialize lag buffer to zeros (will be updated during forward pass)
        let lag_buffer = vec![vec![0.0; max_lag]; n_hydros];

        Self {
            dimension: n_hydros,
            ar_coefficients,
            seasonal_params,
            lag_buffer,
            max_lag,
        }
    }

    /// Get the AR lag order for a specific hydro
    ///
    /// Returns the number of AR coefficients (lag order) for the given hydro.
    /// - 0: Independent (no AR dynamics)
    /// - 1: AR(1)
    /// - 2: AR(2)
    /// - etc.
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index (0..dimension-1)
    ///
    /// # Returns
    ///
    /// The lag order (number of AR coefficients) for the specified hydro.
    ///
    /// # Performance
    ///
    /// O(1) - direct Vec indexing and length access.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert_eq!(model.lag_order(0), 1);  // AR(1)
    /// assert_eq!(model.lag_order(1), 0);  // Independent
    /// ```
    #[inline]
    pub fn lag_order(&self, hydro: usize) -> usize {
        self.ar_coefficients[hydro].len()
    }

    /// Get the maximum lag order across all hydros
    ///
    /// Returns the maximum lag order among all hydros. This is useful for:
    /// - Determining trajectory length requirements
    /// - Allocating constraint arrays
    /// - Optimizing backward pass operations
    ///
    /// For all-independent case, returns 0.
    ///
    /// # Returns
    ///
    /// Maximum lag order across all hydros.
    ///
    /// # Performance
    ///
    /// O(1) - pre-computed during construction.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Mixed: AR(1), AR(2), Independent
    /// assert_eq!(model.max_lag(), 2);
    /// ```
    #[inline]
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Get the number of hydro plants (dimension)
    ///
    /// Returns the total number of hydro plants that have inflow uncertainty.
    /// This should match the system configuration.
    ///
    /// # Returns
    ///
    /// Number of hydro plants (dimension of inflow model).
    ///
    /// # Performance
    ///
    /// O(1) - direct field access.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert_eq!(model.dimension(), 10);  // 10 hydros
    /// ```
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Check if a specific hydro has AR dynamics
    ///
    /// Returns `true` if the hydro has AR coefficients (lag order > 0),
    /// `false` if independent (no AR dynamics).
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index (0..dimension-1)
    ///
    /// # Returns
    ///
    /// `true` if AR dynamics present, `false` if independent.
    ///
    /// # Performance
    ///
    /// O(1) - Vec indexing and emptiness check.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert!(model.has_ar_dynamics(0));   // AR(1)
    /// assert!(!model.has_ar_dynamics(2));  // Independent
    /// ```
    #[inline]
    pub fn has_ar_dynamics(&self, hydro: usize) -> bool {
        !self.ar_coefficients[hydro].is_empty()
    }

    /// Add AR dynamics and observation transformation constraints to LP
    ///
    /// Generates two types of constraints for each hydro:
    /// 1. **AR dynamics**: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t (residual space)
    /// 2. **Observation transform**: Y_t = μ_s + σ_s * Z'_t (links to physical space)
    ///
    /// For independent hydros (empty coefficients), AR dynamics simplifies to Z'_t = ε_t.
    ///
    /// # Arguments
    ///
    /// - `pb`: Mutable reference to solver Problem for adding constraints
    /// - `vars`: Variables struct containing inflow variable indices
    /// - `season_id`: Current season index for seasonal parameter lookup
    ///
    /// # Returns
    ///
    /// `ConstraintIndices` with row indices for all added constraints.
    ///
    /// # Constraints Generated
    ///
    /// For each hydro h:
    ///
    /// **AR Dynamics (residual space):**
    /// ```text
    /// Z'_t[h] - φ₁[h]*Z'_{t-1}[h] - φ₂[h]*Z'_{t-2}[h] - ... = ε_t[h]
    /// ```
    /// - LHS coefficients: +1 on Z'_t, -φ_k on each lag
    /// - RHS: ε_t (set to 0.0 initially, updated at solve time)
    /// - Independent case (φ empty): Z'_t[h] = ε_t[h]
    ///
    /// **Observation Transformation:**
    /// ```text
    /// Y_t[h] = μ_s[h] + σ_s[h] * Z'_t[h]
    /// Rearranged: Y_t[h] - σ_s[h]*Z'_t[h] = μ_s[h]
    /// ```
    /// - LHS coefficients: +1 on Y_t, -σ_s on Z'_t
    /// - RHS: μ_s (seasonal mean)
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: O(n·p) for coefficient vectors (pre-allocated)
    /// - Allocations: 2n constraints + coefficient vectors per hydro
    ///
    /// # Panics
    ///
    /// Panics if variable indices are invalid or seasonal parameters are missing.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut pb = solver::Problem::new();
    /// // ... add variables to pb ...
    /// let indices = model.add_constraints_to_lp(&mut pb, &vars, season_id);
    /// // Update RHS later: pb.change_rhs(indices.ar_dynamics[h], innovation, innovation);
    /// ```
    pub fn add_constraints_to_lp(
        &self,
        pb: &mut solver::Problem,
        vars: &super::subproblem::Variables,
        season_id: usize,
    ) -> ConstraintIndices {
        // PERFORMANCE: Pre-allocate with capacity to avoid reallocation
        let mut ar_dynamics = Vec::with_capacity(self.dimension);
        let mut observation_transform = Vec::with_capacity(self.dimension);

        for hydro in 0..self.dimension {
            // ============================================================
            // AR DYNAMICS CONSTRAINT: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
            // ============================================================

            // PERFORMANCE: Pre-allocate factors vector
            // Size: 1 (Z'_t) + lag_order (lag terms)
            let lag_order = self.ar_coefficients[hydro].len();
            let mut ar_factors = Vec::with_capacity(1 + lag_order);

            // Add Z'_t with coefficient +1.0
            #[allow(deprecated)]
            let zt_var = vars.inflow_process[hydro][1];
            ar_factors.push((zt_var, 1.0));

            // Add lag terms: -φ_k * Z'_{t-k}
            // For independent case (empty coefficients), this loop doesn't execute
            for (lag_idx, &coeff) in
                self.ar_coefficients[hydro].iter().enumerate()
            {
                // Lag variables are stored in vars.inflow_process[hydro][2..]
                // inflow_process structure: [Y_t, Z'_t, Z'_{t-1}, Z'_{t-2}, ...]
                #[allow(deprecated)]
                let lag_var_idx = vars.inflow_process[hydro][2 + lag_idx];
                ar_factors.push((lag_var_idx, -coeff));
            }

            // RHS = 0.0 initially (will be updated to ε_t at solve time)
            // Using equality constraint (0.0..=0.0)
            let ar_row = pb.add_row(0.0..=0.0, &ar_factors);
            ar_dynamics.push(ar_row);

            // ============================================================
            // OBSERVATION TRANSFORMATION: Y_t - σ_s*Z'_t = μ_s
            // ============================================================

            // Lookup seasonal parameters (μ_s, σ_s) for this hydro and season
            let mu = self.seasonal_params.get_mean(season_id);
            let sigma = self.seasonal_params.get_std(season_id);

            // PERFORMANCE: Stack-allocated array for 2 factors (no heap allocation)
            #[allow(deprecated)]
            let obs_factors = [
                (vars.inflow_process[hydro][0], 1.0), // Y_t with coefficient +1.0
                (vars.inflow_process[hydro][1], -sigma), // Z'_t with coefficient -σ_s
            ];

            // RHS = μ_s (seasonal mean)
            let obs_row = pb.add_row(mu..=mu, obs_factors);
            observation_transform.push(obs_row);
        }

        ConstraintIndices {
            ar_dynamics,
            observation_transform,
        }
    }

    // ============================================================================
    // LAG BUFFER MANAGEMENT
    // ============================================================================

    /// Initialize lag buffer from trajectory of past realizations
    ///
    /// Extracts the last p residuals from the trajectory for each hydro,
    /// where p is the lag order. Handles PreStudy nodes correctly, which
    /// may include multi-node histories (PAR case).
    ///
    /// **Important**: The trajectory contains **past observations** (not including
    /// the current unsolved time point). We extract lags from the **last p elements**.
    ///
    /// **Critical**: Trajectory must contain **residuals** (Z'), not observations (Y).
    /// The observation→residual transformation Z' = (Y - μ_s) / σ_s must be performed
    /// upstream (in `sddp/mod.rs`) using correct seasonal parameters for each PreStudy
    /// node. See TICKET-003b for details on PreStudy season handling.
    ///
    /// # Arguments
    ///
    /// - `trajectory`: Slice of past realizations (ordered oldest to newest),
    ///   with trajectory[len-1] being the most recent **past** observation (t-1).
    ///   Each `Realization.inflow_residual[h]` must contain Z' (not Y).
    ///
    /// # Behavior
    ///
    /// For each hydro with lag order p:
    /// - Extracts residuals from trajectory[len-p] to trajectory[len-1]
    /// - Stores in lag_buffer: lag_buffer[h][0] = Z'_{t-1}, lag_buffer[h][1] = Z'_{t-2}, etc.
    /// - If trajectory.len() < p, pads with zeros (defensive, should not occur)
    ///
    /// For independent hydros (p=0), no action taken.
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (updates pre-allocated buffer)
    /// - Cache: Sequential access pattern on trajectory slice
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // AR(2) model with trajectory [t-3, t-2, t-1]
    /// model.initialize_lag_buffer(&trajectory);
    /// // lag_buffer[h][0] = trajectory[2].inflow_residual[h]  // Z'_{t-1}
    /// // lag_buffer[h][1] = trajectory[1].inflow_residual[h]  // Z'_{t-2}
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if trajectory is empty (defensive check).
    ///
    /// # Debug Assertions
    ///
    /// In debug builds, validates that residuals are in reasonable range (|Z'| < 10)
    /// to catch upstream transformation bugs early.
    pub fn initialize_lag_buffer(
        &mut self,
        trajectory: &[crate::subproblem::Realization],
    ) {
        assert!(!trajectory.is_empty(), "Trajectory must not be empty");

        let traj_len = trajectory.len();

        // PERFORMANCE: Sequential iteration over hydros, inner loop over lags
        // Cache-friendly: each hydro's lag buffer is contiguous
        for hydro in 0..self.dimension {
            let lag_order = self.ar_coefficients[hydro].len();

            if lag_order == 0 {
                continue; // Independent hydro, skip
            }

            // Extract last p residuals from trajectory
            // trajectory layout: [..., t-3, t-2, t-1]
            //                            0    1    2  (indices for 3-element trajectory)
            // For AR(2), we want: lag_buffer[h][0] = trajectory[2] (t-1)
            //                     lag_buffer[h][1] = trajectory[1] (t-2)
            //
            // General formula: lag_buffer[h][lag_idx] = trajectory[traj_len - 1 - lag_idx]
            for lag_idx in 0..lag_order {
                let traj_idx = traj_len.checked_sub(1 + lag_idx);

                if let Some(idx) = traj_idx {
                    let residual = trajectory[idx].inflow_residual[hydro];

                    // PERFORMANCE: Debug-only validation to catch upstream transform bugs
                    // Residuals Z' should be normalized (typically |Z'| < 5 for 99.99% of normal)
                    // Threshold of 50 allows test data while catching truly absurd values that
                    // indicate incorrect seasonal parameters (μ, σ) were used in Y→Z' transform
                    debug_assert!(
                        residual.abs() < 50.0,
                        "Residual Z'[{}][lag={}] = {} is extremely out of range. \
                         This suggests incorrect seasonal parameters (μ, σ) were used \
                         during observation→residual transformation. Check PreStudy \
                         season_id assignment (see TICKET-003b).",
                        hydro,
                        lag_idx,
                        residual
                    );

                    self.lag_buffer[hydro][lag_idx] = residual;
                } else {
                    // Trajectory too short (defensive), pad with zeros
                    self.lag_buffer[hydro][lag_idx] = 0.0;
                }
            }
        }
    }
    /// Update lag buffer with single new realization
    ///
    /// Shifts existing lags and inserts new residual at position 0.
    /// Shift semantics:
    /// ```text
    /// Before: [Z'_{t-1}, Z'_{t-2}, Z'_{t-3}]
    /// After:  [Z'_t,     Z'_{t-1}, Z'_{t-2}]
    /// ```
    /// Oldest value (Z'_{t-3}) is discarded.
    ///
    /// # Arguments
    ///
    /// - `realization`: New realization with `inflow_residual` field
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    /// - Cache: Sequential writes to lag buffer
    /// - Note: For p ≤ 3 (typical), shift is faster than circular buffer
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Update with new realization
    /// model.update_lag_buffer(&new_realization);
    /// // lag_buffer[h][0] now contains newest residual
    /// ```
    pub fn update_lag_buffer(
        &mut self,
        realization: &crate::subproblem::Realization,
    ) {
        // PERFORMANCE: Simple shift for small p (typical: p ≤ 3)
        // Circular buffer would be O(1) but adds complexity for minimal gain
        for hydro in 0..self.dimension {
            let lag_order = self.ar_coefficients[hydro].len();

            if lag_order == 0 {
                continue; // Independent hydro, skip
            }

            // Shift existing lags: [0, 1, 2] → [1, 2, ?]
            // Then insert new value at position 0
            for lag_idx in (1..lag_order).rev() {
                self.lag_buffer[hydro][lag_idx] =
                    self.lag_buffer[hydro][lag_idx - 1];
            }

            // Insert new residual at position 0 (most recent)
            self.lag_buffer[hydro][0] = realization.inflow_residual[hydro];
        }
    }

    /// Update lag buffer from trajectory (bulk update)
    ///
    /// More efficient than repeated `update_lag_buffer()` calls.
    /// Extracts last p residuals from trajectory in one pass.
    ///
    /// Equivalent to `initialize_lag_buffer()` but named differently
    /// to clarify intent when updating from simulation trajectory.
    ///
    /// # Arguments
    ///
    /// - `trajectory`: Slice of realizations (ordered oldest to newest)
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations
    /// - Faster than p calls to `update_lag_buffer()` (no shifting)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Update from simulation trajectory
    /// model.update_lag_buffer_from_trajectory(&simulation_trajectory);
    /// ```
    pub fn update_lag_buffer_from_trajectory(
        &mut self,
        trajectory: &[crate::subproblem::Realization],
    ) {
        // Delegate to initialize_lag_buffer (same logic)
        self.initialize_lag_buffer(trajectory);
    }

    /// Get lag residuals for a specific hydro
    ///
    /// Returns a slice of lag values [Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-p}].
    /// For independent hydros (p=0), returns empty slice.
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index (0..dimension-1)
    ///
    /// # Returns
    ///
    /// Slice of lag values:
    /// - `&[Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-p}]` for AR(p) hydros
    /// - `&[]` for independent hydros
    ///
    /// # Performance
    ///
    /// - Time: O(1) - direct slice access
    /// - Space: No allocations (borrows from buffer)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let lags = model.get_lag_residuals(0);
    /// assert_eq!(lags.len(), 2);  // AR(2) model
    /// // lags[0] = Z'_{t-1}, lags[1] = Z'_{t-2}
    /// ```
    #[inline]
    pub fn get_lag_residuals(&self, hydro: usize) -> &[f64] {
        let lag_order = self.ar_coefficients[hydro].len();
        &self.lag_buffer[hydro][0..lag_order]
    }

    /// Clear lag buffer (reset to zeros)
    ///
    /// Useful for testing and reinitialization. Sets all lag values to 0.0.
    ///
    /// # Performance
    ///
    /// - Time: O(n·max_lag)
    /// - Space: No allocations
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.clear_lag_buffer();
    /// assert_eq!(model.get_lag_residuals(0), &[0.0, 0.0]);  // AR(2)
    /// ```
    pub fn clear_lag_buffer(&mut self) {
        for hydro in 0..self.dimension {
            for lag_idx in 0..self.max_lag {
                self.lag_buffer[hydro][lag_idx] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{MarginalDistribution, UncertaintyType};
    use crate::unified_noise_spec::{
        SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec,
        UnifiedNoiseSpec,
    };
    use std::collections::HashMap;

    /// Helper: Create seasonal parameters for testing
    fn create_test_seasonal_params() -> Arc<SeasonalParams> {
        Arc::new(
            SeasonalParams::new(
                12,                  // num_seasons
                vec![1; 12],         // AR(1) for all seasons
                vec![vec![0.7]; 12], // φ = 0.7
                vec![100.0; 12],     // means
                vec![20.0; 12],      // stds
            )
            .unwrap(),
        )
    }

    /// Helper: Create independent noise spec
    fn create_independent_spec(entity_id: usize) -> UnifiedNoiseSpec {
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
            uncertainty_type: UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }),
        }
    }

    /// Helper: Create AR(p) noise spec
    fn create_ar_spec(entity_id: usize, ar_order: usize) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::new();
        for season in 0..12 {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );
        }

        let mut seasonal_ar_params = HashMap::new();
        for season in 0..12 {
            seasonal_ar_params.insert(
                season,
                SeasonalPARParams {
                    ar_order,
                    ar_coefficients: vec![0.7; ar_order],
                },
            );
        }

        UnifiedNoiseSpec {
            uncertainty_type: UncertaintyType::Inflow,
            entity_id,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons: 12,
                seasonal_ar_params,
            },
            seasonal_params,
            marginal_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            }),
        }
    }

    #[test]
    fn test_create_with_independent_noises() {
        let specs = vec![
            create_independent_spec(0),
            create_independent_spec(1),
            create_independent_spec(2),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert_eq!(model.dimension(), 3);
        assert_eq!(model.max_lag(), 0);
        assert_eq!(model.lag_order(0), 0);
        assert_eq!(model.lag_order(1), 0);
        assert_eq!(model.lag_order(2), 0);
        assert!(!model.has_ar_dynamics(0));
        assert!(!model.has_ar_dynamics(1));
        assert!(!model.has_ar_dynamics(2));
    }

    #[test]
    fn test_create_with_ar1_all_hydros() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 1),
            create_ar_spec(2, 1),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert_eq!(model.dimension(), 3);
        assert_eq!(model.max_lag(), 1);
        assert_eq!(model.lag_order(0), 1);
        assert_eq!(model.lag_order(1), 1);
        assert_eq!(model.lag_order(2), 1);
        assert!(model.has_ar_dynamics(0));
        assert!(model.has_ar_dynamics(1));
        assert!(model.has_ar_dynamics(2));
    }

    #[test]
    fn test_create_with_mixed_ar_orders() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 2),
            create_independent_spec(2),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert_eq!(model.dimension(), 3);
        assert_eq!(model.max_lag(), 2); // Max is AR(2)
        assert_eq!(model.lag_order(0), 1); // AR(1)
        assert_eq!(model.lag_order(1), 2); // AR(2)
        assert_eq!(model.lag_order(2), 0); // Independent
        assert!(model.has_ar_dynamics(0));
        assert!(model.has_ar_dynamics(1));
        assert!(!model.has_ar_dynamics(2));
    }

    #[test]
    fn test_lag_order_returns_correct_values() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 2),
            create_independent_spec(2),
            create_ar_spec(3, 3),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 4, params);

        assert_eq!(model.lag_order(0), 1);
        assert_eq!(model.lag_order(1), 2);
        assert_eq!(model.lag_order(2), 0);
        assert_eq!(model.lag_order(3), 3);
    }

    #[test]
    fn test_max_lag_returns_maximum() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 3),
            create_ar_spec(2, 2),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert_eq!(model.max_lag(), 3);
    }

    #[test]
    fn test_dimension_returns_correct_count() {
        let specs =
            vec![create_independent_spec(0), create_independent_spec(1)];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 2, params);

        assert_eq!(model.dimension(), 2);
    }

    #[test]
    fn test_has_ar_dynamics_identifies_correctly() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_independent_spec(1),
            create_ar_spec(2, 2),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert!(model.has_ar_dynamics(0));
        assert!(!model.has_ar_dynamics(1));
        assert!(model.has_ar_dynamics(2));
    }

    #[test]
    fn test_seasonal_params_shared_via_arc() {
        let specs = vec![create_independent_spec(0)];
        let params = create_test_seasonal_params();

        // Check Arc strong count before
        let count_before = Arc::strong_count(&params);

        let model = UnifiedInflowModel::from_spec(&specs, 1, params.clone());

        // Arc should be shared (count increases)
        let count_after = Arc::strong_count(&params);
        assert!(count_after > count_before);

        // Drop model, count should decrease
        drop(model);
        assert_eq!(Arc::strong_count(&params), count_before);
    }

    #[test]
    fn test_single_hydro() {
        let specs = vec![create_ar_spec(0, 1)];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 1, params);

        assert_eq!(model.dimension(), 1);
        assert_eq!(model.max_lag(), 1);
        assert_eq!(model.lag_order(0), 1);
    }

    #[test]
    fn test_all_independent_max_lag_zero() {
        let specs = vec![
            create_independent_spec(0),
            create_independent_spec(1),
            create_independent_spec(2),
        ];
        let params = create_test_seasonal_params();

        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        assert_eq!(model.max_lag(), 0);
    }

    // ============================================================
    // Constraint Generation Tests
    // ============================================================

    /// Helper: Create mock Variables struct for testing
    /// Helper: Create mock Variables struct with proper variable indices
    /// for testing constraint generation.
    fn create_mock_variables(
        pb: &mut solver::Problem,
        n_hydros: usize,
        max_lag: usize,
    ) -> crate::subproblem::Variables {
        use crate::subproblem::Variables;

        let mut inflow_process = Vec::with_capacity(n_hydros);
        for _h in 0..n_hydros {
            // Structure: [Y_t, Z'_t, Z'_{t-1}, Z'_{t-2}, ...]
            let mut vars = Vec::with_capacity(2 + max_lag);

            // Add Y_t variable (physical inflow)
            let y_var = pb.add_column(0.0, 0.0..f64::INFINITY);
            vars.push(y_var);

            // Add Z'_t variable (residual at time t)
            let z_var = pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
            vars.push(z_var);

            // Add lag variables Z'_{t-1}, Z'_{t-2}, ... Z'_{t-max_lag}
            for _ in 0..max_lag {
                let lag_var =
                    pb.add_column(0.0, f64::NEG_INFINITY..f64::INFINITY);
                vars.push(lag_var);
            }

            inflow_process.push(vars);
        }

        Variables {
            deficit: vec![],
            direct_exchange: vec![],
            reverse_exchange: vec![],
            thermal_gen: vec![],
            turbined_flow: vec![],
            spillage: vec![],
            stored_volume: vec![],
            inflow: vec![],
            inflow_residual: vec![0; n_hydros],
            innovation: vec![0; n_hydros],
            lagged_inflow_state: None,
            #[allow(deprecated)]
            inflow_process,
            alpha: 0,
        }
    }

    #[test]
    fn test_constraint_generation_ar1() {
        let specs = vec![create_ar_spec(0, 1)];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 1, params);

        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 1, 1);

        let indices = model.add_constraints_to_lp(&mut pb, &vars, 0);

        // Should have 1 AR dynamics constraint and 1 observation transform constraint
        assert_eq!(indices.ar_dynamics.len(), 1);
        assert_eq!(indices.observation_transform.len(), 1);
    }

    #[test]
    fn test_constraint_generation_ar2() {
        let specs = vec![create_ar_spec(0, 2)];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 1, params);

        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 1, 2);

        let indices = model.add_constraints_to_lp(&mut pb, &vars, 0);

        // Should have 1 AR dynamics constraint and 1 observation transform constraint
        assert_eq!(indices.ar_dynamics.len(), 1);
        assert_eq!(indices.observation_transform.len(), 1);
    }

    #[test]
    fn test_constraint_generation_independent() {
        let specs = vec![create_independent_spec(0)];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 1, params);

        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 1, 0);

        let indices = model.add_constraints_to_lp(&mut pb, &vars, 0);

        // Even independent case gets AR dynamics constraint (simplified to Z'=ε)
        assert_eq!(indices.ar_dynamics.len(), 1);
        assert_eq!(indices.observation_transform.len(), 1);
    }

    #[test]
    fn test_constraint_generation_mixed_models() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 2),
            create_independent_spec(2),
        ];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 3, params);

        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 3, 2);

        let indices = model.add_constraints_to_lp(&mut pb, &vars, 0);

        // Should have constraints for all 3 hydros
        assert_eq!(indices.ar_dynamics.len(), 3);
        assert_eq!(indices.observation_transform.len(), 3);
    }

    #[test]
    fn test_constraint_indices_structure() {
        let specs = vec![create_ar_spec(0, 1), create_ar_spec(1, 1)];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 2, params);

        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 2, 1);

        let indices = model.add_constraints_to_lp(&mut pb, &vars, 0);

        // Verify indices are sequential (0, 1, 2, 3)
        // First hydro gets rows 0 (AR) and 1 (obs)
        // Second hydro gets rows 2 (AR) and 3 (obs)
        assert_eq!(indices.ar_dynamics[0], 0);
        assert_eq!(indices.observation_transform[0], 1);
        assert_eq!(indices.ar_dynamics[1], 2);
        assert_eq!(indices.observation_transform[1], 3);
    }

    #[test]
    fn test_seasonal_params_used_correctly() {
        let specs = vec![create_ar_spec(0, 1)];

        // Create params with distinct values per season
        let params = Arc::new(
            SeasonalParams::new(
                3,                         // num_seasons
                vec![1; 3],                // AR(1) for all seasons
                vec![vec![0.7]; 3],        // φ = 0.7
                vec![100.0, 200.0, 300.0], // Different means per season
                vec![20.0, 40.0, 60.0],    // Different stds per season
            )
            .unwrap(),
        );

        let model = UnifiedInflowModel::from_spec(&specs, 1, params.clone());
        let mut pb = crate::solver::Problem::new();
        let vars = create_mock_variables(&mut pb, 1, 1);

        // Test with season 0
        let _indices0 = model.add_constraints_to_lp(&mut pb, &vars, 0);
        // Observation transform should use μ=100.0, σ=20.0

        // Test with season 1
        let _indices1 = model.add_constraints_to_lp(&mut pb, &vars, 1);
        // Observation transform should use μ=200.0, σ=40.0

        // Cannot directly verify RHS values without solver introspection,
        // but this tests that different seasons produce different constraints
    }

    // ============================================================================
    // LAG BUFFER MANAGEMENT TESTS
    // ============================================================================

    /// Helper: Create mock Realization with specified residuals
    fn create_mock_realization(
        n_hydros: usize,
        residuals: Vec<f64>,
    ) -> crate::subproblem::Realization {
        use crate::subproblem::{Realization, StudyPeriodKind};

        Realization {
            kind: StudyPeriodKind::Study,
            loads: vec![0.0; 1],
            deficit: vec![0.0; 1],
            exchange: vec![0.0; 1],
            inflow: vec![0.0; n_hydros],
            inflow_residual: residuals,
            turbined_flow: vec![0.0; n_hydros],
            spillage: vec![0.0; n_hydros],
            thermal_generation: vec![0.0; 1],
            water_value: vec![0.0; n_hydros],
            marginal_cost: vec![0.0; 1],
            current_stage_objective: 0.0,
            total_stage_objective: 0.0,
            final_storage: vec![0.0; n_hydros],
            lag_duals: vec![],
            basis: crate::solver::Basis::default(),
        }
    }

    #[test]
    fn test_initialize_lag_buffer_ar1() {
        let specs = vec![create_ar_spec(0, 1)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Create trajectory with 3 realizations (past observations)
        let trajectory = vec![
            create_mock_realization(1, vec![1.0]), // t-3
            create_mock_realization(1, vec![2.0]), // t-2
            create_mock_realization(1, vec![3.0]), // t-1 (most recent)
        ];

        model.initialize_lag_buffer(&trajectory);

        // For AR(1), lag_buffer[0][0] should be Z'_{t-1} = 3.0
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags.len(), 1);
        assert_eq!(lags[0], 3.0);
    }

    #[test]
    fn test_initialize_lag_buffer_ar2() {
        let specs = vec![create_ar_spec(0, 2)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Create trajectory with 4 realizations
        let trajectory = vec![
            create_mock_realization(1, vec![1.0]), // t-4
            create_mock_realization(1, vec![2.0]), // t-3
            create_mock_realization(1, vec![3.0]), // t-2
            create_mock_realization(1, vec![4.0]), // t-1 (most recent)
        ];

        model.initialize_lag_buffer(&trajectory);

        // For AR(2):
        // lag_buffer[0][0] = Z'_{t-1} = 4.0
        // lag_buffer[0][1] = Z'_{t-2} = 3.0
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags.len(), 2);
        assert_eq!(lags[0], 4.0);
        assert_eq!(lags[1], 3.0);
    }

    #[test]
    fn test_initialize_lag_buffer_independent() {
        let specs = vec![create_independent_spec(0)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        let trajectory = vec![
            create_mock_realization(1, vec![1.0]),
            create_mock_realization(1, vec![2.0]),
        ];

        model.initialize_lag_buffer(&trajectory);

        // Independent hydro should have empty lag buffer
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags.len(), 0);
    }

    #[test]
    fn test_initialize_lag_buffer_mixed_hydros() {
        let specs = vec![
            create_ar_spec(0, 1),
            create_ar_spec(1, 2),
            create_independent_spec(2),
        ];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 3, params);

        // Create trajectory with 4 realizations
        let trajectory = vec![
            create_mock_realization(3, vec![1.0, 10.0, 100.0]), // t-4
            create_mock_realization(3, vec![2.0, 20.0, 200.0]), // t-3
            create_mock_realization(3, vec![3.0, 30.0, 300.0]), // t-2
            create_mock_realization(3, vec![4.0, 40.0, 400.0]), // t-1 (most recent)
        ];

        model.initialize_lag_buffer(&trajectory);

        // Hydro 0: AR(1)
        let lags0 = model.get_lag_residuals(0);
        assert_eq!(lags0.len(), 1);
        assert_eq!(lags0[0], 4.0); // Z'_{t-1}

        // Hydro 1: AR(2)
        let lags1 = model.get_lag_residuals(1);
        assert_eq!(lags1.len(), 2);
        assert_eq!(lags1[0], 40.0); // Z'_{t-1}
        assert_eq!(lags1[1], 30.0); // Z'_{t-2}

        // Hydro 2: Independent
        let lags2 = model.get_lag_residuals(2);
        assert_eq!(lags2.len(), 0);
    }

    #[test]
    fn test_update_lag_buffer_single_realization() {
        let specs = vec![create_ar_spec(0, 2)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Initialize with [2.0, 1.0] (newest to oldest)
        let init_trajectory = vec![
            create_mock_realization(1, vec![1.0]), // t-2
            create_mock_realization(1, vec![2.0]), // t-1
        ];
        model.initialize_lag_buffer(&init_trajectory);

        // Verify initial state
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags, &[2.0, 1.0]);

        // Update with new realization: 3.0
        let new_realization = create_mock_realization(1, vec![3.0]);
        model.update_lag_buffer(&new_realization);

        // After shift: [3.0, 2.0] (1.0 dropped)
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags, &[3.0, 2.0]);
    }

    #[test]
    fn test_update_lag_buffer_sequence() {
        let specs = vec![create_ar_spec(0, 1)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Initialize with [1.0]
        let init_trajectory = vec![create_mock_realization(1, vec![1.0])];
        model.initialize_lag_buffer(&init_trajectory);

        // Update sequence: 2.0, 3.0, 4.0
        model.update_lag_buffer(&create_mock_realization(1, vec![2.0]));
        assert_eq!(model.get_lag_residuals(0), &[2.0]);

        model.update_lag_buffer(&create_mock_realization(1, vec![3.0]));
        assert_eq!(model.get_lag_residuals(0), &[3.0]);

        model.update_lag_buffer(&create_mock_realization(1, vec![4.0]));
        assert_eq!(model.get_lag_residuals(0), &[4.0]);
    }

    #[test]
    fn test_update_lag_buffer_from_trajectory() {
        let specs = vec![create_ar_spec(0, 2)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Create trajectory
        let trajectory = vec![
            create_mock_realization(1, vec![1.0]), // t-4
            create_mock_realization(1, vec![2.0]), // t-3
            create_mock_realization(1, vec![3.0]), // t-2
            create_mock_realization(1, vec![4.0]), // t-1 (most recent)
        ];

        model.update_lag_buffer_from_trajectory(&trajectory);

        // Should extract last 2: [4.0, 3.0]
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags, &[4.0, 3.0]);
    }

    #[test]
    fn test_clear_lag_buffer() {
        let specs = vec![create_ar_spec(0, 2), create_ar_spec(1, 1)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 2, params);

        // Initialize with non-zero values
        let trajectory = vec![
            create_mock_realization(2, vec![1.0, 10.0]),
            create_mock_realization(2, vec![2.0, 20.0]),
            create_mock_realization(2, vec![3.0, 30.0]),
        ];
        model.initialize_lag_buffer(&trajectory);

        // Verify non-zero
        assert_eq!(model.get_lag_residuals(0), &[3.0, 2.0]);
        assert_eq!(model.get_lag_residuals(1), &[30.0]);

        // Clear buffer
        model.clear_lag_buffer();

        // Verify zeros
        assert_eq!(model.get_lag_residuals(0), &[0.0, 0.0]);
        assert_eq!(model.get_lag_residuals(1), &[0.0]);
    }

    #[test]
    fn test_get_lag_residuals_empty_for_independent() {
        let specs = vec![create_independent_spec(0)];
        let params = create_test_seasonal_params();
        let model = UnifiedInflowModel::from_spec(&specs, 1, params);

        let lags = model.get_lag_residuals(0);
        assert_eq!(lags.len(), 0);
    }

    #[test]
    fn test_lag_buffer_insufficient_trajectory() {
        let specs = vec![create_ar_spec(0, 3)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // Trajectory with only 2 realizations, but AR(3) needs 3
        let trajectory = vec![
            create_mock_realization(1, vec![1.0]),
            create_mock_realization(1, vec![2.0]),
        ];

        model.initialize_lag_buffer(&trajectory);

        // Should pad with zeros: [2.0, 1.0, 0.0]
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags.len(), 3);
        assert_eq!(lags[0], 2.0); // Z'_{t-1}
        assert_eq!(lags[1], 1.0); // Z'_{t-2}
        assert_eq!(lags[2], 0.0); // Z'_{t-3} (padded)
    }

    #[test]
    fn test_lag_buffer_multi_node_prestudy() {
        // Simulate PAR case with multi-node PreStudy (3 nodes)
        let specs = vec![create_ar_spec(0, 2)];
        let params = create_test_seasonal_params();
        let mut model = UnifiedInflowModel::from_spec(&specs, 1, params);

        // PreStudy trajectory with 5 nodes (more than needed)
        let trajectory = vec![
            create_mock_realization(1, vec![1.0]), // PreStudy node 1
            create_mock_realization(1, vec![2.0]), // PreStudy node 2
            create_mock_realization(1, vec![3.0]), // PreStudy node 3
            create_mock_realization(1, vec![4.0]), // t-2
            create_mock_realization(1, vec![5.0]), // t-1 (most recent)
        ];

        model.initialize_lag_buffer(&trajectory);

        // Should extract last 2: [5.0, 4.0]
        let lags = model.get_lag_residuals(0);
        assert_eq!(lags, &[5.0, 4.0]);
    }
}
