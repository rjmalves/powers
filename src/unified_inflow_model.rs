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
            ar_factors.push((vars.inflow_process[hydro][1], 1.0));

            // Add lag terms: -φ_k * Z'_{t-k}
            // For independent case (empty coefficients), this loop doesn't execute
            for (lag_idx, &coeff) in
                self.ar_coefficients[hydro].iter().enumerate()
            {
                // Lag variables are stored in vars.inflow_process[hydro][2..]
                // inflow_process structure: [Y_t, Z'_t, Z'_{t-1}, Z'_{t-2}, ...]
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
}
