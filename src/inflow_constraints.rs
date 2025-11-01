//! Inflow constraint generation for LP subproblems
//!
//! This module handles the generation of AR dynamics and observation transform
//! constraints for the LP subproblem, working with the new `UncertaintyModel` API.
//!
//! # Architecture
//!
//! Replaces the constraint generation logic from `UnifiedInflowModel` with a cleaner
//! separation of concerns:
//! - `InflowConstraintManager`: Manages lag buffers and constraint indices
//! - `add_inflow_constraints_to_lp()`: Generates constraints from UncertaintyModel
//!
//! # Constraints Generated
//!
//! **AR Dynamics**: Z'_t - Σ(φₖ * Z'_{t-k}) = ε_t
//! - Independent: Z'_t = ε_t (empty coefficients)
//! - AR(p): Full AR equation with lag terms
//!
//! **Observation Transform**: Y_t - σ_s * Z'_t = μ_s
//! - Links residual space (Z') to observation space (Y)

use crate::input::UncertaintyType;
use crate::solver;
use crate::uncertainty_model::UncertaintyModel;

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

/// Manager for inflow constraint generation and lag buffer tracking
///
/// # Responsibilities
///
/// 1. **Lag Buffer Management**: Track historical residuals Z'_{t-k} for AR models
/// 2. **Constraint Indices**: Store row indices for efficient RHS updates
/// 3. **AR Coefficients**: Store per-hydro, per-season AR coefficients for RHS updates
///
/// # Performance
///
/// - Size: O(n·p·s) where n = hydros, p = max AR order, s = seasons
/// - Access: O(1) for all operations
/// - No allocations during hot path (solve loop)
#[derive(Debug, Clone)]
pub struct InflowConstraintManager {
    /// Number of hydro plants
    dimension: usize,

    /// Lag buffer for each hydro: lag_buffer[hydro][lag_idx]
    /// lag_buffer[h][0] = Z'_{t-1}, lag_buffer[h][1] = Z'_{t-2}, etc.
    lag_buffer: Vec<Vec<f64>>,

    /// Maximum AR order across all hydros (cached for efficiency)
    max_lag: usize,

    /// AR coefficients for each hydro and season: ar_coefficients[hydro][season][lag]
    /// For independent models, this is empty (zero-length inner vectors)
    ar_coefficients: Vec<Vec<Vec<f64>>>,

    /// Constraint indices for AR dynamics and observation transform
    constraint_indices: Option<ConstraintIndices>,
}

impl InflowConstraintManager {
    /// Create from uncertainty models
    ///
    /// Extracts inflow models and initializes lag buffers with appropriate size.
    /// Also extracts AR coefficients for each season.
    ///
    /// # Arguments
    ///
    /// - `uncertainty_models`: Slice of uncertainty models (filters for Inflow type)
    /// - `num_seasons`: Number of seasons in the problem
    ///
    /// # Returns
    ///
    /// New manager with initialized lag buffers (all zeros) and AR coefficients
    ///
    /// # Performance
    ///
    /// - Time: O(n·s) where n = number of models, s = seasons
    /// - Space: O(n·p·s) where p = max AR order
    pub fn from_uncertainty_models(
        uncertainty_models: &[UncertaintyModel],
        num_seasons: usize,
    ) -> Self {
        // Find all inflow models and compute max lag
        let mut max_lag = 0;
        let mut dimension = 0;

        for model in uncertainty_models.iter() {
            if matches!(model.entity_type(), UncertaintyType::Inflow) {
                dimension += 1;
                let max_order = model.max_ar_order();
                max_lag = max_lag.max(max_order);
            }
        }

        // Initialize lag buffer with zeros
        let lag_buffer = vec![vec![0.0; max_lag]; dimension];

        // Extract AR coefficients per hydro, per season
        let mut ar_coefficients = Vec::with_capacity(dimension);

        for model in uncertainty_models.iter() {
            if matches!(model.entity_type(), UncertaintyType::Inflow) {
                let mut season_coeffs = Vec::with_capacity(num_seasons);

                for season in 0..num_seasons {
                    let coeffs = match model {
                        UncertaintyModel::Independent { .. } => vec![],
                        UncertaintyModel::PeriodicAR { par_params, .. } => {
                            par_params.ar_coefficients(season).to_vec()
                        }
                    };
                    season_coeffs.push(coeffs);
                }

                ar_coefficients.push(season_coeffs);
            }
        }

        Self {
            dimension,
            lag_buffer,
            max_lag,
            ar_coefficients,
            constraint_indices: None,
        }
    }

    /// Get the number of hydro plants
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the maximum lag order across all hydros
    #[inline]
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Initialize lag buffer from trajectory of past realizations
    ///
    /// Extracts the last p residuals from the trajectory for each hydro,
    /// where p is the lag order for that hydro.
    ///
    /// # Arguments
    ///
    /// - `trajectory`: Slice of past realizations (ordered oldest to newest)
    /// - `uncertainty_models`: Models to determine AR orders
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (updates pre-allocated buffer)
    ///
    /// # Panics
    ///
    /// Panics if trajectory is empty (defensive check)
    pub fn initialize_lag_buffer(
        &mut self,
        trajectory: &[crate::subproblem::Realization],
        uncertainty_models: &[UncertaintyModel],
    ) {
        assert!(!trajectory.is_empty(), "Trajectory must not be empty");

        let traj_len = trajectory.len();

        // Extract lag order for each hydro from uncertainty models
        for model in uncertainty_models.iter() {
            if !matches!(model.entity_type(), UncertaintyType::Inflow) {
                continue;
            }

            let hydro = model.entity_id();
            let lag_order = model.max_ar_order();

            if lag_order == 0 {
                continue; // Independent hydro, skip
            }

            // Extract last p residuals from trajectory
            // trajectory layout: [..., t-3, t-2, t-1]
            // For AR(2), we want: lag_buffer[h][0] = trajectory[len-1] (t-1)
            //                     lag_buffer[h][1] = trajectory[len-2] (t-2)
            for lag_idx in 0..lag_order {
                let traj_idx = traj_len.checked_sub(1 + lag_idx);

                if let Some(idx) = traj_idx {
                    let residual = trajectory[idx].inflow_residual[hydro];

                    // Debug validation: residuals should be normalized
                    debug_assert!(
                        residual.abs() < 50.0,
                        "Residual Z'[{}][lag={}] = {} is out of range",
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
    ///
    /// # Arguments
    ///
    /// - `realization`: New realization with `inflow_residual` field
    /// - `uncertainty_models`: Models to determine AR orders
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    pub fn update_lag_buffer(
        &mut self,
        realization: &crate::subproblem::Realization,
        uncertainty_models: &[UncertaintyModel],
    ) {
        for model in uncertainty_models.iter() {
            if !matches!(model.entity_type(), UncertaintyType::Inflow) {
                continue;
            }

            let hydro = model.entity_id();
            let lag_order = model.max_ar_order();

            if lag_order == 0 {
                continue; // Independent hydro, skip
            }

            // Shift existing lags: [0, 1, 2] → [1, 2, ?]
            for lag_idx in (1..lag_order).rev() {
                self.lag_buffer[hydro][lag_idx] =
                    self.lag_buffer[hydro][lag_idx - 1];
            }

            // Insert new residual at position 0 (most recent)
            self.lag_buffer[hydro][0] = realization.inflow_residual[hydro];
        }
    }

    /// Get lag residuals for a specific hydro
    ///
    /// Returns slice of historical residuals [Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-p}]
    ///
    /// # Performance: O(1) slice reference
    pub fn get_lag_residuals(&self, hydro: usize, lag_order: usize) -> &[f64] {
        &self.lag_buffer[hydro][0..lag_order]
    }

    /// Get AR coefficients for a specific hydro and season
    ///
    /// Returns slice of AR coefficients [φ_1, φ_2, ..., φ_p] for the given season.
    /// For independent models, returns an empty slice.
    ///
    /// # Performance: O(1) slice reference
    pub fn get_ar_coefficients(&self, hydro: usize, season: usize) -> &[f64] {
        &self.ar_coefficients[hydro][season]
    }

    /// Clear lag buffer (reset to zeros)
    ///
    /// Useful for testing and reinitialization.
    pub fn clear_lag_buffer(&mut self) {
        for hydro in 0..self.dimension {
            for lag_idx in 0..self.max_lag {
                self.lag_buffer[hydro][lag_idx] = 0.0;
            }
        }
    }

    /// Get reference to lag buffer for debugging
    pub fn lag_buffer(&self) -> &[Vec<f64>] {
        &self.lag_buffer
    }

    /// Store constraint indices after constraint generation
    pub fn set_constraint_indices(&mut self, indices: ConstraintIndices) {
        self.constraint_indices = Some(indices);
    }

    /// Get constraint indices (if set)
    pub fn constraint_indices(&self) -> Option<&ConstraintIndices> {
        self.constraint_indices.as_ref()
    }
}

/// Add inflow constraints to LP problem
///
/// Generates AR dynamics and observation transform constraints for all inflow
/// uncertainty models.
///
/// # Constraints Generated
///
/// **AR Dynamics**: Z'_t - Σ(φₖ * Z'_{t-k}) = ε_t
/// - Two modes based on state type:
///   - **StorageAndInflowState**: Lag variables included in constraint
///   - **StorageState**: Lags from lag_buffer (updated in RHS)
///
/// **Observation Transform**: Y_t - σ_s * Z'_t = μ_s
/// - Links residual space to observation space
///
/// # Arguments
///
/// - `pb`: LP problem to add constraints to
/// - `vars`: Variables (must include inflow and inflow_residual)
/// - `season_id`: Current season ID for seasonal parameter lookup
/// - `uncertainty_models`: All uncertainty models (filters for Inflow type)
///
/// # Returns
///
/// Constraint indices for efficient RHS updates
///
/// # Performance
///
/// - Time: O(n·p) where n = hydros, p = max AR order
/// - Space: O(n) for constraint indices storage
pub fn add_inflow_constraints_to_lp(
    pb: &mut solver::Problem,
    vars: &crate::subproblem::Variables,
    season_id: usize,
    uncertainty_models: &[UncertaintyModel],
) -> ConstraintIndices {
    let mut ar_dynamics = Vec::new();
    let mut observation_transform = Vec::new();

    for model in uncertainty_models.iter() {
        if !matches!(model.entity_type(), UncertaintyType::Inflow) {
            continue;
        }

        let hydro = model.entity_id();
        let seasonal_params = model.seasonal_params(season_id);

        // ============================================================
        // AR DYNAMICS CONSTRAINT
        // ============================================================
        // Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
        //
        // Two cases based on state type:
        //
        // **StorageAndInflowState**: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
        //   Lag variables are part of the state, included as LP variables
        //
        // **StorageState**: Z'_t = RHS
        //   RHS = Σ(φ_k * lag_buffer_k) + ε_t (updated at solve time)
        //   Lags are tracked externally in InflowConstraintManager.lag_buffer

        let ar_row = match model {
            UncertaintyModel::Independent { .. } => {
                // Independent: Z'_t = ε_t (RHS updated at solve time)
                let ar_factors = [(vars.inflow_residual[hydro], 1.0)];
                pb.add_row(0.0..=0.0, ar_factors)
            }
            UncertaintyModel::PeriodicAR { par_params, .. } => {
                let ar_coeffs = par_params.ar_coefficients(season_id);
                let ar_order = ar_coeffs.len();

                if let Some(ref lag_vars) = vars.lagged_inflow_state {
                    // StorageAndInflowState: Include lag variables in constraint
                    let mut ar_factors = Vec::with_capacity(1 + ar_order);

                    // Add Z'_t with coefficient +1.0
                    ar_factors.push((vars.inflow_residual[hydro], 1.0));

                    // Add lag terms: -φ_k * Z'_{t-k}
                    for (k, &coeff) in ar_coeffs.iter().enumerate() {
                        let lag_var_idx = lag_vars[hydro][k];
                        ar_factors.push((lag_var_idx, -coeff));
                    }

                    // RHS = 0.0 initially (will be updated to ε_t at solve time)
                    pb.add_row(0.0..=0.0, &ar_factors)
                } else {
                    // StorageState: Only Z'_t variable, RHS includes lag contributions
                    // Constraint: Z'_t = RHS
                    // where RHS = Σ(φ_k * lag_buffer[k]) + ε_t
                    let ar_factors = [(vars.inflow_residual[hydro], 1.0)];

                    // RHS = 0.0 initially (will be updated to include lags + ε_t)
                    pb.add_row(0.0..=0.0, ar_factors)
                }
            }
        };

        ar_dynamics.push(ar_row);

        // ============================================================
        // OBSERVATION TRANSFORMATION: Y_t - σ_s*Z'_t = μ_s
        // ============================================================

        let mu = seasonal_params.mean;
        let sigma = seasonal_params.std_dev;

        let obs_factors = [
            (vars.inflow[hydro], 1.0), // Y_t with coefficient +1.0
            (vars.inflow_residual[hydro], -sigma), // Z'_t with coefficient -σ_s
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncertainty_model::{
        DistributionType, PARParams, SeasonalParams,
    };

    fn create_independent_model(entity_id: usize) -> UncertaintyModel {
        let seasonal_params = vec![
            SeasonalParams {
                mean: 100.0,
                std_dev: 20.0,
                distribution: DistributionType::Normal,
            };
            12
        ];

        UncertaintyModel::Independent {
            entity_type: UncertaintyType::Inflow,
            entity_id,
            seasonal_params,
        }
    }

    fn create_ar1_model(entity_id: usize) -> UncertaintyModel {
        let par_params = PARParams {
            num_seasons: 12,
            ar_orders: vec![1; 12],
            ar_coefficients: vec![vec![0.7]; 12],
            seasonal_means: vec![100.0; 12],
            seasonal_stds: vec![20.0; 12],
            seasonal_distributions: vec![DistributionType::Normal; 12],
            max_ar_order: 1,
        };

        UncertaintyModel::PeriodicAR {
            entity_type: UncertaintyType::Inflow,
            entity_id,
            par_params,
        }
    }

    #[test]
    fn test_create_manager_from_independent_models() {
        let models = vec![
            create_independent_model(0),
            create_independent_model(1),
            create_independent_model(2),
        ];

        let manager =
            InflowConstraintManager::from_uncertainty_models(&models, 12);

        assert_eq!(manager.dimension(), 3);
        assert_eq!(manager.max_lag(), 0);

        // Check AR coefficients are empty for independent models
        for hydro in 0..3 {
            for season in 0..12 {
                assert!(manager.get_ar_coefficients(hydro, season).is_empty());
            }
        }
    }

    #[test]
    fn test_create_manager_from_ar1_models() {
        let models = vec![
            create_ar1_model(0),
            create_ar1_model(1),
            create_ar1_model(2),
        ];

        let manager =
            InflowConstraintManager::from_uncertainty_models(&models, 12);

        assert_eq!(manager.dimension(), 3);
        assert_eq!(manager.max_lag(), 1);
        assert_eq!(manager.lag_buffer[0].len(), 1);
        assert_eq!(manager.lag_buffer[1].len(), 1);
        assert_eq!(manager.lag_buffer[2].len(), 1);

        // Check AR coefficients for AR(1) models
        for hydro in 0..3 {
            for season in 0..12 {
                let coeffs = manager.get_ar_coefficients(hydro, season);
                assert_eq!(coeffs.len(), 1);
                assert!((coeffs[0] - 0.7).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_create_manager_from_mixed_models() {
        let models = vec![
            create_ar1_model(0),
            create_independent_model(1),
            create_ar1_model(2),
        ];

        let manager =
            InflowConstraintManager::from_uncertainty_models(&models, 12);

        assert_eq!(manager.dimension(), 3);
        assert_eq!(manager.max_lag(), 1);
    }

    #[test]
    fn test_clear_lag_buffer() {
        let models = vec![create_ar1_model(0)];
        let mut manager =
            InflowConstraintManager::from_uncertainty_models(&models, 12);

        // Set some values
        manager.lag_buffer[0][0] = 5.0;

        manager.clear_lag_buffer();

        assert_eq!(manager.lag_buffer[0][0], 0.0);
    }
}
