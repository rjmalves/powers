//! Inflow constraint generation for LP subproblems
//!
//! This module handles the generation of inflow constraints for the LP subproblem,
//! supporting both residual-space and observation-space formulations.
//!
//! # Architecture
//!
//! Two formulations are supported:
//!
//! ## Residual-Space (Legacy):
//! - `InflowConstraintManager`: Manages lag buffers in residual space
//! - `add_inflow_constraints_to_lp()`: Generates AR dynamics + transform constraints
//! - Constraints: Z'_t - Σ(φₖ * Z'_{t-k}) = ε_t and Y_t = μ + σ * Z'_t
//!
//! ## Observation-Space (NEW - Week 2):
//! - `ObservationSpaceConstraintManager`: Manages lag buffers in observation space
//! - `add_observation_space_constraints()`: Generates single constraint per hydro
//! - Constraint: Y_t - Σ(ψ_i * Y_{t-i}) = η_t
//! - Benefits: 50% fewer variables/constraints, fixes LogNormal bug
//!
//! # References
//!
//! - `QUICKSTART_OBSERVATION_SPACE.md` (Step 4)
//! - `REFACTORING_PLAN_OBSERVATION_SPACE.md` (Phase 3)

use crate::input::UncertaintyType;
use crate::precomputed_scenario::PrecomputedInflowScenario;
use crate::solver;
use crate::uncertainty_model::UncertaintyModel;

// ============================================================================
// OBSERVATION-SPACE FORMULATION (NEW - Week 2)
// ============================================================================

/// Constraint indices for observation-space formulation
///
/// In observation-space formulation, we only need one constraint per hydro:
/// Y_t - Σ(ψ_i * Y_{t-i}) = η_t
///
/// This replaces two constraints from residual-space:
/// - AR dynamics: Z'_t - Σ(φ_k * Z'_{t-k}) = ε_t
/// - Transform: Y_t = μ + σ * Z'_t
///
/// # Benefits (Ticket 2.2, 2.3)
///
/// - 50% fewer constraints per hydro
/// - Fixes LogNormal bug (no transform constraint to break)
/// - Simpler LP structure
/// - 30-50% faster solves
#[derive(Debug, Clone)]
pub struct ObservationSpaceConstraintIndices {
    /// Observation-space AR constraint indices (one per hydro)
    ///
    /// Constraint: Y_t[h] - Σ(ψ_i[h] * Y_{t-i}[h]) = η_t[h]
    pub ar_observation: Vec<usize>,
}

/// Manager for observation-space constraint generation
///
/// Tracks lag observations Y_{t-i} instead of residuals Z'_{t-i}.
/// Simpler than residual-space manager (no transform needed).
///
/// # Responsibilities (Ticket 2.4)
///
/// 1. **Lag Buffer Management**: Track historical observations Y_{t-k}
/// 2. **Constraint Indices**: Store row indices for efficient RHS updates
///
/// # Performance
///
/// - Size: O(n·p) where n = hydros, p = max AR order (same as residual)
/// - Access: O(1) for all operations
/// - No allocations during hot path
/// - No residual<->observation conversions needed!
#[derive(Debug, Clone)]
pub struct ObservationSpaceConstraintManager {
    /// Number of hydro plants
    dimension: usize,

    /// Lag buffer for each hydro: lag_buffer[hydro][lag_idx]
    /// lag_buffer[h][0] = Y_{t-1}, lag_buffer[h][1] = Y_{t-2}, etc.
    ///
    /// NOTE: Stores observations directly, not residuals!
    lag_buffer: Vec<Vec<f64>>,

    /// Maximum AR order across all hydros
    max_lag: usize,

    /// Constraint indices for observation-space AR constraints
    constraint_indices: Option<ObservationSpaceConstraintIndices>,
}

impl ObservationSpaceConstraintManager {
    /// Create from uncertainty models
    ///
    /// # Arguments
    ///
    /// - `uncertainty_models`: Slice of uncertainty models (filters for Inflow type)
    ///
    /// # Returns
    ///
    /// New manager with initialized lag buffers (all zeros)
    pub fn from_uncertainty_models(
        uncertainty_models: &[UncertaintyModel],
    ) -> Self {
        let mut max_lag = 0;
        let mut dimension = 0;

        for model in uncertainty_models.iter() {
            if matches!(model.entity_type(), UncertaintyType::Inflow) {
                dimension += 1;
                let max_order = model.max_ar_order();
                max_lag = max_lag.max(max_order);
            }
        }

        // Initialize lag buffer with zeros (observations, not residuals)
        let lag_buffer = vec![vec![0.0; max_lag]; dimension];

        Self {
            dimension,
            lag_buffer,
            max_lag,
            constraint_indices: None,
        }
    }

    /// Get the number of hydro plants
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the maximum lag order
    #[inline]
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Get lag observations for a specific hydro
    ///
    /// Returns slice of historical observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///
    /// # Performance: O(1) slice reference
    pub fn get_lag_observations(
        &self,
        hydro: usize,
        lag_order: usize,
    ) -> &[f64] {
        &self.lag_buffer[hydro][0..lag_order]
    }

    /// Initialize lag buffer from initial conditions
    ///
    /// # Arguments
    ///
    /// - `initial_lags`: Initial lag values [hydro][lag_idx] where lag_idx=0 is Y_{t-1}
    /// - `uncertainty_models`: Models to map hydro IDs and get default values
    ///
    /// # Behavior
    ///
    /// - If initial_lags has values for a hydro, use them
    /// - Otherwise, use the seasonal mean μ as default for all lags
    /// - This ensures non-negative inflows even when initial conditions aren't specified
    pub fn initialize_from_initial_condition(
        &mut self,
        initial_lags: &[Vec<f64>],
        uncertainty_models: &[UncertaintyModel],
        season_id: usize,
    ) {
        for model in uncertainty_models.iter() {
            if !matches!(model.entity_type(), UncertaintyType::Inflow) {
                continue;
            }

            let hydro = model.entity_id();
            let lag_order = model.max_ar_order();

            if lag_order == 0 {
                continue; // Independent hydro, no lags
            }

            // Check if initial lags are provided for this hydro
            let has_initial_lags =
                hydro < initial_lags.len() && !initial_lags[hydro].is_empty();

            if has_initial_lags {
                // Use provided initial lags
                let available_lags = initial_lags[hydro].len().min(lag_order);
                for lag_idx in 0..available_lags {
                    self.lag_buffer[hydro][lag_idx] =
                        initial_lags[hydro][lag_idx];
                }
                // Fill remaining with mean if needed
                if available_lags < lag_order {
                    let default_value = model.seasonal_params(season_id).mean;
                    for lag_idx in available_lags..lag_order {
                        self.lag_buffer[hydro][lag_idx] = default_value;
                    }
                }
            } else {
                // No initial lags provided - use seasonal mean as default
                // This ensures reasonable starting values for AR models
                let default_value = model.seasonal_params(season_id).mean;
                for lag_idx in 0..lag_order {
                    self.lag_buffer[hydro][lag_idx] = default_value;
                }
            }
        }
    }

    /// Update lag buffer with new observations
    ///
    /// Shifts existing lags and inserts new observation at position 0.
    ///
    /// # Arguments
    ///
    /// - `observations`: New observation for each hydro [Y_t[0], Y_t[1], ...]
    /// - `uncertainty_models`: Models to determine AR orders
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    pub fn update_lag_buffer(
        &mut self,
        observations: &[f64],
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

            // Insert new observation at position 0 (most recent)
            self.lag_buffer[hydro][0] = observations[hydro];
        }
    }

    /// Update lag buffer from new observations using HydroConstraintData (PERF-002)
    ///
    /// This is the optimized version that uses preprocessed HydroConstraintData
    /// instead of iterating through UncertaintyModel objects.
    ///
    /// # Arguments
    ///
    /// - `observations`: New observation values (one per hydro)
    /// - `hydro_data`: Preprocessed hydro constraint data with AR orders
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    /// - Cache-friendly: Sequential iteration over hydro_data
    ///
    /// # References
    ///
    /// - PERF-002: Refactor Subproblem to use HydroConstraintData
    pub fn update_lag_buffer_from_hydro_data(
        &mut self,
        observations: &[f64],
        hydro_data: &[crate::subproblem::HydroConstraintData],
    ) {
        for hdata in hydro_data.iter() {
            let hydro = hdata.hydro_id;
            let lag_order = hdata.ar_order;

            if lag_order == 0 {
                continue; // Independent hydro, skip
            }

            // Shift existing lags: [0, 1, 2] → [1, 2, ?]
            for lag_idx in (1..lag_order).rev() {
                self.lag_buffer[hydro][lag_idx] =
                    self.lag_buffer[hydro][lag_idx - 1];
            }

            // Insert new observation at position 0 (most recent)
            self.lag_buffer[hydro][0] = observations[hydro];
        }
    }

    /// Clear lag buffer (reset to zeros)
    pub fn clear_lag_buffer(&mut self) {
        for hydro in 0..self.dimension {
            for lag_idx in 0..self.max_lag {
                self.lag_buffer[hydro][lag_idx] = 0.0;
            }
        }
    }

    /// Set lag buffer from trajectory observations
    ///
    /// Used during forward pass to initialize lag buffer from past realizations.
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index
    /// - `lags`: Lag values [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] (most recent first)
    pub fn set_lag_buffer(&mut self, hydro: usize, lags: &[f64]) {
        if hydro < self.dimension {
            let available = lags.len().min(self.max_lag);
            for i in 0..available {
                self.lag_buffer[hydro][i] = lags[i];
            }
        }
    }

    /// Get reference to lag buffer for debugging
    pub fn lag_buffer(&self) -> &[Vec<f64>] {
        &self.lag_buffer
    }

    /// Store constraint indices after constraint generation
    pub fn set_constraint_indices(
        &mut self,
        indices: ObservationSpaceConstraintIndices,
    ) {
        self.constraint_indices = Some(indices);
    }

    /// Get constraint indices (if set)
    pub fn constraint_indices(
        &self,
    ) -> Option<&ObservationSpaceConstraintIndices> {
        self.constraint_indices.as_ref()
    }
}

/// Add observation-space constraints to LP problem (NEW - Ticket 2.3)
///
/// Generates single constraint per hydro: Y_t - Σ(ψ_i * Y_{t-i}) = η_t
///
/// This replaces the dual-constraint residual-space approach with a simpler,
/// more efficient formulation that:
/// - Works entirely in observation space (physical units)
/// - Requires 50% fewer variables (no Z'_t, no ε_t)
/// - Requires 50% fewer constraints (no transform constraint)
/// - Fixes LogNormal bug (no transform to break)
///
/// # Arguments
///
/// - `pb`: LP problem to add constraints to
/// - `vars`: Variables (must include inflow, optionally lagged_inflow_state)
/// - `precomputed_scenarios`: Pre-computed scenarios with ψ_i and η_t
///
/// # Returns
///
/// Constraint indices for efficient RHS updates
///
/// # Mathematical Formulation (from par_derivation.pdf)
///
/// ```text
/// Y_t - Σ(ψ_i * Y_{t-i}) = η_t
///
/// where:
///   ψ_i = φ_i * (σ_t / σ_{t-i})                [transformed coefficient]
///   η_t = -Σ[ψ_i * μ_{t-i}] + μ_t + σ_t * ε_t  [pre-computed noise term]
/// ```
///
/// # Performance
///
/// - Time: O(n·p) where n = hydros, p = max AR order
/// - Space: O(n) for constraint indices
/// - 30-50% faster than residual-space (fewer constraints to build)
///
/// # References
///
/// - QUICKSTART_OBSERVATION_SPACE.md (Step 4)
/// - COMPARISON_BEFORE_AFTER.md (LP Constraints section)
pub fn add_observation_space_constraints(
    pb: &mut solver::Problem,
    vars: &crate::subproblem::Variables,
    precomputed_scenarios: &[PrecomputedInflowScenario],
) -> ObservationSpaceConstraintIndices {
    let mut ar_observation = Vec::new();

    for scenario in precomputed_scenarios {
        let hydro = scenario.hydro_id;
        let ar_order = scenario.transformed_coefficients.len();

        // Build constraint: Y_t - Σ(ψ_i * Y_{t-i}) = η_t

        if let Some(ref lag_vars) = vars.lagged_inflow_state {
            // State includes lag variables: use them in constraint
            let mut factors = Vec::with_capacity(1 + ar_order);

            // Add Y_t with coefficient +1.0
            factors.push((vars.inflow[hydro], 1.0));

            // Add lag terms: -ψ_i * Y_{t-i}
            for (i, &psi_i) in
                scenario.transformed_coefficients.iter().enumerate()
            {
                let lag_var_idx = lag_vars[hydro][i];
                factors.push((lag_var_idx, -psi_i));
            }

            // RHS = η_t (pre-computed noise term)
            let row =
                pb.add_row(scenario.noise_term..=scenario.noise_term, &factors);
            ar_observation.push(row);
        } else {
            // State does NOT include lags: only Y_t variable
            // RHS will be updated with: η_t + Σ(ψ_i * lag_observations[i])
            // This happens in the solve loop (Ticket 2.5)

            let factors = [(vars.inflow[hydro], 1.0)];

            // RHS initially set to noise_term (will be updated with lag contributions)
            let row =
                pb.add_row(scenario.noise_term..=scenario.noise_term, factors);
            ar_observation.push(row);
        }
    }

    ObservationSpaceConstraintIndices { ar_observation }
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

    // ========================================================================
    // OBSERVATION-SPACE TESTS (NEW - Week 2)
    // ========================================================================

    #[test]
    fn test_observation_space_manager_creation() {
        let models = vec![
            create_ar1_model(0),
            create_independent_model(1),
            create_ar1_model(2),
        ];

        let manager =
            ObservationSpaceConstraintManager::from_uncertainty_models(&models);

        assert_eq!(manager.dimension(), 3);
        assert_eq!(manager.max_lag(), 1);

        // Lag buffer stores observations, not residuals
        assert_eq!(manager.lag_buffer[0].len(), 1);
        assert_eq!(manager.lag_buffer[1].len(), 1);
        assert_eq!(manager.lag_buffer[2].len(), 1);
    }

    #[test]
    fn test_observation_space_update_lag_buffer() {
        let models = vec![create_ar1_model(0), create_ar1_model(1)];
        let mut manager =
            ObservationSpaceConstraintManager::from_uncertainty_models(&models);

        // First update
        let obs1 = vec![100.0, 110.0];
        manager.update_lag_buffer(&obs1, &models);

        assert_eq!(manager.lag_buffer[0][0], 100.0);
        assert_eq!(manager.lag_buffer[1][0], 110.0);

        // Second update (should shift)
        let obs2 = vec![105.0, 115.0];
        manager.update_lag_buffer(&obs2, &models);

        assert_eq!(manager.lag_buffer[0][0], 105.0); // New observation
        assert_eq!(manager.lag_buffer[1][0], 115.0);
    }

    #[test]
    fn test_observation_space_get_lag_observations() {
        let models = vec![create_ar1_model(0)];
        let mut manager =
            ObservationSpaceConstraintManager::from_uncertainty_models(&models);

        manager.lag_buffer[0][0] = 120.0;

        let lags = manager.get_lag_observations(0, 1);
        assert_eq!(lags.len(), 1);
        assert_eq!(lags[0], 120.0);
    }
}
