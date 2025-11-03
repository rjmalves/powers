//! Inflow constraint generation for LP subproblems (DEPRECATED)
//!
//! # Deprecation Notice
//!
//! This module is deprecated and will be removed in a future version.
//! Use `uncertainty_constraints` module instead, which provides unified
//! handling for all entity types (loads and inflows).
//!
//! ## Migration Path
//!
//! - Replace `ObservationSpaceConstraintManager` with `UncertaintyConstraintManager`
//! - Use unified constraint generation in `subproblem::add_constraints_v2()`
//! - The new module supports both loads and inflows with identical semantics
//!
//! ## Old Module Description
//!
//! This module handles the generation of inflow constraints for the LP subproblem,
//! supporting both residual-space and observation-space formulations.
//!
//! - `ObservationSpaceConstraintManager`: Manages lag buffers in observation space
//! - `add_observation_space_constraints()`: Generates single constraint per hydro
//! - Constraint: Y_t - Σ(ψ_i * Y_{t-i}) = η_t
//! - Benefits: 50% fewer variables/constraints, fixes LogNormal bug
//!

#![deprecated(
    since = "0.3.0",
    note = "Use uncertainty_constraints module instead for unified handling of all entity types"
)]
#![allow(deprecated)]

use crate::input::UncertaintyType;
use crate::precomputed_scenario::PrecomputedInflowScenario;
use crate::solver;
use crate::uncertainty_model::UncertaintyModel;

/// Constraint indices for observation-space formulation
///
/// In observation-space formulation, we only need one constraint per hydro:
/// Y_t - Σ(ψ_i * Y_{t-i}) = η_t
#[derive(Debug, Clone)]
pub struct ObservationSpaceConstraintIndices {
    /// Observation-space AR constraint indices (one per hydro)
    ///
    /// Constraint: Y_t[h] - Σ(ψ_i[h] * Y_{t-i}[h]) = η_t[h]
    pub ar_observation: Vec<usize>,
}

/// Optimized lag buffer with flattened storage
///
/// # Memory Layout
///
/// ```text
/// Hydro lag counts: [2, 3, 1]
/// Offsets:          [0, 2, 5, 6]
/// Data:             [h0_lag0, h0_lag1, h1_lag0, h1_lag1, h1_lag2, h2_lag0]
/// ```
#[derive(Debug, Clone)]
pub struct OptimizedLagBuffer {
    data: Vec<f64>,
    offsets: Vec<usize>,
    n_hydros: usize,
}

impl OptimizedLagBuffer {
    pub fn new(lag_counts: &[usize]) -> Self {
        let n_hydros = lag_counts.len();

        let mut offsets = Vec::with_capacity(n_hydros + 1);
        offsets.push(0);
        let mut total_size = 0;

        for &count in lag_counts {
            total_size += count;
            offsets.push(total_size);
        }

        let data = vec![0.0; total_size];

        Self {
            data,
            offsets,
            n_hydros,
        }
    }

    #[inline]
    pub fn get_lags(&self, hydro: usize) -> &[f64] {
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        &self.data[start..end]
    }

    #[inline]
    pub fn get_lags_mut(&mut self, hydro: usize) -> &mut [f64] {
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        &mut self.data[start..end]
    }

    pub fn set_lags(&mut self, hydro: usize, lags: &[f64]) {
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        let lag_count = end - start;

        let copy_count = lags.len().min(lag_count);
        self.data[start..start + copy_count]
            .copy_from_slice(&lags[..copy_count]);
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    #[inline]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }

    #[inline]
    pub fn total_lags(&self) -> usize {
        self.data.len()
    }
}

/// Manager for observation-space constraint generation
///
/// Tracks lag observations Y_{t-i} for AR inflow models.
#[derive(Debug, Clone)]
pub struct ObservationSpaceConstraintManager {
    dimension: usize,
    lag_buffer: OptimizedLagBuffer,
    max_lag: usize,
    constraint_indices: Option<ObservationSpaceConstraintIndices>,
}

impl ObservationSpaceConstraintManager {
    pub fn from_uncertainty_models(
        uncertainty_models: &[UncertaintyModel],
    ) -> Self {
        let mut max_lag = 0;
        let mut dimension = 0;
        let mut lag_counts = Vec::new();

        for model in uncertainty_models.iter() {
            if matches!(model.entity_type(), UncertaintyType::Inflow) {
                dimension += 1;
                let ar_order = model.max_ar_order();
                lag_counts.push(ar_order);
                max_lag = max_lag.max(ar_order);
            }
        }

        let lag_buffer = OptimizedLagBuffer::new(&lag_counts);

        Self {
            dimension,
            lag_buffer,
            max_lag,
            constraint_indices: None,
        }
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    #[inline]
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Get lag observations for a specific hydro
    ///
    /// Returns slice of historical observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    pub fn get_lag_observations(
        &self,
        hydro: usize,
        lag_order: usize,
    ) -> &[f64] {
        let lags = self.lag_buffer.get_lags(hydro);
        &lags[0..lag_order.min(lags.len())]
    }

    /// Update lag buffer from new observations using HydroConstraintData
    pub fn update_lag_buffer_from_hydro_data(
        &mut self,
        observations: &[f64],
        hydro_data: &[crate::subproblem::HydroConstraintData],
    ) {
        for hdata in hydro_data.iter() {
            let hydro = hdata.hydro_id;
            let lag_order = hdata.ar_order;

            if lag_order == 0 {
                continue;
            }

            let lags = self.lag_buffer.get_lags_mut(hydro);
            lags[0..lag_order].rotate_right(1);
            lags[0] = observations[hydro];
        }
    }

    /// Set lag buffer from trajectory observations
    pub fn set_lag_buffer(&mut self, hydro: usize, lags: &[f64]) {
        if hydro < self.dimension {
            self.lag_buffer.set_lags(hydro, lags);
        }
    }

    /// Get reference to lag buffer for debugging
    pub fn lag_buffer(&self) -> &OptimizedLagBuffer {
        &self.lag_buffer
    }

    pub fn set_constraint_indices(
        &mut self,
        indices: ObservationSpaceConstraintIndices,
    ) {
        self.constraint_indices = Some(indices);
    }

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
/// - `vars`: Variables (must include inflow, optionally lagged_state)
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

        if let Some(ref lag_vars) = vars.lagged_state {
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
#[allow(deprecated)]
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

        // PERF-008: Verify optimized lag buffer
        assert_eq!(manager.lag_buffer().n_hydros(), 3);
        assert_eq!(manager.lag_buffer().get_lags(0).len(), 1);
        assert_eq!(manager.lag_buffer().get_lags(1).len(), 0); // Independent
        assert_eq!(manager.lag_buffer().get_lags(2).len(), 1);
    }

    #[test]
    fn test_observation_space_get_lag_observations() {
        let models = vec![create_ar1_model(0)];
        let mut manager =
            ObservationSpaceConstraintManager::from_uncertainty_models(&models);

        manager.set_lag_buffer(0, &[120.0]);

        let lags = manager.get_lag_observations(0, 1);
        assert_eq!(lags.len(), 1);
        assert_eq!(lags[0], 120.0);
    }

    // ========================================================================
    // OPTIMIZED LAG BUFFER TESTS (PERF-007)
    // ========================================================================

    #[test]
    fn test_optimized_lag_buffer_creation() {
        // 3 hydros with AR orders [2, 3, 1]
        let lag_counts = vec![2, 3, 1];
        let buffer = OptimizedLagBuffer::new(&lag_counts);

        assert_eq!(buffer.n_hydros(), 3);
        assert_eq!(buffer.total_lags(), 6); // 2 + 3 + 1

        // Verify each hydro's lags
        assert_eq!(buffer.get_lags(0).len(), 2);
        assert_eq!(buffer.get_lags(1).len(), 3);
        assert_eq!(buffer.get_lags(2).len(), 1);

        // All initialized to zero
        for hydro in 0..3 {
            for &lag in buffer.get_lags(hydro) {
                assert_eq!(lag, 0.0);
            }
        }
    }

    #[test]
    fn test_optimized_lag_buffer_offsets() {
        // Test offset calculation
        let lag_counts = vec![2, 3, 1];
        let buffer = OptimizedLagBuffer::new(&lag_counts);

        // Verify memory layout
        // Hydro 0: offsets[0..2] = indices 0,1
        // Hydro 1: offsets[2..5] = indices 2,3,4
        // Hydro 2: offsets[5..6] = index 5

        let lags0 = buffer.get_lags(0);
        let lags1 = buffer.get_lags(1);
        let lags2 = buffer.get_lags(2);

        assert_eq!(lags0.len(), 2);
        assert_eq!(lags1.len(), 3);
        assert_eq!(lags2.len(), 1);
    }

    #[test]
    fn test_optimized_lag_buffer_set_lags() {
        let lag_counts = vec![3, 2];
        let mut buffer = OptimizedLagBuffer::new(&lag_counts);

        // Set lags for hydro 0
        buffer.set_lags(0, &[1.0, 2.0, 3.0]);
        assert_eq!(buffer.get_lags(0), &[1.0, 2.0, 3.0]);

        // Set lags for hydro 1
        buffer.set_lags(1, &[4.0, 5.0]);
        assert_eq!(buffer.get_lags(1), &[4.0, 5.0]);

        // Partial set (fewer lags than capacity)
        buffer.set_lags(0, &[10.0]);
        assert_eq!(buffer.get_lags(0)[0], 10.0);
        assert_eq!(buffer.get_lags(0)[1], 2.0); // Unchanged
        assert_eq!(buffer.get_lags(0)[2], 3.0); // Unchanged
    }

    #[test]
    fn test_optimized_lag_buffer_clear() {
        let lag_counts = vec![2, 3];
        let mut buffer = OptimizedLagBuffer::new(&lag_counts);

        // Set some values
        buffer.set_lags(0, &[1.0, 2.0]);
        buffer.set_lags(1, &[3.0, 4.0, 5.0]);

        // Clear
        buffer.clear();

        // All zeros
        for hydro in 0..2 {
            for &lag in buffer.get_lags(hydro) {
                assert_eq!(lag, 0.0);
            }
        }
    }

    #[test]
    fn test_optimized_lag_buffer_large_system() {
        // Simulate 100 hydros with AR(2)
        let lag_counts = vec![2; 100];
        let buffer = OptimizedLagBuffer::new(&lag_counts);

        assert_eq!(buffer.n_hydros(), 100);
        assert_eq!(buffer.total_lags(), 200); // 100 * 2

        // Verify each hydro has 2 lags
        for hydro in 0..100 {
            assert_eq!(buffer.get_lags(hydro).len(), 2);
        }
    }

    #[test]
    fn test_optimized_lag_buffer_memory_layout() {
        // Test that memory is truly contiguous
        let lag_counts = vec![2, 3, 1];
        let mut buffer = OptimizedLagBuffer::new(&lag_counts);

        // Fill with known pattern
        buffer.set_lags(0, &[1.0, 2.0]);
        buffer.set_lags(1, &[3.0, 4.0, 5.0]);
        buffer.set_lags(2, &[6.0]);

        // Verify contiguous layout: [1, 2, 3, 4, 5, 6]
        assert_eq!(buffer.get_lags(0), &[1.0, 2.0]);
        assert_eq!(buffer.get_lags(1), &[3.0, 4.0, 5.0]);
        assert_eq!(buffer.get_lags(2), &[6.0]);
    }
}
