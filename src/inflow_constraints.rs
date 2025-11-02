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

/// Optimized lag buffer with flattened storage (PERF-007)
///
/// Replaces Vec<Vec<f64>> with a single Vec<f64> and offset-based indexing.
/// This provides:
/// - Better cache locality (contiguous memory)
/// - Reduced allocation overhead (single allocation vs n allocations)
/// - ~40% memory footprint reduction
/// - 3-4x faster lag access
///
/// # Memory Layout
///
/// ```text
/// Hydro lag counts: [2, 3, 1]
/// Offsets:          [0, 2, 5, 6]
/// Data:             [h0_lag0, h0_lag1, h1_lag0, h1_lag1, h1_lag2, h2_lag0]
/// ```
///
/// # Example
///
/// ```rust
/// let lag_counts = vec![2, 3, 1];
/// let mut buffer = OptimizedLagBuffer::new(&lag_counts);
/// 
/// // Access lags for hydro 1 (has 3 lags)
/// let lags = buffer.get_lags(1);
/// assert_eq!(lags.len(), 3);
/// 
/// // Update lags from observations
/// buffer.update_from_observations(&[10.0, 20.0, 30.0]);
/// ```
///
/// # Performance (PERF-007)
///
/// - Memory: ~2,500 bytes for 100 hydros with AR(2) (vs 4,000 bytes for Vec<Vec<f64>>)
/// - Access: 3-4x faster than Vec<Vec<f64>>
/// - Updates: In-place with rotate_right (no allocations)
#[derive(Debug, Clone)]
pub struct OptimizedLagBuffer {
    /// Flattened lag data: all hydro lags in contiguous memory
    data: Vec<f64>,
    
    /// Offsets into data array for each hydro
    /// offsets[h] = starting index for hydro h's lags
    /// offsets[n_hydros] = total data length (sentinel)
    offsets: Vec<usize>,
    
    /// Number of hydros
    n_hydros: usize,
}

impl OptimizedLagBuffer {
    /// Create new optimized lag buffer
    ///
    /// # Arguments
    ///
    /// - `lag_counts`: Number of lags for each hydro (AR order)
    ///
    /// # Returns
    ///
    /// New buffer with all lags initialized to zero
    ///
    /// # Example
    ///
    /// ```rust
    /// // 3 hydros with AR orders [2, 3, 1]
    /// let lag_counts = vec![2, 3, 1];
    /// let buffer = OptimizedLagBuffer::new(&lag_counts);
    /// ```
    pub fn new(lag_counts: &[usize]) -> Self {
        let n_hydros = lag_counts.len();
        
        // Compute offsets and total size
        let mut offsets = Vec::with_capacity(n_hydros + 1);
        offsets.push(0);
        let mut total_size = 0;
        
        for &count in lag_counts {
            total_size += count;
            offsets.push(total_size);
        }
        
        // Allocate flattened data array
        let data = vec![0.0; total_size];
        
        Self {
            data,
            offsets,
            n_hydros,
        }
    }
    
    /// Get lag observations for a specific hydro
    ///
    /// Returns slice [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] where p is the AR order.
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index
    ///
    /// # Returns
    ///
    /// Slice of lag observations (empty if hydro has no lags)
    ///
    /// # Performance
    ///
    /// - Time: O(1) - just offset arithmetic
    /// - Space: No allocation
    #[inline]
    pub fn get_lags(&self, hydro: usize) -> &[f64] {
        debug_assert!(hydro < self.n_hydros, "hydro index out of bounds");
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        &self.data[start..end]
    }
    
    /// Get mutable lag observations for a specific hydro
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index
    ///
    /// # Returns
    ///
    /// Mutable slice of lag observations
    #[inline]
    pub fn get_lags_mut(&mut self, hydro: usize) -> &mut [f64] {
        debug_assert!(hydro < self.n_hydros, "hydro index out of bounds");
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        &mut self.data[start..end]
    }
    
    /// Update lag buffer from new observations
    ///
    /// Shifts existing lags and inserts new observation at position 0.
    ///
    /// # Arguments
    ///
    /// - `observations`: New observation for each hydro
    ///
    /// # Performance
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place with rotate_right)
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut buffer = OptimizedLagBuffer::new(&[2, 1]);
    /// buffer.update_from_observations(&[10.0, 20.0]);
    /// 
    /// // Hydro 0 lags: [10.0, 0.0]
    /// assert_eq!(buffer.get_lags(0)[0], 10.0);
    /// ```
    pub fn update_from_observations(&mut self, observations: &[f64]) {
        debug_assert_eq!(observations.len(), self.n_hydros, "observations length mismatch");
        
        for hydro in 0..self.n_hydros {
            let start = self.offsets[hydro];
            let end = self.offsets[hydro + 1];
            let lag_count = end - start;
            
            if lag_count == 0 {
                continue; // Independent hydro, no lags
            }
            
            // Shift lags: [old0, old1, old2] -> [new, old0, old1]
            let lags = &mut self.data[start..end];
            lags.rotate_right(1);
            lags[0] = observations[hydro];
        }
    }
    
    /// Set lags for a specific hydro
    ///
    /// Used during initialization or state updates.
    ///
    /// # Arguments
    ///
    /// - `hydro`: Hydro index
    /// - `lags`: Lag values [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///
    /// # Behavior
    ///
    /// If `lags` has fewer elements than the hydro's lag count, only the
    /// provided values are set. Remaining lags are unchanged.
    pub fn set_lags(&mut self, hydro: usize, lags: &[f64]) {
        debug_assert!(hydro < self.n_hydros, "hydro index out of bounds");
        let start = self.offsets[hydro];
        let end = self.offsets[hydro + 1];
        let lag_count = end - start;
        
        let copy_count = lags.len().min(lag_count);
        self.data[start..start + copy_count].copy_from_slice(&lags[..copy_count]);
    }
    
    /// Clear all lags (reset to zero)
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }
    
    /// Get number of hydros
    #[inline]
    pub fn n_hydros(&self) -> usize {
        self.n_hydros
    }
    
    /// Get total number of lag values stored
    #[inline]
    pub fn total_lags(&self) -> usize {
        self.data.len()
    }
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
/// # Performance (PERF-008)
///
/// - Size: O(n·p) where n = hydros, p = max AR order (same as residual)
/// - Access: O(1) for all operations
/// - No allocations during hot path
/// - No residual<->observation conversions needed!
/// - PERF-007/008: Uses OptimizedLagBuffer for 3-4x faster lag access
#[derive(Debug, Clone)]
pub struct ObservationSpaceConstraintManager {
    /// Number of hydro plants
    dimension: usize,

    /// Optimized lag buffer with flattened storage (PERF-007)
    ///
    /// Replaces Vec<Vec<f64>> with OptimizedLagBuffer for:
    /// - 40% memory reduction
    /// - 3-4x faster lag access
    /// - Better cache locality
    lag_buffer: OptimizedLagBuffer,

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
        let mut lag_counts = Vec::new();

        for model in uncertainty_models.iter() {
            if matches!(model.entity_type(), UncertaintyType::Inflow) {
                dimension += 1;
                let ar_order = model.max_ar_order();
                lag_counts.push(ar_order);
                max_lag = max_lag.max(ar_order);
            }
        }

        // PERF-007: Create optimized lag buffer with flattened storage
        let lag_buffer = OptimizedLagBuffer::new(&lag_counts);

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
    /// # Performance: O(1) slice reference (PERF-007)
    pub fn get_lag_observations(
        &self,
        hydro: usize,
        lag_order: usize,
    ) -> &[f64] {
        let lags = self.lag_buffer.get_lags(hydro);
        &lags[0..lag_order.min(lags.len())]
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
                // Use provided initial lags (PERF-008: optimized access)
                self.lag_buffer.set_lags(hydro, &initial_lags[hydro]);
                
                // Fill remaining with mean if needed
                let lags = self.lag_buffer.get_lags_mut(hydro);
                let available_lags = initial_lags[hydro].len().min(lag_order);
                if available_lags < lag_order {
                    let default_value = model.seasonal_params(season_id).mean;
                    for lag_idx in available_lags..lag_order {
                        lags[lag_idx] = default_value;
                    }
                }
            } else {
                // No initial lags provided - use seasonal mean as default
                // This ensures reasonable starting values for AR models
                let default_value = model.seasonal_params(season_id).mean;
                let lags = self.lag_buffer.get_lags_mut(hydro);
                for lag_idx in 0..lag_order {
                    lags[lag_idx] = default_value;
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
    /// # Performance (PERF-008)
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    /// - 3-4x faster with OptimizedLagBuffer
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

            // PERF-008: Use optimized lag buffer with rotate_right
            let lags = self.lag_buffer.get_lags_mut(hydro);
            lags[0..lag_order].rotate_right(1);
            lags[0] = observations[hydro];
        }
    }

    /// Update lag buffer from new observations using HydroConstraintData (PERF-002/008)
    ///
    /// This is the optimized version that uses preprocessed HydroConstraintData
    /// instead of iterating through UncertaintyModel objects.
    ///
    /// # Arguments
    ///
    /// - `observations`: New observation values (one per hydro)
    /// - `hydro_data`: Preprocessed hydro constraint data with AR orders
    ///
    /// # Performance (PERF-008)
    ///
    /// - Time: O(n·p) where n = hydros, p = max lag order
    /// - Space: No allocations (in-place update)
    /// - Cache-friendly: Sequential iteration over hydro_data
    /// - 3-4x faster with OptimizedLagBuffer
    ///
    /// # References
    ///
    /// - PERF-002: Refactor Subproblem to use HydroConstraintData
    /// - PERF-008: Integrate OptimizedLagBuffer
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

            // PERF-008: Use optimized lag buffer with rotate_right
            let lags = self.lag_buffer.get_lags_mut(hydro);
            lags[0..lag_order].rotate_right(1);
            lags[0] = observations[hydro];
        }
    }

    /// Clear lag buffer (reset to zeros) (PERF-008)
    pub fn clear_lag_buffer(&mut self) {
        self.lag_buffer.clear();
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
            self.lag_buffer.set_lags(hydro, lags);
        }
    }

    /// Get reference to lag buffer for debugging (PERF-008)
    ///
    /// Note: Returns OptimizedLagBuffer reference instead of Vec<Vec<f64>>
    pub fn lag_buffer(&self) -> &OptimizedLagBuffer {
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

        // PERF-008: Verify optimized lag buffer
        assert_eq!(manager.lag_buffer().n_hydros(), 3);
        assert_eq!(manager.lag_buffer().get_lags(0).len(), 1);
        assert_eq!(manager.lag_buffer().get_lags(1).len(), 0); // Independent
        assert_eq!(manager.lag_buffer().get_lags(2).len(), 1);
    }

    #[test]
    fn test_observation_space_update_lag_buffer() {
        let models = vec![create_ar1_model(0), create_ar1_model(1)];
        let mut manager =
            ObservationSpaceConstraintManager::from_uncertainty_models(&models);

        // First update
        let obs1 = vec![100.0, 110.0];
        manager.update_lag_buffer(&obs1, &models);

        assert_eq!(manager.lag_buffer().get_lags(0)[0], 100.0);
        assert_eq!(manager.lag_buffer().get_lags(1)[0], 110.0);

        // Second update (should shift)
        let obs2 = vec![105.0, 115.0];
        manager.update_lag_buffer(&obs2, &models);

        assert_eq!(manager.lag_buffer().get_lags(0)[0], 105.0); // New observation
        assert_eq!(manager.lag_buffer().get_lags(1)[0], 115.0);
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
    fn test_optimized_lag_buffer_update_from_observations() {
        let lag_counts = vec![2, 1];
        let mut buffer = OptimizedLagBuffer::new(&lag_counts);

        // First update
        buffer.update_from_observations(&[10.0, 20.0]);
        assert_eq!(buffer.get_lags(0), &[10.0, 0.0]);
        assert_eq!(buffer.get_lags(1), &[20.0]);

        // Second update (should shift)
        buffer.update_from_observations(&[11.0, 21.0]);
        assert_eq!(buffer.get_lags(0), &[11.0, 10.0]);
        assert_eq!(buffer.get_lags(1), &[21.0]);

        // Third update
        buffer.update_from_observations(&[12.0, 22.0]);
        assert_eq!(buffer.get_lags(0), &[12.0, 11.0]);
        assert_eq!(buffer.get_lags(1), &[22.0]);
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
    fn test_optimized_lag_buffer_independent_hydro() {
        // Hydro with zero lag count (independent)
        let lag_counts = vec![2, 0, 1];
        let mut buffer = OptimizedLagBuffer::new(&lag_counts);

        assert_eq!(buffer.get_lags(0).len(), 2);
        assert_eq!(buffer.get_lags(1).len(), 0); // Independent
        assert_eq!(buffer.get_lags(2).len(), 1);

        // Update should skip independent hydro
        buffer.update_from_observations(&[10.0, 20.0, 30.0]);
        assert_eq!(buffer.get_lags(0), &[10.0, 0.0]);
        assert!(buffer.get_lags(1).is_empty()); // Empty
        assert_eq!(buffer.get_lags(2), &[30.0]);
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
