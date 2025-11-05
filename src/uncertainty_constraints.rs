//! Unified uncertainty constraint management for LP subproblems
//!
//! This module handles observation-space constraints for all uncertain entities
//! (both loads and inflows), using a unified approach:
//!
//! Y_t[i] = deterministic_base[i] + σ[i]·η_t[i] + Σ_{k=1}^{p} ψ_k[i]·Y_{t-k}[i]
//!
//! Where:
//! - Y_t[i]: Observation variable (load or inflow)
//! - η_t[i]: Innovation variable (from SAA)
//! - deterministic_base[i]: Precomputed μ - Σ(φ_k·μ_{season-k})
//! - σ[i]: Seasonal standard deviation
//! - ψ_k[i]: Transformed AR coefficients
//! - p: AR order (can be 0 for independent models)

use crate::temporal_model::TemporalModel;

/// Constraint indices for observation-space formulation
///
/// Stores LP constraint indices for uncertainty constraints.
/// One constraint per entity (load or inflow).
#[derive(Debug, Clone)]
pub struct UncertaintyConstraintIndices {
    /// Constraint indices for all entities (loads + inflows)
    ///
    /// Constraint: Y_t[i] = deterministic_base + σ·η_t + Σ(ψ_k·Y_{t-k})
    pub observation_constraints: Vec<usize>,
}

/// Unified lag buffer for all uncertain entities
///
/// Manages lag observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}] for both
/// loads and inflows in a single, cache-friendly data structure.
///
/// # Memory Layout
///
/// Entities (loads + inflows) with lag counts [2, 0, 3, 1]:
/// Offsets:  [0, 2, 2, 5, 6]
/// Data:     [e0_lag0, e0_lag1, e2_lag0, e2_lag1, e2_lag2, e3_lag0]
///
/// Entity 1 has ar_order=0, so no lags stored.
#[derive(Debug, Clone)]
pub struct UnifiedLagBuffer {
    data: Vec<f64>,
    offsets: Vec<usize>,
    n_entities: usize,
}

impl UnifiedLagBuffer {
    /// Create lag buffer from temporal models
    ///
    /// Automatically determines lag counts from max_ar_order of each model.
    pub fn from_temporal_models(models: &[TemporalModel]) -> Self {
        let lag_counts: Vec<usize> =
            models.iter().map(|m| m.max_ar_order).collect();
        Self::new(&lag_counts)
    }

    /// Create a new lag buffer with specified lag counts per entity
    ///
    /// # Arguments
    ///
    /// * `lag_counts` - Number of lags to store for each entity
    ///
    /// # Memory Layout
    ///
    /// Lags are stored contiguously for cache efficiency.
    /// Offsets array enables O(1) access to each entity's lags.
    pub fn new(lag_counts: &[usize]) -> Self {
        let n_entities = lag_counts.len();

        // Build offsets array: cumulative sum of lag counts
        let mut offsets = Vec::with_capacity(n_entities + 1);
        let mut cumsum = 0;
        for &count in lag_counts {
            offsets.push(cumsum);
            cumsum += count;
        }
        offsets.push(cumsum);

        // Allocate data array
        let data = vec![0.0; cumsum];

        Self {
            data,
            offsets,
            n_entities,
        }
    }

    /// Get lag observations for entity i: [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///
    /// Returns empty slice if entity has ar_order == 0
    #[inline]
    pub fn get_lags(&self, entity: usize) -> &[f64] {
        let start = self.offsets[entity];
        let end = self.offsets[entity + 1];
        &self.data[start..end]
    }

    /// Get mutable lag observations for entity i
    #[inline]
    pub fn get_lags_mut(&mut self, entity: usize) -> &mut [f64] {
        let start = self.offsets[entity];
        let end = self.offsets[entity + 1];
        &mut self.data[start..end]
    }

    /// Update lag buffer with new observation
    ///
    /// Shifts lags: [Y_{t-1}, Y_{t-2}, ...] → [Y_t, Y_{t-1}, ...]
    ///
    /// If entity has ar_order == 0, this is a no-op.
    pub fn update_lags(&mut self, entity: usize, new_observation: f64) {
        let lags = self.get_lags_mut(entity);
        if lags.is_empty() {
            return; // No lags for this entity
        }

        // Shift lags: move everything right by one position
        for i in (1..lags.len()).rev() {
            lags[i] = lags[i - 1];
        }

        // Insert new observation at front
        lags[0] = new_observation;
    }

    /// Clear all lag values (set to zero)
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Set initial lag values for an entity
    ///
    /// Used to initialize lag buffer from initial conditions.
    /// Lags should be ordered as [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    pub fn set_lags(&mut self, entity: usize, lags: &[f64]) {
        let entity_lags = self.get_lags_mut(entity);
        assert_eq!(
            entity_lags.len(),
            lags.len(),
            "Lag count mismatch for entity {}: expected {}, got {}",
            entity,
            entity_lags.len(),
            lags.len()
        );
        entity_lags.copy_from_slice(lags);
    }

    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.n_entities
    }

    /// Get total number of lags stored
    pub fn total_lags(&self) -> usize {
        self.data.len()
    }
}

/// Manager for uncertainty constraints (loads and inflows)
///
/// Replaces `ObservationSpaceConstraintManager` with unified handling.
#[derive(Debug, Clone)]
pub struct UncertaintyConstraintManager {
    /// Total number of uncertain entities (loads + inflows)
    dimension: usize,

    /// Lag buffer for all entities
    lag_buffer: UnifiedLagBuffer,

    /// Maximum lag order across all entities
    max_lag: usize,

    /// LP constraint indices (set during subproblem construction)
    constraint_indices: Option<UncertaintyConstraintIndices>,
}

impl UncertaintyConstraintManager {
    /// Create from temporal models
    pub fn from_temporal_models(models: &[TemporalModel]) -> Self {
        let dimension = models.len();
        let max_lag = models.iter().map(|m| m.max_ar_order).max().unwrap_or(0);
        let lag_buffer = UnifiedLagBuffer::from_temporal_models(models);

        Self {
            dimension,
            lag_buffer,
            max_lag,
            constraint_indices: None,
        }
    }

    /// Get lag observations for entity i
    #[inline]
    pub fn get_lag_observations(&self, entity: usize) -> &[f64] {
        self.lag_buffer.get_lags(entity)
    }

    /// Update lag buffer after LP solve
    pub fn update_lag_buffer(&mut self, entity: usize, observation: f64) {
        self.lag_buffer.update_lags(entity, observation);
    }

    /// Set initial lag values for an entity from initial conditions
    ///
    /// Initializes the lag buffer with historical observations from `recourse.json`
    /// before the first stage optimization. These values directly affect first-stage
    /// inflow realizations through the AR dynamics: Y_t = Σ ψ_j * Y_{t-j} + η_t
    ///
    /// This method should be called during SDDP handler construction (in
    /// `SddpTrainHandler::new()` and `SddpSimulationHandler::new()`).
    ///
    /// # Arguments
    ///
    /// * `entity` - Entity index (matches position in temporal_models)
    /// * `lags` - Initial lag observations [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///   Order: newest to oldest (lag 1 first, lag p last)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For AR(2) model with entity_idx=0, lags from recourse.json
    /// let lags = initial_condition.get_inflow(hydro_id); // e.g., [70.0, 65.0]
    /// uncertainty_manager.set_initial_lags(entity_idx, lags);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if lag count doesn't match entity's AR order.
    pub fn set_initial_lags(&mut self, entity: usize, lags: &[f64]) {
        self.lag_buffer.set_lags(entity, lags);
    }

    /// Set constraint indices (called during subproblem construction)
    pub fn set_constraint_indices(
        &mut self,
        indices: UncertaintyConstraintIndices,
    ) {
        self.constraint_indices = Some(indices);
    }

    /// Get constraint indices (if set)
    pub fn constraint_indices(&self) -> Option<&UncertaintyConstraintIndices> {
        self.constraint_indices.as_ref()
    }

    /// Get dimension (total number of entities)
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get maximum lag order
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    /// Clear all lag buffers
    pub fn clear_lags(&mut self) {
        self.lag_buffer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{MarginalDistribution, UncertaintyType};

    fn make_test_model(
        entity_type: UncertaintyType,
        entity_id: usize,
        max_ar_order: usize,
    ) -> TemporalModel {
        let num_seasons = 3;
        let means = vec![100.0; num_seasons];
        let stds = vec![10.0; num_seasons];
        let dists = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 10.0,
            };
            num_seasons
        ];

        if max_ar_order == 0 {
            TemporalModel::from_independent(
                entity_type,
                entity_id,
                means,
                stds,
                dists,
            )
            .unwrap()
        } else {
            let ar_orders = vec![max_ar_order; num_seasons];
            let ar_coefficients = vec![vec![0.7; max_ar_order]; num_seasons];
            TemporalModel::from_par(
                entity_type,
                entity_id,
                num_seasons,
                means,
                stds,
                dists,
                ar_orders,
                ar_coefficients,
            )
            .unwrap()
        }
    }

    #[test]
    fn test_unified_lag_buffer_creation() {
        let lag_counts = vec![2, 0, 3, 1];
        let buffer = UnifiedLagBuffer::new(&lag_counts);

        assert_eq!(buffer.num_entities(), 4);
        assert_eq!(buffer.total_lags(), 6); // 2 + 0 + 3 + 1 = 6
    }

    #[test]
    fn test_unified_lag_buffer_offsets() {
        let lag_counts = vec![2, 0, 3, 1];
        let buffer = UnifiedLagBuffer::new(&lag_counts);

        // Check offsets
        assert_eq!(buffer.offsets, vec![0, 2, 2, 5, 6]);

        // Check lag access
        assert_eq!(buffer.get_lags(0).len(), 2);
        assert_eq!(buffer.get_lags(1).len(), 0); // No lags
        assert_eq!(buffer.get_lags(2).len(), 3);
        assert_eq!(buffer.get_lags(3).len(), 1);
    }

    #[test]
    fn test_unified_lag_buffer_update() {
        let lag_counts = vec![3];
        let mut buffer = UnifiedLagBuffer::new(&lag_counts);

        // Update with sequence: 1.0, 2.0, 3.0, 4.0
        buffer.update_lags(0, 1.0);
        assert_eq!(buffer.get_lags(0), &[1.0, 0.0, 0.0]);

        buffer.update_lags(0, 2.0);
        assert_eq!(buffer.get_lags(0), &[2.0, 1.0, 0.0]);

        buffer.update_lags(0, 3.0);
        assert_eq!(buffer.get_lags(0), &[3.0, 2.0, 1.0]);

        buffer.update_lags(0, 4.0);
        assert_eq!(buffer.get_lags(0), &[4.0, 3.0, 2.0]);
    }

    #[test]
    fn test_unified_lag_buffer_no_op_for_zero_lags() {
        let lag_counts = vec![0];
        let mut buffer = UnifiedLagBuffer::new(&lag_counts);

        // Should not panic
        buffer.update_lags(0, 42.0);

        // Should still be empty
        assert_eq!(buffer.get_lags(0).len(), 0);
    }

    #[test]
    fn test_unified_lag_buffer_clear() {
        let lag_counts = vec![2, 3];
        let mut buffer = UnifiedLagBuffer::new(&lag_counts);

        // Set some values
        buffer.update_lags(0, 1.0);
        buffer.update_lags(0, 2.0);
        buffer.update_lags(1, 3.0);

        // Clear
        buffer.clear();

        // All should be zero
        assert_eq!(buffer.get_lags(0), &[0.0, 0.0]);
        assert_eq!(buffer.get_lags(1), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unified_lag_buffer_from_temporal_models() {
        let models = vec![
            make_test_model(UncertaintyType::Load, 0, 2),
            make_test_model(UncertaintyType::Load, 1, 0),
            make_test_model(UncertaintyType::Inflow, 0, 1),
        ];

        let buffer = UnifiedLagBuffer::from_temporal_models(&models);

        assert_eq!(buffer.num_entities(), 3);
        assert_eq!(buffer.total_lags(), 3); // 2 + 0 + 1
        assert_eq!(buffer.get_lags(0).len(), 2);
        assert_eq!(buffer.get_lags(1).len(), 0);
        assert_eq!(buffer.get_lags(2).len(), 1);
    }

    #[test]
    fn test_uncertainty_constraint_manager_creation() {
        let models = vec![
            make_test_model(UncertaintyType::Load, 0, 1),
            make_test_model(UncertaintyType::Inflow, 0, 2),
        ];

        let manager =
            UncertaintyConstraintManager::from_temporal_models(&models);

        assert_eq!(manager.dimension(), 2);
        assert_eq!(manager.max_lag(), 2);
        assert!(manager.constraint_indices().is_none());
    }

    #[test]
    fn test_uncertainty_constraint_manager_lag_operations() {
        let models = vec![
            make_test_model(UncertaintyType::Load, 0, 2),
            make_test_model(UncertaintyType::Inflow, 0, 1),
        ];

        let mut manager =
            UncertaintyConstraintManager::from_temporal_models(&models);

        // Update lags
        manager.update_lag_buffer(0, 10.0);
        manager.update_lag_buffer(1, 20.0);

        // Check retrieval
        assert_eq!(manager.get_lag_observations(0), &[10.0, 0.0]);
        assert_eq!(manager.get_lag_observations(1), &[20.0]);

        // Clear
        manager.clear_lags();
        assert_eq!(manager.get_lag_observations(0), &[0.0, 0.0]);
        assert_eq!(manager.get_lag_observations(1), &[0.0]);
    }

    #[test]
    fn test_uncertainty_constraint_manager_set_indices() {
        let models = vec![make_test_model(UncertaintyType::Load, 0, 0)];

        let mut manager =
            UncertaintyConstraintManager::from_temporal_models(&models);

        let indices = UncertaintyConstraintIndices {
            observation_constraints: vec![10, 20, 30],
        };

        manager.set_constraint_indices(indices);

        assert!(manager.constraint_indices().is_some());
        assert_eq!(
            manager
                .constraint_indices()
                .unwrap()
                .observation_constraints,
            vec![10, 20, 30]
        );
    }

    #[test]
    fn test_large_system_with_mixed_orders() {
        // Simulate a large system with varying AR orders
        let models: Vec<TemporalModel> = (0..10)
            .map(|i| {
                let ar_order = i % 4; // 0, 1, 2, 3, 0, 1, 2, 3, 0, 1
                make_test_model(UncertaintyType::Load, i, ar_order)
            })
            .collect();

        let manager =
            UncertaintyConstraintManager::from_temporal_models(&models);

        assert_eq!(manager.dimension(), 10);
        assert_eq!(manager.max_lag(), 3);

        // Total lags: 0+1+2+3+0+1+2+3+0+1 = 13
        assert_eq!(manager.lag_buffer.total_lags(), 13);
    }
}
