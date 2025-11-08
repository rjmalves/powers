/// Initial condition for SDDP algorithm.
///
/// # Storage Initialization
///
/// Initial storage values are used to set the starting reservoir levels in the
/// first pre-study node. These values become the boundary conditions
/// for the first-stage optimization problem.
///
/// # Lagged Inflows Format
///
/// For PAR(p) models, `inflow[hydro_id][lag_idx]` specifies historical inflows where:
/// - `lag_idx = 0` corresponds to Y_{t-1} (most recent lag, 1 stage ago)
/// - `lag_idx = 1` corresponds to Y_{t-2} (2 stages ago)
/// - `lag_idx = p-1` corresponds to Y_{t-p} (oldest lag, p stages ago)
///
pub struct InitialCondition {
    storage: Vec<f64>,
    inflow: Vec<Vec<f64>>,
    /// Deprecated: Pre-study now uses single node with automatic season computation
    #[allow(dead_code)]
    season_ids: Option<Vec<usize>>,
}

impl InitialCondition {
    /// Create initial condition with automatic PreStudy season computation
    pub fn new(storage: Vec<f64>, inflow: Vec<Vec<f64>>) -> Self {
        Self {
            storage,
            inflow,
            season_ids: None,
        }
    }

    /// Create initial condition with explicit PreStudy season IDs
    pub fn with_seasons(
        storage: Vec<f64>,
        inflow: Vec<Vec<f64>>,
        season_ids: Vec<usize>,
    ) -> Self {
        Self {
            storage,
            inflow,
            season_ids: Some(season_ids),
        }
    }

    pub fn get_storage(&self) -> &[f64] {
        &self.storage
    }

    /// Get season ID for a specific PreStudy node
    pub fn get_season_id(&self, prestudy_node_idx: usize) -> Option<usize> {
        self.season_ids
            .as_ref()
            .and_then(|ids| ids.get(prestudy_node_idx).copied())
    }

    /// Get lagged inflows for a specific hydro.
    ///
    /// Returns slice where index 0 = Y_{-1}, index 1 = Y_{-2}, etc.
    /// Returns empty slice if no lags exist for this hydro.
    pub fn get_inflow(&self, hydro_id: usize) -> &[f64] {
        self.inflow
            .get(hydro_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn get_lagged_inflows(&self) -> &[Vec<f64>] {
        &self.inflow
    }

    pub fn lag_count(&self) -> usize {
        self.inflow.iter().map(|v| v.len()).max().unwrap_or(0)
    }

    pub fn has_lags(&self) -> bool {
        !self.inflow.is_empty() && self.inflow.iter().any(|v| !v.is_empty())
    }

    /// Flattens the initial state into a coefficient vector matching State representation.
    ///
    /// Returns `[storage_0, ..., storage_n, lag_0_1, lag_0_2, ..., lag_n_p]`
    /// matching the layout used by `StorageAndInflowState::coefficients()`.
    ///
    /// # Layout
    ///
    /// For each hydro i with AR order p_i:
    /// - storage_i (1 value)
    /// - lag_i_1, lag_i_2, ..., lag_i_{p_i} (p_i values, may be 0)
    ///
    /// # Example
    ///
    /// 3 hydros: AR(0), AR(1), AR(2):
    /// ```text
    /// [V₀, V₁, Y₁⁽¹⁾, V₂, Y₂⁽¹⁾, Y₂⁽²⁾]
    /// ```
    pub fn flatten_state(&self) -> Vec<f64> {
        let mut state = Vec::with_capacity(
            self.storage.len()
                + self.inflow.iter().map(|v| v.len()).sum::<usize>(),
        );

        for (hydro_id, &storage_val) in self.storage.iter().enumerate() {
            state.push(storage_val);

            // Add lags for this hydro (if any)
            if let Some(lags) = self.inflow.get(hydro_id) {
                state.extend_from_slice(lags);
            }
        }

        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_storage_only() {
        let storage = vec![100.0, 200.0];
        let ic = InitialCondition::new(storage.clone(), vec![]);

        assert_eq!(ic.get_storage(), &storage[..]);
        assert_eq!(ic.lag_count(), 0);
        assert!(!ic.has_lags());
        assert_eq!(ic.get_lagged_inflows().len(), 0);
    }

    #[test]
    fn test_new_with_lagged_inflows() {
        let storage = vec![100.0, 200.0];
        let inflow = vec![
            vec![10.0, 9.0],  // hydro 0: [Y_{-1}, Y_{-2}]
            vec![20.0, 18.0], // hydro 1: [Y_{-1}, Y_{-2}]
        ];
        let ic = InitialCondition::new(storage.clone(), inflow.clone());

        assert_eq!(ic.get_storage(), &storage[..]);
        assert_eq!(ic.get_inflow(0), &inflow[0][..]);
        assert_eq!(ic.get_inflow(1), &inflow[1][..]);
        assert_eq!(ic.lag_count(), 2);
        assert!(ic.has_lags());
        assert_eq!(ic.get_lagged_inflows(), &inflow[..]);
    }

    #[test]
    fn test_lag_count_returns_max() {
        // Different lag counts per hydro (PAR models can vary)
        let storage = vec![100.0, 200.0, 300.0];
        let inflow = vec![
            vec![10.0, 9.0, 8.0], // hydro 0: PAR(3)
            vec![20.0, 18.0],     // hydro 1: PAR(2)
            vec![30.0],           // hydro 2: PAR(1)
        ];
        let ic = InitialCondition::new(storage, inflow);

        assert_eq!(ic.lag_count(), 3, "Should return max lag count");
    }

    #[test]
    fn test_has_lags_true_when_lags_exist() {
        let storage = vec![100.0];
        let inflow = vec![vec![10.0]];
        let ic = InitialCondition::new(storage, inflow);

        assert!(ic.has_lags());
    }

    #[test]
    fn test_has_lags_false_when_empty() {
        let storage = vec![100.0];
        let ic = InitialCondition::new(storage, vec![]);

        assert!(!ic.has_lags());
    }

    #[test]
    fn test_has_lags_false_when_all_empty_vecs() {
        let storage = vec![100.0, 200.0];
        let inflow = vec![vec![], vec![]];
        let ic = InitialCondition::new(storage, inflow);

        assert!(!ic.has_lags(), "Empty lag vectors should count as no lags");
    }

    #[test]
    fn test_get_inflow_returns_empty_for_invalid_id() {
        let storage = vec![100.0];
        let inflow = vec![vec![10.0]];
        let ic = InitialCondition::new(storage, inflow);

        let empty: &[f64] = &[];
        assert_eq!(
            ic.get_inflow(999),
            empty,
            "Should return empty for invalid hydro_id"
        );
    }

    #[test]
    fn test_get_inflow_returns_empty_when_no_lags() {
        let storage = vec![100.0, 200.0];
        let ic = InitialCondition::new(storage, vec![]);

        let empty: &[f64] = &[];
        assert_eq!(ic.get_inflow(0), empty);
        assert_eq!(ic.get_inflow(1), empty);
    }

    #[test]
    fn test_get_lagged_inflows_full_structure() {
        let storage = vec![50.0, 60.0];
        let inflow = vec![
            vec![100.0, 95.0, 90.0],   // PAR(3) for hydro 0
            vec![120.0, 115.0, 110.0], // PAR(3) for hydro 1
        ];
        let ic = InitialCondition::new(storage, inflow.clone());

        let lags = ic.get_lagged_inflows();
        assert_eq!(lags.len(), 2);
        assert_eq!(lags[0], inflow[0]);
        assert_eq!(lags[1], inflow[1]);
    }

    #[test]
    fn test_backward_compatibility() {
        // Existing code pattern should still work
        let storage = vec![100.0, 200.0];
        let inflow = vec![vec![10.0], vec![20.0]];
        let ic = InitialCondition::new(storage.clone(), inflow.clone());

        assert_eq!(ic.get_storage(), &storage[..]);
        assert_eq!(ic.get_inflow(0), &inflow[0][..]);
        assert_eq!(ic.get_inflow(1), &inflow[1][..]);
    }

    // ========================================================================
    // TICKET-003b: PreStudy Season Handling Tests
    // ========================================================================

    #[test]
    fn test_with_seasons_constructor() {
        let storage = vec![50.0, 60.0];
        let inflow = vec![vec![100.0, 95.0], vec![120.0, 115.0]];
        let season_ids = vec![6, 5, 4]; // Newest to oldest

        let ic = InitialCondition::with_seasons(
            storage.clone(),
            inflow.clone(),
            season_ids.clone(),
        );

        assert_eq!(ic.get_storage(), &storage[..]);
        assert_eq!(ic.get_inflow(0), &inflow[0][..]);
        assert_eq!(ic.get_inflow(1), &inflow[1][..]);

        // Check season_ids: [newest=6, 5, oldest=4]
        assert_eq!(ic.get_season_id(0), Some(6)); // Newest
        assert_eq!(ic.get_season_id(1), Some(5));
        assert_eq!(ic.get_season_id(2), Some(4)); // Oldest
    }

    #[test]
    fn test_get_season_id_with_explicit_seasons() {
        let ic = InitialCondition::with_seasons(
            vec![50.0],
            vec![vec![100.0, 95.0]],
            vec![0, 11, 10], // Wraparound: newest=0, 11, oldest=10
        );

        // Verify all season IDs are accessible
        assert_eq!(ic.get_season_id(0), Some(0)); // Newest PreStudy
        assert_eq!(ic.get_season_id(1), Some(11));
        assert_eq!(ic.get_season_id(2), Some(10)); // Oldest PreStudy
    }

    #[test]
    fn test_get_season_id_returns_none_for_automatic() {
        // When using new(), season_ids should be None
        let ic = InitialCondition::new(vec![50.0], vec![vec![100.0]]);

        // All indices should return None (use automatic cycle-back)
        assert_eq!(ic.get_season_id(0), None);
        assert_eq!(ic.get_season_id(1), None);
        assert_eq!(ic.get_season_id(100), None); // Out of bounds also None
    }

    #[test]
    fn test_get_season_id_out_of_bounds() {
        let ic = InitialCondition::with_seasons(
            vec![50.0],
            vec![vec![100.0]],
            vec![6, 5], // Newest to oldest
        );

        // Valid indices
        assert_eq!(ic.get_season_id(0), Some(6)); // Newest
        assert_eq!(ic.get_season_id(1), Some(5)); // Oldest

        // Out of bounds should return None
        assert_eq!(ic.get_season_id(2), None);
        assert_eq!(ic.get_season_id(100), None);
    }

    #[test]
    fn test_new_with_seasons_backward_compatible() {
        // Ensure new() still works as before (backward compatibility)
        let ic1 = InitialCondition::new(vec![50.0], vec![vec![100.0]]);
        let ic2 = InitialCondition::new(vec![50.0], vec![vec![100.0]]);

        // Both should have no explicit seasons
        assert_eq!(ic1.get_season_id(0), None);
        assert_eq!(ic2.get_season_id(0), None);
    }

    #[test]
    fn test_with_seasons_empty_season_ids() {
        // Edge case: empty season_ids vector
        let ic = InitialCondition::with_seasons(
            vec![50.0],
            vec![vec![100.0]],
            vec![],
        );

        // Should return None for any index
        assert_eq!(ic.get_season_id(0), None);
    }

    #[test]
    fn test_with_seasons_single_prestudy_node() {
        // lag_order=0 case: single PreStudy node
        let ic = InitialCondition::with_seasons(
            vec![50.0],
            vec![vec![100.0]],
            vec![5],
        );

        assert_eq!(ic.get_season_id(0), Some(5));
        assert_eq!(ic.get_season_id(1), None);
    }
}
