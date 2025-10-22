/// Initial condition for SDDP algorithm.
///
/// Specifies:
/// - Initial storage values for each hydro
/// - Historical inflow lags for PAR model initialization
///
/// # Lagged Inflows Format
///
/// `inflow[hydro_id][lag_idx]` where:
/// - `lag_idx = 0` corresponds to Y_{-1} (most recent lag, 1 stage ago)
/// - `lag_idx = 1` corresponds to Y_{-2} (2 stages ago)
/// - `lag_idx = p-1` corresponds to Y_{-p} (oldest lag, p stages ago)
///
/// For storage-only states, `inflow` can be empty.
///
/// # Example
///
/// ```rust,ignore
/// // Storage-only (backward compatible)
/// let ic = InitialCondition::new(vec![50.0, 60.0], vec![]);
///
/// // PAR(2) with 2 hydros
/// let ic = InitialCondition::new(
///     vec![50.0, 60.0],  // storage
///     vec![
///         vec![100.0, 95.0],  // hydro 0: [Y_{-1}, Y_{-2}]
///         vec![120.0, 115.0], // hydro 1: [Y_{-1}, Y_{-2}]
///     ],
/// );
/// ```
pub struct InitialCondition {
    storage: Vec<f64>,
    /// Historical inflow lags: inflow[hydro_id][lag_idx]
    /// where lag_idx=0 is Y_{-1}, lag_idx=1 is Y_{-2}, etc.
    inflow: Vec<Vec<f64>>,
}

impl InitialCondition {
    /// Create initial condition with storage and optional lagged inflows.
    ///
    /// # Arguments
    ///
    /// * `storage` - Initial storage for each hydro (MWh)
    /// * `inflow` - Historical inflows for PAR initialization.
    ///
    pub fn new(storage: Vec<f64>, inflow: Vec<Vec<f64>>) -> Self {
        Self { storage, inflow }
    }

    /// Get initial storage values.
    ///
    /// Returns slice of storage values indexed by hydro_id.
    pub fn get_storage(&self) -> &[f64] {
        &self.storage
    }

    /// Get lagged inflows for a specific hydro.
    ///
    /// Returns slice where index 0 = Y_{-1}, index 1 = Y_{-2}, etc.
    /// Returns empty slice if no lags exist for this hydro.
    ///
    /// # Arguments
    ///
    /// * `hydro_id` - Index of the hydro
    pub fn get_inflow(&self, hydro_id: usize) -> &[f64] {
        self.inflow
            .get(hydro_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all lagged inflows.
    ///
    /// Returns reference to the full inflow structure:
    /// `inflow[hydro_id][lag_idx]`
    pub fn get_lagged_inflows(&self) -> &[Vec<f64>] {
        &self.inflow
    }

    /// Get the maximum lag count across all hydros.
    ///
    /// For PAR(p), this returns p (the AR order).
    /// Returns 0 for storage-only states.
    pub fn lag_count(&self) -> usize {
        self.inflow.iter().map(|v| v.len()).max().unwrap_or(0)
    }

    /// Check if this initial condition has any lagged inflows.
    ///
    /// Returns `true` if at least one hydro has historical inflows.
    pub fn has_lags(&self) -> bool {
        !self.inflow.is_empty() && self.inflow.iter().any(|v| !v.is_empty())
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
}
