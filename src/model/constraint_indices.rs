//! Constraint index ranges for efficient dual extraction.
//!
//! This module provides [`ConstraintIndices`] which precomputes index ranges
//! from the existing `Constraints` struct for O(1) access during dual extraction.

use std::ops::Range;

use crate::subproblem::Constraints;

/// Precomputed constraint index ranges for efficient dual extraction.
///
/// Used to extract marginal costs, water values, and lag duals from LP solutions.
///
/// # Design for SoA Migration
///
/// The range-based API enables future SoA layouts where duals are extracted
/// directly into contiguous arrays rather than per-realization structs.
#[derive(Clone, Debug)]
pub struct ConstraintIndices {
    /// Load balance constraint range (for marginal costs)
    load_balance: Range<usize>,
    /// Hydro balance constraint range (for water values)
    hydro_balance: Range<usize>,
    /// Load lag constraint indices by bus_id: `load_lag[bus_id][lag_idx]`
    load_lag: Option<Vec<Vec<usize>>>,
    /// Inflow lag constraint indices by hydro_id: `inflow_lag[hydro_id][lag_idx]`
    inflow_lag: Option<Vec<Vec<usize>>>,
}

impl ConstraintIndices {
    /// Create from existing Constraints struct.
    pub fn from_constraints(cons: &Constraints) -> Self {
        let load_balance = Self::vec_to_range(&cons.load_balance);
        let hydro_balance = Self::vec_to_range(&cons.hydro_balance);

        let load_lag = cons
            .load_lag_constraints
            .as_ref()
            .map(|lc| lc.constraints_by_bus.clone());

        let inflow_lag = cons
            .inflow_lag_constraints
            .as_ref()
            .map(|ic| ic.constraints_by_hydro.clone());

        Self {
            load_balance,
            hydro_balance,
            load_lag,
            inflow_lag,
        }
    }

    /// Convert contiguous `Vec<usize>` to `Range<usize>`.
    fn vec_to_range(indices: &[usize]) -> Range<usize> {
        let first = *indices.first().expect("indices must not be empty");
        let last = *indices.last().expect("indices must not be empty");
        first..last + 1
    }

    /// Get load balance constraint range (for marginal cost extraction).
    #[inline]
    pub fn load_balance_range(&self) -> Range<usize> {
        self.load_balance.clone()
    }

    /// Get hydro balance constraint range (for water value extraction).
    #[inline]
    pub fn hydro_balance_range(&self) -> Range<usize> {
        self.hydro_balance.clone()
    }

    /// Get lag constraints for a specific bus.
    ///
    /// Returns an empty slice if the bus has no AR dynamics or load lags don't exist.
    #[inline]
    pub fn load_lag_constraints(&self, bus_id: usize) -> &[usize] {
        self.load_lag
            .as_ref()
            .and_then(|lags| lags.get(bus_id))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get lag constraints for a specific hydro.
    ///
    /// Returns an empty slice if the hydro has no AR dynamics or inflow lags don't exist.
    #[inline]
    pub fn inflow_lag_constraints(&self, hydro_id: usize) -> &[usize] {
        self.inflow_lag
            .as_ref()
            .and_then(|lags| lags.get(hydro_id))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check if load lag constraints exist.
    #[inline]
    pub fn has_load_lags(&self) -> bool {
        self.load_lag.is_some()
    }

    /// Check if inflow lag constraints exist.
    #[inline]
    pub fn has_inflow_lags(&self) -> bool {
        self.inflow_lag.is_some()
    }

    /// Get number of buses with load lag constraints.
    pub fn num_buses(&self) -> usize {
        self.load_lag.as_ref().map(|v| v.len()).unwrap_or(0)
    }

    /// Get number of hydros with inflow lag constraints.
    pub fn num_hydros(&self) -> usize {
        self.inflow_lag.as_ref().map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_range() {
        let indices = vec![0, 1, 2, 3];
        let range = ConstraintIndices::vec_to_range(&indices);
        assert_eq!(range, 0..4);
    }

    #[test]
    fn test_load_lag_constraints_missing() {
        let indices = ConstraintIndices {
            load_balance: 0..5,
            hydro_balance: 5..10,
            load_lag: None,
            inflow_lag: None,
        };
        assert!(indices.load_lag_constraints(0).is_empty());
        assert!(!indices.has_load_lags());
    }

    #[test]
    fn test_load_lag_constraints_present() {
        let indices = ConstraintIndices {
            load_balance: 0..5,
            hydro_balance: 5..10,
            load_lag: Some(vec![vec![10, 11], vec![], vec![12]]),
            inflow_lag: None,
        };
        assert_eq!(indices.load_lag_constraints(0), &[10, 11]);
        assert!(indices.load_lag_constraints(1).is_empty());
        assert_eq!(indices.load_lag_constraints(2), &[12]);
        assert!(indices.has_load_lags());
        assert_eq!(indices.num_buses(), 3);
    }

    #[test]
    fn test_inflow_lag_constraints() {
        let indices = ConstraintIndices {
            load_balance: 0..5,
            hydro_balance: 5..10,
            load_lag: None,
            inflow_lag: Some(vec![vec![20], vec![21, 22]]),
        };
        assert_eq!(indices.inflow_lag_constraints(0), &[20]);
        assert_eq!(indices.inflow_lag_constraints(1), &[21, 22]);
        assert!(indices.has_inflow_lags());
        assert_eq!(indices.num_hydros(), 2);
    }
}
