//! Variable index ranges for efficient solution extraction.
//!
//! This module provides [`VariableIndices`] which precomputes index ranges
//! from the existing `Variables` struct for O(1) access during hot-path extraction.

use std::ops::Range;

use crate::subproblem::Variables;

/// Precomputed variable index ranges for efficient solution extraction.
///
/// These ranges are computed once during subproblem construction and
/// used repeatedly during SDDP forward/backward passes.
///
/// # Design for SoA Migration
///
/// The range-based API enables future SoA layouts where data is extracted
/// directly into contiguous arrays rather than per-realization structs.
///
/// # Example
///
/// ```ignore
/// let indices = VariableIndices::from_variables(&subproblem.variables);
/// let range = indices.deficit_range();
/// target.copy_from_slice(&solution.colvalue[range]);
/// ```
#[derive(Clone, Debug)]
pub struct VariableIndices {
    /// Deficit variable range in LP solution
    deficit: Range<usize>,
    /// Direct exchange variable range (None if no exchanges)
    direct_exchange: Option<Range<usize>>,
    /// Reverse exchange variable range (None if no exchanges)
    reverse_exchange: Option<Range<usize>>,
    /// Thermal generation variable range (None if no thermals)
    thermal_gen: Option<Range<usize>>,
    /// Turbined flow variable range
    turbined_flow: Range<usize>,
    /// Spillage variable range
    spillage: Range<usize>,
    /// Stored volume (final storage) variable range
    stored_volume: Range<usize>,
    /// Load observation variable indices (may not be contiguous)
    load: Vec<usize>,
    /// Inflow observation variable indices (may not be contiguous)
    inflow: Vec<usize>,
}

impl VariableIndices {
    /// Create from existing Variables struct.
    ///
    /// This is the bridge between old and new code.
    pub fn from_variables(vars: &Variables) -> Self {
        Self {
            deficit: Self::vec_to_range(&vars.deficit),
            direct_exchange: Self::vec_to_optional_range(&vars.direct_exchange),
            reverse_exchange: Self::vec_to_optional_range(
                &vars.reverse_exchange,
            ),
            thermal_gen: Self::vec_to_optional_range(&vars.thermal_gen),
            turbined_flow: Self::vec_to_range(&vars.turbined_flow),
            spillage: Self::vec_to_range(&vars.spillage),
            stored_volume: Self::vec_to_range(&vars.stored_volume),
            load: vars.load.clone(),
            inflow: vars.inflow.clone(),
        }
    }

    /// Convert contiguous `Vec<usize>` to `Range<usize>`.
    ///
    /// Assumes indices are contiguous. The range is `[first, last + 1)`.
    fn vec_to_range(indices: &[usize]) -> Range<usize> {
        let first = *indices.first().expect("indices must not be empty");
        let last = *indices.last().expect("indices must not be empty");
        first..last + 1
    }

    /// Convert optional `Vec<usize>` to `Option<Range<usize>>`.
    ///
    /// Returns `None` if the input is empty.
    fn vec_to_optional_range(indices: &[usize]) -> Option<Range<usize>> {
        if indices.is_empty() {
            None
        } else {
            Some(Self::vec_to_range(indices))
        }
    }

    /// Get deficit variable range.
    #[inline]
    pub fn deficit_range(&self) -> Range<usize> {
        self.deficit.clone()
    }

    /// Get direct exchange variable range, if exchanges exist.
    #[inline]
    pub fn direct_exchange_range(&self) -> Option<Range<usize>> {
        self.direct_exchange.clone()
    }

    /// Get reverse exchange variable range, if exchanges exist.
    #[inline]
    pub fn reverse_exchange_range(&self) -> Option<Range<usize>> {
        self.reverse_exchange.clone()
    }

    /// Get thermal generation variable range, if thermals exist.
    #[inline]
    pub fn thermal_gen_range(&self) -> Option<Range<usize>> {
        self.thermal_gen.clone()
    }

    /// Get turbined flow variable range.
    #[inline]
    pub fn turbined_flow_range(&self) -> Range<usize> {
        self.turbined_flow.clone()
    }

    /// Get spillage variable range.
    #[inline]
    pub fn spillage_range(&self) -> Range<usize> {
        self.spillage.clone()
    }

    /// Get stored volume (final storage) variable range.
    #[inline]
    pub fn stored_volume_range(&self) -> Range<usize> {
        self.stored_volume.clone()
    }

    /// Get load observation variable indices (non-contiguous).
    #[inline]
    pub fn load_indices(&self) -> &[usize] {
        &self.load
    }

    /// Get inflow observation variable indices (non-contiguous).
    #[inline]
    pub fn inflow_indices(&self) -> &[usize] {
        &self.inflow
    }

    /// Check if exchange variables exist.
    #[inline]
    pub fn has_exchange(&self) -> bool {
        self.direct_exchange.is_some()
    }

    /// Check if thermal generation variables exist.
    #[inline]
    pub fn has_thermal(&self) -> bool {
        self.thermal_gen.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_to_range_normal() {
        let indices = vec![5, 6, 7, 8, 9];
        let range = VariableIndices::vec_to_range(&indices);
        assert_eq!(range, 5..10);
    }

    #[test]
    fn test_vec_to_range_single() {
        let indices = vec![42];
        let range = VariableIndices::vec_to_range(&indices);
        assert_eq!(range, 42..43);
    }

    #[test]
    fn test_optional_range_empty() {
        let indices: Vec<usize> = vec![];
        let range = VariableIndices::vec_to_optional_range(&indices);
        assert!(range.is_none());
    }

    #[test]
    fn test_optional_range_present() {
        let indices = vec![10, 11, 12];
        let range = VariableIndices::vec_to_optional_range(&indices);
        assert_eq!(range, Some(10..13));
    }
}
