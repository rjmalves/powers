//! Solution extraction from LP solver results.
//!
//! This module provides [`SolutionExtractor`] which extracts LP solution values
//! into domain types (primarily [`Realization`]).
//!
//! # Dual API Pattern
//!
//! Each extraction method has two variants:
//! - `extract_X_into(&self, solution, target_slice)` - Low-level, works with raw slices
//! - `extract_X(&self, solution, realization)` - High-level, works with [`Realization`]
//!
//! This enables future SoA migration while preserving current functionality.
//!
//! # Example
//!
//! ```ignore
//! let extractor = SolutionExtractor::new(var_indices, con_indices);
//!
//! // Low-level API (SoA-ready)
//! extractor.extract_deficit_into(&solution, &mut deficit_buffer);
//!
//! // High-level API (current pattern)
//! extractor.extract_deficit(&solution, &mut realization);
//! ```

use crate::model::{ConstraintIndices, VariableIndices};
use crate::solver::Solution;
use crate::subproblem::Realization;

/// Extracts LP solution values into domain types.
///
/// This struct encapsulates all solution extraction logic, providing both
/// low-level slice-based APIs (for future SoA layouts) and high-level
/// Realization-based APIs (for current usage).
///
/// # Design Principles
///
/// 1. **Zero allocation**: All extraction uses preallocated buffers
/// 2. **Dual API**: `extract_X_into()` for slices, `extract_X()` for Realization
/// 3. **Inlined hot paths**: All extraction methods are `#[inline]`
/// 4. **Single responsibility**: Only extraction logic, no constraint building
#[derive(Clone, Debug)]
pub struct SolutionExtractor {
    var_indices: VariableIndices,
    con_indices: ConstraintIndices,
}

impl SolutionExtractor {
    /// Create a new `SolutionExtractor` from index structs.
    #[must_use]
    pub fn new(
        var_indices: VariableIndices,
        con_indices: ConstraintIndices,
    ) -> Self {
        Self {
            var_indices,
            con_indices,
        }
    }

    /// Create from existing Variables and Constraints.
    ///
    /// This is the bridge for migrating existing code.
    #[must_use]
    pub fn from_subproblem_types(
        variables: &crate::subproblem::Variables,
        constraints: &crate::subproblem::Constraints,
    ) -> Self {
        Self::new(
            VariableIndices::from_variables(variables),
            ConstraintIndices::from_constraints(constraints),
        )
    }

    // =========================================================================
    // Primal Variable Extraction (from solution.colvalue)
    // =========================================================================

    /// Extract deficit values into target slice.
    #[inline]
    pub fn extract_deficit_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.var_indices.deficit_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }

    /// Extract deficit values into Realization.
    #[inline]
    pub fn extract_deficit(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_deficit_into(solution, &mut realization.deficit);
    }

    /// Extract net exchange values (direct - reverse) into target slice.
    #[inline]
    pub fn extract_exchange_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        if let (Some(direct_range), Some(reverse_range)) = (
            self.var_indices.direct_exchange_range(),
            self.var_indices.reverse_exchange_range(),
        ) {
            target.copy_from_slice(&solution.colvalue[direct_range.clone()]);
            target
                .iter_mut()
                .zip(&solution.colvalue[reverse_range])
                .for_each(|(direct, reverse)| *direct -= *reverse);
        }
    }

    /// Extract net exchange values into Realization.
    #[inline]
    pub fn extract_exchange(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_exchange_into(solution, &mut realization.exchange);
    }

    /// Extract thermal generation values into target slice.
    #[inline]
    pub fn extract_thermal_gen_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        if let Some(range) = self.var_indices.thermal_gen_range() {
            target.copy_from_slice(&solution.colvalue[range]);
        }
    }

    /// Extract thermal generation values into Realization.
    #[inline]
    pub fn extract_thermal_gen(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_thermal_gen_into(
            solution,
            &mut realization.thermal_generation,
        );
    }

    /// Extract spillage values into target slice.
    #[inline]
    pub fn extract_spillage_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.var_indices.spillage_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }

    /// Extract spillage values into Realization.
    #[inline]
    pub fn extract_spillage(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_spillage_into(solution, &mut realization.spillage);
    }

    /// Extract turbined flow values into target slice.
    #[inline]
    pub fn extract_turbined_flow_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.var_indices.turbined_flow_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }

    /// Extract turbined flow values into Realization.
    #[inline]
    pub fn extract_turbined_flow(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_turbined_flow_into(
            solution,
            &mut realization.turbined_flow,
        );
    }

    /// Extract final storage values into target slice.
    #[inline]
    pub fn extract_final_storage_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.var_indices.stored_volume_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }

    /// Extract final storage values into Realization.
    #[inline]
    pub fn extract_final_storage(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_final_storage_into(
            solution,
            &mut realization.final_storage,
        );
    }

    /// Extract load observation values into target slice (non-contiguous).
    #[inline]
    pub fn extract_load_into(&self, solution: &Solution, target: &mut [f64]) {
        for (i, &var_idx) in self.var_indices.load_indices().iter().enumerate()
        {
            target[i] = solution.colvalue[var_idx];
        }
    }

    /// Extract load observation values into Realization.
    #[inline]
    pub fn extract_load(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_load_into(solution, &mut realization.loads);
    }

    /// Extract inflow observation values into target slice (non-contiguous).
    #[inline]
    pub fn extract_inflow_into(&self, solution: &Solution, target: &mut [f64]) {
        for (h, &var_idx) in
            self.var_indices.inflow_indices().iter().enumerate()
        {
            target[h] = solution.colvalue[var_idx];
        }
    }

    /// Extract inflow observation values into Realization.
    #[inline]
    pub fn extract_inflow(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_inflow_into(solution, &mut realization.inflow);
    }

    // =========================================================================
    // Dual Variable Extraction (from solution.rowdual)
    // =========================================================================

    /// Extract water values (hydro balance duals) into target slice.
    #[inline]
    pub fn extract_water_values_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.con_indices.hydro_balance_range();
        target.copy_from_slice(&solution.rowdual[range]);
    }

    /// Extract water values into Realization.
    #[inline]
    pub fn extract_water_values(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_water_values_into(solution, &mut realization.water_value);
    }

    /// Extract marginal costs (load balance duals) into target slice.
    #[inline]
    pub fn extract_marginal_costs_into(
        &self,
        solution: &Solution,
        target: &mut [f64],
    ) {
        let range = self.con_indices.load_balance_range();
        target.copy_from_slice(&solution.rowdual[range]);
    }

    /// Extract marginal costs into Realization.
    #[inline]
    pub fn extract_marginal_costs(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_marginal_costs_into(
            solution,
            &mut realization.marginal_cost,
        );
    }

    /// Extract lag duals into Realization.
    ///
    /// This method has complex structure (nested vecs) and allocates.
    /// Future optimization could use preallocated nested structure.
    pub fn extract_lag_duals(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        // Clear existing lag duals
        realization.load_lag_duals.clear();
        realization.inflow_lag_duals.clear();

        // Extract load lag duals by bus_id
        if self.con_indices.has_load_lags() {
            let buses_count = self.con_indices.num_buses();
            realization.load_lag_duals.resize(buses_count, Vec::new());

            for bus_id in 0..buses_count {
                let constraints = self.con_indices.load_lag_constraints(bus_id);
                realization.load_lag_duals[bus_id] = constraints
                    .iter()
                    .map(|&idx| solution.rowdual[idx])
                    .collect();
            }
        }

        // Extract inflow lag duals by hydro_id
        if self.con_indices.has_inflow_lags() {
            let hydros_count = self.con_indices.num_hydros();
            realization
                .inflow_lag_duals
                .resize(hydros_count, Vec::new());

            for hydro_id in 0..hydros_count {
                let constraints =
                    self.con_indices.inflow_lag_constraints(hydro_id);
                realization.inflow_lag_duals[hydro_id] = constraints
                    .iter()
                    .map(|&idx| solution.rowdual[idx])
                    .collect();
            }
        }
    }

    // =========================================================================
    // Convenience Methods
    // =========================================================================

    /// Check if exchange variables exist.
    #[inline]
    #[must_use]
    pub fn has_exchange(&self) -> bool {
        self.var_indices.has_exchange()
    }

    /// Check if thermal generation variables exist.
    #[inline]
    #[must_use]
    pub fn has_thermal(&self) -> bool {
        self.var_indices.has_thermal()
    }

    /// Extract all primal variables into realization.
    ///
    /// This replaces multiple individual calls in the hot path.
    pub fn extract_all_primals(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_deficit(solution, realization);
        if self.has_exchange() {
            self.extract_exchange(solution, realization);
        }
        if self.has_thermal() {
            self.extract_thermal_gen(solution, realization);
        }
        self.extract_spillage(solution, realization);
        self.extract_turbined_flow(solution, realization);
        self.extract_final_storage(solution, realization);
        self.extract_load(solution, realization);
        self.extract_inflow(solution, realization);
    }

    /// Extract all dual variables into realization.
    pub fn extract_all_duals(
        &self,
        solution: &Solution,
        realization: &mut Realization,
    ) {
        self.extract_water_values(solution, realization);
        self.extract_marginal_costs(solution, realization);
        self.extract_lag_duals(solution, realization);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require actual Variables/Constraints/Solution
    // which are complex to construct. Unit tests focus on the extractor creation.

    #[test]
    fn test_has_exchange_and_thermal() {
        // Create minimal indices for testing convenience methods
        let var_indices =
            VariableIndices::from_variables(&create_test_variables(true, true));
        let con_indices =
            ConstraintIndices::from_constraints(&create_test_constraints());

        let extractor = SolutionExtractor::new(var_indices, con_indices);
        assert!(extractor.has_exchange());
        assert!(extractor.has_thermal());
    }

    #[test]
    fn test_no_exchange_no_thermal() {
        let var_indices = VariableIndices::from_variables(
            &create_test_variables(false, false),
        );
        let con_indices =
            ConstraintIndices::from_constraints(&create_test_constraints());

        let extractor = SolutionExtractor::new(var_indices, con_indices);
        assert!(!extractor.has_exchange());
        assert!(!extractor.has_thermal());
    }

    // Helper functions for tests
    fn create_test_variables(
        with_exchange: bool,
        with_thermal: bool,
    ) -> crate::subproblem::Variables {
        crate::subproblem::Variables {
            deficit: vec![0, 1, 2],
            direct_exchange: if with_exchange { vec![3, 4] } else { vec![] },
            reverse_exchange: if with_exchange { vec![5, 6] } else { vec![] },
            thermal_gen: if with_thermal { vec![7, 8, 9] } else { vec![] },
            turbined_flow: vec![10, 11],
            spillage: vec![12, 13],
            stored_volume: vec![14, 15],
            load: vec![16, 17, 18],
            inflow: vec![19, 20],
            innovation: vec![21, 22],
            lagged_state: None,
            load_lags: None,
            inflow_lags: None,
            alpha: 23,
        }
    }

    fn create_test_constraints() -> crate::subproblem::Constraints {
        crate::subproblem::Constraints {
            load_balance: vec![0, 1, 2],
            hydro_balance: vec![3, 4],
            uncertainty_observation: vec![5, 6, 7],
            load_lag_constraints: None,
            inflow_lag_constraints: None,
        }
    }
}
