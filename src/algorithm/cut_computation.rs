//! Cut computation utilities for SDDP backward pass.
//!
//! This module provides a clean interface for computing Benders cuts from
//! branching scenario solutions. The core computation logic is delegated
//! to `Subproblem::compute_cut_data()`.
//!
//! # Architecture
//!
//! Cut computation is part of backward pass Phase 1:
//!
//! 1. Solve branching scenarios (parallel across handlers)
//! 2. Compute cut coefficients from solutions (`compute_cut_data`)
//! 3. Return `CutData` for Phase 2 batch selection
//!
//! # Future Enhancements
//!
//! This module can be extended to include:
//! - Cut normalization strategies
//! - Parallel cut coefficient computation (for large state spaces)
//! - Cut quality metrics and filtering

// Re-export CutData for convenience
pub use crate::fcf::CutData;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_data_reexport() {
        // Verify CutData is accessible through this module
        let _: fn(CutData) = |_| {};
    }
}
