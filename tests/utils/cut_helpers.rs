//! Cut creation helpers for tests
//!
//! Common utilities for creating test cuts and states to eliminate duplication
//! across test files.

use powers_rs::cut::BendersCut;
use powers_rs::state::{State, StorageState};
use powers_rs::system::System;

/// Create a simple test cut with given parameters
///
/// # Arguments
/// * `id` - Unique cut identifier
/// * `coefficients` - Cut coefficients (typically water values, negative for minimization)
/// * `rhs` - Right-hand side constant
///
/// # Returns
/// A BendersCut with stage_id=1 and node_id=0 (suitable for simple tests)
///
/// # Example
/// ```
/// use tests::utils::cut_helpers::create_test_cut;
///
/// let cut = create_test_cut(1, vec![-1.0, -2.0], 10.0);
/// assert_eq!(cut.id, 1);
/// assert_eq!(cut.eval_height_at_state(&[5.0, 3.0]), 10.0 - 1.0*5.0 - 2.0*3.0);
/// ```
pub fn create_test_cut(
    id: usize,
    coefficients: Vec<f64>,
    rhs: f64,
) -> BendersCut {
    BendersCut::new(id, coefficients, rhs, 1, 0)
}

/// Create a test storage state with default system
///
/// Useful for tests that need a State trait object but don't care about
/// system configuration.
///
/// # Returns
/// Boxed StorageState initialized with default system
///
/// # Example
/// ```
/// use tests::utils::cut_helpers::create_test_state;
///
/// let state = create_test_state();
/// assert_eq!(state.num_state_variables(), system_default_hydro_count);
/// ```
pub fn create_test_state() -> Box<dyn State> {
    let system = System::default();
    Box::new(StorageState::new(&system))
}

/// Create a test storage state with custom system
///
/// # Arguments
/// * `system` - System configuration to use
///
/// # Returns
/// Boxed StorageState initialized with provided system
///
/// # Example
/// ```
/// use tests::utils::cut_helpers::create_test_state_with_system;
/// use powers_rs::system::System;
///
/// let mut system = System::default();
/// system.hydros.push(/* ... */);
/// let state = create_test_state_with_system(&system);
/// ```
#[allow(dead_code)]
pub fn create_test_state_with_system(system: &System) -> Box<dyn State> {
    Box::new(StorageState::new(system))
}

/// Create multiple test cuts with sequential IDs
///
/// Convenience function for creating many similar cuts.
///
/// # Arguments
/// * `count` - Number of cuts to create
/// * `coefficients` - Coefficients to use for all cuts
/// * `rhs_base` - Base RHS value (incremented by 1.0 for each cut)
///
/// # Returns
/// Vector of test cuts
///
/// # Example
/// ```
/// use tests::utils::cut_helpers::create_test_cuts;
///
/// let cuts = create_test_cuts(5, vec![-1.0, -2.0], 100.0);
/// assert_eq!(cuts.len(), 5);
/// assert_eq!(cuts[0].id, 0);
/// assert_eq!(cuts[4].id, 4);
/// ```
pub fn create_test_cuts(
    count: usize,
    coefficients: Vec<f64>,
    rhs_base: f64,
) -> Vec<BendersCut> {
    (0..count)
        .map(|i| create_test_cut(i, coefficients.clone(), rhs_base + i as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_cut_basic() {
        let cut = create_test_cut(1, vec![-1.0, -2.0], 10.0);
        assert_eq!(cut.id, 1);
        assert_eq!(cut.coefficients, vec![-1.0, -2.0]);
        assert_eq!(cut.rhs, 10.0);
    }

    #[test]
    fn test_create_test_cut_evaluation() {
        let cut = create_test_cut(1, vec![-1.0, -2.0], 10.0);
        let height = cut.eval_height_at_state(&[5.0, 3.0]);
        // height = 10.0 + (-1.0)*5.0 + (-2.0)*3.0 = 10.0 - 5.0 - 6.0 = -1.0
        assert_eq!(height, -1.0);
    }

    #[test]
    fn test_create_test_state_default() {
        let _state = create_test_state();
        // Just verify it was created successfully - State is an opaque trait object
        // No public methods to test the structure
    }

    #[test]
    fn test_create_test_cuts_multiple() {
        let cuts = create_test_cuts(5, vec![-1.0], 100.0);
        assert_eq!(cuts.len(), 5);

        for (i, cut) in cuts.iter().enumerate() {
            assert_eq!(cut.id, i);
            assert_eq!(cut.rhs, 100.0 + i as f64);
        }
    }
}
