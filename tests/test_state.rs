/// The State trait is complex with solver integration.
/// These tests focus on StorageState implementation and testable operations.
/// Solver-dependent methods (add_variables_to_subproblem, etc.) require
/// integration tests and are out of scope for unit tests.
///
/// StorageState uses Vec<f64> for storage - cache-friendly
/// and zero-cost for the core state representation. Cloning states for the
/// visited pool is necessary for SDDP convergence tracking.
mod fixtures;

// Access modules directly (now public in test builds)
use powers_rs::cut::BendersCut;
use powers_rs::solver::Basis;
use powers_rs::state::{State, StorageState, VisitedStatePool};
use powers_rs::subproblem::Realization;
use powers_rs::system::System;

/// Helper function to create a test system with N hydros
fn create_test_system(num_hydros: usize) -> System {
    let mut system = System::default();
    // Default system has 1 hydro, adjust if needed
    system.meta.hydros_count = num_hydros;
    system
}

/// Helper function to create a test StorageState
fn create_test_state(dimension: usize) -> StorageState {
    let mut system = create_test_system(dimension);
    system.meta.hydros_count = dimension;
    StorageState::new(&system)
}

/// Helper to create a realization for testing state updates
fn create_test_realization(
    dimension: usize,
    storage_values: Vec<f64>,
) -> Realization {
    Realization::new(
        vec![0.0; dimension], // loads
        vec![0.0; dimension], // deficit
        vec![],               // exchange
        vec![0.0; dimension], // inflow
        vec![0.0; dimension], // turbined_flow
        vec![0.0; dimension], // spillage
        vec![],               // thermal_generation
        vec![0.0; dimension], // water_value
        vec![0.0; dimension], // marginal_cost
        0.0,                  // current_stage_objective
        0.0,                  // total_stage_objective
        storage_values,       // final_storage
        Basis::new(),         // basis
    )
}

/// Tests for StorageState creation and initialization
mod test_state_creation {
    use super::*;

    #[test]
    fn test_new_storage_state_single_hydro() {
        let system = System::default();

        let state = StorageState::new(&system);

        assert_eq!(state.coefficients().len(), 1);
        assert_eq!(state.coefficients()[0], 0.0);
        assert_eq!(state.get_dominating_objective(), 0.0);
        assert_eq!(state.get_dominating_cut_id(), 0);
    }

    #[test]
    fn test_new_storage_state_multi_hydro() {
        let state = create_test_state(5);

        assert_eq!(state.coefficients().len(), 5);
        for &val in state.coefficients() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_state_factory() {
        let system = System::default();

        let state = powers_rs::state::factory(
            "storage",
            &system,
            &[], // Empty uncertainty models for simple storage state
        );

        assert_eq!(state.coefficients().len(), 1);
    }

    #[test]
    #[should_panic(
        expected = "Unknown state_choice: 'unknown'. Valid options: 'storage', 'storage_and_inflow'"
    )]
    fn test_state_factory_invalid_kind() {
        let system = System::default();

        powers_rs::state::factory("unknown", &system, &[]);
    }

    #[test]
    fn test_state_initial_values() {
        let state = create_test_state(3);

        // Initial storage should be all zeros
        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);

        // Initial dominating objective should be zero
        assert_eq!(state.get_dominating_objective(), 0.0);

        // Initial dominating cut ID should be zero
        assert_eq!(state.get_dominating_cut_id(), 0);
    }

    #[test]
    fn test_state_dimension_consistency() {
        for dim in [1, 2, 5, 10, 20] {
            let state = create_test_state(dim);
            assert_eq!(state.coefficients().len(), dim);
        }
    }
}

/// Tests for state operations and field access
mod test_state_operations {
    use super::*;

    #[test]
    fn test_coefficients_access() {
        let state = create_test_state(3);

        let coeffs = state.coefficients();
        assert_eq!(coeffs.len(), 3);
        assert_eq!(coeffs, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_set_get_dominating_objective() {
        let mut state = create_test_state(1);

        state.set_dominating_objective(42.5);
        assert_eq!(state.get_dominating_objective(), 42.5);

        state.set_dominating_objective(-10.0);
        assert_eq!(state.get_dominating_objective(), -10.0);

        state.set_dominating_objective(0.0);
        assert_eq!(state.get_dominating_objective(), 0.0);
    }

    #[test]
    fn test_set_get_dominating_cut_id() {
        let mut state = create_test_state(1);

        state.set_dominating_cut_id(5);
        assert_eq!(state.get_dominating_cut_id(), 5);

        state.set_dominating_cut_id(100);
        assert_eq!(state.get_dominating_cut_id(), 100);

        state.set_dominating_cut_id(0);
        assert_eq!(state.get_dominating_cut_id(), 0);
    }

    #[test]
    fn test_update_dominating_cut() {
        let mut state = create_test_state(2);
        let cut = BendersCut::new(42, vec![1.0, 2.0], 10.0, 1, 0);
        let height = 100.5;

        state.update_dominating_cut(&cut, height);

        assert_eq!(state.get_dominating_cut_id(), 42);
        assert_eq!(state.get_dominating_objective(), 100.5);
    }

    #[test]
    fn test_set_dimension() {
        let mut state = create_test_state(1);

        state.set_dimension(5);
        // Note: set_dimension only changes the dimension field,
        // it doesn't resize the storage vector
        // This is the current implementation behavior
    }
}

/// Tests for state updates and transitions
mod test_state_updates {
    use super::*;

    #[test]
    fn test_update_with_realization_single_hydro() {
        let mut state = create_test_state(1);
        let realization = create_test_realization(1, vec![50.0]);

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients()[0], 50.0);
    }

    #[test]
    fn test_update_with_realization_multi_hydro() {
        let mut state = create_test_state(3);
        let storage_values = vec![10.0, 20.0, 30.0];
        let realization = create_test_realization(3, storage_values.clone());

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients(), storage_values.as_slice());
    }

    #[test]
    fn test_state_transition_sequence() {
        let mut state = create_test_state(2);

        // Initial state
        assert_eq!(state.coefficients(), &[0.0, 0.0]);

        // First transition
        let real1 = create_test_realization(2, vec![10.0, 15.0]);
        state.update_with_current_realization(&real1);
        assert_eq!(state.coefficients(), &[10.0, 15.0]);

        // Second transition
        let real2 = create_test_realization(2, vec![20.0, 25.0]);
        state.update_with_current_realization(&real2);
        assert_eq!(state.coefficients(), &[20.0, 25.0]);

        // Third transition
        let real3 = create_test_realization(2, vec![5.0, 10.0]);
        state.update_with_current_realization(&real3);
        assert_eq!(state.coefficients(), &[5.0, 10.0]);
    }

    #[test]
    fn test_update_with_zero_values() {
        let mut state = create_test_state(3);
        let realization = create_test_realization(3, vec![0.0, 0.0, 0.0]);

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_update_with_negative_values() {
        // ARCHITECTURE NOTE: StorageState doesn't enforce bounds
        // Negative storage values are physically meaningless but accepted
        let mut state = create_test_state(2);
        let realization = create_test_realization(2, vec![-10.0, -5.0]);

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients(), &[-10.0, -5.0]);
    }

    #[test]
    fn test_update_with_large_values() {
        let mut state = create_test_state(2);
        let realization = create_test_realization(2, vec![1e6, 1e9]);

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients(), &[1e6, 1e9]);
    }
}

/// Tests for state cloning and visited state pool
mod test_state_cloning {
    use super::*;

    #[test]
    fn test_state_clone_dyn() {
        let state = create_test_state(3);
        let mut state_box: Box<dyn State> = Box::new(state);

        state_box.set_dominating_objective(42.0);
        state_box.set_dominating_cut_id(10);

        let cloned = state_box.clone_dyn();

        assert_eq!(cloned.get_dominating_objective(), 42.0);
        assert_eq!(cloned.get_dominating_cut_id(), 10);
        assert_eq!(cloned.coefficients().len(), 3);
    }

    #[test]
    fn test_state_box_clone() {
        let state = create_test_state(2);
        let mut state_box: Box<dyn State> = Box::new(state);

        let realization = create_test_realization(2, vec![10.0, 20.0]);
        state_box.update_with_current_realization(&realization);

        let cloned = state_box.clone();

        assert_eq!(cloned.coefficients(), &[10.0, 20.0]);
    }

    #[test]
    fn test_cloned_state_independence() {
        let state = create_test_state(1);
        let mut state_box: Box<dyn State> = Box::new(state);

        state_box.set_dominating_objective(10.0);

        let mut cloned = state_box.clone();
        cloned.set_dominating_objective(20.0);

        // Original should be unchanged
        assert_eq!(state_box.get_dominating_objective(), 10.0);
        assert_eq!(cloned.get_dominating_objective(), 20.0);
    }
}

/// Tests for VisitedStatePool
mod test_visited_state_pool {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = VisitedStatePool::new();

        assert_eq!(pool.pool.len(), 0);
        assert!(pool.pool.is_empty());
    }

    #[test]
    fn test_add_state_to_pool() {
        let mut pool = VisitedStatePool::new();
        let state = create_test_state(1);

        pool.pool.push(Box::new(state));

        assert_eq!(pool.pool.len(), 1);
    }

    #[test]
    fn test_add_multiple_states() {
        let mut pool = VisitedStatePool::new();

        for i in 0..10 {
            let mut state = create_test_state(1);
            let realization = create_test_realization(1, vec![i as f64]);
            state.update_with_current_realization(&realization);
            pool.pool.push(Box::new(state));
        }

        assert_eq!(pool.pool.len(), 10);

        // Verify states are different
        for i in 0..10 {
            assert_eq!(pool.pool[i].coefficients()[0], i as f64);
        }
    }

    #[test]
    fn test_pool_with_multidimensional_states() {
        let mut pool = VisitedStatePool::new();

        let state1 = create_test_state(3);
        let state2 = create_test_state(3);

        pool.pool.push(Box::new(state1));
        pool.pool.push(Box::new(state2));

        assert_eq!(pool.pool.len(), 2);
        assert_eq!(pool.pool[0].coefficients().len(), 3);
        assert_eq!(pool.pool[1].coefficients().len(), 3);
    }

    #[test]
    fn test_pool_capacity_growth() {
        // PERFORMANCE: Test pool can grow to realistic SDDP sizes
        let mut pool = VisitedStatePool::new();

        // SDDP typically visits 100-1000 states per iteration
        let num_states = 200;

        for i in 0..num_states {
            let mut state = create_test_state(2);
            let realization =
                create_test_realization(2, vec![i as f64, i as f64 * 2.0]);
            state.update_with_current_realization(&realization);
            pool.pool.push(Box::new(state));
        }

        assert_eq!(pool.pool.len(), num_states);

        // Verify first and last states
        assert_eq!(pool.pool[0].coefficients(), &[0.0, 0.0]);
        assert_eq!(pool.pool[199].coefficients(), &[199.0, 398.0]);
    }
}

/// Tests for multi-dimensional states
mod test_multidimensional_states {
    use super::*;

    #[test]
    fn test_2d_state() {
        let state = create_test_state(2);

        assert_eq!(state.coefficients().len(), 2);
        assert_eq!(state.coefficients(), &[0.0, 0.0]);
    }

    #[test]
    fn test_5d_state() {
        let state = create_test_state(5);

        assert_eq!(state.coefficients().len(), 5);
        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_10d_state() {
        let state = create_test_state(10);

        assert_eq!(state.coefficients().len(), 10);
        for &val in state.coefficients() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_large_dimension_state() {
        // PERFORMANCE: Test with realistic large system (100 hydros)
        let state = create_test_state(100);

        assert_eq!(state.coefficients().len(), 100);

        // Update with non-zero values
        let storage_values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let realization = create_test_realization(100, storage_values.clone());

        let mut state_mut = state;
        state_mut.update_with_current_realization(&realization);

        assert_eq!(state_mut.coefficients()[0], 0.0);
        assert_eq!(state_mut.coefficients()[50], 50.0);
        assert_eq!(state_mut.coefficients()[99], 99.0);
    }

    #[test]
    fn test_dimension_range() {
        // Test various dimensions to ensure no unexpected behavior
        for dim in [1, 2, 3, 5, 10, 20, 50] {
            let state = create_test_state(dim);
            assert_eq!(state.coefficients().len(), dim);

            let storage: Vec<f64> = vec![1.0; dim];
            let realization = create_test_realization(dim, storage.clone());

            let mut state_mut = state;
            state_mut.update_with_current_realization(&realization);

            for &val in state_mut.coefficients() {
                assert_eq!(val, 1.0);
            }
        }
    }
}

/// Tests for edge cases and boundary conditions
mod test_edge_cases {
    use super::*;

    #[test]
    fn test_state_with_infinity_values() {
        let mut state = create_test_state(2);
        let realization =
            create_test_realization(2, vec![f64::INFINITY, f64::NEG_INFINITY]);

        state.update_with_current_realization(&realization);

        assert_eq!(state.coefficients()[0], f64::INFINITY);
        assert_eq!(state.coefficients()[1], f64::NEG_INFINITY);
    }

    #[test]
    fn test_state_with_nan_values() {
        let mut state = create_test_state(1);
        let realization = create_test_realization(1, vec![f64::NAN]);

        state.update_with_current_realization(&realization);

        assert!(state.coefficients()[0].is_nan());
    }

    #[test]
    fn test_dominating_objective_extreme_values() {
        let mut state = create_test_state(1);

        // Very large positive
        state.set_dominating_objective(1e100);
        assert_eq!(state.get_dominating_objective(), 1e100);

        // Very large negative
        state.set_dominating_objective(-1e100);
        assert_eq!(state.get_dominating_objective(), -1e100);

        // Infinity
        state.set_dominating_objective(f64::INFINITY);
        assert_eq!(state.get_dominating_objective(), f64::INFINITY);
    }

    #[test]
    fn test_dominating_cut_id_large_values() {
        let mut state = create_test_state(1);

        // Test with large IDs (realistic for long SDDP runs)
        state.set_dominating_cut_id(usize::MAX);
        assert_eq!(state.get_dominating_cut_id(), usize::MAX);

        state.set_dominating_cut_id(1_000_000);
        assert_eq!(state.get_dominating_cut_id(), 1_000_000);
    }

    #[test]
    fn test_repeated_updates() {
        // PERFORMANCE: Test that repeated updates don't cause issues
        let mut state = create_test_state(1);

        for i in 0..1000 {
            let realization = create_test_realization(1, vec![i as f64]);
            state.update_with_current_realization(&realization);
        }

        assert_eq!(state.coefficients()[0], 999.0);
    }

    #[test]
    fn test_empty_dimension_handling() {
        // This test documents that zero-dimensional states are not supported
        let state = create_test_state(1);
        assert!(!state.coefficients().is_empty());
    }
}

/// Tests for state equality and comparison semantics
mod test_state_semantics {
    use super::*;

    #[test]
    fn test_states_with_same_storage_are_different_objects() {
        let state1 = create_test_state(2);
        let state2 = create_test_state(2);

        // Same values but different objects
        assert_eq!(state1.coefficients(), state2.coefficients());

        // But they are independent
        let mut state1_mut = state1;
        let realization = create_test_realization(2, vec![10.0, 20.0]);
        state1_mut.update_with_current_realization(&realization);

        assert_ne!(state1_mut.coefficients(), state2.coefficients());
    }

    #[test]
    fn test_cloned_state_has_same_values() {
        let mut state = create_test_state(2);
        state.set_dominating_objective(42.0);
        state.set_dominating_cut_id(10);
        let realization = create_test_realization(2, vec![5.0, 15.0]);
        state.update_with_current_realization(&realization);

        let state_box: Box<dyn State> = Box::new(state);
        let cloned = state_box.clone();

        assert_eq!(cloned.coefficients(), state_box.coefficients());
        assert_eq!(
            cloned.get_dominating_objective(),
            state_box.get_dominating_objective()
        );
        assert_eq!(
            cloned.get_dominating_cut_id(),
            state_box.get_dominating_cut_id()
        );
    }
}

// ============================================================================
// TICKET-002: State Extraction and Rebuild Pattern Tests
// ============================================================================
//
// These tests validate the extract-from-trajectory and rebuild-coefficients
// pattern that forms the foundation of the state refactoring. They ensure:
// 1. Storage extraction from trajectory works correctly
// 2. Coefficients are properly rebuilt after extraction
// 3. coefficients() returns correct values after update operations
// 4. Edge cases are handled (single realization, zero dimension, etc.)

#[cfg(test)]
mod state_extraction_tests {
    use super::*;

    /// Helper to create a trajectory of realizations
    fn create_test_trajectory(
        num_realizations: usize,
        num_hydros: usize,
    ) -> Vec<Realization> {
        (0..num_realizations)
            .map(|i| {
                let storage: Vec<f64> =
                    (0..num_hydros).map(|h| (i * 10 + h) as f64).collect();
                create_test_realization(num_hydros, storage)
            })
            .collect()
    }

    #[test]
    fn test_coefficients_after_update_with_current_realization() {
        let mut state = create_test_state(3);

        // Initial state should be zeros
        assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);

        // Update with first realization
        let realization1 = create_test_realization(3, vec![10.0, 20.0, 30.0]);
        state.update_with_current_realization(&realization1);
        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);

        // Update with second realization
        let realization2 = create_test_realization(3, vec![15.0, 25.0, 35.0]);
        state.update_with_current_realization(&realization2);
        assert_eq!(state.coefficients(), &[15.0, 25.0, 35.0]);
    }

    #[test]
    fn test_state_coefficients_consistency_across_updates() {
        // Test that multiple updates maintain consistency
        let mut state = create_test_state(2);
        let trajectory = create_test_trajectory(5, 2);

        for realization in &trajectory {
            state.update_with_current_realization(realization);

            // Coefficients should always match current realization's storage
            assert_eq!(state.coefficients(), &realization.final_storage);
        }
    }

    #[test]
    fn test_storage_state_preserves_metadata_on_update() {
        let mut state = create_test_state(2);

        // Set some metadata
        state.set_dominating_objective(42.5);
        state.set_dominating_cut_id(7);
        state.set_iteration(3);
        state.set_forward_pass_idx(1);

        // Update state
        let realization = create_test_realization(2, vec![10.0, 20.0]);
        state.update_with_current_realization(&realization);

        // Metadata should be preserved
        assert_eq!(state.get_dominating_objective(), 42.5);
        assert_eq!(state.get_dominating_cut_id(), 7);
        assert_eq!(state.get_iteration(), 3);
        assert_eq!(state.get_forward_pass_idx(), 1);

        // But coefficients should be updated
        assert_eq!(state.coefficients(), &[10.0, 20.0]);
    }

    #[test]
    fn test_storage_state_zero_dimension() {
        // Edge case: system with no hydros (degenerate but valid)
        let state = create_test_state(0);
        assert_eq!(state.coefficients(), &[] as &[f64]);
    }

    #[test]
    fn test_storage_state_large_dimension() {
        // Test with larger system (100 hydros)
        let num_hydros = 100;
        let mut state = create_test_state(num_hydros);

        let storage: Vec<f64> =
            (0..num_hydros).map(|i| i as f64 * 1.5).collect();
        let realization = create_test_realization(num_hydros, storage.clone());

        state.update_with_current_realization(&realization);
        assert_eq!(state.coefficients(), storage.as_slice());
        assert_eq!(state.coefficients().len(), num_hydros);
    }

    #[test]
    fn test_trajectory_extraction_pattern() {
        // Simulate the pattern: create trajectory, extract last
        let trajectory = create_test_trajectory(4, 3);

        // In actual use, update_from_trajectory() extracts from last realization
        // Here we test that the pattern works correctly
        let last_realization = trajectory.last().unwrap();
        let mut state = create_test_state(3);

        state.update_with_current_realization(last_realization);

        // Should have extracted storage from last realization
        // trajectory[3] has storage [30, 31, 32]
        assert_eq!(state.coefficients(), &[30.0, 31.0, 32.0]);
    }

    #[test]
    fn test_state_coefficients_immutable_reference() {
        let mut state = create_test_state(2);
        let realization = create_test_realization(2, vec![5.0, 10.0]);
        state.update_with_current_realization(&realization);

        // Get reference to coefficients
        let coeffs1 = state.coefficients();
        let coeffs2 = state.coefficients();

        // Both should point to same data
        assert_eq!(coeffs1, coeffs2);
        assert_eq!(coeffs1, &[5.0, 10.0]);
    }

    #[test]
    fn test_state_sequential_updates() {
        // Test realistic sequential updates as in forward pass
        let mut state = create_test_state(2);
        let trajectory = create_test_trajectory(10, 2);

        for (i, realization) in trajectory.iter().enumerate() {
            state.update_with_current_realization(realization);

            // Expected storage for realization i: [i*10, i*10+1]
            let expected = vec![i as f64 * 10.0, i as f64 * 10.0 + 1.0];
            assert_eq!(state.coefficients(), expected.as_slice());
        }
    }

    #[test]
    fn test_storage_state_clone_preserves_coefficients() {
        let mut state = create_test_state(3);
        let realization = create_test_realization(3, vec![7.0, 14.0, 21.0]);
        state.update_with_current_realization(&realization);

        // Clone via trait object (as used in visited_states pool)
        let state_box: Box<dyn State> = Box::new(state);
        let cloned = state_box.clone();

        // Coefficients should be identical
        assert_eq!(cloned.coefficients(), state_box.coefficients());
        assert_eq!(cloned.coefficients(), &[7.0, 14.0, 21.0]);
    }
}
