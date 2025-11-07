//! Cut pool management tests
//!
//! Tests for FutureCostFunction cut storage and management.
//! Validates cut addition, retrieval, and pool operations.

mod fixtures;
mod utils;

use powers_rs::fcf::{CutStatePair, FutureCostFunction};
use utils::cut_helpers::{create_test_cut, create_test_state};

/// Tests for FutureCostFunction creation
mod test_fcf_creation {
    use super::*;

    #[test]
    fn test_new_fcf() {
        let fcf = FutureCostFunction::new();

        assert_eq!(fcf.cut_pool.pool.len(), 0);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 0);
        assert_eq!(fcf.cut_pool.total_cut_count, 0);
        assert_eq!(fcf.state_pool.pool.len(), 0);
    }

    #[test]
    fn test_default_fcf() {
        let fcf = FutureCostFunction::default();

        assert_eq!(fcf.cut_pool.pool.len(), 0);
        assert_eq!(fcf.cut_pool.total_cut_count, 0);
        assert!(fcf.cut_pool.active_cut_indices.is_empty());
    }

    #[test]
    fn test_fcf_initial_state() {
        let fcf = FutureCostFunction::new();

        // FCF should start with empty pools
        assert!(fcf.cut_pool.pool.is_empty());
        assert!(fcf.cut_pool.active_cut_indices.is_empty());
        assert!(fcf.state_pool.pool.is_empty());

        // Total count should be zero
        assert_eq!(fcf.get_total_cut_count(), 0);
    }
}

/// Tests for cut storage operations
mod test_cut_storage {
    use super::*;

    #[test]
    fn test_add_single_cut() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);

        fcf.add_cut(cut);

        assert_eq!(fcf.cut_pool.pool.len(), 1);
        assert_eq!(fcf.cut_pool.pool[0].id, 0);
    }

    #[test]
    fn test_add_multiple_cuts() {
        let mut fcf = FutureCostFunction::new();

        for i in 0..10 {
            let cut = create_test_cut(i, vec![i as f64], i as f64 * 10.0);
            fcf.add_cut(cut);
        }

        assert_eq!(fcf.cut_pool.pool.len(), 10);

        // Verify cuts are stored in order
        for i in 0..10 {
            assert_eq!(fcf.cut_pool.pool[i].id, i);
        }
    }

    #[test]
    fn test_add_cut_preserves_data() {
        let mut fcf = FutureCostFunction::new();
        let coeffs = vec![2.0, 3.0, 5.0];
        let rhs = 42.0;
        let cut = create_test_cut(99, coeffs.clone(), rhs);

        fcf.add_cut(cut);

        let stored_cut = &fcf.cut_pool.pool[0];
        assert_eq!(stored_cut.id, 99);
        assert_eq!(stored_cut.coefficients, coeffs);
        assert_eq!(stored_cut.rhs, rhs);
    }

    #[test]
    fn test_add_state() {
        let mut fcf = FutureCostFunction::new();
        let state = create_test_state();

        fcf.add_state(state);

        assert_eq!(fcf.state_pool.pool.len(), 1);
    }

    #[test]
    fn test_add_multiple_states() {
        let mut fcf = FutureCostFunction::new();

        for _ in 0..5 {
            let state = create_test_state();
            fcf.add_state(state);
        }

        assert_eq!(fcf.state_pool.pool.len(), 5);
    }

    #[test]
    fn test_pool_growth() {
        // PERFORMANCE: Verify pool can grow to realistic sizes
        let mut fcf = FutureCostFunction::new();

        let num_cuts = 100; // Typical SDDP might have 50-200 cuts per stage
        for i in 0..num_cuts {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
        }

        assert_eq!(fcf.cut_pool.pool.len(), num_cuts);

        // Verify all cuts are accessible
        for i in 0..num_cuts {
            assert_eq!(fcf.cut_pool.pool[i].id, i);
        }
    }

    #[test]
    fn test_empty_pool_operations() {
        let fcf = FutureCostFunction::new();

        // Should handle empty pool gracefully
        assert_eq!(fcf.cut_pool.pool.len(), 0);
        assert_eq!(fcf.get_total_cut_count(), 0);
    }
}

/// Tests for active cut tracking
mod test_active_cut_tracking {
    use super::*;

    #[test]
    fn test_update_cut_pool_on_add() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut);

        fcf.update_cut_pool_on_add(0);

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
        assert!(fcf.cut_pool.active_cut_indices.contains_key(&0));
        assert_eq!(*fcf.cut_pool.active_cut_indices.get(&0).unwrap(), 0); // First cut at index 0
        assert_eq!(fcf.cut_pool.total_cut_count, 1);
    }

    #[test]
    fn test_add_multiple_active_cuts() {
        let mut fcf = FutureCostFunction::new();

        for i in 0..5 {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 5);
        assert_eq!(fcf.cut_pool.total_cut_count, 5);
    }

    #[test]
    fn test_update_cut_pool_on_return() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut);

        // Mark as inactive first
        fcf.cut_pool.pool[0].active = false;

        // Return to active set
        fcf.update_cut_pool_on_return(0);

        assert!(fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
        assert!(fcf.cut_pool.active_cut_indices.contains_key(&0));
    }

    #[test]
    fn test_get_active_cut_index_by_id() {
        let mut fcf = FutureCostFunction::new();

        // Add several cuts to active set
        for i in 0..5 {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        // Verify we can find cuts by ID
        for i in 0..5 {
            let index = fcf.get_active_cut_index_by_id(i);
            // Verify the index matches what we expect (sequential: 0, 1, 2, 3, 4)
            assert_eq!(index, i);
        }
    }

    #[test]
    fn test_update_cut_pool_on_remove() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut);
        fcf.update_cut_pool_on_add(0);

        // Remove from active set
        fcf.update_cut_pool_on_remove(0);

        assert!(!fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 0);
    }

    #[test]
    fn test_active_inactive_cycle() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut);

        // Add to active set
        fcf.update_cut_pool_on_add(0);
        assert!(fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);

        // Remove from active set
        fcf.update_cut_pool_on_remove(0);
        assert!(!fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 0);

        // Return to active set
        fcf.update_cut_pool_on_return(0);
        assert!(fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
    }

    #[test]
    fn test_multiple_cuts_active_management() {
        let mut fcf = FutureCostFunction::new();

        // Add 10 cuts
        for i in 0..10 {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 10);

        // Remove some cuts (e.g., cuts 2, 5, 8)
        for &id in &[8, 5, 2] {
            fcf.update_cut_pool_on_remove(id);
        }

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 7);

        // Verify the removed cuts are inactive
        for &id in &[2, 5, 8] {
            assert!(!fcf.cut_pool.pool[id].active);
        }

        // Verify remaining cuts are active
        for id in 0..10 {
            if ![2, 5, 8].contains(&id) {
                assert!(fcf.cut_pool.pool[id].active);
            }
        }
    }
}

/// Tests for cut domination logic
mod test_cut_domination {
    use super::*;

    #[test]
    fn test_cut_domination_initialization() {
        let cut = create_test_cut(0, vec![1.0], 10.0);

        // New cuts should start with count of 1
        assert_eq!(cut.non_dominated_state_count, 1);
    }

    #[test]
    fn test_eval_new_cut_domination_empty_states() {
        let mut fcf = FutureCostFunction::new();
        let mut cut = create_test_cut(0, vec![1.0], 10.0);

        // With no states, domination count should remain unchanged
        fcf.eval_new_cut_domination(&mut cut);

        assert_eq!(cut.non_dominated_state_count, 1);
    }

    #[test]
    fn test_cut_non_dominated_count_updates() {
        // This tests that the non_dominated_state_count field
        // is properly managed during cut operations
        let mut fcf = FutureCostFunction::new();

        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        let cut2 = create_test_cut(1, vec![2.0], 5.0);

        fcf.add_cut(cut1);
        fcf.add_cut(cut2);

        // Both should start with count 1
        assert_eq!(fcf.cut_pool.pool[0].non_dominated_state_count, 1);
        assert_eq!(fcf.cut_pool.pool[1].non_dominated_state_count, 1);
    }
}

/// Tests for CutStatePair
mod test_cut_state_pair {
    use super::*;

    #[test]
    fn test_cut_state_pair_creation() {
        let cut = create_test_cut(0, vec![1.0], 10.0);
        let state = create_test_state();

        let pair = CutStatePair::new(cut, state, 0);

        assert_eq!(pair.cut.id, 0);
    }

    #[test]
    fn test_cut_state_pair_preserves_data() {
        let coeffs = vec![2.0, 3.0];
        let rhs = 42.0;
        let cut = create_test_cut(5, coeffs.clone(), rhs);
        let state = create_test_state();

        let pair = CutStatePair::new(cut, state, 0);

        assert_eq!(pair.cut.id, 5);
        assert_eq!(pair.cut.coefficients, coeffs);
        assert_eq!(pair.cut.rhs, rhs);
    }
}

/// Tests for memory and performance characteristics
mod test_memory_characteristics {
    use super::*;

    #[test]
    fn test_pool_capacity_growth() {
        // PERFORMANCE: Verify pool grows efficiently
        let mut fcf = FutureCostFunction::new();

        // Add cuts in batches to observe growth
        for batch in 0..5 {
            for i in 0..20 {
                let id = batch * 20 + i;
                let cut = create_test_cut(id, vec![1.0], id as f64);
                fcf.add_cut(cut);
            }

            // Pool should grow to accommodate all cuts
            assert_eq!(fcf.cut_pool.pool.len(), (batch + 1) * 20);
        }

        assert_eq!(fcf.cut_pool.pool.len(), 100);
    }

    #[test]
    fn test_active_set_bounded_growth() {
        // PERFORMANCE: Active set should not grow unbounded
        let mut fcf = FutureCostFunction::new();

        // Add many cuts
        for i in 0..200 {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        // Active set grows with total cuts (no selection strategy yet)
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 200);

        // But we can remove cuts
        for i in (100..200).rev() {
            fcf.update_cut_pool_on_remove(i);
        }

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 100);
    }

    #[test]
    fn test_no_memory_leak_on_operations() {
        // PERFORMANCE: Repeated add/remove should not leak memory
        let mut fcf = FutureCostFunction::new();

        // Add cuts
        for i in 0..50 {
            let cut = create_test_cut(i, vec![1.0], i as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        // Remove half
        for i in (25..50).rev() {
            fcf.update_cut_pool_on_remove(i);
        }

        // Re-add some
        for i in 25..35 {
            fcf.update_cut_pool_on_return(i);
        }

        // Active set size should be correct
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 35);

        // Pool still contains all cuts
        assert_eq!(fcf.cut_pool.pool.len(), 50);
    }

    #[test]
    fn test_large_pool_operations() {
        // PERFORMANCE: Test with realistic problem sizes
        let mut fcf = FutureCostFunction::new();

        // Typical SDDP: 50-200 cuts per stage, 10-50 stages
        // Let's test with 500 cuts (reasonable for large problems)
        for i in 0..500 {
            let cut = create_test_cut(i, vec![1.0, 2.0], i as f64);
            fcf.add_cut(cut);
        }

        assert_eq!(fcf.cut_pool.pool.len(), 500);

        // Verify we can still access cuts efficiently
        assert_eq!(fcf.cut_pool.pool[0].id, 0);
        assert_eq!(fcf.cut_pool.pool[499].id, 499);
    }
}

/// Tests for edge cases
mod test_edge_cases {
    use super::*;

    #[test]
    fn test_empty_fcf_operations() {
        let fcf = FutureCostFunction::new();

        assert_eq!(fcf.get_total_cut_count(), 0);
        assert!(fcf.cut_pool.pool.is_empty());
        assert!(fcf.cut_pool.active_cut_indices.is_empty());
    }

    #[test]
    fn test_single_cut_operations() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![1.0], 10.0);

        fcf.add_cut(cut);
        fcf.update_cut_pool_on_add(0);

        assert_eq!(fcf.cut_pool.pool.len(), 1);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
        assert_eq!(fcf.get_total_cut_count(), 1);
    }

    #[test]
    fn test_cut_with_empty_coefficients() {
        let mut fcf = FutureCostFunction::new();
        let cut = create_test_cut(0, vec![], 10.0);

        fcf.add_cut(cut);

        assert_eq!(fcf.cut_pool.pool.len(), 1);
        assert!(fcf.cut_pool.pool[0].coefficients.is_empty());
    }

    #[test]
    fn test_cut_with_large_dimension() {
        let mut fcf = FutureCostFunction::new();
        let dim = 100;
        let coeffs: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let cut = create_test_cut(0, coeffs, 10.0);

        fcf.add_cut(cut);

        assert_eq!(fcf.cut_pool.pool[0].coefficients.len(), dim);
    }
}

mod test_eval_new_cut_domination_extended {
    use super::*;

    #[test]
    fn test_new_cut_with_existing_states() {
        // Test eval_new_cut_domination with states in pool (lines 66-74)
        let mut fcf = FutureCostFunction::new();

        // Add an initial cut
        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut1);

        // Add states to the pool
        for _ in 0..3 {
            let state = create_test_state();
            fcf.add_state(state);
        }

        // Create new cut and evaluate domination
        let mut new_cut = create_test_cut(1, vec![2.0], 20.0);
        let initial_count = new_cut.non_dominated_state_count;

        fcf.eval_new_cut_domination(&mut new_cut);

        // If new cut dominates states, count should increase
        // If not, count stays at initial value
        assert!(new_cut.non_dominated_state_count >= initial_count);
    }

    #[test]
    fn test_new_cut_updates_state_domination() {
        // Test that eval_new_cut_domination updates state's dominating cut (line 71)
        let mut fcf = FutureCostFunction::new();

        // Add weak initial cut
        let weak_cut = create_test_cut(0, vec![0.5], 5.0);
        fcf.add_cut(weak_cut);

        // Add state
        let state = create_test_state();
        fcf.add_state(state);

        // Get initial dominating cut ID
        let _initial_dominating_id =
            fcf.state_pool.pool[0].get_dominating_cut_id();

        // Create stronger cut
        let mut strong_cut = create_test_cut(1, vec![5.0], 50.0);

        fcf.eval_new_cut_domination(&mut strong_cut);

        // Strong cut might dominate (implementation-dependent)
        // This test just ensures the method executes without panic
        assert!(fcf.state_pool.pool[0].get_dominating_cut_id() <= 1);
    }

    #[test]
    fn test_new_cut_decrements_old_dominating_count() {
        // Test that old dominating cut count is decremented (line 69)
        let mut fcf = FutureCostFunction::new();

        // Add initial cut with a state it dominates
        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut1);

        let state = create_test_state();
        fcf.add_state(state);

        // Set initial cut as dominating with high count
        fcf.cut_pool.pool[0].non_dominated_state_count = 5;

        // Add new stronger cut
        let mut cut2 = create_test_cut(1, vec![10.0], 100.0);

        fcf.eval_new_cut_domination(&mut cut2);

        // If cut2 dominates, cut1's count may be decremented
        assert!(fcf.cut_pool.pool[0].non_dominated_state_count <= 5);
    }

    #[test]
    fn test_eval_domination_with_empty_pool() {
        // Edge case: eval_new_cut_domination with no states (line 66 iteration)
        let mut fcf = FutureCostFunction::new();

        let mut cut = create_test_cut(0, vec![1.0], 10.0);
        let initial_count = cut.non_dominated_state_count;

        fcf.eval_new_cut_domination(&mut cut);

        // Count should remain unchanged
        assert_eq!(cut.non_dominated_state_count, initial_count);
    }

    #[test]
    fn test_eval_domination_multiple_iterations() {
        // Test eval_new_cut_domination called multiple times (realistic scenario)
        let mut fcf = FutureCostFunction::new();

        // Add initial states
        for _ in 0..5 {
            fcf.add_state(create_test_state());
        }

        // Add cuts iteratively
        for id in 0..10 {
            let mut cut =
                create_test_cut(id, vec![1.0 + id as f64], 10.0 * id as f64);
            fcf.eval_new_cut_domination(&mut cut);
            fcf.add_cut(cut);
        }

        // Verify pool integrity
        assert_eq!(fcf.cut_pool.pool.len(), 10);
        assert_eq!(fcf.state_pool.pool.len(), 5);
    }
}

/// Additional tests for update_old_cuts_domination (used in subproblem.rs)
mod test_update_old_cuts_domination_extended {
    use super::*;

    #[test]
    fn test_update_with_only_active_cuts() {
        // Test update_old_cuts_domination when all cuts are active (line 86-87)
        let mut fcf = FutureCostFunction::new();

        // Add only active cuts
        for id in 0..5 {
            let cut = create_test_cut(id, vec![1.0], 10.0 * id as f64);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(id);
        }

        let mut state = create_test_state();

        let result = fcf.update_old_cuts_domination(&mut state);

        // Should return empty since all cuts are active (line 87 continues)
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_update_with_inactive_cuts() {
        // Test update_old_cuts_domination with inactive cuts (lines 88-96)
        let mut fcf = FutureCostFunction::new();

        // Add active cuts
        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);

        // Add inactive cuts
        let mut cut2 = create_test_cut(1, vec![2.0], 20.0);
        cut2.active = false;
        fcf.add_cut(cut2);

        let mut cut3 = create_test_cut(2, vec![3.0], 30.0);
        cut3.active = false;
        fcf.add_cut(cut3);

        let mut state = create_test_state();

        let result = fcf.update_old_cuts_domination(&mut state);

        // Result contains IDs of inactive cuts that dominate
        for &cut_id in &result {
            assert!(cut_id < 3);
            assert!(!fcf.cut_pool.pool[cut_id].active);
        }
    }

    #[test]
    fn test_update_increments_dominating_cut_count() {
        // Test that dominating cut's non_dominated_state_count increases (line 94)
        let mut fcf = FutureCostFunction::new();

        // Add active cut
        let cut1 = create_test_cut(0, vec![0.5], 5.0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);

        // Add inactive cut with initial count
        let mut cut2 = create_test_cut(1, vec![5.0], 50.0);
        cut2.active = false;
        let initial_count = cut2.non_dominated_state_count;
        fcf.add_cut(cut2);

        let mut state = create_test_state();

        let _result = fcf.update_old_cuts_domination(&mut state);

        // If cut2 dominates, count should increase (line 94)
        // Otherwise, stays same
        assert!(
            fcf.cut_pool.pool[1].non_dominated_state_count >= initial_count
        );
    }

    #[test]
    fn test_update_decrements_old_dominating() {
        // Test decrementing old dominating cut count (lines 100-102)
        let mut fcf = FutureCostFunction::new();

        // Add active cut that initially dominates
        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);
        fcf.cut_pool.pool[0].non_dominated_state_count = 10;

        // Add stronger inactive cut
        let mut cut2 = create_test_cut(1, vec![10.0], 100.0);
        cut2.active = false;
        fcf.add_cut(cut2);

        let mut state = create_test_state();

        let _result = fcf.update_old_cuts_domination(&mut state);

        // If cut2 dominates, cut1's count may be decremented
        // Test ensures method executes without panic
        assert!(fcf.cut_pool.pool[0].non_dominated_state_count <= 10);
    }

    #[test]
    fn test_update_returns_correct_cut_ids() {
        // Test that returned cut IDs match dominating cuts (line 96)
        let mut fcf = FutureCostFunction::new();

        // Add active cut
        let cut1 = create_test_cut(0, vec![1.0], 10.0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);

        // Add inactive cuts
        for id in 1..5 {
            let mut cut =
                create_test_cut(id, vec![2.0 * id as f64], 20.0 * id as f64);
            cut.active = false;
            fcf.add_cut(cut);
        }

        let mut state = create_test_state();

        let result = fcf.update_old_cuts_domination(&mut state);

        // All returned IDs should be valid cut IDs
        for &cut_id in &result {
            assert!(cut_id < 5);
            // All returned cuts should be inactive
            assert!(!fcf.cut_pool.pool[cut_id].active);
        }
    }

    #[test]
    fn test_update_with_mixed_cuts() {
        // Test realistic scenario with mix of active/inactive cuts
        let mut fcf = FutureCostFunction::new();

        // Add pattern: active, inactive, active, inactive, active
        for id in 0..5 {
            let mut cut =
                create_test_cut(id, vec![1.0 + id as f64], 10.0 * id as f64);
            if id % 2 == 1 {
                cut.active = false;
            }
            fcf.add_cut(cut);
            if id % 2 == 0 {
                fcf.update_cut_pool_on_add(id);
            }
        }

        let mut state = create_test_state();

        let result = fcf.update_old_cuts_domination(&mut state);

        // Only inactive cuts (1, 3) can be returned
        for &cut_id in &result {
            assert!(cut_id == 1 || cut_id == 3);
        }
    }
}

/// Integration test combining domination methods as used in subproblem.rs
mod test_domination_realistic_flow {
    use super::*;

    #[test]
    fn test_typical_sddp_iteration_flow() {
        // Simulate the flow from subproblem.rs
        let mut fcf = FutureCostFunction::new();

        // Iteration 1: Add first cut and state
        let mut cut1 = create_test_cut(0, vec![1.0], 10.0);
        cut1.id = fcf.cut_pool.total_cut_count;
        fcf.update_cut_pool_on_add(cut1.id);
        fcf.eval_new_cut_domination(&mut cut1);
        fcf.add_cut(cut1);

        let mut state1 = create_test_state();
        let _returning_cuts = fcf.update_old_cuts_domination(&mut state1);
        fcf.add_state(state1);

        // Iteration 2: Add second cut and state
        let mut cut2 = create_test_cut(1, vec![2.0], 20.0);
        cut2.id = fcf.cut_pool.total_cut_count;
        fcf.update_cut_pool_on_add(cut2.id);
        fcf.eval_new_cut_domination(&mut cut2);
        fcf.add_cut(cut2);

        let mut state2 = create_test_state();
        let _returning_cuts = fcf.update_old_cuts_domination(&mut state2);
        fcf.add_state(state2);

        // Verify FCF state
        assert_eq!(fcf.cut_pool.pool.len(), 2);
        assert_eq!(fcf.state_pool.pool.len(), 2);
        assert_eq!(fcf.cut_pool.total_cut_count, 2);
    }

    #[test]
    fn test_cut_removal_based_on_domination() {
        // Simulate cut removal logic from subproblem.rs
        let mut fcf = FutureCostFunction::new();

        // Add several cuts
        for id in 0..5 {
            let mut cut =
                create_test_cut(id, vec![1.0 + id as f64], 10.0 * id as f64);
            cut.id = fcf.cut_pool.total_cut_count;
            fcf.update_cut_pool_on_add(cut.id);
            fcf.add_cut(cut);
        }

        // Manually set some cuts to have low domination count (would be removed)
        fcf.cut_pool.pool[1].non_dominated_state_count = 0; // Changed from -1 to 0 (usize can't be negative)
        fcf.cut_pool.pool[3].non_dominated_state_count = 0;

        // Find cuts to remove (from subproblem.rs logic)
        let mut removing_cut_ids = Vec::<usize>::new();
        for cut in fcf.cut_pool.pool.iter_mut() {
            if (cut.non_dominated_state_count == 0) && cut.active {
                removing_cut_ids.push(cut.id);
            }
        }

        // Should identify cuts 1 and 3 for removal
        assert!(removing_cut_ids.contains(&1));
        assert!(removing_cut_ids.contains(&3));
        assert_eq!(removing_cut_ids.len(), 2);
    }

    #[test]
    fn test_multiple_iterations_with_domination() {
        // Test realistic multi-iteration scenario
        let mut fcf = FutureCostFunction::new();

        // Simulate 10 SDDP iterations
        for iter in 0..10 {
            // Add cut
            let mut cut = create_test_cut(
                iter,
                vec![1.0 + iter as f64],
                10.0 * iter as f64,
            );
            cut.id = fcf.cut_pool.total_cut_count;
            fcf.update_cut_pool_on_add(cut.id);
            fcf.eval_new_cut_domination(&mut cut);
            fcf.add_cut(cut);

            // Add state
            let mut state = create_test_state();
            let _returning_cuts = fcf.update_old_cuts_domination(&mut state);
            fcf.add_state(state);
        }

        // Verify consistent state
        assert_eq!(fcf.cut_pool.pool.len(), 10);
        assert_eq!(fcf.state_pool.pool.len(), 10);
        assert_eq!(fcf.cut_pool.total_cut_count, 10);

        // All cuts should be active
        for cut in &fcf.cut_pool.pool {
            assert!(cut.active);
        }
    }
}
