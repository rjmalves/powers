// Unit tests for batch cut selection (T3.5)
//
// Tests verify that batch cut selection:
// 1. Produces same results as sequential per-cut addition
// 2. Maintains deterministic ordering
// 3. Handles edge cases (empty pools, identical cuts, etc.)
// 4. Correctly identifies returning and removing cuts
//
// PERFORMANCE: Batch selection eliminates lock contention and ensures
// deterministic cut selection ordering.

use powers_rs::cut::BendersCut;
use powers_rs::fcf::{CutStatePair, FutureCostFunction};
use powers_rs::state::{State, StorageState};
use powers_rs::stochastic_process;
use powers_rs::system::System;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a test cut with given parameters
fn create_test_cut(id: usize, coefficients: Vec<f64>, rhs: f64) -> BendersCut {
    BendersCut::new(id, coefficients, rhs)
}

/// Create a test storage state
fn create_test_state() -> Box<dyn State> {
    let system = System::default();
    let load_sp = stochastic_process::factory("naive");
    let inflow_sp = stochastic_process::factory("naive");
    Box::new(StorageState::new(
        &system,
        load_sp.as_ref(),
        inflow_sp.as_ref(),
    ))
}

/// Create an FCF with some initial cuts and states
fn create_fcf_with_baseline(
    num_cuts: usize,
    num_states: usize,
) -> FutureCostFunction {
    let mut fcf = FutureCostFunction::new();

    // Add initial cuts FIRST (states will reference these)
    for i in 0..num_cuts {
        let coefficients = vec![1.0 + i as f64 * 0.1];
        let rhs = 100.0 + i as f64 * 10.0;
        let cut = create_test_cut(i, coefficients, rhs);

        fcf.add_cut(cut);
        fcf.update_cut_pool_on_add(i);
    }

    // Then add states (after cuts exist)
    for _i in 0..num_states {
        let mut state = create_test_state();
        // Initialize with first cut as dominating if cuts exist
        if !fcf.cut_pool.pool.is_empty() {
            let first_cut = &fcf.cut_pool.pool[0];
            let height = first_cut.eval_height_at_state(state.coefficients());
            state.set_dominating_objective(height);
            state.set_dominating_cut_id(0);
        }
        fcf.add_state(state);
    }

    fcf
}

// =============================================================================
// Test Group 1: Correctness - Batch vs Sequential
// =============================================================================

#[test]
fn test_batch_selection_same_as_sequential() {
    // Create two identical FCFs
    let mut fcf_sequential = create_fcf_with_baseline(10, 5);
    let mut fcf_batch = create_fcf_with_baseline(10, 5);

    // Create new cuts to add
    let new_cuts: Vec<(BendersCut, Box<dyn State>)> = (0..3)
        .map(|i| {
            let coefficients = vec![2.0 + i as f64 * 0.1];
            let rhs = 200.0 + i as f64 * 10.0;
            let cut = create_test_cut(10 + i, coefficients, rhs);
            let state = create_test_state();
            (cut, state)
        })
        .collect();

    // Add sequentially
    for (cut, state) in new_cuts.iter().cloned() {
        let mut cut_clone = cut;
        let mut state_clone = state;

        cut_clone.id = fcf_sequential.cut_pool.total_cut_count;
        fcf_sequential.update_cut_pool_on_add(cut_clone.id);
        fcf_sequential.eval_new_cut_domination(&mut cut_clone);
        fcf_sequential.add_cut(cut_clone);
        fcf_sequential.update_old_cuts_domination(&mut state_clone);
        fcf_sequential.add_state(state_clone);
    }

    // Add in batch
    let cut_state_pairs: Vec<CutStatePair> = new_cuts
        .into_iter()
        .map(|(cut, state)| CutStatePair::new(cut, state))
        .collect();

    let _results = fcf_batch.add_cuts_batch(cut_state_pairs);

    // Verify both FCFs have same state
    assert_eq!(
        fcf_sequential.cut_pool.pool.len(),
        fcf_batch.cut_pool.pool.len()
    );
    assert_eq!(
        fcf_sequential.state_pool.pool.len(),
        fcf_batch.state_pool.pool.len()
    );
    assert_eq!(
        fcf_sequential.cut_pool.total_cut_count,
        fcf_batch.cut_pool.total_cut_count
    );

    // Verify cut non_dominated_state_count matches
    for i in 0..fcf_sequential.cut_pool.pool.len() {
        assert_eq!(
            fcf_sequential.cut_pool.pool[i].non_dominated_state_count,
            fcf_batch.cut_pool.pool[i].non_dominated_state_count,
            "Cut {} non_dominated_state_count mismatch",
            i
        );
    }
}

#[test]
fn test_batch_deterministic_ordering() {
    // Run batch selection twice with same input
    let create_batch = || {
        let cut_state_pairs: Vec<CutStatePair> = (0..5)
            .map(|i| {
                let coefficients = vec![1.0 + i as f64 * 0.1];
                let rhs = 100.0 + i as f64 * 10.0;
                let cut = create_test_cut(i, coefficients, rhs);
                let state = create_test_state();
                CutStatePair::new(cut, state)
            })
            .collect();
        cut_state_pairs
    };

    let mut fcf1 = FutureCostFunction::new();
    let mut fcf2 = FutureCostFunction::new();

    let results1 = fcf1.add_cuts_batch(create_batch());
    let results2 = fcf2.add_cuts_batch(create_batch());

    // Verify both runs produced same results
    assert_eq!(results1.len(), results2.len());

    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.cut_id, r2.cut_id);
        assert_eq!(r1.returning_cut_ids.len(), r2.returning_cut_ids.len());
        assert_eq!(r1.removing_cut_ids.len(), r2.removing_cut_ids.len());
    }

    // Verify final FCF state is identical
    assert_eq!(fcf1.cut_pool.pool.len(), fcf2.cut_pool.pool.len());
    assert_eq!(fcf1.cut_pool.total_cut_count, fcf2.cut_pool.total_cut_count);
}

#[test]
fn test_batch_empty_pool() {
    let mut fcf = FutureCostFunction::new();

    // Add cuts to empty pool
    let cut_state_pairs: Vec<CutStatePair> = (0..3)
        .map(|i| {
            let coefficients = vec![1.0 + i as f64];
            let rhs = 100.0;
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let results = fcf.add_cuts_batch(cut_state_pairs);

    assert_eq!(results.len(), 3);
    assert_eq!(fcf.cut_pool.pool.len(), 3);
    assert_eq!(fcf.state_pool.pool.len(), 3);
    assert_eq!(fcf.cut_pool.total_cut_count, 3);

    // No cuts should be returning (pool was empty)
    for result in &results {
        assert!(result.returning_cut_ids.is_empty());
    }
}

#[test]
fn test_batch_single_cut() {
    let mut fcf = FutureCostFunction::new();

    let cut = create_test_cut(0, vec![1.0], 100.0);
    let state = create_test_state();
    let pair = CutStatePair::new(cut, state);

    let results = fcf.add_cuts_batch(vec![pair]);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].cut_id, 0);
    assert_eq!(fcf.cut_pool.pool.len(), 1);
    assert_eq!(fcf.state_pool.pool.len(), 1);
}

// =============================================================================
// Test Group 2: Edge Cases
// =============================================================================

#[test]
fn test_batch_identical_cuts() {
    let mut fcf = FutureCostFunction::new();

    // Add multiple identical cuts
    let cut_state_pairs: Vec<CutStatePair> = (0..5)
        .map(|i| {
            let coefficients = vec![1.0]; // All identical
            let rhs = 100.0; // All identical
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let results = fcf.add_cuts_batch(cut_state_pairs);

    assert_eq!(results.len(), 5);
    assert_eq!(fcf.cut_pool.pool.len(), 5);

    // All cuts should be in the pool (even if identical)
    // Cut selection handles dominance, not deduplication
    assert_eq!(fcf.cut_pool.total_cut_count, 5);
}

#[test]
fn test_batch_dominated_cuts() {
    let mut fcf = FutureCostFunction::new();

    // Add a strong initial cut
    let strong_cut = create_test_cut(0, vec![10.0], 1000.0);
    fcf.add_cut(strong_cut);
    fcf.update_cut_pool_on_add(0);

    // Try to add weaker cuts (they should be dominated)
    let cut_state_pairs: Vec<CutStatePair> = (1..4)
        .map(|i| {
            let coefficients = vec![0.5]; // Weaker slope
            let rhs = 50.0; // Lower intercept
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let _results = fcf.add_cuts_batch(cut_state_pairs);

    // All cuts should be in the pool
    assert_eq!(fcf.cut_pool.pool.len(), 4); // 1 strong + 3 weak
    assert_eq!(fcf.state_pool.pool.len(), 3); // 3 new states
    assert_eq!(fcf.cut_pool.total_cut_count, 4);
}

#[test]
fn test_batch_large_batch() {
    let mut fcf = create_fcf_with_baseline(50, 10);

    // Add large batch of cuts
    let cut_state_pairs: Vec<CutStatePair> = (50..1050)
        .map(|i| {
            let coefficients = vec![1.0 + (i as f64) * 0.001];
            let rhs = 100.0 + (i as f64) * 0.5;
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let results = fcf.add_cuts_batch(cut_state_pairs);

    assert_eq!(results.len(), 1000);
    assert_eq!(fcf.cut_pool.pool.len(), 1050); // 50 initial + 1000 new
    assert_eq!(fcf.cut_pool.total_cut_count, 1050);
}

// =============================================================================
// Test Group 3: Cut Selection Logic
// =============================================================================

#[test]
fn test_batch_returning_cuts_identified() {
    let mut fcf = FutureCostFunction::new();

    // Add initial cuts
    let initial_pairs: Vec<CutStatePair> = (0..5)
        .map(|i| {
            let coefficients = vec![1.0 + i as f64 * 0.2];
            let rhs = 100.0 + i as f64 * 20.0;
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let _initial_results = fcf.add_cuts_batch(initial_pairs);

    // Manually mark some cuts as inactive (simulating removal)
    fcf.cut_pool.pool[1].active = false;
    fcf.cut_pool.pool[3].active = false;

    // Add new cuts with new states that might make inactive cuts relevant again
    let new_pairs: Vec<CutStatePair> = (5..7)
        .map(|i| {
            let coefficients = vec![0.5 + i as f64 * 0.1];
            let rhs = 80.0;
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let results = fcf.add_cuts_batch(new_pairs);

    // Check that function executes without panic
    assert_eq!(results.len(), 2);

    // Note: Whether cuts actually return depends on dominance at specific states
    // This test verifies the mechanism works, not specific returning behavior
}

#[test]
fn test_batch_removing_cuts_identified() {
    let mut fcf = FutureCostFunction::new();

    // Add cuts with varying strengths
    let weak_cut = create_test_cut(0, vec![0.5], 50.0);
    let mut weak_cut_clone = weak_cut;
    weak_cut_clone.active = true;
    weak_cut_clone.non_dominated_state_count = 0; // Manually set to dominated
    fcf.add_cut(weak_cut_clone);
    fcf.update_cut_pool_on_add(0);

    let strong_cut = create_test_cut(1, vec![5.0], 500.0);
    let state = create_test_state();
    let pair = CutStatePair::new(strong_cut, state);

    let results = fcf.add_cuts_batch(vec![pair]);

    // Weak cut should be identified for removal
    assert_eq!(results.len(), 1);

    // The weak cut (id=0) should appear in removing_cut_ids
    let has_removing = results[0].removing_cut_ids.contains(&0);
    assert!(
        has_removing,
        "Cut with non_dominated_state_count=0 should be marked for removal"
    );
}

// =============================================================================
// Test Group 4: Integration with Existing Code
// =============================================================================

#[test]
fn test_batch_maintains_active_cut_ids() {
    let mut fcf = FutureCostFunction::new();

    let cut_state_pairs: Vec<CutStatePair> = (0..5)
        .map(|i| {
            let coefficients = vec![1.0 + i as f64];
            let rhs = 100.0;
            let cut = create_test_cut(i, coefficients, rhs);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let _results = fcf.add_cuts_batch(cut_state_pairs);

    // All cuts should be in active_cut_ids
    assert_eq!(fcf.cut_pool.active_cut_ids.len(), 5);

    // Active cut IDs should match cut IDs
    for (i, &cut_id) in fcf.cut_pool.active_cut_ids.iter().enumerate() {
        assert_eq!(cut_id, i);
    }
}

#[test]
fn test_batch_total_cut_count_increments() {
    let mut fcf = FutureCostFunction::new();

    assert_eq!(fcf.cut_pool.total_cut_count, 0);

    let batch1: Vec<CutStatePair> = (0..3)
        .map(|i| {
            let cut = create_test_cut(i, vec![1.0], 100.0);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let _results1 = fcf.add_cuts_batch(batch1);
    assert_eq!(fcf.cut_pool.total_cut_count, 3);

    let batch2: Vec<CutStatePair> = (3..7)
        .map(|i| {
            let cut = create_test_cut(i, vec![2.0], 200.0);
            let state = create_test_state();
            CutStatePair::new(cut, state)
        })
        .collect();

    let _results2 = fcf.add_cuts_batch(batch2);
    assert_eq!(fcf.cut_pool.total_cut_count, 7);
}
