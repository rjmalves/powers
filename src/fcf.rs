use crate::cut;
use crate::state;
use std::collections::HashSet;

// Epsilon for numerical equality in domination evaluation.
//
// When two cut heights differ by less than this threshold, they are considered
// numerically equal and tie-breaking by cut ID is used to ensure deterministic
// selection. This prevents non-determinism from floating-point rounding errors.
//
// **Value Selection Rationale (1e-6)**:
//
// 1. **Typical Value Magnitudes**:
//    - Cut coefficients: O(1) to O(100) (water values in $/MWh)
//    - State values: O(10^3) to O(10^6) MWh (reservoir storage)
//    - Cut RHS: O(10^6) to O(10^9) (future costs)
//    - Heights: O(10^6) to O(10^9) (RHS - dot product)
//
// 2. **IEEE 754 Double Precision**:
//    - 53-bit mantissa ≈ 15-17 decimal digits
//    - For values ~10^9, machine epsilon is ~10^-6 (absolute)
//    - Kahan summation improves to ~10^-15 relative precision
//    - But intermediate FMA operations still accumulate errors
//
// 3. **Safety Margin**:
//    - 1e-6 is ~10 orders of magnitude below typical cost values
//    - Well above machine epsilon for O(10^9) values (~1e-6 absolute)
//    - Conservative enough to avoid false equality
//    - Large enough to catch genuine floating-point rounding differences
//
// 4. **Why Not Smaller** (e.g., 1e-10):
//    - Too sensitive to numerical noise from Kahan summation
//    - Would not reliably catch FMA-induced differences
//    - Could cause spurious tie-breaking when cuts are genuinely different
//
// 5. **Why Not Larger** (e.g., 1e-3):
//    - Would incorrectly treat distinct cuts as equal
//    - Could mask genuine domination relationships
//    - Would reduce cut selection effectiveness
//    - Costs differing by $1000 are meaningfully different
//
// **Impact on Algorithm**:
// - Heights within 1e-6 (~ $0.000001) are considered equal → tie-break by ID
// - Heights differing by > 1e-6 use standard comparison
// - Ensures same cut dominates same state across all runs
// - Critical for 100% reproducible lower bounds
//
const DOMINATION_EPSILON: f64 = 1e-6;

#[derive(Default)]
pub struct FutureCostFunction {
    pub cut_pool: cut::BendersCutPool,
    pub state_pool: state::VisitedStatePool,
}

impl FutureCostFunction {
    pub fn new() -> Self {
        Self {
            cut_pool: cut::BendersCutPool::new(),
            state_pool: state::VisitedStatePool::new(),
        }
    }

    pub fn add_cut(&mut self, new_cut: cut::BendersCut) {
        self.cut_pool.pool.push(new_cut);
    }

    pub fn add_state(&mut self, new_state: Box<dyn state::State>) {
        self.state_pool.pool.push(new_state);
    }

    pub fn get_total_cut_count(&self) -> usize {
        self.cut_pool.total_cut_count
    }

    /// Tests the new cut on every previously visited state. If this cut dominates,
    /// decrements the previous dominating cut counter and updates this.
    pub fn eval_new_cut_domination(&mut self, new_cut: &mut cut::BendersCut) {
        for state in self.state_pool.pool.iter_mut() {
            let height = new_cut.eval_height_at_state(state.coefficients());
            let current_dominating_obj = state.get_dominating_objective();

            // Use epsilon-based comparison with tie-breaking.
            //
            // When heights are numerically equal (within DOMINATION_EPSILON),
            // prefer lower cut ID for deterministic selection. This ensures
            // the same cut dominates across runs, preventing dominating_cut_id
            // variations that cause diverging lower bounds.
            //
            // **Why Tie-Breaking is Needed**:
            // Even with deterministic height computation, two cuts
            // can have genuinely equal heights OR heights that differ only by
            // floating-point rounding noise. Without tie-breaking, the >= comparison
            // becomes non-deterministic:
            //   - Run 1: height = 100.0000001, dominating = 100.0000000 → replace
            //   - Run 2: height = 100.0000000, dominating = 100.0000001 → don't replace
            //
            // **Tie-Breaking Strategy**: Prefer lower cut ID
            // - Lower ID = added earlier = more "established" cut
            // - Consistent with BTreeMap ordering
            // - Minimizes domination updates (older cuts more central to policy)
            //
            let should_update = if (height - current_dominating_obj).abs()
                < DOMINATION_EPSILON
            {
                // Heights numerically equal - tie-break by ID (prefer lower)
                new_cut.id < state.get_dominating_cut_id()
            } else {
                // Heights clearly different - use standard comparison
                height > current_dominating_obj
            };

            if should_update {
                let old_cut_id = state.get_dominating_cut_id();
                // Only decrement if old_cut_id is valid (within pool bounds)
                if old_cut_id < self.cut_pool.pool.len() {
                    // Use saturating_sub to prevent underflow (stays at 0 if already 0)
                    self.cut_pool.pool[old_cut_id].non_dominated_state_count =
                        self.cut_pool.pool[old_cut_id]
                            .non_dominated_state_count
                            .saturating_sub(1);
                }
                new_cut.non_dominated_state_count += 1;
                state.update_dominating_cut(new_cut, height);
            }
        }
    }

    /// Tests the cuts that are not in the model for the new state. If any of these cuts
    /// dominate the new state, increment their counter and puts them back inside the model
    pub fn update_old_cuts_domination(
        &mut self,
        new_state: &mut Box<dyn state::State>,
    ) -> Vec<usize> {
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();
        for old_cut in self.cut_pool.pool.iter_mut() {
            match old_cut.active {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(new_state.coefficients());
                    let current_dominating_obj =
                        new_state.get_dominating_objective();

                    // Same epsilon-based tie-breaking as eval_new_cut_domination.
                    // Ensures consistent domination decisions when heights are numerically equal.
                    let should_update = if (height - current_dominating_obj)
                        .abs()
                        < DOMINATION_EPSILON
                    {
                        // Heights numerically equal - tie-break by ID (prefer lower)
                        old_cut.id < new_state.get_dominating_cut_id()
                    } else {
                        // Heights clearly different - use standard comparison
                        height > current_dominating_obj
                    };

                    if should_update {
                        cut_non_dominated_decrement_ids
                            .push(new_state.get_dominating_cut_id());

                        old_cut.non_dominated_state_count += 1;
                        new_state.update_dominating_cut(old_cut, height);
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }
        // Decrements the non-dominating counts using saturating_sub
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cut_pool.pool[*cut_id].non_dominated_state_count =
                self.cut_pool.pool[*cut_id]
                    .non_dominated_state_count
                    .saturating_sub(1);
        }

        cut_ids_to_return_to_model
    }

    pub fn update_cut_pool_on_add(&mut self, cut_id: usize) {
        // New cuts are always added at the end of the active list
        let new_index = self.cut_pool.active_cut_indices.len();
        self.cut_pool.active_cut_indices.insert(cut_id, new_index);
        self.cut_pool.total_cut_count += 1;
    }

    pub fn update_cut_pool_on_return(&mut self, cut_id: usize) {
        // Returning cuts are added at the end of the active list
        let new_index = self.cut_pool.active_cut_indices.len();
        self.cut_pool.active_cut_indices.insert(cut_id, new_index);
        self.cut_pool.pool[cut_id].active = true;
    }

    pub fn get_active_cut_index_by_id(&self, cut_id: usize) -> usize {
        // Direct O(1) lookup
        *self.cut_pool.active_cut_indices.get(&cut_id).unwrap()
    }

    pub fn update_cut_pool_on_remove(&mut self, cut_id: usize) {
        // Remove and mark as inactive
        if let Some(removed_index) =
            self.cut_pool.active_cut_indices.remove(&cut_id)
        {
            self.cut_pool.pool[cut_id].active = false;

            // Adjust indices for all cuts after the removed one
            // When we remove a cut from the model, all subsequent constraints shift down
            for (_id, index) in self.cut_pool.active_cut_indices.iter_mut() {
                if *index > removed_index {
                    *index -= 1;
                }
            }
        }
    }

    /// Add multiple cuts in batch (deterministic cut selection)
    ///
    /// This processes cut-state pairs sequentially in a single lock acquisition,
    /// eliminating lock contention and ensuring deterministic ordering.
    ///
    /// Dominated cut detection must happen ONCE after ALL cuts in the batch
    /// are processed. Detecting per-cut would find the SAME dominated cuts multiple times!
    ///
    pub fn add_cuts_batch(
        &mut self,
        cut_state_pairs: Vec<CutStatePair>,
    ) -> BatchCutSelectionResult {
        let mut new_cut_ids = HashSet::new();
        let mut returning_cut_ids = HashSet::new();

        // ============================================================
        // PHASE 1: Process all cuts and update dominance counters
        // ============================================================
        // This updates non_dominated_state_count for each cut but does NOT
        // yet determine which cuts to remove. That happens ONCE at the end.
        // Intra-batch domination is handled: later cuts can dominate earlier ones!

        for pair in cut_state_pairs.into_iter() {
            let mut cut = pair.cut;
            let mut state = pair.state;

            // Assign ID and add to pool
            cut.id = self.cut_pool.total_cut_count;
            new_cut_ids.insert(cut.id);
            self.update_cut_pool_on_add(cut.id);

            // The cut immediately dominates its source state
            // This must happen AFTER assigning the real cut ID
            let cut_height = cut.eval_height_at_state(state.coefficients());
            state.update_dominating_cut(&cut, cut_height);

            // Evaluate dominance against ALL previous states (including from this batch)
            // This handles intra-batch domination correctly!
            self.eval_new_cut_domination(&mut cut);
            self.add_cut(cut);

            // Update with new state and check for cuts to return
            let returning_ids = self.update_old_cuts_domination(&mut state);
            returning_cut_ids.extend(returning_ids);

            self.add_state(state);
        }

        // ============================================================
        // PHASE 2: Identify ALL dominated cuts ONCE
        // ============================================================
        // This happens AFTER all cuts in the batch have been processed,
        // ensuring we don't find the same dominated cuts multiple times.

        let removing_cut_ids: HashSet<usize> = self
            .cut_pool
            .pool
            .iter()
            .filter(|c| c.non_dominated_state_count == 0 && c.active)
            .map(|c| c.id)
            .collect();

        BatchCutSelectionResult {
            new_cut_ids,
            returning_cut_ids,
            removing_cut_ids,
        }
    }
}

/// Pair of cut and state with metadata for deterministic processing.
///
/// The `forward_pass_idx` field is critical for achieving
/// deterministic cut ordering in parallel execution. When multiple forward passes
/// run in parallel, cuts arrive in non-deterministic order based on thread timing.
/// Sorting by this integer ID ensures consistent processing order regardless of
/// thread scheduling, which is essential because intra-batch cut domination is
/// order-dependent.
pub struct CutStatePair {
    pub cut: cut::BendersCut,
    pub state: Box<dyn state::State>,
    pub forward_pass_idx: usize,
}

impl CutStatePair {
    pub fn new(
        cut: cut::BendersCut,
        state: Box<dyn state::State>,
        forward_pass_idx: usize,
    ) -> Self {
        Self {
            cut,
            state,
            forward_pass_idx,
        }
    }
}

/// Result of batch cut selection for an entire batch
///
/// This struct aggregates cut selection results for ALL cuts processed in a single batch.
/// Unlike the old design where each cut had its own result, this returns a single result
/// containing all the information needed to update the model.
pub struct BatchCutSelectionResult {
    /// IDs of all newly added cuts in this batch
    pub new_cut_ids: HashSet<usize>,
    /// IDs of inactive cuts that should be returned to the model
    pub returning_cut_ids: HashSet<usize>,
    /// IDs of active cuts that are dominated and should be removed
    pub removing_cut_ids: HashSet<usize>,
}

/// Result of batch cut selection for one cut
///
/// Contains information about which cuts need to be added/returned/removed
/// from the subproblem model after cut selection.
pub struct CutSelectionResult {
    /// ID of the newly added cut
    pub cut_id: usize,
    /// IDs of cuts that were inactive but should be returned to the model
    pub returning_cut_ids: Vec<usize>,
    /// IDs of cuts that are dominated and should be removed from the model
    pub removing_cut_ids: Vec<usize>,
}

/// Aggregated result of batch cut selection for ALL cuts
///
/// This aggregates results from multiple cuts to ensure ALL handler models
/// receive the SAME updates. This is critical for maintaining:
/// 1. Model consistency across all handlers
/// 2. Lower bound monotonicity (LB is evaluated on handler 0's model)
/// 3. Correctness of the SDDP algorithm
///
pub struct AggregatedCutSelectionResult {
    /// IDs of all newly added cuts
    pub new_cut_ids: HashSet<usize>,
    /// IDs of ALL cuts that should be returned to models (union across all results)
    pub returning_cut_ids: HashSet<usize>,
    /// IDs of ALL cuts that should be removed from models (union across all results)
    pub removing_cut_ids: HashSet<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StorageState;
    use crate::stochastic_process;
    use crate::system;

    #[test]
    fn test_new_future_cost_function() {
        let fcf = FutureCostFunction::new();
        assert_eq!(fcf.cut_pool.total_cut_count, 0);
        assert!(fcf.state_pool.pool.is_empty());
    }

    #[test]
    fn test_add_cut() {
        let mut fcf = FutureCostFunction::new();
        let cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        fcf.add_cut(cut);
        assert_eq!(fcf.cut_pool.pool.len(), 1);
    }

    #[test]
    fn test_add_state() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let state = Box::new(StorageState::new(
            &system,
            load_sp.as_ref(),
            &inflow_processes,
        ));
        fcf.add_state(state);
        assert_eq!(fcf.state_pool.pool.len(), 1);
    }

    #[test]
    fn test_get_total_cut_count() {
        let mut fcf = FutureCostFunction::new();
        assert_eq!(fcf.get_total_cut_count(), 0);

        let cut1 = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);
        assert_eq!(fcf.get_total_cut_count(), 1);

        let cut2 = cut::BendersCut::new(1, vec![2.0], 20.0, 1, 0);
        fcf.add_cut(cut2);
        fcf.update_cut_pool_on_add(1);
        assert_eq!(fcf.get_total_cut_count(), 2);
    }

    #[test]
    fn test_update_cut_pool_on_add() {
        let mut fcf = FutureCostFunction::new();
        let cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        fcf.add_cut(cut);

        fcf.update_cut_pool_on_add(0);

        assert_eq!(fcf.cut_pool.total_cut_count, 1);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
        assert_eq!(*fcf.cut_pool.active_cut_indices.get(&0).unwrap(), 0);
    }

    #[test]
    fn test_update_cut_pool_on_return() {
        let mut fcf = FutureCostFunction::new();
        let mut cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        cut.active = false;
        fcf.add_cut(cut);

        fcf.update_cut_pool_on_return(0);

        assert!(fcf.cut_pool.pool[0].active);
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
    }

    #[test]
    fn test_eval_new_cut_domination_empty_states() {
        let mut fcf = FutureCostFunction::new();
        let mut cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);

        // Cuts start with non_dominated_state_count = 1
        assert_eq!(cut.non_dominated_state_count, 1);

        // Should not crash with empty state pool
        fcf.eval_new_cut_domination(&mut cut);

        // Counter should remain unchanged since there are no states
        assert_eq!(cut.non_dominated_state_count, 1);
    }

    #[test]
    fn test_eval_new_cut_domination_with_state() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];

        // Add a state
        let state = Box::new(StorageState::new(
            &system,
            load_sp.as_ref(),
            &inflow_processes,
        ));
        fcf.add_state(state);

        // Add and evaluate a cut
        let mut cut = cut::BendersCut::new(0, vec![1.0], 100.0, 1, 0);
        fcf.eval_new_cut_domination(&mut cut);

        // Function should execute without crashing
    }

    #[test]
    fn test_update_old_cuts_domination_empty() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];
        let mut state: Box<dyn state::State> = Box::new(StorageState::new(
            &system,
            load_sp.as_ref(),
            &inflow_processes,
        ));

        // Should return empty vector when no cuts exist
        let returned_cuts = fcf.update_old_cuts_domination(&mut state);
        assert!(returned_cuts.is_empty());
    }

    #[test]
    fn test_default_future_cost_function() {
        let fcf = FutureCostFunction::default();
        assert_eq!(fcf.cut_pool.total_cut_count, 0);
        assert!(fcf.state_pool.pool.is_empty());
    }

    #[test]
    fn test_get_active_cut_index_by_id() {
        let mut fcf = FutureCostFunction::new();

        // Add first cut
        let cut1 = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        fcf.add_cut(cut1);
        fcf.update_cut_pool_on_add(0);

        // Add second cut
        let cut2 = cut::BendersCut::new(1, vec![2.0], 20.0, 1, 0);
        fcf.add_cut(cut2);
        fcf.update_cut_pool_on_add(1);

        // Verify indices
        assert_eq!(fcf.get_active_cut_index_by_id(0), 0);
        assert_eq!(fcf.get_active_cut_index_by_id(1), 1);
    }

    #[test]
    fn test_update_cut_pool_on_remove_single() {
        let mut fcf = FutureCostFunction::new();

        // Add and activate a cut
        let cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);
        fcf.add_cut(cut);
        fcf.update_cut_pool_on_add(0);

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
        assert!(fcf.cut_pool.pool[0].active);

        // Remove the cut
        fcf.update_cut_pool_on_remove(0);

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 0);
        assert!(!fcf.cut_pool.pool[0].active);
    }

    #[test]
    fn test_update_cut_pool_on_remove_adjusts_indices() {
        let mut fcf = FutureCostFunction::new();

        // Add three cuts
        for i in 0..3 {
            let cut =
                cut::BendersCut::new(i, vec![1.0], 10.0 * (i as f64), 1, 0);
            fcf.add_cut(cut);
            fcf.update_cut_pool_on_add(i);
        }

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 3);
        assert_eq!(fcf.get_active_cut_index_by_id(0), 0);
        assert_eq!(fcf.get_active_cut_index_by_id(1), 1);
        assert_eq!(fcf.get_active_cut_index_by_id(2), 2);

        // Remove middle cut (id=1, index=1)
        fcf.update_cut_pool_on_remove(1);

        // Verify cut 2's index decreased from 2 to 1
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 2);
        assert_eq!(fcf.get_active_cut_index_by_id(0), 0);
        assert_eq!(fcf.get_active_cut_index_by_id(2), 1); // Shifted down
        assert!(!fcf.cut_pool.pool[1].active);
    }

    #[test]
    fn test_update_old_cuts_domination_with_inactive_cut() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let load_sp = stochastic_process::factory("naive");
        let inflow_sp = stochastic_process::factory("naive");
        let inflow_processes = vec![inflow_sp];

        // Add a cut and mark it inactive
        let mut cut = cut::BendersCut::new(0, vec![1.0], 100.0, 1, 0);
        cut.active = false;
        fcf.add_cut(cut);

        // Create new state
        let mut state: Box<dyn state::State> = Box::new(StorageState::new(
            &system,
            load_sp.as_ref(),
            &inflow_processes,
        ));

        // Update should consider inactive cuts
        let returned_cuts = fcf.update_old_cuts_domination(&mut state);

        // Verify function executes (may or may not return cuts depending on domination)
        assert!(returned_cuts.len() <= 1);
    }

    #[test]
    fn test_aggregated_cut_selection_result_default() {
        // Test that HashSet fields are properly initialized
        let result = AggregatedCutSelectionResult {
            new_cut_ids: HashSet::new(),
            returning_cut_ids: HashSet::new(),
            removing_cut_ids: HashSet::new(),
        };

        assert!(result.new_cut_ids.is_empty());
        assert!(result.returning_cut_ids.is_empty());
        assert!(result.removing_cut_ids.is_empty());
    }
}
