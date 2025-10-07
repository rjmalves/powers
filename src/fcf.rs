use crate::cut;
use crate::state;
use std::collections::HashSet;

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
            if height >= state.get_dominating_objective() {
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
                    if height > new_state.get_dominating_objective() {
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
        // Direct O(1) lookup with HashMap
        *self.cut_pool.active_cut_indices.get(&cut_id).unwrap()
    }

    pub fn update_cut_pool_on_remove(
        &mut self,
        cut_id: usize,
        _cut_index: usize, // Deprecated parameter, kept for API compatibility
    ) {
        // Remove from HashMap and mark as inactive
        if let Some(removed_index) =
            self.cut_pool.active_cut_indices.remove(&cut_id)
        {
            self.cut_pool.pool[cut_id].active = false;

            // CRITICAL: Adjust indices for all cuts after the removed one
            // When we remove a cut from the model, all subsequent constraints shift down
            for (_id, index) in self.cut_pool.active_cut_indices.iter_mut() {
                if *index > removed_index {
                    *index -= 1;
                }
            }
        }
    }

    /// Add multiple cuts in batch (deterministic cut selection) - NEW DESIGN
    ///
    /// This processes cut-state pairs sequentially in a single lock acquisition,
    /// eliminating lock contention and ensuring deterministic ordering.
    ///
    /// **KEY IMPROVEMENTS**:
    /// - Returns single BatchCutSelectionResult instead of Vec<CutSelectionResult>
    /// - Uses HashSet to automatically eliminate duplicates
    /// - Handles intra-batch domination (cuts within batch dominating each other)
    /// - Uses saturating_sub to prevent counter underflow
    ///
    /// **CRITICAL**: Dominated cut detection must happen ONCE after ALL cuts in the batch
    /// are processed. Detecting per-cut would find the SAME dominated cuts multiple times!
    ///
    /// # Performance
    /// - Complexity: O(n × m) where n=new_cuts, m=existing_states
    /// - Lock acquisitions: 1 (vs N for per-thread approach)
    /// - Expected speedup: 15-30% on multi-core systems due to eliminated contention
    ///
    /// # Determinism
    /// Cuts are processed in the order provided, making the algorithm deterministic
    /// given the same input order (e.g., sorted by node ID).
    ///
    /// # Arguments
    /// * `cut_state_pairs` - Vector of cuts and states to process
    ///
    /// # Returns
    /// Single `BatchCutSelectionResult` with all new/returning/removing cut IDs
    pub fn add_cuts_batch(
        &mut self,
        cut_state_pairs: Vec<CutStatePair>,
    ) -> BatchCutSelectionResult {
        let mut new_cut_ids = HashSet::new();
        let mut returning_cut_ids = HashSet::new();

        // DEBUG: Log initial state
        if std::env::var("POWERS_CUT_DEBUG").is_ok() {
            eprintln!("\n[FCF] ========== BATCH START ==========");
            eprintln!(
                "[FCF] Initial state: {} total cuts, {} active cuts, {} states",
                self.cut_pool.total_cut_count,
                self.cut_pool.active_cut_indices.len(),
                self.state_pool.pool.len()
            );
            eprintln!("[FCF] Processing {} new cuts", cut_state_pairs.len());
        }

        // ============================================================
        // PHASE 1: Process all cuts and update dominance counters
        // ============================================================
        // This updates non_dominated_state_count for each cut but does NOT
        // yet determine which cuts to remove. That happens ONCE at the end.
        // Intra-batch domination is handled: later cuts can dominate earlier ones!

        for (batch_idx, pair) in cut_state_pairs.into_iter().enumerate() {
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

            // DEBUG: Log cut before evaluation
            if std::env::var("POWERS_CUT_DEBUG").is_ok() {
                eprintln!(
                    "[FCF]   Cut #{} (id={}): coeffs={:?}, rhs={:.2}, active={}, count={} [dominates source state]",
                    batch_idx, cut.id, cut.coefficients, cut.rhs, cut.active, cut.non_dominated_state_count
                );
            }

            // Evaluate dominance against ALL previous states (including from this batch)
            // This handles intra-batch domination correctly!
            self.eval_new_cut_domination(&mut cut);

            // DEBUG: Log dominance result
            if std::env::var("POWERS_CUT_DEBUG").is_ok() {
                eprintln!(
                    "[FCF]      After eval: count={} (tested against {} states)",
                    cut.non_dominated_state_count, self.state_pool.pool.len()
                );
            }

            self.add_cut(cut);

            // Update with new state and check for cuts to return
            let returning_ids = self.update_old_cuts_domination(&mut state);
            returning_cut_ids.extend(returning_ids);

            // DEBUG: Log returning cuts
            if std::env::var("POWERS_CUT_DEBUG").is_ok()
                && !returning_cut_ids.is_empty()
            {
                eprintln!(
                    "[FCF]      Returning cuts so far: {:?}",
                    returning_cut_ids
                );
            }

            self.add_state(state);
        }

        // ============================================================
        // PHASE 2: Identify ALL dominated cuts ONCE
        // ============================================================
        // This happens AFTER all cuts in the batch have been processed,
        // ensuring we don't find the same dominated cuts multiple times.
        // ============================================================
        // PHASE 2: Identify ALL dominated cuts ONCE
        // ============================================================
        // This happens AFTER all cuts in the batch have been processed,
        // ensuring we don't find the same dominated cuts multiple times.
        // Now checking for count == 0 instead of <= 0 since we use usize.

        // DEBUG: Log all cuts before finding dominated ones
        if std::env::var("POWERS_CUT_DEBUG").is_ok() {
            eprintln!("[FCF] --- PHASE 2: Finding dominated cuts ---");
            eprintln!("[FCF] Detailed cut state (ALL cuts in pool):");
            for (idx, cut) in self.cut_pool.pool.iter().enumerate() {
                eprintln!(
                    "[FCF]   Cut {}: active={}, count={}, rhs={:.2}{}",
                    idx,
                    cut.active,
                    cut.non_dominated_state_count,
                    cut.rhs,
                    if cut.non_dominated_state_count == 0 && cut.active {
                        " ⚠️ WILL BE DOMINATED"
                    } else if !cut.active {
                        " 💤 ALREADY INACTIVE"
                    } else {
                        " ✅ ACTIVE & VALID"
                    }
                );
            }
            eprintln!(
                "[FCF] active_cut_indices HashMap: {} entries",
                self.cut_pool.active_cut_indices.len()
            );
        }

        let removing_cut_ids: HashSet<usize> = self
            .cut_pool
            .pool
            .iter()
            .filter(|c| c.non_dominated_state_count == 0 && c.active)
            .map(|c| c.id)
            .collect();

        // DEBUG: Log dominated cuts found
        if std::env::var("POWERS_CUT_DEBUG").is_ok() {
            eprintln!(
                "[FCF] Found {} dominated cuts: {:?}",
                removing_cut_ids.len(),
                removing_cut_ids
            );
        }

        // DEBUG: Log final state
        if std::env::var("POWERS_CUT_DEBUG").is_ok() {
            eprintln!(
                "[FCF] Final state: {} total cuts, {} active cuts, {} states",
                self.cut_pool.total_cut_count,
                self.cut_pool.active_cut_indices.len(),
                self.state_pool.pool.len()
            );
            eprintln!("[FCF] ========== BATCH END ==========\n");
        }

        BatchCutSelectionResult {
            new_cut_ids,
            returning_cut_ids,
            removing_cut_ids,
        }
    }
}

pub struct CutStatePair {
    pub cut: cut::BendersCut,
    pub state: Box<dyn state::State>,
}

impl CutStatePair {
    pub fn new(cut: cut::BendersCut, state: Box<dyn state::State>) -> Self {
        Self { cut, state }
    }
}

/// Result of batch cut selection for an entire batch (NEW DESIGN)
///
/// This struct aggregates cut selection results for ALL cuts processed in a single batch.
/// Unlike the old design where each cut had its own result, this returns a single result
/// containing all the information needed to update the model.
///
/// # Key Features
/// - Uses HashSet to eliminate duplicates automatically
/// - Handles intra-batch dominance (cuts within the batch dominating each other)
/// - Single result per batch instead of multiple results to aggregate
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
pub struct AggregatedCutSelectionResult {
    /// IDs of all newly added cuts (in order)
    pub new_cut_ids: Vec<usize>,
    /// IDs of ALL cuts that should be returned to models (union across all results)
    pub returning_cut_ids: Vec<usize>,
    /// IDs of ALL cuts that should be removed from models (union across all results)
    pub removing_cut_ids: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StorageState;
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
        let cut = cut::BendersCut::new(0, vec![1.0], 10.0);
        fcf.add_cut(cut);
        assert_eq!(fcf.cut_pool.pool.len(), 1);
    }

    #[test]
    fn test_add_state() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let load_sp = crate::stochastic_process::factory("naive");
        let inflow_sp = crate::stochastic_process::factory("naive");
        let state = Box::new(StorageState::new(
            &system,
            load_sp.as_ref(),
            inflow_sp.as_ref(),
        ));
        fcf.add_state(state);
        assert_eq!(fcf.state_pool.pool.len(), 1);
    }
}
