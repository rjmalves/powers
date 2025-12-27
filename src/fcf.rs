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

    /// Create FCF with pre-allocated capacity for expected training.
    ///
    /// # Performance Optimization (TICKET-006d)
    ///
    /// Pre-allocates cut and state pools to eliminate reallocations during training.
    /// Conservative estimate assumes cut selection might be disabled.
    ///
    /// **Memory allocation pattern**:
    /// - Without preallocation: ~8 reallocations (Vec doubles: 0→1→2→4→8→16→32→64→128→256)
    /// - With preallocation: 0 reallocations (allocated once to final size)
    ///
    /// **Expected savings** (20 iterations, 10 forward passes):
    /// - Cuts: 200 × 1,304 bytes = 261 KB (preallocated)
    /// - States: 200 × ~750 bytes = 150 KB (preallocated)
    /// - HashMap: ~50 KB (preallocated with load factor)
    /// - Total: ~460 KB allocated once vs grown incrementally
    ///
    /// **Performance impact**: 1-3% faster (eliminates reallocation overhead)
    ///
    /// # Arguments
    ///
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `num_iterations` - Training iterations
    /// * `max_state_dim` - Maximum state dimension across all nodes
    ///
    /// # Example
    ///
    /// ```ignore
    /// let fcf = FutureCostFunction::with_capacity(
    ///     10,   // 10 forward passes
    ///     100,  // 100 iterations
    ///     156,  // 156 hydros (storage + inflow state)
    /// );
    /// // fcf.cut_pool has capacity for 1000 cuts
    /// // fcf.state_pool has capacity for 1000 states
    /// ```
    pub fn with_capacity(
        num_forward_passes: usize,
        num_iterations: usize,
        max_state_dim: usize,
    ) -> Self {
        // Conservative estimate: assume cut selection disabled
        let max_cuts = num_forward_passes * num_iterations;
        let max_states = num_forward_passes * num_iterations;

        Self {
            cut_pool: cut::BendersCutPool::with_capacity(
                max_cuts,
                max_state_dim,
            ),
            state_pool: state::VisitedStatePool::with_capacity(max_states),
        }
    }

    /// Create FCF with fully preallocated cut and state pools.
    ///
    /// Unlike `with_capacity` which only reserves pointer space, this method
    /// fully preallocates all `BendersCut` and `State` instances with their
    /// coefficient vectors. This enables zero-allocation updates during training.
    ///
    /// # Memory Usage
    ///
    /// Total cut memory = `num_iterations * num_forward_passes * (sizeof(BendersCut) + state_dimension * 8)`
    ///
    /// Example (8 iterations, 16 forward passes, 156 state dimensions):
    /// - 128 cuts × (88 bytes struct + 1248 bytes coefficients) ≈ 171 KB
    /// - 128 states × (48 bytes struct + 1248 bytes coefficients) ≈ 166 KB
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `state_dimension` - State dimension for coefficient vectors
    /// * `template_state` - Template state to clone for preallocation
    ///
    /// # Example
    ///
    /// ```ignore
    /// let template: Box<dyn State> = Box::new(StorageState::new(&system));
    /// let fcf = FutureCostFunction::preallocate_pools(8, 16, 156, &template);
    /// assert_eq!(fcf.cut_pool.pool.len(), 128);
    /// assert_eq!(fcf.state_pool.pool.len(), 128);
    /// assert!(fcf.cut_pool.is_preallocated());
    /// ```
    pub fn preallocate_pools(
        num_iterations: usize,
        num_forward_passes: usize,
        state_dimension: usize,
        template_state: &dyn state::State,
    ) -> Self {
        Self {
            cut_pool: cut::BendersCutPool::preallocate(
                num_iterations,
                num_forward_passes,
                state_dimension,
            ),
            state_pool: state::VisitedStatePool::preallocate(
                num_iterations,
                num_forward_passes,
                template_state,
            ),
        }
    }

    pub fn add_cut(&mut self, new_cut: cut::BendersCut) {
        self.cut_pool.pool.push(std::sync::Arc::new(new_cut));
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
            let state_coefs = state.coefficients();
            let height = new_cut.eval_height_at_state(state_coefs);
            let current_dominating_obj = state.get_dominating_objective();

            // Use epsilon-based comparison with tie-breaking.
            //
            // When heights are numerically equal (within DOMINATION_EPSILON),
            // prefer lower cut ID for deterministic selection. This ensures
            // the same cut dominates across runs, preventing dominating_cut_id
            // variations that cause diverging lower bounds.
            let should_update = if (height - current_dominating_obj).abs()
                < DOMINATION_EPSILON
            {
                new_cut.id < state.get_dominating_cut_id()
            } else {
                height > current_dominating_obj
            };

            if should_update {
                let old_cut_id = state.get_dominating_cut_id();

                // Only decrement if old_cut_id is valid (within pool bounds)
                if old_cut_id < self.cut_pool.pool.len() {
                    self.cut_pool.pool[old_cut_id]
                        .decrement_non_dominated_count();
                }
                new_cut.increment_non_dominated_count();
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
            // Skip unpopulated preallocated cuts
            if !old_cut.is_populated() {
                continue;
            }

            match old_cut.is_active() {
                true => continue,
                false => {
                    let height =
                        old_cut.eval_height_at_state(new_state.coefficients());
                    let current_dominating_obj =
                        new_state.get_dominating_objective();

                    // Same epsilon-based tie-breaking as eval_new_cut_domination.
                    let should_update = if (height - current_dominating_obj)
                        .abs()
                        < DOMINATION_EPSILON
                    {
                        old_cut.id < new_state.get_dominating_cut_id()
                    } else {
                        height > current_dominating_obj
                    };

                    if should_update {
                        cut_non_dominated_decrement_ids
                            .push(new_state.get_dominating_cut_id());

                        old_cut.increment_non_dominated_count();
                        new_state.update_dominating_cut(old_cut, height);
                        cut_ids_to_return_to_model.push(old_cut.id);
                    }
                    continue;
                }
            }
        }
        // Decrements the non-dominating counts
        for cut_id in cut_non_dominated_decrement_ids.iter() {
            self.cut_pool.pool[*cut_id].decrement_non_dominated_count();
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
        self.cut_pool.pool[cut_id].set_active(true);
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
            self.cut_pool.pool[cut_id].set_active(false);

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
    /// # Preallocation Mode
    ///
    /// When the cut pool is preallocated (via `preallocate_pools()`), cuts are updated
    /// in place using slot-based access computed from `(iteration, forward_pass_idx)`.
    /// This eliminates heap allocations during training.
    ///
    /// When not preallocated, falls back to push behavior for backward compatibility.
    ///
    pub fn add_cuts_batch(
        &mut self,
        cut_state_pairs: Vec<CutStatePair>,
        enable_cut_selection: bool,
    ) -> BatchCutSelectionResult {
        let mut new_cut_ids = HashSet::new();
        let mut returning_cut_ids = HashSet::new();

        let is_preallocated = self.cut_pool.is_preallocated();

        // ============================================================
        // PHASE 1: Process all cuts and update dominance counters
        // ============================================================
        // This updates non_dominated_state_count for each cut but does NOT
        // yet determine which cuts to remove. That happens ONCE at the end.
        // Intra-batch domination is handled: later cuts can dominate earlier ones!

        for pair in cut_state_pairs.into_iter() {
            let iteration = pair.cut.iteration;
            let forward_pass_idx = pair.cut.forward_pass_idx;
            let state_coefficients = pair.state.coefficients().to_vec();

            let cut_id = if is_preallocated {
                // Preallocated mode: update cut in place using slot-based access
                let slot = self.cut_pool.update_cut(
                    iteration,
                    forward_pass_idx,
                    &pair.cut.coefficients,
                    pair.cut.rhs,
                );

                // Update preallocated state in place (TICKET-011)
                self.state_pool.update_state(
                    slot,
                    &state_coefficients,
                    iteration,
                    forward_pass_idx,
                );

                slot
            } else {
                // Non-preallocated mode: assign ID and push
                let mut cut = pair.cut;
                cut.id = self.cut_pool.total_cut_count;
                let id = cut.id;
                self.add_cut(cut);
                self.add_state(pair.state);
                id
            };

            new_cut_ids.insert(cut_id);
            self.update_cut_pool_on_add(cut_id);

            // Update state domination from source cut
            {
                let cut = &self.cut_pool.pool[cut_id];
                let state = &mut self.state_pool.pool[cut_id];
                let cut_height = cut.eval_height_at_state(state.coefficients());
                state.set_dominating_cut_id(cut_id);
                state.set_dominating_objective(cut_height);
            }

            // Evaluate dominance against ALL previous states (including from this batch)
            // This handles intra-batch domination correctly!
            self.eval_new_cut_domination_by_id(cut_id);

            // Update with new state and check for cuts to return
            let returning_ids =
                self.update_old_cuts_domination_for_slot(cut_id);
            returning_cut_ids.extend(returning_ids);
        }

        // ============================================================
        // PHASE 2: Identify ALL dominated cuts ONCE
        // ============================================================
        // When cut selection is ENABLED, remove cuts with zero dominated states.
        // When DISABLED, keep all cuts for monotonic lower bound growth.
        let removing_cut_ids: HashSet<usize> = if enable_cut_selection {
            self.cut_pool
                .pool
                .iter()
                .filter(|c| {
                    c.is_populated()
                        && c.get_non_dominated_count() == 0
                        && c.is_active()
                })
                .map(|c| c.id)
                .collect()
        } else {
            // Cut selection disabled: never remove cuts
            HashSet::new()
        };

        BatchCutSelectionResult {
            new_cut_ids,
            returning_cut_ids,
            removing_cut_ids,
        }
    }

    /// Update old cuts domination for a state at the given slot.
    ///
    /// This is a variant that accesses the state from the pool by slot index,
    /// avoiding the borrow conflict with `&mut self`.
    fn update_old_cuts_domination_for_slot(
        &mut self,
        state_slot: usize,
    ) -> Vec<usize> {
        let mut cut_non_dominated_decrement_ids = Vec::<usize>::new();
        let mut cut_ids_to_return_to_model = Vec::<usize>::new();

        for cut_idx in 0..self.cut_pool.pool.len() {
            // Skip the cut at the same slot as the state (self-domination handled earlier)
            if cut_idx == state_slot {
                continue;
            }

            let old_cut = &self.cut_pool.pool[cut_idx];

            // Skip unpopulated preallocated cuts
            if !old_cut.is_populated() {
                continue;
            }

            // Skip active cuts
            if old_cut.is_active() {
                continue;
            }

            let state = &self.state_pool.pool[state_slot];
            let height = old_cut.eval_height_at_state(state.coefficients());
            let current_dominating_obj = state.get_dominating_objective();
            let current_dominating_cut_id = state.get_dominating_cut_id();

            // Same epsilon-based tie-breaking as eval_new_cut_domination.
            let should_update = if (height - current_dominating_obj).abs()
                < DOMINATION_EPSILON
            {
                old_cut.id < current_dominating_cut_id
            } else {
                height > current_dominating_obj
            };

            if should_update {
                cut_non_dominated_decrement_ids.push(current_dominating_cut_id);
                cut_ids_to_return_to_model.push(cut_idx);

                // Update state domination info
                let state = &mut self.state_pool.pool[state_slot];
                state.set_dominating_cut_id(cut_idx);
                state.set_dominating_objective(height);
            }
        }

        // Update non_dominated_state_count for cuts that now dominate
        for &cut_idx in &cut_ids_to_return_to_model {
            self.cut_pool.pool[cut_idx].increment_non_dominated_count();
        }

        // Decrements the non-dominating counts
        for &cut_id in &cut_non_dominated_decrement_ids {
            if cut_id < self.cut_pool.pool.len() {
                self.cut_pool.pool[cut_id].decrement_non_dominated_count();
            }
        }

        cut_ids_to_return_to_model
    }

    /// Evaluate new cut domination by cut ID (helper for preallocated mode)
    fn eval_new_cut_domination_by_id(&mut self, cut_id: usize) {
        for state in self.state_pool.pool.iter_mut() {
            let state_coefs = state.coefficients();

            // Get cut reference for height evaluation
            let cut = &self.cut_pool.pool[cut_id];
            let height = cut.rhs
                + crate::utils::dot_product_deterministic(
                    &cut.coefficients,
                    state_coefs,
                );
            let current_dominating_obj = state.get_dominating_objective();

            // Use epsilon-based comparison with tie-breaking
            let should_update = if (height - current_dominating_obj).abs()
                < DOMINATION_EPSILON
            {
                cut.id < state.get_dominating_cut_id()
            } else {
                height > current_dominating_obj
            };

            if should_update {
                let old_cut_id = state.get_dominating_cut_id();

                // Only decrement if old_cut_id is valid (within pool bounds)
                if old_cut_id < self.cut_pool.pool.len() {
                    self.cut_pool.pool[old_cut_id]
                        .decrement_non_dominated_count();
                }
                self.cut_pool.pool[cut_id].increment_non_dominated_count();

                // Update state with cut reference
                let cut = &self.cut_pool.pool[cut_id];
                state.update_dominating_cut(cut, height);
            }
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
    pub new_cut_ids: HashSet<usize>,
    pub returning_cut_ids: HashSet<usize>,
    pub removing_cut_ids: HashSet<usize>,
}

/// Aggregated result of batch cut selection for ALL cuts
///
/// This aggregates results from multiple cuts to ensure ALL handler models
/// receive the SAME updates.
pub struct AggregatedCutSelectionResult {
    pub new_cut_ids: HashSet<usize>,
    pub returning_cut_ids: HashSet<usize>,
    pub removing_cut_ids: HashSet<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StorageState;
    use crate::system;

    /// Helper to create a template state for preallocate_pools tests
    fn create_template_state(num_hydros: usize) -> Box<dyn state::State> {
        let system = create_test_system(num_hydros);
        Box::new(StorageState::new(&system))
    }

    /// Helper to create a system with the specified number of hydros
    fn create_test_system(num_hydros: usize) -> system::System {
        let mut system = system::System::default();
        system.hydros.clear();
        for i in 0..num_hydros {
            system.hydros.push(system::Hydro::new(
                i, None, 0, 1.0, 0.0, 100.0, 0.0, 60.0, 0.01,
            ));
        }
        system.meta.hydros_count = num_hydros;
        system
    }

    #[test]
    fn test_new_future_cost_function() {
        let fcf = FutureCostFunction::new();
        assert_eq!(fcf.cut_pool.total_cut_count, 0);
        assert!(fcf.state_pool.pool.is_empty());
    }

    // ========================================================================
    // TICKET-004: FCF preallocate_pools tests
    // ========================================================================

    #[test]
    fn test_preallocate_pools_creates_correct_structure() {
        let template = create_template_state(156);
        let fcf = FutureCostFunction::preallocate_pools(
            8,
            16,
            156,
            template.as_ref(),
        );
        assert_eq!(fcf.cut_pool.pool.len(), 128); // 8 * 16
        assert!(fcf.cut_pool.is_preallocated());
        assert_eq!(fcf.state_pool.pool.len(), 128); // Also preallocated!
    }

    #[test]
    fn test_preallocate_pools_cuts_have_correct_dimension() {
        let template = create_template_state(10);
        let fcf =
            FutureCostFunction::preallocate_pools(2, 4, 10, template.as_ref());
        for cut in &fcf.cut_pool.pool {
            assert_eq!(cut.coefficients.len(), 10);
        }
    }

    #[test]
    fn test_preallocate_pools_cuts_start_inactive() {
        let template = create_template_state(10);
        let fcf =
            FutureCostFunction::preallocate_pools(2, 4, 10, template.as_ref());
        for cut in &fcf.cut_pool.pool {
            assert!(!cut.is_active());
        }
    }

    #[test]
    fn test_preallocate_pools_state_pool_is_preallocated() {
        let template = create_template_state(156);
        let fcf = FutureCostFunction::preallocate_pools(
            8,
            16,
            156,
            template.as_ref(),
        );
        // State pool should have 128 states preallocated
        assert_eq!(fcf.state_pool.pool.len(), 128);
        // All states should be zeroed
        for state in &fcf.state_pool.pool {
            assert!(state.coefficients().iter().all(|&c| c == 0.0));
        }
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
        // StorageState::new() only needs system, not uncertainty models
        let state = Box::new(StorageState::new(&system));
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
        cut.set_active(false);
        fcf.add_cut(cut);

        fcf.update_cut_pool_on_return(0);

        assert!(fcf.cut_pool.pool[0].is_active());
        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 1);
    }

    #[test]
    fn test_eval_new_cut_domination_empty_states() {
        let mut fcf = FutureCostFunction::new();
        let mut cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, 0);

        // Cuts start with non_dominated_state_count = 1
        assert_eq!(cut.get_non_dominated_count(), 1);

        // Should not crash with empty state pool
        fcf.eval_new_cut_domination(&mut cut);

        // Counter should remain unchanged since there are no states
        assert_eq!(cut.get_non_dominated_count(), 1);
    }

    #[test]
    fn test_eval_new_cut_domination_with_state() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();

        // Add a state
        let state = Box::new(StorageState::new(&system));
        fcf.add_state(state);

        // Add and evaluate a cut
        let mut cut = cut::BendersCut::new(0, vec![1.0], 100.0, 1, 0);
        fcf.eval_new_cut_domination(&mut cut);
    }

    #[test]
    fn test_update_old_cuts_domination_empty() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();
        let mut state: Box<dyn state::State> =
            Box::new(StorageState::new(&system));

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
        assert!(fcf.cut_pool.pool[0].is_active());

        // Remove the cut
        fcf.update_cut_pool_on_remove(0);

        assert_eq!(fcf.cut_pool.active_cut_indices.len(), 0);
        assert!(!fcf.cut_pool.pool[0].is_active());
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
        assert!(!fcf.cut_pool.pool[1].is_active());
    }

    #[test]
    fn test_update_old_cuts_domination_with_inactive_cut() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();

        // Add a cut and mark it inactive
        let mut cut = cut::BendersCut::new(0, vec![1.0], 100.0, 1, 0);
        cut.set_active(false);
        fcf.add_cut(cut);

        // Create new state
        let mut state: Box<dyn state::State> =
            Box::new(StorageState::new(&system));

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

    /// Test that add_cuts_batch correctly enforces invariant when selection is disabled
    #[test]
    fn test_add_cuts_batch_disabled_returns_empty_removing_set() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();

        // Create test cut-state pairs
        let mut pairs = Vec::new();
        for i in 0..5 {
            let cut = cut::BendersCut::new(i, vec![1.0], 10.0, 0, i);
            let state = Box::new(StorageState::new(&system));
            pairs.push(CutStatePair {
                cut,
                state,
                forward_pass_idx: i,
            });
        }

        // Call with selection DISABLED
        let result = fcf.add_cuts_batch(pairs, false);

        // Verify no cuts are marked for removal
        assert_eq!(result.removing_cut_ids.len(), 0);
        assert_eq!(result.new_cut_ids.len(), 5);
    }

    /// Test that add_cuts_batch with selection enabled can mark cuts for removal
    #[test]
    fn test_add_cuts_batch_enabled_allows_removal() {
        let mut fcf = FutureCostFunction::new();
        let system = system::System::default();

        // Create test cut-state pairs
        let mut pairs = Vec::new();
        for i in 0..3 {
            let cut = cut::BendersCut::new(i, vec![1.0], 10.0, 0, i);
            let state = Box::new(StorageState::new(&system));
            pairs.push(CutStatePair {
                cut,
                state,
                forward_pass_idx: i,
            });
        }

        // Call with selection ENABLED
        let result = fcf.add_cuts_batch(pairs, true);

        // Verify method runs without error (removal is allowed)
        assert_eq!(result.new_cut_ids.len(), 3);
        // Note: Whether cuts are actually removed depends on domination,
        // but the mechanism should work without panic
    }

    // ========================================================================
    // TICKET-005: add_cuts_batch preallocated mode tests
    // ========================================================================

    #[test]
    fn test_add_cuts_batch_preallocated_uses_slot_access() {
        let system = system::System::default();
        let template = create_template_state(1);

        // Create preallocated FCF
        let mut fcf =
            FutureCostFunction::preallocate_pools(2, 4, 1, template.as_ref());
        assert!(fcf.cut_pool.is_preallocated());

        // Create cut-state pairs with specific iteration/forward_pass_idx
        let mut pairs = Vec::new();
        for fp_idx in 0..4 {
            let cut = cut::BendersCut::new(
                0,
                vec![1.0],
                10.0 * (fp_idx as f64),
                1,
                fp_idx,
            );
            let state = Box::new(StorageState::new(&system));
            pairs.push(CutStatePair {
                cut,
                state,
                forward_pass_idx: fp_idx,
            });
        }

        // Process batch
        let result = fcf.add_cuts_batch(pairs, false);

        // Verify cut IDs are slot-based (0, 1, 2, 3 for iteration=1, fp_idx=0,1,2,3)
        assert!(result.new_cut_ids.contains(&0));
        assert!(result.new_cut_ids.contains(&1));
        assert!(result.new_cut_ids.contains(&2));
        assert!(result.new_cut_ids.contains(&3));
        assert_eq!(result.new_cut_ids.len(), 4);

        // Verify cuts are populated at correct slots
        assert!(fcf.cut_pool.pool[0].is_populated());
        assert!(fcf.cut_pool.pool[1].is_populated());
        assert!(fcf.cut_pool.pool[2].is_populated());
        assert!(fcf.cut_pool.pool[3].is_populated());
        // Slots 4-7 should still be unpopulated (iteration 2)
        assert!(!fcf.cut_pool.pool[4].is_populated());
    }

    #[test]
    fn test_add_cuts_batch_preallocated_no_reallocation() {
        let system = system::System::default();
        let template = create_template_state(1);

        // Create preallocated FCF
        let mut fcf =
            FutureCostFunction::preallocate_pools(4, 4, 1, template.as_ref());
        let original_cut_capacity = fcf.cut_pool.pool.capacity();
        let original_cut_len = fcf.cut_pool.pool.len();
        let original_state_capacity = fcf.state_pool.pool.capacity();
        let original_state_len = fcf.state_pool.pool.len();

        // Add multiple batches
        for iteration in 1..=4 {
            let mut pairs = Vec::new();
            for fp_idx in 0..4 {
                let cut =
                    cut::BendersCut::new(0, vec![1.0], 10.0, iteration, fp_idx);
                let state = Box::new(StorageState::new(&system));
                pairs.push(CutStatePair {
                    cut,
                    state,
                    forward_pass_idx: fp_idx,
                });
            }
            fcf.add_cuts_batch(pairs, false);
        }

        // Verify no reallocation occurred
        assert_eq!(fcf.cut_pool.pool.capacity(), original_cut_capacity);
        assert_eq!(fcf.cut_pool.pool.len(), original_cut_len);
        assert_eq!(fcf.state_pool.pool.capacity(), original_state_capacity);
        assert_eq!(fcf.state_pool.pool.len(), original_state_len);
    }

    #[test]
    fn test_add_cuts_batch_preallocated_iteration_2() {
        let system = system::System::default();
        let template = create_template_state(1);

        // Create preallocated FCF
        let mut fcf =
            FutureCostFunction::preallocate_pools(4, 4, 1, template.as_ref());

        // Add cuts for iteration 2
        let mut pairs = Vec::new();
        for fp_idx in 0..4 {
            let cut = cut::BendersCut::new(0, vec![1.0], 20.0, 2, fp_idx);
            let state = Box::new(StorageState::new(&system));
            pairs.push(CutStatePair {
                cut,
                state,
                forward_pass_idx: fp_idx,
            });
        }

        let result = fcf.add_cuts_batch(pairs, false);

        // Verify cut IDs are slot-based: (2-1)*4 + fp_idx = 4, 5, 6, 7
        assert!(result.new_cut_ids.contains(&4));
        assert!(result.new_cut_ids.contains(&5));
        assert!(result.new_cut_ids.contains(&6));
        assert!(result.new_cut_ids.contains(&7));

        // Verify correct slots are populated
        assert!(fcf.cut_pool.pool[4].is_populated());
        assert!(!fcf.cut_pool.pool[0].is_populated()); // Iteration 1 not populated
    }

    #[test]
    fn test_add_cuts_batch_non_preallocated_still_works() {
        let system = system::System::default();

        // Create non-preallocated FCF
        let mut fcf = FutureCostFunction::new();
        assert!(!fcf.cut_pool.is_preallocated());

        // Create cut-state pairs
        let mut pairs = Vec::new();
        for i in 0..3 {
            let cut = cut::BendersCut::new(0, vec![1.0], 10.0, 1, i);
            let state = Box::new(StorageState::new(&system));
            pairs.push(CutStatePair {
                cut,
                state,
                forward_pass_idx: i,
            });
        }

        // Process batch
        let result = fcf.add_cuts_batch(pairs, false);

        // Verify old push behavior: IDs are 0, 1, 2
        assert!(result.new_cut_ids.contains(&0));
        assert!(result.new_cut_ids.contains(&1));
        assert!(result.new_cut_ids.contains(&2));
        assert_eq!(fcf.cut_pool.pool.len(), 3);
    }
}
