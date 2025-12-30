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

pub struct FutureCostFunction {
    pub cut_pool: cut::BendersCutPool,
    pub state_pool: state::VisitedStatePool,
}

impl FutureCostFunction {
    /// Create a placeholder FCF that must be replaced before use.
    ///
    /// This is used during graph construction where FCFs are later replaced
    /// with preallocated versions via `preallocate_pools()`.
    ///
    /// # Warning
    ///
    /// This FCF is not functional. Any operation requiring preallocated pools
    /// will panic. Always call `preallocate_pools()` before training.
    pub(crate) fn placeholder() -> Self {
        Self {
            cut_pool: cut::BendersCutPool::with_capacity(0, 0),
            state_pool: state::VisitedStatePool::with_capacity(0),
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

    /// Update cut pool state when a cut is added.
    ///
    /// With preallocation, the cut is already in place. This just updates the count
    /// and marks the cut as active.
    pub fn update_cut_pool_on_add(&mut self, cut_id: usize) {
        self.cut_pool.pool[cut_id].set_active(true);
        self.cut_pool.total_cut_count =
            self.cut_pool.total_cut_count.max(cut_id + 1);
    }

    /// Update cut pool state when a cut returns to model.
    ///
    /// With preallocation, just mark the cut as active.
    pub fn update_cut_pool_on_return(&mut self, cut_id: usize) {
        self.cut_pool.pool[cut_id].set_active(true);
    }

    /// Update cut pool state when a cut is removed.
    ///
    /// With preallocation, cuts are never actually removed from the model -
    /// they're deactivated via bound relaxation. This just marks the cut inactive.
    pub fn update_cut_pool_on_remove(&mut self, cut_id: usize) {
        self.cut_pool.pool[cut_id].set_active(false);
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
    /// # Panics
    ///
    /// Panics if the cut pool is not preallocated. Use `preallocate_pools()` before calling.
    ///
    pub fn add_cuts_batch(
        &mut self,
        cut_state_pairs: Vec<CutStatePair>,
        enable_cut_selection: bool,
    ) -> BatchCutSelectionResult {
        assert!(
            self.cut_pool.is_preallocated(),
            "add_cuts_batch requires preallocated pools. Use FutureCostFunction::preallocate_pools()"
        );

        let mut new_cut_ids = HashSet::new();
        let mut returning_cut_ids = HashSet::new();

        // ============================================================
        // PHASE 1: Process all cuts and update dominance counters
        // ============================================================
        // This updates non_dominated_state_count for each cut but does NOT
        // yet determine which cuts to remove. That happens ONCE at the end.
        // Intra-batch domination is handled: later cuts can dominate earlier ones!

        for pair in cut_state_pairs.into_iter() {
            let iteration = pair.cut.iteration;
            let forward_pass_idx = pair.cut.forward_pass_idx;

            // Preallocated mode: update cut in place using slot-based access
            let slot = self.cut_pool.update_cut(
                iteration,
                forward_pass_idx,
                &pair.cut.coefficients,
                pair.cut.rhs,
            );

            // Update preallocated state in place
            // PERF: Pass slice directly, avoid to_vec() allocation
            self.state_pool.update_state(
                slot,
                pair.state.coefficients(),
                iteration,
                forward_pass_idx,
            );

            let cut_id = slot;

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

    /// Finalize a cut at a slot after it has been updated via compute_cut_into_slot.
    ///
    /// # Zero Allocation
    ///
    /// This method is designed to work with `compute_cut_into_slot` which updates
    /// the cut and state pools directly. After the pools are updated, this method
    /// runs the domination evaluation for the new cut.
    ///
    /// # Arguments
    ///
    /// * `slot` - The slot index that was updated (returned by compute_cut_into_slot)
    ///
    /// # Returns
    ///
    /// Set of cut IDs that may need to be returned to the model due to domination changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Update pools directly with zero allocation
    /// let slot = state.compute_cut_into_slot(
    ///     risk_measure,
    ///     &realizations,
    ///     &mut fcf.cut_pool,
    ///     &mut fcf.state_pool,
    ///     iteration,
    ///     forward_pass_idx,
    /// );
    ///
    /// // Run domination evaluation for the slot
    /// let returning_ids = fcf.finalize_cut_at_slot(slot);
    /// ```
    #[inline]
    pub fn finalize_cut_at_slot(&mut self, slot: usize) -> HashSet<usize> {
        // Track this cut as a new cut
        self.update_cut_pool_on_add(slot);

        // Update state domination from source cut
        {
            let cut = &self.cut_pool.pool[slot];
            let state = &mut self.state_pool.pool[slot];
            let cut_height = cut.eval_height_at_state(state.coefficients());
            state.set_dominating_cut_id(slot);
            state.set_dominating_objective(cut_height);
        }

        // Evaluate dominance against ALL previous states
        self.eval_new_cut_domination_by_id(slot);

        // Update with new state and check for cuts to return
        let returning_ids = self.update_old_cuts_domination_for_slot(slot);

        returning_ids.into_iter().collect()
    }

    /// Finalize a batch of cuts at slots after they have been updated.
    ///
    /// # Zero Allocation Path
    ///
    /// Use this method after calling `compute_cut_into_slot` for each cut in a batch.
    /// Slots must be sorted by forward_pass_idx for deterministic ordering.
    ///
    /// # Arguments
    ///
    /// * `slots` - Sorted slice of slot indices that were updated
    /// * `enable_cut_selection` - Whether to identify dominated cuts for removal
    ///
    /// # Returns
    ///
    /// BatchCutSelectionResult with new, returning, and removing cut IDs.
    pub fn finalize_cuts_batch(
        &mut self,
        slots: &[usize],
        enable_cut_selection: bool,
    ) -> BatchCutSelectionResult {
        let mut new_cut_ids = HashSet::new();
        let mut returning_cut_ids = HashSet::new();

        for &slot in slots {
            new_cut_ids.insert(slot);

            // Same logic as add_cuts_batch_from_data, but slot is already updated
            self.update_cut_pool_on_add(slot);

            // Update state domination from source cut
            {
                let cut = &self.cut_pool.pool[slot];
                let state = &mut self.state_pool.pool[slot];
                let cut_height = cut.eval_height_at_state(state.coefficients());
                state.set_dominating_cut_id(slot);
                state.set_dominating_objective(cut_height);
            }

            // Evaluate dominance against ALL previous states
            self.eval_new_cut_domination_by_id(slot);

            // Update with new state and check for cuts to return
            let returning_ids = self.update_old_cuts_domination_for_slot(slot);
            returning_cut_ids.extend(returning_ids);
        }

        // Identify ALL dominated cuts ONCE
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
            HashSet::new()
        };

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

    // =========================================================================
    // T-054: Zero-allocation path verification tests
    // =========================================================================

    #[test]
    fn test_finalize_cut_at_slot() {
        let template = create_template_state(2);

        // Create preallocated FCF
        let mut fcf =
            FutureCostFunction::preallocate_pools(2, 4, 2, template.as_ref());
        assert!(fcf.cut_pool.is_preallocated());

        // Directly update cut slot (simulating what compute_cut_into_slot does)
        let slot = fcf.cut_pool.update_cut_and_state_slots(
            1,
            2,
            &[1.0, 2.0],
            100.0,
            &[10.0, 20.0],
            &mut fcf.state_pool,
        );

        // Finalize the cut
        let returning_ids = fcf.finalize_cut_at_slot(slot);

        // Verify cut is populated and active
        assert!(fcf.cut_pool.pool[slot].is_populated());
        assert!(fcf.cut_pool.pool[slot].is_active());
        assert_eq!(slot, 2); // (1-1)*4 + 2 = 2

        // Verify state domination was set
        let state = &fcf.state_pool.pool[slot];
        assert_eq!(state.get_dominating_cut_id(), slot);

        // No returning cuts expected for first cut
        assert!(returning_ids.is_empty());
    }

    #[test]
    fn test_finalize_cuts_batch() {
        let template = create_template_state(1);

        // Create preallocated FCF
        let mut fcf =
            FutureCostFunction::preallocate_pools(2, 4, 1, template.as_ref());

        // Directly update multiple cut slots (simulating parallel compute_cut_into_slot)
        let mut slots = Vec::new();
        for fp_idx in 0..4 {
            let slot = fcf.cut_pool.update_cut_and_state_slots(
                1,
                fp_idx,
                &[(fp_idx + 1) as f64],
                (fp_idx * 10) as f64,
                &[(fp_idx * 5) as f64],
                &mut fcf.state_pool,
            );
            slots.push(slot);
        }

        // Sort for deterministic ordering
        slots.sort_unstable();

        // Finalize all cuts in batch
        let result = fcf.finalize_cuts_batch(&slots, false);

        // Verify all cuts are in new_cut_ids
        assert_eq!(result.new_cut_ids.len(), 4);
        for &slot in &slots {
            assert!(result.new_cut_ids.contains(&slot));
        }

        // Verify all cuts are populated
        for &slot in &slots {
            assert!(fcf.cut_pool.pool[slot].is_populated());
            assert!(fcf.cut_pool.pool[slot].is_active());
        }
    }
}
