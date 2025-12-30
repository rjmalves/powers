use crate::utils;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Compute slot index for (iteration, forward_pass_idx) pair.
///
/// The slot is computed as: `(iteration - 1) * num_forward_passes + forward_pass_idx`
///
/// # Arguments
///
/// * `iteration` - Current iteration number (1-based, range: 1..=num_iterations)
/// * `forward_pass_idx` - Forward pass index (0-based, range: 0..num_forward_passes)
/// * `num_forward_passes` - Total forward passes per iteration
///
/// # Returns
///
/// Slot index in the preallocated pool (0-based)
///
/// # Example
///
/// ```ignore
/// // With 16 forward passes per iteration:
/// assert_eq!(compute_slot(1, 0, 16), 0);   // First slot
/// assert_eq!(compute_slot(1, 15, 16), 15); // End of first iteration
/// assert_eq!(compute_slot(2, 0, 16), 16);  // Start of second iteration
/// assert_eq!(compute_slot(8, 15, 16), 127); // Last slot for 8 iterations
/// ```
#[inline]
pub fn compute_slot(
    iteration: usize,
    forward_pass_idx: usize,
    num_forward_passes: usize,
) -> usize {
    debug_assert!(iteration >= 1, "iteration must be 1-based, got 0");
    debug_assert!(
        forward_pass_idx < num_forward_passes,
        "forward_pass_idx {} out of bounds for num_forward_passes {}",
        forward_pass_idx,
        num_forward_passes
    );
    (iteration - 1) * num_forward_passes + forward_pass_idx
}

#[derive(Debug)]
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    /// Whether cut is active in the model. Uses atomic for Arc-based sharing.
    active: AtomicBool,
    /// Count of states dominated by this cut. Uses atomic for Arc-based sharing.
    non_dominated_state_count: AtomicUsize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    /// Preallocated slot index in HiGHS model (0-based).
    /// Set when cut is added via preallocation. Used for O(1) deactivation.
    /// Uses AtomicUsize with usize::MAX as sentinel for "None".
    slot_index: AtomicUsize,
    /// Whether this cut has been populated with data.
    /// Preallocated cuts start with `populated = false` and are set to `true`
    /// after the first `update()` call. Cuts created via `new()` are immediately populated.
    /// Used to skip unpopulated preallocated cuts in domination evaluation.
    populated: bool,
}

/// Lightweight result from cut evaluation.
///
/// Holds references to computed cut data without allocation.
/// Used for direct transfer to preallocated cut pool slots.
///
/// # Lifetime
///
/// The `'a` lifetime is tied to the thread-local `CutComputationBuffers`.
/// This struct should not outlive the `with_cut_buffers` closure.
#[derive(Debug)]
pub struct CutEvalResult<'a> {
    /// Reference to computed cut coefficients (water values, lag duals)
    pub coefficients: &'a [f64],
    /// Computed RHS: objective - dot_product(coefficients, state_coefficients)
    pub rhs: f64,
    /// Iteration number (1-based)
    pub iteration: usize,
    /// Forward pass index (0-based)
    pub forward_pass_idx: usize,
}

impl<'a> CutEvalResult<'a> {
    /// Create a new cut evaluation result.
    #[inline]
    pub fn new(
        coefficients: &'a [f64],
        rhs: f64,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> Self {
        Self {
            coefficients,
            rhs,
            iteration,
            forward_pass_idx,
        }
    }
}

/// Sentinel value for slot_index indicating "no slot assigned"
const SLOT_INDEX_NONE: usize = usize::MAX;

impl Clone for BendersCut {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            coefficients: self.coefficients.clone(),
            rhs: self.rhs,
            active: AtomicBool::new(self.active.load(Ordering::Relaxed)),
            non_dominated_state_count: AtomicUsize::new(
                self.non_dominated_state_count.load(Ordering::Relaxed),
            ),
            iteration: self.iteration,
            forward_pass_idx: self.forward_pass_idx,
            slot_index: AtomicUsize::new(
                self.slot_index.load(Ordering::Relaxed),
            ),
            populated: self.populated,
        }
    }
}

impl BendersCut {
    pub fn new(
        id: usize,
        coefficients: Vec<f64>,
        rhs: f64,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> Self {
        Self {
            id,
            coefficients,
            rhs,
            active: AtomicBool::new(true),
            non_dominated_state_count: AtomicUsize::new(1),
            iteration,
            forward_pass_idx,
            slot_index: AtomicUsize::new(SLOT_INDEX_NONE),
            populated: true, // Cuts created via new() are immediately populated
        }
    }

    /// Check if cut is active in the model.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Set cut active status.
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Get non-dominated state count.
    #[inline]
    pub fn get_non_dominated_count(&self) -> usize {
        self.non_dominated_state_count.load(Ordering::Relaxed)
    }

    /// Increment non-dominated state count.
    #[inline]
    pub fn increment_non_dominated_count(&self) {
        self.non_dominated_state_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement non-dominated state count (saturating at 0).
    #[inline]
    pub fn decrement_non_dominated_count(&self) {
        // Use fetch_update for saturating subtraction
        let _ = self.non_dominated_state_count.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |x| Some(x.saturating_sub(1)),
        );
    }

    /// Reset non-dominated state count to a specific value.
    #[inline]
    pub fn set_non_dominated_count(&self, count: usize) {
        self.non_dominated_state_count
            .store(count, Ordering::Relaxed);
    }

    /// Get slot index if set.
    #[inline]
    pub fn get_slot_index(&self) -> Option<usize> {
        let val = self.slot_index.load(Ordering::Relaxed);
        if val == SLOT_INDEX_NONE {
            None
        } else {
            Some(val)
        }
    }

    /// Set slot index.
    #[inline]
    pub fn set_slot_index(&self, slot: usize) {
        self.slot_index.store(slot, Ordering::Relaxed);
    }

    /// Check if this cut has been populated with data.
    ///
    /// Preallocated cuts start unpopulated and become populated after `update()`.
    /// Cuts created via `new()` are immediately populated.
    #[inline]
    pub fn is_populated(&self) -> bool {
        self.populated
    }

    /// Update cut coefficients and RHS in place without allocation.
    ///
    /// This method is used with preallocated cuts to avoid heap allocations
    /// during training. The coefficient vector must have the same dimension
    /// as the preallocated one.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - New coefficient values (must match preallocated length)
    /// * `rhs` - New RHS value
    /// * `iteration` - Iteration that created this cut
    /// * `forward_pass_idx` - Forward pass that created this cut
    ///
    /// # Panics
    ///
    /// Panics in debug mode if coefficient dimensions don't match.
    #[inline]
    pub fn update(
        &mut self,
        coefficients: &[f64],
        rhs: f64,
        iteration: usize,
        forward_pass_idx: usize,
    ) {
        debug_assert_eq!(
            self.coefficients.len(),
            coefficients.len(),
            "coefficient dimension mismatch: expected {}, got {}",
            self.coefficients.len(),
            coefficients.len()
        );
        self.coefficients.copy_from_slice(coefficients);
        self.rhs = rhs;
        self.iteration = iteration;
        self.forward_pass_idx = forward_pass_idx;
        self.set_active(true);
        self.set_non_dominated_count(1);
        self.populated = true;
    }

    pub fn eval_height_at_state(&self, state_coefficients: &[f64]) -> f64 {
        // Use deterministic dot product for domination evaluation.
        //
        // Standard dot product allows compiler to reorder operations (e.g., with FMA
        // instructions), causing different heights across runs even with identical
        // inputs. This leads to:
        //   - Different DominatingObjective values → different dominating_cut_id
        //   - Diverging lower bounds even with identical cut coefficients
        let dot = utils::dot_product_deterministic(
            &self.coefficients,
            state_coefficients,
        );

        self.rhs + dot
    }
}

#[derive(Debug)]
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
    /// Maps cut_id → index in solver model constraints.
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
    /// Number of forward passes per iteration (for slot computation).
    /// Zero if not using preallocated mode.
    num_forward_passes: usize,
}

impl BendersCutPool {
    /// Create cut pool with pre-allocated capacity.
    ///
    /// # Performance Optimization (TICKET-006d)
    ///
    /// Pre-allocates Vec and HashMap to avoid reallocations during training.
    ///
    /// **Memory pattern**:
    /// - `pool`: Pre-allocated to `num_cuts` capacity
    /// - `active_cut_indices`: Pre-allocated with 33% extra for HashMap load factor (~75%)
    ///
    /// **Expected behavior** (200 cuts):
    /// - Without preallocation: ~8 Vec reallocations + ~8 HashMap rehashes
    /// - With preallocation: 0 reallocations, 0 rehashes
    ///
    /// # Arguments
    ///
    /// * `num_cuts` - Expected number of cuts (num_forward_passes × num_iterations)
    /// * `state_dim` - State dimension (used for coefficient vector sizing)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pool = BendersCutPool::with_capacity(
    ///     200,  // 20 iterations × 10 forward passes
    ///     156,  // 156 state dimensions
    /// );
    /// ```
    pub fn with_capacity(num_cuts: usize, _state_dim: usize) -> Self {
        Self {
            pool: Vec::with_capacity(num_cuts),
            // HashMap load factor ~75%, reserve 33% extra buckets to minimize rehashing
            active_cut_indices: HashMap::with_capacity(num_cuts),
            total_cut_count: 0,
            num_forward_passes: 0,
        }
    }

    /// Create cut pool with fully preallocated cuts.
    ///
    /// Unlike `with_capacity` which only reserves pointer space, this method
    /// preallocates all `BendersCut` instances with their coefficient vectors.
    /// This enables zero-allocation cut updates during training.
    ///
    /// # Memory Usage
    ///
    /// Total memory = `num_iterations * num_forward_passes * (sizeof(BendersCut) + state_dimension * 8)`
    ///
    /// Example (8 iterations, 16 forward passes, 156 state dimensions):
    /// - 128 cuts × (88 bytes struct + 1248 bytes coefficients) ≈ 171 KB
    ///
    /// # Arguments
    ///
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `state_dimension` - State dimension for coefficient vectors
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pool = BendersCutPool::preallocate(
    ///     8,    // 8 iterations
    ///     16,   // 16 forward passes
    ///     156,  // 156 state dimensions
    /// );
    /// assert_eq!(pool.pool.len(), 128);
    /// assert_eq!(pool.pool[0].coefficients.len(), 156);
    /// ```
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        state_dimension: usize,
    ) -> Self {
        let total_cuts = num_iterations * num_forward_passes;

        let pool: Vec<Arc<BendersCut>> = (0..total_cuts)
            .map(|id| {
                Arc::new(BendersCut {
                    id,
                    coefficients: vec![0.0; state_dimension],
                    rhs: 0.0,
                    active: AtomicBool::new(false),
                    non_dominated_state_count: AtomicUsize::new(0),
                    iteration: 0,
                    forward_pass_idx: 0,
                    slot_index: AtomicUsize::new(SLOT_INDEX_NONE),
                    populated: false, // Preallocated cuts start unpopulated
                })
            })
            .collect();

        Self {
            pool,
            active_cut_indices: HashMap::with_capacity(total_cuts),
            total_cut_count: 0,
            num_forward_passes,
        }
    }

    /// Update cut at slot computed from (iteration, forward_pass_idx).
    ///
    /// This method computes the slot index from the iteration and forward pass
    /// coordinates, then updates the preallocated cut in place.
    ///
    /// # Returns
    ///
    /// The slot index (which is also the cut_id).
    ///
    /// # Panics
    ///
    /// Panics if the pool was not created with `preallocate()` (num_forward_passes == 0),
    /// or if there are multiple references to the Arc (should not happen during batch update).
    pub fn update_cut(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        coefficients: &[f64],
        rhs: f64,
    ) -> usize {
        debug_assert!(
            self.num_forward_passes > 0,
            "update_cut requires preallocated pool"
        );

        let slot =
            compute_slot(iteration, forward_pass_idx, self.num_forward_passes);

        // Get mutable access to the Arc contents
        // This succeeds when there's only one reference (during batch update)
        let cut = Arc::get_mut(&mut self.pool[slot])
            .expect("Cannot mutate cut with multiple Arc references");
        cut.update(coefficients, rhs, iteration, forward_pass_idx);

        if slot >= self.total_cut_count {
            self.total_cut_count = slot + 1;
        }

        slot
    }

    /// Check if pool was created with preallocate().
    #[inline]
    pub fn is_preallocated(&self) -> bool {
        self.num_forward_passes > 0
    }

    /// Update cut and state slots atomically from buffer references.
    ///
    /// # Zero Allocation
    ///
    /// This method performs no heap allocation. It copies directly from
    /// the provided slices into preallocated pool slots using `copy_from_slice`.
    ///
    /// # Arguments
    ///
    /// * `iteration` - Training iteration (1-based)
    /// * `forward_pass_idx` - Forward pass index (0-based)
    /// * `cut_coefficients` - Slice from thread-local buffer (must match preallocated dimension)
    /// * `cut_rhs` - Computed RHS value
    /// * `state_coefficients` - State coefficients slice (must match preallocated dimension)
    /// * `state_pool` - Mutable reference to state pool to update
    ///
    /// # Returns
    ///
    /// The slot index that was updated (also the cut_id).
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Pool was not created with `preallocate()` (num_forward_passes == 0)
    /// - There are multiple Arc references to the cut (should not happen during batch update)
    /// - Coefficient slice lengths don't match preallocated dimensions
    ///
    /// # Example
    ///
    /// ```ignore
    /// let slot = cut_pool.update_cut_and_state_slots(
    ///     iteration,
    ///     forward_pass_idx,
    ///     &cut_coefficients,
    ///     cut_rhs,
    ///     state.coefficients(),
    ///     &mut state_pool,
    /// );
    /// ```
    #[inline]
    pub fn update_cut_and_state_slots(
        &mut self,
        iteration: usize,
        forward_pass_idx: usize,
        cut_coefficients: &[f64],
        cut_rhs: f64,
        state_coefficients: &[f64],
        state_pool: &mut crate::state::VisitedStatePool,
    ) -> usize {
        debug_assert!(
            self.num_forward_passes > 0,
            "update_cut_and_state_slots requires preallocated pool"
        );

        let slot =
            compute_slot(iteration, forward_pass_idx, self.num_forward_passes);

        // Update cut slot - no allocation, direct copy
        let cut = Arc::get_mut(&mut self.pool[slot])
            .expect("Cannot mutate cut with multiple Arc references");
        cut.update(cut_coefficients, cut_rhs, iteration, forward_pass_idx);

        if slot >= self.total_cut_count {
            self.total_cut_count = slot + 1;
        }

        // Update state slot - no allocation, uses update_coefficients which does copy_from_slice
        state_pool.update_state(
            slot,
            state_coefficients,
            iteration,
            forward_pass_idx,
        );

        slot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_benders_cut() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0, 1, 0);
        assert_eq!(cut.id, 1);
        assert_eq!(cut.coefficients, vec![1.0, 2.0]);
        assert_eq!(cut.rhs, 10.0);
        assert!(cut.is_active());
        assert_eq!(cut.get_non_dominated_count(), 1);
        assert_eq!(cut.iteration, 1);
        assert_eq!(cut.forward_pass_idx, 0);
        assert_eq!(cut.get_slot_index(), None);
    }

    #[test]
    fn test_cut_eval_result_creation() {
        let coefficients = vec![1.0, 2.0, 3.0];
        let result = CutEvalResult::new(&coefficients, 100.0, 1, 0);
        assert_eq!(result.coefficients.len(), 3);
        assert_eq!(result.coefficients, &[1.0, 2.0, 3.0]);
        assert_eq!(result.rhs, 100.0);
        assert_eq!(result.iteration, 1);
        assert_eq!(result.forward_pass_idx, 0);
    }

    #[test]
    fn test_eval_height_at_state() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0, 1, 0);
        let state_coeffs = vec![3.0, 4.0];
        // 10.0 + (1.0 * 3.0 + 2.0 * 4.0) = 10.0 + 3.0 + 8.0 = 21.0
        assert_eq!(cut.eval_height_at_state(&state_coeffs), 21.0);
    }

    // ========================================================================
    // TICKET-001: compute_slot tests
    // ========================================================================

    #[test]
    fn test_compute_slot_first_slot() {
        assert_eq!(compute_slot(1, 0, 16), 0);
    }

    #[test]
    fn test_compute_slot_end_of_first_iteration() {
        assert_eq!(compute_slot(1, 15, 16), 15);
    }

    #[test]
    fn test_compute_slot_start_of_second_iteration() {
        assert_eq!(compute_slot(2, 0, 16), 16);
    }

    #[test]
    fn test_compute_slot_last_slot_for_8_iterations() {
        assert_eq!(compute_slot(8, 15, 16), 127);
    }

    #[test]
    fn test_compute_slot_single_forward_pass() {
        assert_eq!(compute_slot(1, 0, 1), 0);
        assert_eq!(compute_slot(5, 0, 1), 4);
    }

    #[test]
    #[should_panic(expected = "iteration must be 1-based")]
    fn test_compute_slot_zero_iteration_panics() {
        compute_slot(0, 0, 16);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_compute_slot_forward_pass_out_of_bounds_panics() {
        compute_slot(1, 16, 16);
    }

    // ========================================================================
    // TICKET-002: BendersCut::update tests
    // ========================================================================

    #[test]
    fn test_benders_cut_update_modifies_all_fields() {
        let mut cut = BendersCut::new(0, vec![0.0, 0.0, 0.0], 0.0, 0, 0);
        cut.set_active(false);
        cut.set_non_dominated_count(5);

        cut.update(&[1.0, 2.0, 3.0], 100.0, 5, 7);

        assert_eq!(cut.coefficients, vec![1.0, 2.0, 3.0]);
        assert_eq!(cut.rhs, 100.0);
        assert_eq!(cut.iteration, 5);
        assert_eq!(cut.forward_pass_idx, 7);
        assert!(cut.is_active());
        assert_eq!(cut.get_non_dominated_count(), 1);
    }

    #[test]
    fn test_benders_cut_update_preserves_id() {
        let mut cut = BendersCut::new(42, vec![0.0; 10], 0.0, 0, 0);
        cut.update(&[1.0; 10], 50.0, 3, 2);
        assert_eq!(cut.id, 42);
    }

    #[test]
    fn test_benders_cut_update_no_reallocation() {
        let mut cut = BendersCut::new(0, vec![0.0; 156], 0.0, 0, 0);
        let original_capacity = cut.coefficients.capacity();

        cut.update(&[1.0; 156], 100.0, 1, 0);

        assert_eq!(cut.coefficients.capacity(), original_capacity);
    }

    #[test]
    fn test_benders_cut_update_various_dimensions() {
        for dim in [1, 10, 156] {
            let mut cut = BendersCut::new(0, vec![0.0; dim], 0.0, 0, 0);
            let new_coeffs: Vec<f64> = (0..dim).map(|i| i as f64).collect();
            cut.update(&new_coeffs, 50.0, 2, 3);
            assert_eq!(cut.coefficients, new_coeffs);
        }
    }

    #[test]
    #[should_panic(expected = "coefficient dimension mismatch")]
    fn test_benders_cut_update_dimension_mismatch_panics() {
        let mut cut = BendersCut::new(0, vec![0.0; 10], 0.0, 0, 0);
        cut.update(&[1.0; 5], 50.0, 1, 0);
    }

    // ========================================================================
    // TICKET-003: BendersCutPool::preallocate tests
    // ========================================================================

    #[test]
    fn test_preallocate_creates_correct_number_of_cuts() {
        let pool = BendersCutPool::preallocate(8, 16, 156);
        assert_eq!(pool.pool.len(), 128);
    }

    #[test]
    fn test_preallocate_each_cut_has_correct_dimension() {
        let pool = BendersCutPool::preallocate(2, 4, 10);
        for cut in &pool.pool {
            assert_eq!(cut.coefficients.len(), 10);
        }
    }

    #[test]
    fn test_preallocate_cuts_start_inactive() {
        let pool = BendersCutPool::preallocate(2, 4, 10);
        for cut in &pool.pool {
            assert!(!cut.is_active());
            assert_eq!(cut.get_non_dominated_count(), 0);
        }
    }

    #[test]
    fn test_preallocate_cuts_have_sequential_ids() {
        let pool = BendersCutPool::preallocate(2, 4, 10);
        for (expected_id, cut) in pool.pool.iter().enumerate() {
            assert_eq!(cut.id, expected_id);
        }
    }

    #[test]
    fn test_preallocate_is_preallocated_true() {
        let pool = BendersCutPool::preallocate(2, 4, 10);
        assert!(pool.is_preallocated());
    }

    #[test]
    fn test_with_capacity_is_preallocated_false() {
        let pool = BendersCutPool::with_capacity(100, 10);
        assert!(!pool.is_preallocated());
    }

    #[test]
    fn test_update_cut_correctly_computes_slot() {
        let mut pool = BendersCutPool::preallocate(4, 8, 3);

        // Update cut at (1, 0) -> slot 0
        let slot = pool.update_cut(1, 0, &[1.0, 2.0, 3.0], 100.0);
        assert_eq!(slot, 0);
        assert_eq!(pool.pool[0].coefficients, vec![1.0, 2.0, 3.0]);
        assert_eq!(pool.pool[0].rhs, 100.0);
        assert!(pool.pool[0].is_active());

        // Update cut at (2, 3) -> slot 11
        let slot = pool.update_cut(2, 3, &[4.0, 5.0, 6.0], 200.0);
        assert_eq!(slot, 11);
        assert_eq!(pool.pool[11].coefficients, vec![4.0, 5.0, 6.0]);
        assert_eq!(pool.pool[11].rhs, 200.0);
    }

    #[test]
    fn test_update_cut_updates_total_cut_count() {
        let mut pool = BendersCutPool::preallocate(4, 8, 3);
        assert_eq!(pool.total_cut_count, 0);

        pool.update_cut(1, 0, &[1.0, 2.0, 3.0], 100.0);
        assert_eq!(pool.total_cut_count, 1);

        // Updating at higher slot increases count
        pool.update_cut(2, 5, &[1.0, 2.0, 3.0], 100.0);
        assert_eq!(pool.total_cut_count, 14); // slot 13 + 1

        // Updating at lower slot doesn't decrease count
        pool.update_cut(1, 3, &[1.0, 2.0, 3.0], 100.0);
        assert_eq!(pool.total_cut_count, 14);
    }

    #[test]
    fn test_preallocate_coefficients_initialized_to_zero() {
        let pool = BendersCutPool::preallocate(2, 4, 5);
        for cut in &pool.pool {
            assert!(cut.coefficients.iter().all(|&c| c == 0.0));
            assert_eq!(cut.rhs, 0.0);
        }
    }

    // ========================================================================
    // TICKET-006: populated flag tests
    // ========================================================================

    #[test]
    fn test_preallocated_cuts_start_unpopulated() {
        let pool = BendersCutPool::preallocate(2, 4, 5);
        for cut in &pool.pool {
            assert!(!cut.is_populated());
        }
    }

    #[test]
    fn test_update_sets_populated_true() {
        let mut pool = BendersCutPool::preallocate(2, 4, 3);
        assert!(!pool.pool[0].is_populated());

        pool.update_cut(1, 0, &[1.0, 2.0, 3.0], 100.0);
        assert!(pool.pool[0].is_populated());
    }

    #[test]
    fn test_new_cuts_are_populated() {
        let cut = BendersCut::new(0, vec![1.0, 2.0], 10.0, 1, 0);
        assert!(cut.is_populated());
    }

    #[test]
    fn test_unpopulated_cuts_remain_unpopulated_until_update() {
        let pool = BendersCutPool::preallocate(4, 8, 3);
        // Only slot 0 and 15 are populated
        assert!(!pool.pool[5].is_populated());
        assert!(!pool.pool[10].is_populated());
        assert!(!pool.pool[31].is_populated());
    }

    // ========================================================================
    // T-050: update_cut_and_state_slots tests
    // ========================================================================

    #[test]
    fn test_update_cut_and_state_slots_updates_both_pools() {
        use crate::state::{StorageState, VisitedStatePool};
        use crate::system::{Bus, Hydro, System};

        // Create system with 3 hydros (state dimension = 3)
        let hydros: Vec<Hydro> = (0..3)
            .map(|id| {
                Hydro::new(id, None, 0, 1.0, 0.0, 100.0, 0.0, 10.0, 1000.0)
            })
            .collect();
        let buses = vec![Bus::new(0, 1000.0)];
        let system = System::new(buses, vec![], vec![], hydros);

        let template = StorageState::new(&system);

        // Create preallocated pools
        let mut cut_pool = BendersCutPool::preallocate(4, 8, 3); // 32 slots, 3 coefficients
        let mut state_pool = VisitedStatePool::preallocate(4, 8, &template);

        // Update slot using new method
        let cut_coefficients = [1.0, 2.0, 3.0];
        let state_coefficients = [10.0, 20.0, 30.0];
        let cut_rhs = 100.0;

        let slot = cut_pool.update_cut_and_state_slots(
            2, // iteration (1-based)
            3, // forward_pass_idx (0-based)
            &cut_coefficients,
            cut_rhs,
            &state_coefficients,
            &mut state_pool,
        );

        // Verify slot computation
        assert_eq!(slot, compute_slot(2, 3, 8)); // (2-1)*8 + 3 = 11

        // Verify cut was updated
        let cut = &cut_pool.pool[slot];
        assert_eq!(cut.coefficients, vec![1.0, 2.0, 3.0]);
        assert_eq!(cut.rhs, 100.0);
        assert_eq!(cut.iteration, 2);
        assert_eq!(cut.forward_pass_idx, 3);
        assert!(cut.is_active());
        assert!(cut.is_populated());

        // Verify state was updated
        let state = &state_pool.pool[slot];
        assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);
        assert_eq!(state.get_iteration(), 2);
        assert_eq!(state.get_forward_pass_idx(), 3);
    }

    #[test]
    fn test_update_cut_and_state_slots_matches_update_cut_behavior() {
        use crate::state::{StorageState, VisitedStatePool};
        use crate::system::{Bus, Hydro, System};

        // Create system with 3 hydros
        let hydros: Vec<Hydro> = (0..3)
            .map(|id| {
                Hydro::new(id, None, 0, 1.0, 0.0, 100.0, 0.0, 10.0, 1000.0)
            })
            .collect();
        let buses = vec![Bus::new(0, 1000.0)];
        let system = System::new(buses, vec![], vec![], hydros);

        let template = StorageState::new(&system);

        // Create two pools - one for each method
        let mut cut_pool_new = BendersCutPool::preallocate(2, 4, 3);
        let mut cut_pool_old = BendersCutPool::preallocate(2, 4, 3);
        let mut state_pool = VisitedStatePool::preallocate(2, 4, &template);

        let cut_coefficients = [1.5, 2.5, 3.5];
        let cut_rhs = 50.0;
        let state_coefficients = [5.0, 10.0, 15.0];

        // Use new method
        let slot_new = cut_pool_new.update_cut_and_state_slots(
            1,
            2,
            &cut_coefficients,
            cut_rhs,
            &state_coefficients,
            &mut state_pool,
        );

        // Use old method
        let slot_old =
            cut_pool_old.update_cut(1, 2, &cut_coefficients, cut_rhs);

        // Slots should match
        assert_eq!(slot_new, slot_old);

        // Cut contents should match
        assert_eq!(
            cut_pool_new.pool[slot_new].coefficients,
            cut_pool_old.pool[slot_old].coefficients
        );
        assert_eq!(
            cut_pool_new.pool[slot_new].rhs,
            cut_pool_old.pool[slot_old].rhs
        );
        assert_eq!(
            cut_pool_new.pool[slot_new].iteration,
            cut_pool_old.pool[slot_old].iteration
        );
    }
}
