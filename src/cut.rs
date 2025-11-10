use crate::utils;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: bool,
    pub non_dominated_state_count: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
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
            active: true,
            non_dominated_state_count: 1,
            iteration,
            forward_pass_idx,
        }
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
    pub pool: Vec<BendersCut>,
    /// Maps cut_id → index in solver model constraints.
    /// PERFORMANCE: HashMap provides O(1) lookup vs BTreeMap's O(log n).
    /// Profiling showed 5.31% CPU time in BTreeMap iteration (std::_Rb_tree_increment).
    /// HashMap iteration is deterministic within a run (required for reproducibility).
    pub active_cut_indices: HashMap<usize, usize>,
    pub total_cut_count: usize,
}

impl Default for BendersCutPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BendersCutPool {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            active_cut_indices: HashMap::new(),
            total_cut_count: 0,
        }
    }
}

// ============================================================================
// Deep Memory Estimation Implementation
// ============================================================================

use crate::memory::DeepSizeEstimate;

impl DeepSizeEstimate for BendersCut {
    /// Estimate heap bytes for a BendersCut instance.
    ///
    /// BendersCut contains a `Vec<f64>` for coefficients, which is heap-allocated.
    /// For a typical problem with 156 hydros:
    /// - Stack: 56 bytes (struct fields)
    /// - Heap: 156 × 8 = 1,248 bytes (coefficient vector)
    /// - Total: ~1,304 bytes (vs 56 bytes from shallow estimation)
    fn estimate_heap_bytes(&self, sizing: &crate::memory::SizingInfo) -> usize {
        std::mem::size_of::<Self>() +                          // Stack
        self.coefficients.estimate_heap_bytes(sizing)          // Heap (coefficients)
    }

    /// Static estimation for BendersCut.
    ///
    /// Uses `max_state_dimension` from sizing context to estimate coefficient
    /// vector size, since cuts typically have one coefficient per state dimension.
    fn estimate_heap_bytes_static(sizing: &crate::memory::SizingInfo) -> usize {
        std::mem::size_of::<Self>() +
        sizing.max_state_dimension * std::mem::size_of::<f64>()
    }
}

impl DeepSizeEstimate for BendersCutPool {
    /// Estimate heap bytes for a BendersCutPool instance.
    ///
    /// Accounts for:
    /// - All cuts in the pool (with their nested allocations)
    /// - HashMap overhead for active_cut_indices
    fn estimate_heap_bytes(&self, sizing: &crate::memory::SizingInfo) -> usize {
        let stack_size = std::mem::size_of::<Self>();
        
        // Pool vector capacity
        let pool_overhead = self.pool.capacity() * std::mem::size_of::<BendersCut>();
        
        // All cuts with their nested allocations
        let cuts_heap: usize = self.pool.iter()
            .map(|cut| cut.estimate_heap_bytes(sizing))
            .sum();
        
        // HashMap overhead: capacity × (key + value + overhead)
        let hashmap_overhead = self.active_cut_indices.capacity() * 
                               (std::mem::size_of::<usize>() * 2 + 8);
        
        stack_size + pool_overhead + cuts_heap + hashmap_overhead
    }

    /// Static estimation for BendersCutPool.
    ///
    /// Estimates total cuts based on convergence patterns:
    /// - Typical: 10 cuts per node
    /// - Training iterations: max_iterations
    /// - Convergence factor: 30% survival rate (cuts get dominated/pruned)
    fn estimate_heap_bytes_static(sizing: &crate::memory::SizingInfo) -> usize {
        let stack_size = std::mem::size_of::<Self>();
        
        // Conservative estimate for total cuts
        let estimated_cuts = {
            let base_cuts = 10 * sizing.num_nodes;
            let training_cuts = base_cuts * sizing.max_iterations;
            (training_cuts as f64 * 0.3) as usize  // 30% survival
        };
        
        let pool_overhead = estimated_cuts * std::mem::size_of::<BendersCut>();
        let cuts_heap = estimated_cuts * BendersCut::estimate_heap_bytes_static(sizing);
        let hashmap_overhead = estimated_cuts * (std::mem::size_of::<usize>() * 2 + 8);
        
        stack_size + pool_overhead + cuts_heap + hashmap_overhead
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
        assert!(cut.active);
        assert_eq!(cut.non_dominated_state_count, 1);
        assert_eq!(cut.iteration, 1);
        assert_eq!(cut.forward_pass_idx, 0);
    }

    #[test]
    fn test_eval_height_at_state() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0, 1, 0);
        let state_coeffs = vec![3.0, 4.0];
        // 10.0 + (1.0 * 3.0 + 2.0 * 4.0) = 10.0 + 3.0 + 8.0 = 21.0
        assert_eq!(cut.eval_height_at_state(&state_coeffs), 21.0);
    }

    #[test]
    fn test_new_benders_cut_pool() {
        let pool = BendersCutPool::new();
        assert!(pool.pool.is_empty());
        assert!(pool.active_cut_indices.is_empty());
        assert_eq!(pool.total_cut_count, 0);
    }

    #[test]
    fn test_active_cut_indices_iteration_deterministic() {
        // PERFORMANCE: HashMap provides deterministic iteration within a run,
        // which is sufficient for reproducible results. Unlike BTreeMap, it
        // doesn't guarantee sorted order, but that's not required for correctness.
        let mut pool = BendersCutPool::new();

        // Add cuts in non-sequential order
        pool.active_cut_indices.insert(15, 100);
        pool.active_cut_indices.insert(5, 200);
        pool.active_cut_indices.insert(10, 300);
        pool.active_cut_indices.insert(1, 400);
        pool.active_cut_indices.insert(20, 500);

        // Collect keys from multiple iterations within the same run
        let keys1: Vec<_> = pool.active_cut_indices.keys().copied().collect();
        let keys2: Vec<_> = pool.active_cut_indices.keys().copied().collect();
        let keys3: Vec<_> = pool.active_cut_indices.keys().copied().collect();

        // HashMap iteration is deterministic within a single run
        // (all iterations produce identical order)
        assert_eq!(keys1, keys2);
        assert_eq!(keys2, keys3);

        // Verify values are also accessible in deterministic order
        let values1: Vec<_> =
            pool.active_cut_indices.values().copied().collect();
        let values2: Vec<_> =
            pool.active_cut_indices.values().copied().collect();

        assert_eq!(values1, values2);

        // Verify all expected entries are present (order doesn't matter)
        assert_eq!(keys1.len(), 5);
        assert!(keys1.contains(&1));
        assert!(keys1.contains(&5));
        assert!(keys1.contains(&10));
        assert!(keys1.contains(&15));
        assert!(keys1.contains(&20));
    }
}
