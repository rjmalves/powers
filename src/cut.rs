use crate::utils;
use std::collections::BTreeMap;

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
    /// This allows O(log n) lookup of constraint row when removing cuts
    pub active_cut_indices: BTreeMap<usize, usize>,
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
            active_cut_indices: BTreeMap::new(),
            total_cut_count: 0,
        }
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
        // Verify BTreeMap provides deterministic iteration order
        let mut pool = BendersCutPool::new();

        // Add cuts in non-sequential order to test sorting
        pool.active_cut_indices.insert(15, 100);
        pool.active_cut_indices.insert(5, 200);
        pool.active_cut_indices.insert(10, 300);
        pool.active_cut_indices.insert(1, 400);
        pool.active_cut_indices.insert(20, 500);

        // Collect keys from multiple iterations
        let keys1: Vec<_> = pool.active_cut_indices.keys().copied().collect();
        let keys2: Vec<_> = pool.active_cut_indices.keys().copied().collect();
        let keys3: Vec<_> = pool.active_cut_indices.keys().copied().collect();

        // Should be sorted in ascending order: [1, 5, 10, 15, 20]
        assert_eq!(keys1, vec![1, 5, 10, 15, 20]);

        // All iterations should produce identical order (deterministic)
        assert_eq!(keys1, keys2);
        assert_eq!(keys2, keys3);

        // Verify values are also accessible in deterministic order
        let values1: Vec<_> =
            pool.active_cut_indices.values().copied().collect();
        let values2: Vec<_> =
            pool.active_cut_indices.values().copied().collect();

        assert_eq!(values1, vec![400, 200, 300, 100, 500]); // Corresponding to sorted keys
        assert_eq!(values1, values2);
    }
}
