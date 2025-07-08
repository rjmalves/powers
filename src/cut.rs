use crate::utils;

#[derive(Debug)]
pub struct BendersCut {
    pub id: usize,
    pub coefficients: Vec<f64>,
    pub rhs: f64,
    pub active: bool,
    pub non_dominated_state_count: isize,
}

impl BendersCut {
    pub fn new(id: usize, coefficients: Vec<f64>, rhs: f64) -> Self {
        Self {
            id,
            coefficients,
            rhs,
            active: true,
            non_dominated_state_count: 1,
        }
    }

    pub fn eval_height_at_state(&self, state_coefficients: &[f64]) -> f64 {
        self.rhs + utils::dot_product(&self.coefficients, state_coefficients)
    }
}

#[derive(Debug)]
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_ids: Vec<usize>,
    pub total_cut_count: usize,
}

impl BendersCutPool {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            active_cut_ids: vec![],
            total_cut_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_benders_cut() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0);
        assert_eq!(cut.id, 1);
        assert_eq!(cut.coefficients, vec![1.0, 2.0]);
        assert_eq!(cut.rhs, 10.0);
        assert!(cut.active);
        assert_eq!(cut.non_dominated_state_count, 1);
    }

    #[test]
    fn test_eval_height_at_state() {
        let cut = BendersCut::new(1, vec![1.0, 2.0], 10.0);
        let state_coeffs = vec![3.0, 4.0];
        // 10.0 + (1.0 * 3.0 + 2.0 * 4.0) = 10.0 + 3.0 + 8.0 = 21.0
        assert_eq!(cut.eval_height_at_state(&state_coeffs), 21.0);
    }

    #[test]
    fn test_new_benders_cut_pool() {
        let pool = BendersCutPool::new();
        assert!(pool.pool.is_empty());
        assert!(pool.active_cut_ids.is_empty());
        assert_eq!(pool.total_cut_count, 0);
    }
}