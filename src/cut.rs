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
