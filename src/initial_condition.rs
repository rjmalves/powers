pub struct InitialCondition {
    storage: Vec<f64>,
    inflow: Vec<Vec<f64>>,
}

impl InitialCondition {
    pub fn new(storage: Vec<f64>, inflow: Vec<Vec<f64>>) -> Self {
        Self { storage, inflow }
    }

    pub fn get_storage(&self) -> &[f64] {
        &self.storage
    }

    pub fn get_inflow(&self, hydro_id: usize) -> &[f64] {
        &self.inflow.get(hydro_id).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_initial_condition() {
        let storage = vec![100.0, 200.0];
        let inflow = vec![vec![10.0], vec![20.0]];
        let ic = InitialCondition::new(storage.clone(), inflow.clone());
        assert_eq!(ic.get_storage(), &storage[..]);
        assert_eq!(ic.get_inflow(0), &inflow[0][..]);
        assert_eq!(ic.get_inflow(1), &inflow[1][..]);
    }
}