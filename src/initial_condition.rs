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
