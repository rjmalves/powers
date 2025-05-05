pub struct RiskMeasure {}

impl RiskMeasure {
    pub fn new() -> Self {
        Self {}
    }
    pub fn adjust_probabilities(
        &self,
        probabilities: &[f64],
        costs: &[f64],
    ) -> Vec<f64> {
        let num_branchings = costs.len();
        let p = 1.0 / num_branchings as f64;
        return vec![p; num_branchings];
    }
}
