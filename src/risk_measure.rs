#[derive(Debug)]
pub struct RiskMeasure {}

impl RiskMeasure {
    pub fn new() -> Self {
        Self {}
    }
    pub fn adjust_probabilities<'a>(
        &'a self,
        probabilities: &'a [f64],
        _costs: &'a [f64],
    ) -> &'a [f64] {
        return probabilities;
    }
}
