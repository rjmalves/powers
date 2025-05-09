pub trait RiskMeasure {
    fn adjust_probabilities<'a>(
        &'a self,
        probabilities: &'a [f64],
        costs: &'a [f64],
    ) -> &'a [f64];
}

pub struct Expectation {}

impl Expectation {
    pub fn new() -> Self {
        Self {}
    }
}

impl RiskMeasure for Expectation {
    fn adjust_probabilities<'a>(
        &'a self,
        probabilities: &'a [f64],
        _costs: &'a [f64],
    ) -> &'a [f64] {
        return probabilities;
    }
}
