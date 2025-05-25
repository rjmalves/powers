pub trait RiskMeasure: Send + Sync {
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

pub fn factory(kind: &str) -> Box<dyn RiskMeasure> {
    match kind {
        "expectation" => Box::new(Expectation::new()),
        _ => panic!("risk measure kind {} not supported", kind),
    }
}
