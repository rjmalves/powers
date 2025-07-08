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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expectation_adjust_probabilities() {
        let expectation = Expectation::new();
        let probabilities = vec![0.25, 0.75];
        let costs = vec![100.0, 200.0];
        let adjusted = expectation.adjust_probabilities(&probabilities, &costs);
        assert_eq!(adjusted, &probabilities[..]);
    }

    #[test]
    fn test_factory_expectation() {
        let risk_measure = factory("expectation");
        let probabilities = vec![0.5, 0.5];
        let costs = vec![10.0, 20.0];
        let adjusted = risk_measure.adjust_probabilities(&probabilities, &costs);
        assert_eq!(adjusted, &probabilities[..]);
    }

    #[test]
    #[should_panic(expected = "risk measure kind conditional_value_at_risk not supported")]
    fn test_factory_unsupported() {
        factory("conditional_value_at_risk");
    }
}