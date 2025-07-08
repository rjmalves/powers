pub trait StochasticProcess: Send + Sync {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64];
}

#[derive(Debug)]
pub struct Naive {}

impl Naive {
    pub fn new() -> Self {
        Self {}
    }
}

impl StochasticProcess for Naive {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64] {
        return noises;
    }
}

pub fn factory(kind: &str) -> Box<dyn StochasticProcess> {
    match kind {
        "naive" => Box::new(Naive::new()),
        _ => panic!("stochastic process kind {} not supported", kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_realize() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_factory_naive() {
        let sp = factory("naive");
        let noises = vec![4.0, 5.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    #[should_panic(expected = "stochastic process kind arma not supported")]
    fn test_factory_unsupported() {
        factory("arma");
    }
}