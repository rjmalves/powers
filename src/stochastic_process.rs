pub trait StochasticProcess {
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
