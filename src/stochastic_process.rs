pub trait StochasticProcess {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64];
}

#[derive(Debug)]
pub struct Naive {
    _state: Vec<f64>,
}

impl Naive {
    pub fn new(state_size: usize) -> Self {
        Self {
            _state: Vec::<f64>::with_capacity(state_size),
        }
    }
}

impl StochasticProcess for Naive {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64] {
        return noises;
    }
}
