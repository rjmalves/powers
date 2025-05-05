pub struct StochasticProcess {
    state: Vec<f64>,
}

impl StochasticProcess {
    pub fn new(state_size: usize) -> Self {
        Self {
            state: Vec::<f64>::with_capacity(state_size),
        }
    }

    pub fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64] {
        return noises;
    }
}
