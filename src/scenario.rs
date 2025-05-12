use rand::prelude::*;
use rand_distr;
use rand_xoshiro;

pub struct NodeNoiseGenerator<
    L: rand_distr::Distribution<f64>,
    I: rand_distr::Distribution<f64>,
> {
    pub load_distributions: Vec<L>, // indexed by hydro_id
    pub inflow_distributions: Vec<I>, // indexed by hydro_id
    pub num_branchings: usize,
    pub num_load_entities: usize,
    pub num_inflow_entities: usize,
}

pub struct NoiseGenerator<
    L: rand_distr::Distribution<f64>,
    I: rand_distr::Distribution<f64>,
> {
    pub node_generators: Vec<NodeNoiseGenerator<L, I>>,
}

impl<L: rand_distr::Distribution<f64>, I: rand_distr::Distribution<f64>>
    NoiseGenerator<L, I>
{
    pub fn new() -> Self {
        Self {
            node_generators: vec![],
        }
    }

    pub fn add_node_generator(
        &mut self,
        load_distributions: Vec<L>,
        inflow_distributions: Vec<I>,
        num_branchings: usize,
    ) {
        let num_load_entities = load_distributions.len();
        let num_inflow_entities = inflow_distributions.len();
        self.node_generators.push(NodeNoiseGenerator::<L, I> {
            load_distributions,
            inflow_distributions,
            num_branchings,
            num_load_entities,
            num_inflow_entities,
        });
    }

    pub fn get_node_generator(
        &mut self,
        id: usize,
    ) -> Option<&NodeNoiseGenerator<L, I>> {
        self.node_generators.get(id)
    }

    /// Generates an SAA from a set of distributions
    ///
    /// `seed` must be an u64
    ///
    /// ## Example
    ///
    /// ```
    /// let mu = 3.6;
    /// let sigma = 0.6928;
    /// let num_entities = 2;
    /// let mut scenario_generator = powers_rs::scenario::NoiseGenerator::new();
    /// let num_stages = 1;
    /// let num_branchings = 10;
    /// scenario_generator.add_node_generator(
    ///     vec![rand_distr::Normal::new(mu, sigma).unwrap(); num_entities],
    ///     vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_entities],
    ///     num_branchings);
    /// let saa = scenario_generator.generate(0);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().num_load_entities, num_entities);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().num_inflow_entities, num_entities);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().get_load_noises().len(), num_entities);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().get_inflow_noises().len(), num_entities);
    ///
    /// ```
    pub fn generate(&self, seed: u64) -> SAA {
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);

        let mut saa = SAA::new(&self);
        for (stage_id, stage_generator) in
            self.node_generators.iter().enumerate()
        {
            // here, 'noises' is indexed by [entity][branching]
            let load_noises: Vec<Vec<f64>> = stage_generator
                .load_distributions
                .iter()
                .map(|entity_generator| {
                    entity_generator
                        .sample_iter(&mut rng)
                        .take(stage_generator.num_branchings)
                        .collect()
                })
                .collect();
            let inflow_noises: Vec<Vec<f64>> = stage_generator
                .inflow_distributions
                .iter()
                .map(|entity_generator| {
                    entity_generator
                        .sample_iter(&mut rng)
                        .take(stage_generator.num_branchings)
                        .collect()
                })
                .collect();

            saa.set_noises_by_stage(
                stage_id,
                stage_generator.num_branchings,
                stage_generator.num_load_entities,
                stage_generator.num_inflow_entities,
                load_noises,
                inflow_noises,
            );
        }

        saa
    }
}

#[derive(Debug, Clone)]
pub struct SampledBranchingNoises {
    pub load_noises: Vec<f64>,
    pub inflow_noises: Vec<f64>,
    pub num_load_entities: usize,
    pub num_inflow_entities: usize,
}

impl SampledBranchingNoises {
    pub fn new(num_load_entities: usize, num_inflow_entities: usize) -> Self {
        Self {
            load_noises: Vec::<f64>::with_capacity(num_load_entities),
            inflow_noises: Vec::<f64>::with_capacity(num_inflow_entities),
            num_load_entities,
            num_inflow_entities,
        }
    }

    pub fn get_load_noises(&self) -> &[f64] {
        return self.load_noises.as_slice();
    }

    pub fn get_inflow_noises(&self) -> &[f64] {
        return self.inflow_noises.as_slice();
    }

    pub fn set_load_noises(&mut self, noises: &[f64]) {
        self.load_noises.extend_from_slice(noises);
    }

    pub fn set_inflow_noises(&mut self, noises: &[f64]) {
        self.inflow_noises.extend_from_slice(noises);
    }
}

#[derive(Debug, Clone)]
pub struct SampledNodeBranchings {
    pub num_branchings: usize,
    pub branching_noises: Vec<SampledBranchingNoises>,
}

impl SampledNodeBranchings {
    pub fn new<
        L: rand_distr::Distribution<f64>,
        I: rand_distr::Distribution<f64>,
    >(
        stage_generator: &NodeNoiseGenerator<L, I>,
    ) -> Self {
        let num_load_entities = stage_generator.num_load_entities;
        let num_inflow_entities = stage_generator.num_inflow_entities;
        Self {
            num_branchings: stage_generator.num_branchings,
            branching_noises: vec![
                SampledBranchingNoises::new(
                    num_load_entities,
                    num_inflow_entities
                );
                stage_generator.num_branchings
            ],
        }
    }

    pub fn get_noises_by_branching(
        &self,
        branching_id: usize,
    ) -> Option<&SampledBranchingNoises> {
        return self.branching_noises.get(branching_id);
    }

    pub fn set_noises_by_branching(
        &mut self,
        branching_id: usize,
        load_noises: &[f64],
        inflow_noises: &[f64],
    ) {
        self.branching_noises
            .get_mut(branching_id)
            .unwrap()
            .set_load_noises(load_noises);
        self.branching_noises
            .get_mut(branching_id)
            .unwrap()
            .set_inflow_noises(inflow_noises);
    }
}

#[derive(Debug)]
pub struct SAA {
    pub branching_samples: Vec<SampledNodeBranchings>,
    pub index_samplers: Vec<rand_distr::Uniform<usize>>,
}

impl SAA {
    pub fn new<
        L: rand_distr::Distribution<f64>,
        I: rand_distr::Distribution<f64>,
    >(
        scenario_generator: &NoiseGenerator<L, I>,
    ) -> Self {
        let branching_samples: Vec<SampledNodeBranchings> = scenario_generator
            .node_generators
            .iter()
            .map(|g| SampledNodeBranchings::new(g))
            .collect();
        let index_samplers = scenario_generator
            .node_generators
            .iter()
            .map(|g| {
                rand_distr::Uniform::<usize>::try_from(0..g.num_branchings)
                    .unwrap()
            })
            .collect();
        Self {
            branching_samples,
            index_samplers,
        }
    }

    pub fn get_branching_count_at_stage(
        &self,
        stage_id: usize,
    ) -> Option<usize> {
        return Some(self.branching_samples.get(stage_id)?.num_branchings);
    }

    pub fn get_noises_by_stage_and_branching(
        &self,
        stage_id: usize,
        branching_id: usize,
    ) -> Option<&SampledBranchingNoises> {
        return self
            .branching_samples
            .get(stage_id)?
            .get_noises_by_branching(branching_id);
    }

    pub fn sample_scenario(
        &self,
        rng: &mut rand_xoshiro::Xoshiro256Plus,
    ) -> Vec<&SampledBranchingNoises> {
        let branching_indices: Vec<usize> =
            self.index_samplers.iter().map(|d| d.sample(rng)).collect();

        branching_indices
            .iter()
            .enumerate()
            .map(|(id, branching_id)| {
                self.get_noises_by_stage_and_branching(id, *branching_id)
                    .unwrap()
            })
            .collect()
    }

    pub fn set_noises_by_stage(
        &mut self,
        stage_id: usize,
        num_branchings: usize,
        num_load_entities: usize,
        num_inflow_entities: usize,
        load_noises: Vec<Vec<f64>>,
        inflow_noises: Vec<Vec<f64>>,
    ) {
        for branching_id in 0..num_branchings {
            let mut branching_load_noises =
                Vec::<f64>::with_capacity(num_load_entities);
            for entitiy_id in 0..num_load_entities {
                branching_load_noises.push(
                    *load_noises
                        .get(entitiy_id)
                        .unwrap()
                        .get(branching_id)
                        .unwrap(),
                );
            }
            let mut branching_inflow_noises =
                Vec::<f64>::with_capacity(num_inflow_entities);
            for entitiy_id in 0..num_inflow_entities {
                branching_inflow_noises.push(
                    *inflow_noises
                        .get(entitiy_id)
                        .unwrap()
                        .get(branching_id)
                        .unwrap(),
                );
            }
            self.branching_samples
                .get_mut(stage_id)
                .unwrap()
                .set_noises_by_branching(
                    branching_id,
                    branching_load_noises.as_slice(),
                    branching_inflow_noises.as_slice(),
                );
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_generate_saa() {
        let mu = 3.6;
        let sigma = 0.6928;
        let num_entities = 2;
        let mut scenario_generator = NoiseGenerator::new();
        let num_branchings = 10;
        scenario_generator.add_node_generator(
            vec![rand_distr::Normal::new(10.0, 0.0).unwrap(); num_entities],
            vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_entities],
            num_branchings,
        );
        let saa = scenario_generator.generate(0);
        assert!(saa.get_noises_by_stage_and_branching(0, 0).is_some())
    }
}
