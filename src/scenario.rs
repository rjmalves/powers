use rand::prelude::*;
use rand_distr;
use rand_xoshiro;

pub struct StageScenarioGenerator<D: rand_distr::Distribution<f64>> {
    pub distributions: Vec<D>, // indexed by hydro_id
    pub num_branchings: usize,
    pub num_entities: usize,
}

pub struct ScenarioGenerator<D: rand_distr::Distribution<f64>> {
    pub stage_generators: Vec<StageScenarioGenerator<D>>,
}

impl<D: rand_distr::Distribution<f64>> ScenarioGenerator<D> {
    pub fn new() -> Self {
        Self {
            stage_generators: vec![],
        }
    }

    pub fn add_stage_generator(
        &mut self,
        distributions: Vec<D>,
        num_branchings: usize,
    ) {
        let num_entities = distributions.len();
        self.stage_generators.push(StageScenarioGenerator::<D> {
            distributions,
            num_branchings,
            num_entities,
        });
    }

    pub fn get_stage_generator(
        &mut self,
        id: usize,
    ) -> Option<&StageScenarioGenerator<D>> {
        self.stage_generators.get(id)
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
    /// let mut scenario_generator = powers_rs::scenario::ScenarioGenerator::new();
    /// let num_stages = 1;
    /// let num_branchings = 10;
    /// scenario_generator.add_stage_generator(
    ///     vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_entities],
    ///     num_branchings);
    /// let saa = scenario_generator.generate(0);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().num_entities, num_entities);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().get_noises().len(), num_entities);
    ///
    /// ```
    pub fn generate(&self, seed: u64) -> SAA {
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);

        let mut saa = SAA::new(&self);
        for (stage_id, stage_generator) in
            self.stage_generators.iter().enumerate()
        {
            // here, 'noises' is indexed by [entity][branching]
            let noises: Vec<Vec<f64>> = stage_generator
                .distributions
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
                stage_generator.num_entities,
                noises,
            );
        }

        saa
    }
}

#[derive(Debug, Clone)]
pub struct SampledBranchingNoises {
    pub noises: Vec<f64>,
    pub num_entities: usize,
}

impl SampledBranchingNoises {
    pub fn new(num_entities: usize) -> Self {
        Self {
            noises: Vec::<f64>::with_capacity(num_entities),
            num_entities,
        }
    }

    pub fn get_noises(&self) -> &[f64] {
        return self.noises.as_slice();
    }

    pub fn set_noises(&mut self, noises: &[f64]) {
        self.noises.extend_from_slice(noises);
    }
}

#[derive(Debug, Clone)]
pub struct SampledStageBranchings {
    pub num_branchings: usize,
    pub branching_noises: Vec<SampledBranchingNoises>,
}

impl SampledStageBranchings {
    pub fn new<D: rand_distr::Distribution<f64>>(
        stage_generator: &StageScenarioGenerator<D>,
    ) -> Self {
        let num_entities = stage_generator.num_entities;
        Self {
            num_branchings: stage_generator.num_branchings,
            branching_noises: vec![
                SampledBranchingNoises::new(num_entities);
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
        noises: &[f64],
    ) {
        self.branching_noises
            .get_mut(branching_id)
            .unwrap()
            .set_noises(noises);
    }
}

#[derive(Debug)]
pub struct SAA {
    pub branching_samples: Vec<SampledStageBranchings>,
    pub index_samplers: Vec<rand_distr::Uniform<usize>>,
}

impl SAA {
    pub fn new<D: rand_distr::Distribution<f64>>(
        scenario_generator: &ScenarioGenerator<D>,
    ) -> Self {
        let branching_samples: Vec<SampledStageBranchings> = scenario_generator
            .stage_generators
            .iter()
            .map(|g| SampledStageBranchings::new(g))
            .collect();
        let index_samplers = scenario_generator
            .stage_generators
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
        num_entities: usize,
        noises: Vec<Vec<f64>>,
    ) {
        for branching_id in 0..num_branchings {
            let mut branching_noises = Vec::<f64>::with_capacity(num_entities);
            for entitiy_id in 0..num_entities {
                branching_noises.push(
                    *noises.get(entitiy_id).unwrap().get(branching_id).unwrap(),
                );
            }
            self.branching_samples
                .get_mut(stage_id)
                .unwrap()
                .set_noises_by_branching(
                    branching_id,
                    branching_noises.as_slice(),
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
        let mut scenario_generator = ScenarioGenerator::new();
        let num_branchings = 10;
        scenario_generator.add_stage_generator(
            vec![rand_distr::LogNormal::new(mu, sigma).unwrap(); num_entities],
            num_branchings,
        );
        let saa = scenario_generator.generate(0);
        assert!(saa.get_noises_by_stage_and_branching(0, 0).is_some())
    }
}
