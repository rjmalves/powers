use rand::prelude::*;
use rand_distr;
use rand_xoshiro;

pub struct StageScenarioGenerator {
    pub distributions: Vec<rand_distr::LogNormal<f64>>, // indexed by hydro_id
    pub num_branchings: usize,
    pub num_hydros: usize,
}

pub struct ScenarioGenerator {
    pub stage_generators: Vec<StageScenarioGenerator>,
}

impl ScenarioGenerator {
    pub fn new() -> Self {
        Self {
            stage_generators: vec![],
        }
    }

    pub fn add_stage_generator(
        &mut self,
        distributions: Vec<rand_distr::LogNormal<f64>>,
        num_branchings: usize,
    ) {
        let num_hydros = distributions.len();
        self.stage_generators.push(StageScenarioGenerator {
            distributions,
            num_branchings,
            num_hydros,
        });
    }

    pub fn get_stage_generator(
        &mut self,
        id: usize,
    ) -> Option<&StageScenarioGenerator> {
        self.stage_generators.get(id)
    }
}

#[derive(Debug, Clone)]
pub struct SampledBranchingNoises {
    pub noises: Vec<f64>,
    pub num_hydros: usize,
}

impl SampledBranchingNoises {
    pub fn new(num_hydros: usize) -> Self {
        Self {
            noises: Vec::<f64>::with_capacity(num_hydros),
            num_hydros,
        }
    }

    pub fn set_noises(&mut self, noises: &[f64]) {
        for id in 0..self.num_hydros {
            self.noises[id] = noises[id]
        }
    }
}

#[derive(Debug, Clone)]
pub struct SampledStageBranchings {
    pub num_branchings: usize,
    pub branching_noises: Vec<SampledBranchingNoises>,
}

impl SampledStageBranchings {
    pub fn new(stage_generator: &StageScenarioGenerator) -> Self {
        let num_hydros = stage_generator.num_hydros;
        Self {
            num_branchings: stage_generator.num_branchings,
            branching_noises: vec![
                SampledBranchingNoises::new(num_hydros);
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

pub struct SAA {
    pub samples: Vec<SampledStageBranchings>,
}

impl SAA {
    pub fn new(scenario_generator: &ScenarioGenerator) -> Self {
        let samples: Vec<SampledStageBranchings> = scenario_generator
            .stage_generators
            .iter()
            .map(|g| SampledStageBranchings::new(g))
            .collect();
        Self { samples }
    }

    pub fn get_noises_by_stage_and_branching(
        &self,
        stage_id: usize,
        branching_id: usize,
    ) -> Option<&SampledBranchingNoises> {
        return self
            .samples
            .get(stage_id)?
            .get_noises_by_branching(branching_id);
    }

    pub fn set_noises_by_stage(
        &mut self,
        stage_id: usize,
        noises: Vec<Vec<f64>>,
    ) {
        let num_branchings = noises.len();

        for branching_id in 0..num_branchings {
            self.samples
                .get_mut(stage_id)
                .unwrap()
                .set_noises_by_branching(
                    branching_id,
                    noises.get(branching_id).unwrap().as_slice(),
                );
        }
    }
}

pub fn generate_saa(scenario_generator: &ScenarioGenerator, seed: u64) -> SAA {
    let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);

    let mut saa = SAA::new(&scenario_generator);
    for (stage_id, stage_generator) in
        scenario_generator.stage_generators.iter().enumerate()
    {
        let noises: Vec<Vec<f64>> = stage_generator
            .distributions
            .iter()
            .map(|hydro_generator| {
                hydro_generator
                    .sample_iter(&mut rng)
                    .take(stage_generator.num_branchings)
                    .collect()
            })
            .collect();

        saa.set_noises_by_stage(stage_id, noises);
    }

    saa
}
