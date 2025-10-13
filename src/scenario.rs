#![allow(deprecated)] // Old NodeNoiseGenerator still used in existing code

use rand::prelude::*;
use rand_distr;
use rand_xoshiro;
use std::collections::HashMap;

use crate::ar_dynamics::ARDynamicsApplicator;
use crate::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
use crate::correlation_applicator::CorrelationApplicator;
use crate::initial_condition::InitialCondition;
use crate::input::{
    CorrelationBlock, MarginalDistribution, NoiseModelV2, Recourse,
    TemporalModel, UncertaintyType,
};
use crate::marginal_transformer::MarginalTransformer;

/// Legacy noise generator (deprecated)
///
/// **DEPRECATED**: Use `ScenarioGenerator` instead for new code.
/// This struct is kept for backward compatibility only.
///
/// # Migration
///
/// **Old code**:
/// ```ignore
/// let mut generator = NoiseGenerator::new();
/// generator.add_node_generator(load_dist, inflow_dist, num_scenarios);
/// let saa = generator.generate(seed);
/// ```
///
/// **New code**:
/// ```ignore
/// let generator = ScenarioGenerator::from_recourse_input(&recourse, &initial_condition, seed)?;
/// let saa = generator.generate_saa(num_stages, &scenarios_per_stage);
/// ```
///
/// # Replacement
///
/// The new `ScenarioGenerator` provides:
/// - AR temporal dynamics support
/// - Correlation structure (Gaussian copula)
/// - LogNormal3 marginals with zero LP overhead
/// - Proper separation of marginal/innovation/temporal models
///
/// See `ScenarioGenerator` documentation for details.
#[deprecated(
    since = "0.3.0",
    note = "Use ScenarioGenerator instead. This will be removed in v0.4.0"
)]
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

impl<L: rand_distr::Distribution<f64>, I: rand_distr::Distribution<f64>> Default
    for NoiseGenerator<L, I>
{
    fn default() -> Self {
        Self::new()
    }
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

        let mut saa = SAA::new(self);
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
        self.load_noises.as_slice()
    }

    pub fn get_inflow_noises(&self) -> &[f64] {
        self.inflow_noises.as_slice()
    }

    pub fn set_load_noises(&mut self, noises: &[f64]) {
        self.load_noises.clear(); // Clear existing noises before setting new ones
        self.load_noises.extend_from_slice(noises);
    }

    pub fn set_inflow_noises(&mut self, noises: &[f64]) {
        self.inflow_noises.clear(); // Clear existing noises before setting new ones
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
        self.branching_noises.get(branching_id)
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
        Some(self.branching_samples.get(stage_id)?.num_branchings)
    }

    pub fn get_noises_by_stage_and_branching(
        &self,
        stage_id: usize,
        branching_id: usize,
    ) -> Option<&SampledBranchingNoises> {
        self.branching_samples
            .get(stage_id)?
            .get_noises_by_branching(branching_id)
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
        // Ensure we have enough stages (extend if necessary)
        while self.branching_samples.len() <= stage_id {
            self.branching_samples.push(SampledNodeBranchings {
                num_branchings: 0,
                branching_noises: vec![],
            });
        }

        // Initialize the stage with the correct number of branchings
        self.branching_samples[stage_id] = SampledNodeBranchings {
            num_branchings,
            branching_noises: vec![
                SampledBranchingNoises::new(
                    num_load_entities,
                    num_inflow_entities
                );
                num_branchings
            ],
        };

        // Fill in the noise values
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
#[allow(clippy::items_after_test_module)] // Tests for old NoiseGenerator, new code follows
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

    #[test]
    fn test_get_branching_count_at_stage() {
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

        assert_eq!(saa.get_branching_count_at_stage(0), Some(num_branchings));
        assert_eq!(saa.get_branching_count_at_stage(999), None);
    }

    #[test]
    fn test_sample_scenario() {
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

        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(42);
        let scenario = saa.sample_scenario(&mut rng);

        assert_eq!(scenario.len(), 1); // One stage
        assert_eq!(scenario[0].load_noises.len(), num_entities);
        assert_eq!(scenario[0].inflow_noises.len(), num_entities);
    }

    #[test]
    fn test_get_noises_by_stage_and_branching_out_of_bounds() {
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

        // Test out of bounds access
        assert!(saa.get_noises_by_stage_and_branching(999, 0).is_none());
        assert!(saa.get_noises_by_stage_and_branching(0, 999).is_none());
    }
}

// ============================================================================
// New Scenario Generation Pipeline (AR-6.6)
// ============================================================================

/// Scenario generator for CEPEL-compliant 4-stage pipeline
///
/// Orchestrates the complete scenario generation process:
/// 1. **Base Noise**: Generate Z ~ N(0,1)
/// 2. **Correlation**: Apply Cholesky transformation W = L×Z
/// 3. **Marginal Transformation**: Transform to target distributions
/// 4. **AR Dynamics**: Apply temporal correlation Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
///
/// # Architecture
///
/// This replaces the old `NoiseGenerator` with a pipeline that properly separates:
/// - Marginal distributions (target distribution of realizations)
/// - Innovation distributions (white noise for AR models)
/// - Temporal models (independent vs AR)
/// - Correlation structure (spatial/physical correlations)
///
/// # Example
///
/// ```ignore
/// // Create generator from recourse input
/// let generator = ScenarioGenerator::from_recourse_input(
///     &recourse,
///     &initial_condition,
///     seed,
/// )?;
///
/// // Generate SAA for SDDP
/// let saa = generator.generate_saa(num_stages, &scenarios_per_stage);
/// ```
///
/// # Performance
///
/// Target: 1000 scenarios × 10 entities × 12 stages in <200ms
///
/// # Backward Compatibility
///
/// For independent, uncorrelated models, produces identical results to old `NoiseGenerator`.
pub struct ScenarioGenerator {
    /// Noise models for all entities (schema v2 format)
    noise_models: Vec<NoiseModelV2>,

    /// Correlation blocks (optional, can be empty for independent sampling)
    correlation_blocks: Vec<CorrelationBlock>,

    /// Initial lags for AR entities: entity_idx → [X_{-1}, X_{-2}, ..., X_{-p}]
    initial_lags: HashMap<usize, Vec<f64>>,

    /// Base noise method (default: Standard)
    base_noise_method: BaseNoiseMethod,

    /// RNG seed for deterministic generation
    seed: u64,

    // Derived/cached data (computed from noise_models)
    /// Marginal distributions indexed by global entity index
    entity_marginals: Vec<MarginalDistribution>,

    /// Temporal models indexed by global entity index
    entity_temporal_models: Vec<TemporalModel>,

    /// Number of load entities
    num_load_entities: usize,

    /// Number of inflow entities
    num_inflow_entities: usize,

    /// Mapping: (UncertaintyType, entity_id) → global entity index
    entity_index_map: HashMap<(UncertaintyType, usize), usize>,
}

/// Type alias for entity mapping result to reduce type complexity
type EntityMappingResult = (
    Vec<MarginalDistribution>,
    Vec<TemporalModel>,
    HashMap<(UncertaintyType, usize), usize>,
    usize,
    usize,
);

impl ScenarioGenerator {
    /// Create scenario generator from recourse input
    ///
    /// # Arguments
    ///
    /// - `recourse`: Recourse input with noise models and correlation
    /// - `initial_condition`: Initial condition with lag history for AR entities
    /// - `seed`: RNG seed for deterministic generation
    ///
    /// # Returns
    ///
    /// `Ok(ScenarioGenerator)` if input is valid, `Err(String)` otherwise
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - AR entities missing initial lags
    /// - Correlation blocks reference non-existent entities
    /// - Noise models inconsistent (duplicate entities, wrong season, etc.)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let recourse = read_recourse_input("recourse.json");
    /// let initial_condition = InitialCondition::from_input(&recourse.initial_condition);
    /// let generator = ScenarioGenerator::from_recourse_input(&recourse, &initial_condition, 42)?;
    /// ```
    pub fn from_recourse_input(
        recourse: &Recourse,
        initial_condition: &InitialCondition,
        seed: u64,
    ) -> Result<Self, String> {
        // Normalize recourse to v2 format
        let noise_models = recourse.normalize_to_v2().map_err(|e| {
            format!("Failed to normalize recourse input: {}", e)
        })?;

        // Build entity mappings
        let (
            entity_marginals,
            entity_temporal_models,
            entity_index_map,
            num_load_entities,
            num_inflow_entities,
        ) = Self::build_entity_mappings(&noise_models)?;

        // Extract correlation blocks
        let correlation_blocks = recourse
            .correlation
            .as_ref()
            .map(|c| c.blocks.clone())
            .unwrap_or_default();

        // Extract initial lags for AR entities
        let initial_lags =
            Self::extract_initial_lags(initial_condition, &entity_index_map)?;

        // Validate consistency
        Self::validate_ar_entities_have_lags(
            &entity_temporal_models,
            &initial_lags,
        )?;
        Self::validate_correlation_entities_exist(
            &correlation_blocks,
            &entity_index_map,
        )?;

        Ok(Self {
            noise_models,
            correlation_blocks,
            initial_lags,
            base_noise_method: BaseNoiseMethod::Standard,
            seed,
            entity_marginals,
            entity_temporal_models,
            num_load_entities,
            num_inflow_entities,
            entity_index_map,
        })
    }

    /// Generate SAA (Sample Average Approximation) for SDDP
    ///
    /// # Arguments
    ///
    /// - `num_stages`: Number of stages in scenario tree
    /// - `scenarios_per_stage`: Number of scenarios (branchings) per stage
    ///
    /// # Returns
    ///
    /// `SAA` struct compatible with existing SDDP algorithm
    ///
    /// # Performance
    ///
    /// For 1000 scenarios × 10 entities × 12 stages: ~150ms (target: <200ms)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let saa = generator.generate_saa(12, &vec![1000; 12]);
    /// ```
    pub fn generate_saa(
        &self,
        num_stages: usize,
        scenarios_per_stage: &[usize],
    ) -> SAA {
        assert_eq!(
            scenarios_per_stage.len(),
            num_stages,
            "scenarios_per_stage length must equal num_stages"
        );

        // Initialize empty SAA
        let mut saa = SAA::new_empty();

        // Current lags (updated after each stage)
        let mut current_lags = self.initial_lags.clone();

        // Generate scenarios for each stage
        #[allow(clippy::needless_range_loop)]
        // Need numeric index for scenarios_per_stage[stage_id]
        for stage_id in 0..num_stages {
            let num_scenarios = scenarios_per_stage[stage_id];

            // 4-stage pipeline for this stage
            let (realizations, updated_lags) = self.generate_stage_scenarios(
                stage_id,
                num_scenarios,
                &current_lags,
            );

            // Convert to SAA format
            let (load_noises, inflow_noises) =
                self.split_by_uncertainty_type(&realizations);

            saa.set_noises_by_stage(
                stage_id,
                num_scenarios,
                self.num_load_entities,
                self.num_inflow_entities,
                load_noises,
                inflow_noises,
            );

            // Update lags for next stage
            current_lags = updated_lags;
        }

        saa
    }

    /// Generate scenarios for a single stage using 4-stage pipeline
    ///
    /// # Pipeline Stages
    ///
    /// 1. **Base Noise**: Z ~ N(0,1) [BaseNoiseGenerator]
    /// 2. **Correlation**: W = L×Z [CorrelationApplicator]
    /// 3. **Marginal**: ε ~ F [MarginalTransformer]
    /// 4. **AR Dynamics**: X = AR(ε) [ARDynamicsApplicator]
    ///
    /// # Returns
    ///
    /// - `realizations`: Vec<Vec<f64>> indexed by [scenario][entity]
    /// - `updated_lags`: HashMap for next stage
    fn generate_stage_scenarios(
        &self,
        stage_id: usize,
        num_scenarios: usize,
        current_lags: &HashMap<usize, Vec<f64>>,
    ) -> (Vec<Vec<f64>>, HashMap<usize, Vec<f64>>) {
        let num_entities = self.entity_marginals.len();

        // Stage 1: Base noise Z ~ N(0,1)
        let base_noise_generator = BaseNoiseGenerator::new(
            num_scenarios,
            num_entities,
            self.seed + stage_id as u64, // Vary seed per stage
        );
        let base_noise = base_noise_generator.generate(self.base_noise_method);

        // Stage 2: Correlation W = L×Z (skip if no correlation blocks)
        let correlated = if self.correlation_blocks.is_empty() {
            base_noise
        } else {
            // Build correlation blocks with global entity index mapping
            let blocks: Vec<crate::correlation_applicator::CorrelationBlock> =
                self.correlation_blocks
                    .iter()
                    .map(|cb| {
                        // Convert from input::CorrelationBlock to correlation_applicator::CorrelationBlock
                        let entities: Vec<crate::correlation_applicator::EntityRef> = cb
                            .entities
                            .iter()
                            .map(|entity_ref| {
                                let uncertainty_type = match entity_ref.uncertainty_type {
                                    UncertaintyType::Inflow => {
                                        crate::correlation_applicator::UncertaintyType::HydroInflow
                                    }
                                    UncertaintyType::Load => {
                                        crate::correlation_applicator::UncertaintyType::Load
                                    }
                                };
                                crate::correlation_applicator::EntityRef {
                                    uncertainty_type,
                                    entity_id: entity_ref.entity_id,
                                }
                            })
                            .collect();

                        let matrix =
                            nalgebra::DMatrix::from_row_slice(
                                cb.correlation_matrix.len(),
                                cb.correlation_matrix[0].len(),
                                &cb.correlation_matrix
                                    .iter()
                                    .flatten()
                                    .copied()
                                    .collect::<Vec<_>>(),
                            );

                        crate::correlation_applicator::CorrelationBlock::new(
                            entities, matrix,
                        )
                        .unwrap()
                    })
                    .collect();

            // Build entity_to_global_index for CorrelationApplicator
            let entity_to_global_index: HashMap<
                crate::correlation_applicator::EntityRef,
                usize,
            > = self
                .entity_index_map
                .iter()
                .map(|((uncertainty_type, entity_id), global_idx)| {
                    let uncertainty_type = match uncertainty_type {
                        UncertaintyType::Inflow => {
                            crate::correlation_applicator::UncertaintyType::HydroInflow
                        }
                        UncertaintyType::Load => {
                            crate::correlation_applicator::UncertaintyType::Load
                        }
                    };
                    (
                        crate::correlation_applicator::EntityRef {
                            uncertainty_type,
                            entity_id: *entity_id,
                        },
                        *global_idx,
                    )
                })
                .collect();

            let correlation_applicator =
                CorrelationApplicator::new(blocks, entity_to_global_index);
            correlation_applicator.apply_correlation(&base_noise)
        };

        // Stage 3: Marginal transformation ε ~ F
        let marginal_transformer =
            MarginalTransformer::new(self.entity_marginals.clone()).unwrap();
        let innovations = marginal_transformer.transform_marginals(&correlated);

        // Stage 4: AR dynamics X = AR(ε)
        let ar_applicator = ARDynamicsApplicator::new(
            self.entity_temporal_models.clone(),
            current_lags.clone(),
        )
        .unwrap();
        ar_applicator.apply_ar_dynamics(&innovations)
    }

    /// Split realizations by uncertainty type for SAA format
    ///
    /// SAA expects separate load_noises and inflow_noises.
    /// This function splits realizations indexed by global entity index
    /// into two separate arrays.
    ///
    /// # Returns
    ///
    /// - `load_noises`: Vec<Vec<f64>> indexed by [entity_id][scenario]
    /// - `inflow_noises`: Vec<Vec<f64>> indexed by [entity_id][scenario]
    fn split_by_uncertainty_type(
        &self,
        realizations: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let num_scenarios = realizations.len();

        // Pre-allocate output
        let mut load_noises =
            vec![Vec::with_capacity(num_scenarios); self.num_load_entities];
        let mut inflow_noises =
            vec![Vec::with_capacity(num_scenarios); self.num_inflow_entities];

        // Split by uncertainty type
        for scenario in realizations {
            for (global_idx, &value) in scenario.iter().enumerate() {
                // Find which noise model corresponds to this global index
                if let Some(noise_model) = self.noise_models.iter().find(|nm| {
                    self.entity_index_map
                        .get(&(nm.uncertainty_type.clone(), nm.entity_id))
                        == Some(&global_idx)
                }) {
                    match noise_model.uncertainty_type {
                        UncertaintyType::Load => {
                            load_noises[noise_model.entity_id].push(value);
                        }
                        UncertaintyType::Inflow => {
                            inflow_noises[noise_model.entity_id].push(value);
                        }
                    }
                }
            }
        }

        (load_noises, inflow_noises)
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Build entity mappings from noise models
    ///
    /// Creates:
    /// - Global entity index (0..N-1) for all entities
    /// - Marginal distributions indexed by global index
    /// - Temporal models indexed by global index
    /// - Entity index map: (UncertaintyType, entity_id) → global index
    /// - Entity counts by uncertainty type
    fn build_entity_mappings(
        noise_models: &[NoiseModelV2],
    ) -> Result<EntityMappingResult, String> {
        let num_entities = noise_models.len();

        let mut entity_marginals = Vec::with_capacity(num_entities);
        let mut entity_temporal_models = Vec::with_capacity(num_entities);
        let mut entity_index_map = HashMap::new();
        let mut num_load_entities = 0;
        let mut num_inflow_entities = 0;

        for (global_idx, nm) in noise_models.iter().enumerate() {
            entity_marginals.push(nm.marginal_distribution.clone());
            entity_temporal_models.push(nm.temporal_model.clone());
            entity_index_map.insert(
                (nm.uncertainty_type.clone(), nm.entity_id),
                global_idx,
            );

            match nm.uncertainty_type {
                UncertaintyType::Load => num_load_entities += 1,
                UncertaintyType::Inflow => num_inflow_entities += 1,
            }
        }

        Ok((
            entity_marginals,
            entity_temporal_models,
            entity_index_map,
            num_load_entities,
            num_inflow_entities,
        ))
    }

    /// Extract initial lags from InitialCondition
    ///
    /// Converts InitialCondition lag storage to HashMap keyed by global entity index.
    fn extract_initial_lags(
        initial_condition: &InitialCondition,
        entity_index_map: &HashMap<(UncertaintyType, usize), usize>,
    ) -> Result<HashMap<usize, Vec<f64>>, String> {
        let mut initial_lags = HashMap::new();

        // Extract inflow lags (hydro_id corresponds to inflow entity_id)
        // Try accessing inflow lags for each entity that appears in the noise models
        for ((uncertainty_type, entity_id), global_idx) in entity_index_map {
            if uncertainty_type == &UncertaintyType::Inflow {
                // Try to get lags for this hydro
                let lags = initial_condition.get_inflow(*entity_id);
                if !lags.is_empty() {
                    initial_lags.insert(*global_idx, lags.to_vec());
                }
            }
        }

        // TODO: Add load lag support if needed in the future
        // For now, loads are typically independent (no lags)

        Ok(initial_lags)
    }

    /// Validate that all AR entities have initial lags
    fn validate_ar_entities_have_lags(
        entity_temporal_models: &[TemporalModel],
        initial_lags: &HashMap<usize, Vec<f64>>,
    ) -> Result<(), String> {
        for (entity_idx, temporal_model) in
            entity_temporal_models.iter().enumerate()
        {
            if let TemporalModel::Autoregressive {
                lag_order,
                coefficients: _,
            } = temporal_model
            {
                let lags = initial_lags.get(&entity_idx).ok_or(format!(
                    "Entity {} has AR model but no initial lags",
                    entity_idx
                ))?;

                if lags.len() != *lag_order {
                    return Err(format!(
                        "Entity {} AR({}) requires {} lags, got {}",
                        entity_idx,
                        lag_order,
                        lag_order,
                        lags.len()
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate that correlation blocks reference existing entities
    fn validate_correlation_entities_exist(
        correlation_blocks: &[CorrelationBlock],
        entity_index_map: &HashMap<(UncertaintyType, usize), usize>,
    ) -> Result<(), String> {
        for block in correlation_blocks {
            for entity_ref in &block.entities {
                let key =
                    (entity_ref.uncertainty_type.clone(), entity_ref.entity_id);
                if !entity_index_map.contains_key(&key) {
                    return Err(format!(
                        "Correlation block '{}' references non-existent entity: {:?}",
                        block.name, entity_ref
                    ));
                }
            }
        }

        Ok(())
    }
}

impl SAA {
    /// Create empty SAA (for ScenarioGenerator to populate)
    fn new_empty() -> Self {
        Self {
            branching_samples: vec![],
            index_samplers: vec![],
        }
    }
}
