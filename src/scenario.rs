#![allow(deprecated)] // Old NodeNoiseGenerator still used in existing code

use rand::prelude::*;
use rand_distr;
use rand_xoshiro;
use std::collections::HashMap;

use crate::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
use crate::correlation_applicator::CorrelationApplicator;
use crate::initial_condition::InitialCondition;
use crate::input::{
    CorrelationBlock, MarginalDistribution, NoiseModel, Recourse,
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

/// Type alias for seasonal parameter tuple (mean, std_dev, marginal)
///
/// Simplifies NoiseLookupTable HashMap signatures to avoid clippy::type_complexity.
type SeasonalParams = (f64, f64, Option<MarginalDistribution>);

/// Optimized lookup table for O(1) noise parameter access
///
/// Pre-computes and indexes noise parameters for fast lookup during scenario generation.
/// Replaces O(n) linear searches through noise models with O(1) HashMap lookups.
///
/// # Architecture
///
/// The lookup table is built once from `UnifiedNoiseSpec` and provides:
/// - **Seasonal parameters**: O(1) lookup by (uncertainty_type, entity_id, season_id)
/// - **Temporal model info**: Quick check if entity uses PAR or independent model
///
/// # Performance
///
/// - **Construction**: O(n × s) where n = entities, s = seasons per entity
/// - **Lookup**: O(1) average case via HashMap
/// - **Memory**: ~80 bytes per (entity, season) entry
///
/// # Example
///
/// ```ignore
/// // Build from unified specs
/// let unified_specs = UnifiedNoiseSpec::from_noise_models(&noise_models)?;
/// let lookup = NoiseLookupTable::from_unified_specs(&unified_specs);
///
/// // O(1) parameter lookup
/// if let Some(params) = lookup.get_params(UncertaintyType::Inflow, 0, 5) {
///     println!("Inflow[0] season 5: mean={}, std_dev={}", params.mean, params.std_dev);
/// }
///
/// // Check temporal model type
/// if lookup.is_par_model(UncertaintyType::Inflow, 0) {
///     println!("Entity uses PAR model");
/// }
/// ```
///
/// # Design Rationale
///
/// Previous implementation searched through `Vec<NoiseModel>` for each lookup:
/// ```ignore
/// // OLD: O(n) search
/// let model = noise_models.iter()
///     .find(|m| m.uncertainty_type == unc_type && m.entity_id == entity)
///     .expect("Not found");
/// ```
///
/// New implementation uses pre-built HashMap:
/// ```ignore
/// // NEW: O(1) lookup
/// let params = lookup.get_params(unc_type, entity, season)?;
/// ```
///
/// For multi-entity problems with many stages, this reduces lookup overhead
/// from O(n × m × s) to O(m × s) where:
/// - n = number of noise model entries
/// - m = number of entities
/// - s = number of stages
#[derive(Debug, Clone)]
pub struct NoiseLookupTable {
    /// Flattened seasonal parameters for O(1) access
    ///
    /// Key: (UncertaintyType, entity_id, season_id)
    /// Value: (mean, std_dev, marginal_distribution)
    ///
    /// # Performance
    ///
    /// Pre-allocated with capacity = total number of (entity, season) pairs.
    /// Avoids rehashing during construction.
    params: HashMap<(UncertaintyType, usize, usize), SeasonalParams>,

    /// Temporal model type per entity
    ///
    /// Key: (UncertaintyType, entity_id)
    /// Value: true if PAR model, false if independent
    ///
    /// # Performance
    ///
    /// Small HashMap (one entry per entity). O(1) lookup to determine
    /// if entity uses temporal correlation.
    is_par: HashMap<(UncertaintyType, usize), bool>,
}

impl NoiseLookupTable {
    /// Build lookup table from unified noise specifications
    ///
    /// Pre-computes all seasonal parameters and temporal model types for
    /// O(1) access during scenario generation.
    ///
    /// # Arguments
    ///
    /// - `specs`: Unified noise specifications (one per entity)
    ///
    /// # Returns
    ///
    /// Lookup table with pre-computed indices
    ///
    /// # Performance
    ///
    /// - Time: O(n × s) where n = entities, s = avg seasons per entity
    /// - Space: O(n × s) for params HashMap, O(n) for is_par HashMap
    ///
    /// # Example
    ///
    /// ```ignore
    /// let specs = UnifiedNoiseSpec::from_noise_models(&noise_models)?;
    /// let lookup = NoiseLookupTable::from_unified_specs(&specs);
    /// ```
    pub fn from_unified_specs(
        specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
    ) -> Self {
        use crate::unified_noise_spec::TemporalModelSpec;

        // PERFORMANCE: Pre-compute total capacity to avoid rehashing
        let total_params: usize =
            specs.iter().map(|spec| spec.seasonal_params.len()).sum();

        let mut params = HashMap::with_capacity(total_params);
        let mut is_par = HashMap::with_capacity(specs.len());

        for spec in specs {
            let entity_key = (spec.uncertainty_type.clone(), spec.entity_id);

            // Store temporal model type
            let is_par_model = matches!(
                spec.temporal_model,
                TemporalModelSpec::PeriodicAutoregressive { .. }
            );
            is_par.insert(entity_key.clone(), is_par_model);

            // Flatten seasonal parameters into lookup table
            for (&season_id, season_params) in &spec.seasonal_params {
                let marginal = season_params
                    .marginal_override
                    .clone()
                    .or_else(|| spec.marginal_distribution.clone());

                params.insert(
                    (spec.uncertainty_type.clone(), spec.entity_id, season_id),
                    (season_params.mean, season_params.std_dev, marginal),
                );
            }
        }

        Self { params, is_par }
    }

    /// Get seasonal parameters for an entity (O(1) lookup)
    ///
    /// # Arguments
    ///
    /// - `uncertainty_type`: Inflow or Load
    /// - `entity_id`: Entity index (hydro_id or bus_id)
    /// - `season_id`: Season index
    ///
    /// # Returns
    ///
    /// - `Some((mean, std_dev, marginal))`: Parameters for this season
    /// - `None`: Season not defined for this entity
    ///
    /// # Performance
    ///
    /// O(1) average case via HashMap lookup
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some((mean, std_dev, marginal)) =
    ///     lookup.get_params(UncertaintyType::Inflow, 0, 5)
    /// {
    ///     println!("Season 5: μ={}, σ={}", mean, std_dev);
    /// }
    /// ```
    #[inline]
    pub fn get_params(
        &self,
        uncertainty_type: UncertaintyType,
        entity_id: usize,
        season_id: usize,
    ) -> Option<&SeasonalParams> {
        self.params.get(&(uncertainty_type, entity_id, season_id))
    }

    /// Check if entity uses PAR temporal model (O(1) lookup)
    ///
    /// # Arguments
    ///
    /// - `uncertainty_type`: Inflow or Load
    /// - `entity_id`: Entity index (hydro_id or bus_id)
    ///
    /// # Returns
    ///
    /// - `true`: Entity uses PAR model
    /// - `false`: Entity uses independent model or not found
    ///
    /// # Performance
    ///
    /// O(1) average case via HashMap lookup
    ///
    /// # Example
    ///
    /// ```ignore
    /// if lookup.is_par_model(UncertaintyType::Inflow, 0) {
    ///     println!("Entity 0 uses PAR temporal correlation");
    /// }
    /// ```
    #[inline]
    pub fn is_par_model(
        &self,
        uncertainty_type: UncertaintyType,
        entity_id: usize,
    ) -> bool {
        self.is_par
            .get(&(uncertainty_type, entity_id))
            .copied()
            .unwrap_or(false)
    }

    /// Get all parameters for entities of a specific type in a season (bulk retrieval)
    ///
    /// Returns a Vec of (entity_id, params) tuples for all entities of the given
    /// uncertainty type in the specified season. This is more efficient than calling
    /// `get_params()` repeatedly when processing many entities.
    ///
    /// # Arguments
    ///
    /// - `uncertainty_type`: Inflow or Load
    /// - `season_id`: Season index
    ///
    /// # Returns
    ///
    /// Vec of (entity_id, &SeasonalParams) for all matching entities
    ///
    /// # Performance
    ///
    /// - **Time**: O(n) where n = total number of (entity, season) pairs
    /// - **Space**: O(m) where m = number of matching entities
    /// - **Cache-friendly**: Returns Vec for contiguous iteration
    ///
    /// While this is O(n) in the total number of params, it's still faster than
    /// repeated HashMap lookups when processing many entities because:
    /// 1. Single HashMap iteration vs multiple lookups
    /// 2. Returned Vec enables cache-friendly iteration
    /// 3. Predictable access pattern for CPU prefetcher
    ///
    /// # Usage Pattern
    ///
    /// ```ignore
    /// // BEFORE: Multiple O(1) lookups (scattered memory access)
    /// for entity_id in 0..num_hydros {
    ///     if let Some(params) = lookup.get_params(UncertaintyType::Inflow, entity_id, season) {
    ///         process(params);
    ///     }
    /// }
    ///
    /// // AFTER: Single bulk retrieval (cache-friendly iteration)
    /// let all_inflow_params = lookup.get_all_params_for_season(UncertaintyType::Inflow, season);
    /// for (entity_id, params) in &all_inflow_params {
    ///     process(params);
    /// }
    /// ```
    ///
    /// # When to Use
    ///
    /// - Processing all entities of same type in a scenario generation loop
    /// - Iterating over entities in stage-by-stage scenario building
    /// - Pre-fetching parameters for cache locality
    ///
    /// # When NOT to Use
    ///
    /// - Looking up single entity (use `get_params()` instead)
    /// - Sparse entity access patterns
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Generate scenarios for all hydro entities in season 5
    /// let inflow_params = lookup.get_all_params_for_season(UncertaintyType::Inflow, 5);
    ///
    /// for (hydro_id, params) in &inflow_params {
    ///     for scenario in 0..num_scenarios {
    ///         let noise = sample_distribution(params, &mut rng);
    ///         scenarios[scenario][*hydro_id] = noise;
    ///     }
    /// }
    /// ```
    pub fn get_all_params_for_season(
        &self,
        uncertainty_type: UncertaintyType,
        season_id: usize,
    ) -> Vec<(usize, &SeasonalParams)> {
        // PERFORMANCE: Pre-allocate with estimated capacity
        // Typical case: 10-50 entities per type
        let mut result = Vec::with_capacity(32);

        for ((unc_type, entity_id, sid), params) in &self.params {
            if unc_type == &uncertainty_type && *sid == season_id {
                result.push((*entity_id, params));
            }
        }

        // PERFORMANCE: Sort by entity_id for predictable access pattern
        // Helps CPU prefetcher and cache locality
        result.sort_unstable_by_key(|(entity_id, _)| *entity_id);

        result
    }

    /// Get parameter count for performance analysis
    ///
    /// Returns the total number of (entity, season) parameter entries.
    /// Useful for benchmarking and memory profiling.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let lookup = NoiseLookupTable::from_unified_specs(&specs);
    /// println!("Lookup table size: {} entries", lookup.param_count());
    /// ```
    #[inline]
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Get entity count for a specific uncertainty type
    ///
    /// Returns the number of unique entities for the given uncertainty type.
    /// Useful for pre-allocation and performance analysis.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let num_hydros = lookup.entity_count(UncertaintyType::Inflow);
    /// let num_loads = lookup.entity_count(UncertaintyType::Load);
    /// ```
    pub fn entity_count(&self, uncertainty_type: UncertaintyType) -> usize {
        self.is_par
            .keys()
            .filter(|(unc_type, _)| unc_type == &uncertainty_type)
            .count()
    }
}

/// Scenario generator for 4-stage pipeline
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
pub struct ScenarioGenerator {
    /// Noise models for all entities (schema v2 format)
    noise_models: Vec<NoiseModel>,

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
#[allow(dead_code)]
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
    /// **DEPRECATED**: Use NoiseModelCache path instead
    ///
    /// This method is deprecated and will panic. The recommended path is now:
    /// 1. Use `recourse.get_unified_specs()` to get internal representation
    /// 2. Build `NoiseModelCache` from unified specs
    /// 3. Use `recourse.generate_sddp_noises_with_cache()` for scenario generation
    ///
    /// # Panics
    ///
    /// Always panics with migration instructions.
    pub fn from_recourse_input(
        _recourse: &Recourse,
        _initial_condition: &InitialCondition,
        _seed: u64,
    ) -> Result<Self, String> {
        panic!(
            "ScenarioGenerator::from_recourse_input() is deprecated after removal of NoiseModel.\n\
             Migration path:\n\
             1. Use recourse.get_unified_specs() to get UnifiedNoiseSpec\n\
             2. Build NoiseModelCache::from_unified_specs()\n\
             3. Use recourse.generate_sddp_noises_with_cache() for scenarios\n\
             \n\
             If you need direct ScenarioGenerator construction, use from_unified_specs() instead."
        );
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
    /// ## For Independent models:
    /// 1. **Base Noise**: Z ~ N(0,1) [BaseNoiseGenerator]
    /// 2. **Correlation**: W = L×Z [CorrelationApplicator]
    /// 3. **Marginal**: ε ~ F [MarginalTransformer]
    ///
    /// ## For PAR models:
    /// 1. **Base Noise**: Z ~ N(0,1) [BaseNoiseGenerator]
    /// 2. **Correlation**: W = L×Z [CorrelationApplicator]
    /// 3. **Residual Transform**: a ~ F [MarginalTransformer::transform_to_residuals]
    /// 4. **PAR Dynamics**: X = PAR(a) [PeriodicARGenerator]
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
        // Always use PAR pipeline (handles both Independent and PAR models)
        self.generate_stage_scenarios_par(stage_id, num_scenarios, current_lags)
    }

    /// PAR pipeline
    ///
    /// # Pipeline
    ///
    /// 1. **Base Noise**: Z ~ N(0,1)
    /// 2. **Correlation**: W = L×Z
    /// 3. **Residual Transform**: a ~ F (LogNormal3, etc.)
    /// 4. **PAR Dynamics**: X = μₘ + σₘ·[Σφₖₘ·aₜ₋ₖ + aₜ]
    ///
    /// # Note
    ///
    /// For PAR models, marginal distributions are applied to **residuals** (aₜ),
    /// not final values (Xₜ). This ensures non-negativity while preserving
    /// seasonal AR structure.
    #[allow(deprecated)] // Still supports deprecated AR models during soft deprecation (PAR-018)
    fn generate_stage_scenarios_par(
        &self,
        stage_id: usize,
        num_scenarios: usize,
        current_lags: &HashMap<usize, Vec<f64>>,
    ) -> (Vec<Vec<f64>>, HashMap<usize, Vec<f64>>) {
        use crate::par_generator::PeriodicARGenerator;
        use crate::seasonal_params::SeasonalParams;

        let num_entities = self.entity_marginals.len();

        // Stage 1: Base noise Z ~ N(0,1)
        let base_noise_generator = BaseNoiseGenerator::new(
            num_scenarios,
            num_entities,
            self.seed + stage_id as u64,
        );
        let base_noise = base_noise_generator.generate(self.base_noise_method);

        // Stage 2: Correlation W = L×Z
        let correlated = if self.correlation_blocks.is_empty() {
            base_noise
        } else {
            self.apply_correlation(&base_noise)
        };

        // Stage 3: Residual transform a ~ F
        let marginal_transformer =
            MarginalTransformer::new(self.entity_marginals.clone()).unwrap();
        let residuals =
            marginal_transformer.transform_to_residuals(&correlated);

        // Stage 4: PAR dynamics
        // Apply PAR generator to each entity that has PAR model
        let mut realizations = vec![vec![0.0; num_entities]; num_scenarios];
        let mut updated_lags = HashMap::new();

        for entity_idx in 0..num_entities {
            match &self.entity_temporal_models[entity_idx] {
                TemporalModel::PeriodicAutoregressive { .. } => {
                    // Extract seasonal params for this entity
                    let seasonal_params = SeasonalParams::try_from(
                        &self.entity_temporal_models[entity_idx],
                    )
                    .expect("Failed to extract SeasonalParams from PAR model");

                    // Get initial lags for this entity (or empty for cold start)
                    let initial_residuals = current_lags
                        .get(&entity_idx)
                        .cloned()
                        .unwrap_or_default();

                    // Create PAR generator for this entity
                    let mut par_generator = PeriodicARGenerator::new(
                        seasonal_params,
                        initial_residuals,
                    );

                    // Generate all scenarios for this entity
                    let mut new_lags = Vec::new();
                    for scenario_idx in 0..num_scenarios {
                        let a_t = residuals[scenario_idx][entity_idx];
                        let z_t = par_generator.generate_next(a_t);
                        realizations[scenario_idx][entity_idx] = z_t;

                        // Store last residual for lag buffer
                        if scenario_idx == num_scenarios - 1 {
                            // For simplicity, store last scenario's residuals as lags
                            // This is consistent with existing AR dynamics behavior
                            new_lags = par_generator
                                .get_residual_buffer()
                                .iter()
                                .copied()
                                .collect();
                        }
                    }

                    // Update lags for next stage
                    updated_lags.insert(entity_idx, new_lags);
                }
                TemporalModel::Independent => {
                    // Independent: no temporal dynamics, just copy residuals
                    for scenario_idx in 0..num_scenarios {
                        realizations[scenario_idx][entity_idx] =
                            residuals[scenario_idx][entity_idx];
                    }
                }
            }
        }

        (realizations, updated_lags)
    }

    /// Apply correlation transformation to base noise
    ///
    /// Helper method to deduplicate correlation code between standard and PAR pipelines.
    fn apply_correlation(&self, base_noise: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Build correlation blocks with global entity index mapping
        let blocks: Vec<crate::correlation_applicator::CorrelationBlock> = self
            .correlation_blocks
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

                let matrix = nalgebra::DMatrix::from_row_slice(
                    cb.correlation_matrix.len(),
                    cb.correlation_matrix[0].len(),
                    &cb.correlation_matrix
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>(),
                );

                crate::correlation_applicator::CorrelationBlock::new(entities, matrix)
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
        correlation_applicator.apply_correlation(base_noise)
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
    #[allow(dead_code)]
    fn build_entity_mappings(
        noise_models: &[NoiseModel],
    ) -> Result<EntityMappingResult, String> {
        let num_entities = noise_models.len();

        let mut entity_marginals = Vec::with_capacity(num_entities);
        let mut entity_temporal_models = Vec::with_capacity(num_entities);
        let mut entity_index_map = HashMap::new();
        let mut num_load_entities = 0;
        let mut num_inflow_entities = 0;

        for (global_idx, nm) in noise_models.iter().enumerate() {
            entity_marginals.push(nm.distribution.clone());
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    fn validate_ar_entities_have_lags(
        entity_temporal_models: &[TemporalModel],
        initial_lags: &HashMap<usize, Vec<f64>>,
    ) -> Result<(), String> {
        for (entity_idx, temporal_model) in
            entity_temporal_models.iter().enumerate()
        {
            if let TemporalModel::PeriodicAutoregressive {
                ar_orders,
                num_seasons: _,
                ..
            } = temporal_model
            {
                let lags = initial_lags.get(&entity_idx).ok_or(format!(
                    "Entity {} has PAR model but no initial lags",
                    entity_idx
                ))?;

                let max_lag_order =
                    ar_orders.iter().max().copied().unwrap_or(0);
                if lags.len() != max_lag_order {
                    return Err(format!(
                        "Entity {} PAR (max order {}) requires {} lags, got {}",
                        entity_idx,
                        max_lag_order,
                        max_lag_order,
                        lags.len()
                    ));
                }
            }
        }

        Ok(())
    }
    /// Validate that correlation blocks reference existing entities
    #[allow(dead_code)]
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
    pub(crate) fn new_empty() -> Self {
        Self {
            branching_samples: vec![],
            index_samplers: vec![],
        }
    }
}

#[cfg(test)]
mod noise_lookup_table_tests {
    use super::*;
    use crate::unified_noise_spec::{
        SeasonalNoiseParams, SeasonalPARParams, TemporalModelSpec,
        UnifiedNoiseSpec,
    };

    /// Helper: Build simple independent noise spec for testing
    fn build_independent_spec(
        entity_id: usize,
        uncertainty_type: UncertaintyType,
        seasons: &[(usize, f64, f64)],
    ) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::with_capacity(seasons.len());
        for &(season_id, mean, std_dev) in seasons {
            seasonal_params.insert(
                season_id,
                SeasonalNoiseParams {
                    mean,
                    std_dev,
                    marginal_override: None,
                },
            );
        }

        UnifiedNoiseSpec {
            uncertainty_type,
            entity_id,
            temporal_model: TemporalModelSpec::Independent,
            seasonal_params,
            marginal_distribution: Some(MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            }),
        }
    }

    /// Helper: Build PAR noise spec for testing
    fn build_par_spec(
        entity_id: usize,
        uncertainty_type: UncertaintyType,
        num_seasons: usize,
    ) -> UnifiedNoiseSpec {
        let mut seasonal_params = HashMap::with_capacity(num_seasons);
        let mut seasonal_ar_params = HashMap::with_capacity(num_seasons);

        for season in 0..num_seasons {
            seasonal_params.insert(
                season,
                SeasonalNoiseParams {
                    mean: 100.0 + (season as f64) * 10.0,
                    std_dev: 20.0,
                    marginal_override: None,
                },
            );

            seasonal_ar_params.insert(
                season,
                SeasonalPARParams {
                    ar_order: 1,
                    ar_coefficients: vec![0.7],
                },
            );
        }

        UnifiedNoiseSpec {
            uncertainty_type,
            entity_id,
            temporal_model: TemporalModelSpec::PeriodicAutoregressive {
                num_seasons,
                seasonal_ar_params,
            },
            seasonal_params,
            marginal_distribution: Some(MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 0.0,
                sigma: 0.6,
            }),
        }
    }

    #[test]
    fn test_lookup_table_construction_empty() {
        let specs: Vec<UnifiedNoiseSpec> = vec![];
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        // Should construct successfully with empty specs
        assert!(lookup.get_params(UncertaintyType::Inflow, 0, 0).is_none());
        assert!(!lookup.is_par_model(UncertaintyType::Inflow, 0));
    }

    #[test]
    fn test_lookup_table_independent_model() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0), (1, 120.0, 25.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        // Test O(1) parameter lookup
        let params0 = lookup
            .get_params(UncertaintyType::Load, 0, 0)
            .expect("Should find season 0");
        assert_eq!(params0.0, 100.0);
        assert_eq!(params0.1, 20.0);

        let params1 = lookup
            .get_params(UncertaintyType::Load, 0, 1)
            .expect("Should find season 1");
        assert_eq!(params1.0, 120.0);
        assert_eq!(params1.1, 25.0);

        // Missing season returns None
        assert!(lookup.get_params(UncertaintyType::Load, 0, 2).is_none());

        // Check temporal model type
        assert!(!lookup.is_par_model(UncertaintyType::Load, 0));
    }

    #[test]
    fn test_lookup_table_par_model() {
        let spec = build_par_spec(0, UncertaintyType::Inflow, 12);
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        // Test PAR model detection
        assert!(lookup.is_par_model(UncertaintyType::Inflow, 0));

        // Test parameter lookup for multiple seasons
        for season in 0..12 {
            let params = lookup
                .get_params(UncertaintyType::Inflow, 0, season)
                .unwrap_or_else(|| panic!("Should find season {}", season));
            assert_eq!(params.0, 100.0 + (season as f64) * 10.0);
            assert_eq!(params.1, 20.0);
        }

        // Season outside range returns None
        assert!(lookup.get_params(UncertaintyType::Inflow, 0, 12).is_none());
    }

    #[test]
    fn test_lookup_table_multiple_entities() {
        let specs = vec![
            build_independent_spec(
                0,
                UncertaintyType::Load,
                &[(0, 50.0, 10.0)],
            ),
            build_independent_spec(
                1,
                UncertaintyType::Load,
                &[(0, 60.0, 12.0)],
            ),
            build_par_spec(0, UncertaintyType::Inflow, 3),
        ];

        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        // Load entity 0
        let load0 = lookup.get_params(UncertaintyType::Load, 0, 0).unwrap();
        assert_eq!(load0.0, 50.0);

        // Load entity 1
        let load1 = lookup.get_params(UncertaintyType::Load, 1, 0).unwrap();
        assert_eq!(load1.0, 60.0);

        // Inflow entity 0 (PAR)
        let inflow0 = lookup.get_params(UncertaintyType::Inflow, 0, 0).unwrap();
        assert_eq!(inflow0.0, 100.0);
        assert!(lookup.is_par_model(UncertaintyType::Inflow, 0));

        // Different uncertainty types don't interfere
        assert!(lookup.get_params(UncertaintyType::Inflow, 0, 0).is_some());
        assert!(lookup.get_params(UncertaintyType::Load, 0, 0).is_some());
    }

    #[test]
    fn test_lookup_table_missing_entity() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        // Non-existent entity returns None (not panic)
        assert!(lookup.get_params(UncertaintyType::Load, 999, 0).is_none());
        assert!(!lookup.is_par_model(UncertaintyType::Load, 999));

        // Wrong uncertainty type returns None
        assert!(lookup.get_params(UncertaintyType::Inflow, 0, 0).is_none());
    }

    #[test]
    fn test_lookup_table_marginal_distribution() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        let params = lookup.get_params(UncertaintyType::Load, 0, 0).unwrap();
        assert!(params.2.is_some()); // Has marginal distribution

        // Should be Normal(0, 1) as set in helper
        if let Some(MarginalDistribution::Normal { mean, std_dev }) = &params.2
        {
            assert_eq!(*mean, 0.0);
            assert_eq!(*std_dev, 1.0);
        } else {
            panic!("Expected Normal distribution");
        }
    }

    #[test]
    fn test_lookup_table_performance_capacity() {
        // Test that pre-allocation works correctly (no panics)
        let mut specs = Vec::new();
        for entity in 0..100 {
            specs.push(build_independent_spec(
                entity,
                UncertaintyType::Load,
                &[(0, 100.0, 20.0)],
            ));
        }

        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        // All entities should be accessible
        for entity in 0..100 {
            assert!(lookup
                .get_params(UncertaintyType::Load, entity, 0)
                .is_some());
        }
    }

    #[test]
    fn test_bulk_retrieval_empty() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        // Season without any entities
        let result =
            lookup.get_all_params_for_season(UncertaintyType::Inflow, 0);
        assert!(result.is_empty());

        // Season that doesn't exist
        let result =
            lookup.get_all_params_for_season(UncertaintyType::Load, 99);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bulk_retrieval_single_entity() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        let result = lookup.get_all_params_for_season(UncertaintyType::Load, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0); // entity_id
        assert_eq!(result[0].1 .0, 100.0); // mean
        assert_eq!(result[0].1 .1, 20.0); // std_dev
    }

    #[test]
    fn test_bulk_retrieval_multiple_entities() {
        let specs = vec![
            build_independent_spec(
                2,
                UncertaintyType::Inflow,
                &[(0, 50.0, 10.0)],
            ),
            build_independent_spec(
                0,
                UncertaintyType::Inflow,
                &[(0, 100.0, 20.0)],
            ),
            build_independent_spec(
                1,
                UncertaintyType::Inflow,
                &[(0, 75.0, 15.0)],
            ),
        ];
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        let result =
            lookup.get_all_params_for_season(UncertaintyType::Inflow, 0);

        // Should return all 3 entities
        assert_eq!(result.len(), 3);

        // Should be sorted by entity_id
        assert_eq!(result[0].0, 0);
        assert_eq!(result[0].1 .0, 100.0);

        assert_eq!(result[1].0, 1);
        assert_eq!(result[1].1 .0, 75.0);

        assert_eq!(result[2].0, 2);
        assert_eq!(result[2].1 .0, 50.0);
    }

    #[test]
    fn test_bulk_retrieval_mixed_uncertainty_types() {
        let specs = vec![
            build_independent_spec(
                0,
                UncertaintyType::Inflow,
                &[(0, 100.0, 20.0)],
            ),
            build_independent_spec(
                0,
                UncertaintyType::Load,
                &[(0, 50.0, 10.0)],
            ),
            build_independent_spec(
                1,
                UncertaintyType::Inflow,
                &[(0, 120.0, 25.0)],
            ),
        ];
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        // Get only inflow entities
        let inflow_result =
            lookup.get_all_params_for_season(UncertaintyType::Inflow, 0);
        assert_eq!(inflow_result.len(), 2);
        assert_eq!(inflow_result[0].0, 0);
        assert_eq!(inflow_result[1].0, 1);

        // Get only load entities
        let load_result =
            lookup.get_all_params_for_season(UncertaintyType::Load, 0);
        assert_eq!(load_result.len(), 1);
        assert_eq!(load_result[0].0, 0);
        assert_eq!(load_result[0].1 .0, 50.0);
    }

    #[test]
    fn test_bulk_retrieval_multiple_seasons() {
        let spec = build_independent_spec(
            0,
            UncertaintyType::Load,
            &[(0, 100.0, 20.0), (1, 120.0, 25.0), (2, 90.0, 18.0)],
        );
        let lookup = NoiseLookupTable::from_unified_specs(&[spec]);

        // Each season should return its own parameters
        let result0 =
            lookup.get_all_params_for_season(UncertaintyType::Load, 0);
        assert_eq!(result0.len(), 1);
        assert_eq!(result0[0].1 .0, 100.0);

        let result1 =
            lookup.get_all_params_for_season(UncertaintyType::Load, 1);
        assert_eq!(result1.len(), 1);
        assert_eq!(result1[0].1 .0, 120.0);

        let result2 =
            lookup.get_all_params_for_season(UncertaintyType::Load, 2);
        assert_eq!(result2.len(), 1);
        assert_eq!(result2[0].1 .0, 90.0);
    }

    #[test]
    fn test_param_count() {
        let specs = vec![
            build_independent_spec(
                0,
                UncertaintyType::Load,
                &[(0, 100.0, 20.0), (1, 120.0, 25.0)],
            ),
            build_par_spec(0, UncertaintyType::Inflow, 3),
        ];
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        // Load: 2 seasons, Inflow PAR: 3 seasons = 5 total
        assert_eq!(lookup.param_count(), 5);
    }

    #[test]
    fn test_entity_count() {
        let specs = vec![
            build_independent_spec(
                0,
                UncertaintyType::Load,
                &[(0, 100.0, 20.0)],
            ),
            build_independent_spec(
                1,
                UncertaintyType::Load,
                &[(0, 50.0, 10.0)],
            ),
            build_par_spec(0, UncertaintyType::Inflow, 3),
            build_par_spec(1, UncertaintyType::Inflow, 3),
            build_par_spec(2, UncertaintyType::Inflow, 3),
        ];
        let lookup = NoiseLookupTable::from_unified_specs(&specs);

        assert_eq!(lookup.entity_count(UncertaintyType::Load), 2);
        assert_eq!(lookup.entity_count(UncertaintyType::Inflow), 3);
    }
}
