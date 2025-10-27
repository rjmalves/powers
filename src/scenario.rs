use rand::prelude::*;
use rand_distr;
use rand_xoshiro;
use std::collections::HashMap;

use crate::input::{MarginalDistribution, UncertaintyType};
use crate::unified_noise_spec::{TemporalModelSpec, UnifiedNoiseSpec};

/// Simple scenario generator for stage-wise noise sampling
///
/// A lightweight scenario generator that samples from distribution vectors
/// to create Sample Average Approximation (SAA) trees. Useful for:
/// - Unit tests with controlled, deterministic scenarios
/// - Benchmarking with simple distribution patterns
/// - Examples demonstrating basic SDDP concepts
///
/// See `Recourse::generate_sddp_noises()` for the production path.
///
/// # Example
///
/// ```
/// use powers_rs::scenario::NoiseGenerator;
/// use rand_distr::Normal;
///
/// let mut generator = NoiseGenerator::new();
/// generator.add_node_generator(
///     vec![Normal::new(100.0, 20.0).unwrap()],  // Load distributions
///     vec![Normal::new(50.0, 10.0).unwrap()],   // Inflow distributions
///     10  // Number of scenarios
/// );
/// let saa = generator.generate(42);  // Generate with seed
/// ```
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

/// Scenario generator that orchestrates multi-stage noise sampling
///
/// Manages a collection of `NodeNoiseGenerator` instances (one per stage)
/// and generates complete Sample Average Approximation (SAA) trees.
///
/// # Use Cases
///
/// - **Testing**: Simple, deterministic scenario generation for unit tests
/// - **Prototyping**: Quick setup without complex configuration
/// - **Benchmarking**: Controlled scenario patterns for performance testing
///
/// # Production Alternative
///
/// For production SDDP runs, use `NoiseModelCache` via `Recourse::generate_sddp_noises()`:
/// - Optimized caching layer
/// - Full support for PAR models, correlation, and advanced distributions
/// - Season-aware parameter management
///
/// # Example
///
/// ```ignore
/// use powers_rs::scenario::NoiseGenerator;
/// use rand_distr::{Normal, LogNormal};
///
/// let mut generator = NoiseGenerator::new();
///
/// // Stage 0: Add first stage scenarios
/// generator.add_node_generator(
///     vec![Normal::new(100.0, 20.0).unwrap()],  // Load
///     vec![LogNormal::new(4.5, 0.3).unwrap()],  // Inflow
///     10  // 10 scenarios
/// );
///
/// // Stage 1: Add second stage scenarios  
/// generator.add_node_generator(
///     vec![Normal::new(110.0, 25.0).unwrap()],
///     vec![LogNormal::new(4.5, 0.3).unwrap()],
///     5  // 5 scenarios per branch
/// );
///
/// let saa = generator.generate(42);  // Deterministic with seed
/// ```
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

    /// Generates a Sample Average Approximation (SAA) from configured distributions
    ///
    /// Samples noise values from the distribution vectors for each stage and scenario,
    /// creating a complete scenario tree structure. Uses the provided seed for
    /// deterministic, reproducible scenario generation.
    ///
    /// # Example
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

    /// Create empty SAA (for NoiseModelCache pipeline to populate stage-by-stage)
    pub(crate) fn new_empty() -> Self {
        Self {
            branching_samples: vec![],
            index_samplers: vec![],
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
#[derive(Debug, Clone)]
pub struct NoiseLookupTable {
    /// Flattened seasonal parameters for O(1) access
    params: HashMap<(UncertaintyType, usize, usize), SeasonalParams>,

    /// Temporal model type per entity
    is_par: HashMap<(UncertaintyType, usize), bool>,
}

impl NoiseLookupTable {
    /// Build lookup table from unified noise specifications
    ///
    /// Pre-computes all seasonal parameters and temporal model types for
    /// O(1) access during scenario generation.
    pub fn from_unified_specs(specs: &[UnifiedNoiseSpec]) -> Self {
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
