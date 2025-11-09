use rand::prelude::*;
use rand_distr;
use rand_xoshiro;

/// Method used to generate scenario tree
#[derive(Debug, Clone)]
pub enum ScenarioGenerationMethod {
    /// Sample Average Approximation with fixed samples
    SAA { num_samples: usize },
    /// Loaded from external file
    External { source: String },
    /// Custom generation method
    Custom { description: String },
}

/// Metadata about scenario tree generation
#[derive(Debug, Clone)]
pub struct ScenarioTreeMetadata {
    /// Generation method used
    pub generation_method: ScenarioGenerationMethod,
    /// Random seed used for generation
    pub seed: u64,
    /// Timestamp when tree was generated (RFC 3339 format)
    pub generated_at: String,
    /// Number of stages in the tree
    pub num_stages: usize,
}

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
    pub load_distributions: Vec<L>, // indexed by bus_id
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
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().get_load_innovations().len(), num_entities);
    /// assert_eq!(saa.get_noises_by_stage_and_branching(0, 0).unwrap().get_inflow_innovations().len(), num_entities);
    ///
    /// ```
    pub fn generate(&self, seed: u64) -> ScenarioTree {
        let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(seed);

        let mut tree = ScenarioTree::new(self, seed);
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

            tree.set_noises_by_stage(
                stage_id,
                stage_generator.num_branchings,
                stage_generator.num_load_entities,
                stage_generator.num_inflow_entities,
                load_noises,
                inflow_noises,
            );
        }

        tree
    }
}

#[derive(Debug, Clone)]
pub struct OptimizedSampledBranchingNoises {
    pub load_innovations: Vec<f64>,
    pub inflow_innovations: Vec<f64>,
    pub num_load_entities: usize,
    pub num_inflow_entities: usize,
}

impl OptimizedSampledBranchingNoises {
    pub fn new(num_load_entities: usize, num_inflow_entities: usize) -> Self {
        Self {
            load_innovations: Vec::with_capacity(num_load_entities),
            inflow_innovations: Vec::with_capacity(num_inflow_entities),
            num_load_entities,
            num_inflow_entities,
        }
    }

    #[inline]
    pub fn get_load_innovations(&self) -> &[f64] {
        &self.load_innovations
    }

    #[inline]
    pub fn get_inflow_innovations(&self) -> &[f64] {
        &self.inflow_innovations
    }

    /// Get all innovations in unified order: [loads..., inflows...]
    ///
    /// This method provides a unified view of all innovations for the v2 API.
    /// The returned vector has innovations in order: loads first, then inflows.
    ///
    /// # Returns
    ///
    /// Combined innovations vector: [η_load[0], ..., η_load[n], η_inflow[0], ..., η_inflow[m]]
    ///
    /// # Note
    ///
    /// This creates a temporary allocation. For high-performance code, consider
    /// restructuring OptimizedSampledBranchingNoises to store a single unified vector.
    pub fn get_all_innovations(&self) -> Vec<f64> {
        let mut all_innovations = Vec::with_capacity(
            self.num_load_entities + self.num_inflow_entities,
        );
        all_innovations.extend_from_slice(&self.load_innovations);
        all_innovations.extend_from_slice(&self.inflow_innovations);
        all_innovations
    }

    /// Compute observations from residuals (lazy, only when needed for output)
    pub fn compute_observations(
        &self,
        seasonal_means: &[f64],
        seasonal_stds: &[f64],
    ) -> Vec<f64> {
        self.inflow_innovations
            .iter()
            .enumerate()
            .map(|(i, &z_prime)| seasonal_means[i] + seasonal_stds[i] * z_prime)
            .collect()
    }

    pub fn set_load_innovations(&mut self, innovations: &[f64]) {
        self.load_innovations.clear();
        self.load_innovations.extend_from_slice(innovations);
    }

    pub fn set_inflow_data(&mut self, innovations: &[f64]) {
        self.inflow_innovations.clear();
        self.inflow_innovations.extend_from_slice(innovations);
    }
}

#[derive(Debug, Clone)]
pub struct SampledNodeBranchings {
    pub num_branchings: usize,
    pub branching_noises: Vec<OptimizedSampledBranchingNoises>,
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
                OptimizedSampledBranchingNoises::new(
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
    ) -> Option<&OptimizedSampledBranchingNoises> {
        self.branching_noises.get(branching_id)
    }

    pub fn set_noises_by_branching(
        &mut self,
        branching_id: usize,
        load_innovations: &[f64],
        inflow_innovations: &[f64],
    ) {
        let noise = self.branching_noises.get_mut(branching_id).unwrap();
        noise.set_load_innovations(load_innovations);
        noise.set_inflow_data(inflow_innovations);
    }
}

#[derive(Debug)]
/// Scenario tree representation with metadata
///
/// A generic scenario tree structure that can be generated through various methods
/// (SAA, external files, custom generators). Tracks generation provenance for
/// reproducibility and debugging.
///
/// # Example
///
/// ```ignore
/// use powers_rs::scenario::{ScenarioTree, ScenarioTreeMetadata, ScenarioGenerationMethod};
///
/// let tree = ScenarioTree {
///     stage_scenarios: branching_samples,
///     index_samplers,
///     metadata: ScenarioTreeMetadata {
///         generation_method: ScenarioGenerationMethod::SAA { num_samples: 100 },
///         seed: 42,
///         generated_at: "2024-01-01T00:00:00Z".to_string(),
///         num_stages: 12,
///     },
/// };
/// ```
pub struct ScenarioTree {
    /// Sampled noise branchings for each stage
    pub stage_scenarios: Vec<SampledNodeBranchings>,
    /// Uniform samplers for scenario selection
    pub index_samplers: Vec<rand_distr::Uniform<usize>>,
    /// Metadata about tree generation
    pub metadata: ScenarioTreeMetadata,
}

impl ScenarioTree {
    pub fn new<
        L: rand_distr::Distribution<f64>,
        I: rand_distr::Distribution<f64>,
    >(
        scenario_generator: &NoiseGenerator<L, I>,
        seed: u64,
    ) -> Self {
        let stage_scenarios: Vec<SampledNodeBranchings> = scenario_generator
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

        let num_stages = scenario_generator.node_generators.len();
        let num_samples = scenario_generator
            .node_generators
            .first()
            .map(|g| g.num_branchings)
            .unwrap_or(0);

        Self {
            stage_scenarios,
            index_samplers,
            metadata: ScenarioTreeMetadata {
                generation_method: ScenarioGenerationMethod::SAA {
                    num_samples,
                },
                seed,
                generated_at: chrono::Utc::now().to_rfc3339(),
                num_stages,
            },
        }
    }

    /// Create empty ScenarioTree (for NoiseModelCache pipeline to populate stage-by-stage)
    pub fn new_empty() -> Self {
        Self {
            stage_scenarios: vec![],
            index_samplers: vec![],
            metadata: ScenarioTreeMetadata {
                generation_method: ScenarioGenerationMethod::Custom {
                    description: "Empty tree for incremental population"
                        .to_string(),
                },
                seed: 0,
                generated_at: chrono::Utc::now().to_rfc3339(),
                num_stages: 0,
            },
        }
    }

    pub fn get_branching_count_at_stage(
        &self,
        stage_id: usize,
    ) -> Option<usize> {
        Some(self.stage_scenarios.get(stage_id)?.num_branchings)
    }

    pub fn get_noises_by_stage_and_branching(
        &self,
        stage_id: usize,
        branching_id: usize,
    ) -> Option<&OptimizedSampledBranchingNoises> {
        self.stage_scenarios
            .get(stage_id)?
            .get_noises_by_branching(branching_id)
    }

    pub fn sample_scenario(
        &self,
        rng: &mut rand_xoshiro::Xoshiro256Plus,
    ) -> Vec<&OptimizedSampledBranchingNoises> {
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
        load_innovations: Vec<Vec<f64>>,
        inflow_innovations: Vec<Vec<f64>>,
    ) {
        // Ensure we have enough stages (extend if necessary)
        while self.stage_scenarios.len() <= stage_id {
            self.stage_scenarios.push(SampledNodeBranchings {
                num_branchings: 0,
                branching_noises: vec![],
            });
        }

        // Initialize the stage with the correct number of branchings
        self.stage_scenarios[stage_id] = SampledNodeBranchings {
            num_branchings,
            branching_noises: vec![
                OptimizedSampledBranchingNoises::new(
                    num_load_entities,
                    num_inflow_entities
                );
                num_branchings
            ],
        };

        // Fill in the noise values
        for branching_id in 0..num_branchings {
            let mut branching_load_innovations =
                Vec::<f64>::with_capacity(num_load_entities);
            for entity_id in 0..num_load_entities {
                branching_load_innovations.push(
                    *load_innovations
                        .get(entity_id)
                        .unwrap()
                        .get(branching_id)
                        .unwrap(),
                );
            }
            let mut branching_inflow_innovations =
                Vec::<f64>::with_capacity(num_inflow_entities);

            for entity_id in 0..num_inflow_entities {
                branching_inflow_innovations.push(
                    *inflow_innovations
                        .get(entity_id)
                        .unwrap()
                        .get(branching_id)
                        .unwrap(),
                );
            }

            self.stage_scenarios
                .get_mut(stage_id)
                .unwrap()
                .set_noises_by_branching(
                    branching_id,
                    branching_load_innovations.as_slice(),
                    branching_inflow_innovations.as_slice(),
                );
        }

        // Update metadata
        self.metadata.num_stages = self.stage_scenarios.len();
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
        assert_eq!(scenario[0].load_innovations.len(), num_entities);
        assert_eq!(scenario[0].inflow_innovations.len(), num_entities);
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
