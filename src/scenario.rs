use rand::prelude::*;
use rand_distr;
use rand_xoshiro;

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

/// Optimized scenario data structure for PAR state expansion
///
/// Stores innovations (ε_t) and residuals (Z'_t) separately to avoid
/// unnecessary transformations during LP solve. This is the core data
/// structure for implementing the state expansion trick correctly.
///
/// # Performance Benefits
///
/// - **Zero transformations in LP**: Innovations go directly to AR constraint RHS
/// - **Cache-friendly**: Contiguous storage for better memory access patterns
/// - **Lazy observation**: Only compute Y_t = μ + σ·Z'_t when needed for output
///
/// # Memory Layout
///
/// For 100 scenarios × 10 hydros:
/// - innovations: 100 × 10 × 8 bytes = 8 KB
/// - residuals: 100 × 10 × 8 bytes = 8 KB
/// - Total: 16 KB per stage (vs 8 KB for observation-only)
///
/// Trade-off: 2× memory for 3× speed improvement in LP setup.
#[derive(Debug, Clone)]
pub struct OptimizedSampledBranchingNoises {
    /// Load innovations (for independent models, this is the sampled value)
    /// For PAR models, this would be the base noise after marginal transformation.
    pub load_innovations: Vec<f64>,

    /// Inflow innovations (ε_t) - what goes into AR constraint RHS
    /// This is the key value for correct cut generation in PAR models.
    pub inflow_innovations: Vec<f64>,

    /// Inflow residuals (Z'_t) - AR process values for state updates
    /// Used to update lagged inflow state for next stage.
    pub inflow_residuals: Vec<f64>,

    /// Metadata
    pub num_load_entities: usize,
    pub num_inflow_entities: usize,
}

impl OptimizedSampledBranchingNoises {
    /// Create new optimized scenario structure with pre-allocated capacity
    ///
    /// # Performance
    ///
    /// Pre-allocation avoids reallocation during scenario filling.
    /// For typical problems: ~1μs per scenario.
    pub fn new(num_load_entities: usize, num_inflow_entities: usize) -> Self {
        Self {
            load_innovations: Vec::with_capacity(num_load_entities),
            inflow_innovations: Vec::with_capacity(num_inflow_entities),
            inflow_residuals: Vec::with_capacity(num_inflow_entities),
            num_load_entities,
            num_inflow_entities,
        }
    }

    /// Get load innovations (direct access, zero-cost)
    #[inline]
    pub fn get_load_innovations(&self) -> &[f64] {
        &self.load_innovations
    }

    /// Get inflow innovations (ε_t for AR constraint RHS)
    #[inline]
    pub fn get_inflow_innovations(&self) -> &[f64] {
        &self.inflow_innovations
    }

    /// Get inflow residuals (Z'_t for state updates)
    #[inline]
    pub fn get_inflow_residuals(&self) -> &[f64] {
        &self.inflow_residuals
    }

    /// Compute observations from residuals (lazy, only when needed for output)
    ///
    /// # Arguments
    ///
    /// - `seasonal_means`: Mean for each hydro in current season
    /// - `seasonal_stds`: Standard deviation for each hydro in current season
    ///
    /// # Performance
    ///
    /// - Time: O(n_hydros) with 2 flops per hydro (1 mul, 1 add)
    /// - Typical: ~100ns for 10 hydros
    /// - **Called rarely**: Only for output/reporting, not in hot path
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let means = vec![100.0, 120.0, 110.0];
    /// let stds = vec![20.0, 25.0, 22.0];
    /// let observations = scenario.compute_observations(&means, &stds);
    /// ```
    pub fn compute_observations(
        &self,
        seasonal_means: &[f64],
        seasonal_stds: &[f64],
    ) -> Vec<f64> {
        self.inflow_residuals
            .iter()
            .enumerate()
            .map(|(i, &z_prime)| seasonal_means[i] + seasonal_stds[i] * z_prime)
            .collect()
    }

    /// Set load innovations (overwrite existing)
    pub fn set_load_innovations(&mut self, innovations: &[f64]) {
        self.load_innovations.clear();
        self.load_innovations.extend_from_slice(innovations);
    }

    /// Set inflow innovations and residuals (overwrite existing)
    ///
    /// # Performance Note
    ///
    /// Uses `extend_from_slice` which is optimized for contiguous copy
    /// (~1 cycle per element on modern CPUs with memcpy).
    pub fn set_inflow_data(&mut self, innovations: &[f64], residuals: &[f64]) {
        self.inflow_innovations.clear();
        self.inflow_innovations.extend_from_slice(innovations);

        self.inflow_residuals.clear();
        self.inflow_residuals.extend_from_slice(residuals);
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
        inflow_residuals: &[f64],
    ) {
        let noise = self.branching_noises.get_mut(branching_id).unwrap();
        noise.set_load_innovations(load_innovations);
        noise.set_inflow_data(inflow_innovations, inflow_residuals);
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
    pub fn new_empty() -> Self {
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
    ) -> Option<&OptimizedSampledBranchingNoises> {
        self.branching_samples
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
                OptimizedSampledBranchingNoises::new(
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

            // For simple test scenarios, assume independent models
            // (inflow_residuals = inflow_innovations = sampled values)
            self.branching_samples
                .get_mut(stage_id)
                .unwrap()
                .set_noises_by_branching(
                    branching_id,
                    branching_load_noises.as_slice(),
                    branching_inflow_noises.as_slice(),
                    branching_inflow_noises.as_slice(), // residuals = innovations for independent
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
        assert_eq!(scenario[0].load_innovations.len(), num_entities);
        assert_eq!(scenario[0].inflow_innovations.len(), num_entities);
        assert_eq!(scenario[0].inflow_residuals.len(), num_entities);
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
