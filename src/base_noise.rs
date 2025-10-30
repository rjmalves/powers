/// Base Noise Generator - Stage 1 of Scenario Generation Pipeline
///
/// Generates independent standard normal samples Z ~ N(0,1) that serve as the
/// foundation for the multi-stage scenario generation pipeline:
///
/// 1. **Base Noise** (this module): Z ~ N(0,1) independent samples
/// 2. **Correlation**: W = L×Z with Cholesky transformation
/// 3. **Marginal**: Transform to target distributions (Normal/LogNormal3)
/// 4. **Temporal**: Apply AR dynamics Xₜ = Σφᵢ Xₜ₋ᵢ + εₜ
///
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rand_xoshiro::Xoshiro256Plus;

/// Method for generating base noise samples
///
/// Currently only standard Monte Carlo sampling is implemented.
/// Future variance reduction methods may be added as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseNoiseMethod {
    Standard,
}

/// Generator for independent standard normal samples
///
/// Creates Z ~ N(0,1) samples for the base noise stage of scenario generation.
/// Provides deterministic output for a given seed.
///
/// # Example
///
/// ```
/// use powers_rs::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
///
/// let generator = BaseNoiseGenerator::new(1000, 10, 42);
/// let noise = generator.generate(BaseNoiseMethod::Standard);
///
/// // Verify statistical properties
/// let mean: f64 = noise.iter().flatten().sum::<f64>() / (1000.0 * 10.0);
/// assert!((mean.abs()) < 0.1); // Mean ≈ 0
/// ```
pub struct BaseNoiseGenerator {
    num_scenarios: usize,
    num_entities: usize,
    seed: u64,
}

impl BaseNoiseGenerator {
    pub fn new(num_scenarios: usize, num_entities: usize, seed: u64) -> Self {
        Self {
            num_scenarios,
            num_entities,
            seed,
        }
    }

    /// Generate standard normal samples Z ~ N(0,1)
    ///
    /// Returns a matrix of samples indexed by [scenario][entity].
    /// All samples are independent (no correlation applied at this stage).
    ///
    /// # Example
    ///
    /// ```
    /// use powers_rs::base_noise::{BaseNoiseGenerator, BaseNoiseMethod};
    ///
    /// let gen = BaseNoiseGenerator::new(100, 3, 42);
    /// let noise = gen.generate(BaseNoiseMethod::Standard);
    ///
    /// assert_eq!(noise.len(), 100);
    /// assert_eq!(noise[0].len(), 3);
    /// ```
    pub fn generate(&self, method: BaseNoiseMethod) -> Vec<Vec<f64>> {
        // Validation
        self.validate_inputs();

        // Dispatch to appropriate method
        match method {
            BaseNoiseMethod::Standard => self.generate_standard(),
        }
    }

    fn validate_inputs(&self) {
        assert!(
            self.num_scenarios > 0,
            "num_scenarios must be > 0, got {}",
            self.num_scenarios
        );
        assert!(
            self.num_entities > 0,
            "num_entities must be > 0, got {}",
            self.num_entities
        );
    }

    fn generate_standard(&self) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256Plus::seed_from_u64(self.seed);
        let standard_normal = StandardNormal;

        let mut scenarios = Vec::with_capacity(self.num_scenarios);

        for _ in 0..self.num_scenarios {
            let mut scenario = Vec::with_capacity(self.num_entities);

            for _ in 0..self.num_entities {
                scenario.push(standard_normal.sample(&mut rng));
            }

            scenarios.push(scenario);
        }

        scenarios
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_generate_standard_normals_dimensions() {
        // Test: Correct dimensions [scenarios × entities]
        let gen = BaseNoiseGenerator::new(100, 3, 42);
        let noise = gen.generate(BaseNoiseMethod::Standard);

        assert_eq!(noise.len(), 100, "Should have 100 scenarios");
        for scenario in &noise {
            assert_eq!(
                scenario.len(),
                3,
                "Each scenario should have 3 entities"
            );
        }
    }

    #[test]
    fn test_generate_statistical_properties() {
        // Test: Mean ≈ 0, std ≈ 1 for large sample
        let gen = BaseNoiseGenerator::new(10000, 10, 42);
        let noise = gen.generate(BaseNoiseMethod::Standard);

        // Flatten all samples
        let samples: Vec<f64> = noise.iter().flatten().copied().collect();
        let n = samples.len() as f64;

        let mean = samples.iter().sum::<f64>() / n;

        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // With 100,000 samples, mean should be very close to 0
        assert!(mean.abs() < 0.05, "Sample mean should be ≈ 0, got {}", mean);

        // Standard deviation should be close to 1
        assert!(
            (std_dev - 1.0).abs() < 0.05,
            "Sample std should be ≈ 1, got {}",
            std_dev
        );
    }

    #[test]
    fn test_deterministic_generation() {
        // Test: Same seed produces identical samples
        let gen1 = BaseNoiseGenerator::new(100, 5, 42);
        let gen2 = BaseNoiseGenerator::new(100, 5, 42);

        let noise1 = gen1.generate(BaseNoiseMethod::Standard);
        let noise2 = gen2.generate(BaseNoiseMethod::Standard);

        for (s1, s2) in noise1.iter().zip(noise2.iter()) {
            for (v1, v2) in s1.iter().zip(s2.iter()) {
                assert_eq!(
                    v1, v2,
                    "Same seed should produce identical samples"
                );
            }
        }
    }

    #[test]
    fn test_different_seeds_produce_different_samples() {
        // Test: Different seeds produce different samples
        let gen1 = BaseNoiseGenerator::new(100, 5, 42);
        let gen2 = BaseNoiseGenerator::new(100, 5, 43);

        let noise1 = gen1.generate(BaseNoiseMethod::Standard);
        let noise2 = gen2.generate(BaseNoiseMethod::Standard);

        // Check that at least some samples are different
        let mut differences = 0;
        for (s1, s2) in noise1.iter().zip(noise2.iter()) {
            for (v1, v2) in s1.iter().zip(s2.iter()) {
                if (v1 - v2).abs() > 1e-10 {
                    differences += 1;
                }
            }
        }

        assert!(
            differences > 400,
            "Different seeds should produce mostly different samples, got {} differences",
            differences
        );
    }

    #[test]
    fn test_entity_independence() {
        // Test: Low correlation between different entities
        let gen = BaseNoiseGenerator::new(1000, 2, 42);
        let noise = gen.generate(BaseNoiseMethod::Standard);

        let entity0: Vec<f64> = noise.iter().map(|s| s[0]).collect();
        let entity1: Vec<f64> = noise.iter().map(|s| s[1]).collect();

        let n = entity0.len() as f64;
        let mean0 = entity0.iter().sum::<f64>() / n;
        let mean1 = entity1.iter().sum::<f64>() / n;

        let cov: f64 = entity0
            .iter()
            .zip(entity1.iter())
            .map(|(x, y)| (x - mean0) * (y - mean1))
            .sum::<f64>()
            / n;

        let var0 = entity0.iter().map(|x| (x - mean0).powi(2)).sum::<f64>() / n;
        let var1 = entity1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / n;

        let correlation = cov / (var0.sqrt() * var1.sqrt());

        assert!(
            correlation.abs() < 0.1,
            "Entities should be independent (low correlation), got {}",
            correlation
        );
    }

    #[test]
    #[should_panic(expected = "num_scenarios must be > 0")]
    fn test_validation_zero_scenarios() {
        // Test: Validation rejects zero scenarios
        let gen = BaseNoiseGenerator::new(0, 10, 42);
        gen.generate(BaseNoiseMethod::Standard);
    }

    #[test]
    #[should_panic(expected = "num_entities must be > 0")]
    fn test_validation_zero_entities() {
        // Test: Validation rejects zero entities
        let gen = BaseNoiseGenerator::new(100, 0, 42);
        gen.generate(BaseNoiseMethod::Standard);
    }

    #[test]
    fn test_performance_baseline() {
        let gen = BaseNoiseGenerator::new(1000, 10, 42);

        let start = Instant::now();
        let _noise = gen.generate(BaseNoiseMethod::Standard);
        let duration = start.elapsed();

        // Should complete in < 30ms
        assert!(
            duration.as_millis() < 30,
            "Should generate 1000×10 samples in <30ms, took {:?}",
            duration
        );
    }
}
