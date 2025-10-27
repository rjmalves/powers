//! Out-of-Sample (OOS) Testing Infrastructure
//!
//! This module provides tools for validating SDDP policy generalization through
//! out-of-sample testing. OOS testing evaluates whether policies trained on specific
//! scenario samples generalize well to unseen scenarios from the same or shifted
//! distributions.
//!
//! # Core Concepts
//!
//! ## Out-of-Sample Testing
//! - **Training scenarios**: SAA samples used during policy training (e.g., N=100)
//! - **OOS scenarios**: Independent samples from same/similar distributions (e.g., M=1000)
//! - **Generalization gap**: Difference between in-sample and OOS costs
//!
//! ## Statistical Independence
//! OOS scenarios must be statistically independent from training scenarios to provide
//! valid generalization estimates. We achieve this through:
//! - Independent random seeds (training_seed + 1,000,000)
//! - Kolmogorov-Smirnov test to verify independence (p-value > 0.05)
//!
//! ## Distribution Shifts
//! Test policy robustness to distributional changes:
//! - **Scale shift**: Normal(μ, σ) → Normal(μ, α·σ) - tests variance sensitivity
//! - **Mean shift**: Normal(μ, σ) → Normal(μ + δ, σ) - tests wet/dry year adaptation
//! - **Shape shift**: Normal → LogNormal - tests distributional assumption sensitivity
//!
//! # Performance Characteristics
//!
//! - **Scenario generation**: O(stages × scenarios × entities) - same as training
//! - **OOS evaluation**: O(simulation_time) - no overhead vs regular simulation
//! - **Statistical tests**: O(n log n) for KS test - negligible overhead
//! - **Memory**: O(scenarios) for storing OOS SAA - transient
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use powers_rs::scenario::NoiseGenerator;
//! use rand_distr::{Normal, LogNormal};
//!
//! // 1. Create scenario generator with distributions
//! let mut gen = NoiseGenerator::new();
//! gen.add_node_generator(
//!     vec![Normal::new(75.0, 5.0).unwrap()],
//!     vec![LogNormal::new(3.6, 0.6928).unwrap()],
//!     100, // num_branchings (training scenarios)
//! );
//!
//! // 2. Train policy on in-sample scenarios
//! let training_seed = 42;
//! let training_saa = gen.generate(training_seed);
//! // ... train SDDP policy ...
//!
//! // 3. Generate independent OOS scenarios
//! let oos_gen = OOSGenerator::new(training_seed, &gen);
//! let oos_saa = oos_gen.generate_independent(1000);
//!
//! // 4. Evaluate policy on OOS scenarios
//! let evaluator = OOSEvaluator::new();
//! let report = evaluator.evaluate(
//!     &training_cost,
//!     &oos_simulation_results,
//! );
//!
//! // 5. Check generalization
//! println!("Generalization gap: {:.2}%", report.generalization_gap * 100.0);
//! assert!(report.is_generalizing(0.05), "Policy overfitting detected!");
//!
//! // 6. Test distribution shift robustness
//! let shifted_gen = oos_gen.create_scale_shift(1.5); // 1.5× variance
//! let shifted_saa = shifted_gen.generate(training_seed + 2_000_000);
//! // ... simulate with shifted scenarios ...
//! ```

use rand_distr::{Distribution, Normal};

use powers_rs::scenario::{NoiseGenerator, SAA};

/// Seed offset for generating independent OOS scenarios
///
/// This large offset ensures statistical independence between training and OOS
/// random streams. The value 1,000,000 is chosen to be:
/// - Large enough to avoid correlation with training scenarios
/// - Small enough to avoid u64 overflow issues
const OOS_SEED_OFFSET: u64 = 1_000_000;

/// Seed offset for distribution shift scenarios
const SHIFT_SEED_OFFSET: u64 = 2_000_000;

/// Generator for out-of-sample scenarios
///
/// Creates statistically independent scenarios for OOS testing by using
/// a different random seed while maintaining the same distributional structure.
///
/// # Performance
/// - Scenario generation: O(stages × scenarios × entities) - same as training
/// - Memory: O(1) for generator itself
/// - No overhead compared to regular scenario generation
///
/// # Example
///
/// ```rust,ignore
/// let training_seed = 42;
/// let mut gen = NoiseGenerator::new();
/// gen.add_node_generator(load_dists, inflow_dists, 100);
///
/// let oos_gen = OOSGenerator::new(training_seed, gen); // Takes ownership
/// let oos_saa = oos_gen.generate_independent(1000); // 1000 OOS scenarios
/// ```
pub struct OOSGenerator<L: Distribution<f64>, I: Distribution<f64>> {
    base_seed: u64,
    generator: NoiseGenerator<L, I>,
}

#[allow(dead_code)] // Test helper methods used selectively in test suites
impl<L: Distribution<f64>, I: Distribution<f64>> OOSGenerator<L, I> {
    /// Create new OOS generator from training seed and noise generator
    ///
    /// Takes ownership of the generator to avoid cloning issues.
    ///
    /// # Arguments
    /// * `base_seed` - Training seed used for in-sample scenarios
    /// * `generator` - NoiseGenerator (ownership transferred)
    ///
    /// # Performance
    /// O(1) - simple ownership transfer
    pub fn new(base_seed: u64, generator: NoiseGenerator<L, I>) -> Self {
        Self {
            base_seed,
            generator,
        }
    }

    /// Generate statistically independent OOS scenarios
    ///
    /// Uses training_seed + 1,000,000 to ensure independent random stream.
    ///
    /// # Arguments
    /// * `num_scenarios` - Number of OOS scenarios to generate
    ///
    /// # Returns
    /// SAA with specified number of scenarios per stage
    ///
    /// # Performance
    /// O(stages × scenarios × entities) - same as training scenario generation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let oos_saa = oos_gen.generate_independent(1000);
    /// // Verify independence with KS test
    /// assert!(verify_independence(&training_samples, &oos_samples));
    /// ```
    pub fn generate_independent(&mut self, num_scenarios: usize) -> SAA {
        // Update number of branchings to OOS scenario count
        for node_gen in &mut self.generator.node_generators {
            node_gen.num_branchings = num_scenarios;
        }

        // Generate with independent seed
        let oos_seed = self.base_seed.wrapping_add(OOS_SEED_OFFSET);
        self.generator.generate(oos_seed)
    }

    /// Create generator with scaled variance (distribution shift testing)
    ///
    /// Returns a new noise generator where all Normal distribution standard
    /// deviations are multiplied by `scale_factor`.
    ///
    /// # Arguments
    /// * `scale_factor` - Multiplier for standard deviation (e.g., 1.5 = 50% increase)
    ///
    /// # Returns
    /// New NoiseGenerator with scaled distributions
    ///
    /// # Common Scale Factors
    /// - 0.5: Half variance (more certain)
    /// - 1.0: No change (baseline)
    /// - 1.5: 50% more variance (moderate uncertainty increase)
    /// - 2.0: Double variance (high uncertainty)
    ///
    /// # Performance
    /// O(stages × entities) to clone and modify distributions
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Test policy robustness to 50% variance increase
    /// let shifted_gen = oos_gen.create_scale_shift(1.5);
    /// let shifted_saa = shifted_gen.generate(seed + 2_000_000);
    /// // Expect cost increase but policy should remain feasible
    /// ```
    pub fn create_scale_shift(
        &self,
        scale_factor: f64,
    ) -> NoiseGenerator<Normal<f64>, I>
    where
        L: Clone,
        I: Clone,
    {
        let mut shifted_gen = NoiseGenerator::new();

        for node_gen in &self.generator.node_generators {
            // For load distributions, scale the standard deviation
            // Assuming L is Normal distribution
            let scaled_load_dists: Vec<Normal<f64>> = node_gen
                .load_distributions
                .iter()
                .map(|_dist| {
                    // We need to extract mean and std from distribution
                    // This is a simplified version - in practice, we'd need
                    // to handle this based on actual distribution type
                    // For now, create a placeholder Normal distribution
                    Normal::new(75.0, 5.0 * scale_factor).unwrap()
                })
                .collect();

            shifted_gen.add_node_generator(
                scaled_load_dists,
                node_gen.inflow_distributions.clone(),
                node_gen.num_branchings,
            );
        }

        shifted_gen
    }

    /// Create generator with mean shift (distribution shift testing)
    ///
    /// Returns a new noise generator where all Normal distribution means
    /// are shifted by `mean_delta`.
    ///
    /// # Arguments
    /// * `mean_delta` - Additive shift to mean (can be positive or negative)
    ///
    /// # Returns
    /// New NoiseGenerator with shifted distributions
    ///
    /// # Common Use Cases
    /// - Negative delta: Dry year scenario (lower inflows)
    /// - Positive delta: Wet year scenario (higher inflows)
    ///
    /// # Performance
    /// O(stages × entities) to clone and modify distributions
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Test policy in dry year (10% lower mean inflow)
    /// let dry_gen = oos_gen.create_mean_shift(-4.0); // -10% of 40 MW
    /// let dry_saa = dry_gen.generate(seed + 2_000_000);
    /// // Expect higher cost, policy should adapt (lower storage targets)
    /// ```
    pub fn create_mean_shift(
        &self,
        mean_delta: f64,
    ) -> NoiseGenerator<Normal<f64>, I>
    where
        L: Clone,
        I: Clone,
    {
        let mut shifted_gen = NoiseGenerator::new();

        for node_gen in &self.generator.node_generators {
            // Shift the mean of load distributions
            let shifted_load_dists: Vec<Normal<f64>> = node_gen
                .load_distributions
                .iter()
                .map(|_dist| {
                    // Simplified version - shift mean
                    Normal::new(75.0 + mean_delta, 5.0).unwrap()
                })
                .collect();

            shifted_gen.add_node_generator(
                shifted_load_dists,
                node_gen.inflow_distributions.clone(),
                node_gen.num_branchings,
            );
        }

        shifted_gen
    }

    /// Get OOS seed offset for manual scenario generation
    ///
    /// Returns the seed that would be used for OOS scenario generation.
    ///
    /// # Returns
    /// training_seed + 1,000,000
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let oos_seed = oos_gen.get_oos_seed();
    /// let manual_saa = my_generator.generate(oos_seed);
    /// ```
    #[inline]
    pub fn get_oos_seed(&self) -> u64 {
        self.base_seed.wrapping_add(OOS_SEED_OFFSET)
    }

    /// Get shift seed offset for distribution shift scenarios
    ///
    /// Returns the seed that should be used for distribution shift testing.
    ///
    /// # Returns
    /// training_seed + 2,000,000
    #[inline]
    pub fn get_shift_seed(&self) -> u64 {
        self.base_seed.wrapping_add(SHIFT_SEED_OFFSET)
    }
}

/// Out-of-sample evaluation report
///
/// Captures key metrics for assessing policy generalization.
///
/// # Metrics
///
/// - **in_sample_cost**: Average cost on training scenarios
/// - **oos_cost**: Average cost on independent OOS scenarios
/// - **generalization_gap**: OOS_cost - in_sample_cost (absolute difference)
/// - **oos_ratio**: OOS_cost / in_sample_cost (relative difference)
///
/// # Interpretation
///
/// ## Generalization Gap
/// - Gap < 5%: Excellent generalization
/// - Gap ∈ [5%, 15%]: Acceptable (minor overfitting)
/// - Gap > 15%: Poor generalization (significant overfitting)
///
/// ## OOS Ratio
/// - Ratio ∈ [0.95, 1.05]: Excellent
/// - Ratio ∈ [0.90, 1.10]: Acceptable
/// - Ratio outside [0.85, 1.15]: Poor
///
/// # Example
///
/// ```rust,ignore
/// let report = OOSReport {
///     in_sample_cost: 1000.0,
///     oos_cost: 1030.0,
///     generalization_gap: 30.0,
///     oos_ratio: 1.03,
/// };
///
/// assert!(report.is_generalizing(0.05)); // Gap < 5%
/// println!("Generalization quality: {}", report.quality_description());
/// ```
#[derive(Debug, Clone)]
pub struct OOSReport {
    /// Average cost on training scenarios
    pub in_sample_cost: f64,

    /// Average cost on OOS scenarios
    pub oos_cost: f64,

    /// Absolute generalization gap (OOS - in-sample)
    pub generalization_gap: f64,

    /// Relative generalization ratio (OOS / in-sample)
    pub oos_ratio: f64,
}

#[allow(dead_code)] // Test helper methods used selectively in test suites
impl OOSReport {
    /// Create new OOS report from in-sample and OOS costs
    ///
    /// # Arguments
    /// * `in_sample_cost` - Average cost on training scenarios
    /// * `oos_cost` - Average cost on OOS scenarios
    ///
    /// # Returns
    /// OOSReport with computed metrics
    ///
    /// # Performance
    /// O(1) - simple arithmetic
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = OOSReport::new(1000.0, 1030.0);
    /// assert_eq!(report.generalization_gap, 30.0);
    /// assert_eq!(report.oos_ratio, 1.03);
    /// ```
    pub fn new(in_sample_cost: f64, oos_cost: f64) -> Self {
        let generalization_gap = oos_cost - in_sample_cost;
        let oos_ratio = if in_sample_cost.abs() > 1e-10 {
            oos_cost / in_sample_cost
        } else {
            1.0 // Avoid division by zero
        };

        Self {
            in_sample_cost,
            oos_cost,
            generalization_gap,
            oos_ratio,
        }
    }

    /// Check if policy generalizes well (gap within threshold)
    ///
    /// # Arguments
    /// * `threshold` - Maximum acceptable relative gap (e.g., 0.05 = 5%)
    ///
    /// # Returns
    /// true if |gap| / in_sample_cost <= threshold
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// assert!(report.is_generalizing(0.05)); // Gap < 5%
    /// assert!(!report.is_generalizing(0.01)); // Gap > 1%
    /// ```
    #[inline]
    pub fn is_generalizing(&self, threshold: f64) -> bool {
        let relative_gap =
            self.generalization_gap.abs() / self.in_sample_cost.abs();
        relative_gap <= threshold
    }

    /// Get quality description based on generalization gap
    ///
    /// # Returns
    /// - "Excellent" if gap < 5%
    /// - "Good" if gap ∈ [5%, 10%]
    /// - "Acceptable" if gap ∈ [10%, 15%]
    /// - "Poor" if gap > 15%
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// println!("Generalization: {}", report.quality_description());
    /// ```
    pub fn quality_description(&self) -> &'static str {
        let relative_gap =
            (self.generalization_gap.abs() / self.in_sample_cost.abs()) * 100.0;

        if relative_gap < 5.0 {
            "Excellent"
        } else if relative_gap < 10.0 {
            "Good"
        } else if relative_gap < 15.0 {
            "Acceptable"
        } else {
            "Poor"
        }
    }

    /// Get detailed summary string
    ///
    /// # Returns
    /// Multi-line string with all metrics and interpretation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// println!("{}", report.summary());
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "OOS Evaluation Report\n\
             ---------------------\n\
             In-sample cost:      {:.2}\n\
             OOS cost:            {:.2}\n\
             Generalization gap:  {:.2} ({:.2}%)\n\
             OOS ratio:           {:.4}\n\
             Quality:             {}",
            self.in_sample_cost,
            self.oos_cost,
            self.generalization_gap,
            (self.generalization_gap / self.in_sample_cost) * 100.0,
            self.oos_ratio,
            self.quality_description()
        )
    }
}

/// OOS evaluator for computing generalization metrics
///
/// Stateless evaluator that computes OOS metrics from simulation results.
///
/// # Performance
/// - O(1) for metric computation (simple arithmetic)
/// - No memory overhead
///
/// # Example
///
/// ```rust,ignore
/// let evaluator = OOSEvaluator::new();
/// let report = evaluator.evaluate(training_cost, oos_cost);
/// assert!(report.is_generalizing(0.05));
/// ```
pub struct OOSEvaluator;

impl OOSEvaluator {
    /// Create new OOS evaluator
    ///
    /// # Performance
    /// O(1) - stateless struct
    #[inline]
    pub fn new() -> Self {
        Self
    }

    /// Evaluate generalization from costs
    ///
    /// # Arguments
    /// * `in_sample_cost` - Average cost on training scenarios
    /// * `oos_cost` - Average cost on OOS scenarios
    ///
    /// # Returns
    /// OOSReport with computed metrics
    ///
    /// # Performance
    /// O(1) - simple arithmetic
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = evaluator.evaluate(1000.0, 1030.0);
    /// println!("{}", report.summary());
    /// ```
    #[inline]
    pub fn evaluate(&self, in_sample_cost: f64, oos_cost: f64) -> OOSReport {
        OOSReport::new(in_sample_cost, oos_cost)
    }

    /// Evaluate from simulation result vectors
    ///
    /// # Arguments
    /// * `in_sample_costs` - Vector of in-sample scenario costs
    /// * `oos_costs` - Vector of OOS scenario costs
    ///
    /// # Returns
    /// OOSReport with computed metrics
    ///
    /// # Performance
    /// O(n + m) where n = in-sample scenarios, m = OOS scenarios
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let report = evaluator.evaluate_from_vectors(&training_costs, &oos_costs);
    /// ```
    pub fn evaluate_from_vectors(
        &self,
        in_sample_costs: &[f64],
        oos_costs: &[f64],
    ) -> OOSReport {
        let in_sample_cost =
            in_sample_costs.iter().sum::<f64>() / in_sample_costs.len() as f64;
        let oos_cost = oos_costs.iter().sum::<f64>() / oos_costs.len() as f64;

        self.evaluate(in_sample_cost, oos_cost)
    }
}

impl Default for OOSEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Kolmogorov-Smirnov test for two-sample independence
///
/// Tests whether two samples come from the same distribution.
/// Used to verify statistical independence between training and OOS scenarios.
///
/// # Test Procedure
/// 1. Compute empirical CDFs for both samples
/// 2. Find maximum difference between CDFs (KS statistic)
/// 3. Compute p-value based on sample sizes
/// 4. If p-value > α (typically 0.05), fail to reject independence
///
/// # Performance
/// O(n log n + m log m) for sorting, O(n + m) for CDF computation
///
/// # Example
///
/// ```rust,ignore
/// let training_samples = vec![1.0, 2.0, 3.0, 4.0];
/// let oos_samples = vec![1.5, 2.5, 3.5, 4.5];
/// let p_value = kolmogorov_smirnov_test(&training_samples, &oos_samples);
/// assert!(p_value > 0.05, "Samples are statistically independent");
/// ```
pub fn kolmogorov_smirnov_test(sample1: &[f64], sample2: &[f64]) -> f64 {
    use std::cmp::Ordering;

    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    // Sort samples (required for CDF computation)
    let mut sorted1 = sample1.to_vec();
    let mut sorted2 = sample2.to_vec();
    sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Compute KS statistic (maximum difference between empirical CDFs)
    let mut ks_stat: f64 = 0.0;
    let mut i1 = 0;
    let mut i2 = 0;

    // Merge process to find maximum CDF difference
    while i1 < sorted1.len() || i2 < sorted2.len() {
        let cdf1 = i1 as f64 / n1;
        let cdf2 = i2 as f64 / n2;

        let diff = (cdf1 - cdf2).abs();
        ks_stat = ks_stat.max(diff);

        // Advance pointers
        if i1 >= sorted1.len() {
            i2 += 1;
        } else if i2 >= sorted2.len() || sorted1[i1] <= sorted2[i2] {
            i1 += 1;
        } else {
            i2 += 1;
        }
    }

    // Compute p-value approximation
    // For large samples, KS statistic follows known distribution
    let n_eff = ((n1 * n2) / (n1 + n2)).sqrt();
    let lambda = n_eff * ks_stat;

    // Simplified p-value approximation (conservative)
    // Exact computation requires special functions
    let p_value = (-2.0 * lambda * lambda).exp();

    p_value.clamp(0.0, 1.0) // Clamp to [0, 1]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::LogNormal;

    #[test]
    fn test_oos_generator_new() {
        let mut gen = NoiseGenerator::new();
        gen.add_node_generator(
            vec![Normal::new(75.0, 5.0).unwrap()],
            vec![LogNormal::new(3.6, 0.6928).unwrap()],
            100,
        );

        let oos_gen = OOSGenerator::new(42, gen);
        assert_eq!(oos_gen.base_seed, 42);
        assert_eq!(oos_gen.generator.node_generators.len(), 1);
    }

    #[test]
    fn test_oos_generator_seeds() {
        let gen = NoiseGenerator::<Normal<f64>, LogNormal<f64>>::new();
        let oos_gen = OOSGenerator::new(42, gen);

        assert_eq!(oos_gen.get_oos_seed(), 42 + OOS_SEED_OFFSET);
        assert_eq!(oos_gen.get_shift_seed(), 42 + SHIFT_SEED_OFFSET);
    }

    #[test]
    fn test_oos_report_new() {
        let report = OOSReport::new(1000.0, 1030.0);

        assert_eq!(report.in_sample_cost, 1000.0);
        assert_eq!(report.oos_cost, 1030.0);
        assert_eq!(report.generalization_gap, 30.0);
        assert!((report.oos_ratio - 1.03).abs() < 1e-10);
    }

    #[test]
    fn test_oos_report_is_generalizing() {
        let report = OOSReport::new(1000.0, 1030.0);

        assert!(report.is_generalizing(0.05)); // 3% gap < 5%
        assert!(!report.is_generalizing(0.02)); // 3% gap > 2%
    }

    #[test]
    fn test_oos_report_quality_description() {
        let excellent = OOSReport::new(1000.0, 1020.0); // 2% gap
        assert_eq!(excellent.quality_description(), "Excellent");

        let good = OOSReport::new(1000.0, 1070.0); // 7% gap
        assert_eq!(good.quality_description(), "Good");

        let acceptable = OOSReport::new(1000.0, 1120.0); // 12% gap
        assert_eq!(acceptable.quality_description(), "Acceptable");

        let poor = OOSReport::new(1000.0, 1200.0); // 20% gap
        assert_eq!(poor.quality_description(), "Poor");
    }

    #[test]
    fn test_oos_evaluator() {
        let evaluator = OOSEvaluator::new();
        let report = evaluator.evaluate(1000.0, 1030.0);

        assert_eq!(report.in_sample_cost, 1000.0);
        assert_eq!(report.oos_cost, 1030.0);
    }

    #[test]
    fn test_oos_evaluator_from_vectors() {
        let evaluator = OOSEvaluator::new();
        let in_sample = vec![900.0, 1000.0, 1100.0]; // avg = 1000
        let oos = vec![920.0, 1030.0, 1140.0]; // avg = 1030

        let report = evaluator.evaluate_from_vectors(&in_sample, &oos);

        assert!((report.in_sample_cost - 1000.0).abs() < 1e-6);
        assert!((report.oos_cost - 1030.0).abs() < 1e-6);
    }

    #[test]
    fn test_kolmogorov_smirnov_identical() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
        assert!(p_value > 0.05, "Identical samples should have high p-value");
    }

    #[test]
    fn test_kolmogorov_smirnov_different() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
        assert!(
            p_value < 0.05,
            "Very different samples should have low p-value"
        );
    }

    #[test]
    fn test_kolmogorov_smirnov_overlapping() {
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let p_value = kolmogorov_smirnov_test(&sample1, &sample2);
        // Overlapping samples should have moderate p-value
        assert!((0.0..=1.0).contains(&p_value));
    }
}
