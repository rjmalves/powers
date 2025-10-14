//! Marginal Distribution Transformation (CEPEL Pipeline Stage 3)
//!
//! # Overview
//!
//! Transforms correlated standard normal samples W ~ N(0,R) to target marginal
//! distributions while preserving the correlation structure introduced in Stage 2.
//! This implements the Gaussian copula approach: transform each marginal independently
//! using the appropriate inverse CDF while maintaining the correlation in normal space.
//!
//! This is **Stage 3** of the CEPEL 4-stage scenario generation pipeline:
//! 1. Base Noise: Generate Z ~ N(0,1) (independent) [`crate::base_noise`]
//! 2. Correlation: Apply W = L×Z → W ~ N(0,R) [`crate::correlation_applicator`]
//! 3. **Marginal: Transform to target distributions** ← This module
//! 4. Temporal: Apply AR/PAR dynamics [`crate::ar_dynamics`], [`crate::par_generator`]
//!
//! # Usage by Temporal Model
//!
//! This transformer applies to **different targets** depending on the temporal model:
//!
//! | Temporal Model          | Transform Applied To  | Next Stage            |
//! |-------------------------|-----------------------|-----------------------|
//! | `Independent`           | Final series Xₜ       | Output (done)         |
//! | `Autoregressive`        | Innovations εₜ        | AR dynamics           |
//! | `PeriodicAutoregressive`| **Residuals aₜ**      | PAR re-seasonalization|
//!
//! **For PAR models** (CEPEL methodology):
//! - **Input**: Correlated normal variates W ~ MVN(0, R)
//! - **Output**: Transformed residuals aₜ ~ F (e.g., LogNormal3)
//! - **Next**: PAR equation applies seasonal structure: Zₜ = μₘ + σₘ·[Σφₖₘ·aₜ₋ₖ + aₜ]
//!
//! This separation is critical: LogNormal3 is applied to **residuals** (aₜ), not final
//! values (Zₜ), ensuring non-negativity while preserving seasonal AR structure.
//!
//! # Algorithm
//!
//! **Gaussian Copula Theorem**:
//!
//! - Input: W ~ MVN(0, R) (multivariate normal with correlation R)
//! - Transform: Xᵢ = Fᵢ⁻¹(Φ(Wᵢ)) where Fᵢ is target CDF, Φ is standard normal CDF
//! - Result: X has marginals Fᵢ and correlation approximately preserved
//!
//! **For Normal Marginal**: Fᵢ⁻¹(Φ(w)) = μᵢ + σᵢw (linear)
//! - Pearson correlation preserved exactly
//!
//! **For LogNormal3**: Fᵢ⁻¹(Φ(w)) = γ + exp(μ + σw) (nonlinear)
//! - Pearson correlation changes (nonlinear transformation)
//! - Spearman (rank) correlation approximately preserved
//! - Tail dependence increased (copula property)
//!
//! # Performance
//!
//! - **Time**: O(S × E) where S=scenarios, E=entities (linear, embarrassingly parallel)
//! - **Space**: O(S × E) (copies input)
//! - **Target**: <15ms for 1000 scenarios × 10 entities
//!
//! # Mathematical Properties
//!
//! ## Normal Transformation
//!
//! ```text
//! X = μ + σW
//! E[X] = μ
//! Var(X) = σ²
//! Correlation preserved exactly (linear transformation)
//! ```
//!
//! ## LogNormal3 Transformation
//!
//! ```text
//! X = γ + exp(μ + σW)
//! E[X] = γ + exp(μ + σ²/2)
//! Var(X) = exp(2μ + σ²) × (exp(σ²) - 1)
//! Spearman ρ_s ≈ (6/π)arcsin(ρ_pearson/2)
//! ```
//!
//! # PAR Residual Transformation (CEPEL Methodology)
//!
//! For Periodic Autoregressive models, this transformer produces **residuals aₜ**:
//!
//! ```text
//! Pipeline:
//!   W ~ MVN(0, R)              [Correlated noise]
//!   ↓
//!   a = transform(W)           [This transformer: apply LogNormal3 to W]
//!   ↓
//!   Z = μₘ + σₘ·[AR_term + a]  [PAR generator: seasonal re-scaling]
//!
//! Key insight: LogNormal3 applied to RESIDUALS (a), not final series (Z).
//! This ensures non-negativity of residuals while preserving seasonal structure.
//! ```
//!
//! **Why residual transformation?**
//! - Direct transformation: Z = LogNormal3(W) loses seasonal AR structure
//! - Residual transformation: a = LogNormal3(W), then Z = PAR(a) preserves both:
//!   * Non-negativity (LogNormal3 on residuals)
//!   * Seasonal AR dynamics (PAR equation with seasonal μₘ, σₘ, φₖₘ)
//!   * Spatial correlation (preserved through Gaussian copula)
//!
//! # References
//!
//! - CEPEL Technical Reports: Scenario Generation for Hydrothermal Systems
//! - Maceira, M.E.P. et al. (2008): "PAR(p) Model for Hydrological Studies"
//! - Nelsen, R.B. (2006): "An Introduction to Copulas", 2nd Edition
//! - Joe, H. (1997): "Multivariate Models and Dependence Concepts"
//! - PSR SDDP Technical Folder: Non-negativity in Stochastic Optimization

use crate::input::MarginalDistribution;

/// Marginal distribution transformer for CEPEL pipeline stage 3
///
/// Transforms correlated standard normal samples to target marginal distributions
/// while preserving correlation structure (Gaussian copula approach).
///
/// # Examples
///
/// ```
/// use powers_rs::marginal_transformer::MarginalTransformer;
/// use powers_rs::input::MarginalDistribution;
///
/// // Define marginal distributions for each entity
/// let marginals = vec![
///     MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
///     MarginalDistribution::LogNormal3 { gamma: 1.0, mu: 4.5, sigma: 0.3 },
/// ];
///
/// let transformer = MarginalTransformer::new(marginals).unwrap();
///
/// // Transform correlated N(0,1) samples to target marginals
/// let correlated_normal = vec![
///     vec![0.5, -0.3],   // scenario 0: [entity_0, entity_1]
///     vec![-1.0, 0.8],   // scenario 1
/// ];
///
/// let transformed = transformer.transform_marginals(&correlated_normal);
///
/// // entity_0 now ~ N(100, 20²), entity_1 now ~ LogNormal3(1, 4.5, 0.3)
/// // Correlation structure preserved (exactly for Normal, approximately for LogNormal3)
/// assert_eq!(transformed.len(), 2);
/// assert_eq!(transformed[0].len(), 2);
/// ```
#[derive(Debug)]
pub struct MarginalTransformer {
    /// Marginal distribution specification for each entity
    entity_marginals: Vec<MarginalDistribution>,
}

impl MarginalTransformer {
    /// Create a new marginal transformer
    ///
    /// # Arguments
    ///
    /// * `entity_marginals` - Marginal distribution for each entity (length = num_entities)
    ///   - Index corresponds to entity position in samples array
    ///   - Distribution specifies target marginal for that entity
    ///
    /// # Returns
    ///
    /// Transformer ready to apply marginal transformations, or error if validation fails
    ///
    /// # Errors
    ///
    /// - Empty marginals vector
    /// - Invalid marginal parameters (σ ≤ 0, γ < 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::marginal_transformer::MarginalTransformer;
    /// use powers_rs::input::MarginalDistribution;
    ///
    /// let marginals = vec![
    ///     MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
    ///     MarginalDistribution::LogNormal3 { gamma: 1.0, mu: 4.5, sigma: 0.3 },
    /// ];
    ///
    /// let transformer = MarginalTransformer::new(marginals).unwrap();
    /// ```
    pub fn new(
        entity_marginals: Vec<MarginalDistribution>,
    ) -> Result<Self, String> {
        // Validate non-empty
        if entity_marginals.is_empty() {
            return Err("Marginals vector cannot be empty".to_string());
        }

        // Validate each marginal
        for (idx, marginal) in entity_marginals.iter().enumerate() {
            Self::validate_marginal(marginal, idx)?;
        }

        Ok(Self { entity_marginals })
    }

    /// Transform correlated standard normal samples to target marginals
    ///
    /// Applies Gaussian copula transformation: for each entity, transform W ~ N(0,1)
    /// to target marginal distribution while preserving correlation structure.
    ///
    /// # Arguments
    ///
    /// * `correlated_samples` - Correlated standard normal samples [scenario][entity]
    ///   - W ~ MVN(0, R) where R is correlation matrix
    ///   - Shape: [num_scenarios][num_entities]
    ///   - num_entities must match entity_marginals.len()
    ///
    /// # Returns
    ///
    /// Transformed samples [scenario][entity] with target marginals
    /// - X has marginal distributions as specified
    /// - Correlation structure approximately preserved
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - correlated_samples is empty
    /// - num_entities doesn't match entity_marginals.len()
    ///
    /// # Performance
    ///
    /// - Time: O(S × E) where S=scenarios, E=entities
    /// - Space: O(S × E) (allocates output)
    /// - Typical: <15ms for 1000 scenarios × 10 entities
    ///
    /// # Examples
    ///
    /// ```
    /// use powers_rs::marginal_transformer::MarginalTransformer;
    /// use powers_rs::input::MarginalDistribution;
    ///
    /// let marginals = vec![
    ///     MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
    ///     MarginalDistribution::LogNormal3 { gamma: 1.0, mu: 4.5, sigma: 0.3 },
    /// ];
    ///
    /// let transformer = MarginalTransformer::new(marginals).unwrap();
    ///
    /// let correlated = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];
    /// let transformed = transformer.transform_marginals(&correlated);
    ///
    /// assert_eq!(transformed.len(), 2);
    /// assert_eq!(transformed[0].len(), 2);
    /// ```
    pub fn transform_marginals(
        &self,
        correlated_samples: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        assert!(
            !correlated_samples.is_empty(),
            "Correlated samples cannot be empty"
        );

        let num_scenarios = correlated_samples.len();
        let num_entities = correlated_samples[0].len();

        assert_eq!(
            num_entities,
            self.entity_marginals.len(),
            "Number of entities ({}) must match marginals count ({})",
            num_entities,
            self.entity_marginals.len()
        );

        // PERFORMANCE: Pre-allocate output with exact capacity
        let mut transformed = Vec::with_capacity(num_scenarios);

        for scenario in correlated_samples {
            let mut scenario_transformed = Vec::with_capacity(num_entities);

            // Transform each entity independently
            for (entity_idx, &w) in scenario.iter().enumerate() {
                let x = self.transform_entity(w, entity_idx);
                scenario_transformed.push(x);
            }

            transformed.push(scenario_transformed);
        }

        transformed
    }

    /// Transform a single entity value from W ~ N(0,1) to target marginal
    ///
    /// # Arguments
    ///
    /// * `w` - Correlated standard normal value
    /// * `entity_idx` - Index of entity (for marginal lookup)
    ///
    /// # Returns
    ///
    /// Transformed value with target marginal distribution
    #[inline]
    fn transform_entity(&self, w: f64, entity_idx: usize) -> f64 {
        match &self.entity_marginals[entity_idx] {
            MarginalDistribution::Normal { mean, std_dev } => {
                // X = μ + σW (linear transformation)
                // PERFORMANCE: Fast path, single multiply-add
                mean + std_dev * w
            }
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                // X = γ + exp(μ + σW)
                // PERFORMANCE: Clamp exponent to avoid overflow
                // exp(x) overflows at x ≈ 709, so clamp to [-20, 20] is safe
                let exponent = (mu + sigma * w).clamp(-20.0, 20.0);
                gamma + exponent.exp()
            }
        }
    }

    /// Validate marginal distribution parameters
    ///
    /// # Arguments
    ///
    /// * `marginal` - Marginal distribution to validate
    /// * `idx` - Index for error reporting
    ///
    /// # Returns
    ///
    /// Ok if valid, Err with descriptive message otherwise
    fn validate_marginal(
        marginal: &MarginalDistribution,
        idx: usize,
    ) -> Result<(), String> {
        match marginal {
            MarginalDistribution::Normal { mean: _, std_dev } => {
                if *std_dev <= 0.0 {
                    return Err(format!(
                        "Normal marginal[{}]: std_dev must be > 0, got {}",
                        idx, std_dev
                    ));
                }
            }
            MarginalDistribution::LogNormal3 {
                gamma,
                mu: _,
                sigma,
            } => {
                if *gamma < 0.0 {
                    return Err(format!(
                        "LogNormal3 marginal[{}]: gamma must be >= 0, got {}",
                        idx, gamma
                    ));
                }
                if *sigma <= 0.0 {
                    return Err(format!(
                        "LogNormal3 marginal[{}]: sigma must be > 0, got {}",
                        idx, sigma
                    ));
                }
            }
        }
        Ok(())
    }

    /// Transform correlated normal samples to residuals for PAR models
    ///
    /// This is a semantic alias for `transform_marginals()` that clarifies intent
    /// when used in PAR (Periodic Autoregressive) scenario generation.
    ///
    /// # CEPEL PAR Pipeline
    ///
    /// ```text
    /// Stage 1: Z ~ N(0,1)                    [Base noise]
    /// Stage 2: W = L×Z → W ~ MVN(0, R)       [Correlation]
    /// Stage 3: a = transform_to_residuals(W) [This method]
    /// Stage 4: Z = μₘ + σₘ·[AR_term + a]     [PAR generator]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `correlated_samples` - Correlated standard normal samples W ~ MVN(0, R)
    ///
    /// # Returns
    ///
    /// Transformed residuals aₜ with target marginal distributions
    /// - Ready for PAR generator re-seasonalization
    /// - Spatial correlation preserved (Gaussian copula)
    ///
    /// # Example (PAR with LogNormal3 residuals)
    ///
    /// ```
    /// use powers_rs::marginal_transformer::MarginalTransformer;
    /// use powers_rs::input::MarginalDistribution;
    ///
    /// // Define residual distributions for PAR model entities
    /// let residual_dists = vec![
    ///     MarginalDistribution::LogNormal3 { gamma: 1.0, mu: 4.5, sigma: 0.3 },
    ///     MarginalDistribution::LogNormal3 { gamma: 0.5, mu: 4.2, sigma: 0.25 },
    /// ];
    ///
    /// let transformer = MarginalTransformer::new(residual_dists).unwrap();
    ///
    /// // Correlated normal variates from Stage 2
    /// let correlated_w = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];
    ///
    /// // Transform to residuals (LogNormal3 applied)
    /// let residuals_a = transformer.transform_to_residuals(&correlated_w);
    ///
    /// // residuals_a now ready for PAR generator:
    /// // Z_t = μ_m + σ_m·[φ_1m·a_t-1 + ... + a_t]
    /// assert!(residuals_a[0][0] > 1.0); // LogNormal3 ensures > gamma
    /// assert!(residuals_a[0][1] > 0.5);
    /// ```
    ///
    /// # Performance
    ///
    /// Identical to `transform_marginals()` - zero overhead semantic wrapper.
    #[inline]
    pub fn transform_to_residuals(
        &self,
        correlated_samples: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        // PERFORMANCE: Inline to eliminate function call overhead
        self.transform_marginals(correlated_samples)
    }

    /// Get number of entities
    pub fn num_entities(&self) -> usize {
        self.entity_marginals.len()
    }

    /// Get reference to entity marginals
    pub fn entity_marginals(&self) -> &[MarginalDistribution] {
        &self.entity_marginals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::ContinuousCDF;

    #[test]
    fn test_normal_marginal_transformation() {
        // Test: Normal marginal X = μ + σW
        let marginals = vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        let correlated = vec![vec![0.0], vec![1.0], vec![-1.0], vec![2.0]];

        let transformed = transformer.transform_marginals(&correlated);

        // X = 100 + 20W
        assert_eq!(transformed.len(), 4);
        assert_eq!(transformed[0][0], 100.0); // W=0 → X=100
        assert_eq!(transformed[1][0], 120.0); // W=1 → X=120
        assert_eq!(transformed[2][0], 80.0); // W=-1 → X=80
        assert_eq!(transformed[3][0], 140.0); // W=2 → X=140
    }

    #[test]
    fn test_lognormal3_transformation() {
        // Test: LogNormal3 X = γ + exp(μ + σW)
        let marginals = vec![MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.3,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        let correlated = vec![vec![0.0], vec![1.0], vec![-1.0]];

        let transformed = transformer.transform_marginals(&correlated);

        // X = 1 + exp(4.5 + 0.3W)
        assert_eq!(transformed.len(), 3);

        // W=0: X = 1 + exp(4.5) ≈ 1 + 90.02 = 91.02
        assert!((transformed[0][0] - 91.02).abs() < 0.1);

        // W=1: X = 1 + exp(4.8) ≈ 1 + 121.51 = 122.51
        assert!((transformed[1][0] - 122.51).abs() < 0.1);

        // W=-1: X = 1 + exp(4.2) ≈ 1 + 66.69 = 67.69
        assert!((transformed[2][0] - 67.69).abs() < 0.1);
    }

    #[test]
    fn test_mixed_marginals() {
        // Test: Mix of Normal and LogNormal3
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
            MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            },
        ];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        let correlated = vec![vec![0.5, -0.3, 1.2]];

        let transformed = transformer.transform_marginals(&correlated);

        assert_eq!(transformed.len(), 1);
        assert_eq!(transformed[0].len(), 3);

        // Entity 0: X = 100 + 20*0.5 = 110
        assert_eq!(transformed[0][0], 110.0);

        // Entity 1: X = 1 + exp(4.5 + 0.3*(-0.3)) = 1 + exp(4.41) ≈ 83.27
        assert!((transformed[0][1] - 83.27).abs() < 0.1);

        // Entity 2: X = 50 + 10*1.2 = 62
        assert_eq!(transformed[0][2], 62.0);
    }

    #[test]
    fn test_identity_case() {
        // Test: μ=0, σ=1 → no transformation
        let marginals = vec![MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: 1.0,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        let correlated = vec![vec![0.5], vec![-1.3], vec![2.1]];

        let transformed = transformer.transform_marginals(&correlated);

        // Should be unchanged
        for (i, scenario) in transformed.iter().enumerate() {
            assert_eq!(scenario[0], correlated[i][0]);
        }
    }

    #[test]
    fn test_marginal_properties() {
        // Test: Sample mean and variance match theoretical values
        let marginals = vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        // Generate many samples using inverse normal CDF for deterministic test
        let num_scenarios = 10000;
        let correlated: Vec<Vec<f64>> = (0..num_scenarios)
            .map(|i| {
                // Use inverse CDF to generate standard normal samples
                let u = (i as f64 + 0.5) / num_scenarios as f64;
                let w = statrs::distribution::Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(u);
                vec![w]
            })
            .collect();

        let transformed = transformer.transform_marginals(&correlated);

        // Extract samples for entity 0
        let samples: Vec<f64> = transformed.iter().map(|s| s[0]).collect();

        // Compute mean and std dev manually
        let mean = samples.iter().sum::<f64>() / num_scenarios as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / num_scenarios as f64;
        let std_dev = variance.sqrt();

        // With 10000 samples, should be close to theoretical
        assert!(
            (mean - 100.0).abs() < 1.0,
            "Mean should be ~100, got {}",
            mean
        );
        assert!(
            (std_dev - 20.0).abs() < 1.0,
            "Std dev should be ~20, got {}",
            std_dev
        );
    }

    #[test]
    fn test_correlation_preservation_normal() {
        // Test: Linear transformation preserves Pearson correlation exactly
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            },
        ];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        // Generate correlated samples (simulating Stage 2 output)
        // Use proper quasi-random sampling for deterministic test
        let num_scenarios = 5000;
        let correlated: Vec<Vec<f64>> = (0..num_scenarios)
            .map(|i| {
                // Quasi-random grid for deterministic results
                let u1 = (i as f64 + 0.5) / num_scenarios as f64;
                let u2 = ((i as f64 * 1.618033988749895 % 1.0)
                    + 0.5 / num_scenarios as f64)
                    % 1.0; // Golden ratio for quasi-random

                let w1 = statrs::distribution::Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(u1);
                let w2 = statrs::distribution::Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(u2);

                // Apply correlation: W2' = 0.7*W1 + sqrt(1-0.7²)*W2
                let rho = 0.7_f64;
                let w2_corr = rho * w1 + (1.0 - rho * rho).sqrt() * w2;

                vec![w1, w2_corr]
            })
            .collect();

        let transformed = transformer.transform_marginals(&correlated);

        // Compute sample correlation
        let x1: Vec<f64> = transformed.iter().map(|s| s[0]).collect();
        let x2: Vec<f64> = transformed.iter().map(|s| s[1]).collect();

        let mean1 = x1.iter().sum::<f64>() / num_scenarios as f64;
        let mean2 = x2.iter().sum::<f64>() / num_scenarios as f64;

        let cov: f64 = x1
            .iter()
            .zip(&x2)
            .map(|(a, b)| (a - mean1) * (b - mean2))
            .sum::<f64>()
            / num_scenarios as f64;

        let var1 = x1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
            / num_scenarios as f64;
        let var2 = x2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
            / num_scenarios as f64;

        let corr = cov / (var1.sqrt() * var2.sqrt());

        // Linear transformation preserves correlation exactly (within sampling error)
        // Wider tolerance for quasi-random sampling
        assert!(
            (corr - 0.7).abs() < 0.05,
            "Correlation should be ~0.7, got {}",
            corr
        );
    }

    #[test]
    fn test_extreme_values_handled() {
        // Test: Extreme W values don't cause overflow
        let marginals = vec![MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.3,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        // Extreme values: W = ±10 (very rare in N(0,1))
        let correlated = vec![vec![10.0], vec![-10.0], vec![100.0]];

        let transformed = transformer.transform_marginals(&correlated);

        // Should not overflow or underflow
        for scenario in &transformed {
            assert!(scenario[0].is_finite(), "Result should be finite");
            assert!(scenario[0] >= 1.0, "Result should be >= gamma");
        }

        // W=100 gets clamped to 20, so X = 1 + exp(4.5 + 0.3*20) = 1 + exp(10.5)
        // Wait - W=100 with sigma=0.3 → exponent = 4.5 + 0.3*100 = 34.5, clamped to 20
        // So X = 1 + exp(20) ≈ 485M (large but finite)
        assert!(
            transformed[2][0] < 5e8,
            "Should be clamped to reasonable value, got {}",
            transformed[2][0]
        );
        assert!(
            (transformed[2][0] - 485165197.0).abs() < 1.0,
            "Should equal exp(20) ≈ 485M, got {}",
            transformed[2][0]
        );
    }

    #[test]
    fn test_validation_rejects_invalid_normal() {
        // Test: Validation rejects σ ≤ 0
        let marginals = vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 0.0, // Invalid
        }];

        let result = MarginalTransformer::new(marginals);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("std_dev must be > 0"));
    }

    #[test]
    fn test_validation_rejects_invalid_lognormal3() {
        // Test: Validation rejects γ < 0
        let marginals = vec![MarginalDistribution::LogNormal3 {
            gamma: -1.0, // Invalid
            mu: 4.5,
            sigma: 0.3,
        }];

        let result = MarginalTransformer::new(marginals);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("gamma must be >= 0"));

        // Test: Validation rejects σ ≤ 0
        let marginals = vec![MarginalDistribution::LogNormal3 {
            gamma: 1.0,
            mu: 4.5,
            sigma: 0.0, // Invalid
        }];

        let result = MarginalTransformer::new(marginals);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("sigma must be > 0"));
    }

    #[test]
    fn test_empty_marginals_rejected() {
        // Test: Empty marginals vector rejected
        let marginals = vec![];
        let result = MarginalTransformer::new(marginals);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot be empty"));
    }

    #[test]
    #[should_panic(expected = "Correlated samples cannot be empty")]
    fn test_empty_samples_panics() {
        // Test: Empty samples panic
        let marginals = vec![MarginalDistribution::Normal {
            mean: 100.0,
            std_dev: 20.0,
        }];

        let transformer = MarginalTransformer::new(marginals).unwrap();
        let _ = transformer.transform_marginals(&[]);
    }

    #[test]
    #[should_panic(expected = "must match marginals count")]
    fn test_dimension_mismatch_panics() {
        // Test: Dimension mismatch panics
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            },
        ];

        let transformer = MarginalTransformer::new(marginals).unwrap();

        // Only 1 entity in samples, but 2 marginals specified
        let correlated = vec![vec![0.5]];
        let _ = transformer.transform_marginals(&correlated);
    }

    // ========================================================================
    // PAR Residual Transformation Tests (PAR-007)
    // ========================================================================

    #[test]
    fn test_par_residual_transformation_lognormal3() {
        // Test: PAR residual transformation with LogNormal3
        // Scenario: Two hydro entities with LogNormal3 residual distributions
        let residual_dists = vec![
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
            MarginalDistribution::LogNormal3 {
                gamma: 0.5,
                mu: 4.2,
                sigma: 0.25,
            },
        ];

        let transformer = MarginalTransformer::new(residual_dists).unwrap();

        // Correlated normal variates (output from Stage 2: correlation)
        let correlated_w = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];

        // Transform to residuals (Stage 3: marginal transformation)
        let residuals_a = transformer.transform_to_residuals(&correlated_w);

        // Verify structure
        assert_eq!(residuals_a.len(), 2); // 2 scenarios
        assert_eq!(residuals_a[0].len(), 2); // 2 entities

        // Verify LogNormal3 properties: a_t > gamma (non-negativity)
        for scenario in &residuals_a {
            assert!(
                scenario[0] > 1.0,
                "Entity 0 residual should be > gamma=1.0"
            );
            assert!(
                scenario[1] > 0.5,
                "Entity 1 residual should be > gamma=0.5"
            );
        }

        // Verify specific values (hand calculation)
        // Entity 0, scenario 0: W=0.5 → a = 1 + exp(4.5 + 0.3*0.5) = 1 + exp(4.65) ≈ 105.64
        assert!((residuals_a[0][0] - 105.64).abs() < 0.1);

        // Entity 1, scenario 0: W=-0.3 → a = 0.5 + exp(4.2 + 0.25*(-0.3)) = 0.5 + exp(4.125) ≈ 62.45
        assert!((residuals_a[0][1] - 62.45).abs() < 0.1);
    }

    #[test]
    fn test_par_residual_transformation_preserves_correlation() {
        // Test: Residual transformation preserves spatial correlation
        // Critical for PAR models: correlation between entities must be maintained
        let residual_dists = vec![
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.0,
                sigma: 0.3,
            },
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.0,
                sigma: 0.3,
            },
        ];

        let transformer = MarginalTransformer::new(residual_dists).unwrap();

        // Generate correlated samples with known correlation
        let num_scenarios = 5000;
        let rho = 0.7; // Target correlation

        let correlated: Vec<Vec<f64>> = (0..num_scenarios)
            .map(|i| {
                let u1 = (i as f64 + 0.5) / num_scenarios as f64;
                let u2 = ((i as f64 * 1.618033988749895 % 1.0)
                    + 0.5 / num_scenarios as f64)
                    % 1.0;

                let w1 = statrs::distribution::Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(u1);
                let w2 = statrs::distribution::Normal::new(0.0, 1.0)
                    .unwrap()
                    .inverse_cdf(u2);

                // Induce correlation: W2' = ρ·W1 + sqrt(1-ρ²)·W2
                let w2_corr = rho * w1 + (1.0 - rho * rho).sqrt() * w2;

                vec![w1, w2_corr]
            })
            .collect();

        // Transform to residuals
        let residuals = transformer.transform_to_residuals(&correlated);

        // Compute sample correlation of residuals
        let a1: Vec<f64> = residuals.iter().map(|s| s[0]).collect();
        let a2: Vec<f64> = residuals.iter().map(|s| s[1]).collect();

        let mean1 = a1.iter().sum::<f64>() / num_scenarios as f64;
        let mean2 = a2.iter().sum::<f64>() / num_scenarios as f64;

        let cov: f64 = a1
            .iter()
            .zip(&a2)
            .map(|(a, b)| (a - mean1) * (b - mean2))
            .sum::<f64>()
            / num_scenarios as f64;

        let var1 = a1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
            / num_scenarios as f64;
        let var2 = a2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
            / num_scenarios as f64;

        let corr = cov / (var1.sqrt() * var2.sqrt());

        // LogNormal3 transformation changes Pearson correlation
        // but should preserve positive dependence structure
        // With identical marginals and moderate parameters, correlation should be > 0.5
        assert!(
            corr > 0.5,
            "Correlation should be preserved (approx), got {}",
            corr
        );

        // For LogNormal3, Spearman correlation is better preserved than Pearson
        // This is a property of Gaussian copulas with nonlinear transformations
    }

    #[test]
    fn test_par_residuals_always_positive() {
        // Test: PAR residuals with LogNormal3 are always positive
        // Critical for non-negativity constraint in hydro inflows
        let residual_dists = vec![MarginalDistribution::LogNormal3 {
            gamma: 0.0,
            mu: 4.0,
            sigma: 0.5,
        }];

        let transformer = MarginalTransformer::new(residual_dists).unwrap();

        // Test with extreme negative W values
        let correlated =
            vec![vec![-5.0], vec![-10.0], vec![-3.0], vec![0.0], vec![3.0]];

        let residuals = transformer.transform_to_residuals(&correlated);

        // All residuals must be positive (> gamma = 0)
        for (i, scenario) in residuals.iter().enumerate() {
            assert!(
                scenario[0] > 0.0,
                "Scenario {} residual should be positive, got {}",
                i,
                scenario[0]
            );
        }

        // Verify reasonable bounds (clamping works)
        for scenario in &residuals {
            assert!(
                scenario[0] < 1e8,
                "Residual should be bounded, got {}",
                scenario[0]
            );
        }
    }

    #[test]
    fn test_par_residual_vs_marginal_equivalence() {
        // Test: transform_to_residuals() is equivalent to transform_marginals()
        // They are semantic aliases - same implementation
        let residual_dists = vec![
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
            MarginalDistribution::Normal {
                mean: 50.0,
                std_dev: 10.0,
            },
        ];

        let transformer = MarginalTransformer::new(residual_dists).unwrap();

        let correlated = vec![vec![0.5, -0.3], vec![-1.0, 0.8]];

        let via_marginals = transformer.transform_marginals(&correlated);
        let via_residuals = transformer.transform_to_residuals(&correlated);

        // Should be bitwise identical
        assert_eq!(via_marginals.len(), via_residuals.len());
        for (s1, s2) in via_marginals.iter().zip(&via_residuals) {
            assert_eq!(s1.len(), s2.len());
            for (v1, v2) in s1.iter().zip(s2) {
                assert_eq!(v1, v2, "Values should be identical");
            }
        }
    }

    #[test]
    fn test_par_mixed_residual_distributions() {
        // Test: PAR with mixed residual distributions (some LogNormal3, some Normal)
        // Real-world scenario: some entities have non-negativity constraints, others don't
        let residual_dists = vec![
            MarginalDistribution::LogNormal3 {
                gamma: 1.0,
                mu: 4.5,
                sigma: 0.3,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::LogNormal3 {
                gamma: 0.5,
                mu: 4.0,
                sigma: 0.25,
            },
        ];

        let transformer = MarginalTransformer::new(residual_dists).unwrap();

        let correlated = vec![vec![0.5, -0.3, 1.2]];

        let residuals = transformer.transform_to_residuals(&correlated);

        assert_eq!(residuals.len(), 1);
        assert_eq!(residuals[0].len(), 3);

        // Entity 0: LogNormal3 → always > 1.0
        assert!(residuals[0][0] > 1.0);

        // Entity 1: Normal (0,1) → can be negative
        assert_eq!(residuals[0][1], -0.3); // Identity transformation

        // Entity 2: LogNormal3 → always > 0.5
        assert!(residuals[0][2] > 0.5);
    }
}
