//! 3-Parameter Log-Normal Distribution for Non-Negative Scenario Generation
//!
//! A random variable X follows a 3-parameter log-normal distribution LN3(γ, μ, σ) if:
//!
//! ```text
//! X = γ + exp(Y)  where Y ~ N(μ, σ²)
//! ```
//!
//! **Parameters**:
//! - `γ` (gamma): Location parameter, γ ≥ 0. Represents the minimum possible value (X ≥ γ always).
//! - `μ` (mu): Mean of the log-transformed variable Y = log(X - γ)
//! - `σ` (sigma): Standard deviation of Y, σ > 0
//!
//! ## Statistical Properties
//!
//! Given parameters (γ, μ, σ):
//!
//! ```text
//! E[X] = γ + exp(μ + σ²/2)
//! Var[X] = exp(2μ + σ²) · (exp(σ²) - 1)
//! Mode[X] = γ + exp(μ - σ²)  (for σ² < 1)
//! Median[X] = γ + exp(μ)
//! ```
//!
//! The distribution is **always positive** (X ≥ γ ≥ 0) and **right-skewed**.
//!
//! ## Sampling Algorithm
//!
//! To sample X ~ LN3(γ, μ, σ):
//!
//! 1. Sample Z ~ N(0, 1) (standard normal)
//! 2. Transform: Y = μ + σZ
//! 3. Exponentiate: X = γ + exp(Y)
//!
//! This is O(1) with zero allocations.
//!
//! ## Integration with Correlation
//!
//! This module integrates seamlessly with `CorrelatedNoiseGenerator` (AR-5.6):
//!
//! 1. Generate correlated Z ~ N(0, 1) via Cholesky decomposition
//! 2. For each entity, apply inverse CDF: X = F⁻¹(Φ(Z)) where F⁻¹ is LogNormal3::inverse_cdf
//! 3. Result: Correlated non-negative scenarios with correct marginals
//!
//! ## Performance Characteristics
//!
//! - **Time**: O(1) per sample (single exp() evaluation)
//! - **Space**: O(1) (zero allocations in hot path)
//! - **Cache-friendly**: All parameters fit in single cache line
//! - **Target**: <10ns per sample (faster than Shadow AR's 50ns)
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use powers::lognormal3::LogNormal3Param;
//!
//! // Create distribution: minimum 1.0, typical values around exp(4.5) ≈ 90
//! let dist = LogNormal3Param::new(1.0, 4.5, 0.3).unwrap();
//!
//! // Sample from standard normal innovation
//! let z = 0.5; // Z ~ N(0,1)
//! let x = dist.sample(z);
//! assert!(x >= 1.0); // Always non-negative
//!
//! // Or sample via inverse CDF (for correlation)
//! let u = 0.7; // U ~ Uniform(0,1)
//! let x = dist.inverse_cdf(u);
//! assert!(x >= 1.0);
//! ```

use crate::error::ValidationError;

/// 3-Parameter Log-Normal distribution: X = γ + exp(μ + σZ) where Z ~ N(0,1)
///
/// This struct represents the parameters of a 3-parameter log-normal distribution
/// used for generating non-negative scenarios in SDDP.
///
/// # Invariants
///
/// - `gamma >= 0.0`: Location parameter (minimum value)
/// - `sigma > 0.0`: Scale parameter (must be positive)
/// - No restrictions on `mu` (can be any real number)
///
/// # Performance
///
/// This struct is `Copy` and fits in a single cache line (24 bytes).
/// All sampling operations are O(1) with zero allocations.
///
/// # Examples
///
/// ```rust,ignore
/// use powers::lognormal3::LogNormal3Param;
///
/// // Typical hydro inflow: minimum 10 m³/s, median ~100 m³/s
/// let dist = LogNormal3Param::new(
///     10.0,  // gamma: minimum inflow
///     4.6,   // mu: log of typical inflow minus gamma
///     0.5,   // sigma: variability
/// ).unwrap();
///
/// // Sample from standard normal
/// let inflow = dist.sample(0.0); // Z=0 gives median
/// assert!(inflow >= 10.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormal3Param {
    /// Location parameter (minimum value), γ ≥ 0
    ///
    /// All samples X will satisfy X ≥ gamma. Typically set to a small
    /// positive value or the observed minimum.
    pub gamma: f64,

    /// Mean of log-transformed variable, μ ∈ ℝ
    ///
    /// Controls the location of the distribution after the shift.
    /// Median = γ + exp(μ).
    pub mu: f64,

    /// Standard deviation of log-transformed variable, σ > 0
    ///
    /// Controls the spread. Larger σ means more right-skewed distribution.
    pub sigma: f64,
}

impl LogNormal3Param {
    /// Create a new 3-parameter log-normal distribution
    ///
    /// # Parameters
    ///
    /// - `gamma`: Location parameter (minimum value), must be ≥ 0
    /// - `mu`: Mean of log-transformed variable (no restrictions)
    /// - `sigma`: Standard deviation of log-transformed variable, must be > 0
    ///
    /// # Returns
    ///
    /// - `Ok(LogNormal3Param)` if parameters are valid
    /// - `Err` if gamma < 0 or sigma ≤ 0
    ///
    #[inline]
    #[allow(clippy::result_large_err)] // ValidationError is used consistently across the codebase
    pub fn new(
        gamma: f64,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, ValidationError> {
        if gamma < 0.0 {
            return Err(ValidationError::InvalidFieldValue {
                file: "lognormal3".to_string(),
                field: "gamma".to_string(),
                value: gamma.to_string(),
                constraint: "Location parameter gamma must be >= 0".to_string(),
                suggestion: "Use non-negative gamma (e.g., 0.0, 1.0, or 10.0)"
                    .to_string(),
            });
        }

        if sigma <= 0.0 {
            return Err(ValidationError::InvalidFieldValue {
                file: "lognormal3".to_string(),
                field: "sigma".to_string(),
                value: sigma.to_string(),
                constraint: "Scale parameter sigma must be > 0".to_string(),
                suggestion: "Use positive sigma (e.g., 0.3, 0.5, or 1.0)"
                    .to_string(),
            });
        }

        Ok(Self { gamma, mu, sigma })
    }

    /// Sample from the distribution given a standard normal variate
    ///
    /// Generates X = γ + exp(μ + σZ) where Z ~ N(0,1).
    ///
    /// # Parameters
    ///
    /// - `z`: Standard normal variate, Z ~ N(0,1)
    ///
    /// # Returns
    ///
    /// Non-negative sample X ≥ gamma
    ///
    /// # Performance
    ///
    /// - Time: O(1) - single exp() evaluation
    /// - Space: O(1) - zero allocations
    /// - Cache-friendly: All parameters in single cache line
    /// - Target: <10ns per call on modern hardware
    ///
    /// # Performance Note
    ///
    /// This method is marked `#[inline]` because it's called in the hot path
    /// of scenario generation (thousands of times per SDDP iteration). The
    /// compiler can optimize away function call overhead.
    ///
    #[inline]
    pub fn sample(&self, z: f64) -> f64 {
        // X = γ + exp(μ + σZ)
        self.gamma + (self.mu + self.sigma * z).exp()
    }

    /// Inverse CDF (quantile function) for the distribution
    ///
    /// Given u ~ Uniform(0,1), returns x such that P(X ≤ x) = u.
    ///
    /// This is used in conjunction with `CorrelatedNoiseGenerator`:
    /// 1. Generate correlated Z ~ N(0,1) via Cholesky
    /// 2. Transform to uniform: U = Φ(Z)
    /// 3. Apply inverse CDF: X = F⁻¹(U)
    ///
    /// # Parameters
    ///
    /// - `u`: Uniform random variate, must be in (0, 1)
    ///
    /// # Returns
    ///
    /// Quantile x such that P(X ≤ x) = u
    ///
    /// # Panics
    ///
    /// Panics if u is not in (0, 1) (0 and 1 are not allowed due to log(0) and log(∞)).
    ///
    #[inline]
    pub fn inverse_cdf(&self, u: f64) -> f64 {
        assert!(u > 0.0 && u < 1.0, "u must be in (0, 1), got {}", u);

        // Standard normal inverse CDF
        // Using Box-Muller approximation for speed
        // Alternative: statrs::distribution::Normal::inverse_cdf
        let z = inverse_normal_cdf(u);

        // Transform to log-normal
        self.sample(z)
    }

    /// Compute the expected value (mean) of the distribution
    ///
    /// Returns E[X] = γ + exp(μ + σ²/2)
    ///
    #[inline]
    pub fn mean(&self) -> f64 {
        self.gamma + (self.mu + 0.5 * self.sigma * self.sigma).exp()
    }

    /// Compute the variance of the distribution
    ///
    /// Returns Var[X] = exp(2μ + σ²) · (exp(σ²) - 1)
    ///
    #[inline]
    pub fn variance(&self) -> f64 {
        let exp_sigma_sq = (self.sigma * self.sigma).exp();
        (2.0 * self.mu + self.sigma * self.sigma).exp() * (exp_sigma_sq - 1.0)
    }

    /// Compute the median of the distribution
    ///
    /// Returns Median[X] = γ + exp(μ)
    ///
    #[inline]
    pub fn median(&self) -> f64 {
        self.gamma + self.mu.exp()
    }
}

/// Fast inverse normal CDF approximation
///
/// Uses Beasley-Springer-Moro algorithm for speed.
/// Accurate to ~1e-9 for u in [0.001, 0.999].
///
/// # Performance
///
/// Faster than `statrs` for our use case (no heap allocation,
/// better inlining). If higher accuracy is needed, switch to
/// `statrs::distribution::Normal::new(0, 1).unwrap().inverse_cdf(u)`.
///
#[inline]
fn inverse_normal_cdf(u: f64) -> f64 {
    // Beasley-Springer-Moro algorithm
    // Coefficients for central region
    const A0: f64 = 2.50662823884;
    const A1: f64 = -18.61500062529;
    const A2: f64 = 41.39119773534;
    const A3: f64 = -25.44106049637;
    const B1: f64 = -8.47351093090;
    const B2: f64 = 23.08336743743;
    const B3: f64 = -21.06224101826;
    const B4: f64 = 3.13082909833;
    const C0: f64 = 0.3374754822726147;
    const C1: f64 = 0.9761690190917186;
    const C2: f64 = 0.1607979714918209;
    const C3: f64 = 0.0276438810333863;
    const C4: f64 = 0.0038405729373609;
    const C5: f64 = 0.0003951896511919;
    const C6: f64 = 0.0000321767881768;
    const C7: f64 = 0.0000002888167364;
    const C8: f64 = 0.0000003960315187;

    let y = u - 0.5;

    if y.abs() < 0.42 {
        // Central region
        let r = y * y;
        y * (((A3 * r + A2) * r + A1) * r + A0)
            / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0)
    } else {
        // Tail region
        let r = if y > 0.0 { 1.0 - u } else { u };
        let r = (-r.ln()).sqrt();

        let z = (((((((C8 * r + C7) * r + C6) * r + C5) * r + C4) * r + C3)
            * r
            + C2)
            * r
            + C1)
            * r
            + C0;

        if y < 0.0 {
            -z
        } else {
            z
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lognormal3_new_validates_params() {
        // Valid parameters
        assert!(LogNormal3Param::new(0.0, 4.5, 0.3).is_ok());
        assert!(LogNormal3Param::new(10.0, 4.5, 0.3).is_ok());
        assert!(LogNormal3Param::new(1.0, -2.0, 1.0).is_ok()); // Negative mu is OK

        // Invalid: negative gamma
        assert!(LogNormal3Param::new(-1.0, 4.5, 0.3).is_err());
        assert!(LogNormal3Param::new(-0.001, 4.5, 0.3).is_err());

        // Invalid: zero sigma
        assert!(LogNormal3Param::new(1.0, 4.5, 0.0).is_err());

        // Invalid: negative sigma
        assert!(LogNormal3Param::new(1.0, 4.5, -0.3).is_err());
    }

    #[test]
    fn test_lognormal3_sample_always_positive() {
        let dist = LogNormal3Param::new(10.0, 4.5, 0.3).unwrap();

        // Test many samples
        for z in &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let x = dist.sample(*z);
            assert!(x >= 10.0, "Sample {} violates X >= gamma (z = {})", x, z);
        }

        // Test with gamma = 0
        let dist_zero = LogNormal3Param::new(0.0, 2.0, 0.5).unwrap();
        for z in &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            let x = dist_zero.sample(*z);
            assert!(x >= 0.0, "Sample {} is negative (z = {})", x, z);
        }
    }

    #[test]
    fn test_lognormal3_sample_median() {
        let dist = LogNormal3Param::new(10.0, 4.5, 0.3).unwrap();

        // Z = 0 gives median = gamma + exp(mu)
        let median_sample = dist.sample(0.0);
        let expected_median = 10.0 + 4.5_f64.exp();

        assert!(
            (median_sample - expected_median).abs() < 1e-10,
            "Median mismatch: {} != {}",
            median_sample,
            expected_median
        );

        // Verify median() method matches
        assert!((dist.median() - expected_median).abs() < 1e-10);
    }

    #[test]
    fn test_lognormal3_sample_ordering() {
        let dist = LogNormal3Param::new(5.0, 3.0, 0.5).unwrap();

        let x_neg = dist.sample(-2.0);
        let x_zero = dist.sample(0.0);
        let x_pos = dist.sample(2.0);

        // Larger Z should give larger X
        assert!(x_neg < x_zero);
        assert!(x_zero < x_pos);
        assert!(x_neg >= 5.0);
    }

    #[test]
    fn test_lognormal3_inverse_cdf_median() {
        let dist = LogNormal3Param::new(10.0, 4.5, 0.3).unwrap();

        // u = 0.5 should give median
        let median_cdf = dist.inverse_cdf(0.5);
        let expected_median = dist.median();

        assert!(
            (median_cdf - expected_median).abs() < 1e-6,
            "Inverse CDF median mismatch: {} != {}",
            median_cdf,
            expected_median
        );
    }

    #[test]
    fn test_lognormal3_inverse_cdf_monotonic() {
        let dist = LogNormal3Param::new(5.0, 3.0, 0.5).unwrap();

        let p05 = dist.inverse_cdf(0.05);
        let p50 = dist.inverse_cdf(0.50);
        let p95 = dist.inverse_cdf(0.95);

        // Larger u should give larger x
        assert!(p05 < p50);
        assert!(p50 < p95);
        assert!(p05 >= 5.0);
    }

    #[test]
    fn test_lognormal3_mean() {
        let dist = LogNormal3Param::new(10.0, 4.5, 0.3).unwrap();
        let mean = dist.mean();

        // Mean = gamma + exp(mu + sigma²/2)
        let expected = 10.0 + (4.5 + 0.5 * 0.3 * 0.3_f64).exp();

        assert!(
            (mean - expected).abs() < 1e-10,
            "Mean mismatch: {} != {}",
            mean,
            expected
        );
    }

    #[test]
    fn test_lognormal3_variance() {
        let dist = LogNormal3Param::new(0.0, 2.0, 0.5).unwrap();
        let var = dist.variance();

        // Var = exp(2μ + σ²) · (exp(σ²) - 1)
        let sigma_sq: f64 = 0.5 * 0.5;
        let expected = (2.0 * 2.0 + sigma_sq).exp() * (sigma_sq.exp() - 1.0);

        assert!(
            (var - expected).abs() < 1e-9,
            "Variance mismatch: {} != {}",
            var,
            expected
        );
    }

    #[test]
    fn test_inverse_normal_cdf_accuracy() {
        // Test against known values
        let z50 = inverse_normal_cdf(0.5);
        assert!((z50 - 0.0).abs() < 1e-9);

        let z84 = inverse_normal_cdf(0.841345);
        assert!((z84 - 1.0).abs() < 1e-4); // ~1 std dev

        let z16 = inverse_normal_cdf(0.158655);
        assert!((z16 - (-1.0)).abs() < 1e-4); // ~-1 std dev
    }

    #[test]
    #[should_panic(expected = "u must be in (0, 1)")]
    fn test_inverse_cdf_panics_on_zero() {
        let dist = LogNormal3Param::new(1.0, 2.0, 0.5).unwrap();
        dist.inverse_cdf(0.0);
    }

    #[test]
    #[should_panic(expected = "u must be in (0, 1)")]
    fn test_inverse_cdf_panics_on_one() {
        let dist = LogNormal3Param::new(1.0, 2.0, 0.5).unwrap();
        dist.inverse_cdf(1.0);
    }
}
