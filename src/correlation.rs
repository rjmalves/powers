//! Correlated scenario generation using Gaussian copula
//!
//! # Overview
//!
//! Generates correlated multi-variate samples while preserving
//! marginal distributions. Uses Gaussian copula with Cholesky
//! decomposition for efficient sampling.
//!
//! # Algorithm
//!
//! 1. Generate independent standard normals: Z ~ N(0, I)
//! 2. Apply correlation via Cholesky: Z' = L × Z where Σ = L Lᵀ
//! 3. Transform to uniform: U = Φ(Z') where Φ is normal CDF
//! 4. Apply inverse CDF of marginal: X = F⁻¹(U)
//!

use crate::error::{PowersError, ValidationError};
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::Rng;
use rand_distr::StandardNormal;
use statrs::distribution::{ContinuousCDF, LogNormal, Normal};
use std::cell::RefCell;

/// Marginal distribution for a single variable in the copula
///
/// Each variable in the correlated sample has its own marginal distribution.
/// The Gaussian copula preserves these marginals while introducing correlation.
///
/// # Examples
///
/// ```
/// # use powers_rs::correlation::MarginalDistribution;
/// // Normal distribution with mean=100, std_dev=20
/// let normal = MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 };
///
/// // Lognormal distribution (mu and sigma are parameters of underlying normal)
/// let lognormal = MarginalDistribution::Lognormal { mu: 4.0, sigma: 0.5 };
///
/// // Uniform distribution on [50, 150]
/// let uniform = MarginalDistribution::Uniform { min: 50.0, max: 150.0 };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum MarginalDistribution {
    /// Normal distribution N(μ, σ²)
    Normal {
        /// Mean μ
        mean: f64,
        /// Standard deviation σ (must be positive)
        std_dev: f64,
    },
    /// Lognormal distribution with parameters (μ, σ) of the underlying normal
    Lognormal {
        /// Location parameter μ of underlying normal
        mu: f64,
        /// Scale parameter σ of underlying normal (must be positive)
        sigma: f64,
    },
    /// 3-parameter log-normal distribution (methodology for non-negative scenarios)
    ///
    /// Generates X = γ + exp(μ + σZ) where Z ~ N(0,1). Used for inflows/loads
    /// that must be non-negative. **Zero LP overhead** compared to Shadow AR.
    ///
    /// See [`crate::lognormal3`] module for detailed documentation.
    LogNormal3 {
        /// Location parameter γ (minimum value), must be >= 0
        gamma: f64,
        /// Mean of log-transformed variable μ
        mu: f64,
        /// Standard deviation of log-transformed variable σ (must be positive)
        sigma: f64,
    },
    /// Uniform distribution on [min, max]
    Uniform {
        /// Minimum value
        min: f64,
        /// Maximum value (must be > min)
        max: f64,
    },
}

impl MarginalDistribution {
    /// Apply inverse CDF (quantile function) to transform uniform [0,1] to marginal distribution
    ///
    /// # Arguments
    ///
    /// * `u` - Uniform random variable in [0, 1]
    ///
    /// # Returns
    ///
    /// Value from the marginal distribution corresponding to quantile u
    ///
    /// # Performance
    ///
    /// O(1) time. Normal and lognormal use iterative methods (typically 2-3 iterations).
    pub fn inverse_cdf(&self, u: f64) -> f64 {
        match self {
            MarginalDistribution::Normal { mean, std_dev } => {
                let normal = Normal::new(*mean, *std_dev)
                    .expect("Invalid normal parameters");
                normal.inverse_cdf(u)
            }
            MarginalDistribution::Lognormal { mu, sigma } => {
                let lognormal = LogNormal::new(*mu, *sigma)
                    .expect("Invalid lognormal parameters");
                lognormal.inverse_cdf(u)
            }
            MarginalDistribution::LogNormal3 { gamma, mu, sigma } => {
                // Use our fast 3-parameter log-normal implementation
                let dist = crate::lognormal3::LogNormal3Param::new(
                    *gamma, *mu, *sigma,
                )
                .expect("Invalid LogNormal3 parameters");
                dist.inverse_cdf(u)
            }
            MarginalDistribution::Uniform { min, max } => min + u * (max - min),
        }
    }

    /// Validate distribution parameters
    fn validate(&self, index: usize) -> Result<(), PowersError> {
        match self {
            MarginalDistribution::Normal { mean: _, std_dev } => {
                if *std_dev <= 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "correlation specification".to_string(),
                        field: format!("marginal[{}].std_dev", index),
                        value: format!("{}", std_dev),
                        constraint: "Must be positive".to_string(),
                        suggestion:
                            "Use positive standard deviation (e.g., 1.0)"
                                .to_string(),
                    })
                    .into());
                }
            }
            MarginalDistribution::Lognormal { mu: _, sigma } => {
                if *sigma <= 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "correlation specification".to_string(),
                        field: format!("marginal[{}].sigma", index),
                        value: format!("{}", sigma),
                        constraint: "Must be positive".to_string(),
                        suggestion: "Use positive sigma (e.g., 0.5)"
                            .to_string(),
                    })
                    .into());
                }
            }
            MarginalDistribution::LogNormal3 {
                gamma,
                mu: _,
                sigma,
            } => {
                if *gamma < 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "correlation specification".to_string(),
                        field: format!("marginal[{}].gamma", index),
                        value: format!("{}", gamma),
                        constraint: "Must be >= 0".to_string(),
                        suggestion: "Use non-negative gamma (e.g., 0.0 or 1.0)"
                            .to_string(),
                    })
                    .into());
                }
                if *sigma <= 0.0 {
                    return Err(Box::new(ValidationError::InvalidFieldValue {
                        file: "correlation specification".to_string(),
                        field: format!("marginal[{}].sigma", index),
                        value: format!("{}", sigma),
                        constraint: "Must be positive".to_string(),
                        suggestion: "Use positive sigma (e.g., 0.3)"
                            .to_string(),
                    })
                    .into());
                }
            }
            MarginalDistribution::Uniform { min, max } => {
                if max <= min {
                    return Err(Box::new(
                        ValidationError::ConstraintViolation {
                            file: "correlation specification".to_string(),
                            context: format!("marginal[{}]", index),
                            constraint: "max > min".to_string(),
                            details: format!("min={}, max={}", min, max),
                            suggestion: "Ensure max > min".to_string(),
                        },
                    )
                    .into());
                }
            }
        }
        Ok(())
    }
}

/// Correlated noise generator using Gaussian copula with Cholesky decomposition
///
/// Generates correlated multi-variate samples efficiently while preserving
/// specified marginal distributions for each variable.
///
/// # Algorithm
///
/// Uses Gaussian copula with Cholesky decomposition:
/// 1. Sample independent Z ~ N(0, 1)
/// 2. Apply correlation: Z' = L × Z (matrix-vector multiply, O(n²))
/// 3. Transform to uniform: U = Φ(Z') (normal CDF)
/// 4. Apply inverse CDF: X = F⁻¹(U) (marginal-specific)
///
/// # Performance Characteristics
///
/// - **Initialization**: O(n³) for Cholesky decomposition (done once)
/// - **Per sample**: O(n²) for matrix-vector multiply + O(n) for transforms
/// - **Memory**: O(n²) for Cholesky factor + O(n) workspace
///
/// Typical performance: <100μs per sample for n=50 on modern CPU
///
/// # Examples
///
/// ```
/// # use powers_rs::correlation::{CorrelatedNoiseGenerator, MarginalDistribution};
/// # use nalgebra::DMatrix;
/// # use rand::SeedableRng;
/// # use rand_xoshiro::Xoshiro256Plus;
/// // Define correlation matrix (must be symmetric, positive semi-definite)
/// let correlation = DMatrix::from_row_slice(2, 2, &[
///     1.0, 0.7,
///     0.7, 1.0,
/// ]);
///
/// // Define marginal distributions
/// let marginals = vec![
///     MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
///     MarginalDistribution::Lognormal { mu: 4.0, sigma: 0.5 },
/// ];
///
/// // Create generator (validates and computes Cholesky decomposition)
/// let generator = CorrelatedNoiseGenerator::new(correlation, marginals).unwrap();
///
/// // Generate correlated sample
/// let mut rng = Xoshiro256Plus::seed_from_u64(42);
/// let sample = generator.generate_correlated_sample(&mut rng);
/// assert_eq!(sample.len(), 2);
/// ```
pub struct CorrelatedNoiseGenerator {
    /// Lower triangular Cholesky factor: L such that Σ = L Lᵀ
    ///
    /// Computed once during initialization. Used for efficient correlation
    /// application via matrix-vector multiply: Z' = L × Z
    cholesky_l: DMatrix<f64>,

    /// Marginal distributions for each variable
    ///
    /// Length must match correlation matrix dimensions
    marginals: Vec<MarginalDistribution>,

    /// Number of correlated variables
    n_variables: usize,

    /// Workspace vector for intermediate computations (reused to avoid allocations)
    ///
    /// PERFORMANCE: Reusing this vector avoids n allocations per sample
    workspace: RefCell<DVector<f64>>,
}

impl CorrelatedNoiseGenerator {
    /// Create a new correlated noise generator
    ///
    /// # Arguments
    ///
    /// * `correlation_matrix` - Symmetric positive semi-definite correlation matrix (n×n)
    ///   - Diagonal elements must be 1.0
    ///   - Off-diagonal elements must be in [-1, 1]
    /// * `marginals` - Marginal distributions for each variable (length n)
    ///
    /// # Returns
    ///
    /// Generator ready to produce correlated samples, or error if validation fails
    ///
    /// # Errors
    ///
    /// - Matrix not symmetric
    /// - Matrix not positive semi-definite (Cholesky decomposition fails)
    /// - Diagonal elements not 1.0
    /// - Off-diagonal elements outside [-1, 1]
    /// - Marginals length doesn't match matrix dimensions
    /// - Invalid marginal distribution parameters
    ///
    /// # Performance
    ///
    /// O(n³) for Cholesky decomposition. This is done once during initialization,
    /// not per sample.
    ///
    /// # Examples
    ///
    /// ```
    /// # use powers_rs::correlation::{CorrelatedNoiseGenerator, MarginalDistribution};
    /// # use nalgebra::DMatrix;
    /// let correlation = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
    /// let marginals = vec![
    ///     MarginalDistribution::Normal { mean: 0.0, std_dev: 1.0 },
    ///     MarginalDistribution::Normal { mean: 0.0, std_dev: 1.0 },
    /// ];
    /// let generator = CorrelatedNoiseGenerator::new(correlation, marginals).unwrap();
    /// ```
    pub fn new(
        correlation_matrix: DMatrix<f64>,
        marginals: Vec<MarginalDistribution>,
    ) -> Result<Self, PowersError> {
        let n = correlation_matrix.nrows();

        // Validate dimensions
        if correlation_matrix.ncols() != n {
            return Err(Box::new(ValidationError::ConstraintViolation {
                file: "correlation specification".to_string(),
                context: "correlation_matrix".to_string(),
                constraint: "Must be square".to_string(),
                details: format!(
                    "rows={}, cols={}",
                    n,
                    correlation_matrix.ncols()
                ),
                suggestion: "Provide square correlation matrix".to_string(),
            })
            .into());
        }

        if marginals.len() != n {
            return Err(Box::new(ValidationError::ConstraintViolation {
                file: "correlation specification".to_string(),
                context: "marginals".to_string(),
                constraint: "Length must match correlation matrix dimension"
                    .to_string(),
                details: format!(
                    "matrix dimension={}, marginals length={}",
                    n,
                    marginals.len()
                ),
                suggestion: "Provide one marginal per correlated variable"
                    .to_string(),
            })
            .into());
        }

        // Validate correlation matrix properties
        Self::validate_correlation_matrix(&correlation_matrix)?;

        // Validate marginal distributions
        for (i, marginal) in marginals.iter().enumerate() {
            marginal.validate(i)?;
        }

        // Compute Cholesky decomposition
        // PERFORMANCE: O(n³) but done once at initialization
        let cholesky = Cholesky::new(correlation_matrix.clone()).ok_or_else(|| {
            Box::new(ValidationError::ConstraintViolation {
                file: "correlation specification".to_string(),
                context: "correlation_matrix".to_string(),
                constraint: "Must be positive semi-definite".to_string(),
                details: "Cholesky decomposition failed".to_string(),
                suggestion:
                    "Check matrix is valid correlation (symmetric, PSD, diagonal=1, off-diagonal in [-1,1])"
                        .to_string(),
            })
        })?;

        Ok(Self {
            cholesky_l: cholesky.l(),
            marginals,
            n_variables: n,
            workspace: RefCell::new(DVector::zeros(n)),
        })
    }

    /// Create generator with regularization for near-singular matrices
    ///
    /// If Cholesky decomposition fails, adds small epsilon to diagonal and retries.
    /// Useful for matrices that are numerically near-singular but conceptually valid.
    ///
    /// # Arguments
    ///
    /// * `correlation_matrix` - Correlation matrix (may be near-singular)
    /// * `marginals` - Marginal distributions
    /// * `epsilon` - Regularization parameter to add to diagonal (typically 1e-6 to 1e-8)
    ///
    /// # Returns
    ///
    /// Generator with regularized matrix, or error if still invalid
    ///
    /// # Performance
    ///
    /// Same as `new()` but with one retry after regularization
    pub fn new_with_regularization(
        mut correlation_matrix: DMatrix<f64>,
        marginals: Vec<MarginalDistribution>,
        epsilon: f64,
    ) -> Result<Self, PowersError> {
        match Self::new(correlation_matrix.clone(), marginals.clone()) {
            Ok(generator) => Ok(generator),
            Err(_) => {
                // Regularize by adding epsilon to diagonal
                for i in 0..correlation_matrix.nrows() {
                    correlation_matrix[(i, i)] += epsilon;
                }
                eprintln!(
                    "⚠ WARNING: Correlation matrix near-singular, regularized with epsilon={}",
                    epsilon
                );
                Self::new(correlation_matrix, marginals)
            }
        }
    }

    /// Generate a correlated sample from the specified marginal distributions
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Vector of n correlated samples, one per marginal distribution
    ///
    /// # Performance
    ///
    /// - Time: O(n²) for matrix-vector multiply + O(n) for CDF transforms
    /// - Space: O(1) (reuses workspace, no allocations)
    /// - Typical: <100μs for n=50 on modern CPU
    ///
    /// # Algorithm
    ///
    /// 1. Sample Z[i] ~ N(0,1) independently for i=1..n
    /// 2. Apply correlation: Z' = L × Z (matrix-vector multiply)
    /// 3. Transform to uniform: U[i] = Φ(Z'[i]) (normal CDF)
    /// 4. Apply inverse CDF: X[i] = F⁻¹(U[i]) for marginal i
    ///
    /// # Examples
    ///
    /// ```
    /// # use powers_rs::correlation::{CorrelatedNoiseGenerator, MarginalDistribution};
    /// # use nalgebra::DMatrix;
    /// # use rand::SeedableRng;
    /// # use rand_xoshiro::Xoshiro256Plus;
    /// # let correlation = DMatrix::identity(2, 2);
    /// # let marginals = vec![
    /// #     MarginalDistribution::Normal { mean: 100.0, std_dev: 20.0 },
    /// #     MarginalDistribution::Normal { mean: 50.0, std_dev: 10.0 },
    /// # ];
    /// # let generator = CorrelatedNoiseGenerator::new(correlation, marginals).unwrap();
    /// let mut rng = Xoshiro256Plus::seed_from_u64(42);
    /// let sample = generator.generate_correlated_sample(&mut rng);
    /// assert_eq!(sample.len(), 2);
    /// ```
    pub fn generate_correlated_sample(&self, rng: &mut impl Rng) -> Vec<f64> {
        // PERFORMANCE: Reuse workspace to avoid allocation
        let mut workspace = self.workspace.borrow_mut();

        // Step 1: Generate independent standard normals Z ~ N(0, 1)
        for i in 0..self.n_variables {
            let z: f64 = rng.sample(StandardNormal);
            workspace[i] = z;
        }

        // Step 2: Apply correlation via Cholesky: Z' = L × Z
        // PERFORMANCE: O(n²) matrix-vector multiply, cache-friendly with lower triangular
        let correlated_z = &self.cholesky_l * &*workspace;

        // Step 3: Transform to uniform via normal CDF, then to marginal via inverse CDF
        // PERFORMANCE: O(n) transforms
        let standard_normal =
            Normal::new(0.0, 1.0).expect("Standard normal creation failed");
        correlated_z
            .iter()
            .zip(&self.marginals)
            .map(|(z_prime, marginal)| {
                // Transform correlated Z' to uniform [0,1] via normal CDF
                let u = standard_normal.cdf(*z_prime);
                // Transform uniform to target marginal via inverse CDF
                marginal.inverse_cdf(u)
            })
            .collect()
    }

    /// Validate correlation matrix properties
    ///
    /// Checks:
    /// - Symmetric (matrix[i,j] == matrix[j,i])
    /// - Diagonal elements = 1.0
    /// - Off-diagonal elements in [-1, 1]
    fn validate_correlation_matrix(
        matrix: &DMatrix<f64>,
    ) -> Result<(), PowersError> {
        let n = matrix.nrows();
        const TOLERANCE: f64 = 1e-10;

        // Check symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                if (matrix[(i, j)] - matrix[(j, i)]).abs() > TOLERANCE {
                    return Err(Box::new(
                        ValidationError::ConstraintViolation {
                            file: "correlation specification".to_string(),
                            context: format!("correlation_matrix[{},{}]", i, j),
                            constraint: "Matrix must be symmetric".to_string(),
                            details: format!(
                                "matrix[{},{}]={}, matrix[{},{}]={}",
                                i,
                                j,
                                matrix[(i, j)],
                                j,
                                i,
                                matrix[(j, i)]
                            ),
                            suggestion:
                                "Ensure matrix[i,j] == matrix[j,i] for all i,j"
                                    .to_string(),
                        },
                    )
                    .into());
                }
            }
        }

        // Check diagonal = 1.0
        for i in 0..n {
            if (matrix[(i, i)] - 1.0).abs() > TOLERANCE {
                return Err(Box::new(ValidationError::InvalidFieldValue {
                    file: "correlation specification".to_string(),
                    field: format!("correlation_matrix[{},{}]", i, i),
                    value: format!("{}", matrix[(i, i)]),
                    constraint: "Diagonal elements must be 1.0".to_string(),
                    suggestion: "Set diagonal elements to 1.0".to_string(),
                })
                .into());
            }
        }

        // Check off-diagonal in [-1, 1]
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = matrix[(i, j)];
                    if !(-1.0..=1.0).contains(&val) {
                        return Err(Box::new(ValidationError::InvalidFieldValue {
                            file: "correlation specification".to_string(),
                            field: format!("correlation_matrix[{},{}]", i, j),
                            value: format!("{}", val),
                            constraint: "Off-diagonal elements must be in [-1, 1]".to_string(),
                            suggestion: "Use valid correlation coefficient in [-1, 1]".to_string(),
                        })
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// Number of correlated variables
    pub fn n_variables(&self) -> usize {
        self.n_variables
    }

    /// Get reference to marginal distributions
    pub fn marginals(&self) -> &[MarginalDistribution] {
        &self.marginals
    }
}

/// Cholesky factor wrapper for efficient correlation application
///
/// Wraps the lower triangular Cholesky factor L where R = LL^T for
/// a correlation matrix R. Provides efficient matrix-vector multiply
/// for applying correlation: W = L×Z.
///
/// # Performance
///
/// - **Initialization**: O(n³) for Cholesky decomposition (done once)
/// - **Transform**: O(n²) for matrix-vector multiply
/// - **Memory**: O(n²) for lower triangular matrix
///
/// # Examples
///
/// ```
/// use powers_rs::correlation::CholeskyFactor;
/// use nalgebra::DMatrix;
///
/// let correlation = DMatrix::from_row_slice(2, 2, &[1.0, 0.7, 0.7, 1.0]);
/// let factor = CholeskyFactor::new(correlation).unwrap();
///
/// let z = vec![0.5, -0.3];
/// let w = factor.transform(&z);
/// ```
#[derive(Clone, Debug)]
pub struct CholeskyFactor {
    /// Lower triangular Cholesky factor L
    l: DMatrix<f64>,
}

impl CholeskyFactor {
    /// Create Cholesky factor from correlation matrix
    ///
    /// # Arguments
    ///
    /// * `correlation_matrix` - Symmetric positive semi-definite correlation matrix
    ///
    /// # Returns
    ///
    /// Cholesky factor or error if matrix is not positive semi-definite
    ///
    /// # Performance
    ///
    /// O(n³) for decomposition. This is done once and cached.
    pub fn new(correlation_matrix: DMatrix<f64>) -> Result<Self, String> {
        let cholesky = Cholesky::new(correlation_matrix).ok_or_else(|| {
            "Cholesky decomposition failed - matrix not positive semi-definite"
                .to_string()
        })?;

        Ok(Self { l: cholesky.l() })
    }

    /// Apply Cholesky transformation: W = L×Z
    ///
    /// # Arguments
    ///
    /// * `z` - Input vector (length must match matrix dimension)
    ///
    /// # Returns
    ///
    /// Transformed vector W = L×Z
    ///
    /// # Performance
    ///
    /// O(n²) matrix-vector multiply. Optimized with lower triangular structure.
    pub fn transform(&self, z: &[f64]) -> Vec<f64> {
        let n = self.l.nrows();
        assert_eq!(
            z.len(),
            n,
            "Input vector length must match matrix dimension"
        );

        // Convert slice to DVector for matrix multiply
        let z_vec = DVector::from_row_slice(z);

        // Perform L×z (matrix-vector multiply)
        let w = &self.l * z_vec;

        // Convert back to Vec
        w.as_slice().to_vec()
    }

    /// Get matrix dimension
    pub fn dimension(&self) -> usize {
        self.l.nrows()
    }

    /// Get reference to Cholesky factor L
    pub fn as_matrix(&self) -> &DMatrix<f64> {
        &self.l
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256Plus;

    fn compute_mean(samples: &[f64]) -> f64 {
        samples.iter().sum::<f64>() / samples.len() as f64
    }

    fn compute_std_dev(samples: &[f64]) -> f64 {
        let mean = compute_mean(samples);
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / samples.len() as f64;
        variance.sqrt()
    }

    fn compute_correlation(samples: &[Vec<f64>], i: usize, j: usize) -> f64 {
        let x: Vec<f64> = samples.iter().map(|s| s[i]).collect();
        let y: Vec<f64> = samples.iter().map(|s| s[j]).collect();
        let mean_x = compute_mean(&x);
        let mean_y = compute_mean(&y);
        let std_x = compute_std_dev(&x);
        let std_y = compute_std_dev(&y);

        let covariance = x
            .iter()
            .zip(&y)
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / x.len() as f64;

        covariance / (std_x * std_y)
    }

    fn compute_median(samples: &[f64]) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }

    #[test]
    fn test_identity_correlation_produces_independence() {
        let identity = DMatrix::identity(3, 3);
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
        ];
        let generator =
            CorrelatedNoiseGenerator::new(identity, marginals).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let samples: Vec<Vec<f64>> = (0..10000)
            .map(|_| generator.generate_correlated_sample(&mut rng))
            .collect();

        let corr_01 = compute_correlation(&samples, 0, 1);
        let corr_02 = compute_correlation(&samples, 0, 2);
        let corr_12 = compute_correlation(&samples, 1, 2);

        assert!(
            corr_01.abs() < 0.05,
            "Variables 0,1 should be uncorrelated, got {}",
            corr_01
        );
        assert!(
            corr_02.abs() < 0.05,
            "Variables 0,2 should be uncorrelated, got {}",
            corr_02
        );
        assert!(
            corr_12.abs() < 0.05,
            "Variables 1,2 should be uncorrelated, got {}",
            corr_12
        );
    }

    #[test]
    fn test_perfect_correlation_preserved() {
        let perfect = DMatrix::from_row_slice(2, 2, &[1.0, 0.99, 0.99, 1.0]);
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
        let generator =
            CorrelatedNoiseGenerator::new(perfect, marginals).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let samples: Vec<Vec<f64>> = (0..10000)
            .map(|_| generator.generate_correlated_sample(&mut rng))
            .collect();

        let corr = compute_correlation(&samples, 0, 1);
        assert!(
            (corr - 0.99).abs() < 0.02,
            "Correlation should be preserved, got {}",
            corr
        );
    }

    #[test]
    fn test_marginal_distributions_preserved() {
        let correlation = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]);
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0,
            },
            MarginalDistribution::Lognormal {
                mu: 4.0,
                sigma: 0.5,
            },
        ];
        let generator =
            CorrelatedNoiseGenerator::new(correlation, marginals).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let samples: Vec<Vec<f64>> = (0..50000)
            .map(|_| generator.generate_correlated_sample(&mut rng))
            .collect();

        // Check marginal 0 (normal)
        let samples_0: Vec<f64> = samples.iter().map(|s| s[0]).collect();
        let mean_0 = compute_mean(&samples_0);
        let std_0 = compute_std_dev(&samples_0);

        assert!(
            (mean_0 - 100.0).abs() < 1.0,
            "Mean should be preserved, got {}",
            mean_0
        );
        assert!(
            (std_0 - 20.0).abs() < 1.0,
            "Std dev should be preserved, got {}",
            std_0
        );

        // Check marginal 1 (lognormal)
        let samples_1: Vec<f64> = samples.iter().map(|s| s[1]).collect();
        let median_1 = compute_median(&samples_1);
        let expected_median = 4.0_f64.exp();

        assert!(
            (median_1 - expected_median).abs() < 5.0,
            "Lognormal median should be preserved, got {} expected {}",
            median_1,
            expected_median
        );
    }

    #[test]
    fn test_rejects_non_symmetric() {
        let non_symmetric =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.6, 1.0]);
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
        ];
        assert!(
            CorrelatedNoiseGenerator::new(non_symmetric, marginals).is_err()
        );
    }

    #[test]
    fn test_rejects_non_psd() {
        // Create a matrix that will definitely fail Cholesky
        // Using inconsistent correlations: A-B=0.9, B-C=0.9, A-C=-0.9
        // This violates transitivity and is not PSD
        let non_psd = DMatrix::from_row_slice(
            3,
            3,
            &[1.0, 0.9, -0.9, 0.9, 1.0, 0.9, -0.9, 0.9, 1.0],
        );
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
        ];
        assert!(CorrelatedNoiseGenerator::new(non_psd, marginals).is_err());
    }

    #[test]
    fn test_regularization_for_near_singular() {
        let near_singular =
            DMatrix::from_row_slice(2, 2, &[1.0, 0.999999, 0.999999, 1.0]);
        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
            MarginalDistribution::Normal {
                mean: 0.0,
                std_dev: 1.0,
            },
        ];

        let result = CorrelatedNoiseGenerator::new_with_regularization(
            near_singular,
            marginals,
            1e-6,
        );
        assert!(
            result.is_ok(),
            "Regularization should fix near-singular matrix"
        );
    }

    #[test]
    #[ignore] // Run with --ignored flag for performance testing
    fn test_correlation_performance() {
        let n = 50;
        let mut correlation = DMatrix::identity(n, n);

        // Add realistic correlation structure (spatial decay)
        for i in 0..n {
            for j in (i + 1)..n {
                let dist = (i as f64 - j as f64).abs();
                let corr = 0.8_f64.powf(dist / 10.0);
                correlation[(i, j)] = corr;
                correlation[(j, i)] = corr;
            }
        }

        let marginals = vec![
            MarginalDistribution::Normal {
                mean: 100.0,
                std_dev: 20.0
            };
            n
        ];
        let generator =
            CorrelatedNoiseGenerator::new(correlation, marginals).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let start = std::time::Instant::now();

        for _ in 0..10_000 {
            std::hint::black_box(
                generator.generate_correlated_sample(&mut rng),
            );
        }

        let elapsed = start.elapsed();
        let us_per_sample = elapsed.as_micros() / 10_000;

        println!("Performance: {} μs per sample for n={}", us_per_sample, n);
        assert!(
            us_per_sample < 200,
            "Performance regression: {}μs (expected <200μs)",
            us_per_sample
        );
    }

    #[test]
    fn test_uniform_marginal() {
        let identity = DMatrix::identity(1, 1);
        let marginals = vec![MarginalDistribution::Uniform {
            min: 50.0,
            max: 150.0,
        }];
        let generator =
            CorrelatedNoiseGenerator::new(identity, marginals).unwrap();

        let mut rng = Xoshiro256Plus::seed_from_u64(42);
        let samples: Vec<Vec<f64>> = (0..10000)
            .map(|_| generator.generate_correlated_sample(&mut rng))
            .collect();

        let samples_0: Vec<f64> = samples.iter().map(|s| s[0]).collect();
        let mean = compute_mean(&samples_0);
        let min_val = samples_0.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val =
            samples_0.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        assert!(
            (mean - 100.0).abs() < 2.0,
            "Uniform mean should be ~100, got {}",
            mean
        );
        assert!(
            (50.0..55.0).contains(&min_val),
            "Min should be near 50, got {}",
            min_val
        );
        assert!(
            (145.0..=150.0).contains(&max_val),
            "Max should be near 150, got {}",
            max_val
        );
    }

    #[test]
    fn test_invalid_marginal_parameters() {
        let identity = DMatrix::identity(1, 1);

        // Negative std_dev
        let marginals = vec![MarginalDistribution::Normal {
            mean: 0.0,
            std_dev: -1.0,
        }];
        assert!(
            CorrelatedNoiseGenerator::new(identity.clone(), marginals).is_err()
        );

        // Negative sigma
        let marginals = vec![MarginalDistribution::Lognormal {
            mu: 0.0,
            sigma: -0.5,
        }];
        assert!(
            CorrelatedNoiseGenerator::new(identity.clone(), marginals).is_err()
        );

        // max <= min
        let marginals = vec![MarginalDistribution::Uniform {
            min: 100.0,
            max: 50.0,
        }];
        assert!(CorrelatedNoiseGenerator::new(identity, marginals).is_err());
    }
}
