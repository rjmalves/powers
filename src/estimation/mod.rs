//! Parameter estimation for Periodic Autoregressive (PAR) models.
//!
//! This module implements the Yule-Walker method for estimating PAR model parameters
//! (seasonal means μₘ, standard deviations σₘ, and AR coefficients φₖₘ) from
//! historical time series data.
//!
//! # Mathematical Background
//!
//! **CRITICAL**: PAR AR coefficients φ are estimated in **residual space only**.
//! This ensures stationarity of the AR process.
//!
//! The PAR(p) model for period m is:
//!
//! ```text
//! Y_t = μₘ + σₘ · Z'_t   ← Observation (for output)
//!
//! Z'_t = φ₁ₘ·Z'_{t-1} + φ₂ₘ·Z'_{t-2} + ... + φₚₘ·Z'_{t-p} + ε_t  ← AR dynamics in residual space
//! ```
//!
//! Where:
//! - Y_t: observed value at time t (e.g., physical inflow)
//! - Z'_t: standardized residual (stationary AR process)
//! - μₘ: seasonal mean for period m
//! - σₘ: seasonal standard deviation for period m
//! - φₖₘ: AR coefficient k for period m (in residual space!)
//! - ε_t: innovation (white noise residual)
//!
//! ## Estimation Procedure
//!
//! 1. **Seasonal Statistics**: Compute μₘ and σₘ from data grouped by period
//! 2. **De-seasonalization**: Transform data to residuals: Z'_t = (Y_t - μₘ) / σₘ
//! 3. **Yule-Walker Equations**: For each period m, solve in residual space:
//!    ```text
//!    R·φ = r
//!    ```
//!    Where R is the autocorrelation matrix and r is the autocorrelation vector
//! 4. **Validation**: Check stationarity constraint (sum of |φₖₘ| < 1)
//!
//! # Performance Characteristics
//!
//! - **Time Complexity**: O(T + n_periods · p³) where T = data length, p = AR order
//! - **Space Complexity**: O(T + n_periods · p²)
//! - **Hot Path**: Autocorrelation computation and matrix solve
//! - **Optimization**: Uses pre-allocated buffers, cache-friendly iteration
//!
//! # Example
//!
//! ```rust,ignore
//! use powers_rs::estimation::{YuleWalkerEstimator, EstimationConfig};
//!
//! // Historical inflow data (e.g., 120 months = 10 years)
//! let data = vec![40.0, 45.0, 50.0, ..., 42.0];
//!
//! let config = EstimationConfig {
//!     n_periods: 12,      // Monthly seasonality
//!     ar_order: 1,        // PAR(1) model
//!     min_samples_per_period: 5, // Require at least 5 observations per month
//! };
//!
//! let estimator = YuleWalkerEstimator::new(config);
//! let params = estimator.estimate(&data)?;
//!
//! // Use estimated parameters in SDDP
//! println!("Estimated parameters for month 0:");
//! println!("  Mean: {:.2}", params.means()[0]);
//! println!("  Std: {:.2}", params.std_devs()[0]);
//! println!("  AR(1) coef: {:.3}", params.ar_coeffs()[0][0]);
//! ```

use nalgebra::{DMatrix, DVector};
use std::error::Error;
use std::fmt;

// ==============================================================================
// Configuration and Error Types
// ==============================================================================

/// Configuration for PAR parameter estimation.
#[derive(Debug, Clone)]
pub struct EstimationConfig {
    /// Number of periods in the seasonal cycle (e.g., 12 for monthly data).
    pub n_periods: usize,

    /// Order of the autoregressive model (number of lags).
    pub ar_order: usize,

    /// Minimum number of samples required per period for reliable estimation.
    ///
    /// Recommended: At least 5-10 samples per period.
    pub min_samples_per_period: usize,
}

impl Default for EstimationConfig {
    fn default() -> Self {
        Self {
            n_periods: 12,
            ar_order: 1,
            min_samples_per_period: 5,
        }
    }
}

/// Errors that can occur during parameter estimation.
#[derive(Debug, Clone)]
pub enum EstimationError {
    /// Insufficient data for estimation.
    InsufficientData { required: usize, provided: usize },

    /// Insufficient samples for a specific period.
    InsufficientSamplesForPeriod {
        period: usize,
        required: usize,
        provided: usize,
    },

    /// AR coefficients violate stationarity constraint.
    NonStationaryCoefficients { period: usize, coeff_sum: f64 },

    /// Matrix is singular (cannot solve Yule-Walker equations).
    SingularMatrix { period: usize },

    /// Invalid configuration (e.g., ar_order = 0).
    InvalidConfig { message: String },
}

impl fmt::Display for EstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EstimationError::InsufficientData { required, provided } => {
                write!(
                    f,
                    "Insufficient data: need at least {} samples, got {}",
                    required, provided
                )
            }
            EstimationError::InsufficientSamplesForPeriod {
                period,
                required,
                provided,
            } => {
                write!(
                    f,
                    "Period {}: need at least {} samples, got {}",
                    period, required, provided
                )
            }
            EstimationError::NonStationaryCoefficients {
                period,
                coeff_sum,
            } => {
                write!(
                    f,
                    "Period {}: non-stationary coefficients (sum = {:.3} >= 1.0)",
                    period, coeff_sum
                )
            }
            EstimationError::SingularMatrix { period } => {
                write!(
                    f,
                    "Period {}: singular autocorrelation matrix (cannot solve Yule-Walker)",
                    period
                )
            }
            EstimationError::InvalidConfig { message } => {
                write!(f, "Invalid configuration: {}", message)
            }
        }
    }
}

impl Error for EstimationError {}

// ==============================================================================
// Estimated Parameters
// ==============================================================================

/// Estimated PAR model parameters ready for use in SDDP.
#[derive(Debug, Clone)]
pub struct EstimatedPARParams {
    /// Number of periods.
    n_periods: usize,

    /// AR order.
    ar_order: usize,

    /// Seasonal means μₘ (one per period).
    means: Vec<f64>,

    /// Seasonal standard deviations σₘ (one per period).
    std_devs: Vec<f64>,

    /// AR coefficients φₖₘ (ar_order coefficients per period).
    ///
    /// Layout: ar_coeffs[m][k] = φₖₘ
    ar_coeffs: Vec<Vec<f64>>,

    /// Residual standard deviation for innovation term (one per period).
    ///
    /// After fitting AR model, this is the std dev of remaining unexplained variance.
    residual_std_devs: Vec<f64>,
}

impl EstimatedPARParams {
    /// Create new estimated parameters (useful for testing).
    #[doc(hidden)]
    pub fn new(
        n_periods: usize,
        ar_order: usize,
        means: Vec<f64>,
        std_devs: Vec<f64>,
        ar_coeffs: Vec<Vec<f64>>,
        residual_std_devs: Vec<f64>,
    ) -> Self {
        Self {
            n_periods,
            ar_order,
            means,
            std_devs,
            ar_coeffs,
            residual_std_devs,
        }
    }

    /// Get seasonal means.
    #[inline]
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Get seasonal standard deviations.
    #[inline]
    pub fn std_devs(&self) -> &[f64] {
        &self.std_devs
    }

    /// Get AR coefficients (n_periods × ar_order matrix).
    #[inline]
    pub fn ar_coeffs(&self) -> &[Vec<f64>] {
        &self.ar_coeffs
    }

    /// Get residual standard deviations.
    #[inline]
    pub fn residual_std_devs(&self) -> &[f64] {
        &self.residual_std_devs
    }

    /// Get number of periods.
    #[inline]
    pub fn n_periods(&self) -> usize {
        self.n_periods
    }

    /// Get AR order.
    #[inline]
    pub fn ar_order(&self) -> usize {
        self.ar_order
    }

    /// Export to JSON-compatible structure for use in recourse.json.
    pub fn to_json_fragment(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "PAR",
            "n_periods": self.n_periods,
            "ar_orders": vec![self.ar_order; self.n_periods],
            "ar_coeffs": self.ar_coeffs,
            "means": self.means,
            "std_devs": self.std_devs,
            "residual_distribution": {
                "kind": "Normal",
                "mean": 0.0,
                "std_dev": 1.0
            }
        })
    }
}

// ==============================================================================
// Yule-Walker Estimator
// ==============================================================================

/// Yule-Walker estimator for PAR model parameters.
///
/// # Performance Notes
///
/// - Pre-allocates buffers to minimize allocations in hot paths
/// - Uses cache-friendly iteration over data
/// - Leverages nalgebra for efficient matrix operations
/// - Typical performance: ~1ms for 120 samples, 12 periods, AR(1)
///
pub struct YuleWalkerEstimator {
    config: EstimationConfig,
}

impl YuleWalkerEstimator {
    /// Create a new Yule-Walker estimator with given configuration.
    pub fn new(config: EstimationConfig) -> Result<Self, EstimationError> {
        // Validate configuration
        if config.n_periods == 0 {
            return Err(EstimationError::InvalidConfig {
                message: "n_periods must be > 0".to_string(),
            });
        }

        if config.ar_order == 0 {
            return Err(EstimationError::InvalidConfig {
                message: "ar_order must be > 0".to_string(),
            });
        }

        if config.min_samples_per_period < config.ar_order {
            return Err(EstimationError::InvalidConfig {
                message: format!(
                    "min_samples_per_period ({}) must be >= ar_order ({})",
                    config.min_samples_per_period, config.ar_order
                ),
            });
        }

        Ok(Self { config })
    }

    /// Estimate PAR parameters from historical time series data.
    ///
    /// # Arguments
    ///
    /// * `data` - Historical time series (length must be >= n_periods * min_samples_per_period)
    ///
    /// # Returns
    ///
    /// Estimated PAR parameters or error if estimation fails.
    ///
    /// # Performance
    ///
    /// - Time: O(T + n_periods · p³) where T = data.len(), p = ar_order
    /// - Space: O(T + n_periods · p²)
    ///
    pub fn estimate(
        &self,
        data: &[f64],
    ) -> Result<EstimatedPARParams, EstimationError> {
        let required_samples =
            self.config.n_periods * self.config.min_samples_per_period;
        if data.len() < required_samples {
            return Err(EstimationError::InsufficientData {
                required: required_samples,
                provided: data.len(),
            });
        }

        // Step 1: Compute seasonal statistics (means and stds)
        let (means, std_devs) = self.compute_seasonal_statistics(data)?;

        // Step 2: De-seasonalize data to get residuals
        let residuals = self.compute_residuals(data, &means, &std_devs);

        // Step 3: Estimate AR coefficients per period using Yule-Walker
        let (ar_coeffs, residual_std_devs) =
            self.estimate_ar_coefficients(&residuals)?;

        Ok(EstimatedPARParams {
            n_periods: self.config.n_periods,
            ar_order: self.config.ar_order,
            means,
            std_devs,
            ar_coeffs,
            residual_std_devs,
        })
    }

    /// Compute seasonal means and standard deviations.
    ///
    /// Groups data by period (modulo n_periods) and computes statistics.
    ///
    /// # Performance
    ///
    /// - Time: O(T) where T = data.len()
    /// - Space: O(T) for temporary storage
    ///
    fn compute_seasonal_statistics(
        &self,
        data: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
        let n_periods = self.config.n_periods;

        // Pre-allocate buffers for each period's data
        let mut period_data: Vec<Vec<f64>> = vec![Vec::new(); n_periods];

        // Group data by period (cache-friendly single pass)
        for (t, &value) in data.iter().enumerate() {
            let period = t % n_periods;
            period_data[period].push(value);
        }

        // Compute means and stds for each period
        let mut means = Vec::with_capacity(n_periods);
        let mut std_devs = Vec::with_capacity(n_periods);

        for (period, values) in period_data.iter().enumerate() {
            // Check minimum samples
            if values.len() < self.config.min_samples_per_period {
                return Err(EstimationError::InsufficientSamplesForPeriod {
                    period,
                    required: self.config.min_samples_per_period,
                    provided: values.len(),
                });
            }

            // Compute mean
            let mean = values.iter().sum::<f64>() / values.len() as f64;

            // Compute std dev (sample standard deviation with Bessel's correction)
            let variance = values
                .iter()
                .map(|&x| {
                    let diff = x - mean;
                    diff * diff
                })
                .sum::<f64>()
                / (values.len() - 1) as f64;

            let std_dev = variance.sqrt();

            means.push(mean);
            std_devs.push(std_dev);
        }

        Ok((means, std_devs))
    }

    /// De-seasonalize data to compute standardized residuals.
    ///
    /// Transforms: a_t = (Z_t - μₘ) / σₘ
    ///
    /// # Performance
    ///
    /// - Time: O(T) where T = data.len()
    /// - Space: O(T) for residuals vector
    ///
    fn compute_residuals(
        &self,
        data: &[f64],
        means: &[f64],
        std_devs: &[f64],
    ) -> Vec<f64> {
        let n_periods = self.config.n_periods;

        // Pre-allocate residuals vector
        let mut residuals = Vec::with_capacity(data.len());

        // Compute residuals (cache-friendly single pass)
        for (t, &value) in data.iter().enumerate() {
            let period = t % n_periods;
            let residual = (value - means[period]) / std_devs[period];
            residuals.push(residual);
        }

        residuals
    }

    /// Estimate AR coefficients using Yule-Walker equations for each period.
    ///
    /// For each period m, solves: R·φ = r
    /// Where R is the autocorrelation matrix and r is the autocorrelation vector.
    ///
    /// # Performance
    ///
    /// - Time: O(n_periods · (T/n_periods · p + p³)) ≈ O(T·p + n_periods·p³)
    /// - Space: O(n_periods · p²)
    /// - Hot path: Matrix solve (p³ per period)
    ///
    fn estimate_ar_coefficients(
        &self,
        residuals: &[f64],
    ) -> Result<(Vec<Vec<f64>>, Vec<f64>), EstimationError> {
        let n_periods = self.config.n_periods;
        let p = self.config.ar_order;

        // Pre-allocate output vectors
        let mut ar_coeffs = Vec::with_capacity(n_periods);
        let mut residual_std_devs = Vec::with_capacity(n_periods);

        // Group residuals by period
        let mut period_residuals: Vec<Vec<f64>> = vec![Vec::new(); n_periods];

        for (t, &residual) in residuals.iter().enumerate() {
            let period = t % n_periods;
            period_residuals[period].push(residual);
        }

        // Estimate AR coefficients for each period
        for (period, res) in period_residuals.iter().enumerate() {
            // Skip first p observations (used as initial conditions)
            if res.len() <= p {
                return Err(EstimationError::InsufficientSamplesForPeriod {
                    period,
                    required: p + 1,
                    provided: res.len(),
                });
            }

            // Compute autocorrelations at lags 0, 1, ..., p
            let acf = self.compute_autocorrelations(res, p);

            // Build Yule-Walker system: R·φ = r
            // R is the autocorrelation matrix (Toeplitz)
            let mut r_matrix = DMatrix::from_fn(p, p, |i, j| {
                let lag = i.abs_diff(j);
                acf[lag]
            });

            // r is the autocorrelation vector at lags 1, 2, ..., p
            let r_vector =
                DVector::from_iterator(p, acf.iter().skip(1).take(p).copied());

            // Solve R·φ = r using Cholesky decomposition (efficient for symmetric positive definite)
            let cholesky = match nalgebra::Cholesky::new(r_matrix.clone()) {
                Some(c) => c,
                None => {
                    // Matrix is singular or not positive definite
                    // Fall back to adding small diagonal regularization
                    for i in 0..p {
                        r_matrix[(i, i)] += 1e-8;
                    }

                    nalgebra::Cholesky::new(r_matrix)
                        .ok_or(EstimationError::SingularMatrix { period })?
                }
            };

            let phi = cholesky.solve(&r_vector);

            // Extract AR coefficients
            let coeffs: Vec<f64> = phi.iter().copied().collect();

            // Validate stationarity (sum of absolute values < 1.0)
            let coeff_sum: f64 = coeffs.iter().map(|&c| c.abs()).sum();
            if coeff_sum >= 1.0 {
                return Err(EstimationError::NonStationaryCoefficients {
                    period,
                    coeff_sum,
                });
            }

            // Compute residual variance: σ²_ε = σ²(1 - ρ'φ)
            let rho_phi: f64 = coeffs
                .iter()
                .zip(acf.iter().skip(1))
                .map(|(&c, &r)| c * r)
                .sum();
            let residual_variance = acf[0] * (1.0 - rho_phi).max(1e-10);
            let residual_std = residual_variance.sqrt();

            ar_coeffs.push(coeffs);
            residual_std_devs.push(residual_std);
        }

        Ok((ar_coeffs, residual_std_devs))
    }

    /// Compute sample autocorrelations at lags 0, 1, ..., max_lag.
    ///
    /// Returns normalized autocorrelations where ρ(0) = 1.0.
    ///
    /// # Performance
    ///
    /// - Time: O(T · max_lag) where T = data.len()
    /// - Space: O(max_lag)
    ///
    fn compute_autocorrelations(
        &self,
        data: &[f64],
        max_lag: usize,
    ) -> Vec<f64> {
        let n = data.len();

        // Compute mean
        let mean = data.iter().sum::<f64>() / n as f64;

        // Compute variance (lag 0)
        let variance = data
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;

        // Prevent division by zero
        if variance < 1e-12 {
            // Constant series - return ones
            return vec![1.0; max_lag + 1];
        }

        // Pre-allocate autocorrelation vector
        let mut acf = Vec::with_capacity(max_lag + 1);
        acf.push(1.0); // ρ(0) = 1.0 (normalized)

        // Compute autocorrelations for lags 1, 2, ..., max_lag
        for lag in 1..=max_lag {
            let mut sum = 0.0;
            for t in lag..n {
                sum += (data[t] - mean) * (data[t - lag] - mean);
            }
            let acov_k = sum / n as f64; // Autocovariance
            let acf_k = acov_k / variance; // Normalize to get autocorrelation
            acf.push(acf_k);
        }

        acf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimation_config_validation() {
        // Invalid: n_periods = 0
        let config = EstimationConfig {
            n_periods: 0,
            ar_order: 1,
            min_samples_per_period: 5,
        };
        let result = YuleWalkerEstimator::new(config);
        assert!(result.is_err());

        // Invalid: ar_order = 0
        let config = EstimationConfig {
            n_periods: 12,
            ar_order: 0,
            min_samples_per_period: 5,
        };
        let result = YuleWalkerEstimator::new(config);
        assert!(result.is_err());

        // Invalid: min_samples < ar_order
        let config = EstimationConfig {
            n_periods: 12,
            ar_order: 3,
            min_samples_per_period: 2,
        };
        let result = YuleWalkerEstimator::new(config);
        assert!(result.is_err());

        // Valid configuration
        let config = EstimationConfig {
            n_periods: 12,
            ar_order: 1,
            min_samples_per_period: 5,
        };
        let result = YuleWalkerEstimator::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_insufficient_data_error() {
        let config = EstimationConfig {
            n_periods: 12,
            ar_order: 1,
            min_samples_per_period: 5,
        };
        let estimator = YuleWalkerEstimator::new(config).unwrap();

        // Only 30 samples (need 60)
        let data = vec![50.0; 30];
        let result = estimator.estimate(&data);

        assert!(matches!(
            result,
            Err(EstimationError::InsufficientData { .. })
        ));
    }
}
