//! Seasonal parameters container for Periodic Autoregressive (PAR) models
//!
//! This module provides the `SeasonalParams` type, which encapsulates all
//! seasonal parameters (μₘ, σₘ, φₖₘ) for PAR models and validates them
//! according to methodology requirements.
//!
//! # PAR(p) Model
//!
//! The PAR model equation:
//!
//! ```text
//! Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
//! ```
//!
//! where:
//! - `m = season_id` (period index, wraps around: m = t mod period)
//! - `μₘ`: seasonal mean for period m
//! - `σₘ`: seasonal standard deviation for period m (must be > 0)
//! - `φₖₘ`: AR coefficient k for period m (k = 1..pₘ)
//! - `pₘ`: AR order for period m (can vary by season)
//! - `aₜ`: residual from residual_distribution (e.g., LogNormal3)
//!
//! # Validation
//!
//! `SeasonalParams::new()` validates:
//!
//! 1. **Length consistency**: All arrays (ar_orders, ar_coefficients, means, stds)
//!    must have length equal to `period`
//! 2. **Positivity**: All σₘ > 0 (standard deviations must be positive)
//! 3. **Coefficient consistency**: ar_coefficients[m].len() == ar_orders[m]
//! 4. **Stationarity**: Each period's AR polynomial must be stationary
//!
//! # Stationarity Conditions
//!
//! - **AR(0)**: Always stationary (no AR component)
//! - **AR(1)**: |φ₁| < 1
//! - **AR(2)**: |φ₂| < 1, φ₁ + φ₂ < 1, φ₂ - φ₁ < 1
//! - **AR(p)**: Sum of absolute coefficients < 1 (heuristic for MVP)
//!
//! # Example
//!
//! ```rust
//! use powers_rs::seasonal_params::SeasonalParams;
//!
//! // Create 12-period PAR(1) model with all coefficients = 0.7
//! let params = SeasonalParams::new(
//!     12,
//!     vec![1; 12],                    // AR(1) for all periods
//!     vec![vec![0.7]; 12],            // φ = 0.7 for all periods
//!     vec![100.0, 120.0, 150.0, 180.0, 200.0, 180.0,
//!          150.0, 120.0, 100.0, 90.0, 80.0, 90.0],  // Seasonal means
//!     vec![20.0, 25.0, 30.0, 35.0, 40.0, 35.0,
//!          30.0, 25.0, 20.0, 18.0, 15.0, 18.0],     // Seasonal stds
//! ).expect("Valid PAR(1) parameters");
//!
//! // Access parameters for specific period
//! assert_eq!(params.get_mean(1), 120.0);
//! assert_eq!(params.get_ar_coeffs(1), &[0.7]);
//! ```

use crate::error::PowersError;
use crate::input::TemporalModel;

/// Container for periodic AR seasonal parameters
///
/// # Mathematical Notation
///
/// - μₘ: seasonal mean for season m
/// - σₘ: seasonal standard deviation for season m
/// - φₖₘ: AR coefficient k for season m (k = 1..pₘ)
/// - pₘ: AR order for season m
///
/// # PAR(p) Equation
///
/// ```text
/// Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
///
/// where:
///   m = t mod num_seasons (seasonal index, maps to season_id in graph nodes)
///   aₜ ~ residual_distribution (e.g., LogNormal3)
/// ```
///
/// # Stationarity Requirement
///
/// For each season m, the AR polynomial must be stationary:
/// - AR(1): |φ₁ₘ| < 1
/// - AR(2): |φ₂ₘ| < 1, φ₁ₘ + φ₂ₘ < 1, φ₂ₘ - φ₁ₘ < 1
/// - AR(p): spectral radius of companion matrix < 1 (heuristic: sum|φₖ| < 1)
///
/// # Performance
///
/// - **Size**: ~40-80 bytes (depending on num_seasons and AR orders)
/// - **Access**: O(1) via helper methods with seasonal wraparound
/// - **Validation**: O(num_seasons) during construction (one-time cost)
/// - **Cloning**: Inexpensive for small num_seasons (<= 52), use references when possible
///
#[derive(Debug, Clone)]
pub struct SeasonalParams {
    /// Number of seasons in seasonal cycle (e.g., 12 for monthly, 4 for quarterly)
    ///
    /// Must match the num_seasons in PeriodicAutoregressive variant.
    /// All seasonal arrays (ar_orders, ar_coefficients, means, stds) must have this length.
    pub num_seasons: usize,

    /// AR order for each season [p₀, p₁, ..., p_{num_seasons-1}]
    ///
    /// Each element specifies the AR order for that season.
    /// Orders can vary by season (e.g., AR(1) in dry season, AR(2) in wet season).
    pub ar_orders: Vec<usize>,

    /// AR coefficients for each season
    ///
    /// Outer vec length = num_seasons, inner vec[m] length = ar_orders[m].
    /// Example: ar_coefficients[0] = [φ₁₀, φ₂₀] for season 0 with AR(2)
    pub ar_coefficients: Vec<Vec<f64>>,

    /// Seasonal means [μ₀, μ₁, ..., μ_{num_seasons-1}]
    ///
    /// Mean value for each season (μₘ in PAR(p) notation).
    /// Example: For monthly inflows, might be [100.0, 120.0, 150.0, ..., 90.0]
    pub means: Vec<f64>,

    /// Seasonal standard deviations [σ₀, σ₁, ..., σ_{num_seasons-1}]
    ///
    /// Standard deviation for each season (σₘ in PAR(p) notation).
    /// All values must be > 0 (validated during construction).
    /// Example: For monthly inflows, might be [20.0, 25.0, 30.0, ..., 18.0]
    pub stds: Vec<f64>,
}

impl SeasonalParams {
    /// Construct and validate seasonal parameters
    ///
    /// # Validation Steps
    ///
    /// 1. **Length consistency**: All arrays must have length `num_seasons`
    /// 2. **Positivity**: All σₘ must be positive (> 0)
    /// 3. **Coefficient consistency**: ar_coefficients[m].len() must equal ar_orders[m]
    /// 4. **Stationarity**: Each season's AR polynomial must be stationary
    ///
    /// # Arguments
    ///
    /// - `num_seasons`: Seasonal cycle length (e.g., 12 for monthly, 4 for quarterly)
    /// - `ar_orders`: AR order for each season
    /// - `ar_coefficients`: AR coefficients for each season (nested vec)
    /// - `means`: Seasonal means (μₘ)
    /// - `stds`: Seasonal standard deviations (σₘ)
    ///
    /// # Errors
    ///
    /// Returns `PowersError::InvalidInput` if any validation fails, with a descriptive
    /// error message indicating the specific constraint violation.
    ///
    /// # Performance
    ///
    /// O(num_seasons) validation cost during construction. Hot path code (scenario generation)
    /// uses pre-validated instances, so this one-time cost is acceptable.
    ///
    /// # Example
    ///
    /// ```rust
    /// use powers_rs::seasonal_params::SeasonalParams;
    ///
    /// // Valid PAR(1) model
    /// let params = SeasonalParams::new(
    ///     12,
    ///     vec![1; 12],
    ///     vec![vec![0.7]; 12],
    ///     vec![100.0; 12],
    ///     vec![20.0; 12],
    /// );
    /// assert!(params.is_ok());
    ///
    /// // Invalid: coefficient too large
    /// let result = SeasonalParams::new(
    ///     12,
    ///     vec![1; 12],
    ///     vec![vec![1.5]; 12],  // |φ| >= 1
    ///     vec![100.0; 12],
    ///     vec![20.0; 12],
    /// );
    /// assert!(result.is_err());
    /// ```
    pub fn new(
        num_seasons: usize,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        means: Vec<f64>,
        stds: Vec<f64>,
    ) -> Result<Self, PowersError> {
        let params = Self {
            num_seasons,
            ar_orders,
            ar_coefficients,
            means,
            stds,
        };

        // Run all validation checks
        params.validate_lengths()?;
        params.validate_positivity()?;
        params.validate_coefficient_lengths()?;
        params.validate_stationarity()?;

        Ok(params)
    }

    /// Extract seasonal parameters from the first inflow hydro in unified_specs
    ///
    /// This helper constructs `SeasonalParams` from `UnifiedNoiseSpec` data,
    /// which is needed to initialize `UnifiedInflowModel` in the subproblem.
    ///
    /// **Assumptions**:
    /// - All inflow hydros share the same number of seasons and AR structure
    /// - If no inflow specs are found, returns default identity params
    ///
    /// # Arguments
    ///
    /// - `unified_specs`: Slice of unified noise specifications
    /// - `n_hydros`: Number of hydro plants expected
    ///
    /// # Returns
    ///
    /// `SeasonalParams` extracted from the first inflow hydro in the specs.
    /// If no inflow hydros are found, returns single-season AR(0) params
    /// with μ=0, σ=1 (identity transformation).
    ///
    /// # Performance
    ///
    /// - Time: O(n) scan through specs + O(s) for constructing seasonal arrays
    /// - Space: O(s·p) where s = seasons, p = max AR order
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let specs = vec![/* unified noise specs */];
    /// let params = SeasonalParams::from_unified_specs(&specs, 3)?;
    /// ```
    pub fn from_unified_specs(
        unified_specs: &[crate::unified_noise_spec::UnifiedNoiseSpec],
        _n_hydros: usize,
    ) -> Result<Self, PowersError> {
        use crate::input::UncertaintyType;
        use crate::unified_noise_spec::TemporalModelSpec;

        // Find first inflow spec
        let inflow_spec = unified_specs
            .iter()
            .find(|s| matches!(s.uncertainty_type, UncertaintyType::Inflow));

        match inflow_spec {
            Some(spec) => {
                // Extract temporal model info
                let (num_seasons, ar_params) = match &spec.temporal_model {
                    TemporalModelSpec::Independent => {
                        // Independent: single season, AR(0)
                        (1, None)
                    }
                    TemporalModelSpec::PeriodicAutoregressive {
                        num_seasons,
                        seasonal_ar_params,
                    } => (*num_seasons, Some(seasonal_ar_params)),
                };

                // Build seasonal parameter arrays
                let mut ar_orders = Vec::with_capacity(num_seasons);
                let mut ar_coefficients = Vec::with_capacity(num_seasons);
                let mut means = Vec::with_capacity(num_seasons);
                let mut stds = Vec::with_capacity(num_seasons);

                for season_id in 0..num_seasons {
                    // Extract mean and std from seasonal_params
                    let seasonal_params = spec
                        .seasonal_params
                        .get(&season_id)
                        .ok_or_else(|| {
                            PowersError::from(format!(
                                "Missing seasonal_params for season {} in spec for entity {}",
                                season_id, spec.entity_id
                            ))
                        })?;

                    means.push(seasonal_params.mean);
                    stds.push(seasonal_params.std_dev);

                    // Extract AR parameters if available
                    if let Some(ar_params_map) = &ar_params {
                        if let Some(ar_params) = ar_params_map.get(&season_id) {
                            ar_orders.push(ar_params.ar_order);
                            ar_coefficients
                                .push(ar_params.ar_coefficients.clone());
                        } else {
                            // Missing AR params for this season - default to AR(0)
                            ar_orders.push(0);
                            ar_coefficients.push(Vec::new());
                        }
                    } else {
                        // Independent case: AR(0)
                        ar_orders.push(0);
                        ar_coefficients.push(Vec::new());
                    }
                }

                // Construct and validate
                Self::new(num_seasons, ar_orders, ar_coefficients, means, stds)
            }
            None => {
                // No inflow specs - return identity transformation (AR(0))
                Ok(Self {
                    num_seasons: 1,
                    ar_orders: vec![0],
                    ar_coefficients: vec![Vec::new()],
                    means: vec![0.0],
                    stds: vec![1.0],
                })
            }
        }
    }

    /// Validate all arrays have length `num_seasons`
    ///
    /// Ensures consistency: all seasonal parameter arrays must have the same length
    /// as the specified num_seasons.
    fn validate_lengths(&self) -> Result<(), PowersError> {
        if self.ar_orders.len() != self.num_seasons {
            return Err(PowersError::from(format!(
                "PAR parameter validation failed: ar_orders length {} != num_seasons {}. \
                 All seasonal arrays must have length equal to num_seasons.",
                self.ar_orders.len(),
                self.num_seasons
            )));
        }
        if self.ar_coefficients.len() != self.num_seasons {
            return Err(PowersError::from(format!(
                "PAR parameter validation failed: ar_coefficients length {} != num_seasons {}. \
                 All seasonal arrays must have length equal to num_seasons.",
                self.ar_coefficients.len(),
                self.num_seasons
            )));
        }
        if self.means.len() != self.num_seasons {
            return Err(PowersError::from(format!(
                "PAR parameter validation failed: means length {} != num_seasons {}. \
                 All seasonal arrays must have length equal to num_seasons.",
                self.means.len(),
                self.num_seasons
            )));
        }
        if self.stds.len() != self.num_seasons {
            return Err(PowersError::from(format!(
                "PAR parameter validation failed: stds length {} != num_seasons {}. \
                 All seasonal arrays must have length equal to num_seasons.",
                self.stds.len(),
                self.num_seasons
            )));
        }
        Ok(())
    }

    /// Validate all standard deviations are positive
    ///
    /// Physical requirement: σₘ > 0 for all seasons (cannot have zero or negative variance).
    fn validate_positivity(&self) -> Result<(), PowersError> {
        for (m, &std) in self.stds.iter().enumerate() {
            if std <= 0.0 {
                return Err(PowersError::from(format!(
                    "PAR parameter validation failed: Standard deviation for season {} \
                     must be positive (> 0), got {}. This violates physical constraints.",
                    m, std
                )));
            }
        }
        Ok(())
    }

    /// Validate ar_coefficients inner length matches ar_orders
    ///
    /// For each season m, ar_coefficients[m] must have exactly ar_orders[m] elements.
    fn validate_coefficient_lengths(&self) -> Result<(), PowersError> {
        for m in 0..self.num_seasons {
            let expected = self.ar_orders[m];
            let actual = self.ar_coefficients[m].len();
            if actual != expected {
                return Err(PowersError::from(format!(
                    "PAR parameter validation failed: Season {} has AR order {} but got {} \
                     coefficients. Number of coefficients must match AR order.",
                    m, expected, actual
                )));
            }
        }
        Ok(())
    }

    /// Validate stationarity for all seasons
    ///
    /// # Stationarity Conditions
    ///
    /// For each season m, the AR polynomial must be stationary:
    ///
    /// - **AR(0)**: Always stationary (no AR component)
    /// - **AR(1)**: |φ₁| < 1
    /// - **AR(2)**: Three conditions must all hold:
    ///   1. |φ₂| < 1
    ///   2. φ₁ + φ₂ < 1
    ///   3. φ₂ - φ₁ < 1
    /// - **AR(p > 2)**: Heuristic check: sum(|φₖ|) < 1
    ///
    /// # Mathematical Background
    ///
    /// Stationarity requires all roots of the characteristic polynomial
    /// `1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0` to lie outside the unit circle.
    ///
    /// For AR(1) and AR(2), we use closed-form conditions. For AR(p > 2),
    /// we use a sufficient (but not necessary) condition: sum of absolute
    /// coefficients < 1. This is conservative but fast.
    ///
    /// # Performance
    ///
    /// O(num_seasons × max_ar_order) - linear in total coefficients.
    /// Acceptable for one-time validation during construction.
    ///
    fn validate_stationarity(&self) -> Result<(), PowersError> {
        for m in 0..self.num_seasons {
            let order = self.ar_orders[m];
            let coeffs = &self.ar_coefficients[m];

            if order == 0 {
                // No AR component - always stationary
                continue;
            }

            if order == 1 {
                // AR(1): |φ₁| < 1
                let phi = coeffs[0];
                if phi.abs() >= 1.0 {
                    return Err(PowersError::from(format!(
                        "PAR stationarity validation failed: Season {} AR(1) coefficient {} \
                         violates stationarity condition |φ| < 1. For stationary AR(1), the \
                         coefficient must be strictly less than 1 in absolute value.",
                        m, phi
                    )));
                }
            } else if order == 2 {
                // AR(2): three conditions
                let phi1 = coeffs[0];
                let phi2 = coeffs[1];

                // Condition 1: |φ₂| < 1
                if phi2.abs() >= 1.0 {
                    return Err(PowersError::from(format!(
                        "PAR stationarity validation failed: Season {} AR(2) violates |φ₂| < 1. \
                         Got φ₂ = {}. For stationary AR(2), |φ₂| must be strictly less than 1.",
                        m, phi2
                    )));
                }

                // Condition 2: φ₁ + φ₂ < 1
                let sum = phi1 + phi2;
                if sum >= 1.0 {
                    return Err(PowersError::from(format!(
                        "PAR stationarity validation failed: Season {} AR(2) violates φ₁ + φ₂ < 1. \
                         Got φ₁ = {}, φ₂ = {}, sum = {}. For stationary AR(2), the sum must be \
                         strictly less than 1.",
                        m, phi1, phi2, sum
                    )));
                }

                // Condition 3: φ₂ - φ₁ < 1
                let diff = phi2 - phi1;
                if diff >= 1.0 {
                    return Err(PowersError::from(format!(
                        "PAR stationarity validation failed: Season {} AR(2) violates φ₂ - φ₁ < 1. \
                         Got φ₁ = {}, φ₂ = {}, difference = {}. For stationary AR(2), the \
                         difference must be strictly less than 1.",
                        m, phi1, phi2, diff
                    )));
                }
            } else {
                // AR(p > 2): Use heuristic - sum of absolute coefficients < 1
                // This is a sufficient (but not necessary) condition for stationarity.
                // It's conservative but computationally cheap.
                let sum_abs: f64 = coeffs.iter().map(|c| c.abs()).sum();
                if sum_abs >= 1.0 {
                    return Err(PowersError::from(format!(
                        "PAR stationarity validation failed: Season {} AR({}) likely non-stationary. \
                         Sum of absolute coefficients = {} >= 1. For AR(p > 2), we use a heuristic \
                         sufficient condition: sum(|φₖ|) < 1. Consider reducing coefficient magnitudes.",
                        m, order, sum_abs
                    )));
                }
                // Future enhancement: For exact stationarity check, compute eigenvalues
                // of companion matrix. Current heuristic is conservative but adequate.
                // See FUTURE_WORK.md: "Eigenvalue-Based Stationarity Check"
            }
        }
        Ok(())
    }

    /// Get AR coefficients for a specific season
    ///
    /// Returns a slice of AR coefficients for the given season index.
    /// Automatically wraps around using modulo arithmetic.
    ///
    /// # Arguments
    ///
    /// - `season_index`: Season index (0-based, wraps around via modulo)
    ///
    /// # Returns
    ///
    /// Slice of AR coefficients [φ₁ₘ, φ₂ₘ, ..., φₚₘ] for season m = season_index % num_seasons
    ///
    /// # Performance
    ///
    /// O(1) - simple modulo and vec indexing
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// let params = SeasonalParams::new(
    ///     3,
    ///     vec![1, 2, 1],
    ///     vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
    ///     vec![100.0, 120.0, 150.0],
    ///     vec![20.0, 25.0, 30.0],
    /// ).unwrap();
    ///
    /// assert_eq!(params.get_ar_coeffs(1), &[0.5, 0.3]);
    /// assert_eq!(params.get_ar_coeffs(4), &[0.5, 0.3]); // 4 % 3 = 1
    /// ```
    #[inline]
    pub fn get_ar_coeffs(&self, season_index: usize) -> &[f64] {
        &self.ar_coefficients[season_index % self.num_seasons]
    }

    /// Get seasonal mean for a specific season
    ///
    /// Returns the seasonal mean μₘ for the given season index.
    /// Automatically wraps around using modulo arithmetic.
    ///
    /// # Performance
    ///
    /// O(1) - simple modulo and vec indexing
    #[inline]
    pub fn get_mean(&self, season_index: usize) -> f64 {
        self.means[season_index % self.num_seasons]
    }

    /// Get seasonal standard deviation for a specific season
    ///
    /// Returns the seasonal standard deviation σₘ for the given season index.
    /// Automatically wraps around using modulo arithmetic.
    ///
    /// # Performance
    ///
    /// O(1) - simple modulo and vec indexing
    #[inline]
    pub fn get_std(&self, season_index: usize) -> f64 {
        self.stds[season_index % self.num_seasons]
    }

    /// Get AR order for a specific season
    ///
    /// Returns the AR order pₘ for the given season index.
    /// Automatically wraps around using modulo arithmetic.
    ///
    /// # Performance
    ///
    /// O(1) - simple modulo and vec indexing
    #[inline]
    pub fn get_ar_order(&self, season_index: usize) -> usize {
        self.ar_orders[season_index % self.num_seasons]
    }
}

impl TryFrom<&TemporalModel> for SeasonalParams {
    type Error = PowersError;

    /// Convert PeriodicAutoregressive variant to SeasonalParams
    ///
    /// # Errors
    ///
    /// Returns `PowersError::InvalidInput` if:
    /// - The temporal model is not PeriodicAutoregressive
    /// - The parameters fail validation (lengths, positivity, stationarity)
    ///
    /// # Example
    ///
    /// ```rust
    /// use powers_rs::seasonal_params::SeasonalParams;
    /// use powers_rs::input::TemporalModel;
    ///
    /// let model = TemporalModel::PeriodicAutoregressive {
    ///     num_seasons: 12,
    ///     ar_orders: vec![1; 12],
    ///     ar_coefficients: vec![vec![0.7]; 12],
    ///     seasonal_means: vec![100.0; 12],
    ///     seasonal_stds: vec![20.0; 12],
    /// };
    ///
    /// let params = SeasonalParams::try_from(&model);
    /// assert!(params.is_ok());
    /// ```
    fn try_from(model: &TemporalModel) -> Result<Self, Self::Error> {
        match model {
            TemporalModel::PeriodicAutoregressive {
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => SeasonalParams::new(
                *num_seasons,
                ar_orders.clone(),
                ar_coefficients.clone(),
                seasonal_means.clone(),
                seasonal_stds.clone(),
            ),
            _ => Err(PowersError::from(
                "Cannot convert non-PeriodicAutoregressive temporal model to SeasonalParams. \
                 Only TemporalModel::PeriodicAutoregressive can be converted."
                    .to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_par1_all_periods() {
        let params = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![20.0; 12],
        );
        assert!(params.is_ok(), "Valid PAR(1) should construct");
    }

    #[test]
    fn test_invalid_par1_coefficient_too_large() {
        let result = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![1.2]; 12], // |φ| > 1
            vec![100.0; 12],
            vec![20.0; 12],
        );
        assert!(result.is_err(), "PAR(1) with |φ| > 1 should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("stationarity"),
            "Error should mention stationarity"
        );
        assert!(
            err_msg.contains("1.2"),
            "Error should show coefficient value"
        );
    }

    #[test]
    fn test_valid_par2() {
        let params = SeasonalParams::new(
            2,
            vec![2, 2],
            vec![vec![0.5, 0.3], vec![0.6, 0.2]],
            vec![100.0, 120.0],
            vec![20.0, 25.0],
        );
        assert!(params.is_ok(), "Valid PAR(2) should construct");
    }

    #[test]
    fn test_invalid_par2_sum_condition() {
        let result = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.7, 0.5]], // φ₁ + φ₂ = 1.2 > 1
            vec![100.0],
            vec![20.0],
        );
        assert!(
            result.is_err(),
            "PAR(2) violating sum condition should fail"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("φ₁ + φ₂"),
            "Error should mention sum condition"
        );
    }

    #[test]
    fn test_invalid_par2_diff_condition() {
        let result = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![-0.3, 0.8]], // φ₂ - φ₁ = 0.8 - (-0.3) = 1.1 > 1
            vec![100.0],
            vec![20.0],
        );
        assert!(
            result.is_err(),
            "PAR(2) violating diff condition should fail"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("φ₂ - φ₁"),
            "Error should mention difference condition"
        );
    }

    #[test]
    fn test_invalid_par2_phi2_condition() {
        let result = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.3, 1.1]], // |φ₂| >= 1
            vec![100.0],
            vec![20.0],
        );
        assert!(result.is_err(), "PAR(2) with |φ₂| >= 1 should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("|φ₂|"), "Error should mention |φ₂| < 1");
    }

    #[test]
    fn test_length_mismatch_ar_orders() {
        let result = SeasonalParams::new(
            12,
            vec![1; 10], // Wrong length!
            vec![vec![0.7]; 10],
            vec![100.0; 12],
            vec![20.0; 12],
        );
        assert!(result.is_err(), "Length mismatch should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("length"), "Error should mention length");
        assert!(
            err_msg.contains("ar_orders"),
            "Error should specify ar_orders"
        );
    }

    #[test]
    fn test_length_mismatch_means() {
        let result = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 10], // Wrong length!
            vec![20.0; 12],
        );
        assert!(result.is_err(), "Length mismatch should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("means"), "Error should specify means");
    }

    #[test]
    fn test_negative_std() {
        let result = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![-20.0; 12], // Negative!
        );
        assert!(result.is_err(), "Negative std should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("positive"),
            "Error should mention positive"
        );
        assert!(err_msg.contains("-20"), "Error should show value");
    }

    #[test]
    fn test_zero_std() {
        let result = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![0.0; 12], // Zero!
        );
        assert!(result.is_err(), "Zero std should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("positive"),
            "Error should mention positive"
        );
    }

    #[test]
    fn test_coefficient_count_mismatch() {
        let result = SeasonalParams::new(
            2,
            vec![2, 1],
            vec![vec![0.5], vec![0.6]], // First period should have 2 coeffs!
            vec![100.0, 120.0],
            vec![20.0, 25.0],
        );
        assert!(result.is_err(), "Coefficient count mismatch should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("coefficients"),
            "Error should mention coefficients"
        );
    }

    #[test]
    fn test_helper_methods() {
        let params = SeasonalParams::new(
            3,
            vec![1, 2, 1],
            vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
            vec![100.0, 120.0, 150.0],
            vec![20.0, 25.0, 30.0],
        )
        .expect("Valid params should construct");

        // Test direct access
        assert_eq!(params.get_ar_order(0), 1);
        assert_eq!(params.get_ar_order(1), 2);
        assert_eq!(params.get_ar_order(2), 1);
        assert_eq!(params.get_mean(1), 120.0);
        assert_eq!(params.get_std(2), 30.0);
        assert_eq!(params.get_ar_coeffs(1), &[0.5, 0.3]);

        // Test wraparound
        assert_eq!(params.get_ar_order(3), 1); // 3 % 3 = 0
        assert_eq!(params.get_mean(3), 100.0); // 3 % 3 = 0
        assert_eq!(params.get_std(4), 25.0); // 4 % 3 = 1
        assert_eq!(params.get_ar_coeffs(5), &[0.6]); // 5 % 3 = 2
    }

    #[test]
    fn test_helper_methods_wraparound() {
        let params = SeasonalParams::new(
            3,
            vec![1, 2, 1],
            vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
            vec![100.0, 120.0, 150.0],
            vec![20.0, 25.0, 30.0],
        )
        .unwrap();

        // Period 0
        assert_eq!(params.get_ar_coeffs(0), &[0.7]);
        // Period 1
        assert_eq!(params.get_ar_coeffs(1), &[0.5, 0.3]);
        // Period 2
        assert_eq!(params.get_ar_coeffs(2), &[0.6]);
        // Wraparound: period 3 = period 0
        assert_eq!(params.get_ar_coeffs(3), &[0.7]);
        // Wraparound: period 4 = period 1
        assert_eq!(params.get_ar_coeffs(4), &[0.5, 0.3]);
    }

    #[test]
    fn test_try_from_periodic_ar() {
        let model = TemporalModel::PeriodicAutoregressive {
            num_seasons: 12,
            ar_orders: vec![1; 12],
            ar_coefficients: vec![vec![0.7]; 12],
            seasonal_means: vec![100.0; 12],
            seasonal_stds: vec![20.0; 12],
        };

        let params = SeasonalParams::try_from(&model);
        assert!(params.is_ok(), "Conversion from PAR should succeed");

        let params = params.unwrap();
        assert_eq!(params.num_seasons, 12);
        assert_eq!(params.get_mean(0), 100.0);
        assert_eq!(params.get_ar_coeffs(0), &[0.7]);
    }

    #[test]
    fn test_try_from_independent_fails() {
        let model = TemporalModel::Independent;
        let result = SeasonalParams::try_from(&model);
        assert!(result.is_err(), "Conversion from Independent should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-PeriodicAutoregressive"),
            "Error should mention type mismatch"
        );
    }

    #[test]
    fn test_valid_par_with_varying_orders() {
        // Quarterly PAR with varying orders: AR(1), AR(2), AR(1), AR(0)
        let params = SeasonalParams::new(
            4,
            vec![1, 2, 1, 0], // Last quarter has no AR component
            vec![
                vec![0.7],
                vec![0.5, 0.3],
                vec![0.6],
                vec![], // Empty for AR(0)
            ],
            vec![100.0, 150.0, 200.0, 120.0],
            vec![20.0, 30.0, 40.0, 25.0],
        );
        assert!(
            params.is_ok(),
            "PAR with varying orders including AR(0) should be valid"
        );
    }

    #[test]
    fn test_ar_p_heuristic_check() {
        // AR(3) with sum of absolute coefficients < 1 should pass
        let params = SeasonalParams::new(
            1,
            vec![3],
            vec![vec![0.3, 0.2, 0.1]], // sum = 0.6 < 1
            vec![100.0],
            vec![20.0],
        );
        assert!(params.is_ok(), "AR(3) with sum|φ| < 1 should pass");
    }

    #[test]
    fn test_ar_p_heuristic_check_fails() {
        // AR(3) with sum of absolute coefficients >= 1 should fail
        let result = SeasonalParams::new(
            1,
            vec![3],
            vec![vec![0.5, 0.4, 0.2]], // sum = 1.1 >= 1
            vec![100.0],
            vec![20.0],
        );
        assert!(result.is_err(), "AR(3) with sum|φ| >= 1 should fail");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("likely non-stationary"),
            "Error should mention non-stationarity"
        );
        assert!(err_msg.contains("AR(3)"), "Error should mention AR(3)");
    }
}
