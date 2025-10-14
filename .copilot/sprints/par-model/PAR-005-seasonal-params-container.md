# PAR-005: Implement SeasonalParams Container and Validation

## Context

The PAR generator needs a convenient container for seasonal parameters (μₘ, σₘ, φₖₘ) extracted from the `PeriodicAutoregressive` variant. This ticket creates the `SeasonalParams` type and implements runtime validation for:

1. Array length consistency (all must equal period)
2. Positivity constraints (σₘ > 0)
3. Stationarity conditions (spectral radius < 1)

This type serves as an intermediate representation between raw JSON input and the PAR generator.

## Acceptance Criteria

- [ ] `SeasonalParams` struct implemented with all required fields
- [ ] Constructor validates all length and positivity constraints
- [ ] Stationarity validation for AR(1), AR(2), and AR(p) cases
- [ ] Clear error messages for each validation failure type
- [ ] Helper methods for accessing parameters by period index
- [ ] Comprehensive unit tests covering all validation paths
- [ ] Documentation with CEPEL references and stationarity formulas

## Tasks

### Implementation

- [ ] Create `src/seasonal_params.rs` module
- [ ] Implement `SeasonalParams` struct:
  - Fields: period, ar_orders, ar_coefficients, means, stds
  - Constructor: `new()` with validation
  - Getters: `get_ar_coeffs(month)`, `get_mean(month)`, etc.
- [ ] Implement validation methods:
  - `validate_lengths()` - all arrays match period
  - `validate_positivity()` - all σₘ > 0
  - `validate_stationarity()` - AR spectral radius < 1
- [ ] Add conversion: `From<TemporalModel::PeriodicAutoregressive>` trait
- [ ] Add module to `lib.rs`: `pub mod seasonal_params;`

### Testing

- [ ] Unit test: Valid PAR(1) with all φ in (-1, 1)
- [ ] Unit test: Invalid PAR(1) with |φ| > 1 fails
- [ ] Unit test: Valid PAR(2) satisfying all three conditions
- [ ] Unit test: Invalid PAR(2) violating sum condition fails
- [ ] Unit test: Length mismatch (ar_orders.len() != period) fails
- [ ] Unit test: Negative standard deviation fails
- [ ] Unit test: ar_coefficients inner length != ar_orders[m] fails
- [ ] Unit test: Conversion from TemporalModel variant
- [ ] Unit test: Helper methods return correct values

### Documentation

- [ ] Module-level doc explaining CEPEL seasonal parameter semantics
- [ ] Struct doc with field descriptions and mathematical notation
- [ ] Stationarity validation docs with formulas (AR(1), AR(2), AR(p))
- [ ] Example usage showing construction and access patterns
- [ ] Error message documentation

## Technical Notes

### Implementation

````rust
// src/seasonal_params.rs

use crate::error::PowersError;
use crate::input::TemporalModel;

/// Container for periodic AR seasonal parameters (CEPEL methodology)
///
/// # Mathematical Notation
///
/// - μₘ: seasonal mean for period m
/// - σₘ: seasonal standard deviation for period m
/// - φₖₘ: AR coefficient k for period m (k = 1..pₘ)
/// - pₘ: AR order for period m
///
/// # CEPEL PAR(p) Equation
///
/// ```text
/// Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
///
/// where:
///   m = t mod period (seasonal index, maps to season_id in graph nodes)
///   aₜ ~ residual_distribution (e.g., LogNormal3)
/// ```
///
/// # Stationarity Requirement
///
/// For each period m, the AR polynomial must be stationary:
/// - AR(1): |φ₁ₘ| < 1
/// - AR(2): |φ₂ₘ| < 1, φ₁ₘ + φ₂ₘ < 1, φ₂ₘ - φ₁ₘ < 1
/// - AR(p): spectral radius of companion matrix < 1
///
#[derive(Debug, Clone)]
pub struct SeasonalParams {
    /// Number of periods in seasonal cycle (e.g., 12 for monthly, 4 for quarterly)
    /// Must match the period in PeriodicAutoregressive variant
    pub period: usize,

    /// AR order for each period [p₀, p₁, ..., p_{period-1}]
    pub ar_orders: Vec<usize>,

    /// AR coefficients for each period
    /// Outer vec length = period, inner vec[m] length = ar_orders[m]
    pub ar_coefficients: Vec<Vec<f64>>,

    /// Seasonal means [μ₀, μ₁, ..., μ_{period-1}]
    pub means: Vec<f64>,

    /// Seasonal standard deviations [σ₀, σ₁, ..., σ_{period-1}]
    pub stds: Vec<f64>,
}

impl SeasonalParams {
    /// Construct and validate seasonal parameters
    ///
    /// # Validation
    ///
    /// 1. All arrays must have length `period`
    /// 2. All σₘ must be positive
    /// 3. ar_coefficients[m].len() must equal ar_orders[m]
    /// 4. Each AR polynomial must be stationary
    ///
    /// # Errors
    ///
    /// Returns `PowersError::InvalidInput` if any validation fails.
    pub fn new(
        period: usize,
        ar_orders: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        means: Vec<f64>,
        stds: Vec<f64>,
    ) -> Result<Self, PowersError> {
        let params = Self {
            period,
            ar_orders,
            ar_coefficients,
            means,
            stds,
        };

        params.validate_lengths()?;
        params.validate_positivity()?;
        params.validate_coefficient_lengths()?;
        params.validate_stationarity()?;

        Ok(params)
    }

    /// Validate all arrays have length `period`
    fn validate_lengths(&self) -> Result<(), PowersError> {
        if self.ar_orders.len() != self.period {
            return Err(PowersError::InvalidInput(format!(
                "ar_orders length {} != period {}",
                self.ar_orders.len(),
                self.period
            )));
        }
        if self.ar_coefficients.len() != self.period {
            return Err(PowersError::InvalidInput(format!(
                "ar_coefficients length {} != period {}",
                self.ar_coefficients.len(),
                self.period
            )));
        }
        if self.means.len() != self.period {
            return Err(PowersError::InvalidInput(format!(
                "means length {} != period {}",
                self.means.len(),
                self.period
            )));
        }
        if self.stds.len() != self.period {
            return Err(PowersError::InvalidInput(format!(
                "stds length {} != period {}",
                self.stds.len(),
                self.period
            )));
        }
        Ok(())
    }

    /// Validate all standard deviations are positive
    fn validate_positivity(&self) -> Result<(), PowersError> {
        for (m, &std) in self.stds.iter().enumerate() {
            if std <= 0.0 {
                return Err(PowersError::InvalidInput(format!(
                    "Standard deviation for period {} must be positive, got {}",
                    m, std
                )));
            }
        }
        Ok(())
    }

    /// Validate ar_coefficients inner length matches ar_orders
    fn validate_coefficient_lengths(&self) -> Result<(), PowersError> {
        for m in 0..self.period {
            let expected = self.ar_orders[m];
            let actual = self.ar_coefficients[m].len();
            if actual != expected {
                return Err(PowersError::InvalidInput(format!(
                    "Period {} AR order is {} but got {} coefficients",
                    m, expected, actual
                )));
            }
        }
        Ok(())
    }

    /// Validate stationarity for all periods
    ///
    /// # Stationarity Conditions
    ///
    /// - **AR(1)**: |φ₁| < 1
    /// - **AR(2)**: |φ₂| < 1, φ₁ + φ₂ < 1, φ₂ - φ₁ < 1
    /// - **AR(p)**: All roots of 1 - φ₁z - φ₂z² - ... - φₚzᵖ lie outside unit circle
    ///
    /// For AR(p>2), we use the companion matrix method (eigenvalue check).
    fn validate_stationarity(&self) -> Result<(), PowersError> {
        for m in 0..self.period {
            let order = self.ar_orders[m];
            let coeffs = &self.ar_coefficients[m];

            if order == 0 {
                continue; // No AR component, automatically stationary
            }

            if order == 1 {
                // AR(1): |φ₁| < 1
                let phi = coeffs[0];
                if phi.abs() >= 1.0 {
                    return Err(PowersError::InvalidInput(format!(
                        "Period {} AR(1) coefficient {} violates stationarity (|φ| < 1)",
                        m, phi
                    )));
                }
            } else if order == 2 {
                // AR(2): three conditions
                let phi1 = coeffs[0];
                let phi2 = coeffs[1];

                if phi2.abs() >= 1.0 {
                    return Err(PowersError::InvalidInput(format!(
                        "Period {} AR(2) violates |φ₂| < 1: φ₂ = {}",
                        m, phi2
                    )));
                }
                if phi1 + phi2 >= 1.0 {
                    return Err(PowersError::InvalidInput(format!(
                        "Period {} AR(2) violates φ₁ + φ₂ < 1: {} + {} = {}",
                        m, phi1, phi2, phi1 + phi2
                    )));
                }
                if phi2 - phi1 >= 1.0 {
                    return Err(PowersError::InvalidInput(format!(
                        "Period {} AR(2) violates φ₂ - φ₁ < 1: {} - {} = {}",
                        m, phi2, phi1, phi2 - phi1
                    )));
                }
            } else {
                // AR(p): Companion matrix spectral radius < 1
                // For production, use nalgebra eigenvalue check
                // For MVP, skip or use heuristic (sum of abs coefficients < 1)
                let sum_abs: f64 = coeffs.iter().map(|c| c.abs()).sum();
                if sum_abs >= 1.0 {
                    return Err(PowersError::InvalidInput(format!(
                        "Period {} AR({}) likely non-stationary: sum|φₖ| = {} >= 1",
                        m, order, sum_abs
                    )));
                }
                // TODO: Implement full spectral radius check using nalgebra
            }
        }
        Ok(())
    }

    /// Get AR coefficients for a specific period
    pub fn get_ar_coeffs(&self, period_index: usize) -> &[f64] {
        &self.ar_coefficients[period_index % self.period]
    }

    /// Get seasonal mean for a specific period
    pub fn get_mean(&self, period_index: usize) -> f64 {
        self.means[period_index % self.period]
    }

    /// Get seasonal standard deviation for a specific period
    pub fn get_std(&self, period_index: usize) -> f64 {
        self.stds[period_index % self.period]
    }

    /// Get AR order for a specific period
    pub fn get_ar_order(&self, period_index: usize) -> usize {
        self.ar_orders[period_index % self.period]
    }
}

impl TryFrom<&TemporalModel> for SeasonalParams {
    type Error = PowersError;

    fn try_from(model: &TemporalModel) -> Result<Self, Self::Error> {
        match model {
            TemporalModel::PeriodicAutoregressive {
                period,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => SeasonalParams::new(
                *period,
                ar_orders.clone(),
                ar_coefficients.clone(),
                seasonal_means.clone(),
                seasonal_stds.clone(),
            ),
            _ => Err(PowersError::InvalidInput(
                "Cannot convert non-periodic temporal model to SeasonalParams".to_string()
            )),
        }
    }
}
````

### Test Coverage

```rust
// tests/test_seasonal_params.rs

use powers::seasonal_params::SeasonalParams;
use powers::error::PowersError;

#[test]
fn test_valid_par1_all_periods() {
    let params = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    );
    assert!(params.is_ok());
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
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("stationarity"));
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
    assert!(params.is_ok());
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
    assert!(result.is_err());
}

#[test]
fn test_length_mismatch() {
    let result = SeasonalParams::new(
        12,
        vec![1; 10], // Wrong length!
        vec![vec![0.7]; 10],
        vec![100.0; 12],
        vec![20.0; 12],
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("length"));
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
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("positive"));
}

#[test]
fn test_coefficient_count_mismatch() {
    let result = SeasonalParams::new(
        2,
        vec![2, 1],
        vec![vec![0.5], vec![0.6]], // First should have 2 coeffs!
        vec![100.0, 120.0],
        vec![20.0, 25.0],
    );
    assert!(result.is_err());
}

#[test]
fn test_helper_methods() {
    let params = SeasonalParams::new(
        3,
        vec![1, 2, 1],
        vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
        vec![100.0, 120.0, 150.0],
        vec![20.0, 25.0, 30.0],
    ).unwrap();

    assert_eq!(params.get_ar_order(0), 1);
    assert_eq!(params.get_ar_order(1), 2);
    assert_eq!(params.get_mean(1), 120.0);
    assert_eq!(params.get_std(2), 30.0);
    assert_eq!(params.get_ar_coeffs(1), &[0.5, 0.3]);

    // Test wraparound
    assert_eq!(params.get_mean(3), 100.0); // 3 % 3 = 0
}
```

## Dependencies

- **Blocked by**: PAR-001 (TemporalModel enum)
- **Blocks**: PAR-006 (PAR generator needs this type)
- **Related**: PAR-002 (similar validation logic in PeriodicARParams)

## Estimated Effort

**2 story points** (confidence: high)

- 2 hours implementation (struct, validation, helpers)
- 2 hours testing (9 unit tests)
- 1 hour documentation
- 1 hour review

### Breakdown

- Implement struct and constructor: 1 hour
- Implement validation methods: 1.5 hours
- Implement helper methods and trait: 0.5 hours
- Write 9 unit tests: 2 hours
- Documentation with formulas: 1 hour
- Code review: 1 hour
