# PAR-002: Add SeasonalStats and Periodic Parameter Types

## Context

CEPEL's PAR(p) model requires careful management of seasonal statistics (μₘ, σₘ, γₘ per month) and AR coefficients that vary by season. We need clean, type-safe containers for these parameters that will be used throughout the PAR implementation.

This ticket creates the supporting types that will be used by the PAR generator (PAR-006) and parameter estimation tool (PAR-013).

## Acceptance Criteria

- [ ] `SeasonalStats` struct contains all per-season statistical parameters
- [ ] `PeriodicARParams` struct aggregates seasonal stats with AR coefficients
- [ ] Types include validation methods for stationarity and consistency
- [ ] Serde support for JSON serialization/deserialization
- [ ] Clear documentation with CEPEL notation and references
- [ ] Helper methods for accessing parameters by month/season index

## Tasks

### Implementation

- [ ] Create `SeasonalStats` struct in `src/input.rs` or new `src/periodic_ar.rs`
  - Fields: `month`, `mean`, `std_dev`, `skewness`, `ar_order`
- [ ] Create `PeriodicARParams` struct
  - Fields: `period`, `seasonal_stats: Vec<SeasonalStats>`, `ar_coefficients: Vec<Vec<f64>>`
- [ ] Implement `From<TemporalModel::PeriodicAutoregressive>` for `PeriodicARParams`
- [ ] Add helper methods:
  - `get_params_for_month(month: usize) -> &SeasonalStats`
  - `get_ar_coeffs_for_month(month: usize) -> &[f64]`
  - `validate_consistency() -> Result<(), String>` (length checks)
  - `validate_stationarity() -> Result<(), String>` (AR stability)
- [ ] Implement `Debug`, `Clone`, `Serialize`, `Deserialize` for both types

### Testing

- [ ] Unit test: Create SeasonalStats and verify all fields accessible
- [ ] Unit test: Create PeriodicARParams from TemporalModel variant
- [ ] Unit test: `get_params_for_month` returns correct stats for each month
- [ ] Unit test: `validate_consistency` catches length mismatches
- [ ] Unit test: `validate_stationarity` detects unstable AR(1) coefficient (|φ| > 1)
- [ ] Unit test: `validate_stationarity` accepts stable AR(2) coefficients
- [ ] Unit test: Serialize/deserialize round-trip preserves all data

### Documentation

- [ ] Add module-level doc comment explaining PAR parameter structure
- [ ] Document each field with CEPEL notation (μₘ, σₘ, γₘ, φₖₘ)
- [ ] Add examples showing typical monthly parameter structure
- [ ] Reference CEPEL equation (12) from documentation
- [ ] Add inline comments for stationarity validation logic

## Technical Notes

### Implementation Approach

```rust
// In src/periodic_ar.rs (new module) or src/input.rs

/// Container for seasonal statistics at a single period
///
/// Used for PAR(p) parameter estimation and validation. Each season/period
/// (identified by `season_id` in graph nodes) has its own statistics.
///
/// # Fields
///
/// - `period_index`: Index in seasonal cycle (0..period-1, e.g., 0..11 for monthly)
/// - `mean`: Mean value for this period (μₘ)
/// - `std_dev`: Standard deviation for this period (σₘ, must be > 0)
/// - `skewness`: Optional skewness parameter (for distribution fitting)
/// - `ar_order`: AR order for this period (pₘ, can vary by period!)
///
#[derive(Debug, Clone, PartialEq)]
pub struct SeasonalStats {
    /// Period index in cycle (0..period-1)
    pub period_index: usize,

/// Complete parameter set for Periodic AR(p) model
///
/// Aggregates seasonal statistics with AR coefficients for CEPEL PAR(p).
/// Provides validation and convenient access methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicARParams {
    /// Seasonal cycle length (12 for monthly, 4 for quarterly)
    pub period: usize,

    /// Statistical parameters per season
    pub seasonal_stats: Vec<SeasonalStats>,

    /// AR coefficients per season: ar_coefficients[m] has length ar_order[m]
    pub ar_coefficients: Vec<Vec<f64>>,
}

impl PeriodicARParams {
    /// Get seasonal statistics for a specific month
    ///
    /// # Arguments
    /// * `month` - Month index (0-based, wraps around if >= period)
    pub fn get_params_for_month(&self, month: usize) -> &SeasonalStats {
        &self.seasonal_stats[month % self.period]
    }

    /// Get AR coefficients for a specific month
    pub fn get_ar_coeffs_for_month(&self, month: usize) -> &[f64] {
        &self.ar_coefficients[month % self.period]
    }

    /// Validate parameter consistency
    ///
    /// Checks:
    /// - All vectors have length == period
    /// - ar_coefficients[m].len() == seasonal_stats[m].ar_order
    /// - std_dev > 0 for all months
    pub fn validate_consistency(&self) -> Result<(), String> {
        if self.seasonal_stats.len() != self.period {
            return Err(format!(
                "seasonal_stats length {} != period {}",
                self.seasonal_stats.len(), self.period
            ));
        }

        if self.ar_coefficients.len() != self.period {
            return Err(format!(
                "ar_coefficients length {} != period {}",
                self.ar_coefficients.len(), self.period
            ));
        }

        for (m, stats) in self.seasonal_stats.iter().enumerate() {
            if stats.std_dev <= 0.0 {
                return Err(format!(
                    "Month {} std_dev {} must be > 0",
                    m, stats.std_dev
                ));
            }

            if self.ar_coefficients[m].len() != stats.ar_order {
                return Err(format!(
                    "Month {} ar_coefficients length {} != ar_order {}",
                    m, self.ar_coefficients[m].len(), stats.ar_order
                ));
            }
        }

        Ok(())
    }

    /// Validate AR coefficient stationarity
    ///
    /// Checks stability conditions:
    /// - AR(1): |φ₁| < 1
    /// - AR(2): φ₁+φ₂ < 1, φ₂-φ₁ < 1, |φ₂| < 1
    /// - AR(p): Spectral radius < 1 (simplified check: sum of |φᵢ| < 1)
    pub fn validate_stationarity(&self) -> Result<(), String> {
        for (m, coeffs) in self.ar_coefficients.iter().enumerate() {
            let order = coeffs.len();

            match order {
                0 => continue, // White noise, always stationary
                1 => {
                    if coeffs[0].abs() >= 1.0 {
                        return Err(format!(
                            "Month {} AR(1): |φ₁| = {} >= 1 (unstable)",
                            m, coeffs[0].abs()
                        ));
                    }
                }
                2 => {
                    let phi1 = coeffs[0];
                    let phi2 = coeffs[1];
                    if phi1 + phi2 >= 1.0 {
                        return Err(format!(
                            "Month {} AR(2): φ₁+φ₂ = {} >= 1 (unstable)",
                            m, phi1 + phi2
                        ));
                    }
                    if phi2 - phi1 >= 1.0 {
                        return Err(format!(
                            "Month {} AR(2): φ₂-φ₁ = {} >= 1 (unstable)",
                            m, phi2 - phi1
                        ));
                    }
                    if phi2.abs() >= 1.0 {
                        return Err(format!(
                            "Month {} AR(2): |φ₂| = {} >= 1 (unstable)",
                            m, phi2.abs()
                        ));
                    }
                }
                _ => {
                    // Simplified check for AR(p): sum of absolute coefficients < 1
                    // Note: This is sufficient but not necessary for stationarity
                    let sum_abs: f64 = coeffs.iter().map(|c| c.abs()).sum();
                    if sum_abs >= 1.0 {
                        return Err(format!(
                            "Month {} AR({}): Σ|φᵢ| = {} >= 1 (likely unstable)",
                            m, order, sum_abs
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

impl From<&TemporalModel> for Option<PeriodicARParams> {
    fn from(temporal: &TemporalModel) -> Self {
        match temporal {
            TemporalModel::PeriodicAutoregressive {
                period,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            } => {
                let seasonal_stats: Vec<SeasonalStats> = (0..*period)
                    .map(|m| SeasonalStats {
                        month: m,
                        mean: seasonal_means[m],
                        std_dev: seasonal_stds[m],
                        skewness: 0.0, // Will be estimated from data in PAR-013
                        ar_order: ar_orders[m],
                    })
                    .collect();

                Some(PeriodicARParams {
                    period: *period,
                    seasonal_stats,
                    ar_coefficients: ar_coefficients.clone(),
                })
            }
            _ => None,
        }
    }
}
```

### Stationarity Validation References

- **AR(1)**: |φ₁| < 1 (simple condition)
- **AR(2)**: Three conditions (Brockwell & Davis, "Time Series: Theory and Methods")
- **AR(p)**: Roots of characteristic polynomial outside unit circle (complex to check)
  - Simplified: Σ|φᵢ| < 1 is sufficient but not necessary
  - CEPEL uses Yule-Walker, which guarantees stationarity if historical data is stationary

### Edge Cases

- AR(0): White noise, always stationary
- Very small coefficients (< 1e-10): Treat as effective white noise
- Monthly skewness: Set to 0.0 initially, estimate in PAR-013 from residuals

## Dependencies

- **Blocked by**: PAR-001 (needs PeriodicAutoregressive variant)
- **Blocks**: PAR-006 (generator needs these types), PAR-013 (estimation tool)
- **Related**: PAR-005 (validation logic will use these types)

## Estimated Effort

**2 story points** (confidence: high)

- 3 hours implementation (types, validation, conversion)
- 2 hours testing (7 unit tests)
- 1 hour documentation
- 1 hour review

### Breakdown

- Define structs and basic methods: 1.5 hours
- Implement validation logic: 1.5 hours
- Unit tests: 2 hours
- Documentation and examples: 1 hour
- Code review and polish: 1 hour
