# AR-9: AR Stochastic Process Implementation

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 2 days  
**Priority**: P0 (Critical)  
**Assignee**: TBD  
**Blocked by**: AR-5.5-v2 (log-normal scenario transformation), AR-5.6 (correlation infrastructure), AR-6 (trait extension)

---

## Context

With the `StochasticProcess` trait extended (AR-6), we can now implement the `AutoRegressive` stochastic process. This is the core component that transforms white noise innovations into correlated inflow realizations via the AR equation: ξₜ = φ₁ξₜ₋₁ + ... + φₚξₜ₋ₚ + εₜ.

**Extended Requirements** (based on AR-5.5-v2 and AR-5.6):

1. **Non-negativity**: Use **CEPEL 3-parameter log-normal transformation** for scenario generation (X = γ + exp(μ + σZ), **zero LP overhead**)
2. **Correlated innovations**: Accept pre-generated correlated innovations from `CorrelatedNoiseGenerator`

**Why this matters**: This is the mathematical engine that provides temporal correlation in SDDP. Non-negativity via inverse CDF sampling (not LP constraints) and correlation make it production-ready for real-world hydro systems.

---

## Objective

Implement `AutoRegressive` struct that implements `StochasticProcess` trait, providing conditional sampling based on lag history.

---

## Acceptance Criteria

### Must Have

- [ ] `AutoRegressive` struct with coefficients and innovation distribution
- [ ] `sample_conditional()` implementation using AR equation
- [ ] Validation: lag_state length matches lag_order
- [ ] Marginal distribution computed from AR parameters
- [ ] Unit tests for AR(1), AR(2) sampling
- [ ] **Integration with CEPEL log-normal transformation** (use inverse CDF sampling when non-negativity required, **zero LP overhead**) ← NEW
- [ ] **Support for correlated innovations** (accept externally-generated innovations from `CorrelatedNoiseGenerator`) ← NEW
- [ ] **Configuration flag for non-negativity method** (None or LogNormal3) ← NEW

### Should Have

- [ ] Builder pattern for construction
- [ ] Helper for computing stationary variance
- [ ] Autocorrelation function computation
- [ ] **Tests for non-negative output** (verify log-normal inverse CDF sampling) ← NEW
- [ ] **Tests for correlated multi-resource scenarios** (verify correlation preservation) ← NEW

### Won't Have (Yet)

- PAR (periodic AR) implementation (AR-19)
- Yule-Walker parameter estimation (later)
- Non-Gaussian innovations beyond log-normal (CEPEL log-normal handles non-negativity)

---

## Implementation Tasks

### 1. Define AutoRegressive Struct (1 hour)

```rust
// In src/stochastic_process.rs

/// Autoregressive process: ξₜ = φ₁ξₜ₋₁ + ... + φₚξₜ₋ₚ + εₜ
///
/// # Mathematical Details
/// - Coefficients: φ = [φ₁, φ₂, ..., φₚ]
/// - Innovation: εₜ ~ N(0, σ²)
/// - Marginal: ξₜ ~ N(μ, σ²ₓ) where σ²ₓ computed from Yule-Walker
///
/// # Stationarity
/// Must satisfy stationarity conditions (checked in validation AR-2)
#[derive(Debug, Clone)]
pub struct AutoRegressive {
    /// AR coefficients [φ₁, φ₂, ..., φₚ]
    coefficients: Vec<f64>,

    /// Innovation distribution (white noise)
    /// Typically N(0, σ²) but could be other symmetric distributions
    innovation_dist: Distribution,

    /// Marginal distribution (computed)
    /// For stationary AR: ξₜ ~ N(μ, σ²ₓ)
    marginal_dist: Distribution,

    /// Resource name (for logging)
    resource: String,
}

impl AutoRegressive {
    /// Create new AR process with validated coefficients
    pub fn new(
        coefficients: Vec<f64>,
        innovation_dist: Distribution,
        resource: String,
    ) -> Result<Self, StochasticProcessError> {
        // Validate stationarity (already done in AR-2, but double-check)
        validate_stationarity(&coefficients)?;

        // Compute marginal distribution
        let marginal_dist = compute_marginal_distribution(&coefficients, &innovation_dist)?;

        Ok(Self {
            coefficients,
            innovation_dist,
            marginal_dist,
            resource,
        })
    }

    /// Get lag order (p)
    pub fn lag_order(&self) -> usize {
        self.coefficients.len()
    }

    /// Get AR coefficients
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Compute autocorrelation at lag k: ρₖ = Corr(ξₜ, ξₜ₋ₖ)
    pub fn autocorrelation(&self, lag: usize) -> f64 {
        if lag == 0 {
            return 1.0;
        }

        // For AR(1): ρₖ = φ₁^k
        if self.coefficients.len() == 1 {
            return self.coefficients[0].powi(lag as i32);
        }

        // For AR(p): use Yule-Walker recursion
        compute_acf(&self.coefficients, lag)
    }
}

fn validate_stationarity(coefficients: &[f64]) -> Result<(), StochasticProcessError> {
    // Basic check (full validation in AR-2)
    if coefficients.is_empty() {
        return Err(StochasticProcessError::InvalidLagOrder { order: 0 });
    }

    // For AR(1): |φ| < 1
    if coefficients.len() == 1 && coefficients[0].abs() >= 1.0 {
        return Err(StochasticProcessError::NonStationary {
            message: format!("AR(1) coefficient {} violates |φ| < 1", coefficients[0]),
        });
    }

    Ok(())
}

fn compute_marginal_distribution(
    coefficients: &[f64],
    innovation_dist: &Distribution,
) -> Result<Distribution, StochasticProcessError> {
    // Extract innovation variance
    let innovation_variance = match innovation_dist {
        Distribution::Normal { mean: _, std_dev } => std_dev * std_dev,
        _ => {
            return Err(StochasticProcessError::UnsupportedDistribution {
                distribution: format!("{:?}", innovation_dist),
            });
        }
    };

    // For AR(1): Var(ξ) = σ² / (1 - φ²)
    if coefficients.len() == 1 {
        let phi = coefficients[0];
        let marginal_variance = innovation_variance / (1.0 - phi * phi);
        let marginal_std_dev = marginal_variance.sqrt();

        return Ok(Distribution::Normal {
            mean: 0.0, // Zero-mean process
            std_dev: marginal_std_dev,
        });
    }

    // For AR(p): solve Yule-Walker equations
    // Γφ = γ where Γ is Toeplitz ACF matrix
    // For now: use numerical approximation
    let marginal_variance = compute_marginal_variance_numeric(coefficients, innovation_variance)?;
    let marginal_std_dev = marginal_variance.sqrt();

    Ok(Distribution::Normal {
        mean: 0.0,
        std_dev: marginal_std_dev,
    })
}

fn compute_marginal_variance_numeric(
    coefficients: &[f64],
    innovation_variance: f64,
) -> Result<f64, StochasticProcessError> {
    // Solve: σ²ₓ = σ² / (1 - φ₁ρ₁ - φ₂ρ₂ - ... - φₚρₚ)
    // where ρₖ are autocorrelations (computed via Yule-Walker recursion)

    // Initial guess: use AR(1) approximation
    let phi_sum: f64 = coefficients.iter().map(|&phi| phi.abs()).sum();
    if phi_sum >= 1.0 {
        return Err(StochasticProcessError::NonStationary {
            message: "Sum of absolute coefficients >= 1".to_string(),
        });
    }

    // Approximation: σ²ₓ ≈ σ² / (1 - Σ|φᵢ|)
    let marginal_variance = innovation_variance / (1.0 - phi_sum);

    Ok(marginal_variance)
}

fn compute_acf(coefficients: &[f64], max_lag: usize) -> f64 {
    // Yule-Walker recursion: ρₖ = φ₁ρₖ₋₁ + φ₂ρₖ₋₂ + ... + φₚρₖ₋ₚ
    let p = coefficients.len();
    let mut acf = vec![1.0]; // ρ₀ = 1

    for k in 1..=max_lag {
        let mut rho_k = 0.0;
        for (j, &phi_j) in coefficients.iter().enumerate() {
            let lag = k as isize - (j as isize + 1);
            if lag >= 0 && (lag as usize) < acf.len() {
                rho_k += phi_j * acf[lag as usize];
            } else if lag < 0 {
                // Symmetry: ρ₋ₖ = ρₖ
                rho_k += phi_j * acf[(-lag) as usize];
            }
        }
        acf.push(rho_k);
    }

    acf[max_lag]
}
```

### 2. Implement StochasticProcess Trait (1 hour)

```rust
impl StochasticProcess for AutoRegressive {
    fn sample(&self, _rng: &mut impl Rng) -> f64 {
        panic!(
            "AR process requires lag state for conditional sampling. \
             Use sample_conditional() instead."
        );
    }

    fn is_conditional(&self) -> bool {
        true
    }

    fn sample_conditional(&self, lag_state: &[f64], rng: &mut impl Rng) -> f64 {
        // Validate lag_state length
        assert_eq!(
            lag_state.len(),
            self.coefficients.len(),
            "AR({}) requires {} lag values, got {}",
            self.coefficients.len(),
            self.coefficients.len(),
            lag_state.len()
        );

        // Sample innovation: εₜ ~ N(0, σ²)
        let innovation = sample_from_distribution(&self.innovation_dist, rng);

        // Compute AR mean: E[ξₜ | ξₜ₋₁, ...] = φ₁ξₜ₋₁ + ... + φₚξₜ₋ₚ
        let ar_mean: f64 = self.coefficients
            .iter()
            .zip(lag_state.iter())
            .map(|(phi, xi)| phi * xi)
            .sum();

        // Return: ξₜ = ar_mean + εₜ
        let realization = ar_mean + innovation;

        log::trace!(
            "AR sampling for '{}': lag_state={:?}, innovation={:.2}, realization={:.2}",
            self.resource,
            lag_state,
            innovation,
            realization
        );

        realization
    }

    fn lag_order(&self) -> usize {
        self.coefficients.len()
    }

    fn innovation_distribution(&self) -> &Distribution {
        &self.innovation_dist
    }

    fn distribution(&self) -> &Distribution {
        &self.marginal_dist
    }
}
```

### 3. Add Error Types (15 min)

```rust
// In src/error.rs

#[derive(Debug, Error)]
pub enum StochasticProcessError {
    #[error("Invalid lag order: {order} (must be > 0)")]
    InvalidLagOrder { order: usize },

    #[error("Non-stationary AR process: {message}")]
    NonStationary { message: String },

    #[error("Unsupported distribution for AR innovations: {distribution}")]
    UnsupportedDistribution { distribution: String },
}
```

### 4. Tests (3 hours)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_ar1_creation() {
        let ar = AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        assert_eq!(ar.lag_order(), 1);
        assert_eq!(ar.coefficients(), &[0.7]);
    }

    #[test]
    fn test_ar1_conditional_sampling() {
        let ar = AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let lag_state = vec![100.0];

        let realization = ar.sample_conditional(&lag_state, &mut rng);

        // Should be around φ * lag_state[0] = 0.7 * 100 = 70 (plus noise)
        assert!((realization - 70.0).abs() < 50.0); // Wide tolerance for noise
    }

    #[test]
    fn test_ar1_autocorrelation() {
        let ar = AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        // AR(1) autocorrelation: ρₖ = φ^k
        assert_eq!(ar.autocorrelation(0), 1.0);
        assert_relative_eq!(ar.autocorrelation(1), 0.7, epsilon = 1e-9);
        assert_relative_eq!(ar.autocorrelation(2), 0.49, epsilon = 1e-9);
        assert_relative_eq!(ar.autocorrelation(3), 0.343, epsilon = 1e-9);
    }

    #[test]
    fn test_ar1_marginal_variance() {
        let innovation_std = 15.0;
        let phi = 0.7;

        let ar = AutoRegressive::new(
            vec![phi],
            Distribution::Normal { mean: 0.0, std_dev: innovation_std },
            "reservoir_1".to_string(),
        ).unwrap();

        // Marginal variance: σ²ₓ = σ² / (1 - φ²)
        let expected_variance = (innovation_std * innovation_std) / (1.0 - phi * phi);
        let expected_std = expected_variance.sqrt();

        if let Distribution::Normal { mean: _, std_dev } = ar.distribution() {
            assert_relative_eq!(*std_dev, expected_std, epsilon = 1e-6);
        } else {
            panic!("Expected Normal distribution");
        }
    }

    #[test]
    fn test_ar2_conditional_sampling() {
        let ar = AutoRegressive::new(
            vec![0.6, 0.3],
            Distribution::Normal { mean: 0.0, std_dev: 10.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let lag_state = vec![100.0, 95.0]; // [ξₜ₋₁, ξₜ₋₂]

        let realization = ar.sample_conditional(&lag_state, &mut rng);

        // Should be around φ₁*lag[0] + φ₂*lag[1] = 0.6*100 + 0.3*95 = 88.5
        assert!((realization - 88.5).abs() < 40.0);
    }

    #[test]
    #[should_panic(expected = "AR process requires lag state")]
    fn test_ar_unconditional_sample_panics() {
        let ar = AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        ar.sample(&mut rng); // Should panic
    }

    #[test]
    #[should_panic(expected = "requires 2 lag values, got 1")]
    fn test_ar_wrong_lag_count_panics() {
        let ar = AutoRegressive::new(
            vec![0.6, 0.3], // AR(2)
            Distribution::Normal { mean: 0.0, std_dev: 10.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        ar.sample_conditional(&[100.0], &mut rng); // Only 1 lag, needs 2
    }

    #[test]
    fn test_ar_is_conditional() {
        let ar = AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        ).unwrap();

        assert!(ar.is_conditional());
        assert_eq!(ar.lag_order(), 1);
    }

    #[test]
    fn test_non_stationary_rejected() {
        let result = AutoRegressive::new(
            vec![1.05], // Non-stationary
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "reservoir_1".to_string(),
        );

        assert!(result.is_err());
    }
}
```

---

## Documentation Requirements

### Code Documentation

- [ ] Rustdoc for AutoRegressive struct and all methods
- [ ] Mathematical explanation of AR process
- [ ] Examples showing conditional sampling

### User Documentation

- [ ] Explain AR process in user guide
- [ ] Show how to choose AR coefficients
- [ ] Discuss autocorrelation interpretation

---

## Files to Modify

### Core Implementation

- `src/stochastic_process.rs`: Add AutoRegressive struct
- `src/error.rs`: Add StochasticProcessError

### Tests

- `tests/test_stochastic_process.rs`: Add AR tests

---

## Dependencies

### Depends On

- AR-2 (Input validation - coefficients validated)
- AR-6 (Stochastic process trait extension)

### Blocks

- AR-11 (State transition with lag update)
- AR-16 (Scenario generation for AR)

---

## Technical Notes

### AR Equation

**Standard form**: ξₜ = φ₁ξₜ₋₁ + φ₂ξₜ₋₂ + ... + φₚξₜ₋ₚ + εₜ

Where:

- ξₜ: Inflow at time t
- φᵢ: AR coefficients
- εₜ ~ N(0, σ²): White noise innovation

### Marginal Distribution

For stationary AR(1):

- Mean: E[ξₜ] = 0 (zero-mean process)
- Variance: Var(ξₜ) = σ² / (1 - φ²)

For AR(p): Solve Yule-Walker equations (numerical for p > 1).

### Autocorrelation Function

For AR(1): ρₖ = φᵏ (exponential decay)
For AR(p): Yule-Walker recursion

### Performance Considerations

- Sampling: O(p) dot product
- ACF computation: O(k × p) for lag k
- Marginal variance: O(p²) (Yule-Walker)

### Edge Cases

1. **φ = 0**: Degenerates to white noise (wasteful but valid)
2. **φ < 0**: Oscillatory behavior (valid if stationary)
3. **p = 0**: Invalid (should use Naive process)

---

## Validation Checklist

Before marking this ticket complete:

- [ ] AR(1) sampling works correctly
- [ ] AR(2) sampling works correctly
- [ ] Autocorrelation matches theoretical values
- [ ] Marginal distribution correct
- [ ] Non-stationary coefficients rejected
- [ ] Wrong lag count causes panic (fail-fast)
- [ ] All tests passing
- [ ] `cargo clippy` clean
- [ ] Documentation complete

---

## Success Metrics

- ✅ AR(1) and AR(2) processes sample correctly
- ✅ Sample autocorrelation matches theoretical (within statistical error)
- ✅ Marginal variance matches Yule-Walker solution
- ✅ Conditional sampling produces reasonable values
- ✅ Performance overhead <1ms per sample

---

**Created**: 2025-01-10  
**Last Updated**: 2025-01-10  
**Previous Ticket**: AR-8 (State trait refactoring)  
**Next Ticket**: AR-10 (Pre-study nodes for lag initialization)
