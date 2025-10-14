# PAR-006: Implement PeriodicARGenerator with PAR(p) Equation

## Context

**CRITICAL TICKET**: This is the core PAR generator implementing CEPEL's exact equation:

```
Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
```

The generator takes correlated, transformed residuals `aₜ` and applies periodic AR dynamics with seasonal re-scaling. This is the heart of the PAR model implementation.

## Acceptance Criteria

- [ ] `PeriodicARGenerator` struct implements full PAR(p) equation
- [ ] Handles varying AR orders across periods (p₁, p₂, ..., p₁₂)
- [ ] Maintains circular buffer of past residuals (size = max(pₘ))
- [ ] Correct seasonal indexing (month = t mod period)
- [ ] Initial conditions handled properly (warm-up period)
- [ ] Numerically stable for long simulations
- [ ] Comprehensive unit tests with known-output cases
- [ ] Benchmarks show <10% overhead vs stationary AR

## Tasks

### Implementation

- [ ] Create `src/par_generator.rs` module
- [ ] Implement `PeriodicARGenerator` struct:
  - Fields: seasonal_params, residual_buffer, current_stage
  - Constructor: `new(SeasonalParams, initial_residuals)`
  - Main method: `generate_next(a_t) -> Z_t`
- [ ] Implement circular buffer management:
  - Efficient ring buffer (VecDeque or fixed array)
  - Automatic wraparound
  - Resizing for varying AR orders
- [ ] Implement PAR equation evaluation:
  - Seasonal index calculation
  - AR term summation
  - Scaling and offset application
- [ ] Add to `lib.rs` as public module

### Testing

- [ ] Unit test: PAR(1) with φ=0.7 matches hand calculation
- [ ] Unit test: PAR(2) with two coefficients matches formula
- [ ] Unit test: AR(0) (white noise + seasonality) works correctly
- [ ] Unit test: Varying orders (AR(1), AR(2), AR(1), ...) across periods
- [ ] Unit test: Buffer wraparound after max_order iterations
- [ ] Unit test: Initial conditions with warm-up period
- [ ] Integration test: Generate 1000 stages, verify mean/std convergence
- [ ] Regression test: Known CEPEL test case (if available)

### Documentation

- [ ] Module doc explaining CEPEL PAR(p) equation
- [ ] Detailed comments on buffer management
- [ ] Example usage with SeasonalParams
- [ ] Performance notes and complexity analysis
- [ ] Reference to CEPEL papers/manuals

## Technical Notes

### Implementation

````rust
// src/par_generator.rs

use crate::seasonal_params::SeasonalParams;
use std::collections::VecDeque;

/// Periodic Autoregressive PAR(p) generator (CEPEL methodology)
///
/// Implements the CEPEL PAR(p) equation:
///
/// ```text
/// Zₜ = μₘ + σₘ · [φ₁ₘ·aₜ₋₁ + φ₂ₘ·aₜ₋₂ + ... + φₚₘ·aₜ₋ₚ + aₜ]
///
/// where:
///   m = t mod period (seasonal index, maps to season_id in graph nodes)
///   aₜ = transformed residual (e.g., from LogNormal3)
///   μₘ = seasonal mean for period m
///   σₘ = seasonal standard deviation for period m
///   φₖₘ = AR coefficient k for period m
///   pₘ = AR order for period m (can vary!)
/// ```
///
/// # Key Features
///
/// - **Varying AR orders**: Each period can have different pₘ (e.g., AR(1) in dry season, AR(2) in wet)
/// - **Seasonal parameters**: Mean and std dev change each period (map to season_id in graph nodes)
/// - **Efficient buffering**: Circular buffer stores max(pₘ) past residuals
///
/// # Usage
///
/// ```rust
/// use powers::seasonal_params::SeasonalParams;
/// use powers::par_generator::PeriodicARGenerator;
///
/// let params = SeasonalParams::new(12, ...)?;
/// let mut generator = PeriodicARGenerator::new(params, Vec::new());
///
/// // Generate time series
/// for residual in transformed_residuals {
///     let value = generator.generate_next(residual);
///     println!("Generated: {}", value);
/// }
/// ```
pub struct PeriodicARGenerator {
    /// Seasonal parameters (μₘ, σₘ, φₖₘ)
    params: SeasonalParams,

    /// Circular buffer of past residuals [aₜ₋₁, aₜ₋₂, ..., aₜ₋ₘₐₓ₍ₚ₎]
    /// Stored in reverse chronological order for easy indexing
    residual_buffer: VecDeque<f64>,

    /// Current time step (for seasonal indexing)
    current_stage: usize,

    /// Maximum AR order across all periods
    max_order: usize,
}

impl PeriodicARGenerator {
    /// Create a new PAR generator
    ///
    /// # Arguments
    ///
    /// - `params`: Seasonal parameters (validated)
    /// - `initial_residuals`: Past residuals for warm-up (length >= max AR order)
    ///
    /// # Initial Conditions
    ///
    /// If `initial_residuals` is empty, buffer is initialized with zeros.
    /// For better quality, provide at least `max(pₘ)` historical residuals.
    pub fn new(params: SeasonalParams, initial_residuals: Vec<f64>) -> Self {
        let max_order = *params.ar_orders.iter().max().unwrap_or(&0);

        let mut buffer = VecDeque::with_capacity(max_order);

        // Initialize buffer with provided residuals or zeros
        if initial_residuals.is_empty() {
            buffer.resize(max_order, 0.0);
        } else {
            // Take last max_order values
            let start = initial_residuals.len().saturating_sub(max_order);
            for &val in &initial_residuals[start..] {
                buffer.push_back(val);
            }
            // Pad with zeros if needed
            while buffer.len() < max_order {
                buffer.push_front(0.0);
            }
        }

        Self {
            params,
            residual_buffer: buffer,
            current_stage: 0,
            max_order,
        }
    }

    /// Generate next value in the time series
    ///
    /// # Arguments
    ///
    /// - `a_t`: Transformed residual for current time step
    ///
    /// # Returns
    ///
    /// - `Z_t`: Final value with seasonality and AR dynamics applied
    ///
    /// # CEPEL Equation
    ///
    /// ```text
    /// Z_t = μₘ + σₘ · [AR_term + a_t]
    ///
    /// where AR_term = Σ φₖₘ·aₜ₋ₖ for k=1..pₘ
    /// ```
    pub fn generate_next(&mut self, a_t: f64) -> f64 {
        let period_index = self.current_stage % self.params.period;

        // Get seasonal parameters for current period
        let mean = self.params.get_mean(period_index);
        let std = self.params.get_std(period_index);
        let ar_order = self.params.get_ar_order(period_index);
        let ar_coeffs = self.params.get_ar_coeffs(period_index);

        // Compute AR term: Σ φₖ·aₜ₋ₖ
        let mut ar_term = 0.0;
        for k in 0..ar_order {
            // Buffer is in reverse chronological order:
            // buffer[0] = aₜ₋₁, buffer[1] = aₜ₋₂, ...
            let past_residual = self.residual_buffer.get(k).copied().unwrap_or(0.0);
            ar_term += ar_coeffs[k] * past_residual;
        }

        // Apply PAR equation: Z_t = μₘ + σₘ·(AR_term + a_t)
        let z_t = mean + std * (ar_term + a_t);

        // Update buffer: add current residual, remove oldest if full
        if self.residual_buffer.len() >= self.max_order {
            self.residual_buffer.pop_back();
        }
        self.residual_buffer.push_front(a_t);

        self.current_stage += 1;

        z_t
    }

    /// Reset generator to initial state
    pub fn reset(&mut self, initial_residuals: Vec<f64>) {
        self.current_stage = 0;
        self.residual_buffer.clear();

        if initial_residuals.is_empty() {
            self.residual_buffer.resize(self.max_order, 0.0);
        } else {
            let start = initial_residuals.len().saturating_sub(self.max_order);
            for &val in &initial_residuals[start..] {
                self.residual_buffer.push_back(val);
            }
            while self.residual_buffer.len() < self.max_order {
                self.residual_buffer.push_front(0.0);
            }
        }
    }

    /// Get current stage index
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }
}
````

### Test Cases

```rust
// tests/test_par_generator.rs

use powers::seasonal_params::SeasonalParams;
use powers::par_generator::PeriodicARGenerator;

#[test]
fn test_par1_simple() {
    // PAR(1) with φ=0.7, period=2
    let params = SeasonalParams::new(
        2,
        vec![1, 1],
        vec![vec![0.7], vec![0.6]],
        vec![100.0, 120.0],
        vec![20.0, 25.0],
    ).unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Stage 0 (period 0): Z₀ = 100 + 20·(0.7·0 + 1.0) = 120
    let z0 = gen.generate_next(1.0);
    assert!((z0 - 120.0).abs() < 1e-10);

    // Stage 1 (period 1): Z₁ = 120 + 25·(0.6·1.0 + 0.5) = 147.5
    let z1 = gen.generate_next(0.5);
    assert!((z1 - 147.5).abs() < 1e-10);

    // Stage 2 (period 0 again): Z₂ = 100 + 20·(0.7·0.5 + 0.8) = 123
    let z2 = gen.generate_next(0.8);
    assert!((z2 - 123.0).abs() < 1e-10);
}

#[test]
fn test_par2() {
    // PAR(2) with two coefficients
    let params = SeasonalParams::new(
        1,
        vec![2],
        vec![vec![0.5, 0.3]],
        vec![100.0],
        vec![20.0],
    ).unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Stage 0: Z₀ = 100 + 20·(0.5·0 + 0.3·0 + 1.0) = 120
    let z0 = gen.generate_next(1.0);
    assert!((z0 - 120.0).abs() < 1e-10);

    // Stage 1: Z₁ = 100 + 20·(0.5·1.0 + 0.3·0 + 0.5) = 120
    let z1 = gen.generate_next(0.5);
    assert!((z1 - 120.0).abs() < 1e-10);

    // Stage 2: Z₂ = 100 + 20·(0.5·0.5 + 0.3·1.0 + 0.8) = 127
    let z2 = gen.generate_next(0.8);
    assert!((z2 - 127.0).abs() < 1e-10);
}

#[test]
fn test_ar0_white_noise() {
    // AR(0) = white noise with seasonality
    let params = SeasonalParams::new(
        2,
        vec![0, 0],
        vec![vec![], vec![]],
        vec![100.0, 150.0],
        vec![20.0, 30.0],
    ).unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Stage 0: Z₀ = 100 + 20·1.5 = 130
    let z0 = gen.generate_next(1.5);
    assert!((z0 - 130.0).abs() < 1e-10);

    // Stage 1: Z₁ = 150 + 30·(-0.5) = 135
    let z1 = gen.generate_next(-0.5);
    assert!((z1 - 135.0).abs() < 1e-10);
}

#[test]
fn test_varying_orders() {
    // Period 0: AR(1), Period 1: AR(2), Period 2: AR(0)
    let params = SeasonalParams::new(
        3,
        vec![1, 2, 0],
        vec![vec![0.7], vec![0.5, 0.3], vec![]],
        vec![100.0, 120.0, 140.0],
        vec![20.0, 25.0, 30.0],
    ).unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    let z0 = gen.generate_next(1.0);
    let z1 = gen.generate_next(0.5);
    let z2 = gen.generate_next(0.8);

    // Just verify no panics and values are reasonable
    assert!(z0 > 0.0);
    assert!(z1 > 0.0);
    assert!(z2 > 0.0);
}

#[test]
fn test_long_simulation_statistics() {
    // Generate 12000 stages, verify mean/std convergence
    let params = SeasonalParams::new(
        12,
        vec![1; 12],
        vec![vec![0.7]; 12],
        vec![100.0; 12],
        vec![20.0; 12],
    ).unwrap();

    let mut gen = PeriodicARGenerator::new(params, vec![]);

    // Generate white noise residuals
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut values = Vec::new();
    for _ in 0..12000 {
        let a_t: f64 = rng.gen_range(-2.0..2.0);
        let z_t = gen.generate_next(a_t);
        values.push(z_t);
    }

    // Check mean (should be near 100 for all periods)
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    assert!((mean - 100.0).abs() < 10.0); // Within 10% due to AR autocorrelation
}
```

## Dependencies

- **Blocked by**: PAR-005 (SeasonalParams container)
- **Blocks**: PAR-009 (scenario integration needs this)
- **Related**: PAR-007 (residual transformation)

## Estimated Effort

**4 story points** (confidence: medium - core algorithm)

- 3 hours implementation (struct, equation, buffer management)
- 3 hours testing (8 unit tests including long simulation)
- 1 hour documentation
- 1 hour performance validation

### Breakdown

- Implement struct and constructor: 1 hour
- Implement generate_next() with PAR equation: 1.5 hours
- Implement buffer management: 0.5 hours
- Reset and helper methods: 0.5 hours
- Write 8 unit tests: 3 hours
- Documentation with equation details: 1 hour
- Benchmark against stationary AR: 0.5 hours
- Code review: 0.5 hours
