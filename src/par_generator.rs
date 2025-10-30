//! Periodic Autoregressive (PAR) generator implementation
//!
//! This module implements the methodology for PAR(p) models, which are
//! autoregressive models with seasonally varying parameters.
//!
//! # PAR(p) Equation (Residual Space Only)
//!
//! **CRITICAL**: PAR dynamics work **exclusively in residual space** (Z'_t).
//! AR coefficients φ operate on standardized residuals, NOT observations.
//!
//! The PAR(p) model generates time series values according to:
//!
//! ```text
//! Z_t = μ_m + σ_m · Z'_t   ← Observation (for output only)
//!
//! where Z'_t follows the AR(p) process (residual space dynamics):
//! Z'_t = φ_1m·Z'_{t-1} + φ_2m·Z'_{t-2} + ... + φ_pm·Z'_{t-p} + ε_t
//!
//! where:
//!   t = time step (stage)
//!   m = t mod period (seasonal index, maps to season_id in graph nodes)
//!   ε_t = innovation (white noise from marginal transform)
//!   Z'_t = standardized residual (stationary AR process)
//!   μ_m = seasonal mean for period m (for observation transform only)
//!   σ_m = seasonal standard deviation for period m (for observation transform only)
//!   φ_km = AR coefficient k for period m in RESIDUAL SPACE (can vary by period!)
//!   p_m = AR order for period m (e.g., AR(1) in dry season, AR(2) in wet season)
//! ```
//!
//! ## Why Residual Space?
//!
//! AR dynamics in observation space (Y_t = φ·Y_{t-1} + ε_t) are **non-stationary**
//! when seasonal means shift (μ_winter ≠ μ_summer). The process has time-varying
//! mean, violating stationarity assumptions required for AR modeling.
//!
//! Solution: Transform to residuals first:
//! ```text
//! Z'_t = (Y_t - μ_m) / σ_m  ← Remove seasonal effects
//! Z'_t = φ·Z'_{t-1} + ε_t   ← NOW stationary and valid!
//! ```
//!
//! # Key Features
//!
//! - **Varying AR orders**: Each period can have different AR order (p₁, p₂, ..., p_period)
//! - **Seasonal parameters**: Mean and standard deviation change each period
//! - **Efficient buffering**: Circular buffer stores max(p_m) past residuals for O(1) access
//! - **Numerical stability**: No accumulation of numerical errors over long simulations
//!
//! # Integration with Pipeline
//!
//! The PAR generator is Stage 4 in the pipeline:
//!
//! 1. **Base Noise**: Generate Z ~ N(0,1)
//! 2. **Correlation**: Apply correlation matrix → W = L×Z
//! 3. **Marginal Transform**: Apply marginal distribution → ε ~ F (LogNormal3, etc.)
//! 4. **PAR Dynamics** (this module): Apply AR equation → final inflow Z_t
//!
//! The generator receives transformed residuals `a_t` from Stage 3 and produces
//! final values `Z_t` with seasonal AR dynamics applied.
//!
//! # Usage Example
//!
//! ```rust
//! use powers_rs::seasonal_params::SeasonalParams;
//! use powers_rs::par_generator::PeriodicARGenerator;
//!
//! // Create seasonal parameters for 12-period (monthly) PAR(1) model
//! let params = SeasonalParams::new(
//!     12,
//!     vec![1; 12],                    // AR(1) for all periods
//!     vec![vec![0.7]; 12],            // φ = 0.7 for all periods
//!     vec![100.0; 12],                // mean = 100 for all periods
//!     vec![20.0; 12],                 // std = 20 for all periods
//! ).unwrap();
//!
//! // Create generator with zero initial conditions
//! let mut generator = PeriodicARGenerator::new(params, Vec::new());
//!
//! // Generate time series from transformed residuals
//! let residuals = vec![0.5, -0.3, 1.2, 0.0];
//! for a_t in residuals {
//!     let z_t = generator.generate_next(a_t);
//!     println!("Stage {}: Z_t = {:.2}", generator.current_stage() - 1, z_t);
//! }
//! ```
//!
//! # Performance Characteristics
//!
//! - **Time complexity**: O(max_order) per generation (bounded by max AR order)
//! - **Space complexity**: O(max_order) for residual buffer
//! - **Cache efficiency**: Circular buffer provides good spatial locality
//! - **Overhead**: ~10 instructions per generation for AR(1), scales linearly with order

use crate::seasonal_params::SeasonalParams;
use std::collections::VecDeque;

/// Output from PAR generator containing innovation and residual
///
/// This structure separates the innovation (ε_t) from the residual (Z'_t)
/// to enable correct implementation of the state expansion trick in SDDP.
///
/// # Mathematical Relationship
///
/// ```text
/// Z'_t = φ₁·Z'_{t-1} + φ₂·Z'_{t-2} + ... + φₚ·Z'_{t-p} + ε_t
/// Y_t = μ + σ·Z'_t
///
/// where:
///   ε_t = innovation (what we pass to AR constraint RHS)
///   Z'_t = residual (what we store for next stage's lags)
///   Y_t = observation (only computed when needed for output)
/// ```
///
/// # Performance Characteristics
///
/// - **Size**: 16 bytes (2 × f64)
/// - **Copy cost**: ~2 cycles (trivially copyable)
/// - **Return optimization**: Typically returned in registers (RVO)
///
/// # Example
///
/// ```rust,ignore
/// let output = par_gen.generate_innovation_and_residual(season_id, base_noise);
///
/// // Use innovation for AR constraint (LP setup)
/// subproblem.set_ar_constraint_rhs(hydro_id, output.innovation);
///
/// // Use residual for state update (for next stage)
/// state.update_lag(hydro_id, output.residual);
///
/// // Compute observation only when needed for output
/// let observation = seasonal_mean + seasonal_std * output.residual;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PAROutput {
    /// Innovation (ε_t) - goes to AR constraint RHS in LP
    ///
    /// This is the white noise component that drives the AR process.
    /// In SDDP, this is what the scenario represents and what gets
    /// parameterized in the subproblem.
    pub innovation: f64,

    /// Residual (Z'_t) - stored for next stage's lags
    ///
    /// This is the AR process value that includes both the AR terms
    /// from past lags and the current innovation:
    /// Z'_t = Σ(φ_k · Z'_{t-k}) + ε_t
    pub residual: f64,
}

impl PAROutput {
    /// Create new PAR output
    #[inline]
    pub fn new(innovation: f64, residual: f64) -> Self {
        Self {
            innovation,
            residual,
        }
    }

    /// Compute observation from residual with seasonal parameters
    ///
    /// # Performance
    ///
    /// - Time: 2 flops (1 mul, 1 add) ~1ns
    /// - **Cold path**: Only called for output/reporting
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = par_gen.generate_innovation_and_residual(season_id, 0.5);
    /// let observation = output.to_observation(100.0, 20.0);
    /// println!("Observation: {}", observation);
    /// ```
    #[inline]
    pub fn to_observation(&self, seasonal_mean: f64, seasonal_std: f64) -> f64 {
        seasonal_mean + seasonal_std * self.residual
    }
}

/// Periodic Autoregressive PAR(p) generator
///
/// This generator implements the PAR(p) equation with seasonally varying
/// parameters. It maintains a circular buffer of past residuals to compute
/// the AR terms efficiently.
///
/// # Example
///
/// ```rust
/// use powers_rs::seasonal_params::SeasonalParams;
/// use powers_rs::par_generator::PeriodicARGenerator;
///
/// let params = SeasonalParams::new(
///     2,                              // 2 periods (e.g., wet/dry seasons)
///     vec![1, 2],                     // AR(1) for period 0, AR(2) for period 1
///     vec![vec![0.7], vec![0.5, 0.3]], // AR coefficients
///     vec![100.0, 120.0],             // seasonal means
///     vec![20.0, 25.0],               // seasonal standard deviations
/// ).unwrap();
///
/// let mut gen = PeriodicARGenerator::new(params, vec![]);
///
/// // Generate values
/// let z0 = gen.generate_next(1.0);
/// let z1 = gen.generate_next(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct PeriodicARGenerator {
    /// Seasonal parameters (μ_m, σ_m, φ_km) validated at construction
    params: SeasonalParams,

    /// Circular buffer of past AR process values [Z'_{t-1}, Z'_{t-2}, ..., Z'_{t-max(p)}]
    ///
    /// PERFORMANCE: Stored in reverse chronological order for easy indexing:
    /// - buffer[0] = a_t-1 (most recent)
    /// - buffer[1] = a_t-2
    /// - buffer[k-1] = a_t-k
    ///
    /// This layout allows direct indexing: buffer[k-1] for φ_k coefficient.
    residual_buffer: VecDeque<f64>,

    /// Current time step (for seasonal indexing m = t mod period)
    current_stage: usize,

    /// Maximum AR order across all periods (for buffer management)
    ///
    /// PERFORMANCE: Cached to avoid repeated max() calls.
    max_order: usize,
}

impl PeriodicARGenerator {
    /// Create a new PAR generator with validated seasonal parameters
    ///
    /// # Arguments
    ///
    /// - `params`: Seasonal parameters (μ_m, σ_m, φ_km) - must be pre-validated
    /// - `initial_residuals`: Past residuals for warm-up period
    ///
    /// # Initial Conditions
    ///
    /// The quality of initial values affects the first max(p_m) generated values:
    ///
    /// - **Empty vector**: Buffer initialized with zeros (cold start)
    ///   - Simple but may cause initial transients
    ///   - Suitable for long simulations where initial bias fades
    ///
    /// - **Historical residuals**: Provide at least max(p_m) past values
    ///   - Takes the last max(p_m) values if more are provided
    ///   - Best for continuing existing time series
    ///   - Pads with zeros if fewer than max(p_m) values provided
    ///
    /// # Performance
    ///
    /// - **Time**: O(max_order) for buffer initialization
    /// - **Space**: O(max_order) for residual buffer
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(2, vec![1, 1], vec![vec![0.7], vec![0.6]], vec![100.0, 120.0], vec![20.0, 25.0]).unwrap();
    /// // Cold start (zeros)
    /// let gen1 = PeriodicARGenerator::new(params.clone(), vec![]);
    ///
    /// // Warm start (historical residuals)
    /// let gen2 = PeriodicARGenerator::new(params, vec![0.5, -0.3, 1.2]);
    /// ```
    pub fn new(params: SeasonalParams, initial_residuals: Vec<f64>) -> Self {
        let max_order = *params.ar_orders.iter().max().unwrap_or(&0);

        let mut buffer = VecDeque::with_capacity(max_order);

        // Initialize buffer with provided residuals or zeros
        if initial_residuals.is_empty() {
            // Cold start: zeros
            buffer.resize(max_order, 0.0);
        } else {
            // Warm start: take last max_order values
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

    /// Generate next value in the time series using the PAR(p) equation
    ///
    /// # PAR(p) Equation
    ///
    /// ```text
    /// Z_t = μ_m + σ_m · [AR_term + a_t]
    ///
    /// where:
    ///   AR_term = Σ φ_km · a_t-k  for k = 1..p_m
    ///   m = t mod period (seasonal index)
    /// ```
    ///
    /// # Arguments
    ///
    /// - `a_t`: Transformed residual for current time step
    ///   - Should come from marginal transformation (LogNormal3, Normal, etc.)
    ///   - Represents the innovation at time t
    ///
    /// # Returns
    ///
    /// - `Z_t`: Final value with seasonality and AR dynamics applied
    ///   - Ready for use in optimization (e.g., inflow to reservoir)
    ///
    /// # Performance
    ///
    /// - **Time**: O(p_m) where p_m is the AR order for current period
    /// - **Allocations**: Zero (uses pre-allocated buffer)
    /// - **Cache**: Good locality (sequential buffer access)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(1, vec![1], vec![vec![0.7]], vec![100.0], vec![20.0]).unwrap();
    /// let mut gen = PeriodicARGenerator::new(params, vec![]);
    ///
    /// // Generate sequence
    /// let z0 = gen.generate_next(1.0);   // Stage 0
    /// let z1 = gen.generate_next(0.5);   // Stage 1
    /// let z2 = gen.generate_next(-0.3);  // Stage 2
    /// ```
    #[inline]
    pub fn generate_next(&mut self, a_t: f64) -> f64 {
        // PERFORMANCE: Inline this hot-path method for zero call overhead
        let season_index = self.current_stage % self.params.num_seasons;

        // Get seasonal parameters for current season
        let mean = self.params.get_mean(season_index);
        let std = self.params.get_std(season_index);
        let ar_order = self.params.get_ar_order(season_index);
        let ar_coeffs = self.params.get_ar_coeffs(season_index);

        // Compute AR term: Σ φ_k · Z'_{t-k}
        //
        // PERFORMANCE: Loop is bounded by ar_order (typically 1-3),
        // so unrolling is not beneficial. Compiler may auto-vectorize.
        let mut ar_term = 0.0;
        for (k, &coeff) in ar_coeffs.iter().enumerate().take(ar_order) {
            // Buffer layout: buffer[0] = Z'_{t-1}, buffer[1] = Z'_{t-2}, ...
            let past_z_prime =
                self.residual_buffer.get(k).copied().unwrap_or(0.0);
            ar_term += coeff * past_z_prime;
        }

        // Compute AR process value: Z'_t = Σ φ_k · Z'_{t-k} + a_t
        let z_prime = ar_term + a_t;

        // Apply PAR equation: Z_t = μ_m + σ_m · Z'_t
        let z_t = mean + std * z_prime;

        // Update buffer: add current AR process value, remove oldest if full
        //
        // PERFORMANCE: VecDeque provides O(1) push_front and pop_back.
        // Circular buffer avoids allocations and copies.
        if self.residual_buffer.len() >= self.max_order {
            self.residual_buffer.pop_back();
        }
        self.residual_buffer.push_front(z_prime);

        self.current_stage += 1;

        z_t
    }

    /// Generate next value for a specific season
    ///
    /// Like `generate_next()`, but allows explicit season control without
    /// relying on `current_stage` counter. Useful for caching scenarios
    /// where stage ordering may not be sequential.
    ///
    /// # Arguments
    ///
    /// - `season_id`: Season index (0..num_seasons-1)
    /// - `a_t`: Current residual from noise distribution
    ///
    /// # Returns
    ///
    /// Generated value with seasonal mean, std dev, and AR dynamics applied.
    ///
    /// # Performance
    ///
    /// Identical to `generate_next()` (~100ns). Auto-increments `current_stage`
    /// for consistency.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Generate for December (season 11) explicitly
    /// let value = gen.generate_next_for_season(11, base_noise);
    /// ```
    #[inline]
    pub fn generate_next_for_season(
        &mut self,
        season_id: usize,
        a_t: f64,
    ) -> f64 {
        // Get seasonal parameters for specified season
        let mean = self.params.get_mean(season_id);
        let std = self.params.get_std(season_id);
        let ar_order = self.params.get_ar_order(season_id);
        let ar_coeffs = self.params.get_ar_coeffs(season_id);

        // Compute AR term: Σ φ_k · Z'_{t-k}
        let mut ar_term = 0.0;
        for (k, &coeff) in ar_coeffs.iter().enumerate().take(ar_order) {
            let past_z_prime =
                self.residual_buffer.get(k).copied().unwrap_or(0.0);
            ar_term += coeff * past_z_prime;
        }

        // Compute AR process value: Z'_t = Σ φ_k · Z'_{t-k} + a_t
        let z_prime = ar_term + a_t;

        // Apply PAR equation: Z_t = μ_m + σ_m · Z'_t
        let z_t = mean + std * z_prime;

        // Update buffer
        if self.residual_buffer.len() >= self.max_order {
            self.residual_buffer.pop_back();
        }
        self.residual_buffer.push_front(z_prime);

        self.current_stage += 1;

        z_t
    }

    /// Generate innovation and residual for state expansion trick
    ///
    /// This is the optimized method for SDDP with PAR models. It returns both
    /// the innovation (ε_t) and residual (Z'_t) without computing the observation,
    /// avoiding unnecessary floating-point operations in the hot path.
    ///
    /// # Arguments
    ///
    /// - `season_id`: Season index (0..num_seasons-1)
    /// - `base_noise`: Base innovation (ε_t), typically from N(0,1) after marginal transform
    ///
    /// # Returns
    ///
    /// `PAROutput` containing:
    /// - `innovation`: ε_t (for AR constraint RHS in LP)
    /// - `residual`: Z'_t (for state update for next stage)
    ///
    /// # Performance
    ///
    /// - **Time**: O(p) where p is AR order (~50ns for AR(1), ~100ns for AR(2))
    /// - **vs generate_next_for_season**: Saves 2 flops (no μ + σ·Z' computation)
    /// - **Speedup**: ~10-15% faster in hot path
    ///
    /// # Correctness
    ///
    /// This is the mathematically correct approach for state expansion:
    /// - Innovation ε_t parameterizes the AR constraint
    /// - Residual Z'_t becomes the state variable for next stage
    /// - Observation Y_t is computed lazily only when needed for output
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Hot path: Generate scenarios
    /// let base_noise: f64 = rng.sample(StandardNormal);
    /// let output = gen.generate_innovation_and_residual(season_id, base_noise);
    ///
    /// // Use in LP
    /// subproblem.set_ar_constraint_rhs(hydro_id, output.innovation);
    /// state.update_lag(hydro_id, output.residual);
    ///
    /// // Cold path: Compute observation for output
    /// if need_output {
    ///     let obs = output.to_observation(seasonal_mean, seasonal_std);
    ///     writer.write_observation(obs);
    /// }
    /// ```
    #[inline]
    pub fn generate_innovation_and_residual(
        &mut self,
        season_id: usize,
        base_noise: f64,
    ) -> PAROutput {
        // Get AR parameters (no mean/std needed here - applied lazily)
        let ar_order = self.params.get_ar_order(season_id);
        let ar_coeffs = self.params.get_ar_coeffs(season_id);

        // PERFORMANCE: Compute AR term - hot path, bounded loop
        // Compiler can auto-vectorize this for larger AR orders
        let mut ar_term = 0.0;
        for (k, &coeff) in ar_coeffs.iter().enumerate().take(ar_order) {
            let past_z_prime =
                self.residual_buffer.get(k).copied().unwrap_or(0.0);
            ar_term += coeff * past_z_prime;
        }

        // Innovation is the base noise (ε_t)
        let innovation = base_noise;

        // Residual is AR process value: Z'_t = Σ φ_k · Z'_{t-k} + ε_t
        let residual = ar_term + innovation;

        // PERFORMANCE: Update buffer with O(1) operations
        // VecDeque provides efficient push_front/pop_back
        if self.residual_buffer.len() >= self.max_order {
            self.residual_buffer.pop_back();
        }
        self.residual_buffer.push_front(residual);

        self.current_stage += 1;

        PAROutput::new(innovation, residual)
    }

    /// Reset generator to initial state
    ///
    /// This allows reusing the generator for multiple simulation runs
    /// without reallocating the buffer.
    ///
    /// # Arguments
    ///
    /// - `initial_residuals`: New initial conditions (same semantics as constructor)
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(1, vec![1], vec![vec![0.7]], vec![100.0], vec![20.0]).unwrap();
    /// let mut gen = PeriodicARGenerator::new(params, vec![]);
    ///
    /// // First simulation
    /// for _ in 0..100 {
    ///     gen.generate_next(0.0);
    /// }
    ///
    /// // Reset for second simulation
    /// gen.reset(vec![]);
    /// for _ in 0..100 {
    ///     gen.generate_next(0.0);
    /// }
    /// ```
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
    ///
    /// This is the number of values generated so far (incremented by `generate_next`).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(1, vec![1], vec![vec![0.7]], vec![100.0], vec![20.0]).unwrap();
    /// let mut gen = PeriodicARGenerator::new(params, vec![]);
    ///
    /// assert_eq!(gen.current_stage(), 0);
    /// gen.generate_next(1.0);
    /// assert_eq!(gen.current_stage(), 1);
    /// gen.generate_next(0.5);
    /// assert_eq!(gen.current_stage(), 2);
    /// ```
    #[inline]
    pub fn current_stage(&self) -> usize {
        self.current_stage
    }

    /// Get reference to underlying seasonal parameters
    ///
    /// Useful for inspecting configuration or passing to other components.
    #[inline]
    pub fn params(&self) -> &SeasonalParams {
        &self.params
    }

    /// Get current seasonal period index (m = t mod period)
    ///
    /// This is the index used to select seasonal parameters at the current stage.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(3, vec![1, 1, 1], vec![vec![0.7], vec![0.6], vec![0.5]], vec![100.0, 110.0, 120.0], vec![20.0, 25.0, 30.0]).unwrap();
    /// let mut gen = PeriodicARGenerator::new(params, vec![]);
    ///
    /// assert_eq!(gen.current_season_index(), 0);
    /// gen.generate_next(1.0);  // Stage 0 -> period 0
    /// assert_eq!(gen.current_season_index(), 1);
    /// gen.generate_next(0.5);  // Stage 1 -> period 1
    /// assert_eq!(gen.current_season_index(), 2);
    /// gen.generate_next(0.0);  // Stage 2 -> season 2
    /// assert_eq!(gen.current_season_index(), 0);  // Wraps around
    /// ```
    #[inline]
    pub fn current_season_index(&self) -> usize {
        self.current_stage % self.params.num_seasons
    }

    /// Get reference to residual buffer
    ///
    /// Returns the current residual buffer state, useful for extracting lags
    /// for next stage or for debugging.
    ///
    /// # Buffer Layout
    ///
    /// Buffer is stored in reverse chronological order:
    /// - buffer[0] = Z'_{t-1} (most recent AR process value)
    /// - buffer[1] = Z'_{t-2}
    /// - buffer[k-1] = Z'_{t-k}
    ///
    /// # Example
    ///
    /// ```rust
    /// # use powers_rs::seasonal_params::SeasonalParams;
    /// # use powers_rs::par_generator::PeriodicARGenerator;
    /// # let params = SeasonalParams::new(1, vec![2], vec![vec![0.5, 0.3]], vec![100.0], vec![20.0]).unwrap();
    /// let mut gen = PeriodicARGenerator::new(params, vec![]);
    ///
    /// gen.generate_next(1.0);  // a_0 = 1.0 → Z'_0 = 0.5·0 + 0.3·0 + 1.0 = 1.0
    /// gen.generate_next(0.5);  // a_1 = 0.5 → Z'_1 = 0.5·1.0 + 0.3·0 + 0.5 = 1.0
    ///
    /// let buffer = gen.get_residual_buffer();
    /// assert_eq!(buffer[0], 1.0);  // Z'_{t-1}
    /// assert_eq!(buffer[1], 1.0);  // Z'_{t-2}
    /// ```
    #[inline]
    pub fn get_residual_buffer(&self) -> &VecDeque<f64> {
        &self.residual_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_par1_simple() {
        // PAR(1) with φ=0.7 (period 0) and φ=0.6 (period 1)
        // Two periods with different means and stds
        let params = SeasonalParams::new(
            2,
            vec![1, 1],
            vec![vec![0.7], vec![0.6]],
            vec![100.0, 120.0],
            vec![20.0, 25.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Stage 0 (period 0): Z'₀ = 0.7·0 + 1.0 = 1.0
        //                     Z₀ = 100 + 20·1.0 = 120
        // (buffer initialized with zeros, so Z'_{-1} = 0)
        let z0 = gen.generate_next(1.0);
        assert!((z0 - 120.0).abs() < 1e-10, "z0 = {}, expected 120", z0);

        // Stage 1 (period 1): Z'₁ = 0.6·1.0 + 0.5 = 1.1
        //                     Z₁ = 120 + 25·1.1 = 147.5
        // (Z'_{t-1} = 1.0 from previous stage)
        let z1 = gen.generate_next(0.5);
        assert!((z1 - 147.5).abs() < 1e-10, "z1 = {}, expected 147.5", z1);

        // Stage 2 (period 0 again): Z'₂ = 0.7·1.1 + 0.8 = 1.57
        //                           Z₂ = 100 + 20·1.57 = 131.4
        // (Z'_{t-1} = 1.1 from previous stage)
        let z2 = gen.generate_next(0.8);
        assert!((z2 - 131.4).abs() < 1e-10, "z2 = {}, expected 131.4", z2);
    }

    #[test]
    fn test_par2() {
        // PAR(2) with two coefficients: φ₁=0.5, φ₂=0.3
        // Single period (stationary AR(2))
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Stage 0: Z'₀ = 0.5·0 + 0.3·0 + 1.0 = 1.0
        //          Z₀ = 100 + 20·1.0 = 120
        let z0 = gen.generate_next(1.0);
        assert!((z0 - 120.0).abs() < 1e-10, "z0 = {}, expected 120", z0);

        // Stage 1: Z'₁ = 0.5·1.0 + 0.3·0 + 0.5 = 1.0
        //          Z₁ = 100 + 20·1.0 = 120
        let z1 = gen.generate_next(0.5);
        assert!((z1 - 120.0).abs() < 1e-10, "z1 = {}, expected 120", z1);

        // Stage 2: Z'₂ = 0.5·1.0 + 0.3·1.0 + 0.8 = 1.6
        //          Z₂ = 100 + 20·1.6 = 132
        let z2 = gen.generate_next(0.8);
        assert!((z2 - 132.0).abs() < 1e-10, "z2 = {}, expected 132", z2);
    }

    #[test]
    fn test_ar0_white_noise() {
        // AR(0) = white noise with seasonality (no AR terms)
        let params = SeasonalParams::new(
            2,
            vec![0, 0],
            vec![vec![], vec![]],
            vec![100.0, 150.0],
            vec![20.0, 30.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Stage 0 (period 0): Z₀ = 100 + 20·1.5 = 130
        let z0 = gen.generate_next(1.5);
        assert!((z0 - 130.0).abs() < 1e-10, "z0 = {}, expected 130", z0);

        // Stage 1 (period 1): Z₁ = 150 + 30·(-0.5) = 135
        let z1 = gen.generate_next(-0.5);
        assert!((z1 - 135.0).abs() < 1e-10, "z1 = {}, expected 135", z1);

        // Stage 2 (period 0): Z₂ = 100 + 20·0.8 = 116
        let z2 = gen.generate_next(0.8);
        assert!((z2 - 116.0).abs() < 1e-10, "z2 = {}, expected 116", z2);
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
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        let z0 = gen.generate_next(1.0);
        assert!(
            (z0 - 120.0).abs() < 1e-10,
            "z0 = {}, expected around 120",
            z0
        );
        let z1 = gen.generate_next(0.5);
        assert!(
            (z1 - 145.0).abs() < 1e-10,
            "z1 = {}, expected around 145.0",
            z1
        );
        let z2 = gen.generate_next(0.8);
        assert!(
            (z2 - 164.0).abs() < 1e-10,
            "z2 = {}, expected around 164",
            z2
        );

        // Verify period wraparound
        let stage = gen.current_stage();
        let season = gen.current_season_index();
        assert_eq!(stage, 3);
        assert_eq!(season, 0); // Should wrap to period 0
    }

    #[test]
    fn test_buffer_wraparound() {
        // AR(2) - buffer should maintain exactly 2 past residuals
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate many values to test buffer management
        for i in 0..100 {
            gen.generate_next(i as f64 * 0.1);
        }

        // Buffer should still have exactly max_order elements
        let buffer_len = gen.get_residual_buffer().len();
        assert_eq!(buffer_len, 2);
    }

    #[test]
    fn test_initial_conditions_empty() {
        // Empty initial conditions → zero-initialized buffer
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let gen = PeriodicARGenerator::new(params, vec![]);

        // Buffer should be initialized with zeros
        let buffer = gen.get_residual_buffer();
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], 0.0);
        assert_eq!(buffer[1], 0.0);
    }

    #[test]
    fn test_initial_conditions_exact() {
        // Provide exactly max_order initial residuals
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let gen = PeriodicARGenerator::new(params, vec![1.5, 2.5]);

        // Buffer should contain provided values
        let buffer = gen.get_residual_buffer();
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], 1.5); // Most recent (a_t-1)
        assert_eq!(buffer[1], 2.5); // Older (a_t-2)
    }

    #[test]
    fn test_initial_conditions_excess() {
        // Provide more than max_order initial residuals → take last max_order
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let gen =
            PeriodicARGenerator::new(params, vec![0.5, 1.0, 1.5, 2.0, 2.5]);

        // Buffer should contain last 2 values
        let buffer = gen.get_residual_buffer();
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer[0], 2.0);
        assert_eq!(buffer[1], 2.5);
    }

    #[test]
    fn test_initial_conditions_insufficient() {
        // Provide fewer than max_order initial residuals → pad with zeros
        let params = SeasonalParams::new(
            1,
            vec![3],
            vec![vec![0.5, 0.3, 0.1]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let gen = PeriodicARGenerator::new(params, vec![1.5]);

        // Buffer should be padded with zeros at the front

        let buffer = gen.get_residual_buffer();
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer[0], 0.0); // Padded zero
        assert_eq!(buffer[1], 0.0); // Padded zero
        assert_eq!(buffer[2], 1.5); // Provided value
    }

    #[test]
    fn test_reset() {
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate some values
        gen.generate_next(1.0);
        gen.generate_next(0.5);
        gen.generate_next(0.8);

        let stage = gen.current_stage();
        assert_eq!(stage, 3);

        // Reset
        gen.reset(vec![]);

        let stage = gen.current_stage();
        assert_eq!(stage, 0);
        let buffer = gen.get_residual_buffer();
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer[0], 0.0);
    }

    #[test]
    fn test_current_season_index() {
        let params = SeasonalParams::new(
            3,
            vec![1, 1, 1],
            vec![vec![0.7], vec![0.6], vec![0.5]],
            vec![100.0, 110.0, 120.0],
            vec![20.0, 25.0, 30.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        let mut season_index = gen.current_season_index();
        assert_eq!(season_index, 0);
        gen.generate_next(1.0);
        season_index = gen.current_season_index();
        assert_eq!(season_index, 1);
        gen.generate_next(0.5);
        season_index = gen.current_season_index();
        assert_eq!(season_index, 2);
        gen.generate_next(0.0);
        season_index = gen.current_season_index();
        assert_eq!(season_index, 0); // Wraparound
        gen.generate_next(0.0);
        season_index = gen.current_season_index();
        assert_eq!(season_index, 1);
    }

    #[test]
    fn test_long_simulation_stability() {
        // Verify numerical stability over long simulation
        let params = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![20.0; 12],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate 10000 stages with controlled residuals
        for i in 0..10000 {
            let a_t = ((i as f64 * 0.1).sin()) * 0.5; // Bounded residuals
            let z_t = gen.generate_next(a_t);

            // Values should remain reasonable (no overflow/underflow)
            assert!(z_t.is_finite(), "Non-finite value at stage {}", i);
            assert!(
                z_t > -1000.0 && z_t < 1000.0,
                "Value out of range: {}",
                z_t
            );
        }
    }

    #[test]
    fn test_negative_residuals() {
        // Verify correct handling of negative residuals
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Negative residuals should work correctly
        let z0 = gen.generate_next(-1.0);
        assert!((z0 - 80.0).abs() < 1e-10, "z0 = {}, expected 80", z0);

        let z1 = gen.generate_next(-0.5);
        // Z₁ = 100 + 20·(0.7·(-1.0) + (-0.5)) = 100 + 20·(-1.2) = 76
        assert!((z1 - 76.0).abs() < 1e-10, "z1 = {}, expected 76", z1);
    }

    #[test]
    fn test_seasonal_variation() {
        // Verify that seasonal parameters (mean, std) are applied correctly
        let params = SeasonalParams::new(
            4,
            vec![0, 0, 0, 0], // AR(0) to isolate seasonal effects
            vec![vec![], vec![], vec![], vec![]],
            vec![100.0, 200.0, 150.0, 250.0], // Varying means
            vec![10.0, 20.0, 15.0, 25.0],     // Varying stds
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // All with residual = 1.0, different seasonal parameters
        let z0 = gen.generate_next(1.0); // 100 + 10*1.0 = 110
        let z1 = gen.generate_next(1.0); // 200 + 20*1.0 = 220
        let z2 = gen.generate_next(1.0); // 150 + 15*1.0 = 165
        let z3 = gen.generate_next(1.0); // 250 + 25*1.0 = 275

        assert!((z0 - 110.0).abs() < 1e-10, "z0 = {}, expected 110", z0);
        assert!((z1 - 220.0).abs() < 1e-10, "z1 = {}, expected 220", z1);
        assert!((z2 - 165.0).abs() < 1e-10, "z2 = {}, expected 165", z2);
        assert!((z3 - 275.0).abs() < 1e-10, "z3 = {}, expected 275", z3);

        // Verify wraparound
        let z4 = gen.generate_next(1.0); // Back to period 0: 110
        assert!((z4 - 110.0).abs() < 1e-10, "z4 = {}, expected 110", z4);
    }

    #[test]
    fn test_edge_case_all_zero_coefficients() {
        // Test PAR with all AR coefficients = 0 (reduces to white noise + seasonal mean)
        let params = SeasonalParams::new(
            4,
            vec![2, 2, 2, 2], // AR(2) order but with zero coefficients
            vec![vec![0.0, 0.0]; 4],
            vec![100.0, 150.0, 200.0, 150.0],
            vec![20.0, 25.0, 30.0, 25.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // With zero coefficients: Z_t = μ_m + σ_m·a_t (no AR dependence)
        let z0 = gen.generate_next(1.0); // 100 + 20*1 = 120
        let z1 = gen.generate_next(0.5); // 150 + 25*0.5 = 162.5
        let z2 = gen.generate_next(-1.0); // 200 + 30*(-1) = 170
        let z3 = gen.generate_next(2.0); // 150 + 25*2 = 200

        assert!((z0 - 120.0).abs() < 1e-10, "z0 = {}", z0);
        assert!((z1 - 162.5).abs() < 1e-10, "z1 = {}", z1);
        assert!((z2 - 170.0).abs() < 1e-10, "z2 = {}", z2);
        assert!((z3 - 200.0).abs() < 1e-10, "z3 = {}", z3);
    }

    #[test]
    fn test_edge_case_max_ar_order() {
        // Test with maximum practical AR order (p=12)
        let ar_order = 12;
        // Use small coefficients to ensure stationarity: sum(|φ_k|) < 1
        // coeffs = [0.01, 0.02, ..., 0.12] → sum = 0.78
        let coeffs: Vec<f64> =
            (1..=ar_order).map(|k| 0.01 * k as f64).collect(); // Sum = 0.01*(1+2+...+12) = 0.78

        let params = SeasonalParams::new(
            1,
            vec![ar_order],
            vec![coeffs.clone()],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate values to fill buffer
        let residuals = vec![
            1.0, 0.5, 0.8, 0.2, -0.5, 1.2, 0.0, -0.3, 0.9, 0.4, -0.2, 0.7,
        ];
        let mut z_primes = Vec::new();
        for &a_t in &residuals {
            let z_t = gen.generate_next(a_t);
            assert!(z_t.is_finite());
            // Compute Z' = (Z - μ) / σ for verification
            z_primes.push((z_t - 100.0) / 20.0);
        }

        // Generate one more to use all lags
        let z_final = gen.generate_next(0.1);

        // Manual calculation: Z'_t = Σφ_k·Z'_{t-k} + a_t
        // ar_sum = 0.01·Z'_{t-1} + 0.02·Z'_{t-2} + ... + 0.12·Z'_{t-12}
        let ar_sum: f64 = z_primes
            .iter()
            .rev()
            .zip(coeffs.iter())
            .map(|(z_prime, phi)| phi * z_prime)
            .sum();
        let z_prime_final = ar_sum + 0.1;
        let expected = 100.0 + 20.0 * z_prime_final;

        assert!(
            (z_final - expected).abs() < 1e-8,
            "z_final = {}, expected {}",
            z_final,
            expected
        );
    }

    #[test]
    fn test_edge_case_single_period_no_seasonality() {
        // Test with period=1 (no seasonality, just stationary AR)
        let params = SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.6, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Verify it behaves like stationary AR(2)
        let z0 = gen.generate_next(1.0);
        let z1 = gen.generate_next(0.5);
        let z2 = gen.generate_next(0.8);

        // Z'₀ = 0.6·0 + 0.3·0 + 1.0 = 1.0
        // Z₀ = 100 + 20·1.0 = 120
        assert!((z0 - 120.0).abs() < 1e-10);

        // Z'₁ = 0.6·1.0 + 0.3·0 + 0.5 = 1.1
        // Z₁ = 100 + 20·1.1 = 122
        assert!((z1 - 122.0).abs() < 1e-10);

        // Z'₂ = 0.6·1.1 + 0.3·1.0 + 0.8 = 1.76
        // Z₂ = 100 + 20·1.76 = 135.2
        assert!((z2 - 135.2).abs() < 1e-10);
    }

    #[test]
    fn test_edge_case_ar_order_zero() {
        // Test with AR(0) - pure white noise with seasonal mean/std
        let params = SeasonalParams::new(
            3,
            vec![0, 0, 0],
            vec![vec![], vec![], vec![]],
            vec![50.0, 100.0, 150.0],
            vec![10.0, 20.0, 30.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // AR(0): Z_t = μ_m + σ_m·a_t (no dependence on past)
        let z0 = gen.generate_next(2.0); // 50 + 10*2 = 70
        let z1 = gen.generate_next(1.5); // 100 + 20*1.5 = 130
        let z2 = gen.generate_next(-1.0); // 150 + 30*(-1) = 120

        assert!((z0 - 70.0).abs() < 1e-10);
        assert!((z1 - 130.0).abs() < 1e-10);
        assert!((z2 - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_edge_case_very_small_coefficients() {
        // Test numerical stability with very small AR coefficients
        let params = SeasonalParams::new(
            2,
            vec![1, 1],
            vec![vec![1e-10], vec![1e-9]],
            vec![100.0, 100.0],
            vec![1.0, 1.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate values - should essentially behave like AR(0)
        for _ in 0..100 {
            let z = gen.generate_next(1.0);
            // Should be very close to μ + σ·a_t = 100 + 1*1 = 101
            assert!((z - 101.0).abs() < 0.01, "z = {}", z);
        }
    }

    #[test]
    fn test_edge_case_near_unit_root() {
        // Test with coefficient very close to 1 (near unit root, but stationary)
        let phi = 0.99;
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![phi]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // With near-unit-root, process has very slow decay
        // Generate with constant positive residual to build up value
        let mut values = Vec::new();
        for _ in 0..50 {
            let z = gen.generate_next(1.0);
            values.push(z);
            // Values should remain bounded and finite
            assert!(z.is_finite());
        }

        // Check that values are increasing initially (building up momentum)
        assert!(
            values[10] > values[0],
            "Values should increase initially with positive residuals"
        );

        // Eventually should stabilize (stationarity requires |φ| < 1)
        // Generate many more iterations
        for _ in 0..1000 {
            let z = gen.generate_next(1.0);
            assert!(z.is_finite());
            // With φ=0.99 and constant a_t=1, steady state is:
            // Z = μ + σ·[φ/(1-φ) + 1]·a_t = 100 + 20·[99 + 1]·1 = 2100
            // Should remain bounded near this value
            assert!(z < 3000.0, "Value unbounded: {}", z);
        }
    }

    #[test]
    fn test_stability_million_iterations() {
        let params = SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![20.0; 12],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate 1 million stages
        for i in 0..1_000_000 {
            // Use deterministic but varying residuals
            let a_t = ((i as f64 * 0.001).sin()) * 2.0;
            let z_t = gen.generate_next(a_t);

            // Verify no overflow/underflow/NaN
            assert!(z_t.is_finite(), "Non-finite at iteration {}", i);

            // With bounded residuals and stationary AR, values should remain bounded
            assert!(
                z_t.abs() < 500.0,
                "Value out of bounds at iteration {}: {}",
                i,
                z_t
            );

            // Periodic check to ensure we're wrapping correctly
            if i % 100000 == 0 && i > 0 {
                let period_idx = gen.current_season_index();
                assert!(period_idx < 12);
            }
        }
    }

    #[test]
    fn test_stability_extreme_residuals() {
        // Test with extreme (but finite) residuals
        let params = SeasonalParams::new(
            2,
            vec![1, 1],
            vec![vec![0.5], vec![0.6]],
            vec![100.0, 150.0],
            vec![20.0, 30.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Test with large positive residuals
        for _ in 0..100 {
            let z = gen.generate_next(100.0);
            assert!(z.is_finite());
        }

        // Reset and test with large negative residuals
        gen.reset(vec![]);
        for _ in 0..100 {
            let z = gen.generate_next(-100.0);
            assert!(z.is_finite());
        }

        // Test with alternating extreme values
        gen.reset(vec![]);
        for i in 0..100 {
            let a_t = if i % 2 == 0 { 50.0 } else { -50.0 };
            let z = gen.generate_next(a_t);
            assert!(z.is_finite());
        }
    }

    #[test]
    fn test_stability_rapid_oscillation() {
        // Test stability with rapidly oscillating residuals
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.8]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Rapidly alternating residuals
        for i in 0..10000 {
            let a_t = if i % 2 == 0 { 5.0 } else { -5.0 };
            let z = gen.generate_next(a_t);
            assert!(z.is_finite());
            // Should oscillate but remain bounded
            assert!(z.abs() < 500.0, "Value unbounded at i={}: {}", i, z);
        }
    }

    #[test]
    fn test_stationarity_long_run_mean() {
        // Verify long-run mean converges to seasonal mean (after warmup)
        let target_mean = 100.0;
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![target_mean],
            vec![20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Warmup: discard first 1000 values
        for _ in 0..1000 {
            let a_t = (rand::random::<f64>() - 0.5) * 2.0; // U(-1, 1) ≈ mean 0
            gen.generate_next(a_t);
        }

        // Collect statistics over 10000 values
        let mut sum = 0.0;
        let n = 10000;
        for _ in 0..n {
            let a_t = (rand::random::<f64>() - 0.5) * 2.0;
            let z = gen.generate_next(a_t);
            sum += z;
        }

        let empirical_mean = sum / n as f64;

        // With zero-mean residuals and stationary AR, empirical mean should be close to μ
        // Allow 5% tolerance due to finite sample
        let tolerance = target_mean * 0.05;
        assert!(
            (empirical_mean - target_mean).abs() < tolerance,
            "Empirical mean {} not close to target {}",
            empirical_mean,
            target_mean
        );
    }

    #[test]
    fn test_stationarity_variance_convergence() {
        // Verify variance converges for stationary PAR(1)
        let mu = 100.0;
        let sigma = 20.0;
        let phi = 0.7;

        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![phi]],
            vec![mu],
            vec![sigma],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Warmup
        for _ in 0..1000 {
            gen.generate_next((rand::random::<f64>() - 0.5) * 2.0);
        }

        // Collect values
        let n = 10000;
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            let a_t = (rand::random::<f64>() - 0.5) * 2.0;
            values.push(gen.generate_next(a_t));
        }

        // Compute variance
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let variance: f64 =
            values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // For AR(1): Var(Z) = σ²·Var(a)·(1 + φ²/(1-φ²))
        // With a_t ~ U(-1,1), Var(a) ≈ 1/3
        // So Var(Z) ≈ σ²/3 · (1 + φ²/(1-φ²))
        let var_a = 1.0 / 3.0;
        let theoretical_var =
            sigma.powi(2) * var_a * (1.0 + phi.powi(2) / (1.0 - phi.powi(2)));

        // Allow 30% tolerance (variance estimates are noisy)
        let tolerance = theoretical_var * 0.3;
        assert!(
            (variance - theoretical_var).abs() < tolerance,
            "Empirical variance {} not close to theoretical {}",
            variance,
            theoretical_var
        );
    }

    #[test]
    fn test_stationarity_periodic_means() {
        // Verify long-run means converge to seasonal means for each period
        let seasonal_means = vec![80.0, 120.0, 100.0];
        let params = SeasonalParams::new(
            3,
            vec![1, 1, 1],
            vec![vec![0.6], vec![0.7], vec![0.5]],
            seasonal_means.clone(),
            vec![20.0, 25.0, 22.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Warmup
        for _ in 0..300 {
            gen.generate_next((rand::random::<f64>() - 0.5) * 2.0);
        }

        // Collect values by period
        let mut sums = [0.0; 3];
        let mut counts = [0; 3];
        let total_iterations = 3000; // 1000 complete cycles

        for _ in 0..total_iterations {
            let period_idx = gen.current_season_index();
            let a_t = (rand::random::<f64>() - 0.5) * 2.0;
            let z = gen.generate_next(a_t);
            sums[period_idx] += z;
            counts[period_idx] += 1;
        }

        // Check each period's empirical mean
        for period in 0..3 {
            let empirical_mean = sums[period] / counts[period] as f64;
            let target_mean = seasonal_means[period];
            let tolerance = target_mean * 0.1; // 10% tolerance

            assert!(
                (empirical_mean - target_mean).abs() < tolerance,
                "Period {} mean {} not close to target {}",
                period,
                empirical_mean,
                target_mean
            );
        }
    }

    #[test]
    fn test_property_generated_values_always_finite() {
        // Property: All generated values must be finite for any finite residuals
        let params = SeasonalParams::new(
            4,
            vec![2, 1, 2, 1],
            vec![vec![0.6, 0.3], vec![0.7], vec![0.5, 0.4], vec![0.8]],
            vec![90.0, 110.0, 130.0, 100.0],
            vec![15.0, 20.0, 25.0, 18.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Test with various residual patterns
        for _ in 0..1000 {
            let a_t = (rand::random::<f64>() - 0.5) * 10.0; // Range [-5, 5]
            let z = gen.generate_next(a_t);
            assert!(
                z.is_finite(),
                "Non-finite value generated for residual {}",
                a_t
            );
        }
    }

    #[test]
    fn test_property_stationarity_implies_boundedness() {
        // Property: For stationary PAR (|φ| < 1), bounded residuals → bounded output
        let params = SeasonalParams::new(
            2,
            vec![1, 1],
            vec![vec![0.7], vec![0.6]], // Stationary
            vec![100.0, 100.0],
            vec![20.0, 20.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Use bounded residuals [-2, 2]
        let mut max_value = f64::NEG_INFINITY;
        let mut min_value = f64::INFINITY;

        for i in 0..10000 {
            let a_t = ((i as f64 * 0.1).sin()) * 2.0; // Bounded in [-2, 2]
            let z = gen.generate_next(a_t);

            max_value = max_value.max(z);
            min_value = min_value.min(z);

            // Output should remain bounded
            assert!(
                z.abs() < 300.0,
                "Unbounded output {} at iteration {}",
                z,
                i
            );
        }

        // Verify we actually explored the space
        assert!(
            max_value > 100.0,
            "Max value {} too low, not exploring state space",
            max_value
        );
        assert!(
            min_value < 100.0,
            "Min value {} too high, not exploring state space",
            min_value
        );
    }

    #[test]
    fn test_property_zero_residual_converges_to_mean() {
        // Property: With a_t = 0 always, Z_t → μ_m asymptotically
        let params = SeasonalParams::new(
            2,
            vec![1, 1],
            vec![vec![0.8], vec![0.6]],
            vec![100.0, 150.0],
            vec![20.0, 25.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Feed zero residuals for long time
        for _ in 0..100 {
            gen.generate_next(0.0);
        }

        // After many iterations with a_t=0, should converge to seasonal means
        let z0 = gen.generate_next(0.0);
        let z1 = gen.generate_next(0.0);

        // Should be very close to seasonal means
        assert!((z0 - 100.0).abs() < 0.01, "z0 = {} not near 100", z0);
        assert!((z1 - 150.0).abs() < 0.01, "z1 = {} not near 150", z1);
    }

    #[test]
    fn test_property_linear_in_residual() {
        // Property: For fixed history, Z_t is linear in a_t
        let params = SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut gen1 = PeriodicARGenerator::new(params.clone(), vec![]);
        let mut gen2 = PeriodicARGenerator::new(params, vec![]);

        // Build same history
        gen1.generate_next(1.0);
        gen2.generate_next(1.0);

        // Now test linearity: Z(a1) - Z(a2) = σ·(a1 - a2)
        let a1 = 2.0;
        let a2 = -1.0;
        let z1 = gen1.generate_next(a1);
        let z2 = gen2.generate_next(a2);

        let expected_diff = 20.0 * (a1 - a2); // σ = 20
        let actual_diff = z1 - z2;

        assert!(
            (actual_diff - expected_diff).abs() < 1e-10,
            "Linearity violated: diff = {}, expected {}",
            actual_diff,
            expected_diff
        );
    }

    #[test]
    fn test_property_reset_restores_initial_state() {
        // Property: reset() should make generator produce same sequence again
        let params = SeasonalParams::new(
            3,
            vec![1, 1, 1],
            vec![vec![0.7], vec![0.6], vec![0.8]],
            vec![100.0, 120.0, 110.0],
            vec![20.0, 25.0, 22.0],
        )
        .unwrap();

        let mut gen = PeriodicARGenerator::new(params, vec![]);

        // Generate sequence
        let residuals = vec![1.0, -0.5, 0.8, 1.2, 0.3, -0.7];
        let mut first_run = Vec::new();
        for &a_t in &residuals {
            first_run.push(gen.generate_next(a_t));
        }

        // Reset and generate again
        gen.reset(vec![]);
        let mut second_run = Vec::new();
        for &a_t in &residuals {
            second_run.push(gen.generate_next(a_t));
        }

        // Both runs should be identical
        for (i, (&z1, &z2)) in first_run.iter().zip(&second_run).enumerate() {
            assert!(
                (z1 - z2).abs() < 1e-10,
                "Mismatch at index {}: {} vs {}",
                i,
                z1,
                z2
            );
        }
    }
}
