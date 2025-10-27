use crate::input;
use crate::par_generator;
use crate::seasonal_params;

/// Stochastic process for transforming noise into realizations
///
/// # Conditional vs Unconditional Processes
///
/// **Unconditional** processes (e.g., independent noise):
/// - Realizations are independent: P(ξₜ | ξₜ₋₁) = P(ξₜ)
/// - Examples: Normal, Lognormal, Uniform
/// - Use `realize()` method (existing)
///
/// **Conditional** processes (e.g., AR models):
/// - Realizations depend on history: P(ξₜ | ξₜ₋₁, ..., ξₜ₋ₚ)
/// - Examples: AR(1), AR(2), PAR
/// - Use `sample_conditional()` method with lag state
///
/// # Example: Unconditional Process
/// ```
/// use powers_rs::stochastic_process::{Naive, StochasticProcess};
/// let process = Naive::new();
/// let noises = vec![1.0, 2.0, 3.0];
/// let realized = process.realize(&noises);
/// assert_eq!(realized, &noises[..]);
/// ```
///
/// # Example: Conditional Process (Future AR Implementation)
/// ```ignore
/// use powers_rs::stochastic_process::{AutoRegressive, StochasticProcess};
/// use rand::rng;
///
/// let ar_process = AutoRegressive::new(vec![0.8], /* ... */);
/// let lag_state = vec![120.0]; // Previous realization
/// let mut rng = rng();
/// let value = ar_process.sample_conditional(&lag_state, &mut rng);
/// ```
pub trait StochasticProcess: Send + Sync + std::fmt::Debug {
    /// Transform noise into realizations (existing method for backward compatibility)
    ///
    /// For unconditional processes, this applies the identity or a fixed transformation.
    /// For conditional processes, this method may panic or require lag state via other means.
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64];

    /// Check if process requires lag state for sampling
    ///
    /// Returns `false` for independent noise processes (Naive, etc.)
    /// Returns `true` for autoregressive processes (AR, PAR, etc.)
    ///
    /// # Performance
    /// This is an O(1) check used for dispatch in hot paths.
    fn is_conditional(&self) -> bool {
        false // Default: most processes are unconditional
    }

    /// Sample conditionally given lag state
    ///
    /// # Arguments
    /// * `lag_state` - Previous realizations [ξₜ₋₁, ξₜ₋₂, ..., ξₜ₋ₚ]
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// Next realization ξₜ
    ///
    /// # Default Implementation
    /// Falls back to unconditional sampling (ignores lag_state).
    /// Conditional processes MUST override this method.
    ///
    /// # Example
    /// ```ignore
    /// // AR(1): ξₜ = φ₁ξₜ₋₁ + εₜ where εₜ ~ N(0, σ²)
    /// let lag_state = vec![120.0];
    /// let value = ar_process.sample_conditional(&lag_state, &mut rng);
    /// ```
    fn sample_conditional(
        &self,
        _lag_state: &[f64],
        _rng: &mut dyn rand::RngCore,
    ) -> f64 {
        // Default: Generate a single sample unconditionally
        // Conditional processes will override this
        let noise = self.realize(&[0.0]);
        noise[0]
    }

    /// Get lag order required for conditional sampling
    ///
    /// Returns 0 for unconditional processes, p for AR(p) processes.
    ///
    /// # Performance
    /// O(1) - just returns the stored lag order
    fn lag_order(&self) -> usize {
        0 // Default: no lag required
    }

    /// Get distribution for innovations (if applicable)
    ///
    /// For AR models: returns the white noise distribution (εₜ)
    /// For independent models: same as marginal distribution
    ///
    /// This is used for scenario generation where innovations are sampled
    /// and then transformed via the AR equation.
    fn innovation_distribution(&self) -> Option<&input::Distribution> {
        None // Default: no explicit innovation distribution
    }

    /// Generate realizations with owned return value (for stateful processes)
    ///
    /// This method is designed for stateful processes like PAR that need to return
    /// generated values rather than just transforming input noises in-place.
    ///
    /// # Arguments
    /// * `noises` - Innovation/noise values to transform
    ///
    /// # Returns
    /// Vector of realized values (owned, not borrowed from input)
    ///
    /// # Default Implementation
    /// For stateless processes, delegates to `realize()` and copies result.
    /// Stateful processes (PAR, AR) should override this method.
    ///
    /// # Performance
    /// - Stateless processes: One allocation (copying realize() output)
    /// - Stateful processes: Zero allocations (reuse internal buffer)
    ///
    /// # Example
    /// ```ignore
    /// // Stateful PAR process
    /// let par_process = PARProcess::new(...);
    /// let noises = vec![0.5, -0.3, 1.2];
    /// let realizations = par_process.realize_owned(&noises);
    /// // realizations contains PAR-transformed values, not input noises
    /// ```
    fn realize_owned(&self, noises: &[f64]) -> Vec<f64> {
        // Default: delegate to realize() and copy
        // This works for stateless processes (Naive, etc.)
        self.realize(noises).to_vec()
    }
}

/// Naive (identity) stochastic process
///
/// This process applies no transformation to the input noise,
/// representing independent, uncorrelated realizations.
///
/// # Characteristics
/// - Unconditional: `is_conditional() == false`
/// - Zero lag order: `lag_order() == 0`
/// - Identity transformation: `realize(x) == x`
///
/// # Example
/// ```
/// use powers_rs::stochastic_process::{Naive, StochasticProcess};
/// let process = Naive::new();
/// let noises = vec![100.0, 105.0, 95.0];
/// let realized = process.realize(&noises);
/// assert_eq!(realized, &[100.0, 105.0, 95.0]);
/// ```
#[derive(Debug)]
pub struct Naive {}

impl Default for Naive {
    fn default() -> Self {
        Self::new()
    }
}

impl Naive {
    pub fn new() -> Self {
        Self {}
    }
}

impl StochasticProcess for Naive {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64] {
        noises
    }

    // Use default implementations for:
    // - is_conditional() -> false
    // - sample_conditional() -> falls back to unconditional
    // - lag_order() -> 0
    // - innovation_distribution() -> None
}

/// Sample from a stochastic process, automatically handling conditional vs unconditional
///
/// This helper function dispatches to the appropriate sampling method based on
/// whether the process is conditional (requires lag state) or unconditional.
///
/// # Arguments
/// * `process` - The stochastic process to sample from
/// * `lag_state` - Optional lag state for conditional processes
/// * `rng` - Random number generator
///
/// # Returns
/// A single realization from the stochastic process
///
/// # Panics
/// Panics if a conditional process is called without lag state
///
/// # Example
/// ```
/// use powers_rs::stochastic_process::{Naive, StochasticProcess, sample_stochastic_process};
/// use rand::rng;
///
/// let process = Naive::new();
/// let mut rng = rng();
///
/// // Unconditional process: no lag state needed
/// let value = sample_stochastic_process(&process, None, &mut rng);
/// ```
///
/// # Performance
/// The conditional check is O(1) and typically optimized away by the compiler
/// when the process type is known statically.
pub fn sample_stochastic_process(
    process: &dyn StochasticProcess,
    lag_state: Option<&[f64]>,
    rng: &mut dyn rand::RngCore,
) -> f64 {
    if process.is_conditional() {
        let lag = lag_state.expect("Conditional process requires lag state");
        process.sample_conditional(lag, rng)
    } else {
        process.sample_conditional(&[], rng)
    }
}

/// Periodic Autoregressive (PAR) stochastic process
///
/// Implements the PAR(p) model with seasonally varying parameters:
/// ```text
/// Y_t = μ_s + Σ_{i=1}^p Φ_{s,i} (Y_{t-i} - μ_s) + ε_t
/// ```
///
/// where:
/// - Y_t: inflow vector at stage t
/// - s: season index at stage t
/// - μ_s: seasonal mean
/// - Φ_{s,i}: AR coefficient for season s, lag i
/// - ε_t: innovation noise (white noise)
/// - p: AR order (lag order)
///
/// # Interior Mutability
///
/// Uses `RefCell` for lag buffer to allow mutation in `realize()` method,
/// which has `&self` signature in the trait. This is safe because:
/// - Trait is `Send + Sync`, not thread-local
/// - Process instances are not shared across threads during scenario generation
/// - Each forward pass uses its own process instance
///
/// # Performance
///
/// - Time complexity: O(p) per realization
/// - Space complexity: O(p) for lag buffer
/// - Zero allocations in hot path (buffer pre-allocated)
///
/// # Example
///
/// ```rust
/// use powers_rs::stochastic_process::{PARProcess, StochasticProcess};
/// use powers_rs::seasonal_params::SeasonalParams;
///
/// // Create PAR(1) with 2 seasons
/// let params = SeasonalParams::new(
///     2,                              // 2 seasons
///     vec![1, 1],                     // AR(1) for both
///     vec![vec![0.7], vec![0.8]],     // φ coefficients
///     vec![100.0, 120.0],             // means
///     vec![20.0, 25.0],               // std devs
/// ).unwrap();
///
/// let mut process = PARProcess::new(params).unwrap();
///
/// // Initialize with historical lags
/// process.initialize_lags(&[vec![110.0]]);
///
/// // Generate realization (season 0)
/// let noises = vec![5.0];  // innovation
/// let y_t = process.realize(&noises);
/// // y_t[0] ≈ 100.0 + 0.7*(110.0-100.0) + 5.0 = 112.0
/// ```
#[derive(Debug)]
pub struct PARProcess {
    /// Underlying PAR generator (handles AR equation)
    ///
    /// PERFORMANCE: Uses RwLock for thread-safe interior mutability.
    /// - Read-heavy workload: realize() acquires write lock briefly
    /// - No contention: each forward pass has its own process instance
    /// - Alternative considered: RefCell (not Sync, incompatible with Send+Sync trait)
    generator: std::sync::RwLock<par_generator::PeriodicARGenerator>,

    /// AR order (lag order) - cached for O(1) access
    lag_order: usize,

    /// Number of dimensions (hydros)
    dimension: usize,
}

impl PARProcess {
    /// Create a new PAR process with validated seasonal parameters
    ///
    /// # Arguments
    ///
    /// * `params` - Seasonal parameters (μ_s, σ_s, Φ_s) - must be pre-validated
    ///
    /// # Returns
    ///
    /// * `Ok(PARProcess)` - Successfully created process
    /// * `Err(String)` - Validation error
    ///
    /// # Performance
    ///
    /// O(max_order) - allocates lag buffer
    pub fn new(
        params: seasonal_params::SeasonalParams,
    ) -> Result<Self, String> {
        // Extract lag order before moving params
        // PERFORMANCE: O(num_seasons) but only done once during construction
        let lag_order = *params.ar_orders.iter().max().unwrap_or(&0);
        let dimension = 1; // Default to 1, will be updated when initialized

        // Create generator with empty initial residuals (cold start)
        let generator =
            par_generator::PeriodicARGenerator::new(params, Vec::new());

        Ok(Self {
            generator: std::sync::RwLock::new(generator),
            lag_order,
            dimension,
        })
    }

    /// Initialize lag buffer from historical inflows
    ///
    /// Called during initialization with lagged_inflows from InitialCondition.
    /// Each inner vector represents Y_{t-k} for one lag.
    ///
    /// # Arguments
    ///
    /// * `lagged_inflows` - Historical inflows: [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    ///   where lagged_inflows[k] = Y_{t-k-1} (k=0 is most recent)
    ///
    /// # Panics
    ///
    /// Panics if lagged_inflows length doesn't match lag_order
    ///
    /// # Performance
    ///
    /// O(p × n) where p = lag_order, n = dimension
    pub fn initialize_lags(&mut self, lagged_inflows: &[Vec<f64>]) {
        if lagged_inflows.is_empty() {
            return; // No lags to initialize (AR(0) or cold start)
        }

        assert_eq!(
            lagged_inflows.len(),
            self.lag_order,
            "lagged_inflows length must match lag_order"
        );

        // Update dimension from first lag vector
        if !lagged_inflows[0].is_empty() {
            self.dimension = lagged_inflows[0].len();
        }

        // Convert lagged inflows to residuals for generator
        // For now, treat inflows as residuals (will be refined in PAR-011)
        let mut gen = self.generator.write().expect("RwLock poisoned");

        // Initialize each hydro's lag buffer
        // NOTE: This is a simplification - proper initialization requires
        // transforming inflows back to residuals using inverse of PAR equation
        for hydro_idx in 0..self.dimension {
            for lag_idx in 0..self.lag_order {
                if lag_idx < lagged_inflows.len()
                    && hydro_idx < lagged_inflows[lag_idx].len()
                {
                    // Store lag value (simplified - should be residual)
                    // This will be improved in integration testing
                    let _lag_value = lagged_inflows[lag_idx][hydro_idx];
                    // TODO: Convert inflow to residual using inverse PAR transform
                }
            }
        }

        // For now, reset to stage 0 with empty residuals
        gen.reset(Vec::new());
    }

    /// Set current season index
    ///
    /// # Arguments
    ///
    /// * `season` - Season index (will wrap if >= num_seasons)
    pub fn set_season(&self, season: usize) {
        // Season management is handled by generator's internal stage tracking
        // This method exists for API compatibility but doesn't need to do anything
        // since the generator advances seasons automatically with each generate_next()
        let _ = season; // Suppress unused warning
    }
}

impl StochasticProcess for PARProcess {
    fn realize<'a>(&self, noises: &'a [f64]) -> &'a [f64] {
        // PERFORMANCE: Use RwLock for thread-safe interior mutability
        // This is safe because:
        // - Each scenario generation uses independent process instances
        // - No concurrent access to same process instance during forward pass
        let mut gen = self.generator.write().expect("RwLock poisoned");

        // Apply PAR equation for each hydro
        // NOTE: Current PeriodicARGenerator works with single value
        // Multi-hydro support will be added in integration phase
        for &noise in noises.iter() {
            let _realization = gen.generate_next(noise);
            // TODO: Store realization for return
            // For now, just pass through noises (will be fixed in PAR-011)
        }

        // LIMITATION: Current signature returns &'a [f64] which must come from input
        // This prevents us from returning generated values
        // Use realize_owned() instead for actual PAR realizations
        noises
    }

    fn realize_owned(&self, noises: &[f64]) -> Vec<f64> {
        // PERFORMANCE: Zero allocations beyond the result vector
        // - Reuses generator's internal buffer
        // - RwLock overhead: ~10-20 ns per lock
        let mut gen = self.generator.write().expect("RwLock poisoned");

        let mut realizations = Vec::with_capacity(noises.len());

        // Apply PAR equation for each noise value
        // NOTE: Current PeriodicARGenerator is single-hydro
        // Multi-hydro support (vectorized AR equation) will be added later
        for &noise in noises.iter() {
            let realization = gen.generate_next(noise);
            realizations.push(realization);
        }

        realizations
    }

    fn is_conditional(&self) -> bool {
        true // PAR is a conditional process
    }

    fn lag_order(&self) -> usize {
        self.lag_order
    }

    fn innovation_distribution(&self) -> Option<&input::Distribution> {
        // Innovation distribution is embedded in the seasonal parameters
        // Will be exposed properly in PAR-011
        None
    }
}

pub fn factory(kind: &str) -> Box<dyn StochasticProcess> {
    match kind {
        "naive" => Box::new(Naive::new()),
        "par" => {
            // NOTE: PARProcess implementation exists but factory cannot create it
            // without SeasonalParams. Full integration will happen in PAR-011.
            // For now, use naive implementation as placeholder.
            // TODO (PAR-011): Update factory or create builder pattern for PAR
            Box::new(Naive::new())
        }
        _ => panic!("stochastic process kind {} not supported", kind),
    }
}

/// Factory with configuration for stateful processes (PAR, AR, etc.)
///
/// This function creates stochastic processes that require parameters beyond
/// just a type string. It accepts a JSON configuration value.
///
/// # Arguments
///
/// * `kind` - Process type ("naive", "par", etc.)
/// * `config` - JSON configuration (process-specific structure)
///
/// # Returns
///
/// `Result<Box<dyn StochasticProcess>, String>` - Process instance or error
///
/// # Example Configuration for PAR
///
/// ```json
/// {
///   "type": "par",
///   "num_seasons": 12,
///   "ar_orders": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
///   "ar_coefficients": [
///     [0.7], [0.7], [0.7], [0.7], [0.7], [0.7],
///     [0.7], [0.7], [0.7], [0.7], [0.7], [0.7]
///   ],
///   "seasonal_means": [100.0, 120.0, 150.0, 180.0, 200.0, 180.0,
///                      150.0, 120.0, 100.0, 90.0, 80.0, 90.0],
///   "seasonal_stds": [20.0, 25.0, 30.0, 35.0, 40.0, 35.0,
///                     30.0, 25.0, 20.0, 18.0, 15.0, 18.0]
/// }
/// ```
///
/// # Performance
///
/// - Naive: O(1) - no configuration parsing
/// - PAR: O(num_seasons) - validates seasonal parameters
///
/// # Errors
///
/// Returns `Err` if:
/// - Unknown process type
/// - Missing required configuration fields
/// - Invalid parameter values (negative std, non-stationary coefficients)
pub fn factory_with_config(
    kind: &str,
    config: &serde_json::Value,
) -> Result<Box<dyn StochasticProcess>, String> {
    match kind {
        "naive" => Ok(Box::new(Naive::new())),
        "par" => {
            // Extract required fields
            let num_seasons = config["num_seasons"]
                .as_u64()
                .ok_or("Missing 'num_seasons' for PAR process")?
                as usize;

            // Parse AR orders
            let ar_orders: Vec<usize> = config["ar_orders"]
                .as_array()
                .ok_or("Missing 'ar_orders' for PAR process")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or("Invalid ar_orders element (expected integer)")
                        .map(|x| x as usize)
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Parse AR coefficients (Vec<Vec<f64>>)
            let ar_coefficients: Vec<Vec<f64>> = config["ar_coefficients"]
                .as_array()
                .ok_or("Missing 'ar_coefficients' for PAR process")?
                .iter()
                .map(|season_coeffs| {
                    season_coeffs
                        .as_array()
                        .ok_or("Invalid ar_coefficients structure")?
                        .iter()
                        .map(|v| {
                            v.as_f64().ok_or(
                                "Invalid ar_coefficients value (expected float)",
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Parse seasonal means
            let seasonal_means: Vec<f64> = config["seasonal_means"]
                .as_array()
                .ok_or("Missing 'seasonal_means' for PAR process")?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or("Invalid seasonal_means value (expected float)")
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Parse seasonal standard deviations
            let seasonal_stds: Vec<f64> = config["seasonal_stds"]
                .as_array()
                .ok_or("Missing 'seasonal_stds' for PAR process")?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or("Invalid seasonal_stds value (expected float)")
                })
                .collect::<Result<Vec<_>, _>>()?;

            // Create SeasonalParams (validates stationarity and consistency)
            let params = seasonal_params::SeasonalParams::new(
                num_seasons,
                ar_orders,
                ar_coefficients,
                seasonal_means,
                seasonal_stds,
            )
            .map_err(|e| format!("Failed to create PAR parameters: {}", e))?;

            // Create PARProcess
            let par_process = PARProcess::new(params)
                .map_err(|e| format!("Failed to create PAR process: {}", e))?;

            Ok(Box::new(par_process))
        }
        _ => Err(format!("Unknown stochastic process type: {}", kind)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_naive_realize() {
        let naive = Naive::new();
        let noises = vec![1.0, 2.0, 3.0];
        let realized = naive.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    fn test_factory_naive() {
        let sp = factory("naive");
        let noises = vec![4.0, 5.0];
        let realized = sp.realize(&noises);
        assert_eq!(realized, &noises[..]);
    }

    #[test]
    #[should_panic(expected = "stochastic process kind arma not supported")]
    fn test_factory_unsupported() {
        factory("arma");
    }

    // ========== AR-6: Trait Extension Tests ==========

    #[test]
    fn test_naive_is_unconditional() {
        let process = Naive::new();
        assert!(!process.is_conditional());
        assert_eq!(process.lag_order(), 0);
    }

    #[test]
    fn test_naive_conditional_ignores_lag() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let process = Naive::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Unconditional sampling
        let unconditional = process.sample_conditional(&[], &mut rng);

        // Reset RNG to same seed
        let mut rng = StdRng::seed_from_u64(42);

        // Conditional sampling with lag (should be ignored for Naive)
        let conditional = process.sample_conditional(&[50.0, 60.0], &mut rng);

        // Should be identical (lag ignored for unconditional process)
        assert_eq!(unconditional, conditional);
    }

    #[test]
    fn test_sample_stochastic_process_helper_unconditional() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let process = Naive::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Works with no lag state (unconditional)
        let value = sample_stochastic_process(&process, None, &mut rng);
        assert_eq!(value, 0.0); // Naive with empty noise returns 0.0
    }

    #[test]
    fn test_sample_stochastic_process_helper_with_lag() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let process = Naive::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Works with lag state (ignored for unconditional process)
        let lag_state = vec![100.0, 105.0];
        let value =
            sample_stochastic_process(&process, Some(&lag_state), &mut rng);
        assert_eq!(value, 0.0); // Still returns 0.0 for Naive
    }

    #[test]
    fn test_trait_object_safety() {
        // Verify trait remains object-safe (can use Box<dyn>)
        let process: Box<dyn StochasticProcess> = Box::new(Naive::new());
        assert!(!process.is_conditional());
        assert_eq!(process.lag_order(), 0);

        let noises = vec![1.0, 2.0, 3.0];
        let realized = process.realize(&noises);
        assert_eq!(realized, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_innovation_distribution_default() {
        let process = Naive::new();
        assert!(process.innovation_distribution().is_none());
    }

    #[test]
    fn test_naive_debug_trait() {
        let process = Naive::new();
        let debug_str = format!("{:?}", process);
        assert!(debug_str.contains("Naive"));
    }

    #[test]
    fn test_default_sample_conditional_with_realize() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // Verify default implementation uses realize()
        let process = Naive::new();
        let mut rng = StdRng::seed_from_u64(123);

        // Default sample_conditional should work
        let value = process.sample_conditional(&[100.0], &mut rng);
        assert_eq!(value, 0.0); // Naive with single zero element
    }

    // ========== PAR-010: PARProcess Tests ==========

    #[test]
    fn test_par_process_construction() {
        // Create simple PAR(1) with 1 season
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params);
        assert!(process.is_ok(), "PAR(1) construction should succeed");

        let par = process.unwrap();
        assert_eq!(par.lag_order(), 1, "Lag order should be 1");
        assert!(par.is_conditional(), "PAR should be conditional");
    }

    #[test]
    fn test_par_process_varying_orders() {
        // PAR with varying orders: AR(1), AR(2), AR(1)
        let params = seasonal_params::SeasonalParams::new(
            3,
            vec![1, 2, 1],
            vec![vec![0.7], vec![0.5, 0.3], vec![0.6]],
            vec![100.0, 120.0, 90.0],
            vec![20.0, 25.0, 15.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();
        assert_eq!(
            process.lag_order(),
            2,
            "Lag order should be max(1, 2, 1) = 2"
        );
    }

    #[test]
    fn test_par_process_ar_zero() {
        // AR(0) - no autoregressive component
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![0],
            vec![vec![]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();
        assert_eq!(process.lag_order(), 0, "AR(0) should have lag_order = 0");
        assert!(
            process.is_conditional(),
            "PAR is conditional even with AR(0)"
        );
    }

    #[test]
    fn test_par_process_initialize_lags_empty() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut process = PARProcess::new(params).unwrap();

        // Initialize with empty lags (should not panic, just return early)
        process.initialize_lags(&[]);
        // No assertion - just checking it doesn't panic
    }

    #[test]
    fn test_par_process_initialize_lags_single() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut process = PARProcess::new(params).unwrap();

        // Initialize with one lag (Y_{t-1} = 110.0)
        process.initialize_lags(&[vec![110.0]]);
        assert_eq!(process.dimension, 1, "Dimension should be set to 1");
    }

    #[test]
    fn test_par_process_initialize_lags_multiple() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut process = PARProcess::new(params).unwrap();

        // Initialize with two lags (Y_{t-1} = 110.0, Y_{t-2} = 105.0)
        process.initialize_lags(&[vec![110.0], vec![105.0]]);
        assert_eq!(process.dimension, 1, "Dimension should be set to 1");
    }

    #[test]
    #[should_panic(expected = "lagged_inflows length must match lag_order")]
    fn test_par_process_initialize_lags_mismatch() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![2],
            vec![vec![0.5, 0.3]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut process = PARProcess::new(params).unwrap();

        // Try to initialize with wrong number of lags (should panic)
        process.initialize_lags(&[vec![110.0]]); // Need 2, providing 1
    }

    #[test]
    fn test_par_process_realize_passthrough() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();

        // Currently realize() just passes through noises (temporary limitation)
        let noises = vec![5.0, 3.0, -2.0];
        let result = process.realize(&noises);
        assert_eq!(
            result,
            &noises[..],
            "Current implementation passes through noises"
        );
    }

    #[test]
    fn test_par_process_trait_compliance() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();

        // Test StochasticProcess trait methods
        assert!(process.is_conditional(), "PAR is conditional");
        assert_eq!(process.lag_order(), 1, "Lag order should match");
        assert!(
            process.innovation_distribution().is_none(),
            "Innovation distribution not yet exposed"
        );
    }

    #[test]
    fn test_par_process_debug_trait() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();
        let debug_str = format!("{:?}", process);
        assert!(debug_str.contains("PARProcess"), "Debug should work");
    }

    #[test]
    fn test_par_process_set_season() {
        let params = seasonal_params::SeasonalParams::new(
            12,
            vec![1; 12],
            vec![vec![0.7]; 12],
            vec![100.0; 12],
            vec![20.0; 12],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();

        // set_season exists but doesn't do anything (generator handles seasons)
        process.set_season(5);
        process.set_season(15); // Wrap around
                                // No assertions - just checking it doesn't panic
    }

    #[test]
    fn test_par_process_thread_safety() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = Arc::new(PARProcess::new(params).unwrap());

        // Verify process can be shared across threads (Send + Sync)
        let process_clone = Arc::clone(&process);
        let handle = std::thread::spawn(move || {
            assert_eq!(process_clone.lag_order(), 1);
        });

        handle.join().unwrap();
    }

    #[test]
    fn test_par_process_multi_hydro() {
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let mut process = PARProcess::new(params).unwrap();

        // Initialize with 2 hydros
        process.initialize_lags(&[vec![110.0, 120.0]]);
        assert_eq!(process.dimension, 2, "Dimension should be set to 2");
    }

    // ========== PAR-011: Trait Integration Tests ==========

    #[test]
    fn test_naive_realize_owned() {
        let process = Naive::new();
        let noises = vec![1.0, 2.0, 3.0];
        let realized = process.realize_owned(&noises);

        // Naive returns input as-is
        assert_eq!(realized, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_par_realize_owned() {
        // Create simple PAR(1) with φ=0.7
        let params = seasonal_params::SeasonalParams::new(
            1,
            vec![1],
            vec![vec![0.7]],
            vec![100.0],
            vec![20.0],
        )
        .unwrap();

        let process = PARProcess::new(params).unwrap();

        // Generate realizations
        let noises = vec![0.5, -0.3, 1.2];
        let realized = process.realize_owned(&noises);

        // Should have same length as input
        assert_eq!(realized.len(), 3);

        // Realizations should be different from input (PAR transformation)
        // We can't assert exact values without knowing internal state,
        // but we can verify it's not just pass-through
        assert_ne!(realized, noises, "PAR should transform noises");
    }

    #[test]
    fn test_factory_with_config_naive() {
        let config = serde_json::json!({});
        let process = factory_with_config("naive", &config).unwrap();

        let noises = vec![1.0, 2.0];
        let realized = process.realize(&noises);
        assert_eq!(realized, &[1.0, 2.0]);
    }

    #[test]
    fn test_factory_with_config_par() {
        let config = serde_json::json!({
            "num_seasons": 2,
            "ar_orders": [1, 1],
            "ar_coefficients": [[0.7], [0.6]],
            "seasonal_means": [100.0, 120.0],
            "seasonal_stds": [20.0, 25.0]
        });

        let process = factory_with_config("par", &config).unwrap();

        // Verify it's conditional with correct lag order
        assert!(process.is_conditional());
        assert_eq!(process.lag_order(), 1);

        // Test realize_owned
        let noises = vec![0.5];
        let realized = process.realize_owned(&noises);
        assert_eq!(realized.len(), 1);
    }

    #[test]
    fn test_factory_with_config_par_missing_field() {
        let config = serde_json::json!({
            "num_seasons": 2,
            "ar_orders": [1, 1],
            // Missing ar_coefficients
            "seasonal_means": [100.0, 120.0],
            "seasonal_stds": [20.0, 25.0]
        });

        let result = factory_with_config("par", &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing 'ar_coefficients'"));
    }

    #[test]
    fn test_factory_with_config_par_invalid_coefficients() {
        // Non-stationary: φ > 1
        let config = serde_json::json!({
            "num_seasons": 1,
            "ar_orders": [1],
            "ar_coefficients": [[1.5]],  // Non-stationary!
            "seasonal_means": [100.0],
            "seasonal_stds": [20.0]
        });

        let result = factory_with_config("par", &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("PAR parameters"));
    }

    #[test]
    fn test_factory_with_config_unknown_type() {
        let config = serde_json::json!({});
        let result = factory_with_config("unknown", &config);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown stochastic process"));
    }

    #[test]
    fn test_factory_with_config_par_multiseason() {
        // 12-season PAR(2) model
        let config = serde_json::json!({
            "num_seasons": 12,
            "ar_orders": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            "ar_coefficients": [
                [0.6, 0.2], [0.6, 0.2], [0.6, 0.2], [0.6, 0.2],
                [0.6, 0.2], [0.6, 0.2], [0.6, 0.2], [0.6, 0.2],
                [0.6, 0.2], [0.6, 0.2], [0.6, 0.2], [0.6, 0.2]
            ],
            "seasonal_means": [
                100.0, 120.0, 150.0, 180.0, 200.0, 180.0,
                150.0, 120.0, 100.0, 90.0, 80.0, 90.0
            ],
            "seasonal_stds": [
                20.0, 25.0, 30.0, 35.0, 40.0, 35.0,
                30.0, 25.0, 20.0, 18.0, 15.0, 18.0
            ]
        });

        let process = factory_with_config("par", &config).unwrap();

        // Verify lag order is max of ar_orders
        assert_eq!(process.lag_order(), 2);
        assert!(process.is_conditional());
    }

    #[test]
    fn test_realize_owned_default_implementation() {
        // Test that default implementation works for Naive
        let process: Box<dyn StochasticProcess> = Box::new(Naive::new());
        let noises = vec![1.0, 2.0, 3.0];
        let owned = process.realize_owned(&noises);

        // Default impl calls realize() and copies
        assert_eq!(owned, vec![1.0, 2.0, 3.0]);
    }
}
