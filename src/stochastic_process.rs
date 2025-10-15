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
    fn innovation_distribution(&self) -> Option<&crate::input::Distribution> {
        None // Default: no explicit innovation distribution
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

pub fn factory(kind: &str) -> Box<dyn StochasticProcess> {
    match kind {
        "naive" => Box::new(Naive::new()),
        "par" => {
            // For now, use naive implementation for PAR to get the example running
            // TODO: Implement proper PAR stochastic process with conditional sampling
            Box::new(Naive::new())
        },
        _ => panic!("stochastic process kind {} not supported", kind),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
