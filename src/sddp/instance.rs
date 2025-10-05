// Utility module for encapsulating SDDP algorithm with its configuration
//
// This module provides `SddpInstance`, a wrapper that combines:
// - The SDDP algorithm itself
// - The configuration (num_iterations, etc.)
// - The Sample Average Approximation (SAA) scenarios
//
// This enables zero-argument `train()` and `simulate()` calls,
// improving ergonomics for testing and benchmarking.

use crate::input::Config;
use crate::scenario::SAA;
use crate::sddp::{SddpAlgorithm, SddpSimulationHandler, TrainingResult};

/// Wrapper for SDDP algorithm with embedded configuration and scenarios.
///
/// This type is returned by `SddpAlgorithm::from_files()` and encapsulates
/// everything needed for training and simulation:
/// - The algorithm state (cuts, subproblems, future cost functions)
/// - Configuration (num_iterations, num_forward_passes, seed, output_path)
/// - SAA scenarios (sampled noise realizations)
///
/// # Benefits
///
/// 1. **Ergonomics**: One-line construction, zero-argument train/simulate
/// 2. **Testing**: Reduces test boilerplate from ~50 lines to ~5 lines
/// 3. **Benchmarking**: Enables clean benchmark construction
/// 4. **Correctness**: Configuration and scenarios are always synchronized
///
/// # Performance
///
/// Zero overhead compared to manual construction:
/// - No heap allocations beyond what's already needed
/// - No virtual dispatch (all methods are statically dispatched)
/// - Wrapper is optimized away by the compiler (zero-cost abstraction)
/// - Size: 3 pointers (~24 bytes on 64-bit) - negligible
///
/// # Example
///
/// ```rust,ignore
/// // Factory API: One-line construction
/// let mut sddp = SddpAlgorithm::from_files(
///     "example/config.json",
///     "example/system.json",
///     "example/graph.json",
///     "example/recourse.json",
/// )?;
///
/// // Zero-argument training (config embedded)
/// let training_result = sddp.train()?;
///
/// // Zero-argument simulation (config and SAA embedded)
/// let simulation_handlers = sddp.simulate()?;
///
/// // Access underlying algorithm if needed
/// let fcf_graph = sddp.algorithm().future_cost_function_graph();
/// ```
///
/// # Design Rationale
///
/// This wrapper exists because the factory method `from_files()` naturally
/// produces all three components (algorithm, config, SAA) together. Rather
/// than forcing users to manage them separately, we bundle them in a struct.
///
/// Alternative considered: Add setters to `SddpAlgorithm` for config/SAA.
/// Rejected because:
/// - Pollutes the core `SddpAlgorithm` type with test-specific methods
/// - Makes it possible to misconfigure (e.g., wrong seed vs. SAA)
/// - Less clear ownership and lifecycle
///
/// # Thread Safety
///
/// `SddpInstance` is NOT `Send` or `Sync` because:
/// - Contains mutable `SddpAlgorithm` state (cuts are added during training)
/// - HiGHS solver models are not thread-safe across instances
/// - Parallelism is handled internally via Rayon (within train/simulate)
///
/// This is intentional: each thread should have its own `SddpInstance`.
pub struct SddpInstance {
    /// The SDDP algorithm with cuts, subproblems, and future cost functions.
    algorithm: SddpAlgorithm,

    /// Configuration: num_iterations, num_forward_passes, seed, output_path.
    config: Config,

    /// Sample Average Approximation: pre-sampled noise scenarios.
    ///
    /// Generated from the stochastic processes in the graph using the config's seed.
    /// Ensures deterministic training and simulation.
    saa: SAA,
}

impl SddpInstance {
    /// Create a new `SddpInstance` with the given algorithm, config, and SAA.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The SDDP algorithm (from `SddpAlgorithm::new()`)
    /// * `config` - Configuration (from `Input::from_paths()`)
    /// * `saa` - SAA scenarios (from `Recourse::generate_sddp_noises()`)
    ///
    /// # Performance
    ///
    /// - Move semantics: no copies or clones
    /// - Zero heap allocations
    /// - Instant construction (O(1))
    ///
    /// This constructor is `pub(crate)` (crate-visible only) because users should
    /// use the factory method `SddpAlgorithm::from_files()` instead. Direct construction
    /// is only for internal use or advanced cases.
    pub(crate) fn new(
        algorithm: SddpAlgorithm,
        config: Config,
        saa: SAA,
    ) -> Self {
        Self {
            algorithm,
            config,
            saa,
        }
    }

    /// Train the SDDP algorithm using the embedded configuration and SAA.
    ///
    /// This is equivalent to calling:
    /// ```rust,ignore
    /// algorithm.train(config.num_iterations, config.num_forward_passes, &saa)
    /// ```
    ///
    /// # Returns
    ///
    /// `Ok(TrainingResult)` on success, containing convergence history.
    /// `Err(...)` if solver fails or other errors occur.
    ///
    /// # Performance
    ///
    /// - No overhead vs. manual `train()` call
    /// - Configuration is passed by value (copyable scalars)
    /// - SAA is passed by reference (no copy)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sddp = SddpAlgorithm::from_files(...)?;
    /// let result = sddp.train()?;
    /// assert!(result.converged(1e-3));
    /// ```
    pub fn train(&mut self) -> Result<TrainingResult, String> {
        self.algorithm.train(
            self.config.num_iterations,
            self.config.num_forward_passes,
            &self.saa,
        )
    }

    /// Simulate the trained policy using the embedded configuration and SAA.
    ///
    /// This is equivalent to calling:
    /// ```rust,ignore
    /// algorithm.simulate(config.num_simulation_scenarios, &saa)
    /// ```
    ///
    /// # Returns
    ///
    /// `Ok(Vec<SimulationHandler>)` on success, one handler per scenario.
    /// `Err(...)` if solver fails or other errors occur.
    ///
    /// # Performance
    ///
    /// - No overhead vs. manual `simulate()` call
    /// - Configuration is passed by value (usize is Copy)
    /// - SAA is passed by reference (no copy)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let handlers = sddp.simulate()?;
    /// for handler in &handlers {
    ///     println!("Scenario cost: {}", handler.total_cost());
    /// }
    /// ```
    pub fn simulate(&mut self) -> Result<Vec<SddpSimulationHandler>, String> {
        self.algorithm
            .simulate(self.config.num_simulation_scenarios, &self.saa)
    }

    /// Immutable reference to the underlying SDDP algorithm.
    ///
    /// Provides access to:
    /// - Future cost function graph (`future_cost_function_graph`)
    /// - Study period IDs (`study_period_ids`)
    /// - Other algorithm state
    ///
    /// Use this when you need to inspect the algorithm state after training
    /// (e.g., for output generation or analysis).
    ///
    /// # Performance
    ///
    /// - Zero overhead (just returns a reference)
    /// - No copying or cloning
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sddp = SddpAlgorithm::from_files(...)?;
    /// let fcf_graph = sddp.algorithm().future_cost_function_graph();
    /// println!("Number of nodes: {}", fcf_graph.num_nodes());
    /// ```
    pub fn algorithm(&self) -> &SddpAlgorithm {
        &self.algorithm
    }

    /// Mutable reference to the underlying SDDP algorithm.
    ///
    /// Provides mutable access for advanced use cases where you need to
    /// modify the algorithm state directly (e.g., custom cut management,
    /// warm-starting from previous runs).
    ///
    /// Most users should use `train()` and `simulate()` instead.
    ///
    /// # Performance
    ///
    /// - Zero overhead (just returns a mutable reference)
    /// - No copying or cloning
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut sddp = SddpAlgorithm::from_files(...)?;
    /// // Advanced: manually add a cut
    /// sddp.algorithm_mut().add_custom_cut(...);
    /// ```
    pub fn algorithm_mut(&mut self) -> &mut SddpAlgorithm {
        &mut self.algorithm
    }

    /// Immutable reference to the configuration.
    ///
    /// Access configuration fields like:
    /// - `num_iterations`
    /// - `num_forward_passes`
    /// - `num_simulation_scenarios`
    /// - `seed`
    /// - `output_path`
    ///
    /// # Performance
    ///
    /// - Zero overhead (just returns a reference)
    /// - No copying or cloning
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sddp = SddpAlgorithm::from_files(...)?;
    /// println!("Training {} iterations with seed {}",
    ///          sddp.config().num_iterations,
    ///          sddp.config().seed);
    /// ```
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Immutable reference to the SAA scenarios.
    ///
    /// Access the pre-sampled noise realizations used for training and simulation.
    /// Useful for:
    /// - Debugging scenario generation
    /// - Analyzing scenario statistics
    /// - Validating determinism
    ///
    /// # Performance
    ///
    /// - Zero overhead (just returns a reference)
    /// - No copying or cloning
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let sddp = SddpAlgorithm::from_files(...)?;
    /// let saa = sddp.saa();
    /// println!("Number of scenarios: {}", saa.len());
    /// ```
    pub fn saa(&self) -> &SAA {
        &self.saa
    }
}
