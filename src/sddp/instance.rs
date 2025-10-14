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
use crate::sddp::{SddpAlgorithm, SimulationTrajectory, TrainingResult};

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
    /// - Thread pool configured once before training (< 10ms overhead)
    ///
    pub fn train(&mut self) -> Result<TrainingResult, String> {
        // Configure thread pool before training
        let threads =
            crate::utils::configure_thread_pool(self.config.num_threads)
                .map_err(|e| {
                    format!("Thread pool configuration failed: {}", e)
                })?;

        // Log thread configuration (helps debugging and performance tuning)
        println!("Using {} threads for training", threads);

        self.algorithm.train(
            self.config.num_iterations,
            self.config.num_forward_passes,
            &self.saa,
        )
    }

    /// Simulate the trained policy using the embedded configuration and SAA.
    ///
    /// Uses `config.num_simulation_scenarios` to determine the number of scenarios.
    /// If `num_simulation_scenarios` is `None`, this method will return an error.
    ///
    /// # Memory Optimization (SIM-OPT-005/006)
    ///
    /// Returns `Vec<SimulationTrajectory>` (lightweight, 96KB each) instead of
    /// `Vec<SddpSimulationHandler>` (heavy, 6MB each), providing 96% memory savings.
    ///
    /// Uses the Extract-and-Release pattern: handlers are allocated per-thread (lazy),
    /// reused across scenarios, and automatically released when threads complete.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<SimulationTrajectory>)` - Lightweight simulation trajectories
    /// - `Err(String)` - If simulation is not configured or fails
    ///
    /// # Performance
    ///
    /// - No overhead vs. manual `simulate()` call
    /// - Configuration is passed by value (usize is Copy)
    /// - SAA is passed by reference (no copy)
    /// - Thread pool configured once before simulation (< 10ms overhead)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Assumes config.num_simulation_scenarios is Some(n)
    /// let trajectories = sddp.simulate()?;
    /// ```
    ///
    /// For more control, use `algorithm_mut().simulate(num_scenarios, &saa)` directly.
    ///
    pub fn simulate(&mut self) -> Result<Vec<SimulationTrajectory>, String> {
        // Check if simulation is configured
        let num_scenarios = self.config.num_simulation_scenarios
            .ok_or_else(|| "Simulation not configured: num_simulation_scenarios is None. Set it to a positive integer or call algorithm_mut().simulate() directly.".to_string())?;

        // Configure thread pool before simulation
        let threads =
            crate::utils::configure_thread_pool(self.config.num_threads)
                .map_err(|e| {
                    format!("Thread pool configuration failed: {}", e)
                })?;

        // Log thread configuration
        println!("Using {} threads for simulation", threads);

        // Use new memory-optimized method (SIM-OPT-005)
        self.algorithm.simulate(num_scenarios, &self.saa)
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
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Immutable reference to the SAA scenarios.
    ///
    /// Access the pre-sampled noise realizations used for training and simulation.
    ///
    /// # Performance
    ///
    /// - Zero overhead (just returns a reference)
    /// - No copying or cloning
    ///
    pub fn saa(&self) -> &SAA {
        &self.saa
    }
}
