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
    algorithm: SddpAlgorithm,
    config: Config,
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
    /// `Ok(TrainingResult)` on success, containing iteration history.
    /// `Err(...)` if solver fails or other errors occur.
    ///
    pub fn train(&mut self) -> Result<TrainingResult, String> {
        
        crate::utils::configure_thread_pool(self.config.num_threads)
                .map_err(|e| {
                    format!("Thread pool configuration failed: {}", e)
                })?;

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
    /// Uses the Extract-and-Release pattern: handlers are allocated per-thread (lazy),
    /// reused across scenarios, and automatically released when threads complete.
    ///
    /// # Returns
    ///
    /// - `Ok(Vec<SimulationTrajectory>)` - Simulation trajectories
    /// - `Err(String)` - If simulation is not configured or fails
    ///
    pub fn simulate(&mut self) -> Result<Vec<SimulationTrajectory>, String> {
        let num_scenarios = self.config.num_simulation_scenarios
            .ok_or_else(|| "Simulation not configured: set it to a positive integer".to_string())?;

        crate::utils::configure_thread_pool(self.config.num_threads)
                .map_err(|e| {
                    format!("Thread pool configuration failed: {}", e)
                })?;

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
    pub fn algorithm(&self) -> &SddpAlgorithm {
        &self.algorithm
    }

    /// Mutable reference to the underlying SDDP algorithm.
    ///
    /// Provides mutable access for advanced use cases where you need to
    /// modify the algorithm state directly (e.g., custom cut management,
    /// warm-starting from previous runs).
    ///
    pub fn algorithm_mut(&mut self) -> &mut SddpAlgorithm {
        &mut self.algorithm
    }

    /// Immutable reference to the configuration.
    /// 
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Immutable reference to the SAA scenarios.
    ///
    /// Access the pre-sampled noise realizations used for training and simulation.
    ///
    pub fn saa(&self) -> &SAA {
        &self.saa
    }
}
