// Utility module for encapsulating SDDP algorithm with its configuration

use crate::input::Config;
use crate::scenario::SAA;
use crate::sddp::{SddpAlgorithm, SimulationTrajectory, TrainingResult};

/// Wrapper for SDDP algorithm with embedded configuration and scenarios.
pub struct SddpInstance {
    algorithm: SddpAlgorithm,
    config: Config,
    saa: SAA,
}

impl SddpInstance {
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
    pub fn train(&mut self) -> Result<TrainingResult, String> {
        crate::utils::configure_thread_pool(self.config.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        self.algorithm.train(
            self.config.num_iterations,
            self.config.num_forward_passes,
            &self.saa,
        )
    }

    /// Simulate the trained policy using the embedded configuration and SAA.
    pub fn simulate(&mut self) -> Result<Vec<SimulationTrajectory>, String> {
        let num_scenarios =
            self.config.num_simulation_scenarios.ok_or_else(|| {
                "Simulation not configured: set it to a positive integer"
                    .to_string()
            })?;

        crate::utils::configure_thread_pool(self.config.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        self.algorithm.simulate(num_scenarios, &self.saa)
    }

    /// Immutable reference to the underlying SDDP algorithm.
    pub fn algorithm(&self) -> &SddpAlgorithm {
        &self.algorithm
    }

    /// Mutable reference to the underlying SDDP algorithm.
    pub fn algorithm_mut(&mut self) -> &mut SddpAlgorithm {
        &mut self.algorithm
    }

    /// Immutable reference to the configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Immutable reference to the SAA scenarios.
    pub fn saa(&self) -> &SAA {
        &self.saa
    }
}
