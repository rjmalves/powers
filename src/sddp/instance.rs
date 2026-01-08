// Utility module for encapsulating SDDP algorithm with its configuration

use crate::input::Config;
use crate::scenario::ScenarioTree;
use crate::sddp::{SddpAlgorithm, SimulationTrajectory, TrainingResult};

/// Wrapper for SDDP algorithm with embedded configuration and scenarios.
pub struct SddpInstance {
    algorithm: SddpAlgorithm,
    config: Config,
    saa: ScenarioTree,
}

impl SddpInstance {
    pub(crate) fn new(
        algorithm: SddpAlgorithm,
        config: Config,
        saa: ScenarioTree,
    ) -> Self {
        Self {
            algorithm,
            config,
            saa,
        }
    }

    /// Train the SDDP algorithm using the embedded configuration and ScenarioTree.
    pub fn train(&mut self) -> Result<TrainingResult, String> {
        crate::utils::configure_thread_pool(self.config.general.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        self.algorithm.train(
            self.config.training.num_iterations,
            self.config.training.num_forward_passes,
            self.config.training.enable_cut_selection,
            &self.saa,
            self.config.output.export_forward_detail,
            self.config.output.export_backward_detail,
            None, // No callback for basic train()
        )
    }

    /// Train with integrated display output.
    pub fn train_with_display(
        &mut self,
        renderer: &dyn crate::display::DisplayRenderer,
        display_config: &crate::display::DisplayConfig,
    ) -> Result<TrainingResult, String> {
        use crate::display::{DisplayContext, IterationTracker};
        use std::io::Write;

        crate::utils::configure_thread_pool(self.config.general.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        // Print header
        let header = renderer.render_header(
            display_config,
            self.config.training.num_iterations,
            self.config.training.num_forward_passes,
            self.config.training.enable_cut_selection,
        );
        print!("{}", header);
        std::io::stdout().flush().ok();

        // Create iteration tracker
        let mut tracker = IterationTracker::new();
        tracker.start();

        // Create callback for real-time display
        let mut callback = |iteration_result: &crate::sddp::IterationResult| {
            // Build DisplayContext
            let ctx = DisplayContext::from_iteration(
                iteration_result.iteration,
                self.config.training.num_iterations,
                iteration_result,
                tracker.previous_lower_bound(),
                tracker.elapsed(),
                display_config.target_gap,
            );

            // Render if should print
            if ctx.should_print {
                let output = renderer.render_iteration(&ctx, display_config);
                print!("{}", output);
                std::io::stdout().flush().ok();
            }

            // Update tracker
            tracker.update(iteration_result.lower_bound, ctx.gap_percent);
        };

        // Train with callback
        let result = self.algorithm.train(
            self.config.training.num_iterations,
            self.config.training.num_forward_passes,
            self.config.training.enable_cut_selection,
            &self.saa,
            self.config.output.export_forward_detail,
            self.config.output.export_backward_detail,
            Some(&mut callback),
        )?;

        // Training summary
        let summary = renderer.render_training_summary(&result, display_config);
        print!("{}", summary);
        std::io::stdout().flush().ok();

        Ok(result)
    }

    /// Simulate the trained policy using the embedded configuration and ScenarioTree.
    pub fn simulate(&mut self) -> Result<Vec<SimulationTrajectory>, String> {
        let num_scenarios =
            self.config.simulation.num_scenarios.ok_or_else(|| {
                "Simulation not configured: set simulation.num_scenarios to a positive integer"
                    .to_string()
            })?;

        crate::utils::configure_thread_pool(self.config.general.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        self.algorithm.simulate(num_scenarios, &self.saa)
    }

    /// Simulate with integrated display output.
    pub fn simulate_with_display(
        &mut self,
        renderer: &dyn crate::display::DisplayRenderer,
        display_config: &crate::display::DisplayConfig,
    ) -> Result<Vec<SimulationTrajectory>, String> {
        use std::io::Write;
        use std::time::Instant;

        let num_scenarios =
            self.config.simulation.num_scenarios.ok_or_else(|| {
                "Simulation not configured: set simulation.num_scenarios to a positive integer"
                    .to_string()
            })?;

        crate::utils::configure_thread_pool(self.config.general.num_threads)
            .map_err(|e| format!("Thread pool configuration failed: {}", e))?;

        // Render simulation start
        let start_output =
            renderer.render_simulation_start(num_scenarios, display_config);
        print!("{}", start_output);
        std::io::stdout().flush().ok();

        // Run simulation
        let begin = Instant::now();
        let trajectories = self.algorithm.simulate(num_scenarios, &self.saa)?;
        let elapsed = begin.elapsed();

        // Render simulation summary
        let summary = renderer.render_simulation_summary(
            &trajectories,
            elapsed,
            display_config,
        );
        print!("{}", summary);
        std::io::stdout().flush().ok();

        Ok(trajectories)
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

    /// Immutable reference to the ScenarioTree scenarios.
    pub fn saa(&self) -> &ScenarioTree {
        &self.saa
    }

    /// Get the system from the first node's data.
    ///
    /// Returns a reference to the power system configuration.
    pub fn system(&self) -> &crate::system::System {
        // Get any node's system (they're all the same)
        &self
            .algorithm
            .node_data_graph
            .iter_nodes()
            .next()
            .expect("Graph must have at least one node")
            .data
            .system
    }

    /// Get the maximum AR order across all temporal models.
    ///
    /// Returns 0 if there are no AR models or all have order 0.
    pub fn max_ar_order(&self) -> usize {
        // Get any node's uncertainty models (they're all the same)
        self.algorithm
            .node_data_graph
            .iter_nodes()
            .next()
            .expect("Graph must have at least one node")
            .data
            .uncertainty_models
            .iter()
            .map(|model| model.max_ar_order)
            .max()
            .unwrap_or(0)
    }

    /// Get AR orders for each hydro entity.
    ///
    /// Returns a vector where index i contains the AR order for hydro i.
    /// Returns 0 for hydros without AR models.
    ///
    /// # Returns
    ///
    /// Vector of AR orders indexed by hydro ID
    pub fn hydro_ar_orders(&self) -> Vec<usize> {
        let num_hydros = self.system().hydros.len();
        let mut ar_orders = vec![0; num_hydros];

        // Get any node's uncertainty models (they're all the same)
        if let Some(node) = self.algorithm.node_data_graph.iter_nodes().next() {
            for model in node.data.uncertainty_models.iter() {
                // Only process inflow models (hydro uncertainties)
                if model.entity_type == crate::input::UncertaintyType::Inflow
                    && model.entity_id < num_hydros
                {
                    ar_orders[model.entity_id] = model.max_ar_order;
                }
            }
        }

        ar_orders
    }
}
