//! Display renderer trait and common utilities.
//!
//! The `DisplayRenderer` trait provides a common interface for all display
//! profile implementations, enabling polymorphic rendering.

use crate::display::config::DisplayConfig;
use crate::display::context::DisplayContext;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

/// Trait for rendering display output.
///
/// Implementors produce formatted strings for different phases of execution.
/// The trait is object-safe to allow dynamic dispatch based on profile selection.
///
/// # Thread Safety
///
/// Renderers must be `Send + Sync` to support potential future async logging.
pub trait DisplayRenderer: Send + Sync {
    /// Render the header displayed at program start.
    ///
    /// Includes program name, version, and configuration summary.
    ///
    /// # Arguments
    ///
    /// * `config` - Display configuration
    /// * `iterations` - Total planned iterations
    /// * `forward_passes` - Forward passes per iteration
    /// * `cut_selection` - Whether cut selection is enabled
    ///
    /// # Returns
    ///
    /// Formatted string ready for output.
    fn render_header(
        &self,
        config: &DisplayConfig,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) -> String;

    /// Render the table header row (if applicable).
    ///
    /// For table-based renderers, this produces column headers.
    /// For non-table renderers (JSON), this may return empty string.
    fn render_table_header(&self, config: &DisplayConfig) -> String;

    /// Render a single iteration's output.
    ///
    /// This is called after each training iteration completes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Complete iteration context with all metrics
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted string for this iteration. May be multi-line.
    fn render_iteration(
        &self,
        ctx: &DisplayContext,
        config: &DisplayConfig,
    ) -> String;

    /// Render the training completion summary.
    ///
    /// Called once after all iterations complete.
    ///
    /// # Arguments
    ///
    /// * `result` - Complete training result with statistics
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted summary string.
    fn render_training_summary(
        &self,
        result: &TrainingResult,
        config: &DisplayConfig,
    ) -> String;

    /// Render the simulation start message.
    ///
    /// Called before simulation begins.
    ///
    /// # Arguments
    ///
    /// * `num_scenarios` - Number of simulation scenarios
    /// * `config` - Display configuration
    fn render_simulation_start(
        &self,
        num_scenarios: usize,
        config: &DisplayConfig,
    ) -> String;

    /// Render the simulation completion summary.
    ///
    /// Called after all simulation scenarios complete.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - All simulation trajectories
    /// * `elapsed` - Simulation duration
    /// * `config` - Display configuration
    ///
    /// # Returns
    ///
    /// Formatted summary with cost statistics.
    fn render_simulation_summary(
        &self,
        trajectories: &[SimulationTrajectory],
        elapsed: Duration,
        config: &DisplayConfig,
    ) -> String;

    /// Render an error message.
    ///
    /// For styled renderers, this adds error formatting (color, icon).
    ///
    /// # Arguments
    ///
    /// * `message` - Error message text
    /// * `config` - Display configuration
    fn render_error(&self, message: &str, config: &DisplayConfig) -> String;

    /// Render a warning message.
    ///
    /// For styled renderers, this adds warning formatting.
    fn render_warning(&self, message: &str, config: &DisplayConfig) -> String;

    /// Get the display profile this renderer implements.
    fn profile(&self) -> super::config::DisplayProfile;

    /// Whether this renderer uses colors.
    ///
    /// Used for testing and documentation.
    fn uses_color(&self) -> bool;
}

/// Create a renderer for the given profile.
///
/// # Arguments
///
/// * `config` - Display configuration for initialization
///
/// # Returns
///
/// Boxed trait object implementing the selected profile.
pub fn create_renderer(config: &DisplayConfig) -> Box<dyn DisplayRenderer> {
    use super::config::DisplayProfile;
    use super::renderers::{
        AdvancedRenderer, AutomationRenderer, MinimalRenderer, StandardRenderer,
    };

    match config.profile {
        DisplayProfile::Advanced => Box::new(AdvancedRenderer::new()),
        DisplayProfile::Standard => Box::new(StandardRenderer::new()),
        DisplayProfile::Minimal => Box::new(MinimalRenderer::new()),
        DisplayProfile::Automation => Box::new(AutomationRenderer::new()),
    }
}

/// High-level display manager.
///
/// Wraps a renderer and provides convenient methods for the SDDP algorithm.
pub struct DisplayManager {
    renderer: Box<dyn DisplayRenderer>,
    config: DisplayConfig,
}

impl DisplayManager {
    /// Create a new display manager with the given configuration.
    pub fn new(config: DisplayConfig) -> Self {
        let renderer = create_renderer(&config);
        Self { renderer, config }
    }

    /// Output iteration display.
    ///
    /// Checks `should_print` flag before rendering.
    pub fn iteration(&self, ctx: &DisplayContext) {
        if ctx.should_print {
            let output = self.renderer.render_iteration(ctx, &self.config);
            print!("{}", output);
        }
    }

    /// Output header.
    pub fn header(
        &self,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) {
        let output = self.renderer.render_header(
            &self.config,
            iterations,
            forward_passes,
            cut_selection,
        );
        print!("{}", output);
    }

    /// Output table header.
    pub fn table_header(&self) {
        let output = self.renderer.render_table_header(&self.config);
        if !output.is_empty() {
            print!("{}", output);
        }
    }

    /// Output training summary.
    pub fn training_summary(&self, result: &TrainingResult) {
        let output =
            self.renderer.render_training_summary(result, &self.config);
        print!("{}", output);
    }

    /// Output simulation start.
    pub fn simulation_start(&self, num_scenarios: usize) {
        let output = self
            .renderer
            .render_simulation_start(num_scenarios, &self.config);
        print!("{}", output);
    }

    /// Output simulation summary.
    pub fn simulation_summary(
        &self,
        trajectories: &[SimulationTrajectory],
        elapsed: Duration,
    ) {
        let output = self.renderer.render_simulation_summary(
            trajectories,
            elapsed,
            &self.config,
        );
        print!("{}", output);
    }

    /// Output error message.
    pub fn error(&self, message: &str) {
        let output = self.renderer.render_error(message, &self.config);
        eprint!("{}", output);
    }

    /// Output warning message.
    pub fn warning(&self, message: &str) {
        let output = self.renderer.render_warning(message, &self.config);
        print!("{}", output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::config::DisplayProfile;

    #[test]
    fn test_create_renderer() {
        let config = DisplayConfig::default();
        let renderer = create_renderer(&config);
        // Should not panic and should be object-safe
        assert_eq!(renderer.profile(), DisplayProfile::Advanced);
    }

    #[test]
    fn test_create_automation_renderer() {
        let mut config = DisplayConfig::default();
        config.profile = DisplayProfile::Automation;
        let renderer = create_renderer(&config);
        assert_eq!(renderer.profile(), DisplayProfile::Automation);
        assert!(!renderer.uses_color());
    }

    #[test]
    fn test_display_manager() {
        let config = DisplayConfig::default();
        let _manager = DisplayManager::new(config);
        // Should not panic
    }
}
