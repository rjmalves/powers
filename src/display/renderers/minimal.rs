//! Minimal display renderer.
//!
//! Minimal output with progress bar and final summary only.

use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::DisplayContext;
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

// TODO: Implement MinimalRenderer (Epic 2)

/// Minimal renderer with progress bar.
#[derive(Debug, Clone, Default)]
pub struct MinimalRenderer;

impl MinimalRenderer {
    /// Create a new minimal renderer.
    pub fn new() -> Self {
        Self
    }
}

impl DisplayRenderer for MinimalRenderer {
    fn render_header(
        &self,
        _config: &DisplayConfig,
        _iterations: usize,
        _forward_passes: usize,
        _cut_selection: bool,
    ) -> String {
        String::new()
    }

    fn render_table_header(&self, _config: &DisplayConfig) -> String {
        String::new()
    }

    fn render_iteration(
        &self,
        _ctx: &DisplayContext,
        _config: &DisplayConfig,
    ) -> String {
        String::new()
    }

    fn render_training_summary(
        &self,
        _result: &TrainingResult,
        _config: &DisplayConfig,
    ) -> String {
        String::new()
    }

    fn render_simulation_start(
        &self,
        _num_scenarios: usize,
        _config: &DisplayConfig,
    ) -> String {
        String::new()
    }

    fn render_simulation_summary(
        &self,
        _trajectories: &[SimulationTrajectory],
        _elapsed: Duration,
        _config: &DisplayConfig,
    ) -> String {
        String::new()
    }

    fn render_error(&self, _message: &str, _config: &DisplayConfig) -> String {
        String::new()
    }

    fn render_warning(
        &self,
        _message: &str,
        _config: &DisplayConfig,
    ) -> String {
        String::new()
    }

    fn profile(&self) -> DisplayProfile {
        DisplayProfile::Minimal
    }

    fn uses_color(&self) -> bool {
        true
    }
}
