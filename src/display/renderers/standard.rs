//! Standard display renderer.
//!
//! Key metrics with simplified layout.

use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::DisplayContext;
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

// TODO: Implement StandardRenderer (Epic 2)

/// Standard renderer with key metrics.
#[derive(Debug, Clone, Default)]
pub struct StandardRenderer;

impl StandardRenderer {
    /// Create a new standard renderer.
    pub fn new() -> Self {
        Self
    }
}

impl DisplayRenderer for StandardRenderer {
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
        DisplayProfile::Standard
    }

    fn uses_color(&self) -> bool {
        true
    }
}
