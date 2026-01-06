//! Minimal display renderer.
//!
//! Minimal output with progress bar and final summary only.

use crate::display::components::{
    color::{colorize, ColorConfig, SemanticColor},
    progress::{ProgressBar, ProgressBarConfig},
    statistics::{format_cost, format_duration_hms},
};
use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::DisplayContext;
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

/// Minimal renderer with progress bar.
///
/// Output during training:
/// ```text
/// Training: [████████░░░░░░░░░░░░] 40% | 4/10 iter | ETA: 00:01:23
/// ```
///
/// Output for summary:
/// ```text
/// Training Complete ✓
///   Final gap: 2.47% | Time: 00:00:00.511
/// ```
#[derive(Debug, Clone)]
pub struct MinimalRenderer {
    color_config: ColorConfig,
}

impl MinimalRenderer {
    /// Create a new minimal renderer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            color_config: ColorConfig::new(true),
        }
    }
}

impl Default for MinimalRenderer {
    fn default() -> Self {
        Self::new()
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
        // Minimal: No header during training (progress bar is enough)
        String::new()
    }

    fn render_table_header(&self, _config: &DisplayConfig) -> String {
        String::new()
    }

    fn render_iteration(
        &self,
        ctx: &DisplayContext,
        _config: &DisplayConfig,
    ) -> String {
        if !ctx.should_print {
            return String::new();
        }

        // Create progress bar for this iteration
        let mut bar = ProgressBar::new(
            ctx.total_iterations,
            ProgressBarConfig {
                width: 30,
                show_percentage: true,
                show_eta: true,
                show_count: true,
                ..Default::default()
            },
        );
        bar.set(ctx.iteration);

        let progress = bar.render(&self.color_config);

        // Use carriage return to overwrite previous line
        format!("\rTraining: {}", progress)
    }

    fn render_training_summary(
        &self,
        result: &TrainingResult,
        config: &DisplayConfig,
    ) -> String {
        let color_enabled = config.color_enabled;
        let color_config = ColorConfig::new(color_enabled);

        // Calculate gap percentage from final_gap()
        let gap_abs = result.final_gap();
        let gap_pct = if result.final_lower_bound != 0.0 {
            (gap_abs / result.final_lower_bound.abs()) * 100.0
        } else {
            0.0
        };
        let converged = gap_pct < 5.0;

        let icon = if converged {
            colorize("✓", SemanticColor::Good, &color_config)
        } else {
            colorize("⋯", SemanticColor::Caution, &color_config)
        };

        let title = format!("\nTraining Complete {}", icon);

        let gap_str = format!("{:.2}%", gap_pct);
        let gap_colored = if color_enabled {
            if gap_pct < 5.0 {
                colorize(&gap_str, SemanticColor::Good, &color_config)
            } else if gap_pct < 20.0 {
                colorize(&gap_str, SemanticColor::Caution, &color_config)
            } else {
                colorize(&gap_str, SemanticColor::Bad, &color_config)
            }
        } else {
            gap_str
        };

        format!(
            "{}\n  Final gap: {} | Time: {}",
            title,
            gap_colored,
            format_duration_hms(result.total_time)
        )
    }

    fn render_simulation_start(
        &self,
        _num_scenarios: usize,
        _config: &DisplayConfig,
    ) -> String {
        "\nRunning simulation...".to_string()
    }

    fn render_simulation_summary(
        &self,
        trajectories: &[SimulationTrajectory],
        _elapsed: Duration,
        config: &DisplayConfig,
    ) -> String {
        let color_enabled = config.color_enabled;
        let color_config = ColorConfig::new(color_enabled);

        let count = trajectories.len();

        // Calculate mean total cost from trajectories
        let mean = if count > 0 {
            let total: f64 = trajectories
                .iter()
                .map(|t| {
                    t.realizations
                        .iter()
                        .map(|r| r.total_stage_objective)
                        .sum::<f64>()
                })
                .sum();
            total / count as f64
        } else {
            0.0
        };

        let icon = colorize("✓", SemanticColor::Good, &color_config);

        format!(
            "\nSimulation Complete {}\n  {} trajectories | Mean: {}",
            icon,
            count,
            format_cost(mean, true)
        )
    }

    fn render_error(&self, message: &str, config: &DisplayConfig) -> String {
        let color_enabled = config.color_enabled;
        let color_config = ColorConfig::new(color_enabled);

        format!(
            "\n❌ ERROR: {}",
            colorize(message, SemanticColor::Bad, &color_config)
        )
    }

    fn render_warning(&self, message: &str, config: &DisplayConfig) -> String {
        let color_enabled = config.color_enabled;
        let color_config = ColorConfig::new(color_enabled);

        format!(
            "\n⚠️  Warning: {}",
            colorize(message, SemanticColor::Caution, &color_config)
        )
    }

    fn profile(&self) -> DisplayProfile {
        DisplayProfile::Minimal
    }

    fn uses_color(&self) -> bool {
        self.color_config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::context::{CostStatistics, GapTrend};
    use crate::timing::{BackwardTimingOutput, ForwardTimingOutput};

    fn create_test_context() -> DisplayContext {
        DisplayContext {
            iteration: 3,
            total_iterations: 10,
            should_print: true,
            lower_bound: 100.0,
            previous_lower_bound: None,
            target_gap: None,
            gap_percent: 10.0,
            gap_trend: GapTrend::Unknown,
            forward_costs: vec![],
            forward_cost_stats: CostStatistics::default(),
            forward_timing: ForwardTimingOutput::default(),
            backward_timing: BackwardTimingOutput::default(),
            first_stage_bound: 100.0,
            first_stage_branching_costs: vec![],
            first_stage_stats: CostStatistics::default(),
            cuts_added: 0,
            cuts_removed: 0,
            cuts_active: 0,
            cuts_returned: 0,
            solver_calls: 0,
            iteration_time: Duration::from_secs(1),
            elapsed_total: Duration::from_secs(3),
        }
    }

    #[test]
    fn test_minimal_render_iteration() {
        let renderer = MinimalRenderer::new();
        let ctx = create_test_context();

        let output = renderer.render_iteration(&ctx, &DisplayConfig::default());
        assert!(output.starts_with("\rTraining:"));
        assert!(output.contains("30%")); // 3/10 = 30%
        assert!(output.contains("3/10 iter"));
    }

    #[test]
    fn test_minimal_render_iteration_skip() {
        let renderer = MinimalRenderer::new();
        let mut ctx = create_test_context();
        ctx.should_print = false;

        assert!(renderer
            .render_iteration(&ctx, &DisplayConfig::default())
            .is_empty());
    }

    #[test]
    fn test_minimal_render_header_empty() {
        let renderer = MinimalRenderer::new();
        let config = DisplayConfig::default();
        assert!(renderer.render_header(&config, 10, 4, true).is_empty());
    }

    #[test]
    fn test_minimal_uses_color() {
        let renderer = MinimalRenderer::new();
        assert!(renderer.uses_color());
    }

    #[test]
    fn test_minimal_profile() {
        let renderer = MinimalRenderer::new();
        assert_eq!(renderer.profile(), DisplayProfile::Minimal);
    }
}
