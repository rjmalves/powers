//! Standard display renderer with key metrics and simplified layout.
//!
//! Provides a balance between detail and simplicity with core metrics,
//! box-drawing tables, and colors but without detailed statistics.

use crate::display::components::color::{
    bold, color_gap_percentage, colorize, ColorConfig, SemanticColor,
};
use crate::display::components::statistics::{
    format_cost, format_duration_hms, format_timing_pair,
};
use crate::display::components::table::{Alignment, BorderStyle};
use crate::display::components::table_format::{
    build_bottom_border, build_row, build_table_header, format_cell,
    TableColumnConfig,
};
use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::DisplayContext;
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

/// Standard renderer with key metrics and simplified layout.
///
/// Provides a balance between detail and simplicity:
/// - Single-line header
/// - Core metrics without statistics detail
/// - Box-drawing tables with colors
/// - Compact summary
#[derive(Debug, Clone)]
pub struct StandardRenderer {
    color_config: ColorConfig,
    terminal_width: u16,
}

impl StandardRenderer {
    /// Create a new standard renderer.
    pub fn new() -> Self {
        let terminal_width =
            crossterm::terminal::size().map(|(w, _)| w).unwrap_or(80);

        Self {
            color_config: ColorConfig::default(),
            terminal_width,
        }
    }

    fn render_header_box(
        &self,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) -> String {
        let width = (self.terminal_width as usize).clamp(50, 70);
        let inner_width = width.saturating_sub(4);

        let cut_str = if cut_selection { "enabled" } else { "disabled" };
        let content = format!(
            "POWE.RS Training: {} iterations × {} forward passes | Cut selection: {}",
            iterations, forward_passes, cut_str
        );

        let content_styled = bold(&content, &self.color_config);
        let content_display_len = content.len();
        let padding = inner_width.saturating_sub(content_display_len);
        let content_padded =
            format!("{}{}", content_styled, " ".repeat(padding));

        let top = format!("╭{}╮", "─".repeat(width.saturating_sub(2)));
        let row = format!("│ {} │", content_padded);
        let bottom = format!("╰{}╯", "─".repeat(width.saturating_sub(2)));

        format!("{}\n{}\n{}\n", top, row, bottom)
    }

    fn render_table_top_and_header(&self) -> String {
        let config = TableColumnConfig::standard();
        build_table_header(&config, BorderStyle::Standard)
    }

    fn render_data_row(&self, ctx: &DisplayContext) -> String {
        let config = TableColumnConfig::standard();
        let border = BorderStyle::Standard.chars().unwrap();

        // Format gap value with color
        let gap_value = format!("{:.1}%", ctx.gap_percent);
        let gap_colored = color_gap_percentage(
            ctx.gap_percent,
            &gap_value,
            &self.color_config,
        );

        let cells = vec![
            format_cell(
                &format!("{}", ctx.iteration),
                config.widths[0],
                Alignment::Right,
            ),
            format_cell(
                &format_cost(ctx.lower_bound, true),
                config.widths[1],
                Alignment::Center,
            ),
            format_cell(
                &format_cost(ctx.forward_cost_stats.mean, true),
                config.widths[2],
                Alignment::Center,
            ),
            format_cell(&gap_colored, config.widths[3], Alignment::Center),
            format_cell(
                &format_timing_pair(
                    ctx.forward_timing.total,
                    ctx.backward_timing.total,
                ),
                config.widths[4],
                Alignment::Center,
            ),
        ];

        build_row(&cells, &border)
    }

    fn render_table_bottom(&self) -> String {
        let config = TableColumnConfig::standard();
        let border = BorderStyle::Standard.chars().unwrap();
        build_bottom_border(&config.widths, &border)
    }
}

impl Default for StandardRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DisplayRenderer for StandardRenderer {
    fn render_header(
        &self,
        config: &DisplayConfig,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) -> String {
        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);
        renderer.render_header_box(iterations, forward_passes, cut_selection)
    }

    fn render_table_header(&self, config: &DisplayConfig) -> String {
        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);
        renderer.render_table_top_and_header()
    }

    fn render_iteration(
        &self,
        ctx: &DisplayContext,
        config: &DisplayConfig,
    ) -> String {
        if !ctx.should_print {
            return String::new();
        }

        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);

        let mut output = String::new();

        if ctx.iteration == 1 {
            output.push_str(&renderer.render_table_top_and_header());
            output.push('\n');
        }

        output.push_str(&renderer.render_data_row(ctx));
        output.push('\n');

        output
    }

    fn render_training_summary(
        &self,
        result: &TrainingResult,
        config: &DisplayConfig,
    ) -> String {
        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);

        let mut lines = Vec::new();

        // Close the table
        lines.push(renderer.render_table_bottom());
        lines.push(String::new());

        // Determine convergence
        let converged = result.relative_gap() < 0.05;

        // Compact title with icon
        let status_icon = if converged {
            colorize("✓", SemanticColor::Good, &renderer.color_config)
        } else {
            colorize("⋯", SemanticColor::Caution, &renderer.color_config)
        };

        lines.push(format!(
            "{} {}",
            bold("Training Complete", &renderer.color_config),
            status_icon
        ));

        // One-line summary
        let gap_pct = result.relative_gap() * 100.0;
        let gap_str = format!("{:.2}%", gap_pct);
        let gap_colored =
            color_gap_percentage(gap_pct, &gap_str, &renderer.color_config);

        lines.push(format!(
            "  Final gap: {} | Time: {}",
            gap_colored,
            format_duration_hms(result.total_time)
        ));

        lines.join("\n") + "\n"
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
        trajectories: &[SimulationTrajectory],
        _elapsed: Duration,
        config: &DisplayConfig,
    ) -> String {
        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);

        if trajectories.is_empty() {
            return format!(
                "{}\n  0 trajectories",
                bold("Simulation Complete", &renderer.color_config)
            );
        }

        // Calculate mean cost
        let costs: Vec<f64> = trajectories
            .iter()
            .map(|t| {
                t.realizations
                    .iter()
                    .map(|r| r.current_stage_objective)
                    .sum()
            })
            .collect();

        let mean = costs.iter().sum::<f64>() / costs.len() as f64;

        let icon = colorize("✓", SemanticColor::Good, &renderer.color_config);

        format!(
            "{} {}\n  {} trajectories | Mean: {}",
            bold("Simulation Complete", &renderer.color_config),
            icon,
            trajectories.len(),
            format_cost(mean, true)
        )
    }

    fn render_error(&self, message: &str, config: &DisplayConfig) -> String {
        let prefix = if config.color_enabled {
            "\x1b[31m✗\x1b[0m"
        } else {
            "✗"
        };
        format!("{} Error: {}\n", prefix, message)
    }

    fn render_warning(&self, message: &str, config: &DisplayConfig) -> String {
        let prefix = if config.color_enabled {
            "\x1b[33m⚠\x1b[0m"
        } else {
            "⚠"
        };
        format!("{} Warning: {}\n", prefix, message)
    }

    fn profile(&self) -> DisplayProfile {
        DisplayProfile::Standard
    }

    fn uses_color(&self) -> bool {
        self.color_config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::CostStatistics;
    use crate::timing::{BackwardTimingOutput, ForwardTimingOutput};

    fn create_test_ctx(iteration: usize) -> DisplayContext {
        let mut ctx = DisplayContext::new(iteration, 10);
        ctx.should_print = true;
        ctx.lower_bound = 101_480.0;
        ctx.forward_cost_stats = CostStatistics {
            mean: 128_230.0,
            std_dev: 3200.0,
            min: 124_000.0,
            max: 135_000.0,
            count: 4,
        };
        ctx.gap_percent = 26.4;
        ctx.forward_timing = ForwardTimingOutput {
            total: Duration::from_millis(18),
            ..Default::default()
        };
        ctx.backward_timing = BackwardTimingOutput {
            total: Duration::from_millis(34),
            ..Default::default()
        };
        ctx
    }

    fn create_test_training_result(
        final_lower_bound: f64,
        statistical_upper_bound: f64,
        best_upper_bound: f64,
        total_time: Duration,
    ) -> TrainingResult {
        TrainingResult::test_new(
            final_lower_bound,
            statistical_upper_bound,
            best_upper_bound,
            total_time,
            32,
        )
    }

    #[test]
    fn test_render_header_contains_brand() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let header = renderer.render_header(&config, 8, 4, true);

        assert!(header.contains("POWE.RS Training"));
        assert!(header.contains("8 iterations"));
        assert!(header.contains("4 forward passes"));
        // Should NOT have tagline (simpler than Advanced)
        assert!(!header.contains("Power Optimization for the World"));
    }

    #[test]
    fn test_render_header_cut_selection() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        let header_enabled = renderer.render_header(&config, 8, 4, true);
        assert!(header_enabled.contains("enabled"));

        let header_disabled = renderer.render_header(&config, 8, 4, false);
        assert!(header_disabled.contains("disabled"));
    }

    #[test]
    fn test_standard_has_fewer_columns_than_advanced() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(1);

        let output = renderer.render_iteration(&ctx, &config);

        // Should have 5 columns (not 6 like Advanced)
        assert!(output.contains("Iter"));
        assert!(output.contains("Lower Bound"));
        assert!(output.contains("Simul Cost"));
        assert!(output.contains("Gap %"));
        assert!(output.contains("Time"));

        // Should NOT have first-stage column
        assert!(!output.contains("1st Stage"));
    }

    #[test]
    fn test_no_statistics_continuation_row() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(2);

        let output = renderer.render_iteration(&ctx, &config);

        // Should NOT have statistics symbols
        assert!(!output.contains("μ="));
        assert!(!output.contains("σ="));
        assert!(!output.contains("n="));

        // Should be a single data row (no statistics continuation)
        // The output ends with newline, so trimming and counting
        let trimmed = output.trim();
        let line_count = trimmed.lines().count();
        assert_eq!(
            line_count, 1,
            "Should have exactly one line (the data row)"
        );
    }

    #[test]
    fn test_first_iteration_includes_table_header() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(1);

        let output = renderer.render_iteration(&ctx, &config);

        assert!(output.contains("Iter"));
        assert!(output.contains("Lower Bound"));
        assert!(output.contains("Gap %"));
    }

    #[test]
    fn test_subsequent_iterations_no_header() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(2);

        let output = renderer.render_iteration(&ctx, &config);

        // Should have data but not the header row
        let trimmed = output.trim();
        let line_count = trimmed.lines().count();
        assert_eq!(
            line_count, 1,
            "Should have exactly one line (the data row, no header)"
        );

        // Should not contain header labels
        assert!(!output.contains("Lower Bound ($)"));
    }

    #[test]
    fn test_skip_when_should_print_false() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();
        let mut ctx = DisplayContext::new(1, 10);
        ctx.should_print = false;

        assert!(renderer.render_iteration(&ctx, &config).is_empty());
    }

    #[test]
    fn test_compact_training_summary() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        let result = create_test_training_result(
            124_130.0,
            127_200.0,
            130_200.0,
            Duration::from_millis(511),
        );

        let output = renderer.render_training_summary(&result, &config);

        // Should be compact (not full breakdown)
        assert!(output.contains("Training Complete"));
        assert!(output.contains("Final gap:"));
        assert!(output.contains("Time:"));
        assert!(output.contains("00:00:00.511"));

        // Should NOT have full breakdown
        assert!(!output.contains("Total cuts:"));
        assert!(!output.contains("Policy cost:"));
        assert!(!output.contains("Iterations:"));

        // Should be 3 lines: table bottom + blank + title + summary
        let line_count = output.lines().count();
        assert!(line_count <= 4);
    }

    #[test]
    fn test_training_summary_converged() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        // Low gap = converged
        let result = create_test_training_result(
            100_000.0,
            102_000.0,
            103_000.0,
            Duration::from_secs(5),
        );

        let output = renderer.render_training_summary(&result, &config);
        assert!(output.contains("✓"));
    }

    #[test]
    fn test_training_summary_not_converged() {
        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        // High gap = not converged
        let result = create_test_training_result(
            100_000.0,
            120_000.0,
            125_000.0,
            Duration::from_secs(5),
        );

        let output = renderer.render_training_summary(&result, &config);
        assert!(output.contains("⋯") || !output.contains("✓"));
    }

    #[test]
    fn test_simulation_summary_compact() {
        use crate::sddp::{RealizationData, SimulationTrajectory};

        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        let trajectories = vec![SimulationTrajectory {
            scenario_id: 0,
            realizations: vec![RealizationData {
                stage_id: 0,
                loads: vec![],
                deficit: vec![],
                exchange: vec![],
                inflow: vec![],
                turbined_flow: vec![],
                spillage: vec![],
                thermal_generation: vec![],
                water_value: vec![],
                marginal_cost: vec![],
                current_stage_objective: 100_000.0,
                total_stage_objective: 100_000.0,
                final_storage: vec![],
            }],
        }];

        let output = renderer.render_simulation_summary(
            &trajectories,
            Duration::from_secs(5),
            &config,
        );

        assert!(output.contains("Simulation Complete"));
        assert!(output.contains("1 trajectories"));
        assert!(output.contains("Mean:"));

        // Should be compact (2 lines max)
        let line_count = output.lines().count();
        assert!(line_count <= 2);
    }

    #[test]
    fn test_simulation_summary_empty() {
        use crate::sddp::SimulationTrajectory;

        let renderer = StandardRenderer::new();
        let config = DisplayConfig::default();

        let trajectories: Vec<SimulationTrajectory> = vec![];

        let output = renderer.render_simulation_summary(
            &trajectories,
            Duration::from_secs(0),
            &config,
        );

        assert!(output.contains("Simulation Complete"));
        assert!(output.contains("0 trajectories"));
    }

    #[test]
    fn test_render_table_bottom() {
        let renderer = StandardRenderer::new();
        let bottom = renderer.render_table_bottom();

        assert!(bottom.contains("└"));
        assert!(bottom.contains("┘"));
        assert!(bottom.contains("┴"));
    }

    #[test]
    fn test_profile() {
        let renderer = StandardRenderer::new();
        assert_eq!(renderer.profile(), DisplayProfile::Standard);
    }

    #[test]
    fn test_uses_color() {
        let mut renderer = StandardRenderer::new();
        renderer.color_config = ColorConfig::new(true);
        assert!(renderer.uses_color());

        renderer.color_config = ColorConfig::new(false);
        assert!(!renderer.uses_color());
    }
}
