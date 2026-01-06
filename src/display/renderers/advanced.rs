//! Advanced display renderer with full metrics and visual styling.

use crate::display::components::color::{
    bold, color_gap_percentage, ColorConfig,
};
use crate::display::components::indicators::{
    bound_trend, gap_trend, trend_arrow_colored, TrendConfig,
};
use crate::display::components::statistics::{
    format_cost, format_cost_stats, format_gap, format_percentage_change,
    format_timing_pair, StatisticsFormat,
};
use crate::display::components::table::BorderStyle;
use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::DisplayContext;
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use std::time::Duration;

/// Advanced renderer with full metrics, colors, and visual indicators.
#[derive(Debug, Clone)]
pub struct AdvancedRenderer {
    color_config: ColorConfig,
    terminal_width: u16,
}

impl AdvancedRenderer {
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
        let width = (self.terminal_width as usize).clamp(60, 85);
        let inner_width = width.saturating_sub(4);

        let line1 = "POWE.RS - Power Optimization for the World of Energy";
        let line1_styled = bold(line1, &self.color_config);

        let cut_selection_str =
            if cut_selection { "enabled" } else { "disabled" };
        let line2 = format!(
            "Training: {} iterations × {} forward passes | Cut selection: {}",
            iterations, forward_passes, cut_selection_str
        );

        let line1_display_len = line1.len();
        let line1_padding = inner_width.saturating_sub(line1_display_len);
        let line1_padded =
            format!("{}{}", line1_styled, " ".repeat(line1_padding));

        let line2_padding = inner_width.saturating_sub(line2.len());
        let line2_padded = format!("{}{}", line2, " ".repeat(line2_padding));

        let top = format!("╭{}╮", "─".repeat(width.saturating_sub(2)));
        let row1 = format!("│ {} │", line1_padded);
        let row2 = format!("│ {} │", line2_padded);
        let bottom = format!("╰{}╯", "─".repeat(width.saturating_sub(2)));

        format!("{}\n{}\n{}\n{}\n", top, row1, row2, bottom)
    }

    fn render_table_top_and_header(&self) -> String {
        let border = BorderStyle::Standard.chars().unwrap();
        let col_widths = [5, 16, 16, 16, 7, 17];
        let headers = [
            "Iter",
            "Lower Bound ($)",
            "Simul Cost ($)",
            "1st Stage ($)",
            "Gap %",
            "Time (fwd/bwd)",
        ];

        let top = format!(
            "{}{}{}",
            border.top_left,
            col_widths
                .iter()
                .map(|&w| border.horizontal.to_string().repeat(w))
                .collect::<Vec<_>>()
                .join(&border.top_tee.to_string()),
            border.top_right
        );

        let header_row = format!(
            "{}{}{}",
            border.vertical,
            headers
                .iter()
                .zip(&col_widths)
                .map(|(header, &width)| format!(
                    " {:^width$} ",
                    header,
                    width = width - 2
                ))
                .collect::<Vec<_>>()
                .join(&border.vertical.to_string()),
            border.vertical
        );

        let separator = format!(
            "{}{}{}",
            border.left_tee,
            col_widths
                .iter()
                .map(|&w| border.horizontal.to_string().repeat(w))
                .collect::<Vec<_>>()
                .join(&border.cross.to_string()),
            border.right_tee
        );

        format!("{}\n{}\n{}", top, header_row, separator)
    }

    fn render_separator(&self) -> String {
        let border = BorderStyle::Standard.chars().unwrap();
        let col_widths = [5, 16, 16, 16, 7, 17];

        format!(
            "{}{}{}",
            border.left_tee,
            col_widths
                .iter()
                .map(|&w| border.horizontal.to_string().repeat(w))
                .collect::<Vec<_>>()
                .join(&border.cross.to_string()),
            border.right_tee
        )
    }

    fn render_data_row(&self, ctx: &DisplayContext) -> String {
        let border = BorderStyle::Standard.chars().unwrap();

        // Format each column
        let iter_cell = format!(" {:>3} ", ctx.iteration);

        let bound_value = format_cost(ctx.lower_bound, true);
        let bound_indicator = if let Some(prev) = ctx.previous_lower_bound {
            let trend = bound_trend(
                ctx.lower_bound,
                Some(prev),
                &TrendConfig::default(),
            );
            let arrow = trend_arrow_colored(trend, &self.color_config);
            format!(" {}", arrow)
        } else {
            String::new()
        };
        let bound_cell =
            format!(" {:^14} ", format!("{}{}", bound_value, bound_indicator));

        let simul_cost_cell =
            format!(" {:^14} ", format_cost(ctx.forward_cost_stats.mean, true));
        let first_stage_cell =
            format!(" {:^14} ", format_cost(ctx.first_stage_bound, true));

        let gap_value = format_gap(ctx.gap_percent);
        let gap_trend_dir = gap_trend(
            ctx.gap_percent,
            ctx.previous_lower_bound.map(|prev| {
                ((ctx.forward_cost_stats.mean - prev) / prev.abs()) * 100.0
            }),
            &TrendConfig::default(),
        );
        let gap_arrow = trend_arrow_colored(gap_trend_dir, &self.color_config);
        let gap_text = format!("{}{}", gap_value, gap_arrow);
        let gap_cell_text = color_gap_percentage(
            ctx.gap_percent,
            &gap_text,
            &self.color_config,
        );
        let gap_cell = format!(" {:>5} ", gap_cell_text);

        let timing_cell = format!(
            " {:^15} ",
            format_timing_pair(
                ctx.forward_timing.total,
                ctx.backward_timing.total,
            )
        );

        format!(
            "{}{}{}{}{}{}{}{}{}{}{}{}{}",
            border.vertical,
            iter_cell,
            border.vertical,
            bound_cell,
            border.vertical,
            simul_cost_cell,
            border.vertical,
            first_stage_cell,
            border.vertical,
            gap_cell,
            border.vertical,
            timing_cell,
            border.vertical
        )
    }

    fn render_stats_continuation(&self, ctx: &DisplayContext) -> String {
        let border = BorderStyle::Standard.chars().unwrap();

        let bound_change = if let Some(prev) = ctx.previous_lower_bound {
            if prev.abs() > 1e-10 {
                let change_pct =
                    ((ctx.lower_bound - prev) / prev.abs()) * 100.0;
                format_percentage_change(change_pct)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let stats = format_cost_stats(
            &ctx.forward_cost_stats,
            &StatisticsFormat::default(),
        );

        format!(
            "{} {:>3} {} {:<73} {}",
            border.vertical,
            bound_change,
            border.vertical,
            stats,
            border.vertical
        )
    }
}

impl Default for AdvancedRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl DisplayRenderer for AdvancedRenderer {
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
        output.push_str(&renderer.render_stats_continuation(ctx));
        output.push('\n');
        output.push_str(&renderer.render_separator());
        output.push('\n');

        output
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
        DisplayProfile::Advanced
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

    fn create_test_ctx(iteration: usize, has_previous: bool) -> DisplayContext {
        let mut ctx = DisplayContext::new(iteration, 10);
        ctx.should_print = true;
        ctx.lower_bound = 101_480.0;
        ctx.previous_lower_bound =
            if has_previous { Some(100_000.0) } else { None };
        ctx.forward_cost_stats = CostStatistics {
            mean: 128_230.0,
            std_dev: 3200.0,
            min: 124_000.0,
            max: 135_000.0,
            count: 4,
        };
        ctx.first_stage_bound = 101_480.0;
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

    #[test]
    fn test_render_header_contains_brand() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();
        let header = renderer.render_header(&config, 8, 4, true);
        assert!(header.contains("POWE.RS"));
        assert!(header.contains("Power Optimization"));
    }

    #[test]
    fn test_render_iteration_first_includes_table_header() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(1, false);
        let output = renderer.render_iteration(&ctx, &config);
        assert!(output.contains("Iter"));
        assert!(output.contains("Lower Bound"));
        assert!(output.contains("Gap %"));
    }

    #[test]
    fn test_render_iteration_skip_when_not_print() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();
        let mut ctx = DisplayContext::new(1, 10);
        ctx.should_print = false;
        assert!(renderer.render_iteration(&ctx, &config).is_empty());
    }

    #[test]
    fn test_profile() {
        let renderer = AdvancedRenderer::new();
        assert_eq!(renderer.profile(), DisplayProfile::Advanced);
    }
}
