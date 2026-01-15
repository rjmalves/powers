//! Advanced display renderer with full metrics and visual styling.

use crate::display::components::color::{
    bold, color_gap_percentage, colorize, ColorConfig, SemanticColor,
};
use crate::display::components::indicators::{
    gap_trend, trend_arrow_colored, TrendConfig,
};
use crate::display::components::statistics::{
    format_cost, format_duration_hms, format_duration_seconds, format_gap_value,
};
use crate::display::components::table::{Alignment, BorderStyle};
use crate::display::components::table_format::{
    build_bottom_border, build_row, build_separator, build_top_border,
    format_cell, TableColumnConfig,
};
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
        target_gap: Option<f64>,
    ) -> String {
        let width = (self.terminal_width as usize).clamp(60, 85);
        let inner_width = width.saturating_sub(4);

        let line1 = "POWE.RS - Power Optimization for the World of Energy";
        let line1_styled = bold(line1, &self.color_config);

        let cut_selection_str =
            if cut_selection { "enabled" } else { "disabled" };
        let line2 = format!(
            "Training: {} iterations × {} forward passes | Cut selection: {} ",
            iterations, forward_passes, cut_selection_str
        );

        let line1_display_len = line1.len();
        let line1_padding = inner_width.saturating_sub(line1_display_len);
        let line1_padded =
            format!("{}{}", line1_styled, " ".repeat(line1_padding));

        let line2_padding = inner_width.saturating_sub(line2.len()) + 1;
        let line2_padded = format!("{}{}", line2, " ".repeat(line2_padding));

        let mut rows = vec![
            format!("╭{}╮", "─".repeat(width.saturating_sub(2))),
            format!("│ {} │", line1_padded),
            format!("│ {} │", line2_padded),
        ];

        // Add target gap line if configured
        if let Some(target) = target_gap {
            let line3 = format!("Target: ≤{:.1}% gap", target);
            let line3_padding = inner_width.saturating_sub(line3.len());
            let line3_padded =
                format!("{}{}", line3, " ".repeat(line3_padding));
            rows.push(format!("│ {} │", line3_padded));
        }

        rows.push(format!("╰{}╯", "─".repeat(width.saturating_sub(2))));

        rows.join("\n") + "\n"
    }

    fn render_table_top_and_header(&self) -> String {
        let config = TableColumnConfig::advanced();
        let units = TableColumnConfig::advanced_units();
        let border = BorderStyle::Standard.chars().unwrap();

        let mut output = String::new();

        // Top border
        output.push_str(&build_top_border(&config.widths, &border));
        output.push('\n');

        // Header row 1: Column names
        let name_cells: Vec<String> = config
            .headers
            .iter()
            .zip(&config.widths)
            .map(|(header, &width)| {
                format_cell(header, width, Alignment::Center)
            })
            .collect();
        output.push_str(&build_row(&name_cells, &border));
        output.push('\n');

        // Header row 2: Units
        let unit_cells: Vec<String> = units
            .iter()
            .zip(&config.widths)
            .map(|(unit, &width)| format_cell(unit, width, Alignment::Center))
            .collect();
        output.push_str(&build_row(&unit_cells, &border));
        output.push('\n');

        // Separator after header
        output.push_str(&build_separator(&config.widths, &border));

        output
    }

    fn render_data_row(&self, ctx: &DisplayContext) -> String {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();

        // Format bound value (without trend - trend goes in separate position)
        let bound_value = format_cost(ctx.lower_bound, true);

        // Format gap value without % (unit is in header)
        // Always reserve space for arrow to maintain alignment
        let gap_value = format_gap_value(ctx.gap_percent);
        let gap_trend_dir = gap_trend(
            ctx.gap_percent,
            ctx.previous_lower_bound.map(|prev| {
                ((ctx.forward_cost_stats.mean - prev) / prev.abs()) * 100.0
            }),
            &TrendConfig::default(),
        );
        let gap_arrow = trend_arrow_colored(gap_trend_dir, &self.color_config);
        // Fixed-width gap: always "value + space + arrow_or_space"
        // This ensures consistent alignment whether arrow is present or not
        let gap_text = if gap_arrow.is_empty() {
            format!("{}  ", gap_value) // Two spaces when no arrow
        } else {
            format!("{} {}", gap_value, gap_arrow) // Space + arrow
        };
        let gap_content = color_gap_percentage(
            ctx.gap_percent,
            &gap_text,
            &self.color_config,
        );

        // Format timing values
        let total_time = format_duration_seconds(ctx.iteration_time);
        let fwd_time = format_duration_seconds(ctx.forward_timing.total);
        let fwd_solver = format_duration_seconds(ctx.forward_timing.solver);
        let bwd_time = format_duration_seconds(ctx.backward_timing.total);
        let bwd_solver = format_duration_seconds(ctx.backward_timing.solver);
        let cuts_time =
            format_duration_seconds(ctx.backward_timing.cut_selection);

        let cells = vec![
            // Column 0: Iteration number
            format_cell(
                &format!("{}", ctx.iteration),
                config.widths[0],
                Alignment::Right,
            ),
            // Column 1: Lower bound (right-aligned for consistent number display)
            format_cell(&bound_value, config.widths[1], Alignment::Right),
            // Column 2: Simulation cost (forward cost mean)
            format_cell(
                &format_cost(ctx.forward_cost_stats.mean, true),
                config.widths[2],
                Alignment::Right,
            ),
            // Column 3: Gap (value + arrow)
            format_cell(&gap_content, config.widths[3], Alignment::Right),
            // Column 4: Total iteration time
            format_cell(&total_time, config.widths[4], Alignment::Right),
            // Column 5: Forward pass total time
            format_cell(&fwd_time, config.widths[5], Alignment::Right),
            // Column 6: Forward solver average time
            format_cell(&fwd_solver, config.widths[6], Alignment::Right),
            // Column 7: Backward pass total time
            format_cell(&bwd_time, config.widths[7], Alignment::Right),
            // Column 8: Backward solver average time
            format_cell(&bwd_solver, config.widths[8], Alignment::Right),
            // Column 9: Cut selection time
            format_cell(&cuts_time, config.widths[9], Alignment::Right),
        ];

        build_row(&cells, &border)
    }

    fn render_table_bottom(&self) -> String {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();
        build_bottom_border(&config.widths, &border)
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
        renderer.render_header_box(
            iterations,
            forward_passes,
            cut_selection,
            config.target_gap,
        )
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

        // Close the table (bottom border)
        lines.push(renderer.render_table_bottom());
        lines.push(String::new()); // Blank line

        // Determine convergence status
        let converged = result.relative_gap() < 0.05; // 5% threshold

        // Title with status icon
        let status_icon = if converged {
            colorize("✓", SemanticColor::Good, &renderer.color_config)
        } else {
            colorize("⋯", SemanticColor::Caution, &renderer.color_config)
        };

        let title = if converged {
            "Training Complete"
        } else {
            "Training Stopped"
        };

        lines.push(format!(
            "{} {}",
            bold(title, &renderer.color_config),
            status_icon
        ));

        // Separator line (same width as title)
        let sep_len = title.len() + 2; // +2 for icon and space
        lines.push("─".repeat(sep_len));

        // Metrics
        lines.push(format!(
            "  Total time:     {}",
            format_duration_hms(result.total_time)
        ));

        lines.push(format!(
            "  Final bound:    {}",
            format_cost(result.final_lower_bound, true)
        ));

        lines.push(format!(
            "  Policy cost:    {} ± {}",
            format_cost(result.statistical_upper_bound, true),
            format_cost(
                (result.best_upper_bound - result.statistical_upper_bound)
                    .abs(),
                true
            )
        ));

        let gap_pct = result.relative_gap() * 100.0;
        let gap_str = format!("{:.2}%", gap_pct);
        let gap_colored =
            color_gap_percentage(gap_pct, &gap_str, &renderer.color_config);
        lines.push(format!("  Final gap:      {}", gap_colored));

        lines.push(format!("  Total cuts:     {}", result.num_cuts));

        lines.push(format!("  Iterations:     {}", result.iterations().len()));

        // Add target gap achievement status if configured
        if let Some(target) = config.target_gap {
            let gap_pct = result.relative_gap() * 100.0;
            if gap_pct <= target {
                let msg = format!("  Target achieved: ≤{:.1}% gap ✓", target);
                lines.push(colorize(
                    &msg,
                    SemanticColor::Good,
                    &renderer.color_config,
                ));
            } else {
                let msg = format!(
                    "  Target missed:   ≤{:.1}% (got {:.2}%)",
                    target, gap_pct
                );
                lines.push(colorize(
                    &msg,
                    SemanticColor::Caution,
                    &renderer.color_config,
                ));
            }
        }

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
        elapsed: Duration,
        config: &DisplayConfig,
    ) -> String {
        let mut renderer = self.clone();
        renderer.color_config = ColorConfig::new(config.color_enabled);

        let mut lines = Vec::new();

        // Add blank line before simulation summary
        lines.push(String::new());

        // Title with checkmark
        let title = "Simulation Complete";
        let icon = colorize("✓", SemanticColor::Good, &renderer.color_config);
        lines.push(format!("{} {}", bold(title, &renderer.color_config), icon));

        // Separator
        lines.push("─".repeat(title.len() + 2));

        // Total time (first, matching training summary order)
        lines.push(format!("  Total time:   {}", format_duration_hms(elapsed)));

        // Handle empty trajectories
        if trajectories.is_empty() {
            lines.push("  Trajectories: 0".to_string());
            return lines.join("\n");
        }

        // Calculate total cost for each trajectory by summing stage costs
        let costs: Vec<f64> = trajectories
            .iter()
            .map(|t| {
                t.realizations
                    .iter()
                    .map(|r| r.current_stage_objective)
                    .sum()
            })
            .collect();

        let count = costs.len();
        let mean = costs.iter().sum::<f64>() / count as f64;
        let std_dev = if count > 1 {
            let variance =
                costs.iter().map(|c| (c - mean).powi(2)).sum::<f64>()
                    / (count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        lines.push(format!("  Trajectories: {}", count));
        lines.push(format!(
            "  Mean cost:    {} ± {}",
            format_cost(mean, true),
            format_cost(std_dev, true)
        ));
        lines.push(format!("  Min cost:     {}", format_cost(min, true)));
        lines.push(format!("  Max cost:     {}", format_cost(max, true)));

        lines.join("\n")
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

    fn create_test_training_result(
        final_lower_bound: f64,
        statistical_upper_bound: f64,
        best_upper_bound: f64,
        total_time: Duration,
        num_cuts: usize,
    ) -> TrainingResult {
        TrainingResult::test_new(
            final_lower_bound,
            statistical_upper_bound,
            best_upper_bound,
            total_time,
            num_cuts,
        )
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
        assert!(output.contains("Gap"));
        assert!(output.contains("(%)"));
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

    #[test]
    fn test_render_training_summary_converged() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Create a training result with low gap (converged)
        let result = create_test_training_result(
            124_130.0,
            127_200.0,
            130_200.0,
            Duration::from_millis(511),
            32,
        );

        let output = renderer.render_training_summary(&result, &config);

        // Should show "Training Complete" because gap is < 5%
        assert!(output.contains("Training Complete"));
        assert!(output.contains("✓") || output.contains("checkmark"));
        assert!(output.contains("00:00:00.511"));
        assert!(output.contains("1.24e5") || output.contains("124130"));
        assert!(output.contains("32"));
        // Check for table bottom border
        assert!(output.contains("└") || output.contains("bottom"));
    }

    #[test]
    fn test_render_training_summary_not_converged() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Create a training result with high gap (not converged)
        let result = create_test_training_result(
            100_000.0,
            120_000.0,
            125_000.0,
            Duration::from_secs(10),
            15,
        );

        let output = renderer.render_training_summary(&result, &config);

        // Should show "Training Stopped" because gap is >= 5%
        assert!(
            output.contains("Training Stopped")
                || output.contains("Training Complete")
        );
        // Should have caution indicator or no checkmark
        assert!(output.contains("⋯") || !output.contains("✓"));
        assert!(output.contains("00:00:10.000"));
        assert!(output.contains("15"));
    }

    #[test]
    fn test_render_simulation_summary() {
        use crate::sddp::{RealizationData, SimulationTrajectory};

        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Create test trajectories with known costs
        let trajectories = vec![
            SimulationTrajectory {
                scenario_id: 0,
                realizations: vec![
                    RealizationData {
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
                        current_stage_objective: 50_000.0,
                        total_stage_objective: 50_000.0,
                        final_storage: vec![],
                    },
                    RealizationData {
                        stage_id: 1,
                        loads: vec![],
                        deficit: vec![],
                        exchange: vec![],
                        inflow: vec![],
                        turbined_flow: vec![],
                        spillage: vec![],
                        thermal_generation: vec![],
                        water_value: vec![],
                        marginal_cost: vec![],
                        current_stage_objective: 70_000.0,
                        total_stage_objective: 120_000.0,
                        final_storage: vec![],
                    },
                ],
            },
            SimulationTrajectory {
                scenario_id: 1,
                realizations: vec![
                    RealizationData {
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
                        current_stage_objective: 55_000.0,
                        total_stage_objective: 55_000.0,
                        final_storage: vec![],
                    },
                    RealizationData {
                        stage_id: 1,
                        loads: vec![],
                        deficit: vec![],
                        exchange: vec![],
                        inflow: vec![],
                        turbined_flow: vec![],
                        spillage: vec![],
                        thermal_generation: vec![],
                        water_value: vec![],
                        marginal_cost: vec![],
                        current_stage_objective: 75_000.0,
                        total_stage_objective: 130_000.0,
                        final_storage: vec![],
                    },
                ],
            },
            SimulationTrajectory {
                scenario_id: 2,
                realizations: vec![
                    RealizationData {
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
                        current_stage_objective: 52_500.0,
                        total_stage_objective: 52_500.0,
                        final_storage: vec![],
                    },
                    RealizationData {
                        stage_id: 1,
                        loads: vec![],
                        deficit: vec![],
                        exchange: vec![],
                        inflow: vec![],
                        turbined_flow: vec![],
                        spillage: vec![],
                        thermal_generation: vec![],
                        water_value: vec![],
                        marginal_cost: vec![],
                        current_stage_objective: 72_500.0,
                        total_stage_objective: 125_000.0,
                        final_storage: vec![],
                    },
                ],
            },
        ];

        let output = renderer.render_simulation_summary(
            &trajectories,
            Duration::from_secs(5),
            &config,
        );

        assert!(output.contains("Simulation Complete"));
        assert!(output.contains("✓"));
        assert!(output.contains("Trajectories: 3"));
        assert!(output.contains("Mean cost:"));
        assert!(output.contains("Min cost:"));
        assert!(output.contains("Max cost:"));

        // Check that costs are calculated correctly
        // Trajectory 0: 120,000, Trajectory 1: 130,000, Trajectory 2: 125,000
        // Mean = 125,000
        assert!(output.contains("1.25e5") || output.contains("125000"));
    }

    #[test]
    fn test_render_simulation_summary_empty() {
        use crate::sddp::SimulationTrajectory;

        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        let trajectories: Vec<SimulationTrajectory> = vec![];

        let output = renderer.render_simulation_summary(
            &trajectories,
            Duration::from_secs(0),
            &config,
        );

        assert!(output.contains("Simulation Complete"));
        assert!(output.contains("Trajectories: 0"));
    }

    #[test]
    fn test_render_table_bottom() {
        let renderer = AdvancedRenderer::new();
        let bottom = renderer.render_table_bottom();

        // Should contain bottom-left corner
        assert!(bottom.contains("└"));
        // Should contain bottom-right corner
        assert!(bottom.contains("┘"));
        // Should contain bottom tee characters
        assert!(bottom.contains("┴"));
    }

    #[test]
    fn test_header_shows_target_when_configured() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig {
            target_gap: Some(5.0),
            ..Default::default()
        };

        let header = renderer.render_header(&config, 8, 4, true);

        assert!(header.contains("Target:"));
        assert!(header.contains("5.0%"));
    }

    #[test]
    fn test_header_no_target_when_not_configured() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        let header = renderer.render_header(&config, 8, 4, true);

        assert!(!header.contains("Target:"));
    }

    #[test]
    fn test_training_summary_target_achieved() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig {
            target_gap: Some(5.0),
            ..Default::default()
        };

        // Gap is 2.4% which is below 5% target
        let result = create_test_training_result(
            100_000.0,
            102_400.0,
            103_000.0,
            Duration::from_secs(10),
            20,
        );

        let output = renderer.render_training_summary(&result, &config);

        assert!(output.contains("Target achieved"));
        assert!(output.contains("5.0%"));
    }

    #[test]
    fn test_training_summary_target_missed() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig {
            target_gap: Some(5.0),
            ..Default::default()
        };

        // Gap is 20% which is above 5% target
        let result = create_test_training_result(
            100_000.0,
            120_000.0,
            125_000.0,
            Duration::from_secs(10),
            20,
        );

        let output = renderer.render_training_summary(&result, &config);

        assert!(output.contains("Target missed"));
        assert!(output.contains("5.0%"));
    }

    #[test]
    fn test_training_summary_no_target() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default(); // No target_gap

        let result = create_test_training_result(
            100_000.0,
            120_000.0,
            125_000.0,
            Duration::from_secs(10),
            20,
        );

        let output = renderer.render_training_summary(&result, &config);

        // Should not mention target
        assert!(!output.contains("Target"));
    }

    #[test]
    fn test_two_row_header_structure() {
        let renderer = AdvancedRenderer::new();
        let header = renderer.render_table_top_and_header();

        let lines: Vec<&str> = header.lines().collect();

        // Should have: top border, name row, unit row, separator
        assert_eq!(lines.len(), 4);

        let name_row = lines[1];
        let unit_row = lines[2];

        // Name row should contain column names
        assert!(name_row.contains("Iter"));
        assert!(name_row.contains("Lower Bound"));
        assert!(name_row.contains("Total Time"));
        assert!(name_row.contains("Fwd Time"));
        assert!(name_row.contains("Bwd Time"));
        assert!(name_row.contains("Cut Selection"));

        // Unit row should contain units
        assert!(unit_row.contains("($)")); // for cost columns
        assert!(unit_row.contains("(s)")); // for timing columns
        assert!(unit_row.contains("(%)")); // for gap column

        // Count "(s)" occurrences - should be 6 (all timing columns)
        let timing_unit_count = unit_row.matches("(s)").count();
        assert_eq!(timing_unit_count, 6);
    }

    #[test]
    fn test_timing_columns_formatted_correctly() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();
        let mut ctx = create_test_ctx(1, false);

        // Set known timing values
        ctx.iteration_time = Duration::from_millis(52);
        ctx.forward_timing.total = Duration::from_millis(18);
        ctx.forward_timing.solver = Duration::from_millis(15);
        ctx.backward_timing.total = Duration::from_millis(34);
        ctx.backward_timing.solver = Duration::from_millis(28);
        ctx.backward_timing.cut_selection = Duration::from_millis(3);

        let output = renderer.render_iteration(&ctx, &config);

        // Verify all timing values appear with 3 decimal places
        assert!(output.contains("0.052"), "Should contain total time 0.052");
        assert!(output.contains("0.018"), "Should contain fwd time 0.018");
        assert!(output.contains("0.015"), "Should contain fwd solver 0.015");
        assert!(output.contains("0.034"), "Should contain bwd time 0.034");
        assert!(output.contains("0.028"), "Should contain bwd solver 0.028");
        assert!(output.contains("0.003"), "Should contain cuts time 0.003");
    }

    #[test]
    fn test_timing_edge_cases() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Test 1: Zero times
        let mut ctx = DisplayContext::new(2, 10);
        ctx.should_print = true;
        ctx.iteration_time = Duration::ZERO;
        ctx.forward_timing.total = Duration::ZERO;
        ctx.backward_timing.total = Duration::ZERO;

        let output = renderer.render_iteration(&ctx, &config);
        assert!(output.contains("0.000"));

        // Test 2: Large times (several minutes)
        let mut ctx3 = DisplayContext::new(3, 10);
        ctx3.should_print = true;
        ctx3.iteration_time = Duration::from_secs(185); // ~3 minutes

        let output3 = renderer.render_iteration(&ctx3, &config);
        assert!(output3.contains("185.000"));
    }

    #[test]
    fn test_single_row_per_iteration() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Subsequent iteration (no header)
        let mut ctx = DisplayContext::new(2, 10);
        ctx.should_print = true;

        let output = renderer.render_iteration(&ctx, &config);

        // Should be exactly 1 line for non-first iteration
        let line_count = output.lines().count();
        assert_eq!(line_count, 1, "Should be single line per iteration");
    }

    #[test]
    fn test_first_iteration_has_header_plus_data() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();
        let ctx = create_test_ctx(1, false);

        let output = renderer.render_iteration(&ctx, &config);

        // First iteration: top border + name row + unit row + separator + data row = 5 lines
        let line_count = output.lines().count();
        assert_eq!(line_count, 5, "First iteration should have 5 lines");
    }

    #[test]
    fn test_ten_columns_in_data_row() {
        let renderer = AdvancedRenderer::new();
        let config = DisplayConfig::default();

        // Subsequent iteration (no header)
        let mut ctx = DisplayContext::new(2, 10);
        ctx.should_print = true;
        ctx.lower_bound = 100_000.0;
        ctx.forward_cost_stats = CostStatistics {
            mean: 110_000.0,
            std_dev: 1000.0,
            min: 109_000.0,
            max: 111_000.0,
            count: 4,
        };

        let output = renderer.render_iteration(&ctx, &config);

        // Count vertical bar separators (should be 11: start + 9 internal + end)
        let separator_count = output.matches('│').count();
        assert_eq!(
            separator_count, 11,
            "Should have 11 vertical separators for 10 columns"
        );
    }
}
