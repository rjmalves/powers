//! Automation renderer producing JSON Lines output.
//!
//! Outputs one JSON object per line for easy parsing by external tools.
//! No ANSI codes, no Unicode special characters.

use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::{DisplayContext, GapTrend};
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{SimulationTrajectory, TrainingResult};
use serde::Serialize;
use std::time::Duration;

/// JSON Lines renderer for automation and machine parsing.
#[derive(Debug, Clone, Default)]
pub struct AutomationRenderer;

impl AutomationRenderer {
    /// Create a new automation renderer.
    pub fn new() -> Self {
        Self
    }
}

impl DisplayRenderer for AutomationRenderer {
    fn render_header(
        &self,
        _config: &DisplayConfig,
        iterations: usize,
        forward_passes: usize,
        cut_selection: bool,
    ) -> String {
        let event = HeaderEvent {
            event_type: "header",
            program: "POWE.RS",
            version: env!("CARGO_PKG_VERSION"),
            iterations,
            forward_passes,
            cut_selection,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_table_header(&self, _config: &DisplayConfig) -> String {
        // No table header in JSON mode
        String::new()
    }

    fn render_iteration(
        &self,
        ctx: &DisplayContext,
        _config: &DisplayConfig,
    ) -> String {
        let event = IterationEvent::from_context(ctx);
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_training_summary(
        &self,
        result: &TrainingResult,
        _config: &DisplayConfig,
    ) -> String {
        let event = TrainingSummaryEvent::from_result(result);
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_simulation_start(
        &self,
        num_scenarios: usize,
        _config: &DisplayConfig,
    ) -> String {
        let event = SimulationStartEvent {
            event_type: "simulation_start",
            num_scenarios,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_simulation_summary(
        &self,
        trajectories: &[SimulationTrajectory],
        elapsed: Duration,
        _config: &DisplayConfig,
    ) -> String {
        let event =
            SimulationSummaryEvent::from_trajectories(trajectories, elapsed);
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_error(&self, message: &str, _config: &DisplayConfig) -> String {
        let event = LogEvent {
            event_type: "error",
            level: "error",
            message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn render_warning(&self, message: &str, _config: &DisplayConfig) -> String {
        let event = LogEvent {
            event_type: "warning",
            level: "warning",
            message,
            timestamp: chrono::Utc::now().to_rfc3339(),
        };
        serde_json::to_string(&event).unwrap() + "\n"
    }

    fn profile(&self) -> DisplayProfile {
        DisplayProfile::Automation
    }

    fn uses_color(&self) -> bool {
        false
    }
}

// JSON event structures

#[derive(Serialize)]
struct HeaderEvent<'a> {
    event_type: &'a str,
    program: &'a str,
    version: &'a str,
    iterations: usize,
    forward_passes: usize,
    cut_selection: bool,
    timestamp: String,
}

#[derive(Serialize)]
struct IterationEvent {
    event_type: &'static str,
    iteration: usize,
    total_iterations: usize,

    // Convergence
    lower_bound: f64,
    gap_percent: f64,
    gap_trend: &'static str,

    // Forward pass
    forward_cost_mean: f64,
    forward_cost_std: f64,
    forward_cost_min: f64,
    forward_cost_max: f64,
    forward_cost_count: usize,

    // First stage
    first_stage_bound: f64,
    first_stage_mean: f64,
    first_stage_std: f64,

    // Cuts
    cuts_added: usize,
    cuts_removed: usize,
    cuts_active: usize,

    // Timing (milliseconds)
    forward_time_ms: u64,
    backward_time_ms: u64,
    iteration_time_ms: u64,
    solver_calls: usize,

    timestamp: String,
}

impl IterationEvent {
    fn from_context(ctx: &DisplayContext) -> Self {
        Self {
            event_type: "iteration",
            iteration: ctx.iteration,
            total_iterations: ctx.total_iterations,
            lower_bound: ctx.lower_bound,
            gap_percent: ctx.gap_percent,
            gap_trend: match ctx.gap_trend {
                GapTrend::Improving => "improving",
                GapTrend::Worsening => "worsening",
                GapTrend::Stable => "stable",
                GapTrend::Unknown => "unknown",
            },
            forward_cost_mean: ctx.forward_cost_stats.mean,
            forward_cost_std: ctx.forward_cost_stats.std_dev,
            forward_cost_min: ctx.forward_cost_stats.min,
            forward_cost_max: ctx.forward_cost_stats.max,
            forward_cost_count: ctx.forward_cost_stats.count,
            first_stage_bound: ctx.first_stage_bound,
            first_stage_mean: ctx.first_stage_stats.mean,
            first_stage_std: ctx.first_stage_stats.std_dev,
            cuts_added: ctx.cuts_added,
            cuts_removed: ctx.cuts_removed,
            cuts_active: ctx.cuts_active,
            forward_time_ms: ctx.forward_timing.total.as_millis() as u64,
            backward_time_ms: ctx.backward_timing.total.as_millis() as u64,
            iteration_time_ms: ctx.iteration_time.as_millis() as u64,
            solver_calls: ctx.solver_calls,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Serialize)]
struct TrainingSummaryEvent {
    event_type: &'static str,
    final_lower_bound: f64,
    policy_cost: f64,
    final_gap_percent: f64,
    total_cuts: usize,
    total_iterations: usize,
    total_time_ms: u64,
    timestamp: String,
}

impl TrainingSummaryEvent {
    fn from_result(result: &TrainingResult) -> Self {
        Self {
            event_type: "training_summary",
            final_lower_bound: result.final_lower_bound,
            policy_cost: result.statistical_upper_bound,
            final_gap_percent: result.relative_gap() * 100.0,
            total_cuts: result.num_cuts,
            total_iterations: result.iterations().len(),
            total_time_ms: result.total_time.as_millis() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Serialize)]
struct SimulationStartEvent<'a> {
    event_type: &'a str,
    num_scenarios: usize,
    timestamp: String,
}

#[derive(Serialize)]
struct SimulationSummaryEvent {
    event_type: &'static str,
    num_scenarios: usize,
    expected_cost: f64,
    cost_std: f64,
    cost_min: f64,
    cost_max: f64,
    elapsed_time_ms: u64,
    timestamp: String,
}

impl SimulationSummaryEvent {
    fn from_trajectories(
        trajectories: &[SimulationTrajectory],
        elapsed: Duration,
    ) -> Self {
        let costs: Vec<f64> = trajectories
            .iter()
            .map(|t| {
                t.realizations
                    .iter()
                    .map(|r| r.current_stage_objective)
                    .sum()
            })
            .collect();

        let stats = crate::display::context::CostStatistics::from_costs(&costs);

        Self {
            event_type: "simulation_summary",
            num_scenarios: trajectories.len(),
            expected_cost: stats.mean,
            cost_std: stats.std_dev,
            cost_min: stats.min,
            cost_max: stats.max,
            elapsed_time_ms: elapsed.as_millis() as u64,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

#[derive(Serialize)]
struct LogEvent<'a> {
    event_type: &'a str,
    level: &'a str,
    message: &'a str,
    timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::context::CostStatistics;

    fn create_test_context() -> DisplayContext {
        let mut ctx = DisplayContext::new(1, 10);
        ctx.lower_bound = 100000.0;
        ctx.forward_cost_stats =
            CostStatistics::from_costs(&[110000.0, 115000.0]);
        ctx.iteration_time = Duration::from_millis(500);
        ctx
    }

    #[test]
    fn test_iteration_event_valid_json() {
        let renderer = AutomationRenderer::new();
        let ctx = create_test_context();
        let config = DisplayConfig::default();

        let output = renderer.render_iteration(&ctx, &config);

        // Verify it's valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).unwrap();

        // Verify key fields
        assert_eq!(parsed["event_type"], "iteration");
        assert!(parsed["iteration"].is_number());
        assert!(parsed["lower_bound"].is_number());
        assert!(parsed["timestamp"].is_string());
    }

    #[test]
    fn test_no_ansi_codes() {
        let renderer = AutomationRenderer::new();
        let ctx = create_test_context();
        let config = DisplayConfig::default();

        let output = renderer.render_iteration(&ctx, &config);

        // Check no ANSI escape sequences
        assert!(!output.contains("\x1b["));
        assert!(!output.contains("\u{001b}"));
    }

    #[test]
    fn test_json_lines_format() {
        let renderer = AutomationRenderer::new();
        let ctx = create_test_context();
        let config = DisplayConfig::default();

        let output = renderer.render_iteration(&ctx, &config);

        // Single line ending with newline
        assert_eq!(output.matches('\n').count(), 1);
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn test_header_event() {
        let renderer = AutomationRenderer::new();
        let config = DisplayConfig::default();

        let output = renderer.render_header(&config, 100, 10, true);

        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["event_type"], "header");
        assert_eq!(parsed["program"], "POWE.RS");
        assert_eq!(parsed["iterations"], 100);
        assert_eq!(parsed["forward_passes"], 10);
        assert_eq!(parsed["cut_selection"], true);
    }

    #[test]
    fn test_error_event() {
        let renderer = AutomationRenderer::new();
        let config = DisplayConfig::default();

        let output = renderer.render_error("Test error", &config);

        let parsed: serde_json::Value =
            serde_json::from_str(output.trim()).unwrap();
        assert_eq!(parsed["event_type"], "error");
        assert_eq!(parsed["level"], "error");
        assert_eq!(parsed["message"], "Test error");
    }

    #[test]
    fn test_profile() {
        let renderer = AutomationRenderer::new();
        assert_eq!(renderer.profile(), DisplayProfile::Automation);
        assert!(!renderer.uses_color());
    }
}
