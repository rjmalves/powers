# T-007: Implement AutomationRenderer (JSON output)

> **Epic**: [Foundation](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-005](./ticket-005-terminal-detection.md), [T-006](./ticket-006-define-display-renderer.md)
> **Blocks**: Sprint 2 integration tickets

## Files to Read Before Starting

- `src/display/renderer.rs` - DisplayRenderer trait (from T-006)
- `src/display/context.rs` - DisplayContext (from T-004)
- `src/logging/formatters/json.rs` - Existing JSON formatter pattern
- `src/sddp/mod.rs` - TrainingResult, SimulationTrajectory types

## Context

### Background

The `AutomationRenderer` produces machine-readable JSON Lines output for CI/CD pipelines, log aggregation systems, and external tool integration. This is the simplest renderer to implement first, validating the architecture before building complex visual renderers.

### Current State

`LogFormat::Json` in the logging module produces basic JSON logs. The new `AutomationRenderer` will produce richer, structured output with all metrics.

## Specification

### AutomationRenderer Struct

```rust
//! Automation renderer producing JSON Lines output.
//!
//! Outputs one JSON object per line for easy parsing by external tools.
//! No ANSI codes, no Unicode special characters.

use serde::Serialize;
use crate::display::config::{DisplayConfig, DisplayProfile};
use crate::display::context::{DisplayContext, CostStatistics, GapTrend};
use crate::display::renderer::DisplayRenderer;
use crate::sddp::{TrainingResult, SimulationTrajectory};
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
        let event = SimulationSummaryEvent::from_trajectories(trajectories, elapsed);
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
```

### JSON Event Structures

```rust
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
    policy_std: f64,
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
            policy_std: 0.0, // TODO: compute from iterations
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
    fn from_trajectories(trajectories: &[SimulationTrajectory], elapsed: Duration) -> Self {
        let costs: Vec<f64> = trajectories.iter()
            .map(|t| t.realizations.iter().map(|r| r.current_stage_objective).sum())
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
```

## Acceptance Criteria

- [ ] `AutomationRenderer` implements `DisplayRenderer` trait
- [ ] All methods produce valid JSON (parseable by `serde_json::from_str`)
- [ ] Each output line is a complete JSON object (JSON Lines format)
- [ ] No ANSI escape codes in any output
- [ ] Timestamps in RFC 3339 format
- [ ] All DisplayContext fields represented in iteration output
- [ ] Event types clearly distinguish different message kinds
- [ ] Unit tests validate JSON structure

## Implementation Guide

### Step 1: Create automation.rs

Define `AutomationRenderer` and event structs.

### Step 2: Implement DisplayRenderer methods

Each method serializes appropriate event to JSON.

### Step 3: Add to renderers/mod.rs

Export `AutomationRenderer`.

### Step 4: Write tests

Validate JSON output is parseable and contains expected fields.

## Pitfalls to Avoid

- ⚠️ Don't use `serde_json::to_string_pretty()` - JSON Lines needs compact format
- ⚠️ Include `\n` at end of each JSON line
- ⚠️ Handle `unwrap()` carefully - serialization shouldn't fail for these types
- ⚠️ Use `&'static str` for event_type to avoid allocation

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iteration_event_valid_json() {
        let renderer = AutomationRenderer::new();
        let ctx = create_test_context();
        let config = DisplayConfig::default();
        
        let output = renderer.render_iteration(&ctx, &config);
        
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        
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
}

fn create_test_context() -> DisplayContext {
    let mut ctx = DisplayContext::new(1, 10);
    ctx.lower_bound = 100000.0;
    ctx.forward_cost_stats = CostStatistics::from_costs(&[110000.0, 115000.0]);
    ctx.iteration_time = Duration::from_millis(500);
    ctx
}
```

### Integration Tests

- [ ] Parse output of actual run with `--profile automation`
- [ ] Verify all event types appear in expected order
- [ ] Verify external tools (jq, Python) can parse output

## Documentation Requirements

- [ ] Module-level docs explaining JSON Lines format
- [ ] Doc comments on all event structs
- [ ] Example JSON output in documentation
- [ ] Note about parsing recommendations

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward serialization. Following existing JSON formatter pattern.

## Definition of Done

- [ ] All trait methods implemented
- [ ] All event types serializing correctly
- [ ] Tests passing for JSON validity
- [ ] No ANSI codes in output
- [ ] Renderer exported from `src/display/mod.rs`
- [ ] PR reviewed and merged
