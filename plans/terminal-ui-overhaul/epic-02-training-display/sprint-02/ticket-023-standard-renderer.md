# T-023: Implement StandardRenderer

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: Sprint 1 complete (all components)
> **Blocks**: T-025 (visual polish)

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - AdvancedRenderer for patterns to follow
- `src/display/renderer.rs` - DisplayRenderer trait
- `src/display/components/` - All display components
- Master plan - Display elements by profile table

## Context

### Background

The StandardRenderer provides a middle-ground between the detailed AdvancedRenderer and the minimal MinimalRenderer. It shows key metrics with colors and box-drawing but without the detailed statistics and continuation rows.

### Current State

AdvancedRenderer and MinimalRenderer are implemented. StandardRenderer needs to provide a simplified version of AdvancedRenderer.

## Specification

### What StandardRenderer Shows (from Master Plan)

| Element | Standard |
|---------|----------|
| Iteration number | ✓ |
| Lower bound | ✓ |
| Simulation cost (mean) | ✓ |
| Forward cost stats | — (not shown) |
| First-stage bound | ✓ |
| Gap % | ✓ |
| Gap trend indicator | — |
| Forward timing | ✓ |
| Backward timing | ✓ |
| Detailed timing breakdown | — |
| Cuts info | — |
| Box-drawing borders | ✓ |
| Colors | ✓ |

### Target Output Format

```
╭─────────────────────────────────────────────────────────────────╮
│ POWE.RS Training: 8 iterations × 4 forward passes               │
╰─────────────────────────────────────────────────────────────────╯

┌─────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter│ Lower Bound ($)│ Simul Cost ($) │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │   1.0148e+05   │   1.2823e+05   │ 26.4% │ 0.018s / 0.034s │
│   2 │   1.2041e+05   │   1.2982e+05   │  7.8% │ 0.004s / 0.037s │
└─────┴────────────────┴────────────────┴───────┴─────────────────┘

Training Complete ✓
  Final gap: 2.47% | Time: 00:00:00.511
```

### Key Differences from AdvancedRenderer

1. **Simpler header**: Single line, no tagline
2. **Fewer columns**: No first-stage column
3. **No continuation rows**: Just main metrics per iteration
4. **No trend arrows**: Just the values
5. **Simpler summary**: One-line summary instead of full breakdown

### StandardRenderer Struct

```rust
/// Standard display renderer with key metrics and simplified layout.
///
/// Provides a balance between detail and simplicity:
/// - Single-line header
/// - Core metrics without statistics detail
/// - Box-drawing tables with colors
/// - Compact summary
pub struct StandardRenderer {
    color_config: ColorConfig,
    terminal_width: u16,
    border_style: BorderStyle,
}
```

## Acceptance Criteria

- [ ] `StandardRenderer` implements `DisplayRenderer` trait
- [ ] Simpler header (single line)
- [ ] Fewer columns than Advanced (5 vs 6)
- [ ] No continuation rows with statistics
- [ ] No trend arrows on gap
- [ ] Compact one-line summary
- [ ] Colors applied correctly
- [ ] Box-drawing borders work
- [ ] Unit tests for all render methods

## Implementation Guide

### Step 1: Create standard.rs

```rust
//! Standard display renderer with key metrics and simplified layout.

use crate::display::{
    config::DisplayConfig,
    context::DisplayContext,
    renderer::DisplayRenderer,
    components::{
        color::{ColorConfig, colorize, SemanticColor},
        table::{BorderStyle, Alignment},
        statistics::{format_cost, format_timing_pair, format_duration_compact},
    },
};

pub struct StandardRenderer {
    color_config: ColorConfig,
    terminal_width: u16,
    border_style: BorderStyle,
}

impl StandardRenderer {
    pub fn new(color_config: ColorConfig, terminal_width: u16) -> Self {
        Self {
            color_config,
            terminal_width,
            border_style: BorderStyle::Standard,
        }
    }
}
```

### Step 2: Implement render_header

```rust
fn render_header(&self, config: &DisplayConfig) -> String {
    let width = self.terminal_width.min(70).max(50) as usize;
    let inner_width = width - 4;
    
    let content = format!(
        "POWE.RS Training: {} iterations × {} forward passes",
        config.max_iterations,
        config.forward_passes
    );
    
    let content_padded = format!("{:<width$}", content, width = inner_width);
    
    format!(
        "╭{}╮\n│ {} │\n╰{}╯\n",
        "─".repeat(width - 2),
        content_padded,
        "─".repeat(width - 2)
    )
}
```

### Step 3: Implement column layout

```rust
fn get_column_layout(&self) -> ColumnLayout {
    ColumnLayout {
        widths: vec![4, 14, 14, 6, 15],
        headers: vec![
            "Iter".to_string(),
            "Lower Bound ($)".to_string(),
            "Simul Cost ($)".to_string(),
            "Gap %".to_string(),
            "Time (fwd/bwd)".to_string(),
        ],
        alignments: vec![
            Alignment::Right,
            Alignment::Center,
            Alignment::Center,
            Alignment::Right,
            Alignment::Center,
        ],
    }
}
```

### Step 4: Implement render_iteration

```rust
fn render_iteration(&self, ctx: &DisplayContext) -> String {
    if !ctx.should_print {
        return String::new();
    }
    
    let layout = self.get_column_layout();
    let mut lines = Vec::new();
    
    // First iteration: table header
    if ctx.iteration == 1 {
        lines.push(self.render_table_top(&layout));
        lines.push(self.render_header_row(&layout));
        lines.push(self.render_separator(&layout));
    }
    
    // Simple data row (no continuation)
    let cells = vec![
        format!("{:>4}", ctx.iteration),
        format_cost(ctx.lower_bound, true),
        format_cost(ctx.forward_cost_stats.mean, true),
        format!("{:5.1}%", ctx.gap_percentage()),
        format_timing_pair(ctx.forward_timing.total, ctx.backward_timing.total),
    ];
    
    lines.push(self.format_row(&cells, &layout));
    
    lines.join("\n")
}
```

### Step 5: Implement render_training_summary

```rust
fn render_training_summary(&self, result: &TrainingResult) -> String {
    let mut lines = Vec::new();
    
    // Close table
    let layout = self.get_column_layout();
    lines.push(self.render_table_bottom(&layout));
    lines.push(String::new());
    
    // Compact summary
    let icon = if result.converged { "✓" } else { "⋯" };
    let icon_colored = if result.converged {
        colorize(icon, SemanticColor::Good, &self.color_config)
    } else {
        colorize(icon, SemanticColor::Caution, &self.color_config)
    };
    
    lines.push(format!("Training Complete {}", icon_colored));
    
    let gap_str = format!("{:.2}%", result.gap_percentage);
    let gap_colored = color_gap_percentage(result.gap_percentage, &gap_str, &self.color_config);
    
    lines.push(format!("  Final gap: {} | Time: {}", 
        gap_colored,
        format_duration_hms(result.total_time)));
    
    lines.join("\n")
}
```

### Step 6: Implement render_simulation_summary

```rust
fn render_simulation_summary(&self, trajectories: &[SimulationTrajectory]) -> String {
    let count = trajectories.len();
    let mean = trajectories.iter().map(|t| t.total_cost).sum::<f64>() / count.max(1) as f64;
    
    format!(
        "Simulation Complete ✓\n  {} trajectories | Mean: {}",
        count,
        format_cost(mean, true)
    )
}
```

### Step 7: Update renderers/mod.rs

```rust
pub mod standard;
pub use standard::StandardRenderer;
```

### Patterns to Follow

- Reuse helper methods from AdvancedRenderer where sensible
- Keep output more compact than Advanced
- Maintain same color scheme

### Pitfalls to Avoid

- ⚠️ Don't just copy AdvancedRenderer - this should be genuinely simpler
- ⚠️ Ensure column count matches header and data
- ⚠️ Keep summary to 1-2 lines

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_standard_render_header() {
    let renderer = StandardRenderer::new(ColorConfig::new(false), 80);
    let config = DisplayConfig {
        max_iterations: 8,
        forward_passes: 4,
        ..Default::default()
    };
    
    let header = renderer.render_header(&config);
    assert!(header.contains("POWE.RS Training"));
    assert!(header.contains("8 iterations"));
    assert!(!header.contains("Power Optimization")); // No tagline
}

#[test]
fn test_standard_fewer_columns() {
    let renderer = StandardRenderer::new(ColorConfig::new(false), 80);
    let ctx = DisplayContext {
        iteration: 1,
        should_print: true,
        ..Default::default()
    };
    
    let output = renderer.render_iteration(&ctx);
    
    // Should NOT have first-stage column
    assert!(!output.contains("1st Stage"));
    
    // Should have gap column
    assert!(output.contains("Gap %"));
}

#[test]
fn test_standard_no_statistics_continuation() {
    let renderer = StandardRenderer::new(ColorConfig::new(false), 80);
    let ctx = DisplayContext {
        iteration: 2,
        should_print: true,
        forward_cost_stats: CostStatistics {
            mean: 128000.0,
            std_dev: 3200.0,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let output = renderer.render_iteration(&ctx);
    
    // Should NOT have statistics line
    assert!(!output.contains("μ="));
    assert!(!output.contains("σ="));
}

#[test]
fn test_standard_compact_summary() {
    let renderer = StandardRenderer::new(ColorConfig::new(false), 80);
    let result = TrainingResult {
        gap_percentage: 2.47,
        total_time: Duration::from_millis(511),
        converged: true,
        ..Default::default()
    };
    
    let output = renderer.render_training_summary(&result);
    
    // Compact format on one line
    assert!(output.contains("Final gap:"));
    assert!(output.contains("Time:"));
    
    // Should NOT have full breakdown
    assert!(!output.contains("Total cuts:"));
    assert!(!output.contains("Policy cost:"));
}
```

## Documentation Requirements

- [ ] Doc comments on StandardRenderer struct
- [ ] Example output in module docs
- [ ] Note differences from AdvancedRenderer

## Effort Estimate

**Points**: 4
**Confidence**: High
**Rationale**: Pattern established by AdvancedRenderer. Simpler output but still needs all renderer methods.

## Definition of Done

- [ ] Implementation complete
- [ ] All DisplayRenderer methods implemented
- [ ] Simpler than Advanced but more than Minimal
- [ ] Colors and box-drawing work
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
