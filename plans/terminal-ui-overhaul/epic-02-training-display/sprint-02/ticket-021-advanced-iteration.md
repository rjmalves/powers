# T-021: Implement AdvancedRenderer iteration row

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 2](./00-sprint-overview.md)
> **Dependencies**: T-020 (AdvancedRenderer header)
> **Blocks**: T-022, T-024

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - AdvancedRenderer skeleton
- `src/display/context.rs` - DisplayContext with all metrics
- `src/display/components/table.rs` - Table builder
- `src/display/components/statistics.rs` - Statistics formatting
- `src/display/components/indicators.rs` - Trend indicators
- Master plan sample output - Target iteration row appearance

## Context

### Background

The iteration row is the core of the AdvancedRenderer, showing all per-iteration metrics in a richly formatted table. This is where users see training progress in real-time with colors, trends, and statistics.

### Current State

T-020 created the AdvancedRenderer skeleton with header implementation. Now we implement the most complex method: `render_iteration`.

## Specification

### Target Output Format

```
┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter│ Lower Bound ($)│ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │   1.0148e+05   │   1.2823e+05   │   1.0148e+05   │ 26.4↓ │ 0.018s / 0.034s │
│     │                │ μ=1.28e5 σ=3.2e3 [1.24e5..1.35e5] n=4                     │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   2 │   1.2041e+05 ▲ │   1.2982e+05   │   1.2041e+05   │  7.8↓ │ 0.004s / 0.037s │
│     │ +18.6%         │ μ=1.30e5 σ=2.8e3 [1.26e5..1.34e5] n=4                     │
└─────┴────────────────┴────────────────┴────────────────┴───────┴─────────────────┘
```

### Iteration Row Structure

Each iteration produces:
1. **Main row**: Core metrics in table columns
2. **Continuation row**: Forward cost statistics spanning columns

### Column Definitions

| Column | Header | Width | Alignment | Content |
|--------|--------|-------|-----------|---------|
| 1 | Iter | 4 | Right | Iteration number |
| 2 | Lower Bound ($) | 14 | Center | Lower bound + change indicator |
| 3 | Simul Cost ($) | 14 | Center | Mean simulation cost |
| 4 | 1st Stage ($) | 14 | Center | First-stage bound |
| 5 | Gap % | 6 | Right | Gap percentage + trend arrow |
| 6 | Time (fwd/bwd) | 15 | Center | Forward/backward timing |

### Color Rules

| Element | Condition | Color |
|---------|-----------|-------|
| Lower Bound | Improved | Green + ▲ |
| Lower Bound | Worsened | Red + ▼ |
| Gap % | < 5% | Green |
| Gap % | 5-20% | Yellow |
| Gap % | > 20% | Red |
| Trend Arrow | Improving | Green ↓ |
| Trend Arrow | Worsening | Red ↑ |
| Trend Arrow | Stable | Yellow → |

### Implementation Approach

For iteration rendering, we need special handling:
1. First iteration: Render table header with top border
2. Each iteration: Render data row with separator
3. Last iteration: Render bottom border (in summary)

This allows streaming output while maintaining table structure.

## Acceptance Criteria

- [ ] First iteration renders column headers with top border
- [ ] Each iteration renders data row with metrics
- [ ] Continuation row shows forward cost statistics
- [ ] Colors applied correctly based on rules
- [ ] Trend indicators (▲, ▼, ↓, ↑, →) appear correctly
- [ ] Bound change percentage shown
- [ ] Timing formatted correctly
- [ ] Row separator between iterations
- [ ] Table width adapts to terminal
- [ ] Unit tests for row rendering

## Implementation Guide

### Step 1: Add internal state

```rust
impl AdvancedRenderer {
    // Track if we've printed the header row
    fn is_first_iteration(&self, ctx: &DisplayContext) -> bool {
        ctx.iteration == 1
    }
}
```

### Step 2: Implement table column layout

```rust
struct ColumnLayout {
    widths: Vec<usize>,
    headers: Vec<String>,
    alignments: Vec<Alignment>,
}

impl AdvancedRenderer {
    fn get_column_layout(&self) -> ColumnLayout {
        ColumnLayout {
            widths: vec![4, 14, 14, 14, 6, 15],
            headers: vec![
                "Iter".to_string(),
                "Lower Bound ($)".to_string(),
                "Simul Cost ($)".to_string(),
                "1st Stage ($)".to_string(),
                "Gap %".to_string(),
                "Time (fwd/bwd)".to_string(),
            ],
            alignments: vec![
                Alignment::Right,
                Alignment::Center,
                Alignment::Center,
                Alignment::Center,
                Alignment::Right,
                Alignment::Center,
            ],
        }
    }
}
```

### Step 3: Implement render_iteration

```rust
fn render_iteration(&self, ctx: &DisplayContext) -> String {
    if !ctx.should_print {
        return String::new();
    }
    
    let layout = self.get_column_layout();
    let mut lines = Vec::new();
    
    // First iteration: render table header
    if self.is_first_iteration(ctx) {
        lines.push(self.render_table_top(&layout));
        lines.push(self.render_header_row(&layout));
        lines.push(self.render_separator(&layout));
    }
    
    // Render data row
    lines.push(self.render_data_row(ctx, &layout));
    
    // Render continuation with statistics
    lines.push(self.render_stats_continuation(ctx, &layout));
    
    // Add separator for next iteration
    lines.push(self.render_separator(&layout));
    
    lines.join("\n")
}
```

### Step 4: Implement render_data_row

```rust
fn render_data_row(&self, ctx: &DisplayContext, layout: &ColumnLayout) -> String {
    let border = self.border_style.chars().unwrap();
    
    // Format each cell
    let iter_cell = format!("{:>4}", ctx.iteration);
    
    let bound_cell = self.format_bound_cell(
        ctx.lower_bound,
        ctx.previous_lower_bound,
    );
    
    let cost_cell = format_cost(ctx.forward_cost_stats.mean, true);
    
    let first_stage_cell = format_cost(ctx.first_stage_bound, true);
    
    let gap_cell = self.format_gap_cell(
        ctx.gap_percentage(),
        ctx.previous_gap_percentage(),
    );
    
    let timing_cell = format_timing_pair(
        ctx.forward_timing.total,
        ctx.backward_timing.total,
    );
    
    // Build row with cells
    let cells = vec![
        iter_cell,
        bound_cell,
        cost_cell,
        first_stage_cell,
        gap_cell,
        timing_cell,
    ];
    
    self.format_row(&cells, layout, border.vertical)
}
```

### Step 5: Implement format_bound_cell

```rust
fn format_bound_cell(&self, current: f64, previous: Option<f64>) -> String {
    let value = format_cost(current, true);
    let indicator = bound_change_indicator(current, previous);
    
    let result = format!("{} {}", value, indicator);
    
    // Apply color
    let direction = bound_trend(current, previous, &TrendConfig::default());
    match direction {
        TrendDirection::Improving => colorize(&result, SemanticColor::Good, &self.color_config),
        TrendDirection::Worsening => colorize(&result, SemanticColor::Bad, &self.color_config),
        _ => result,
    }
}
```

### Step 6: Implement format_gap_cell

```rust
fn format_gap_cell(&self, current: f64, previous: Option<f64>) -> String {
    let gap_text = format!("{:5.1}", current);
    let trend = gap_trend(current, previous, &TrendConfig::default());
    let arrow = trend_arrow(trend);
    
    let result = format!("{}{}", gap_text, arrow);
    color_gap_percentage(current, &result, &self.color_config)
}
```

### Step 7: Implement render_stats_continuation

```rust
fn render_stats_continuation(&self, ctx: &DisplayContext, layout: &ColumnLayout) -> String {
    let border = self.border_style.chars().unwrap();
    
    // Calculate bound change if available
    let bound_change = if let Some(prev) = ctx.previous_lower_bound {
        let pct = ((ctx.lower_bound - prev) / prev.abs()) * 100.0;
        format_percentage_change(pct)
    } else {
        String::new()
    };
    
    // Format forward cost statistics
    let stats = format_cost_stats(&ctx.forward_cost_stats, &StatisticsFormat::default());
    
    // First column: bound change, rest: stats spanning remaining columns
    let total_width: usize = layout.widths.iter().sum::<usize>() + layout.widths.len() - 1;
    let col1_width = layout.widths[0];
    let remaining_width = total_width - col1_width - 1;
    
    format!("{} {:>col1$} {} {:<remaining$} {}",
        border.vertical,
        "",
        border.vertical,
        format!("{} {}", bound_change, stats),
        border.vertical,
        col1 = col1_width,
        remaining = remaining_width
    )
}
```

### Patterns to Follow

- Use components for all formatting
- Keep color application at cell level
- Handle missing previous values gracefully

### Pitfalls to Avoid

- ⚠️ ANSI codes affect visual width - use plain text for calculations
- ⚠️ Handle first iteration specially (no previous values)
- ⚠️ Ensure row separators align with columns
- ⚠️ Unicode width can differ from byte length

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_render_iteration_first() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 100);
    let ctx = DisplayContext {
        iteration: 1,
        total_iterations: 10,
        should_print: true,
        lower_bound: 101480.0,
        previous_lower_bound: None,
        forward_cost_stats: CostStatistics {
            mean: 128230.0,
            std_dev: 3200.0,
            min: 124000.0,
            max: 135000.0,
            count: 4,
        },
        first_stage_bound: 101480.0,
        forward_timing: ForwardTimingOutput { total: Duration::from_millis(18), .. },
        backward_timing: BackwardTimingOutput { total: Duration::from_millis(34), .. },
        ..Default::default()
    };
    
    let output = renderer.render_iteration(&ctx);
    
    // Should contain table header
    assert!(output.contains("Iter"));
    assert!(output.contains("Lower Bound"));
    assert!(output.contains("Gap %"));
    
    // Should contain iteration data
    assert!(output.contains("1")); // iteration number
    assert!(output.contains("1.01e+05")); // lower bound
}

#[test]
fn test_render_iteration_with_improvement() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 100);
    let ctx = DisplayContext {
        iteration: 2,
        lower_bound: 120410.0,
        previous_lower_bound: Some(101480.0),
        should_print: true,
        ..Default::default()
    };
    
    let output = renderer.render_iteration(&ctx);
    
    // Should show improvement indicator
    assert!(output.contains("▲") || output.contains("+"));
}

#[test]
fn test_render_iteration_stats_continuation() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 100);
    let ctx = DisplayContext {
        iteration: 1,
        should_print: true,
        forward_cost_stats: CostStatistics {
            mean: 128230.0,
            std_dev: 3200.0,
            min: 124000.0,
            max: 135000.0,
            count: 4,
        },
        ..Default::default()
    };
    
    let output = renderer.render_iteration(&ctx);
    
    // Should contain statistics
    assert!(output.contains("μ="));
    assert!(output.contains("σ="));
    assert!(output.contains("n=4"));
}

#[test]
fn test_render_iteration_skip_when_not_print() {
    let renderer = AdvancedRenderer::new(ColorConfig::new(false), 100);
    let ctx = DisplayContext {
        should_print: false,
        ..Default::default()
    };
    
    assert!(renderer.render_iteration(&ctx).is_empty());
}
```

## Documentation Requirements

- [ ] Doc comments on render_iteration explaining output format
- [ ] Inline comments for complex formatting logic
- [ ] Note about first iteration special handling

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Most complex rendering logic with multiple columns, colors, and continuation rows. Many edge cases.

## Definition of Done

- [ ] Implementation complete
- [ ] First iteration shows table header
- [ ] Data row shows all metrics
- [ ] Continuation row shows statistics
- [ ] Colors applied correctly
- [ ] Trend indicators work
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
