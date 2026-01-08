# [TABLED-005] Migrate AdvancedRenderer iteration rows

> **Epic**: [Epic 2: Renderer Migration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TABLED-004](./ticket-004-advanced-header.md)  
> **Blocks**: [TABLED-006](./ticket-006-advanced-summary.md)

## Context

### Background

The `AdvancedRenderer` has three methods for rendering iteration data:
1. `render_data_row()` - Single iteration row with metrics
2. `render_stats_continuation()` - Statistics row spanning columns
3. `render_separator()` - Horizontal separator between iterations

These need to be migrated to use `tabled`, with special handling for the statistics continuation row.

### Relation to Epic

This is the most complex migration ticket, as it involves the statistics row that spans columns.

### Current State

Lines 160-267 in `advanced.rs`:
- `render_data_row()`: Manual cell formatting with hardcoded widths
- `render_stats_continuation()`: Merged column calculation with fragile formula
- `render_separator()`: Manual separator construction

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Lines 160-267
- `src/display/components/statistics.rs` - `format_cost_stats`, `format_timing_pair`
- `src/display/components/indicators.rs` - `trend_arrow_colored`
- `src/display/context.rs` - `DisplayContext` fields

## Specification

### Inputs

- `&DisplayContext` - All iteration metrics
- `&self` - Renderer with color config

### Outputs

- `String` containing data row, optional stats row, and separator

### Behavior

#### Option A: Rebuild table each iteration (Recommended)

Build a complete table each iteration containing:
1. Header row (iteration 1 only)
2. Data row
3. Statistics panel row

```rust
fn render_iteration(&self, ctx: &DisplayContext, config: &DisplayConfig) -> String {
    let mut builder = Builder::default();
    
    if ctx.iteration == 1 {
        builder.push_record(["Iter", "Lower Bound ($)", ...]);
    }
    
    // Data row
    builder.push_record([
        format!("{}", ctx.iteration),
        format!("{}{}", format_cost(ctx.lower_bound, true), bound_indicator),
        format_cost(ctx.forward_cost_stats.mean, true),
        format_cost(ctx.first_stage_bound, true),
        format!("{}{}", gap_value, gap_arrow),
        format_timing_pair(ctx.forward_timing.total, ctx.backward_timing.total),
    ]);
    
    let mut table = builder.build();
    table.with(Style::modern());
    
    // Add statistics as panel
    let stats = format_cost_stats(&ctx.forward_cost_stats, &StatisticsFormat::default());
    // ... build output
}
```

#### Option B: Render row-by-row without full table structure

Print individual rows with manual borders, similar to current approach but using `tabled` for width calculation.

**Recommendation**: Option A is cleaner but may show separator differences. Prototype and evaluate.

### Error Handling

No errors - all paths produce valid output.

## Acceptance Criteria

- [ ] `render_iteration()` produces complete iteration output
- [ ] No hardcoded `col_widths` arrays
- [ ] No manual border character construction  
- [ ] Statistics row displays mean, std_dev, min, max, count
- [ ] Trend indicators (↑, ↓) still appear
- [ ] Gap percentage is colored appropriately
- [ ] Timing displays in `fwd/bwd` format
- [ ] Output is properly aligned (no column drift)
- [ ] Method is ~50 lines instead of ~100
- [ ] All iteration tests pass

## Implementation Guide

### Suggested Approach

1. **Simplify the structure**: Instead of separate `render_data_row()`, `render_stats_continuation()`, and `render_separator()` methods, consolidate into one iteration rendering flow.

2. **Handle statistics row**: Use `tabled` Panel or a second row with merged content:
   ```rust
   // Option 1: Panel (full-width)
   table.with(Panel::horizontal(2, stats_text));
   
   // Option 2: Second row with empty first cell
   builder.push_record(["", &format!("      {}", stats), "", "", "", ""]);
   ```

3. **Handle progressive display**: For iteration 1, include header. For subsequent iterations, just the data.

4. **Preserve coloring**: Color functions return ANSI-escaped strings. `tabled` with `ansi` feature handles these correctly for width calculation.

### Code Template

```rust
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

    let mut builder = Builder::default();

    // Header on first iteration
    if ctx.iteration == 1 {
        builder.push_record([
            "Iter",
            "Lower Bound ($)",
            "Simul Cost ($)",
            "1st Stage ($)",
            "Gap %",
            "Time (fwd/bwd)",
        ]);
    }

    // Format data cells
    let iter_cell = format!("{}", ctx.iteration);
    
    let bound_indicator = if let Some(prev) = ctx.previous_lower_bound {
        let trend = bound_trend(ctx.lower_bound, Some(prev), &TrendConfig::default());
        format!(" {}", trend_arrow_colored(trend, &renderer.color_config))
    } else {
        String::new()
    };
    let bound_cell = format!("{}{}", format_cost(ctx.lower_bound, true), bound_indicator);
    
    let simul_cell = format_cost(ctx.forward_cost_stats.mean, true);
    let first_stage_cell = format_cost(ctx.first_stage_bound, true);
    
    let gap_value = format_gap(ctx.gap_percent);
    let gap_trend_dir = gap_trend(
        ctx.gap_percent,
        ctx.previous_lower_bound.map(|prev| {
            ((ctx.forward_cost_stats.mean - prev) / prev.abs()) * 100.0
        }),
        &TrendConfig::default(),
    );
    let gap_arrow = trend_arrow_colored(gap_trend_dir, &renderer.color_config);
    let gap_cell = color_gap_percentage(
        ctx.gap_percent,
        &format!("{}{}", gap_value, gap_arrow),
        &renderer.color_config,
    );
    
    let timing_cell = format_timing_pair(
        ctx.forward_timing.total,
        ctx.backward_timing.total,
    );

    // Data row
    builder.push_record([
        iter_cell,
        bound_cell,
        simul_cell,
        first_stage_cell,
        gap_cell,
        timing_cell,
    ]);

    // Statistics row (as a regular row with first cell being change %)
    let bound_change = if let Some(prev) = ctx.previous_lower_bound {
        if prev.abs() > 1e-10 {
            format_percentage_change(((ctx.lower_bound - prev) / prev.abs()) * 100.0)
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    
    let stats = format_cost_stats(&ctx.forward_cost_stats, &StatisticsFormat::default());
    
    // Second row: change % in first column, stats spanning the rest
    builder.push_record([
        bound_change,
        stats.clone(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
    ]);

    let mut table = builder.build();
    table.with(Style::modern());
    
    if ctx.iteration == 1 {
        table.with(Modify::new(Rows::first()).with(Alignment::center()));
    }

    let mut output = table.to_string();
    output.push('\n');
    output
}
```

### Key Files to Modify

- `src/display/renderers/advanced.rs`:
  - Replace `render_data_row()`, `render_stats_continuation()`, `render_separator()`
  - Update `render_iteration()` to consolidate logic

### Patterns to Follow

- Use `format_cost()`, `format_gap()`, etc. from `statistics.rs`
- Use `color_gap_percentage()` from `color.rs`
- Use `trend_arrow_colored()` from `indicators.rs`

### Pitfalls to Avoid

- ⚠️ Statistics row alignment: May need to span columns or use Panel
- ⚠️ Row separators: `Style::modern()` adds separators between all rows by default
- ⚠️ Header row: Only show on iteration 1
- ⚠️ ANSI codes: Must use `ansi` feature for correct width

### Note on Row Separators

`Style::modern()` puts separators between every row. To get separator after header only:

```rust
table.with(Style::modern().remove_horizontal());
// Then add horizontal line after header
table.with(HorizontalLine::new(1, Style::modern().get_horizontal()));
```

Or use `Style::sharp()` which only has header separator.

## Testing Requirements

### Unit Tests

- [ ] `test_render_iteration_first_includes_table_header()` passes
- [ ] `test_render_iteration_skip_when_not_print()` passes
- [ ] Iteration 1 output contains "Iter", "Lower Bound", etc.
- [ ] Iteration 2+ output does NOT contain header row
- [ ] Statistics row contains formatted stats
- [ ] Gap percentage is colored

### Manual Testing

- [ ] Run training and verify table looks correct
- [ ] Verify columns are aligned across iterations
- [ ] Verify ANSI colors appear correctly

## Documentation Requirements

- [ ] Update method doc comments
- [ ] Remove docs for deleted methods (render_data_row, etc.)

## Dependencies

- **Blocked By**: TABLED-004
- **Blocks**: TABLED-006, TABLED-008
- **Related**: TABLED-007 (similar work for StandardRenderer)

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Statistics row and progressive display add complexity; may need iteration

## Definition of Done

- [ ] Implementation complete
- [ ] All iteration-related tests passing
- [ ] Statistics row displays correctly
- [ ] Manual terminal test shows correct alignment
- [ ] ~50 lines of code reduction
- [ ] Code reviewed
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
