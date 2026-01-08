# [T-033] Migrate AdvancedRenderer to new utilities

> **Epic**: [Epic 2.6: Robust Table Alignment](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-032](./ticket-032-create-table-format-utils.md)  
> **Blocks**: [T-035](./ticket-035-add-alignment-tests.md)

## Context

### Background

The `AdvancedRenderer` currently has hardcoded `col_widths` arrays in multiple locations and uses manual format strings that don't prevent expansion. This ticket migrates the renderer to use the new centralized `TableColumnConfig` and `format_cell()` utilities.

### Current State

`AdvancedRenderer` (src/display/renderers/advanced.rs) has:
- `col_widths = [6, 17, 16, 15, 7, 17]` defined in 4+ locations
- Manual `format!(" {:^14} ", ...)` calls that can expand
- Statistics continuation row with complex width calculation
- Manual border construction duplicated across methods

### Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Current implementation
- `src/display/components/table_format.rs` - New utilities (from T-032)
- `src/display/components/table.rs` - BorderStyle enum

## Specification

### Changes Required

1. **Add import at top of file**:
   ```rust
   use crate::display::components::table_format::{
       Alignment, TableColumnConfig, format_cell, build_row,
       build_top_border, build_separator, build_bottom_border,
       build_header_row,
   };
   ```

2. **Replace `render_table_top_and_header()`**:
   ```rust
   fn render_table_top_and_header(&self) -> String {
       let config = TableColumnConfig::advanced();
       let border = BorderStyle::Standard.chars().unwrap();
       
       let top = build_top_border(&config.widths, &border);
       let header = build_header_row(&config, &border);
       let sep = build_separator(&config.widths, &border);
       
       format!("{}\n{}\n{}", top, header, sep)
   }
   ```

3. **Replace `render_separator()`**:
   ```rust
   fn render_separator(&self) -> String {
       let config = TableColumnConfig::advanced();
       let border = BorderStyle::Standard.chars().unwrap();
       build_separator(&config.widths, &border)
   }
   ```

4. **Replace `render_data_row()`**:
   Use `format_cell()` for each column with proper alignment:
   ```rust
   fn render_data_row(&self, ctx: &DisplayContext) -> String {
       let config = TableColumnConfig::advanced();
       let border = BorderStyle::Standard.chars().unwrap();
       
       let cells = vec![
           format_cell(&format!("{}", ctx.iteration), config.widths[0], Alignment::Right),
           format_cell(&self.format_lower_bound(ctx), config.widths[1], Alignment::Center),
           format_cell(&format_cost(ctx.forward_cost_stats.mean, true), config.widths[2], Alignment::Center),
           format_cell(&format_cost(ctx.first_stage_bound, true), config.widths[3], Alignment::Center),
           format_cell(&self.format_gap(ctx), config.widths[4], Alignment::Center),
           format_cell(&self.format_timing(ctx), config.widths[5], Alignment::Center),
       ];
       
       build_row(&cells, &border)
   }
   ```

5. **Replace `render_stats_continuation()`**:
   The continuation row needs special handling for merged columns:
   ```rust
   fn render_stats_continuation(&self, ctx: &DisplayContext) -> String {
       let config = TableColumnConfig::advanced();
       let border = BorderStyle::Standard.chars().unwrap();
       
       // First column: bound change percentage
       let bound_change = self.format_bound_change(ctx);
       let first_cell = format_cell(&bound_change, config.widths[0], Alignment::Center);
       
       // Merged columns 2-6: statistics
       let stats = format_cost_stats(
           ctx.forward_cost_stats.mean,
           ctx.forward_cost_stats.std_dev,
           ctx.forward_cost_stats.min,
           ctx.forward_cost_stats.max,
           ctx.forward_cost_stats.count,
           &StatisticsFormat::default(),
       );
       
       // Calculate merged width: sum of columns 2-6 + internal separators
       let merged_width: usize = config.widths[1..].iter().sum::<usize>() 
           + (config.widths.len() - 2); // internal │ chars
       
       let merged_cell = format_cell(&stats, merged_width, Alignment::Left);
       
       format!(
           "{}{}{}{}{}",
           border.vertical,
           first_cell,
           border.vertical,
           merged_cell,
           border.vertical
       )
   }
   ```

6. **Replace `render_table_bottom()`**:
   ```rust
   fn render_table_bottom(&self) -> String {
       let config = TableColumnConfig::advanced();
       let border = BorderStyle::Standard.chars().unwrap();
       build_bottom_border(&config.widths, &border)
   }
   ```

7. **Remove all hardcoded `col_widths` arrays**

### Helper Methods

Extract formatted content to helper methods for clarity:

```rust
impl AdvancedRenderer {
    fn format_lower_bound(&self, ctx: &DisplayContext) -> String {
        let value = format_cost(ctx.lower_bound, true);
        let indicator = if let Some(prev) = ctx.previous_lower_bound {
            if (ctx.lower_bound - prev).abs() / prev.abs() > 0.001 {
                trend_arrow_colored(
                    bound_trend(ctx.lower_bound, prev),
                    &TrendConfig::bound(),
                    &self.color_config,
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        format!("{}{}", value, indicator)
    }
    
    fn format_gap(&self, ctx: &DisplayContext) -> String {
        let value = format_gap(ctx.gap_percent);
        let trend = gap_trend(
            ctx.gap_percent,
            ctx.previous_gap_percent,
            ctx.gap_change_threshold,
        );
        let indicator = trend_arrow_colored(
            trend,
            &TrendConfig::gap(),
            &self.color_config,
        );
        format!("{}{}", value, indicator)
    }
    
    fn format_timing(&self, ctx: &DisplayContext) -> String {
        format_timing_pair(
            ctx.timing.forward.total,
            ctx.timing.backward.total,
        )
    }
    
    fn format_bound_change(&self, ctx: &DisplayContext) -> String {
        if let Some(prev) = ctx.previous_lower_bound {
            if prev.abs() > 1e-10 {
                let change = (ctx.lower_bound - prev) / prev.abs() * 100.0;
                format_percentage_change(change, &self.color_config)
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    }
}
```

## Acceptance Criteria

- [ ] No hardcoded `col_widths` arrays anywhere in advanced.rs
- [ ] All table rows use `format_cell()` for content
- [ ] All borders use `build_*_border()` functions
- [ ] Table alignment is perfect (all rows same width)
- [ ] ANSI-colored content aligns correctly
- [ ] Statistics continuation row aligns with table structure
- [ ] `cargo build -j 1` succeeds
- [ ] `cargo test -j 1 -- --test-threads=1` passes
- [ ] Manual terminal testing shows correct alignment

## Testing Requirements

### Manual Testing

Run with a real example to verify alignment:
```bash
cargo run -j 1 -- train examples/01-newsvendor
```

Verify:
- [ ] All table borders align vertically
- [ ] Header row aligns with data rows
- [ ] Statistics row doesn't overflow
- [ ] Colors appear correctly
- [ ] Large numbers don't break alignment

### Unit Tests

Existing tests should continue to pass. Update any tests that check exact output strings.

## Implementation Guide

### Key Files to Modify

- `src/display/renderers/advanced.rs`

### Patterns to Follow

- Keep `ColorConfig` usage unchanged
- Keep `DisplayContext` interface unchanged
- Use the new utilities for ALL cell formatting

### Pitfalls to Avoid

- ⚠️ Don't forget to update the training summary table too
- ⚠️ The stats continuation row has different structure (1 + merged)
- ⚠️ Preserve trend arrows and color formatting
- ⚠️ Use `-j 1` for builds to avoid RAM issues

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Mechanical refactoring with clear patterns

## Definition of Done

- [x] All methods migrated to new utilities
- [x] No hardcoded width arrays
- [x] All tests passing
- [x] Manual verification of alignment
- [x] No clippy warnings
- [x] Formatted with rustfmt

---

**Status**: ✅ COMPLETE
