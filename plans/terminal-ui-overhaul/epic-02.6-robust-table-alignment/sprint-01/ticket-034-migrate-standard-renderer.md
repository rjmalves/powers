# [T-034] Migrate StandardRenderer to new utilities

> **Epic**: [Epic 2.6: Robust Table Alignment](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-032](./ticket-032-create-table-format-utils.md)  
> **Blocks**: [T-035](./ticket-035-add-alignment-tests.md)

## Context

### Background

The `StandardRenderer` uses similar manual table construction to `AdvancedRenderer` but with fewer columns (no 1st Stage column). This ticket migrates it to use the centralized utilities.

### Current State

`StandardRenderer` (src/display/renderers/standard.rs) has:
- `col_widths` arrays defined in 2 locations
- Manual format strings that can expand
- Simpler structure (no statistics continuation row)

### Files to Read Before Starting

- `src/display/renderers/standard.rs` - Current implementation
- `src/display/components/table_format.rs` - New utilities (from T-032)
- `src/display/renderers/advanced.rs` - Reference for migration pattern (from T-033)

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
       let config = TableColumnConfig::standard();
       let border = BorderStyle::Standard.chars().unwrap();
       
       let top = build_top_border(&config.widths, &border);
       let header = build_header_row(&config, &border);
       let sep = build_separator(&config.widths, &border);
       
       format!("{}\n{}\n{}", top, header, sep)
   }
   ```

3. **Replace `render_data_row()`**:
   ```rust
   fn render_data_row(&self, ctx: &DisplayContext) -> String {
       let config = TableColumnConfig::standard();
       let border = BorderStyle::Standard.chars().unwrap();
       
       let cells = vec![
           format_cell(&format!("{}", ctx.iteration), config.widths[0], Alignment::Right),
           format_cell(&format_cost(ctx.lower_bound, true), config.widths[1], Alignment::Center),
           format_cell(&format_cost(ctx.forward_cost_stats.mean, true), config.widths[2], Alignment::Center),
           format_cell(&format_gap(ctx.gap_percent), config.widths[3], Alignment::Center),
           format_cell(&format_timing_pair(ctx.timing.forward.total, ctx.timing.backward.total), config.widths[4], Alignment::Center),
       ];
       
       build_row(&cells, &border)
   }
   ```

4. **Replace `render_separator()`**:
   ```rust
   fn render_separator(&self) -> String {
       let config = TableColumnConfig::standard();
       let border = BorderStyle::Standard.chars().unwrap();
       build_separator(&config.widths, &border)
   }
   ```

5. **Replace `render_table_bottom()`**:
   ```rust
   fn render_table_bottom(&self) -> String {
       let config = TableColumnConfig::standard();
       let border = BorderStyle::Standard.chars().unwrap();
       build_bottom_border(&config.widths, &border)
   }
   ```

6. **Remove all hardcoded `col_widths` arrays**

### Key Differences from AdvancedRenderer

- Standard has 5 columns instead of 6 (no "1st Stage ($)")
- No statistics continuation row
- Simpler gap formatting (no trend arrows)
- No bound change percentage display

## Acceptance Criteria

- [ ] No hardcoded `col_widths` arrays anywhere in standard.rs
- [ ] All table rows use `format_cell()` for content
- [ ] All borders use `build_*_border()` functions
- [ ] Table alignment is perfect (all rows same width)
- [ ] `cargo build -j 1` succeeds
- [ ] `cargo test -j 1 -- --test-threads=1` passes
- [ ] Manual terminal testing shows correct alignment

## Testing Requirements

### Manual Testing

Run with a real example:
```bash
cargo run -j 1 -- train examples/01-newsvendor --profile standard
```

Verify:
- [ ] All table borders align vertically
- [ ] Header row aligns with data rows
- [ ] Colors appear correctly (if enabled)

### Unit Tests

Existing tests should continue to pass.

## Implementation Guide

### Key Files to Modify

- `src/display/renderers/standard.rs`

### Patterns to Follow

- Follow the same pattern as AdvancedRenderer migration
- Use `TableColumnConfig::standard()` for configuration

### Pitfalls to Avoid

- ⚠️ Standard uses 5 columns, not 6
- ⚠️ Don't add AdvancedRenderer features to StandardRenderer
- ⚠️ Use `-j 1` for builds to avoid RAM issues

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simpler than AdvancedRenderer, pattern already established

## Definition of Done

- [x] All methods migrated to new utilities
- [x] No hardcoded width arrays
- [x] All tests passing
- [x] Manual verification of alignment
- [x] No clippy warnings
- [x] Formatted with rustfmt

---

**Status**: ✅ COMPLETE
