# [TABLED-007] Migrate StandardRenderer tables

> **Epic**: [Epic 2: Renderer Migration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 1 complete  
> **Blocks**: [TABLED-008](./ticket-008-update-tests.md)

## Context

### Background

The `StandardRenderer` is similar to `AdvancedRenderer` but simpler:
- 5 columns instead of 6 (no "1st Stage" column)
- No statistics continuation row
- Simpler training summary

The migration follows the same pattern as the Advanced renderer.

### Relation to Epic

This ticket parallelizes with TABLED-004/005/006 as it's independent.

### Current State

`standard.rs` lines 72-184:
- `render_table_top_and_header()`: Manual construction
- `render_data_row()`: Manual cell formatting
- `render_table_bottom()`: Manual bottom border

## Files to Read Before Starting

- `src/display/renderers/standard.rs` - Lines 72-184
- `src/display/renderers/advanced.rs` - After migration (for reference)

## Specification

### Inputs

Same as Advanced renderer:
- `&DisplayContext` for iteration
- `&TrainingResult` for summary
- `&DisplayConfig` for configuration

### Outputs

- Formatted table strings using `tabled`

### Behavior

Migrate all table-related methods to use `tabled::Builder`:
1. `render_table_top_and_header()` → Use Builder with 5 columns
2. `render_data_row()` → Remove, inline into `render_iteration()`
3. `render_table_bottom()` → Remove or simplify
4. `render_iteration()` → Consolidate table construction
5. `render_training_summary()` → Simplify

### Error Handling

No errors - all paths produce valid output.

## Acceptance Criteria

- [ ] All table methods use `tabled::Builder`
- [ ] No hardcoded `col_widths` arrays
- [ ] No manual border character construction
- [ ] 5 columns: Iter, Lower Bound, Simul Cost, Gap %, Time
- [ ] No statistics row (simpler than Advanced)
- [ ] All Standard renderer tests pass
- [ ] Code reduction of ~150 lines

## Implementation Guide

### Suggested Approach

1. **Update imports**:
   ```rust
   use tabled::{
       builder::Builder,
       settings::{Alignment, Modify, Style, object::Rows},
   };
   ```

2. **Replace `render_table_top_and_header()`**:
   ```rust
   fn render_table_top_and_header(&self) -> String {
       let mut builder = Builder::default();
       builder.push_record([
           "Iter",
           "Lower Bound ($)",
           "Simul Cost ($)",
           "Gap %",
           "Time (fwd/bwd)",
       ]);
       
       let mut table = builder.build();
       table
           .with(Style::modern())
           .with(Modify::new(Rows::first()).with(Alignment::center()));
       
       table.to_string()
   }
   ```

3. **Consolidate `render_iteration()`**:
   ```rust
   fn render_iteration(
       &self,
       ctx: &DisplayContext,
       config: &DisplayConfig,
   ) -> String {
       if !ctx.should_print {
           return String::new();
       }
       
       let color_config = ColorConfig::new(config.color_enabled);
       let mut output = String::new();
       
       if ctx.iteration == 1 {
           output.push_str(&self.render_table_top_and_header());
           output.push('\n');
       }
       
       // Build single-row table for this iteration
       let mut builder = Builder::default();
       builder.push_record([
           format!("{}", ctx.iteration),
           format_cost(ctx.lower_bound, true),
           format_cost(ctx.forward_cost_stats.mean, true),
           color_gap_percentage(ctx.gap_percent, &format!("{:.1}%", ctx.gap_percent), &color_config),
           format_timing_pair(ctx.forward_timing.total, ctx.backward_timing.total),
       ]);
       
       let table = builder.build().with(Style::modern()).to_string();
       
       // Extract just the data row (skip borders that duplicate header)
       // OR: Print full row with borders
       
       output.push_str(&table);
       output.push('\n');
       output
   }
   ```

4. **Simplify `render_training_summary()`**:
   - Remove call to `render_table_bottom()`
   - Keep summary text formatting

5. **Remove unused methods**:
   - `render_data_row()` - inlined
   - `render_table_bottom()` - no longer needed

### Key Files to Modify

- `src/display/renderers/standard.rs`: Full refactoring

### Patterns to Follow

- Same patterns as migrated Advanced renderer
- Simpler (no statistics row)

### Pitfalls to Avoid

- ⚠️ Standard has 5 columns, Advanced has 6 - don't copy/paste column lists
- ⚠️ No statistics row - simpler iteration output
- ⚠️ Gap coloring: Use `color_gap_percentage()` which handles ANSI

## Testing Requirements

### Unit Tests

- [ ] `test_render_header_contains_brand()` passes
- [ ] `test_render_header_cut_selection()` passes
- [ ] `test_standard_has_fewer_columns_than_advanced()` passes
- [ ] `test_no_statistics_continuation_row()` passes
- [ ] `test_first_iteration_includes_table_header()` passes
- [ ] `test_subsequent_iterations_no_header()` passes
- [ ] `test_skip_when_should_print_false()` passes
- [ ] `test_compact_training_summary()` passes
- [ ] `test_render_table_bottom()` - may need removal/update

### Manual Testing

- [ ] Run training with `--profile standard` flag
- [ ] Verify table alignment in terminal

## Documentation Requirements

- [ ] Update method doc comments
- [ ] Remove docs for deleted methods

## Dependencies

- **Blocked By**: Epic 1 complete
- **Blocks**: TABLED-008
- **Related**: TABLED-004, TABLED-005, TABLED-006 (parallel work)

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Similar to Advanced but simpler; can follow established patterns

## Definition of Done

- [ ] Implementation complete
- [ ] All Standard renderer tests passing
- [ ] ~150 lines of code removed
- [ ] Manual terminal test shows correct alignment
- [ ] Code reviewed
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
