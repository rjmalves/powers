# [TABLED-004] Migrate AdvancedRenderer table header

> **Epic**: [Epic 2: Renderer Migration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 1 complete  
> **Blocks**: [TABLED-005](./ticket-005-advanced-iteration.md)

## Context

### Background

The `AdvancedRenderer::render_table_top_and_header()` method manually constructs the table header using hardcoded column widths and border characters. This is the first method to migrate to `tabled`.

### Relation to Epic

This ticket establishes the pattern for migrating other renderer methods.

### Current State

Lines 86-142 in `advanced.rs`:
- Hardcoded `col_widths = [6, 17, 16, 15, 7, 17]`
- Manual border construction with `border.top_left`, `border.top_tee`, etc.
- Manual header row formatting with `format!(" {:^width$} ", ...)`
- Manual separator construction

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Lines 86-142, current implementation
- `src/display/components/tabled_utils.rs` - Utilities from Epic 1
- `src/display/components/statistics.rs` - Format functions used

## Specification

### Inputs

- `&self` reference to renderer
- Color configuration from `self.color_config`

### Outputs

- `String` containing the table top border, header row, and separator

### Behavior

Replace manual construction with `tabled::Builder`:
1. Create builder
2. Add header row
3. Build table with `Style::modern()`
4. Apply centered alignment to header
5. Return rendered string

The output should be functionally equivalent to the current output, but may have minor differences in exact spacing due to automatic width calculation.

### Error Handling

No errors - table construction always succeeds.

## Acceptance Criteria

- [ ] `render_table_top_and_header()` uses `tabled::Builder`
- [ ] No references to `col_widths` array
- [ ] No manual border character construction
- [ ] Output contains all header labels: "Iter", "Lower Bound ($)", etc.
- [ ] Output uses modern box-drawing characters (┌, │, ─)
- [ ] Headers are centered
- [ ] Method is ~10-15 lines instead of ~40
- [ ] `cargo test` passes for this method
- [ ] `cargo clippy` clean

## Implementation Guide

### Suggested Approach

1. Add imports at top of `advanced.rs`:
   ```rust
   use tabled::{
       builder::Builder,
       settings::{Alignment, Modify, Style, object::Rows},
   };
   ```

2. Replace method body:
   ```rust
   fn render_table_top_and_header(&self) -> String {
       let mut builder = Builder::default();
       
       builder.push_record([
           "Iter",
           "Lower Bound ($)",
           "Simul Cost ($)",
           "1st Stage ($)",
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

3. Remove unused imports:
   - Remove `use crate::display::components::table::BorderStyle;` if no longer needed
   - Keep it if other methods still use it (will be cleaned in later tickets)

4. Run tests and fix any failures

### Key Files to Modify

- `src/display/renderers/advanced.rs`: Replace `render_table_top_and_header()` method

### Patterns to Follow

- Keep the method signature unchanged
- Use `Style::modern()` to match current `BorderStyle::Standard` appearance

### Pitfalls to Avoid

- ⚠️ Don't remove `BorderStyle` import yet - other methods still use it
- ⚠️ The `tabled` output includes newlines between rows - may need to adjust calling code
- ⚠️ Don't change `render_iteration()` yet - that's a separate ticket

## Testing Requirements

### Unit Tests

- [ ] `test_render_header_contains_brand()` still passes
- [ ] `test_render_iteration_first_includes_table_header()` still passes
- [ ] New test: Header contains all column labels
- [ ] New test: Header uses box-drawing characters

### Integration Tests

- [ ] Run a training iteration and verify output looks correct

### Manual Testing

- [ ] Run `cargo run -- train` on a test case
- [ ] Verify table header appears correctly aligned in terminal

## Documentation Requirements

- [ ] Update method doc comment to mention `tabled` usage

## Dependencies

- **Blocked By**: Epic 1 (TABLED-001, TABLED-002, TABLED-003)
- **Blocks**: TABLED-005
- **Related**: TABLED-007 (similar work for StandardRenderer)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward replacement, clear before/after

## Definition of Done

- [ ] Implementation complete
- [ ] All tests passing
- [ ] Code reduction of ~30 lines
- [ ] Manual terminal test shows correct output
- [ ] Code reviewed
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
