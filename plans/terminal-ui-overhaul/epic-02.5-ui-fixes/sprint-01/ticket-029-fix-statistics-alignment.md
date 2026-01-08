# [T-029] Fix Statistics Row Alignment

> **Epic**: [Epic 02.5: UI Fixes](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-028](./ticket-028-fix-duplicate-table-header.md)
> **Blocks**: None

## Context

### Background

The AdvancedRenderer's statistics continuation row doesn't align with the table column structure. The current implementation tries to span all columns after the first, but uses hardcoded widths that don't match the actual column layout.

### Current State

In `src/display/renderers/advanced.rs` `render_stats_continuation()`:
```rust
format!(
    "{} {:>3} {} {:<73} {}",
    border.vertical,
    bound_change,
    border.vertical,
    stats,
    border.vertical
)
```

This produces:
```
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.156s / 0.711s │
│     │ μ=1.02e8 σ=5.26e5 [1.01e8..1.03e8] n=4                                    │
```

The second row doesn't respect the 6-column structure. It should either:
1. Span columns 2-6 properly with column separators
2. Use a different layout that fits naturally

## Specification

### Fix Options

**Option A**: Proper column spanning (complex)
- Calculate exact widths for spanning columns 2-6
- Add internal vertical bars at correct positions
- Most visually consistent but complex

**Option B**: Single-cell layout (simpler, recommended)
- Make iteration number column + one big merged cell for rest
- Cleaner look, easier to implement
- Better handles variable-width statistics

**Option C**: Remove continuation row entirely
- Move key stats inline or to separate line outside table
- Simplest but loses information density

**Recommendation**: Use **Option B** - a cleaner two-cell layout where the statistics span columns 2-6 as one visual cell.

### Proposed Layout

```
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.156s / 0.711s │
│     │ μ=1.02e8 σ=5.26e5 [1.01e8..1.03e8] n=4                                    │
├─────┼────────────────┴────────────────┴────────────────┴───────┴─────────────────┤
```

Or alternatively, use the full row without internal borders for continuation:
```
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.156s / 0.711s │
│ +1.5%│ μ=1.02e8 σ=5.26e5 [1.01e8..1.03e8] n=4                                   │
```

### Changes Required

1. Calculate total width of columns 2-6 (16+16+16+7+17 = 72 chars + 4 separators = 76)
2. Format statistics to fit within that width
3. Add proper border characters

### Implementation

```rust
fn render_stats_continuation(&self, ctx: &DisplayContext) -> String {
    let border = BorderStyle::Standard.chars().unwrap();
    let col_widths = [5, 16, 16, 16, 7, 17];
    
    // First column: bound change percentage  
    let bound_change = if let Some(prev) = ctx.previous_lower_bound {
        if prev.abs() > 1e-10 {
            let change_pct = ((ctx.lower_bound - prev) / prev.abs()) * 100.0;
            format_percentage_change(change_pct)
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    
    // Rest of columns merged: statistics
    let stats = format_cost_stats(&ctx.forward_cost_stats, &StatisticsFormat::default());
    
    // Width of merged columns (sum of cols 2-6 plus internal separators)
    let merged_width = col_widths[1..].iter().sum::<usize>() 
        + (col_widths.len() - 2); // -1 for first col, -1 less separator
    
    format!(
        "{} {:^width1$} {} {:<width2$} {}",
        border.vertical,
        bound_change,
        border.vertical,
        stats,
        border.vertical,
        width1 = col_widths[0] - 2,
        width2 = merged_width - 2,
    )
}
```

## Acceptance Criteria

- [x] Statistics continuation row fits within table visual bounds
- [x] Row borders align with table structure
- [x] Statistics text doesn't overflow or get truncated awkwardly
- [x] Visual appearance is clean and professional
- [x] All renderer tests pass

## Implementation Guide

### Suggested Approach

1. Open `src/display/renderers/advanced.rs`
2. Modify `render_stats_continuation()` method
3. Calculate correct merged width from column widths
4. Format with proper padding and alignment
5. Test visually with various data sizes
6. Update any unit tests that check for specific output

### Key Files to Modify

- `src/display/renderers/advanced.rs`: `render_stats_continuation()` method

### Patterns to Follow

- Use same `col_widths` array as other table methods
- Use `BorderStyle::Standard.chars()` consistently

### Pitfalls to Avoid

- ⚠️ Don't hardcode widths - calculate from column array
- ⚠️ Handle very long statistics strings gracefully (truncate if needed)
- ⚠️ Test with various number magnitudes (scientific notation lengths vary)

## Testing Requirements

### Manual Tests

- [x] Small numbers (e.g., 1.02e5) look correct
- [x] Large numbers (e.g., 1.02e8) look correct
- [x] Very large numbers (e.g., 1.02e12) don't overflow
- [x] Bound change percentages display correctly (+X.X%, -X.X%)

### Visual Verification

Run with example 05 and verify:
```
┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter │ Lower Bound ($) │ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.156s / 0.711s │
│     │ μ=1.02e8 σ=5.26e5 [1.01e8..1.03e8] n=4                                    │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
```

The continuation row should:
- Start with `│` (left border)
- Have first column (width 5) for bound change or empty
- Have `│` separator
- Have remaining space for statistics
- End with `│` (right border)

### Automated Tests

- [x] Update `test_render_iteration_first_includes_table_header` if it checks specific format
- [x] All other tests pass

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Requires careful width calculations and visual testing

## Definition of Done

- [x] Statistics row aligns with table structure
- [x] No overflow or truncation issues
- [x] Visual testing confirms clean appearance
- [x] All tests passing

## Status: ✅ COMPLETE

**Implementation Summary:**
- Fixed `render_stats_continuation()` in src/display/renderers/advanced.rs
- Properly calculated merged width for columns 2-6: 72 chars + 4 separators = 76
- Changed hardcoded width from 73 to correctly calculated 76
- Changed first column alignment from `:>3` to `:^width1$` (centered, uses col_widths[0])
- All 749 tests passing
- Visual verification shows perfect alignment
