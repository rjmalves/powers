# Epic 2: Renderer Migration

## Summary

Migrate `AdvancedRenderer` and `StandardRenderer` from manual table construction to using `tabled`. This is the core work that eliminates alignment issues and reduces code complexity.

## Scope

### Included

- Refactor `AdvancedRenderer` to use `tabled` for all tables
- Refactor `StandardRenderer` to use `tabled` for all tables
- Update all related unit tests
- Maintain identical visual output (modulo alignment fixes)

### Excluded

- `MinimalRenderer` (uses `ProgressBar`, no tables)
- `AutomationRenderer` (uses JSON, no tables)
- Removal of old `table.rs` (Epic 3)
- New features or visual changes

## Dependencies

- **Requires**: Epic 1 (Core Integration) - `tabled` dependency and utilities
- **Enables**: Epic 3 (Cleanup) - can remove old code after migration

## Acceptance Criteria

- [ ] `AdvancedRenderer` renders all tables via `tabled`
- [ ] `StandardRenderer` renders all tables via `tabled`
- [ ] All existing tests pass (with updated expected strings where needed)
- [ ] Manual terminal testing shows correct alignment
- [ ] No performance regression (subjective - display still feels instant)
- [ ] Code reduction: ~300 lines removed from `advanced.rs`, ~200 from `standard.rs`

## Technical Approach

### Migration Strategy

**Incremental migration per renderer**:
1. Migrate `render_table_top_and_header()` first
2. Migrate `render_data_row()` and `render_iteration()` 
3. Migrate `render_stats_continuation()` (Advanced only)
4. Migrate `render_table_bottom()` and `render_training_summary()`
5. Remove now-unused imports from `components::table`
6. Update tests

### Key Patterns

#### Before (Manual):
```rust
fn render_table_top_and_header(&self) -> String {
    let col_widths = [6, 17, 16, 15, 7, 17];  // Fragile!
    let headers = ["Iter", "Lower Bound ($)", ...];
    
    // 50+ lines of manual border construction
    let top = format!("{}{}{}",
        border.top_left,
        col_widths.iter()
            .map(|&w| border.horizontal.to_string().repeat(w))
            ...
    );
    // ... more manual construction
}
```

#### After (tabled):
```rust
fn render_iteration_table(&self, ctx: &DisplayContext) -> String {
    let mut builder = Builder::default();
    
    if ctx.iteration == 1 {
        builder.push_record([
            "Iter", "Lower Bound ($)", "Simul Cost ($)", 
            "1st Stage ($)", "Gap %", "Time (fwd/bwd)"
        ]);
    }
    
    builder.push_record([
        format!("{}", ctx.iteration),
        format_cost(ctx.lower_bound, true),
        // ... other cells
    ]);
    
    builder.build()
        .with(Style::modern())
        .to_string()
}
```

### Handling Statistics Continuation Row

The Advanced renderer has a statistics row that spans multiple columns. Use `Panel::horizontal()`:

```rust
// Add statistics as a full-width panel after data row
let stats_text = format!("{}    {}", change_pct, format_cost_stats(...));
table.with(Panel::horizontal(row_index + 1, stats_text));
```

### Handling Progressive Display

Current behavior: 
- Iteration 1: Print table header + first row
- Iteration 2-N: Print row only

New approach:
- Store accumulated rows in renderer state, OR
- Print header once (iteration 1), then individual rows (simpler)

Recommendation: Keep current approach - print header only on iteration 1.

## Estimated Effort

**Sprints**: 1 (5-6 days)
**Story Points**: 13

## Sprint Breakdown

### Sprint 1: Full Migration

| Ticket | Title | Points |
|--------|-------|--------|
| TABLED-004 | Migrate AdvancedRenderer table header | 2 |
| TABLED-005 | Migrate AdvancedRenderer iteration rows | 3 |
| TABLED-006 | Migrate AdvancedRenderer training summary | 2 |
| TABLED-007 | Migrate StandardRenderer tables | 3 |
| TABLED-008 | Update renderer tests for new output format | 3 |

---

**Epic Status**: Ready for implementation (blocked by Epic 1)
