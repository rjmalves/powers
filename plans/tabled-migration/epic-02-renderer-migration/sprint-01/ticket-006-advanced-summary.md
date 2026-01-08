# [TABLED-006] Migrate AdvancedRenderer training summary

> **Epic**: [Epic 2: Renderer Migration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TABLED-005](./ticket-005-advanced-iteration.md)  
> **Blocks**: [TABLED-008](./ticket-008-update-tests.md)

## Context

### Background

The `render_training_summary()` method produces the final summary after training completes. It includes:
1. Table bottom border (to close the iteration table)
2. Convergence status
3. Final metrics (time, bound, cost, gap, cuts, iterations)
4. Target gap achievement status

Most of this is plain text formatting, but the table bottom border uses manual construction.

### Relation to Epic

This completes the Advanced renderer migration.

### Current State

Lines 345-439 in `advanced.rs`:
- `render_table_bottom()` uses manual border construction
- Rest is string formatting (not table-related)

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Lines 269-284 (`render_table_bottom`), 345-439 (`render_training_summary`)
- `src/sddp/mod.rs` - `TrainingResult` struct for available fields

## Specification

### Inputs

- `&TrainingResult` - Training outcome with statistics
- `&DisplayConfig` - Display configuration

### Outputs

- `String` containing table bottom + summary text

### Behavior

The table bottom border should be generated to close the iteration table. Since we're now using `tabled`, we have two options:

**Option A**: Generate a standalone bottom border
```rust
// Create a dummy table just to get the bottom border
let bottom = Builder::default()
    .push_record([""; 6])  // Match column count
    .build()
    .with(Style::modern())
    // Extract just the bottom line somehow
```

**Option B**: Remove explicit table bottom, rely on `tabled`'s automatic closure

**Recommendation**: Option B - the iteration table already includes proper borders. The summary can start after a blank line.

The summary text (metrics, status, etc.) remains as plain text formatting - no tables needed.

## Acceptance Criteria

- [ ] Training summary renders correctly after iteration table
- [ ] No manual border character construction
- [ ] All metrics displayed: time, bound, cost, gap, cuts, iterations
- [ ] Convergence status icon (✓ or ⋯) appears
- [ ] Target gap achievement appears when configured
- [ ] Colors applied correctly
- [ ] All summary tests pass

## Implementation Guide

### Suggested Approach

1. **Remove `render_table_bottom()` method** - It's no longer needed if `render_iteration()` produces complete bordered tables.

2. **Simplify `render_training_summary()`**:
   ```rust
   fn render_training_summary(
       &self,
       result: &TrainingResult,
       config: &DisplayConfig,
   ) -> String {
       let color_config = ColorConfig::new(config.color_enabled);
       
       let mut lines = Vec::new();
       lines.push(String::new()); // Blank line after table
       
       // Convergence status
       let converged = result.relative_gap() < 0.05;
       let status_icon = if converged {
           colorize("✓", SemanticColor::Good, &color_config)
       } else {
           colorize("⋯", SemanticColor::Caution, &color_config)
       };
       
       let title = if converged { "Training Complete" } else { "Training Stopped" };
       lines.push(format!("{} {}", bold(title, &color_config), status_icon));
       lines.push("─".repeat(title.len() + 2));
       
       // Metrics
       lines.push(format!("  Total time:     {}", format_duration_hms(result.total_time)));
       lines.push(format!("  Final bound:    {}", format_cost(result.final_lower_bound, true)));
       // ... rest of metrics
       
       lines.join("\n") + "\n"
   }
   ```

3. **Handle table closure**: If the iteration table doesn't include a bottom border, add one at the start of summary. But since each iteration produces a complete table segment, this may not be needed.

### Alternative: Keep table bottom

If the iteration table needs an explicit bottom border:

```rust
fn render_table_bottom(&self) -> String {
    // Use tabled to generate a consistent bottom border
    let mut builder = Builder::default();
    // Add empty row to get border
    builder.push_record([""; 6]);
    let table = builder.build().with(Style::modern());
    
    // Extract just the bottom line
    let full = table.to_string();
    full.lines().last().unwrap_or("").to_string()
}
```

### Key Files to Modify

- `src/display/renderers/advanced.rs`:
  - Remove or simplify `render_table_bottom()`
  - Update `render_training_summary()`

### Pitfalls to Avoid

- ⚠️ Don't break the visual flow between iteration table and summary
- ⚠️ Ensure blank line separation
- ⚠️ Keep all existing metrics in summary

## Testing Requirements

### Unit Tests

- [ ] `test_render_training_summary_converged()` passes
- [ ] `test_render_training_summary_not_converged()` passes
- [ ] `test_training_summary_target_achieved()` passes
- [ ] `test_training_summary_target_missed()` passes
- [ ] `test_training_summary_no_target()` passes

### Manual Testing

- [ ] Run full training and verify summary appears correctly
- [ ] Verify visual separation from iteration table

## Documentation Requirements

- [ ] Update method doc comments

## Dependencies

- **Blocked By**: TABLED-005
- **Blocks**: TABLED-008
- **Related**: None

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Mostly text formatting, minimal table work

## Definition of Done

- [ ] Implementation complete
- [ ] All summary tests passing
- [ ] Manual terminal test shows correct output
- [ ] Code reviewed
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
