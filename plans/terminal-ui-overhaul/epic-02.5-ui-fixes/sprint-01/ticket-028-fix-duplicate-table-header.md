# [T-028] Fix Duplicate Table Header Rendering

> **Epic**: [Epic 02.5: UI Fixes](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-029](./ticket-029-fix-statistics-alignment.md)

## Context

### Background

The table header is being rendered twice: once by `train_with_display()` calling `render_table_header()`, and again by `render_iteration()` when `iteration == 1`. This causes duplicate headers to appear in the output.

### Current State

In `src/sddp/instance.rs`:
```rust
// Line 65-67
let table_header = renderer.render_table_header(display_config);
print!("{}", table_header);
std::io::stdout().flush().ok();
```

In `src/display/renderers/advanced.rs`:
```rust
// Line 315-318 in render_iteration()
if ctx.iteration == 1 {
    output.push_str(&renderer.render_table_top_and_header());
    output.push('\n');
}
```

This produces:
```
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤┌─────┬...
```
(Note the row separator immediately followed by another table top border)

## Specification

### Root Cause

The table header is rendered in two places:
1. `train_with_display()` explicitly calls `render_table_header()` before the training loop
2. `render_iteration()` renders the table header when `iteration == 1`

### Fix Options

**Option A**: Remove `render_table_header()` call from `train_with_display()` (Preferred)
- Let `render_iteration()` handle header on first iteration
- Consistent with current renderer logic
- Simplest fix

**Option B**: Remove header logic from `render_iteration()`
- Keep explicit call in `train_with_display()`
- Requires changing all renderer implementations
- More invasive

**Recommendation**: Use **Option A** - remove the explicit call in `train_with_display()`.

### Changes Required

1. In `src/sddp/instance.rs` `train_with_display()`:
   - Remove lines 65-67 (the `render_table_header()` call)

2. Verify all renderers handle first iteration correctly:
   - `AdvancedRenderer`: Already handles in `render_iteration()`
   - `StandardRenderer`: Already handles in `render_iteration()`  
   - `MinimalRenderer`: No table header (correct)
   - `AutomationRenderer`: No table header (correct)

## Acceptance Criteria

- [x] Table header appears exactly ONCE in training output
- [x] First iteration row includes table header (for table-based profiles)
- [x] Subsequent iterations don't repeat header
- [x] All 4 profiles work correctly
- [x] All existing tests pass

## Implementation Guide

### Suggested Approach

1. Open `src/sddp/instance.rs`
2. Remove lines 65-67:
   ```rust
   // DELETE these lines:
   let table_header = renderer.render_table_header(display_config);
   print!("{}", table_header);
   std::io::stdout().flush().ok();
   ```
3. Test with `cargo run --release -- examples/05-large-scale-brazilian`
4. Verify header appears exactly once
5. Test all 4 profiles

### Key Files to Modify

- `src/sddp/instance.rs`: Remove lines 65-67

### Pitfalls to Avoid

- ⚠️ Don't modify renderer implementations - they're correct
- ⚠️ Ensure `render_table_header()` method still exists (used by tests)

## Testing Requirements

### Manual Tests

- [x] Advanced profile: Table header appears once, table looks correct
- [x] Standard profile: Table header appears once
- [x] Minimal profile: No table header (progress bar only)
- [x] Automation profile: No table header (JSON only)

### Visual Verification

Expected output (Advanced profile, first iteration):
```
╭───────────────────────────────────────────────────────────────────────────────────╮
│ POWE.RS - Power Optimization for the World of Energy                              │
│ Training: 3 iterations × 4 forward passes | Cut selection: enabled               │
╰───────────────────────────────────────────────────────────────────────────────────╯
┌─────┬────────────────┬────────────────┬────────────────┬───────┬─────────────────┐
│ Iter │ Lower Bound ($) │ Simul Cost ($) │ 1st Stage ($)  │ Gap % │ Time (fwd/bwd)  │
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤
│   1 │     1.02e8     │     1.02e8     │     1.02e8     │  0.3% │ 0.156s / 0.711s │
```

NOT:
```
├─────┼────────────────┼────────────────┼────────────────┼───────┼─────────────────┤┌─────┬...
```

### Automated Tests

- [x] All existing tests pass (especially renderer tests)

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple 3-line removal, clear fix

## Definition of Done

- [x] Lines 65-67 removed from `train_with_display()`
- [x] Table header appears exactly once
- [x] All profiles tested manually
- [x] All tests passing

## Status: ✅ COMPLETE

**Implementation Summary:**
- Removed lines 65-67 from `train_with_display()` in src/sddp/instance.rs
- Table header now rendered only once by `render_iteration()` on first iteration
- All 749 tests passing
- All 4 profiles verified manually
- Output is clean and properly formatted
