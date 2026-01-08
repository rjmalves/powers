# [T-030] Fix Minor Display Issues

> **Epic**: [Epic 02.5: UI Fixes](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [T-026](./ticket-026-silence-legacy-logging-lib.md), [T-027](./ticket-027-silence-legacy-logging-simulate.md)
> **Blocks**: None

## Context

### Background

Several minor display issues remain after the major fixes. These are polish items that improve the overall output quality.

### Issues to Fix

1. **Minimal profile progress bar stacking**: Progress bar outputs stack instead of updating in-place
2. **Negative std dev in training summary**: Shows `± -7.70e6` (impossible for std dev)
3. **Missing newlines**: Training summary runs directly into next section without spacing

## Specification

### Issue 1: Progress Bar Stacking (MinimalRenderer)

**Current behavior** (in `src/display/renderers/minimal.rs`):
```rust
format!("\rTraining: {}", progress)
```

The carriage return `\r` should work, but the output shows:
```
Training: [█████████░░░░░░░░░░░░░░░░░░░] |  33% | 1/3 iter | ETA: 0.000sTraining: [███████████████████░░░░░░░░░] |  67% | 2/3 iter | ETA: 0.000sTraining: [████████████████████████████] | 100% | 3/3 iter | ETA: 0.000s
```

**Root cause**: When output is piped or buffered, `\r` may not work as expected. Also, the terminal may not be flushing properly.

**Fix**: 
- Ensure stdout is flushed after each progress update
- Add newline only on final iteration (or use clearing escape sequence)

### Issue 2: Negative Standard Deviation (AdvancedRenderer)

**Current behavior** (in `src/display/renderers/advanced.rs` line 382-386):
```rust
lines.push(format!(
    "  Policy cost:    {} ± {}",
    format_cost(result.statistical_upper_bound, true),
    format_cost(
        result.best_upper_bound - result.statistical_upper_bound,
        true
    )
));
```

**Problem**: `best_upper_bound - statistical_upper_bound` can be negative when `best_upper_bound < statistical_upper_bound`.

**Fix**: Use `(result.best_upper_bound - result.statistical_upper_bound).abs()` or compute actual std dev from the training data.

### Issue 3: Missing Newlines

**Current behavior**: Training summary ends without newline, next output appears on same line:
```
  Iterations:     3[INFO]
```

**Fix**: Add trailing newline to training summary output.

## Acceptance Criteria

- [x] Minimal profile progress bar updates in-place (or at least newline on each update for piped output)
- [x] Training summary shows positive std dev value
- [x] Proper spacing between training summary and subsequent output
- [x] All profiles produce clean, properly spaced output

## Implementation Guide

### Fix 1: Progress Bar (MinimalRenderer)

In `src/display/renderers/minimal.rs`:

```rust
fn render_iteration(&self, ctx: &DisplayContext, _config: &DisplayConfig) -> String {
    if !ctx.should_print {
        return String::new();
    }

    // ... progress bar creation ...

    // For last iteration, add newline; otherwise just \r
    if ctx.iteration == ctx.total_iterations {
        format!("\rTraining: {}\n", progress)
    } else {
        format!("\rTraining: {}", progress)
    }
}
```

Also ensure `train_with_display()` flushes after each iteration (it does, via `std::io::stdout().flush().ok()`).

### Fix 2: Std Dev Calculation (AdvancedRenderer)

In `src/display/renderers/advanced.rs`:

```rust
// Option A: Use absolute value
lines.push(format!(
    "  Policy cost:    {} ± {}",
    format_cost(result.statistical_upper_bound, true),
    format_cost(
        (result.best_upper_bound - result.statistical_upper_bound).abs(),
        true
    )
));

// Option B: Better - compute from actual data if available
// This requires access to the iteration costs which may not be in TrainingResult
```

For now, use Option A (absolute value).

### Fix 3: Trailing Newlines (All Renderers)

In each renderer's `render_training_summary()`, ensure the return ends with `\n`:

```rust
fn render_training_summary(...) -> String {
    // ... build lines ...
    lines.join("\n") + "\n"  // Add trailing newline
}
```

### Key Files to Modify

- `src/display/renderers/minimal.rs`: `render_iteration()` (add newline on last iteration)
- `src/display/renderers/advanced.rs`: `render_training_summary()` (fix std dev, add newline)
- `src/display/renderers/standard.rs`: `render_training_summary()` (check trailing newline)

### Pitfalls to Avoid

- ⚠️ Don't break automation profile (JSON output should remain unchanged)
- ⚠️ Test with both interactive terminal and piped output (`| head`)
- ⚠️ Don't add double newlines

## Testing Requirements

### Manual Tests

- [x] Minimal profile: Progress bar updates cleanly (or shows one per line if piped)
- [x] Advanced profile: Training summary shows positive std dev (no `± -X`)
- [x] All profiles: Proper spacing between sections

### Edge Cases

- [x] Very fast training (< 1 second) - progress bar still works
- [x] Piped output (`| cat`) - no garbled output
- [x] `statistical_upper_bound > best_upper_bound` scenario

### Automated Tests

- [x] All existing tests pass
- [x] Consider adding test for positive std dev in summary

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Three small, isolated fixes

## Definition of Done

- [x] Progress bar behavior is reasonable (update in-place or newline each)
- [x] Std dev always shows positive value
- [x] Proper newlines and spacing throughout
- [x] All tests passing
- [x] Manual verification with all profiles

## Status: ✅ COMPLETE

**Implementation Summary:**

### Fix 1: Negative std dev (AdvancedRenderer)
- Changed calculation from `result.best_upper_bound - result.statistical_upper_bound` to `.abs()`
- Now always shows positive value: `1.10e8 ± 7.70e6`

### Fix 2: Missing newlines (All Renderers)
- Added trailing `\n` to `render_training_summary()` in:
  - `src/display/renderers/advanced.rs`: Changed `lines.join("\n")` to `lines.join("\n") + "\n"`
  - `src/display/renderers/standard.rs`: Same change
  - `src/display/renderers/minimal.rs`: Added `\n` to format string

### Fix 3: Progress bar stacking (MinimalRenderer)
- Added newline on final iteration in `render_iteration()`
- Progress bar now properly terminates with `\n` on last iteration
- Intermediate iterations still use `\r` for in-place updates

**Testing:**
- All 749 tests passing
- All 4 profiles verified manually
- Clean output with proper spacing throughout
