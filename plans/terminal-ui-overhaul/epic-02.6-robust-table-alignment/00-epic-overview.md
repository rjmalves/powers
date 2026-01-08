# Epic 02.6: Robust Table Alignment

## Summary

Replace the fragile manual table width management with a robust system that uses generous width budgets, explicit content truncation, and ANSI-aware width calculation. This eliminates alignment issues caused by Rust's `format!` macro expanding fields when content exceeds the specified width.

Additionally, remove the `tabled` dependency that was added during an attempted migration (which was abandoned due to architectural incompatibility with progressive rendering).

## Problem Statement

### Root Cause Analysis

The fundamental issue with table alignment is **Rust's `format!` macro behavior**:

```rust
// Expected: truncate to fit
let formatted = format!(" {:^14} ", "15-char-string");
// Actual: EXPANDS to 17 chars!

// This causes the row to be wider than expected,
// breaking alignment with other rows
```

### Current Problems

1. **Format Expansion**: When content exceeds format width, the field expands instead of truncating
2. **Distributed Width Definitions**: `col_widths` arrays defined in 4+ locations per renderer
3. **Format/Width Mismatch**: Easy to update `col_widths` but forget corresponding `format!` widths
4. **No ANSI Handling**: ANSI color codes add to string length but not display width
5. **Fragile Merged Rows**: Statistics continuation row requires complex width calculation
6. **Unused Dependency**: `tabled` crate was added but is incompatible with progressive rendering

### Failed Migration Attempt

A migration to the `tabled` crate was attempted but abandoned because:
- `tabled` builds complete tables, not individual rows
- Current architecture requires progressive rendering (header once, then rows incrementally)
- Changing to full-table-rebuild would alter output behavior significantly

## Solution: Robust Manual Table with Safety Guarantees

### Key Design Principles

1. **Generous Width Budget**: Add 25-35% slack to each column
2. **Explicit Truncation**: Never allow format expansion - truncate content first
3. **Centralized Configuration**: Single source of truth for all column widths
4. **ANSI-Aware Measurement**: Strip ANSI codes before calculating display width
5. **Compile-Time Validation**: Tests verify widths are sufficient for all content types

### Architecture

```rust
// Single source of truth for column widths
pub struct TableColumnConfig {
    pub widths: &'static [usize],
    pub headers: &'static [&'static str],
    pub alignments: &'static [Alignment],
}

// Safe cell formatting that NEVER expands
pub fn format_cell(content: &str, width: usize, align: Alignment) -> String {
    let visible_len = strip_ansi(content).chars().count();
    let format_width = width.saturating_sub(2);
    
    // CRITICAL: Truncate if necessary
    let safe_content = if visible_len > format_width {
        truncate_preserving_ansi(content, format_width)
    } else {
        content.to_string()
    };
    
    // Now safe - content guaranteed to fit
    match align {
        Left => format!(" {:<w$} ", safe_content, w = format_width),
        Center => format!(" {:^w$} ", safe_content, w = format_width),
        Right => format!(" {:>w$} ", safe_content, w = format_width),
    }
}
```

## Scope

### Included

- Remove `tabled` dependency from `Cargo.toml`
- Remove `src/display/components/tabled_utils.rs`
- Revert any tabled-related changes to renderers
- Create `src/display/components/table_format.rs` with:
  - `TableColumnConfig` for centralized width management
  - `format_cell()` with truncation safety
  - `strip_ansi_codes()` for width calculation
  - `truncate_preserving_ansi()` for safe truncation
  - Row and border construction utilities
- Update `AdvancedRenderer` to use new utilities
- Update `StandardRenderer` to use new utilities
- Add comprehensive tests for alignment
- Verify all profiles produce correct output

### Excluded

- Changes to Minimal/Automation profiles (no tables)
- New display features
- Performance optimizations beyond what's needed for correctness

## Dependencies

- **Requires**: Epic 02.5 complete (UI fixes applied)
- **Enables**: Epic 03 (clean foundation for simulation display)

## Acceptance Criteria

- [x] `tabled` dependency completely removed from project
- [x] No references to `tabled` or `tabled_utils` in codebase
- [x] All table rows have consistent width (validated by tests)
- [x] ANSI-colored content aligns correctly
- [x] Large numeric values (e.g., `$123,456,789.12`) don't break alignment
- [x] Long timing strings (e.g., `12.34s / 12.34s`) don't break alignment
- [x] Column widths defined in exactly ONE location per renderer
- [x] All existing tests pass
- [x] Manual terminal testing shows perfect alignment
- [x] `cargo clippy` clean
- [x] `cargo fmt` applied

## Technical Approach

### Column Width Strategy

| Column | Header | Max Content | Current Width | New Width | Slack |
|--------|--------|-------------|---------------|-----------|-------|
| Iter | "Iter" (4) | "9999" (4) | 6 | 8 | +33% |
| Lower Bound | "Lower Bound ($)" (15) | "$123,456,789.12" (16) | 17 | 22 | +29% |
| Simul Cost | "Simul Cost ($)" (14) | "$123,456,789.12" (16) | 16 | 20 | +25% |
| 1st Stage | "1st Stage ($)" (13) | "$123,456,789.12" (16) | 15 | 20 | +25% |
| Gap % | "Gap %" (5) | "100.00% ↓" (9) | 7 | 12 | +33% |
| Time | "Time (fwd/bwd)" (14) | "12.34s / 12.34s" (15) | 17 | 20 | +18% |

### ANSI Code Handling

```rust
/// Strip ANSI escape codes for width calculation.
pub fn strip_ansi_codes(s: &str) -> String {
    // Match: ESC [ ... m (SGR sequences for colors/styles)
    let re = regex::Regex::new(r"\x1b\[[0-9;]*m").unwrap();
    re.replace_all(s, "").to_string()
}

/// Calculate display width (visible characters only).
pub fn display_width(s: &str) -> usize {
    strip_ansi_codes(s).chars().count()
}
```

### Safe Truncation

```rust
/// Truncate string to max visible width, preserving ANSI codes.
pub fn truncate_preserving_ansi(s: &str, max_width: usize) -> String {
    let mut result = String::new();
    let mut visible_count = 0;
    let mut in_escape = false;
    
    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            result.push(c);
        } else if in_escape {
            result.push(c);
            if c == 'm' {
                in_escape = false;
            }
        } else if visible_count < max_width {
            result.push(c);
            visible_count += 1;
        }
    }
    
    // Close any open ANSI sequences
    if result.contains("\x1b[") && !result.ends_with("\x1b[0m") {
        result.push_str("\x1b[0m");
    }
    
    result
}
```

## Estimated Effort

- **Sprint**: 1 sprint
- **Story Points**: 13 total
- **Duration**: 2-3 days

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Truncation affects readability | Low | Medium | Generous widths prevent truncation in normal use |
| ANSI handling edge cases | Medium | Low | Comprehensive tests with various color combinations |
| Regex performance overhead | Low | Very Low | Table rendering is not a hot path |
| Test string comparisons break | High | Low | Update tests to verify content presence, not exact format |

## Sprint Breakdown

### Sprint 1: Full Implementation

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| T-031 | Remove tabled dependency and revert related code | 2 | None |
| T-032 | Create robust table formatting utilities | 5 | T-031 |
| T-033 | Migrate AdvancedRenderer to new utilities | 3 | T-032 |
| T-034 | Migrate StandardRenderer to new utilities | 2 | T-032 |
| T-035 | Add comprehensive alignment tests | 2 | T-033, T-034 |

## Definition of Done

- [x] All tickets complete
- [x] `tabled` dependency removed
- [x] All tests passing (including new alignment tests)
- [x] Manual testing with all 4 profiles
- [x] Perfect table alignment verified in terminal
- [x] No clippy warnings
- [x] Code formatted with rustfmt
- [x] Documentation updated

---

**Epic Status**: ✅ COMPLETE
