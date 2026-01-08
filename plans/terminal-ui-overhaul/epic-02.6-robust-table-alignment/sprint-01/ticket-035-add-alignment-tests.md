# [T-035] Add comprehensive alignment tests

> **Epic**: [Epic 2.6: Robust Table Alignment](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-033](./ticket-033-migrate-advanced-renderer.md), [T-034](./ticket-034-migrate-standard-renderer.md)  
> **Blocks**: None (final ticket)

## Context

### Background

Table alignment issues have recurred multiple times. This ticket adds comprehensive tests that validate alignment invariants, ensuring future changes don't reintroduce misalignment.

### Goals

1. Validate that all table rows have consistent width
2. Test edge cases: large numbers, long timing strings, colored content
3. Prevent regression in alignment

### Files to Read Before Starting

- `src/display/components/table_format.rs` - Utilities being tested
- `src/display/renderers/advanced.rs` - Advanced renderer
- `src/display/renderers/standard.rs` - Standard renderer
- `tests/` - Existing test patterns

## Specification

### New Test File: `tests/test_table_alignment.rs`

Create a dedicated integration test file for alignment validation:

```rust
//! Table Alignment Tests
//!
//! These tests ensure that table rendering produces consistent row widths
//! across all content types and edge cases.

use powers_rs::display::components::table_format::{
    display_width, format_cell, Alignment, TableColumnConfig,
    build_top_border, build_separator, build_bottom_border,
    build_header_row, build_row,
};
use powers_rs::display::components::table::BorderStyle;

// ============================================================================
// INVARIANT: All rows in a table must have the same display width
// ============================================================================

#[test]
fn test_advanced_table_row_widths_consistent() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();
    let expected = config.total_width();
    
    // Test borders
    let top = build_top_border(&config.widths, &border);
    assert_eq!(
        display_width(&top), expected,
        "Top border width mismatch: expected {}, got {}",
        expected, display_width(&top)
    );
    
    let header = build_header_row(&config, &border);
    assert_eq!(
        display_width(&header), expected,
        "Header row width mismatch: expected {}, got {}",
        expected, display_width(&header)
    );
    
    let sep = build_separator(&config.widths, &border);
    assert_eq!(
        display_width(&sep), expected,
        "Separator width mismatch: expected {}, got {}",
        expected, display_width(&sep)
    );
    
    let bottom = build_bottom_border(&config.widths, &border);
    assert_eq!(
        display_width(&bottom), expected,
        "Bottom border width mismatch: expected {}, got {}",
        expected, display_width(&bottom)
    );
}

#[test]
fn test_standard_table_row_widths_consistent() {
    let config = TableColumnConfig::standard();
    let border = BorderStyle::Standard.chars().unwrap();
    let expected = config.total_width();
    
    let top = build_top_border(&config.widths, &border);
    let header = build_header_row(&config, &border);
    let sep = build_separator(&config.widths, &border);
    let bottom = build_bottom_border(&config.widths, &border);
    
    assert_eq!(display_width(&top), expected);
    assert_eq!(display_width(&header), expected);
    assert_eq!(display_width(&sep), expected);
    assert_eq!(display_width(&bottom), expected);
}

// ============================================================================
// EDGE CASE: Large numeric values
// ============================================================================

#[test]
fn test_large_cost_values_dont_overflow() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();
    
    // Simulate a row with large values
    let large_cost = "$123,456,789,012.34"; // 19 chars - larger than expected
    let cells: Vec<String> = config.widths.iter()
        .map(|&w| format_cell(large_cost, w, Alignment::Center))
        .collect();
    
    let row = build_row(&cells, &border);
    
    assert_eq!(
        display_width(&row), config.total_width(),
        "Large cost values caused row overflow"
    );
}

#[test]
fn test_maximum_iteration_number() {
    let config = TableColumnConfig::advanced();
    
    // Test iteration numbers up to 99999
    for iter in [1, 10, 100, 1000, 9999, 99999] {
        let cell = format_cell(&format!("{}", iter), config.widths[0], Alignment::Right);
        assert_eq!(
            display_width(&cell), config.widths[0],
            "Iteration {} caused cell overflow", iter
        );
    }
}

// ============================================================================
// EDGE CASE: Long timing strings
// ============================================================================

#[test]
fn test_maximum_timing_strings() {
    let config = TableColumnConfig::advanced();
    let timing_width = config.widths[5]; // Last column is timing
    
    // Test various timing formats
    let timing_strings = [
        "0.001s / 0.001s",   // Short
        "1.234s / 5.678s",   // Medium
        "12.34s / 12.34s",   // Maximum compact
        "1m 23s / 2m 34s",   // Minutes
        "1h 23m / 2h 34m",   // Hours
    ];
    
    for timing in &timing_strings {
        let cell = format_cell(timing, timing_width, Alignment::Center);
        assert_eq!(
            display_width(&cell), timing_width,
            "Timing '{}' caused cell overflow", timing
        );
    }
}

// ============================================================================
// EDGE CASE: ANSI colored content
// ============================================================================

#[test]
fn test_colored_content_alignment() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();
    
    // Content with ANSI color codes
    let green = "\x1b[32m$1,234.56\x1b[0m";
    let red = "\x1b[31m$9,876.54\x1b[0m";
    let bold_blue = "\x1b[1;34m5.23%\x1b[0m";
    
    let cells = vec![
        format_cell("1", config.widths[0], Alignment::Right),
        format_cell(green, config.widths[1], Alignment::Center),
        format_cell(red, config.widths[2], Alignment::Center),
        format_cell("$500.00", config.widths[3], Alignment::Center),
        format_cell(bold_blue, config.widths[4], Alignment::Center),
        format_cell("0.1s / 0.2s", config.widths[5], Alignment::Center),
    ];
    
    let row = build_row(&cells, &border);
    
    assert_eq!(
        display_width(&row), config.total_width(),
        "ANSI colored content caused row width mismatch"
    );
}

#[test]
fn test_mixed_plain_and_colored_rows() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();
    
    // Plain row
    let plain_cells: Vec<String> = config.widths.iter()
        .map(|&w| format_cell("test", w, Alignment::Center))
        .collect();
    let plain_row = build_row(&plain_cells, &border);
    
    // Colored row
    let colored_cells: Vec<String> = config.widths.iter()
        .map(|&w| format_cell("\x1b[32mtest\x1b[0m", w, Alignment::Center))
        .collect();
    let colored_row = build_row(&colored_cells, &border);
    
    assert_eq!(
        display_width(&plain_row),
        display_width(&colored_row),
        "Plain and colored rows have different widths"
    );
}

// ============================================================================
// EDGE CASE: Empty and minimal content
// ============================================================================

#[test]
fn test_empty_cells() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();
    
    let cells: Vec<String> = config.widths.iter()
        .map(|&w| format_cell("", w, Alignment::Center))
        .collect();
    
    let row = build_row(&cells, &border);
    
    assert_eq!(
        display_width(&row), config.total_width(),
        "Empty cells caused row width mismatch"
    );
}

#[test]
fn test_single_char_content() {
    let config = TableColumnConfig::advanced();
    
    for &width in &config.widths {
        let cell = format_cell("X", width, Alignment::Center);
        assert_eq!(
            display_width(&cell), width,
            "Single char cell has wrong width"
        );
    }
}

// ============================================================================
// HEADER VALIDATION
// ============================================================================

#[test]
fn test_headers_fit_in_columns() {
    let config = TableColumnConfig::advanced();
    
    for (header, &width) in config.headers.iter().zip(&config.widths) {
        let header_len = header.len();
        let available = width.saturating_sub(2); // Space for padding
        assert!(
            header_len <= available,
            "Header '{}' ({} chars) doesn't fit in column width {} (available: {})",
            header, header_len, width, available
        );
    }
}

#[test]
fn test_standard_headers_fit_in_columns() {
    let config = TableColumnConfig::standard();
    
    for (header, &width) in config.headers.iter().zip(&config.widths) {
        let header_len = header.len();
        let available = width.saturating_sub(2);
        assert!(
            header_len <= available,
            "Header '{}' ({} chars) doesn't fit in column width {}",
            header, header_len, width
        );
    }
}

// ============================================================================
// ALIGNMENT MODES
// ============================================================================

#[test]
fn test_all_alignments_produce_same_width() {
    let content = "test";
    let width = 15;
    
    let left = format_cell(content, width, Alignment::Left);
    let center = format_cell(content, width, Alignment::Center);
    let right = format_cell(content, width, Alignment::Right);
    
    assert_eq!(display_width(&left), width);
    assert_eq!(display_width(&center), width);
    assert_eq!(display_width(&right), width);
}

// ============================================================================
// TRUNCATION BEHAVIOR
// ============================================================================

#[test]
fn test_content_truncation() {
    let long_content = "this is a very long string that exceeds the column width";
    let width = 15;
    
    let cell = format_cell(long_content, width, Alignment::Left);
    
    assert_eq!(
        display_width(&cell), width,
        "Truncated cell should have exact width"
    );
}

#[test]
fn test_truncation_preserves_ansi() {
    let colored_long = "\x1b[32mthis is a very long green string\x1b[0m";
    let width = 15;
    
    let cell = format_cell(colored_long, width, Alignment::Left);
    
    assert_eq!(display_width(&cell), width);
    // Should still contain color codes
    assert!(cell.contains("\x1b[32m") || cell.contains("\x1b[0m"));
}
```

### Unit Tests in table_format.rs

Ensure the existing unit tests from T-032 are comprehensive. Add any missing edge cases.

## Acceptance Criteria

- [ ] New test file `tests/test_table_alignment.rs` created
- [ ] All alignment invariant tests pass
- [ ] Edge cases covered: large numbers, long strings, colors
- [ ] Both Advanced and Standard configurations tested
- [ ] `cargo test -j 1 -- --test-threads=1` passes
- [ ] Tests prevent future regression

## Testing Requirements

### Run All Tests

```bash
cargo test -j 1 -- --test-threads=1
cargo test -j 1 -- --test-threads=1 test_table_alignment
```

### Verify Coverage

- [ ] Top border alignment
- [ ] Header row alignment
- [ ] Separator alignment
- [ ] Data row alignment
- [ ] Bottom border alignment
- [ ] Large values
- [ ] ANSI colors
- [ ] Empty content
- [ ] Truncation

## Implementation Guide

### Key Files to Create

- `tests/test_table_alignment.rs`

### Patterns to Follow

- See existing integration tests in `tests/` directory
- Use descriptive assertion messages
- Group related tests with comments

### Pitfalls to Avoid

- ⚠️ Don't rely on string equality - use `display_width()` for comparison
- ⚠️ Remember that ANSI codes add bytes but not display width
- ⚠️ Use `-j 1` and `--test-threads=1` for running tests

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Test writing with clear specification

## Definition of Done

- [x] All test cases implemented
- [x] All tests passing
- [x] Edge cases verified
- [x] No clippy warnings in test code
- [x] Tests documented

---

**Status**: ✅ COMPLETE
