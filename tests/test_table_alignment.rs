//! Table Alignment Integration Tests
//!
//! These tests ensure that table rendering produces consistent row widths
//! across all content types and edge cases.
//!
//! The tests validate the alignment invariants that prevent regression
//! in table formatting, specifically ensuring that:
//! - All rows have the same display width
//! - ANSI colored content doesn't break alignment
//! - Large values are truncated rather than expanding

use powers_rs::display::components::table::{Alignment, BorderStyle};
use powers_rs::display::components::table_format::{
    build_bottom_border, build_header_row, build_row, build_separator,
    build_top_border, display_width, format_cell, TableColumnConfig,
};

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
        display_width(&top),
        expected,
        "Top border width mismatch: expected {}, got {}",
        expected,
        display_width(&top)
    );

    let header = build_header_row(&config, &border);
    assert_eq!(
        display_width(&header),
        expected,
        "Header row width mismatch: expected {}, got {}",
        expected,
        display_width(&header)
    );

    let sep = build_separator(&config.widths, &border);
    assert_eq!(
        display_width(&sep),
        expected,
        "Separator width mismatch: expected {}, got {}",
        expected,
        display_width(&sep)
    );

    let bottom = build_bottom_border(&config.widths, &border);
    assert_eq!(
        display_width(&bottom),
        expected,
        "Bottom border width mismatch: expected {}, got {}",
        expected,
        display_width(&bottom)
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
    let cells: Vec<String> = config
        .widths
        .iter()
        .map(|&w| format_cell(large_cost, w, Alignment::Center))
        .collect();

    let row = build_row(&cells, &border);

    assert_eq!(
        display_width(&row),
        config.total_width(),
        "Large cost values caused row overflow"
    );
}

#[test]
fn test_maximum_iteration_number() {
    let config = TableColumnConfig::advanced();

    // Test iteration numbers up to 99999
    for iter in [1, 10, 100, 1000, 9999, 99999] {
        let cell = format_cell(
            &format!("{}", iter),
            config.widths[0],
            Alignment::Right,
        );
        assert_eq!(
            display_width(&cell),
            config.widths[0],
            "Iteration {} caused cell overflow",
            iter
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
        "0.001s / 0.001s", // Short
        "1.234s / 5.678s", // Medium
        "12.34s / 12.34s", // Maximum compact
        "1m 23s / 2m 34s", // Minutes
        "1h 23m / 2h 34m", // Hours
    ];

    for timing in &timing_strings {
        let cell = format_cell(timing, timing_width, Alignment::Center);
        assert_eq!(
            display_width(&cell),
            timing_width,
            "Timing '{}' caused cell overflow",
            timing
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
        display_width(&row),
        config.total_width(),
        "ANSI colored content caused row width mismatch"
    );
}

#[test]
fn test_mixed_plain_and_colored_rows() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();

    // Plain row
    let plain_cells: Vec<String> = config
        .widths
        .iter()
        .map(|&w| format_cell("test", w, Alignment::Center))
        .collect();
    let plain_row = build_row(&plain_cells, &border);

    // Colored row
    let colored_cells: Vec<String> = config
        .widths
        .iter()
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

    let cells: Vec<String> = config
        .widths
        .iter()
        .map(|&w| format_cell("", w, Alignment::Center))
        .collect();

    let row = build_row(&cells, &border);

    assert_eq!(
        display_width(&row),
        config.total_width(),
        "Empty cells caused row width mismatch"
    );
}

#[test]
fn test_single_char_content() {
    let config = TableColumnConfig::advanced();

    for &width in &config.widths {
        let cell = format_cell("X", width, Alignment::Center);
        assert_eq!(
            display_width(&cell),
            width,
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
            header,
            header_len,
            width
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
    let long_content =
        "this is a very long string that exceeds the column width";
    let width = 15;

    let cell = format_cell(long_content, width, Alignment::Left);

    assert_eq!(
        display_width(&cell),
        width,
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

// ============================================================================
// DATA ROW CONSISTENCY
// ============================================================================

#[test]
fn test_data_row_matches_header_width_advanced() {
    let config = TableColumnConfig::advanced();
    let border = BorderStyle::Standard.chars().unwrap();

    // Simulate a typical data row
    let cells = vec![
        format_cell("42", config.widths[0], Alignment::Right),
        format_cell("$1,234,567.89", config.widths[1], Alignment::Center),
        format_cell("$2,345,678.90", config.widths[2], Alignment::Center),
        format_cell("$999,999.99", config.widths[3], Alignment::Center),
        format_cell("5.23%", config.widths[4], Alignment::Center),
        format_cell("1.23s / 4.56s", config.widths[5], Alignment::Center),
    ];

    let data_row = build_row(&cells, &border);
    let header = build_header_row(&config, &border);

    assert_eq!(
        display_width(&data_row),
        display_width(&header),
        "Data row width should match header width"
    );
}

#[test]
fn test_data_row_matches_header_width_standard() {
    let config = TableColumnConfig::standard();
    let border = BorderStyle::Standard.chars().unwrap();

    // Simulate a typical data row
    let cells = vec![
        format_cell("42", config.widths[0], Alignment::Right),
        format_cell("$1,234,567.89", config.widths[1], Alignment::Center),
        format_cell("$2,345,678.90", config.widths[2], Alignment::Center),
        format_cell("5.23%", config.widths[3], Alignment::Center),
        format_cell("1.23s / 4.56s", config.widths[4], Alignment::Center),
    ];

    let data_row = build_row(&cells, &border);
    let header = build_header_row(&config, &border);

    assert_eq!(
        display_width(&data_row),
        display_width(&header),
        "Data row width should match header width"
    );
}

// ============================================================================
// EXTREME VALUES
// ============================================================================

#[test]
fn test_extreme_gap_percentages() {
    let config = TableColumnConfig::advanced();
    let gap_width = config.widths[4];

    // Test extreme gap values
    let extreme_gaps = [
        "0.00%", "100.00%", "999.99%", "0.01%", "-50.00%", "+999.99%",
    ];

    for gap in &extreme_gaps {
        let cell = format_cell(gap, gap_width, Alignment::Center);
        assert_eq!(
            display_width(&cell),
            gap_width,
            "Gap '{}' caused cell overflow",
            gap
        );
    }
}

#[test]
fn test_extreme_cost_values() {
    let config = TableColumnConfig::advanced();

    // Test various cost formats
    let costs = [
        "$0.00",
        "$1.00",
        "$1,000.00",
        "$1,000,000.00",
        "$1,000,000,000.00",
        "-$1,000,000.00",
        "$1.23e9",
        "$123,456,789,012.34",
    ];

    for &cost in &costs {
        for (&width, &align) in config.widths.iter().zip(&config.alignments) {
            let cell = format_cell(cost, width, align);
            assert_eq!(
                display_width(&cell),
                width,
                "Cost '{}' caused overflow in column width {}",
                cost,
                width
            );
        }
    }
}

// ============================================================================
// BORDER STYLE VARIATIONS
// ============================================================================

#[test]
fn test_all_border_styles_consistent_width() {
    let config = TableColumnConfig::advanced();
    let expected = config.total_width();

    for style in [
        BorderStyle::Ascii,
        BorderStyle::Standard,
        BorderStyle::Rounded,
        BorderStyle::Heavy,
    ] {
        if let Some(border) = style.chars() {
            let top = build_top_border(&config.widths, &border);
            let header = build_header_row(&config, &border);
            let sep = build_separator(&config.widths, &border);
            let bottom = build_bottom_border(&config.widths, &border);

            assert_eq!(
                display_width(&top),
                expected,
                "{:?} top border width mismatch",
                style
            );
            assert_eq!(
                display_width(&header),
                expected,
                "{:?} header width mismatch",
                style
            );
            assert_eq!(
                display_width(&sep),
                expected,
                "{:?} separator width mismatch",
                style
            );
            assert_eq!(
                display_width(&bottom),
                expected,
                "{:?} bottom border width mismatch",
                style
            );
        }
    }
}
