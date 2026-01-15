//! Robust table formatting utilities with alignment guarantees.
//!
//! This module provides table formatting utilities that guarantee
//! consistent row widths by:
//!
//! 1. Using ANSI-aware width calculation
//! 2. Truncating content before formatting (preventing expansion)
//! 3. Providing centralized column configuration
//!
//! # Example
//!
//! ```
//! use powers_rs::display::components::table_format::{
//!     format_cell, TableColumnConfig, build_header_row,
//!     build_top_border, display_width,
//! };
//! use powers_rs::display::components::table::{Alignment, BorderStyle};
//!
//! let config = TableColumnConfig::advanced();
//! let border = BorderStyle::Standard.chars().unwrap();
//!
//! // All rows will have the same display width
//! let top = build_top_border(&config.widths, &border);
//! let header = build_header_row(&config, &border);
//!
//! assert_eq!(display_width(&top), display_width(&header));
//! ```

use super::table::{Alignment, BorderChars, BorderStyle};

// ============================================================================
// Column Configuration
// ============================================================================

/// Configuration for a table's column structure.
///
/// Provides a single source of truth for column widths, headers, and alignments.
#[derive(Debug, Clone)]
pub struct TableColumnConfig {
    /// Width of each column (including padding spaces).
    pub widths: Vec<usize>,
    /// Header text for each column.
    pub headers: Vec<&'static str>,
    /// Alignment for each column.
    pub alignments: Vec<Alignment>,
}

impl TableColumnConfig {
    /// Advanced renderer column configuration with detailed timing breakdown.
    ///
    /// Uses 10 columns optimized for ~140 character terminal width:
    /// - Iter: 6 (allows up to 9999)
    /// - Lower Bound: 15 (allows 1.23e8 + arrow with fixed width)
    /// - Simulation Cost: 17 (allows 1.23e8)
    /// - Gap: 8 (allows 99.9 + arrow)
    /// - Total Time: 12 (allows 1800.000s = 30 min)
    /// - Fwd Time: 10 (allows 1800.000s)
    /// - Avg Fwd Solver: 16 (allows 1800.000s avg)
    /// - Bwd Time: 10 (allows 1800.000s)
    /// - Avg Bwd Solver: 16 (allows 1800.000s avg)
    /// - Cut Selection: 15 (allows 999.999s)
    ///
    /// Total width: ~136 characters (fits 140-char terminal)
    #[must_use]
    pub fn advanced() -> Self {
        Self {
            widths: vec![
                6,  // Iter
                15, // Lower Bound ($)
                17, // Simulation Cost ($)
                8,  // Gap (%)
                12, // Total Time (s)
                10, // Fwd Time (s)
                16, // Avg Fwd Solver (s)
                10, // Bwd Time (s)
                16, // Avg Bwd Solver (s)
                15, // Cut Selection (s)
            ],
            headers: vec![
                "Iter",
                "Lower Bound",
                "Simulation Cost",
                "Gap",
                "Total Time",
                "Fwd Time",
                "Avg Fwd Solver",
                "Bwd Time",
                "Avg Bwd Solver",
                "Cut Selection",
            ],
            alignments: vec![
                Alignment::Right, // Iter
                Alignment::Right, // Lower Bound (right-align for consistent number display)
                Alignment::Right, // Simulation Cost
                Alignment::Right, // Gap
                Alignment::Right, // Total Time
                Alignment::Right, // Fwd Time
                Alignment::Right, // Avg Fwd Solver
                Alignment::Right, // Bwd Time
                Alignment::Right, // Avg Bwd Solver
                Alignment::Right, // Cut Selection
            ],
        }
    }

    /// Advanced profile header row 2 (units row).
    ///
    /// Returns the unit labels for the second header row.
    #[must_use]
    pub fn advanced_units() -> Vec<&'static str> {
        vec![
            "",    // Iter (no unit)
            "($)", // Lower Bound
            "($)", // Simulation Cost
            "(%)", // Gap
            "(s)", // Total Time
            "(s)", // Fwd Time
            "(s)", // Avg Fwd Solver
            "(s)", // Bwd Time
            "(s)", // Avg Bwd Solver
            "(s)", // Cut Selection
        ]
    }

    /// Standard renderer column configuration.
    ///
    /// Compact layout with total iteration time (seconds with 3 decimal places).
    /// Column widths:
    /// - Iter: 8 (allows up to 99999)
    /// - Lower Bound: 18 (allows 1.23e8 with padding)
    /// - Simulation Cost: 18 (allows 1.23e8)
    /// - Gap: 10 (allows 99.9)
    /// - Total Time: 12 (allows 1800.000 = 30 min)
    ///
    /// Total width: ~72 characters (fits standard 80-char terminal)
    #[must_use]
    pub fn standard() -> Self {
        Self {
            widths: vec![8, 18, 18, 10, 12],
            headers: vec![
                "Iter",
                "Lower Bound",
                "Simulation Cost",
                "Gap",
                "Total Time",
            ],
            alignments: vec![
                Alignment::Right, // Iter
                Alignment::Right, // Lower Bound
                Alignment::Right, // Simulation Cost
                Alignment::Right, // Gap
                Alignment::Right, // Total Time
            ],
        }
    }

    /// Standard profile header row 2 (units row).
    ///
    /// Returns the unit labels for the second header row.
    #[must_use]
    pub fn standard_units() -> Vec<&'static str> {
        vec![
            "",    // Iter (no unit)
            "($)", // Lower Bound
            "($)", // Simulation Cost
            "(%)", // Gap
            "(s)", // Total Time
        ]
    }

    /// Total width of the table including borders.
    ///
    /// Calculated as: sum of column widths + number of separators.
    #[must_use]
    pub fn total_width(&self) -> usize {
        self.widths.iter().sum::<usize>() + self.widths.len() + 1
    }
}

// ============================================================================
// ANSI Code Handling
// ============================================================================

/// Strip ANSI escape sequences from a string.
///
/// Returns only the visible characters without color/style codes.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::strip_ansi_codes;
///
/// assert_eq!(strip_ansi_codes("plain text"), "plain text");
/// assert_eq!(strip_ansi_codes("\x1b[32mgreen\x1b[0m"), "green");
/// assert_eq!(strip_ansi_codes("\x1b[1;31mbold red\x1b[0m"), "bold red");
/// ```
#[must_use]
pub fn strip_ansi_codes(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_escape = false;

    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if c == 'm' {
                in_escape = false;
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Calculate the display width of a string (visible characters only).
///
/// This counts visible characters, ignoring ANSI escape sequences.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::display_width;
///
/// assert_eq!(display_width("hello"), 5);
/// assert_eq!(display_width("\x1b[32mgreen\x1b[0m"), 5);
/// ```
#[must_use]
pub fn display_width(s: &str) -> usize {
    strip_ansi_codes(s).chars().count()
}

// ============================================================================
// Safe Truncation
// ============================================================================

/// Truncate a string to a maximum display width, preserving ANSI codes.
///
/// If the visible content exceeds `max_width`, it is truncated.
/// ANSI escape sequences are preserved in the output, and any
/// open sequences are properly closed with a reset code.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::{truncate_to_width, display_width};
///
/// assert_eq!(truncate_to_width("hello world", 5), "hello");
/// assert_eq!(truncate_to_width("hi", 10), "hi");
///
/// // Colored content is truncated but color codes are preserved
/// let colored = "\x1b[32mgreen text\x1b[0m";
/// let truncated = truncate_to_width(colored, 5);
/// assert_eq!(display_width(&truncated), 5);
/// assert!(truncated.contains("\x1b[32m")); // Color preserved
/// ```
#[must_use]
pub fn truncate_to_width(s: &str, max_width: usize) -> String {
    let mut result = String::with_capacity(s.len());
    let mut visible_count = 0;
    let mut in_escape = false;
    let mut has_ansi = false;

    for c in s.chars() {
        if c == '\x1b' {
            in_escape = true;
            has_ansi = true;
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
        // Characters beyond max_width are dropped (except ANSI codes)
    }

    // Close any open ANSI sequences with reset
    if has_ansi && !result.ends_with("\x1b[0m") {
        result.push_str("\x1b[0m");
    }

    result
}

// ============================================================================
// Safe Cell Formatting
// ============================================================================

/// Format content into a table cell with guaranteed width.
///
/// This function NEVER allows the cell to exceed the specified width.
/// Content is truncated if necessary to prevent format expansion.
///
/// # Arguments
///
/// * `content` - The cell content (may contain ANSI codes)
/// * `width` - Total cell width (including padding spaces)
/// * `alignment` - How to align content within the cell
///
/// # Returns
///
/// A string of exactly `width` display characters (plus any ANSI codes).
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::{format_cell, display_width};
/// use powers_rs::display::components::table::Alignment;
///
/// let cell = format_cell("test", 10, Alignment::Center);
/// assert_eq!(display_width(&cell), 10);
///
/// // Long content is truncated
/// let cell = format_cell("very long content", 10, Alignment::Left);
/// assert_eq!(display_width(&cell), 10);
/// ```
#[must_use]
pub fn format_cell(
    content: &str,
    width: usize,
    alignment: Alignment,
) -> String {
    // Format width is cell width minus 2 padding spaces
    let format_width = width.saturating_sub(2);

    let visible_len = display_width(content);

    // Truncate if necessary to prevent expansion
    let safe_content = if visible_len > format_width {
        truncate_to_width(content, format_width)
    } else {
        content.to_string()
    };

    // Calculate padding needed
    let safe_visible_len = display_width(&safe_content);
    let padding_total = format_width.saturating_sub(safe_visible_len);

    let (pad_left, pad_right) = match alignment {
        Alignment::Left => (0, padding_total),
        Alignment::Right => (padding_total, 0),
        Alignment::Center => {
            let left = padding_total / 2;
            let right = padding_total - left;
            (left, right)
        }
    };

    format!(
        " {}{}{} ",
        " ".repeat(pad_left),
        safe_content,
        " ".repeat(pad_right)
    )
}

// ============================================================================
// Row Construction Utilities
// ============================================================================

/// Build a table row from formatted cells.
///
/// # Arguments
///
/// * `cells` - Pre-formatted cell strings (from `format_cell`)
/// * `border` - Border characters to use
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::{format_cell, build_row};
/// use powers_rs::display::components::table::{Alignment, BorderStyle};
///
/// let border = BorderStyle::Standard.chars().unwrap();
/// let cells = vec![
///     format_cell("A", 5, Alignment::Left),
///     format_cell("B", 5, Alignment::Right),
/// ];
/// let row = build_row(&cells, &border);
/// assert!(row.starts_with('│'));
/// assert!(row.ends_with('│'));
/// ```
#[must_use]
pub fn build_row(cells: &[String], border: &BorderChars) -> String {
    format!(
        "{}{}{}",
        border.vertical,
        cells.join(&border.vertical.to_string()),
        border.vertical
    )
}

/// Build a horizontal border line.
#[must_use]
fn build_horizontal_border(
    widths: &[usize],
    border: &BorderChars,
    left: char,
    middle: char,
    right: char,
) -> String {
    format!(
        "{}{}{}",
        left,
        widths
            .iter()
            .map(|&w| border.horizontal.to_string().repeat(w))
            .collect::<Vec<_>>()
            .join(&middle.to_string()),
        right
    )
}

/// Build the top border of a table.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table_format::build_top_border;
/// use powers_rs::display::components::table::BorderStyle;
///
/// let border = BorderStyle::Standard.chars().unwrap();
/// let top = build_top_border(&[10, 15, 10], &border);
/// assert!(top.starts_with('┌'));
/// assert!(top.ends_with('┐'));
/// ```
#[must_use]
pub fn build_top_border(widths: &[usize], border: &BorderChars) -> String {
    build_horizontal_border(
        widths,
        border,
        border.top_left,
        border.top_tee,
        border.top_right,
    )
}

/// Build a separator row between data rows.
///
/// Used between header and body, or between data sections.
#[must_use]
pub fn build_separator(widths: &[usize], border: &BorderChars) -> String {
    build_horizontal_border(
        widths,
        border,
        border.left_tee,
        border.cross,
        border.right_tee,
    )
}

/// Build the bottom border of a table.
#[must_use]
pub fn build_bottom_border(widths: &[usize], border: &BorderChars) -> String {
    build_horizontal_border(
        widths,
        border,
        border.bottom_left,
        border.bottom_tee,
        border.bottom_right,
    )
}

/// Build a header row with centered headers.
///
/// All headers are centered regardless of the column's data alignment.
#[must_use]
pub fn build_header_row(
    config: &TableColumnConfig,
    border: &BorderChars,
) -> String {
    let cells: Vec<String> = config
        .headers
        .iter()
        .zip(&config.widths)
        .map(|(header, &width)| format_cell(header, width, Alignment::Center))
        .collect();

    build_row(&cells, border)
}

/// Build a complete table header (top border + header row + separator).
///
/// Convenience function that combines the three header components.
#[must_use]
pub fn build_table_header(
    config: &TableColumnConfig,
    border_style: BorderStyle,
) -> String {
    let Some(border) = border_style.chars() else {
        return String::new();
    };

    let top = build_top_border(&config.widths, &border);
    let header = build_header_row(config, &border);
    let sep = build_separator(&config.widths, &border);

    format!("{}\n{}\n{}", top, header, sep)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ANSI Code Handling Tests
    // ========================================================================

    #[test]
    fn test_strip_ansi_codes_plain() {
        assert_eq!(strip_ansi_codes("hello"), "hello");
        assert_eq!(strip_ansi_codes("hello world"), "hello world");
        assert_eq!(strip_ansi_codes(""), "");
    }

    #[test]
    fn test_strip_ansi_codes_colored() {
        assert_eq!(strip_ansi_codes("\x1b[32mgreen\x1b[0m"), "green");
        assert_eq!(strip_ansi_codes("\x1b[1;31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi_codes("\x1b[0m"), "");
    }

    #[test]
    fn test_strip_ansi_codes_mixed() {
        assert_eq!(
            strip_ansi_codes("hello \x1b[32mworld\x1b[0m!"),
            "hello world!"
        );
    }

    #[test]
    fn test_display_width_plain() {
        assert_eq!(display_width("hello"), 5);
        assert_eq!(display_width(""), 0);
        assert_eq!(display_width("12345"), 5);
    }

    #[test]
    fn test_display_width_ansi() {
        assert_eq!(display_width("\x1b[32mgreen\x1b[0m"), 5);
        assert_eq!(display_width("\x1b[1;31mbold red\x1b[0m"), 8);
    }

    // ========================================================================
    // Truncation Tests
    // ========================================================================

    #[test]
    fn test_truncate_to_width_no_truncation() {
        assert_eq!(truncate_to_width("hello", 10), "hello");
        assert_eq!(truncate_to_width("hi", 5), "hi");
    }

    #[test]
    fn test_truncate_to_width_basic() {
        assert_eq!(truncate_to_width("hello", 3), "hel");
        assert_eq!(truncate_to_width("abcdefghij", 5), "abcde");
    }

    #[test]
    fn test_truncate_to_width_zero() {
        assert_eq!(truncate_to_width("hello", 0), "");
    }

    #[test]
    fn test_truncate_to_width_ansi_preserved() {
        let colored = "\x1b[32mgreen\x1b[0m";
        let truncated = truncate_to_width(colored, 3);

        // Should contain the color start
        assert!(truncated.contains("\x1b[32m"));
        // Should end with reset
        assert!(truncated.ends_with("\x1b[0m"));
        // Should have correct display width
        assert_eq!(display_width(&truncated), 3);
    }

    #[test]
    fn test_truncate_to_width_ansi_in_middle() {
        let text = "ab\x1b[32mcd\x1b[0mef";
        let truncated = truncate_to_width(text, 3);
        assert_eq!(display_width(&truncated), 3);
    }

    // ========================================================================
    // Cell Formatting Tests
    // ========================================================================

    #[test]
    fn test_format_cell_exact_fit() {
        // Content exactly fills format width (8 - 2 padding = 6)
        let cell = format_cell("abcdef", 8, Alignment::Center);
        assert_eq!(display_width(&cell), 8);
    }

    #[test]
    fn test_format_cell_short_content() {
        let cell = format_cell("ab", 10, Alignment::Center);
        assert_eq!(display_width(&cell), 10);
    }

    #[test]
    fn test_format_cell_truncation() {
        let long_content = "this is very long content that exceeds width";
        let cell = format_cell(long_content, 15, Alignment::Left);
        assert_eq!(display_width(&cell), 15);
    }

    #[test]
    fn test_format_cell_alignment_left() {
        let cell = format_cell("ab", 8, Alignment::Left);
        assert_eq!(display_width(&cell), 8);
        // Content should be at the left (after space padding)
        let stripped = strip_ansi_codes(&cell);
        assert!(stripped.starts_with(" ab"));
    }

    #[test]
    fn test_format_cell_alignment_right() {
        let cell = format_cell("ab", 8, Alignment::Right);
        assert_eq!(display_width(&cell), 8);
        // Content should be at the right (before space padding)
        let stripped = strip_ansi_codes(&cell);
        assert!(stripped.ends_with("ab "));
    }

    #[test]
    fn test_format_cell_alignment_center() {
        let cell = format_cell("ab", 8, Alignment::Center);
        assert_eq!(display_width(&cell), 8);
    }

    #[test]
    fn test_format_cell_with_ansi() {
        let colored = "\x1b[32m$1,234.56\x1b[0m";
        let cell = format_cell(colored, 15, Alignment::Center);
        assert_eq!(display_width(&cell), 15);
    }

    #[test]
    fn test_format_cell_empty() {
        let cell = format_cell("", 10, Alignment::Center);
        assert_eq!(display_width(&cell), 10);
    }

    // ========================================================================
    // Column Configuration Tests
    // ========================================================================

    #[test]
    fn test_table_column_config_advanced() {
        let config = TableColumnConfig::advanced();
        assert_eq!(config.widths.len(), 10);
        assert_eq!(config.headers.len(), 10);
        assert_eq!(config.alignments.len(), 10);
    }

    #[test]
    fn test_table_column_config_advanced_units() {
        let units = TableColumnConfig::advanced_units();
        assert_eq!(units.len(), 10);
        // Count timing columns with "(s)" unit
        let timing_count = units.iter().filter(|&&u| u == "(s)").count();
        assert_eq!(timing_count, 6);
    }

    #[test]
    fn test_table_column_config_standard() {
        let config = TableColumnConfig::standard();
        assert_eq!(config.widths.len(), 5);
        assert_eq!(config.headers.len(), 5);
        assert_eq!(config.alignments.len(), 5);
    }

    #[test]
    fn test_total_width_calculation() {
        let config = TableColumnConfig::advanced();
        // 6+15+17+8+12+10+16+10+16+15 = 125 column chars
        // + 11 border chars (│ between each column and at ends)
        let expected = 125 + 11;
        assert_eq!(config.total_width(), expected);
    }

    // ========================================================================
    // Row Construction Tests
    // ========================================================================

    #[test]
    fn test_build_row() {
        let border = BorderStyle::Standard.chars().unwrap();
        let cells = vec![
            format_cell("A", 5, Alignment::Left),
            format_cell("B", 5, Alignment::Right),
        ];
        let row = build_row(&cells, &border);

        assert!(row.starts_with('│'));
        assert!(row.ends_with('│'));
        assert!(row.contains('A'));
        assert!(row.contains('B'));
    }

    #[test]
    fn test_build_top_border() {
        let border = BorderStyle::Standard.chars().unwrap();
        let top = build_top_border(&[10, 15], &border);

        assert!(top.starts_with('┌'));
        assert!(top.ends_with('┐'));
        assert!(top.contains('┬'));
        assert!(top.contains('─'));
    }

    #[test]
    fn test_build_separator() {
        let border = BorderStyle::Standard.chars().unwrap();
        let sep = build_separator(&[10, 15], &border);

        assert!(sep.starts_with('├'));
        assert!(sep.ends_with('┤'));
        assert!(sep.contains('┼'));
    }

    #[test]
    fn test_build_bottom_border() {
        let border = BorderStyle::Standard.chars().unwrap();
        let bottom = build_bottom_border(&[10, 15], &border);

        assert!(bottom.starts_with('└'));
        assert!(bottom.ends_with('┘'));
        assert!(bottom.contains('┴'));
    }

    #[test]
    fn test_build_header_row() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();
        let row = build_header_row(&config, &border);

        // Each header should appear in the row
        for header in &config.headers {
            assert!(row.contains(header), "Missing header: {}", header);
        }
    }

    // ========================================================================
    // Alignment Invariant Tests
    // ========================================================================

    #[test]
    fn test_all_row_types_same_width_advanced() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();
        let expected = config.total_width();

        let top = build_top_border(&config.widths, &border);
        let header = build_header_row(&config, &border);
        let sep = build_separator(&config.widths, &border);
        let bottom = build_bottom_border(&config.widths, &border);

        assert_eq!(display_width(&top), expected, "Top border width mismatch");
        assert_eq!(
            display_width(&header),
            expected,
            "Header row width mismatch"
        );
        assert_eq!(display_width(&sep), expected, "Separator width mismatch");
        assert_eq!(
            display_width(&bottom),
            expected,
            "Bottom border width mismatch"
        );
    }

    #[test]
    fn test_all_row_types_same_width_standard() {
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

    #[test]
    fn test_data_row_matches_header_width() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();

        // Build a data row with various content
        let cells: Vec<String> = config
            .widths
            .iter()
            .zip(&config.alignments)
            .map(|(&w, &a)| format_cell("test", w, a))
            .collect();

        let data_row = build_row(&cells, &border);
        let header = build_header_row(&config, &border);

        assert_eq!(
            display_width(&data_row),
            display_width(&header),
            "Data row width should match header width"
        );
    }

    #[test]
    fn test_large_values_dont_break_alignment() {
        let config = TableColumnConfig::advanced();
        let _border = BorderStyle::Standard.chars().unwrap();

        // Test with large values that might overflow
        let large_values = [
            "$123,456,789,012.34", // Very large cost
            "99999",               // Large iteration
            "1000.00%",            // Large percentage
            "999m 59s / 999m 59s", // Large timing
        ];

        for value in &large_values {
            for (&width, &align) in config.widths.iter().zip(&config.alignments)
            {
                let cell = format_cell(value, width, align);
                assert_eq!(
                    display_width(&cell),
                    width,
                    "Cell width mismatch for value '{}' in column width {}",
                    value,
                    width
                );
            }
        }
    }

    #[test]
    fn test_colored_and_plain_same_width() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();

        // Plain cells
        let plain_cells: Vec<String> = config
            .widths
            .iter()
            .map(|&w| format_cell("test", w, Alignment::Center))
            .collect();

        // Colored cells
        let colored_cells: Vec<String> = config
            .widths
            .iter()
            .map(|&w| format_cell("\x1b[32mtest\x1b[0m", w, Alignment::Center))
            .collect();

        let plain_row = build_row(&plain_cells, &border);
        let colored_row = build_row(&colored_cells, &border);

        assert_eq!(
            display_width(&plain_row),
            display_width(&colored_row),
            "Plain and colored rows should have same display width"
        );
    }

    #[test]
    fn test_headers_fit_in_columns() {
        let config = TableColumnConfig::advanced();

        for (header, &width) in config.headers.iter().zip(&config.widths) {
            let available = width.saturating_sub(2); // Space for padding
            assert!(
                header.len() <= available,
                "Header '{}' ({} chars) doesn't fit in column width {} (available: {})",
                header,
                header.len(),
                width,
                available
            );
        }
    }

    #[test]
    fn test_standard_headers_fit_in_columns() {
        let config = TableColumnConfig::standard();

        for (header, &width) in config.headers.iter().zip(&config.widths) {
            let available = width.saturating_sub(2);
            assert!(
                header.len() <= available,
                "Header '{}' doesn't fit in column width {}",
                header,
                width
            );
        }
    }

    // ========================================================================
    // Build Table Header Tests
    // ========================================================================

    #[test]
    fn test_build_table_header() {
        let config = TableColumnConfig::advanced();
        let header = build_table_header(&config, BorderStyle::Standard);

        // Should contain all three parts
        assert!(header.contains('┌')); // Top border
        assert!(header.contains("Iter")); // Header content
        assert!(header.contains('├')); // Separator

        // Should be 3 lines
        let lines: Vec<&str> = header.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_build_table_header_no_border() {
        let config = TableColumnConfig::advanced();
        let header = build_table_header(&config, BorderStyle::None);
        assert!(header.is_empty());
    }
}
