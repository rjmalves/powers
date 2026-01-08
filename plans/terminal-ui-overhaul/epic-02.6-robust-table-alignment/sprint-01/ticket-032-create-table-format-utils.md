# [T-032] Create robust table formatting utilities

> **Epic**: [Epic 2.6: Robust Table Alignment](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-031](./ticket-031-remove-tabled-dependency.md)  
> **Blocks**: [T-033](./ticket-033-migrate-advanced-renderer.md), [T-034](./ticket-034-migrate-standard-renderer.md)

## Context

### Background

The root cause of table alignment issues is Rust's `format!` macro expanding fields when content exceeds the specified width. This ticket creates a new utility module that guarantees alignment by:

1. Calculating display width correctly (ignoring ANSI codes)
2. Truncating content before formatting (preventing expansion)
3. Providing centralized column configuration
4. Adding generous width slack for unexpected content

### Files to Read Before Starting

- `src/display/components/table.rs` - Existing `BorderStyle` enum and `BorderChars`
- `src/display/components/statistics.rs` - Content formatters that produce variable-length strings
- `src/display/renderers/advanced.rs` - Current manual width management approach
- `docs/TABLE_ALIGNMENT_ANALYSIS.md` - Detailed analysis of alignment issues

## Specification

### New File: `src/display/components/table_format.rs`

Create a new module with the following components:

### 1. Column Alignment Enum

```rust
/// Text alignment within a table cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    Left,
    Center,
    Right,
}
```

### 2. Table Column Configuration

```rust
/// Configuration for a table's column structure.
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
    /// Advanced renderer column configuration.
    /// Generous widths to prevent overflow.
    pub fn advanced() -> Self {
        Self {
            widths: vec![8, 22, 20, 20, 12, 20],
            headers: vec![
                "Iter",
                "Lower Bound ($)",
                "Simul Cost ($)",
                "1st Stage ($)",
                "Gap %",
                "Time (fwd/bwd)",
            ],
            alignments: vec![
                Alignment::Right,   // Iter
                Alignment::Center,  // Lower Bound
                Alignment::Center,  // Simul Cost
                Alignment::Center,  // 1st Stage
                Alignment::Center,  // Gap %
                Alignment::Center,  // Time
            ],
        }
    }

    /// Standard renderer column configuration.
    pub fn standard() -> Self {
        Self {
            widths: vec![8, 22, 20, 12, 20],
            headers: vec![
                "Iter",
                "Lower Bound ($)",
                "Simul Cost ($)",
                "Gap %",
                "Time (fwd/bwd)",
            ],
            alignments: vec![
                Alignment::Right,
                Alignment::Center,
                Alignment::Center,
                Alignment::Center,
                Alignment::Center,
            ],
        }
    }

    /// Total width of the table including borders.
    pub fn total_width(&self) -> usize {
        self.widths.iter().sum::<usize>() + self.widths.len() + 1
    }
}
```

### 3. ANSI Code Handling

```rust
/// Strip ANSI escape sequences from a string.
/// Returns only the visible characters.
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
pub fn display_width(s: &str) -> usize {
    strip_ansi_codes(s).chars().count()
}
```

### 4. Safe Truncation

```rust
/// Truncate a string to a maximum display width, preserving ANSI codes.
/// 
/// If the visible content exceeds `max_width`, it is truncated.
/// Any open ANSI sequences are properly closed with a reset code.
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
```

### 5. Safe Cell Formatting

```rust
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
/// A string of exactly `width` characters (plus any ANSI codes).
pub fn format_cell(content: &str, width: usize, alignment: Alignment) -> String {
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
```

### 6. Row Construction Utilities

```rust
use super::table::{BorderChars, BorderStyle};

/// Build a table row from cells.
pub fn build_row(cells: &[String], border: &BorderChars) -> String {
    format!(
        "{}{}{}",
        border.vertical,
        cells.join(&border.vertical.to_string()),
        border.vertical
    )
}

/// Build a horizontal border line.
pub fn build_horizontal_border(
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
pub fn build_header_row(config: &TableColumnConfig, border: &BorderChars) -> String {
    let cells: Vec<String> = config
        .headers
        .iter()
        .zip(&config.widths)
        .map(|(header, &width)| format_cell(header, width, Alignment::Center))
        .collect();
    
    build_row(&cells, border)
}
```

### 7. Module Export

Update `src/display/components/mod.rs`:

```rust
pub mod table_format;
```

## Acceptance Criteria

- [ ] New file `src/display/components/table_format.rs` created
- [ ] All functions documented with doc comments
- [ ] `strip_ansi_codes()` correctly handles nested ANSI sequences
- [ ] `truncate_to_width()` preserves ANSI codes and closes sequences
- [ ] `format_cell()` never produces cells wider than specified width
- [ ] `TableColumnConfig::advanced()` and `standard()` defined
- [ ] Module exported from `components/mod.rs`
- [ ] Unit tests for all functions
- [ ] `cargo build -j 1` succeeds
- [ ] `cargo test -j 1 -- --test-threads=1` passes
- [ ] `cargo clippy` clean

## Testing Requirements

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_ansi_codes() {
        assert_eq!(strip_ansi_codes("hello"), "hello");
        assert_eq!(strip_ansi_codes("\x1b[32mgreen\x1b[0m"), "green");
        assert_eq!(strip_ansi_codes("\x1b[1;31mred\x1b[0m"), "red");
    }

    #[test]
    fn test_display_width() {
        assert_eq!(display_width("hello"), 5);
        assert_eq!(display_width("\x1b[32mgreen\x1b[0m"), 5);
    }

    #[test]
    fn test_truncate_to_width() {
        assert_eq!(truncate_to_width("hello", 3), "hel");
        assert_eq!(truncate_to_width("hello", 10), "hello");
        // With ANSI - should preserve codes and add reset
        let colored = "\x1b[32mgreen\x1b[0m";
        let truncated = truncate_to_width(colored, 3);
        assert!(truncated.contains("\x1b[32m"));
        assert!(truncated.ends_with("\x1b[0m"));
        assert_eq!(display_width(&truncated), 3);
    }

    #[test]
    fn test_format_cell_exact_fit() {
        let cell = format_cell("test", 8, Alignment::Center);
        assert_eq!(display_width(&cell), 8);
    }

    #[test]
    fn test_format_cell_truncation() {
        let long_content = "this is very long content";
        let cell = format_cell(long_content, 10, Alignment::Left);
        assert_eq!(display_width(&cell), 10);
    }

    #[test]
    fn test_format_cell_with_ansi() {
        let colored = "\x1b[32m$1,234.56\x1b[0m";
        let cell = format_cell(colored, 15, Alignment::Center);
        assert_eq!(display_width(&cell), 15);
    }

    #[test]
    fn test_table_column_config_advanced() {
        let config = TableColumnConfig::advanced();
        assert_eq!(config.widths.len(), 6);
        assert_eq!(config.headers.len(), 6);
        assert_eq!(config.alignments.len(), 6);
    }

    #[test]
    fn test_header_row_alignment() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();
        let row = build_header_row(&config, &border);
        
        // Each header should appear in the row
        for header in &config.headers {
            assert!(row.contains(header));
        }
    }

    #[test]
    fn test_row_widths_match() {
        let config = TableColumnConfig::advanced();
        let border = BorderStyle::Standard.chars().unwrap();
        
        let top = build_top_border(&config.widths, &border);
        let header = build_header_row(&config, &border);
        let sep = build_separator(&config.widths, &border);
        let bottom = build_bottom_border(&config.widths, &border);
        
        let expected_width = config.total_width();
        assert_eq!(display_width(&top), expected_width);
        assert_eq!(display_width(&header), expected_width);
        assert_eq!(display_width(&sep), expected_width);
        assert_eq!(display_width(&bottom), expected_width);
    }
}
```

## Implementation Guide

### Key Files to Modify

- Create: `src/display/components/table_format.rs`
- Modify: `src/display/components/mod.rs` (add export)

### Patterns to Follow

- See `src/display/components/statistics.rs` for similar utility module structure
- Use `#[must_use]` on functions that return values
- Add `/// # Examples` sections to doc comments

### Pitfalls to Avoid

- ⚠️ Don't use regex for ANSI stripping (adds dependency, slower than char iteration)
- ⚠️ Don't forget to close ANSI sequences after truncation
- ⚠️ Width must account for the two padding spaces (` content `)
- ⚠️ Use `saturating_sub` to avoid underflow

## Effort Estimate

**Points**: 5  
**Confidence**: High  
**Rationale**: Core utility module with clear specification and comprehensive tests

## Definition of Done

- [x] All specified functions implemented
- [x] All unit tests passing
- [x] Doc comments complete
- [x] No clippy warnings
- [x] Formatted with rustfmt

---

**Status**: ✅ COMPLETE
