# T-018: Implement table builder component

> **Epic**: [Training Display](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: T-014 (color utilities)
> **Blocks**: T-020, T-021, T-023

## Files to Read Before Starting

- `src/display/components/color.rs` - Color utilities
- `src/display/terminal.rs` - Terminal width detection
- Master plan sample output - Target table appearance

## Context

### Background

The Advanced and Standard renderers display metrics in formatted tables with box-drawing borders. This component provides a flexible table builder that handles column alignment, borders, and multi-line rows.

### Current State

No table formatting exists. Current output uses simple space-padded columns without borders.

## Specification

### Create `src/display/components/table.rs`

#### Types

```rust
/// Border style for tables
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BorderStyle {
    /// No borders
    None,
    /// ASCII: +--+, |
    Ascii,
    /// Standard box drawing: ┌──┐, │
    #[default]
    Standard,
    /// Rounded corners: ╭──╮, │
    Rounded,
    /// Heavy lines: ┏━━┓, ┃
    Heavy,
}

/// Text alignment within a column
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alignment {
    #[default]
    Left,
    Center,
    Right,
}

/// Column definition
pub struct Column {
    /// Column header text
    pub header: String,
    /// Minimum width (characters)
    pub min_width: usize,
    /// Maximum width (characters, 0 = unlimited)
    pub max_width: usize,
    /// Text alignment
    pub alignment: Alignment,
}

/// A row of data
pub struct Row {
    /// Cell values (one per column)
    pub cells: Vec<String>,
    /// Optional continuation lines for this row
    pub continuation: Option<String>,
}

/// Table builder for constructing formatted tables
pub struct TableBuilder {
    columns: Vec<Column>,
    rows: Vec<Row>,
    border_style: BorderStyle,
    header_separator: bool,
}
```

#### Border Characters

```rust
struct BorderChars {
    top_left: char,
    top_right: char,
    bottom_left: char,
    bottom_right: char,
    horizontal: char,
    vertical: char,
    cross: char,
    top_tee: char,
    bottom_tee: char,
    left_tee: char,
    right_tee: char,
}

impl BorderStyle {
    fn chars(&self) -> Option<BorderChars> {
        match self {
            BorderStyle::None => None,
            BorderStyle::Ascii => Some(BorderChars {
                top_left: '+', top_right: '+',
                bottom_left: '+', bottom_right: '+',
                horizontal: '-', vertical: '|',
                cross: '+', top_tee: '+', bottom_tee: '+',
                left_tee: '+', right_tee: '+',
            }),
            BorderStyle::Standard => Some(BorderChars {
                top_left: '┌', top_right: '┐',
                bottom_left: '└', bottom_right: '┘',
                horizontal: '─', vertical: '│',
                cross: '┼', top_tee: '┬', bottom_tee: '┴',
                left_tee: '├', right_tee: '┤',
            }),
            BorderStyle::Rounded => Some(BorderChars {
                top_left: '╭', top_right: '╮',
                bottom_left: '╰', bottom_right: '╯',
                horizontal: '─', vertical: '│',
                cross: '┼', top_tee: '┬', bottom_tee: '┴',
                left_tee: '├', right_tee: '┤',
            }),
            BorderStyle::Heavy => Some(BorderChars {
                top_left: '┏', top_right: '┓',
                bottom_left: '┗', bottom_right: '┛',
                horizontal: '━', vertical: '┃',
                cross: '╋', top_tee: '┳', bottom_tee: '┻',
                left_tee: '┣', right_tee: '┫',
            }),
        }
    }
}
```

#### TableBuilder Methods

```rust
impl TableBuilder {
    /// Create a new table builder
    pub fn new() -> Self;
    
    /// Set border style
    pub fn border_style(self, style: BorderStyle) -> Self;
    
    /// Enable/disable header separator line
    pub fn header_separator(self, enabled: bool) -> Self;
    
    /// Add a column definition
    pub fn column(self, header: &str, min_width: usize, alignment: Alignment) -> Self;
    
    /// Add a column with all options
    pub fn column_full(self, column: Column) -> Self;
    
    /// Add a data row
    pub fn row(self, cells: Vec<String>) -> Self;
    
    /// Add a row with continuation line
    pub fn row_with_continuation(self, cells: Vec<String>, continuation: String) -> Self;
    
    /// Calculate column widths based on content
    fn calculate_widths(&self) -> Vec<usize>;
    
    /// Render the complete table as a string
    pub fn build(&self) -> String;
    
    /// Render a single row (for streaming output)
    pub fn render_row(&self, row_index: usize, widths: &[usize]) -> String;
    
    /// Render just the header section
    pub fn render_header(&self) -> String;
    
    /// Render the top border line
    pub fn render_top_border(&self, widths: &[usize]) -> String;
    
    /// Render a separator line (between header and body)
    pub fn render_separator(&self, widths: &[usize]) -> String;
    
    /// Render the bottom border line
    pub fn render_bottom_border(&self, widths: &[usize]) -> String;
}
```

### Expected Output (Standard Border)

```
┌─────┬────────────────┬────────────────┬───────┐
│ Iter│ Lower Bound ($)│ Simul Cost ($) │ Gap % │
├─────┼────────────────┼────────────────┼───────┤
│   1 │   1.0148e+05   │   1.2823e+05   │ 26.4↓ │
│   2 │   1.2041e+05   │   1.2982e+05   │  7.8↓ │
└─────┴────────────────┴────────────────┴───────┘
```

### Width Calculation

1. For each column, find the maximum width of header and all cells
2. Clamp to min_width and max_width
3. If total width exceeds terminal width, proportionally shrink columns

## Acceptance Criteria

- [ ] All border styles implemented (None, Ascii, Standard, Rounded, Heavy)
- [ ] Column alignment works (left, center, right)
- [ ] Header separator optional
- [ ] Multi-line row continuation works
- [ ] Width calculation respects min/max constraints
- [ ] Builder pattern is ergonomic
- [ ] Streaming render methods work for incremental output
- [ ] Unit tests for all border styles

## Implementation Guide

### Step 1: Add to components/mod.rs

```rust
pub mod table;
```

### Step 2: Implement BorderChars

Define the border character sets as shown above.

### Step 3: Implement TableBuilder

```rust
impl TableBuilder {
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            border_style: BorderStyle::Standard,
            header_separator: true,
        }
    }
    
    pub fn border_style(mut self, style: BorderStyle) -> Self {
        self.border_style = style;
        self
    }
    
    pub fn column(mut self, header: &str, min_width: usize, alignment: Alignment) -> Self {
        self.columns.push(Column {
            header: header.to_string(),
            min_width,
            max_width: 0,
            alignment,
        });
        self
    }
    
    pub fn row(mut self, cells: Vec<String>) -> Self {
        self.rows.push(Row { cells, continuation: None });
        self
    }
}
```

### Step 4: Implement build

```rust
pub fn build(&self) -> String {
    let widths = self.calculate_widths();
    let mut lines = Vec::new();
    
    // Top border
    if self.border_style != BorderStyle::None {
        lines.push(self.render_top_border(&widths));
    }
    
    // Header row
    lines.push(self.render_header_row(&widths));
    
    // Header separator
    if self.header_separator && self.border_style != BorderStyle::None {
        lines.push(self.render_separator(&widths));
    }
    
    // Data rows
    for (i, row) in self.rows.iter().enumerate() {
        lines.push(self.render_data_row(row, &widths));
        if let Some(cont) = &row.continuation {
            lines.push(self.render_continuation(cont, &widths));
        }
    }
    
    // Bottom border
    if self.border_style != BorderStyle::None {
        lines.push(self.render_bottom_border(&widths));
    }
    
    lines.join("\n")
}
```

### Step 5: Implement alignment

```rust
fn align_text(text: &str, width: usize, alignment: Alignment) -> String {
    let text_len = text.chars().count();
    if text_len >= width {
        return text.chars().take(width).collect();
    }
    
    let padding = width - text_len;
    match alignment {
        Alignment::Left => format!("{}{}", text, " ".repeat(padding)),
        Alignment::Right => format!("{}{}", " ".repeat(padding), text),
        Alignment::Center => {
            let left = padding / 2;
            let right = padding - left;
            format!("{}{}{}", " ".repeat(left), text, " ".repeat(right))
        }
    }
}
```

### Patterns to Follow

- Builder pattern for ergonomic API
- Separate calculation from rendering
- Unicode-aware character counting

### Pitfalls to Avoid

- ⚠️ Use `chars().count()` not `len()` for Unicode
- ⚠️ Handle empty tables gracefully
- ⚠️ Continuation lines need special handling

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_simple_table() {
    let table = TableBuilder::new()
        .border_style(BorderStyle::Ascii)
        .column("A", 3, Alignment::Left)
        .column("B", 3, Alignment::Right)
        .row(vec!["1".to_string(), "2".to_string()])
        .build();
    
    let expected = r#"+---+---+
| A |   B |
+---+---+
| 1 |   2 |
+---+---+"#;
    assert_eq!(table, expected);
}

#[test]
fn test_alignment() {
    assert_eq!(align_text("a", 5, Alignment::Left), "a    ");
    assert_eq!(align_text("a", 5, Alignment::Right), "    a");
    assert_eq!(align_text("a", 5, Alignment::Center), "  a  ");
}

#[test]
fn test_border_styles() {
    // Test that each style produces correct corner characters
    for style in [BorderStyle::Ascii, BorderStyle::Standard, 
                  BorderStyle::Rounded, BorderStyle::Heavy] {
        let table = TableBuilder::new()
            .border_style(style)
            .column("X", 1, Alignment::Left)
            .row(vec!["1".to_string()])
            .build();
        
        assert!(!table.is_empty());
    }
}

#[test]
fn test_no_border() {
    let table = TableBuilder::new()
        .border_style(BorderStyle::None)
        .column("A", 3, Alignment::Left)
        .row(vec!["1".to_string()])
        .build();
    
    // Should not contain box-drawing characters
    assert!(!table.contains('│'));
    assert!(!table.contains('─'));
    assert!(!table.contains('+'));
}

#[test]
fn test_continuation_row() {
    let table = TableBuilder::new()
        .border_style(BorderStyle::None)
        .column("A", 5, Alignment::Left)
        .row_with_continuation(
            vec!["main".to_string()],
            "extra info".to_string()
        )
        .build();
    
    assert!(table.contains("main"));
    assert!(table.contains("extra info"));
}
```

## Documentation Requirements

- [ ] Module-level docs with example of building a table
- [ ] Doc comments on all public types and methods
- [ ] Example showing different border styles

## Effort Estimate

**Points**: 4
**Confidence**: Medium
**Rationale**: Most complex component with many edge cases for alignment, borders, and continuation rows.

## Definition of Done

- [ ] Implementation complete
- [ ] All border styles work
- [ ] All alignment modes work
- [ ] Continuation rows work
- [ ] Tests passing
- [ ] Documentation complete
- [ ] PR reviewed and merged
