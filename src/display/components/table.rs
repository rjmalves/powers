//! Table builder component for formatted box-drawing tables.
//!
//! Provides flexible table construction with multiple border styles, column alignment,
//! and support for multi-line rows.

/// Border style for tables.
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

/// Text alignment within a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Alignment {
    #[default]
    Left,
    Center,
    Right,
}

/// Border characters for different styles.
#[derive(Debug, Clone, Copy)]
pub struct BorderChars {
    pub top_left: char,
    pub top_right: char,
    pub bottom_left: char,
    pub bottom_right: char,
    pub horizontal: char,
    pub vertical: char,
    pub cross: char,
    pub top_tee: char,
    pub bottom_tee: char,
    pub left_tee: char,
    pub right_tee: char,
}

impl BorderStyle {
    /// Get the border characters for this style.
    #[must_use]
    pub const fn chars(self) -> Option<BorderChars> {
        match self {
            Self::None => None,
            Self::Ascii => Some(BorderChars {
                top_left: '+',
                top_right: '+',
                bottom_left: '+',
                bottom_right: '+',
                horizontal: '-',
                vertical: '|',
                cross: '+',
                top_tee: '+',
                bottom_tee: '+',
                left_tee: '+',
                right_tee: '+',
            }),
            Self::Standard => Some(BorderChars {
                top_left: '┌',
                top_right: '┐',
                bottom_left: '└',
                bottom_right: '┘',
                horizontal: '─',
                vertical: '│',
                cross: '┼',
                top_tee: '┬',
                bottom_tee: '┴',
                left_tee: '├',
                right_tee: '┤',
            }),
            Self::Rounded => Some(BorderChars {
                top_left: '╭',
                top_right: '╮',
                bottom_left: '╰',
                bottom_right: '╯',
                horizontal: '─',
                vertical: '│',
                cross: '┼',
                top_tee: '┬',
                bottom_tee: '┴',
                left_tee: '├',
                right_tee: '┤',
            }),
            Self::Heavy => Some(BorderChars {
                top_left: '┏',
                top_right: '┓',
                bottom_left: '┗',
                bottom_right: '┛',
                horizontal: '━',
                vertical: '┃',
                cross: '╋',
                top_tee: '┳',
                bottom_tee: '┻',
                left_tee: '┣',
                right_tee: '┫',
            }),
        }
    }
}

/// Column definition.
#[derive(Debug, Clone)]
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

impl Column {
    /// Create a new column with specified properties.
    #[must_use]
    pub fn new(
        header: impl Into<String>,
        min_width: usize,
        alignment: Alignment,
    ) -> Self {
        Self {
            header: header.into(),
            min_width,
            max_width: 0,
            alignment,
        }
    }
}

/// A row of data.
#[derive(Debug, Clone)]
pub struct Row {
    /// Cell values (one per column)
    pub cells: Vec<String>,
    /// Optional continuation line for this row
    pub continuation: Option<String>,
}

impl Row {
    /// Create a new row from cell values.
    #[must_use]
    pub fn new(cells: Vec<String>) -> Self {
        Self {
            cells,
            continuation: None,
        }
    }

    /// Create a row with a continuation line.
    #[must_use]
    pub fn with_continuation(cells: Vec<String>, continuation: String) -> Self {
        Self {
            cells,
            continuation: Some(continuation),
        }
    }
}

/// Table builder for constructing formatted tables.
///
/// # Examples
///
/// ```
/// use powers_rs::display::components::table::{TableBuilder, BorderStyle, Alignment};
///
/// let table = TableBuilder::new()
///     .border_style(BorderStyle::Ascii)
///     .column("Name", 10, Alignment::Left)
///     .column("Value", 8, Alignment::Right)
///     .row(vec!["Test".to_string(), "123".to_string()])
///     .build();
///
/// assert!(table.contains("Name"));
/// assert!(table.contains("Value"));
/// assert!(table.contains("Test"));
/// assert!(table.contains("123"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct TableBuilder {
    columns: Vec<Column>,
    rows: Vec<Row>,
    border_style: BorderStyle,
    header_separator: bool,
}

impl TableBuilder {
    /// Create a new table builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            border_style: BorderStyle::Standard,
            header_separator: true,
        }
    }

    /// Set border style.
    #[must_use]
    pub fn border_style(mut self, style: BorderStyle) -> Self {
        self.border_style = style;
        self
    }

    /// Enable/disable header separator line.
    #[must_use]
    pub fn header_separator(mut self, enabled: bool) -> Self {
        self.header_separator = enabled;
        self
    }

    /// Add a column definition.
    #[must_use]
    pub fn column(
        mut self,
        header: &str,
        min_width: usize,
        alignment: Alignment,
    ) -> Self {
        self.columns.push(Column::new(header, min_width, alignment));
        self
    }

    /// Add a data row.
    #[must_use]
    pub fn row(mut self, cells: Vec<String>) -> Self {
        self.rows.push(Row::new(cells));
        self
    }

    /// Add a row with continuation line.
    #[must_use]
    pub fn row_with_continuation(
        mut self,
        cells: Vec<String>,
        continuation: String,
    ) -> Self {
        self.rows.push(Row::with_continuation(cells, continuation));
        self
    }

    /// Calculate column widths based on content.
    fn calculate_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> =
            self.columns.iter().map(|c| c.min_width).collect();

        // Consider header widths
        for (i, col) in self.columns.iter().enumerate() {
            let header_len = strip_ansi(&col.header).chars().count();
            widths[i] = widths[i].max(header_len);
        }

        // Consider cell widths
        for row in &self.rows {
            for (i, cell) in row.cells.iter().enumerate() {
                if i < widths.len() {
                    let cell_len = strip_ansi(cell).chars().count();
                    widths[i] = widths[i].max(cell_len);
                }
            }
        }

        // Apply max_width constraints
        for (i, col) in self.columns.iter().enumerate() {
            if col.max_width > 0 {
                widths[i] = widths[i].min(col.max_width);
            }
        }

        widths
    }

    /// Render the complete table as a string.
    #[must_use]
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
        for row in &self.rows {
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

    /// Render the top border line.
    #[must_use]
    pub fn render_top_border(&self, widths: &[usize]) -> String {
        let Some(chars) = self.border_style.chars() else {
            return String::new();
        };

        let mut parts = vec![chars.top_left.to_string()];
        for (i, &width) in widths.iter().enumerate() {
            parts.push(chars.horizontal.to_string().repeat(width));
            if i < widths.len() - 1 {
                parts.push(chars.top_tee.to_string());
            }
        }
        parts.push(chars.top_right.to_string());
        parts.join("")
    }

    /// Render a separator line (between header and body).
    #[must_use]
    pub fn render_separator(&self, widths: &[usize]) -> String {
        let Some(chars) = self.border_style.chars() else {
            return String::new();
        };

        let mut parts = vec![chars.left_tee.to_string()];
        for (i, &width) in widths.iter().enumerate() {
            parts.push(chars.horizontal.to_string().repeat(width));
            if i < widths.len() - 1 {
                parts.push(chars.cross.to_string());
            }
        }
        parts.push(chars.right_tee.to_string());
        parts.join("")
    }

    /// Render the bottom border line.
    #[must_use]
    pub fn render_bottom_border(&self, widths: &[usize]) -> String {
        let Some(chars) = self.border_style.chars() else {
            return String::new();
        };

        let mut parts = vec![chars.bottom_left.to_string()];
        for (i, &width) in widths.iter().enumerate() {
            parts.push(chars.horizontal.to_string().repeat(width));
            if i < widths.len() - 1 {
                parts.push(chars.bottom_tee.to_string());
            }
        }
        parts.push(chars.bottom_right.to_string());
        parts.join("")
    }

    fn render_header_row(&self, widths: &[usize]) -> String {
        let vertical = self.border_style.chars().map_or(' ', |c| c.vertical);

        let cells: Vec<String> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, col)| align_text(&col.header, widths[i], col.alignment))
            .collect();

        format!(
            "{}{}{}",
            vertical,
            cells.join(&vertical.to_string()),
            vertical
        )
    }

    fn render_data_row(&self, row: &Row, widths: &[usize]) -> String {
        let vertical = self.border_style.chars().map_or(' ', |c| c.vertical);

        let cells: Vec<String> = row
            .cells
            .iter()
            .enumerate()
            .map(|(i, cell)| {
                let alignment = self
                    .columns
                    .get(i)
                    .map_or(Alignment::Left, |c| c.alignment);
                let width = widths.get(i).copied().unwrap_or(0);
                align_text(cell, width, alignment)
            })
            .collect();

        format!(
            "{}{}{}",
            vertical,
            cells.join(&vertical.to_string()),
            vertical
        )
    }

    fn render_continuation(
        &self,
        continuation: &str,
        widths: &[usize],
    ) -> String {
        let vertical = self.border_style.chars().map_or(' ', |c| c.vertical);

        let total_width: usize =
            widths.iter().sum::<usize>() + widths.len() - 1;
        let padded = format!("{:<width$}", continuation, width = total_width);

        format!("{}{}{}", vertical, padded, vertical)
    }
}

/// Align text within a field of given width.
fn align_text(text: &str, width: usize, alignment: Alignment) -> String {
    // Strip ANSI codes for length calculation, but preserve them in output
    let visible_text = strip_ansi(text);
    let text_len = visible_text.chars().count();

    if text_len >= width {
        // Truncate if too long
        return visible_text.chars().take(width).collect();
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

/// Strip ANSI escape codes from a string.
fn strip_ansi(s: &str) -> String {
    let mut result = String::new();
    let mut in_escape = false;

    for ch in s.chars() {
        if ch == '\x1b' {
            in_escape = true;
        } else if in_escape {
            if ch == 'm' {
                in_escape = false;
            }
        } else {
            result.push(ch);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_table() {
        let table = TableBuilder::new()
            .border_style(BorderStyle::Ascii)
            .column("A", 3, Alignment::Left)
            .column("B", 3, Alignment::Right)
            .row(vec!["1".to_string(), "2".to_string()])
            .build();

        assert!(table.contains('+'));
        assert!(table.contains('|'));
        assert!(table.contains("A"));
        assert!(table.contains("B"));
        assert!(table.contains("1"));
        assert!(table.contains("2"));
    }

    #[test]
    fn test_alignment() {
        assert_eq!(align_text("a", 5, Alignment::Left), "a    ");
        assert_eq!(align_text("a", 5, Alignment::Right), "    a");
        assert_eq!(align_text("a", 5, Alignment::Center), "  a  ");
    }

    #[test]
    fn test_border_styles() {
        for style in [
            BorderStyle::Ascii,
            BorderStyle::Standard,
            BorderStyle::Rounded,
            BorderStyle::Heavy,
        ] {
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
                "extra info".to_string(),
            )
            .build();

        assert!(table.contains("main"));
        assert!(table.contains("extra info"));
    }

    #[test]
    fn test_strip_ansi() {
        assert_eq!(strip_ansi("plain text"), "plain text");
        assert_eq!(strip_ansi("\x1b[32mgreen\x1b[m"), "green");
        assert_eq!(strip_ansi("\x1b[1;31mbold red\x1b[0m"), "bold red");
    }
}
