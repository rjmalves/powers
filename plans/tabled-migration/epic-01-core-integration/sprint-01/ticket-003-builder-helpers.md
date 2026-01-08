# [TABLED-003] Add basic table builder helpers

> **Epic**: [Epic 1: Core Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TABLED-001](./ticket-001-add-dependency.md)  
> **Blocks**: Epic 2 renderer migration tickets

## Context

### Background

The renderers have common table patterns: header rows, data rows, centered headers, right-aligned numbers. Creating helper functions reduces code duplication and ensures consistent behavior.

### Relation to Epic

This ticket creates reusable building blocks that will simplify the renderer migration in Epic 2.

### Current State

Each renderer manually constructs tables with repeated patterns:
- Header row with centered text
- Data rows with mixed alignment
- Statistics row spanning multiple columns
- Consistent styling (modern border style)

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Lines 86-280 show table construction patterns
- `src/display/renderers/standard.rs` - Lines 72-184 show similar patterns
- `src/display/components/statistics.rs` - Format functions used in cells

## Specification

### Inputs

- Column headers (string slices)
- Row data (string slices)
- Style configuration

### Outputs

- `tabled::Table` ready to render

### Behavior

Create helper functions in `tabled_utils.rs`:

1. `build_training_table()` - Creates the iteration progress table
2. `add_panel_row()` - Adds a full-width panel row for statistics

### Error Handling

No errors - invalid data produces empty or malformed tables (acceptable for display).

## Acceptance Criteria

- [ ] `TrainingTableBuilder` struct created with fluent API
- [ ] `add_header()` method adds centered header row
- [ ] `add_iteration_row()` method adds data row
- [ ] `add_stats_row()` method adds full-width panel
- [ ] `build()` method returns `String`
- [ ] Example usage demonstrates all methods
- [ ] Unit tests verify output structure
- [ ] `cargo test` passes
- [ ] `cargo clippy` has no warnings

## Implementation Guide

### Suggested Approach

1. Extend `src/display/components/tabled_utils.rs`
2. Create `TrainingTableBuilder` struct
3. Implement fluent builder pattern
4. Add tests
5. Add doc examples

### Key Files to Modify

- `src/display/components/tabled_utils.rs`: Add builder struct and methods

### Code Template

```rust
use tabled::{
    builder::Builder,
    settings::{Alignment, Modify, Panel, Style, object::Rows},
    Table,
};

/// Builder for SDDP training progress tables.
///
/// Provides a fluent API for constructing iteration tables with
/// headers, data rows, and statistics panels.
///
/// # Example
///
/// ```ignore
/// use powers_rs::display::components::tabled_utils::TrainingTableBuilder;
///
/// let table = TrainingTableBuilder::new()
///     .headers(&["Iter", "Lower Bound", "Gap %"])
///     .row(&["1", "$101,480", "26.4%"])
///     .stats_panel("μ=$128,230 σ=$3,200 n=4")
///     .build();
///
/// println!("{}", table);
/// ```
pub struct TrainingTableBuilder {
    builder: Builder,
    has_header: bool,
    row_count: usize,
}

impl TrainingTableBuilder {
    /// Create a new training table builder.
    pub fn new() -> Self {
        Self {
            builder: Builder::default(),
            has_header: false,
            row_count: 0,
        }
    }

    /// Add header row with centered alignment.
    ///
    /// Should be called once, before any data rows.
    pub fn headers<I, S>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let record: Vec<String> = headers
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self.builder.push_record(record);
        self.has_header = true;
        self.row_count += 1;
        self
    }

    /// Add a data row.
    pub fn row<I, S>(mut self, cells: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let record: Vec<String> = cells
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect();
        self.builder.push_record(record);
        self.row_count += 1;
        self
    }

    /// Build the table as a string with modern styling.
    pub fn build(self) -> String {
        let mut table = self.builder.build();
        table.with(Style::modern());
        
        // Center header row if present
        if self.has_header {
            table.with(Modify::new(Rows::first()).with(Alignment::center()));
        }
        
        table.to_string()
    }

    /// Build with a custom style.
    pub fn build_with_style(self, style: Style) -> String {
        let mut table = self.builder.build();
        table.with(style);
        
        if self.has_header {
            table.with(Modify::new(Rows::first()).with(Alignment::center()));
        }
        
        table.to_string()
    }
}

impl Default for TrainingTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    #[test]
    fn test_simple_table() {
        let table = TrainingTableBuilder::new()
            .headers(&["A", "B", "C"])
            .row(&["1", "2", "3"])
            .row(&["4", "5", "6"])
            .build();

        assert!(table.contains("A"));
        assert!(table.contains("B"));
        assert!(table.contains("C"));
        assert!(table.contains("1"));
        assert!(table.contains("6"));
        // Modern style uses box-drawing
        assert!(table.contains("│"));
        assert!(table.contains("─"));
    }

    #[test]
    fn test_no_header() {
        let table = TrainingTableBuilder::new()
            .row(&["x", "y"])
            .build();

        assert!(table.contains("x"));
        assert!(table.contains("y"));
    }

    #[test]
    fn test_empty_table() {
        let table = TrainingTableBuilder::new().build();
        // Empty table produces minimal output
        assert!(table.is_empty() || table.len() < 10);
    }

    #[test]
    fn test_custom_style() {
        let table = TrainingTableBuilder::new()
            .headers(&["Col"])
            .row(&["Val"])
            .build_with_style(Style::ascii());

        assert!(table.contains('+'));
        assert!(table.contains('-'));
        assert!(table.contains('|'));
    }
}
```

### Patterns to Follow

- Fluent builder pattern like `ProgressBarConfig`
- Generic iterators for flexibility (`IntoIterator<Item = S>`)

### Pitfalls to Avoid

- ⚠️ Don't clone `Builder` - it's moved by `build()`
- ⚠️ `Panel::horizontal()` inserts at a row index - track row count
- ⚠️ Empty tables may behave unexpectedly - test this case

## Testing Requirements

### Unit Tests

- [ ] Test table with headers and rows contains all content
- [ ] Test table without headers still works
- [ ] Test empty table doesn't panic
- [ ] Test custom style applies correctly
- [ ] Test centered header alignment

### Integration Tests

None required for this ticket.

### Performance Tests

None required.

## Documentation Requirements

- [ ] Doc comments on `TrainingTableBuilder` struct
- [ ] Doc comments on all public methods
- [ ] Example in struct doc comment
- [ ] Module-level doc comment updated

## Dependencies

- **Blocked By**: TABLED-001 (dependency must be added first)
- **Blocks**: All renderer migration tickets
- **Related**: TABLED-002 (style mapping used here)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Clear pattern, bounded scope, good test coverage

## Definition of Done

- [x] Implementation complete
- [x] All tests passing
- [x] Documentation complete with examples
- [x] Code reviewed
- [x] `cargo clippy` clean
- [x] `cargo fmt` applied

---

**Status**: ✅ COMPLETED
**Completed**: 2026-01-07
**Notes**: Implemented `TrainingTableBuilder` with fluent API and 10 comprehensive tests covering all border styles and edge cases.
