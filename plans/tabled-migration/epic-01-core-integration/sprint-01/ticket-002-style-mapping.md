# [TABLED-002] Create style mapping utilities

> **Epic**: [Epic 1: Core Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TABLED-001](./ticket-001-add-dependency.md)  
> **Blocks**: Epic 2 renderer migration tickets

## Context

### Background

POWE.RS has a `BorderStyle` enum in `src/display/components/table.rs` that defines table border styles (Ascii, Standard, Rounded, Heavy, None). We need to map these to the equivalent `tabled::settings::Style` types.

### Relation to Epic

This ticket creates the bridge between our existing configuration and `tabled`'s styling system.

### Current State

```rust
// src/display/components/table.rs
pub enum BorderStyle {
    None,
    Ascii,
    Standard,  // ┌──┐, │ (box-drawing)
    Rounded,   // ╭──╮, │
    Heavy,     // ┏━━┓, ┃
}
```

## Files to Read Before Starting

- `src/display/components/table.rs` - Current `BorderStyle` enum (lines 7-21)
- `src/display/renderers/advanced.rs` - How `BorderStyle::Standard` is used (lines 86-142)

## Specification

### Inputs

- `BorderStyle` enum value

### Outputs

- Corresponding `tabled::settings::Style` configuration

### Behavior

Create a new module `src/display/components/tabled_utils.rs` with:

1. A function to convert `BorderStyle` to `tabled::settings::Style`
2. Re-export relevant `tabled` types for convenience

### Error Handling

No errors possible - all enum variants have mappings.

## Acceptance Criteria

- [ ] New file `src/display/components/tabled_utils.rs` created
- [ ] Function `to_tabled_style(BorderStyle) -> impl tabled::settings::TableOption<...>` implemented
- [ ] All 5 `BorderStyle` variants mapped correctly
- [ ] Module exported from `src/display/components/mod.rs`
- [ ] Unit tests verify each mapping produces expected characters
- [ ] `cargo test` passes
- [ ] `cargo clippy` has no warnings

## Implementation Guide

### Suggested Approach

1. Create `src/display/components/tabled_utils.rs`
2. Import `tabled::settings::Style`
3. Import `super::table::BorderStyle`
4. Implement conversion function
5. Add unit tests
6. Export from `mod.rs`

### Key Files to Modify

- `src/display/components/tabled_utils.rs`: New file (create)
- `src/display/components/mod.rs`: Add `pub mod tabled_utils;`

### Code Template

```rust
//! Utilities for integrating with the `tabled` crate.
//!
//! Provides conversion functions and re-exports for table rendering.

use tabled::settings::Style;

use super::table::BorderStyle;

/// Convert our `BorderStyle` to a `tabled` style.
///
/// # Arguments
///
/// * `style` - The border style to convert
///
/// # Returns
///
/// A `tabled::settings::Style` configuration.
///
/// # Examples
///
/// ```ignore
/// use powers_rs::display::components::tabled_utils::to_tabled_style;
/// use powers_rs::display::components::table::BorderStyle;
///
/// let style = to_tabled_style(BorderStyle::Standard);
/// ```
pub fn to_tabled_style(style: BorderStyle) -> Style {
    match style {
        BorderStyle::None => Style::empty(),
        BorderStyle::Ascii => Style::ascii(),
        BorderStyle::Standard => Style::modern(),
        BorderStyle::Rounded => Style::rounded(),
        BorderStyle::Heavy => Style::extended(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tabled::{builder::Builder, Table};

    fn render_simple_table(style: Style) -> String {
        let mut builder = Builder::default();
        builder.push_record(["A", "B"]);
        builder.push_record(["1", "2"]);
        let mut table = builder.build();
        table.with(style);
        table.to_string()
    }

    #[test]
    fn test_ascii_style() {
        let table = render_simple_table(to_tabled_style(BorderStyle::Ascii));
        assert!(table.contains('+'));
        assert!(table.contains('-'));
        assert!(table.contains('|'));
    }

    #[test]
    fn test_standard_style() {
        let table = render_simple_table(to_tabled_style(BorderStyle::Standard));
        assert!(table.contains('┌'));
        assert!(table.contains('│'));
        assert!(table.contains('─'));
    }

    #[test]
    fn test_rounded_style() {
        let table = render_simple_table(to_tabled_style(BorderStyle::Rounded));
        assert!(table.contains('╭'));
        assert!(table.contains('╮'));
    }

    #[test]
    fn test_heavy_style() {
        let table = render_simple_table(to_tabled_style(BorderStyle::Heavy));
        assert!(table.contains('╔'));
        assert!(table.contains('║'));
        assert!(table.contains('═'));
    }

    #[test]
    fn test_none_style() {
        let table = render_simple_table(to_tabled_style(BorderStyle::None));
        // No box-drawing characters
        assert!(!table.contains('│'));
        assert!(!table.contains('─'));
        assert!(!table.contains('+'));
        assert!(!table.contains('|'));
    }
}
```

### Patterns to Follow

- Follow the module pattern in `src/display/components/color.rs`
- Doc comments on public functions

### Pitfalls to Avoid

- ⚠️ `tabled::Style` is generic - you may need to return `impl TableOption<...>` or use `Style::modern()` directly (check API)
- ⚠️ Ensure `BorderStyle::Heavy` maps to `Style::extended()` (not `heavy` which doesn't exist)

## Testing Requirements

### Unit Tests

- [ ] Test `BorderStyle::Ascii` produces `+`, `-`, `|` characters
- [ ] Test `BorderStyle::Standard` produces `┌`, `│`, `─` characters
- [ ] Test `BorderStyle::Rounded` produces `╭`, `╮` characters
- [ ] Test `BorderStyle::Heavy` produces `╔`, `║`, `═` characters
- [ ] Test `BorderStyle::None` produces no border characters

### Integration Tests

None required for this ticket.

### Performance Tests

None required.

## Documentation Requirements

- [ ] Doc comments on `to_tabled_style` function
- [ ] Module-level doc comment explaining purpose

## Dependencies

- **Blocked By**: TABLED-001 (dependency must be added first)
- **Blocks**: TABLED-003, all renderer migration tickets
- **Related**: TABLED-003 (uses this utility)

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple mapping function, clear requirements, straightforward tests

## Definition of Done

- [x] Implementation complete
- [x] All tests passing
- [x] Documentation updated
- [x] Code reviewed
- [x] `cargo clippy` clean
- [x] `cargo fmt` applied

---

**Status**: ✅ COMPLETED
**Completed**: 2026-01-07
**Notes**: Implemented `apply_border_style()` function and 5 comprehensive tests for all border styles.
