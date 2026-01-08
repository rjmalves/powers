# Epic 1: Core `tabled` Integration

## Summary

Add the `tabled` crate as a dependency and create foundational utilities for table rendering. This epic establishes the infrastructure that renderers will use in Epic 2.

## Scope

### Included

- Add `tabled` dependency to `Cargo.toml` with correct feature flags
- Create style mapping utilities (`BorderStyle` → `tabled::Style`)
- Create helper functions for common table patterns
- Verify basic compilation and functionality

### Excluded

- Actual renderer migration (Epic 2)
- Removal of old code (Epic 3)
- Changes to `DisplayRenderer` trait

## Dependencies

- **Requires**: None (first epic)
- **Enables**: Epic 2 (Renderer Migration)

## Acceptance Criteria

- [x] `cargo build` succeeds with `tabled` dependency
- [x] `cargo test` passes (no changes to existing behavior yet)
- [x] Style mapping function converts `BorderStyle` → `tabled::Style`
- [x] Basic table can be constructed and rendered to string
- [x] ANSI feature is enabled and working

## Technical Approach

### Dependency Configuration

```toml
[dependencies]
tabled = { version = "0.20", default-features = false, features = ["std", "ansi"] }
```

**Features:**
- `std`: Required for `Table`, `Builder`, etc.
- `ansi`: Required for correct width calculation with colored text
- Exclude `derive`, `macros`, `assert` to minimize footprint

### Style Mapping

Create a function to convert our `BorderStyle` enum to `tabled::Style`:

```rust
use tabled::settings::Style;

pub fn border_style_to_tabled(style: BorderStyle) -> Style {
    match style {
        BorderStyle::None => Style::empty(),
        BorderStyle::Ascii => Style::ascii(),
        BorderStyle::Standard => Style::modern(),
        BorderStyle::Rounded => Style::rounded(),
        BorderStyle::Heavy => Style::extended(),
    }
}
```

### Helper Module Location

Create utilities in `src/display/components/table_utils.rs` (new file) to avoid touching existing `table.rs` until Epic 3.

## Estimated Effort

**Sprints**: 0.5 (2-3 days)
**Story Points**: 5

## Sprint Breakdown

### Sprint 1: Core Setup

| Ticket | Title | Points |
|--------|-------|--------|
| TABLED-001 | Add tabled dependency to Cargo.toml | 1 |
| TABLED-002 | Create style mapping utilities | 2 |
| TABLED-003 | Add basic table builder helpers | 2 |

---

**Epic Status**: ✅ COMPLETED  
**Completed**: 2026-01-07

## Implementation Summary

Successfully completed all sprint 1 tickets:

### Deliverables

1. **TABLED-001**: Added `tabled` v0.20 dependency with `std` and `ansi` features
2. **TABLED-002**: Created `apply_border_style()` utility function with 5 comprehensive tests
3. **TABLED-003**: Implemented `TrainingTableBuilder` fluent API with 10 comprehensive tests

### New File

- `src/display/components/tabled_utils.rs` (220 lines)
  - `apply_border_style()` - Maps `BorderStyle` to tabled styles
  - `TrainingTableBuilder` - Fluent API for table construction
  - 15 unit tests covering all border styles and edge cases

### Test Results

- Total tests: 764 (15 new)
- All tests passing ✅
- `cargo clippy` clean ✅
- `cargo fmt` applied ✅

### Next Steps

Ready to proceed with Epic 2: Renderer Migration
