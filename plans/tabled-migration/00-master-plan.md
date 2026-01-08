# Master Plan: Table Rendering Migration to `tabled`

## Executive Summary

Migrate POWE.RS terminal display system from manual table rendering with hardcoded column widths to the `tabled` crate, eliminating persistent alignment issues, reducing code complexity by ~400 lines, and establishing a maintainable foundation for future display enhancements.

## Goals & Non-Goals

### Goals

1. **Eliminate alignment issues**: Replace fragile manual width calculations with automatic width handling via `tabled`
2. **Reduce code complexity**: Remove ~400 lines of manual table construction code
3. **Improve maintainability**: Single source of truth for table layout, no distributed width definitions
4. **Preserve all functionality**: All display profiles continue to work identically
5. **Handle ANSI correctly**: Use `tabled`'s `ansi` feature to fix color-related width miscalculations
6. **Clean architecture**: Remove all deprecated manual table code after migration

### Non-Goals (Explicit Scope Exclusions)

- **New display features**: No new columns, metrics, or visual elements
- **Performance optimization**: Table rendering is not a hot path; no benchmarking required
- **Display profile changes**: Minimal, Standard, Advanced, Automation profiles remain as-is
- **API changes**: `DisplayRenderer` trait interface remains unchanged
- **Progress bar changes**: `ProgressBar` component (used by Minimal) is unaffected

## Architecture Overview

### Current State

```
src/display/
├── components/
│   ├── table.rs          # Manual table builder (554 lines) - TO BE REMOVED
│   ├── color.rs          # Color utilities - KEEP
│   ├── indicators.rs     # Trend arrows - KEEP
│   ├── progress.rs       # Progress bar - KEEP
│   └── statistics.rs     # Number formatting - KEEP
├── renderers/
│   ├── advanced.rs       # Uses manual table (930 lines)
│   ├── standard.rs       # Uses manual table (660 lines)
│   ├── minimal.rs        # Uses ProgressBar only (293 lines)
│   └── automation.rs     # JSON output, no tables
└── ...
```

**Problems:**
- `col_widths` arrays defined in 4+ locations per renderer
- Format widths must match column widths (error-prone)
- No ANSI code stripping for width calculation
- Statistics continuation row requires fragile merged-column formula

### Target State

```
src/display/
├── components/
│   ├── table.rs          # REMOVED (or minimal wrapper if needed)
│   ├── color.rs          # Color utilities - UNCHANGED
│   ├── indicators.rs     # Trend arrows - UNCHANGED
│   ├── progress.rs       # Progress bar - UNCHANGED
│   └── statistics.rs     # Number formatting - UNCHANGED
├── renderers/
│   ├── advanced.rs       # Uses tabled::Builder (~600 lines, -35%)
│   ├── standard.rs       # Uses tabled::Builder (~450 lines, -32%)
│   ├── minimal.rs        # UNCHANGED (no tables)
│   └── automation.rs     # UNCHANGED (JSON output)
└── ...
```

**Benefits:**
- Single table construction point per method
- Automatic width calculation and ANSI handling
- No distributed width definitions
- Simpler merged row handling via `Panel`

### Key Design Decisions

1. **Use `tabled::Builder` pattern**: We construct tables dynamically (not from structs), so `Builder` is the right approach over `#[derive(Tabled)]`

2. **Use `Style::modern()` for Standard border style**: Maps to current `BorderStyle::Standard` box-drawing characters

3. **Keep `BorderStyle` enum for compatibility**: Renderers can still accept `BorderStyle` config and map to `tabled::Style`

4. **Use `Panel::horizontal()` for statistics row**: Replaces manual merged-column calculation

5. **Feature configuration**: Use `tabled = { version = "0.20", default-features = false, features = ["std", "ansi"] }` for minimal footprint with ANSI support

6. **Progressive table building**: Buffer rows for each iteration, rebuild table each time (simple, correct, fast enough for display)

## Technical Approach

### Core Abstractions

#### `tabled::Builder` Usage Pattern

```rust
use tabled::{
    builder::Builder,
    settings::{Style, Alignment, Panel, Modify, object::Rows},
};

fn render_iteration_table(&self, ctx: &DisplayContext) -> String {
    let mut builder = Builder::default();
    
    // Headers (first iteration only)
    if ctx.iteration == 1 {
        builder.push_record(["Iter", "Lower Bound ($)", "Gap %", ...]);
    }
    
    // Data row
    builder.push_record([
        format!("{}", ctx.iteration),
        format_cost(ctx.lower_bound, true),
        format_gap(ctx.gap_percent),
        // ...
    ]);
    
    let mut table = builder.build();
    table
        .with(Style::modern())
        .with(Modify::new(Rows::first()).with(Alignment::center()));
    
    table.to_string()
}
```

#### Style Mapping

| Current `BorderStyle` | `tabled` Equivalent |
|----------------------|---------------------|
| `Ascii` | `Style::ascii()` |
| `Standard` | `Style::modern()` |
| `Rounded` | `Style::rounded()` |
| `Heavy` | `Style::extended()` |
| `None` | `Style::empty()` |

#### Statistics Row via Panel

```rust
// Instead of manual merged-column width calculation:
let stats_text = format!("{}    {}", change_pct, format_cost_stats(...));
table.with(Panel::horizontal(row_index, stats_text));
```

### Data Flow

1. `DisplayContext` populated by SDDP algorithm
2. `DisplayManager::iteration()` calls renderer
3. Renderer constructs `tabled::Builder` with data
4. `Builder::build()` produces `Table`
5. `Table::to_string()` renders with automatic width handling
6. String output to terminal

### Parallelism Strategy

N/A - Display rendering is single-threaded and not performance-critical.

### Performance Strategy

- **No benchmarking required**: Table rendering is ~0.01% of SDDP runtime
- **Acceptable overhead**: `tabled` builds tables in microseconds
- **No buffering needed**: Progressive display rebuilds are fast enough

## Phases & Milestones

| Phase | Description | Duration | Milestone |
|-------|-------------|----------|-----------|
| 1 | Core Integration | 2 days | `tabled` dependency added, basic table renders |
| 2 | AdvancedRenderer Migration | 2 days | Advanced profile uses `tabled`, tests pass |
| 3 | StandardRenderer Migration | 1 day | Standard profile uses `tabled`, tests pass |
| 4 | Cleanup & Removal | 1 day | Manual `table.rs` removed, docs updated |

**Total Estimated Duration**: 6 days (1.2 sprints)

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Visual regression in terminal | Medium | Medium | Compare screenshots before/after |
| `tabled` API changes | Low | Low | Pin to specific version 0.20 |
| ANSI handling differs from expectation | Low | Medium | Test with actual colored output |
| Performance regression | Very Low | Very Low | Display is not a hot path |
| Test failures due to exact string matching | High | Low | Update tests to match new format |

## Success Metrics

- [x] All existing tests pass (with updated expected strings)
- [ ] Manual terminal testing shows correct alignment
- [ ] ~400 lines of code removed from `table.rs`
- [ ] ~200 lines reduced from renderers
- [x] No new runtime dependencies beyond `tabled` + `papergrid`
- [x] `cargo clippy` and `cargo fmt` pass
- [ ] Documentation updated

## File Change Summary

### Files to Modify

| File | Change Type | Estimated Lines |
|------|-------------|-----------------|
| `Cargo.toml` | Add dependency | +1 |
| `src/display/renderers/advanced.rs` | Major refactor | -300 |
| `src/display/renderers/standard.rs` | Major refactor | -200 |
| `src/display/components/mod.rs` | Remove table export | -1 |
| `docs/TABLE_ALIGNMENT_ANALYSIS.md` | Archive/remove | -469 |
| `docs/TABLED_MIGRATION_ANALYSIS.md` | Archive/update | Update |

### Files to Remove

| File | Reason |
|------|--------|
| `src/display/components/table.rs` | Replaced by `tabled` |

### Files Unchanged

| File | Reason |
|------|--------|
| `src/display/renderers/minimal.rs` | Uses `ProgressBar`, no tables |
| `src/display/renderers/automation.rs` | Uses JSON, no tables |
| `src/display/components/color.rs` | Independent utility |
| `src/display/components/indicators.rs` | Independent utility |
| `src/display/components/progress.rs` | Independent utility |
| `src/display/components/statistics.rs` | Independent utility |
| `src/display/renderer.rs` | Trait unchanged |
| `src/display/context.rs` | Data types unchanged |
| `src/display/config.rs` | Config unchanged |

---

**Plan Version**: 1.0  
**Created**: 2026-01-07  
**Status**: Epic 1 Completed ✅ (2026-01-07) | Epic 2 Ready

## Epic Status

### Epic 1: Core Integration - ✅ COMPLETED (2026-01-07)

All tickets completed:
- TABLED-001: Added `tabled` v0.20 dependency ✅
- TABLED-002: Created style mapping utilities ✅
- TABLED-003: Implemented `TrainingTableBuilder` ✅

**Deliverables:**
- New file: `src/display/components/tabled_utils.rs` (220 lines)
- 15 new unit tests (all passing)
- Total test count: 764

### Epic 2: Renderer Migration - READY

Ready to begin migration of AdvancedRenderer and StandardRenderer.

### Epic 3: Cleanup - BLOCKED

Awaiting completion of Epic 2.
