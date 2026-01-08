# ⚠️ ARCHIVED: Tabled Migration Plan

**Status**: ABANDONED (2026-01-07)

## Why This Migration Was Abandoned

The migration to the `tabled` crate was abandoned due to a fundamental architectural incompatibility:

### The Problem

1. **`tabled` builds complete tables**: The library constructs entire tables at once, producing a complete output string.

2. **POWE.RS requires progressive rendering**: The current architecture prints the table header once (on iteration 1), then individual data rows progressively as each iteration completes.

3. **These are incompatible**: To use `tabled`, we would need to either:
   - Buffer all rows and rebuild the entire table each iteration (changes output behavior)
   - Use `tabled` only for headers and manually render rows (defeats the purpose)

### The Decision

After analysis, we decided to:

1. **Keep the manual table approach** - it supports progressive rendering naturally
2. **Fix the root cause** - Rust's `format!` expansion behavior
3. **Create robust utilities** - with explicit truncation and ANSI-aware width calculation
4. **Remove `tabled` dependency** - clean up unused code

## Replacement

The replacement plan is documented in:

**[Epic 2.6: Robust Table Alignment](../terminal-ui-overhaul/epic-02.6-robust-table-alignment/00-epic-overview.md)**

This epic creates `table_format.rs` with:
- Centralized column configuration
- ANSI-aware width calculation
- Explicit truncation (prevents format expansion)
- Comprehensive alignment tests

## Original Plan Files

The original plan files are preserved below for reference but should not be implemented:

- `00-master-plan.md` - Original migration plan
- `epic-01-core-integration/` - Add tabled dependency (was completed, now reverted)
- `epic-02-renderer-migration/` - Migrate renderers (never implemented)
- `epic-03-cleanup/` - Remove old code (never implemented)

## Cleanup Required

When implementing Epic 2.6 (T-031), the following must be done:

1. Remove `tabled` from `Cargo.toml`
2. Delete `src/display/components/tabled_utils.rs`
3. Remove `pub mod tabled_utils;` from `src/display/components/mod.rs`
4. This directory can be deleted after Epic 2.6 is complete
