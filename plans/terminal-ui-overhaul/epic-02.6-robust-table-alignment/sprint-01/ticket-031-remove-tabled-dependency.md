# [T-031] Remove tabled dependency and revert related code

> **Epic**: [Epic 2.6: Robust Table Alignment](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [T-032](./ticket-032-create-table-format-utils.md)

## Context

### Background

The `tabled` crate was added as part of an attempted migration to use a table rendering library. However, this migration was abandoned because `tabled` builds complete tables while our architecture requires progressive rendering (print header once, then rows incrementally).

The `tabled` dependency and related code must be removed to clean up the codebase before implementing the new robust table formatting approach.

### Current State

- `Cargo.toml` contains: `tabled = { version = "0.20", default-features = false, features = ["std", "ansi"] }`
- `src/display/components/tabled_utils.rs` exists with utilities that will never be used
- `src/display/components/mod.rs` exports the `tabled_utils` module
- Various renderer files may have minor changes that should be reviewed

### Files to Read Before Starting

- `Cargo.toml` - Dependency to remove
- `src/display/components/tabled_utils.rs` - File to delete
- `src/display/components/mod.rs` - Export to remove
- `git diff src/display/` - Review pending changes to revert if tabled-related

## Specification

### Tasks

1. **Remove `tabled` from Cargo.toml**
   - Delete the `tabled = { ... }` line from `[dependencies]`

2. **Delete tabled_utils.rs**
   - Remove `src/display/components/tabled_utils.rs` entirely

3. **Update mod.rs**
   - Remove `pub mod tabled_utils;` from `src/display/components/mod.rs`

4. **Review and revert tabled-related changes**
   - Check `git diff src/display/renderers/` for any tabled-related imports or changes
   - Revert any changes that were specifically for tabled migration

5. **Run Cargo.lock update**
   - Run `cargo update` to clean up the lock file

6. **Verify build**
   - Run `cargo build -j 1` to ensure no broken imports

## Acceptance Criteria

- [ ] `tabled` does not appear in `Cargo.toml`
- [ ] `tabled` does not appear in `Cargo.lock`
- [ ] `src/display/components/tabled_utils.rs` does not exist
- [ ] No file in `src/` contains `use tabled` or `mod tabled`
- [ ] `cargo build -j 1` succeeds
- [ ] `cargo test -j 1 -- --test-threads=1` passes

## Implementation Guide

### Step-by-Step

1. Remove dependency from Cargo.toml:
   ```bash
   # Edit Cargo.toml to remove the tabled line
   ```

2. Delete the tabled_utils file:
   ```bash
   rm src/display/components/tabled_utils.rs
   ```

3. Update mod.rs:
   ```rust
   // Remove this line from src/display/components/mod.rs:
   // pub mod tabled_utils;
   ```

4. Check for any tabled imports in renderers:
   ```bash
   grep -r "tabled" src/
   ```

5. Clean up Cargo.lock:
   ```bash
   cargo update
   ```

6. Verify:
   ```bash
   cargo build -j 1
   cargo test -j 1 -- --test-threads=1
   ```

### Pitfalls to Avoid

- ⚠️ Don't revert UI fixes from Epic 02.5 (only tabled-specific changes)
- ⚠️ Keep the existing `table.rs` with `BorderStyle` - we'll build on it
- ⚠️ Use `-j 1` to avoid RAM issues

## Testing Requirements

### Verification Tests

- [ ] `grep -r "tabled" src/` returns no results
- [ ] `grep "tabled" Cargo.toml` returns no results
- [ ] `grep "tabled" Cargo.lock` returns no results
- [ ] `cargo build -j 1` succeeds
- [ ] `cargo test -j 1 -- --test-threads=1` passes

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward cleanup task with clear steps

## Definition of Done

- [x] All tabled references removed
- [x] Build succeeds
- [x] Tests pass
- [x] No regression in existing functionality

---

**Status**: ✅ COMPLETE
