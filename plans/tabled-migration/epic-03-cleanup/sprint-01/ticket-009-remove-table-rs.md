# [TABLED-009] Remove table.rs and update module exports

> **Epic**: [Epic 3: Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 2 complete  
> **Blocks**: [TABLED-010](./ticket-010-cleanup-imports.md)

## Context

### Background

With the migration to `tabled` complete, the manual table implementation in `src/display/components/table.rs` is no longer used. This ticket removes it entirely.

### Relation to Epic

This is the primary cleanup action that removes the deprecated code.

### Current State

- `table.rs` contains 554 lines of manual table code
- Exported from `components/mod.rs`
- Previously imported by `advanced.rs` and `standard.rs`

## Files to Read Before Starting

- `src/display/components/table.rs` - File to be deleted
- `src/display/components/mod.rs` - Module exports to update

## Specification

### Inputs

None - this is a deletion task.

### Outputs

- `table.rs` deleted
- `mod.rs` updated
- Build still succeeds

### Behavior

1. Delete `src/display/components/table.rs`
2. Remove `pub mod table;` from `src/display/components/mod.rs`
3. Verify no compilation errors

### Error Handling

Compilation errors indicate missed migration - fix in renderers first.

## Acceptance Criteria

- [ ] `src/display/components/table.rs` deleted
- [ ] `pub mod table;` removed from `mod.rs`
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] No references to `components::table` anywhere

## Implementation Guide

### Suggested Approach

1. **Verify no usage** (before deletion):
   ```bash
   grep -r "components::table" src/
   grep -r "use.*table::" src/display/
   grep -r "BorderStyle" src/display/renderers/
   grep -r "BorderChars" src/display/
   grep -r "TableBuilder" src/display/
   ```

2. **Delete the file**:
   ```bash
   rm src/display/components/table.rs
   ```

3. **Update mod.rs**:
   ```rust
   // Before
   pub mod color;
   pub mod indicators;
   pub mod progress;
   pub mod statistics;
   pub mod table;        // REMOVE THIS LINE
   pub mod tabled_utils; // Keep this (new)
   
   // After
   pub mod color;
   pub mod indicators;
   pub mod progress;
   pub mod statistics;
   pub mod tabled_utils;
   ```

4. **Build and test**:
   ```bash
   cargo build
   cargo test
   ```

5. **Handle any failures**:
   - If compilation fails, some code still references `table.rs`
   - Fix the reference, then retry

### Key Files to Modify

- `src/display/components/table.rs` - DELETE
- `src/display/components/mod.rs` - Remove export

### Pitfalls to Avoid

- ⚠️ Don't delete before verifying no usage
- ⚠️ Don't forget to update mod.rs
- ⚠️ Check for any `#[cfg(test)]` usage of table.rs

## Testing Requirements

### Unit Tests

- [ ] `cargo test` passes after deletion
- [ ] No tests were in `table.rs` that are still needed

### Integration Tests

- [ ] `cargo test --all` passes

## Documentation Requirements

- [ ] No documentation references `table.rs`

## Dependencies

- **Blocked By**: Epic 2 complete (all renderers migrated)
- **Blocks**: TABLED-010, TABLED-011
- **Related**: None

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Simple deletion, clear verification steps

## Definition of Done

- [ ] File deleted
- [ ] Module export removed
- [ ] Build succeeds
- [ ] Tests pass
