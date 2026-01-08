# [TABLED-010] Clean up unused imports in renderers

> **Epic**: [Epic 3: Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [TABLED-009](./ticket-009-remove-table-rs.md)  
> **Blocks**: [TABLED-011](./ticket-011-update-docs.md)

## Context

### Background

After removing `table.rs`, there may be unused imports or dead code in the renderers. This ticket cleans up any remaining cruft.

### Relation to Epic

This ensures the codebase is clean and warning-free after migration.

### Current State

Renderers may have:
- Unused imports from old table module (now removed)
- Dead helper methods
- Commented-out code
- Unused variables

## Files to Read Before Starting

- `src/display/renderers/advanced.rs` - Check imports section
- `src/display/renderers/standard.rs` - Check imports section
- `src/display/components/tabled_utils.rs` - Verify exports

## Specification

### Inputs

- Compiler warnings
- Clippy warnings

### Outputs

- Clean, warning-free code

### Behavior

1. Run `cargo clippy` to find all warnings
2. Fix each warning
3. Run `cargo fmt` to ensure consistent formatting
4. Verify tests still pass

## Acceptance Criteria

- [ ] `cargo build` produces no warnings
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo fmt --check` passes
- [ ] No unused imports in renderers
- [ ] No dead code
- [ ] No commented-out code blocks

## Implementation Guide

### Suggested Approach

1. **Find all warnings**:
   ```bash
   cargo build 2>&1 | grep -i warning
   cargo clippy -- -D warnings 2>&1
   ```

2. **Fix unused imports**:
   ```rust
   // Remove lines like:
   use crate::display::components::table::BorderStyle;  // DELETE
   use crate::display::components::table::BorderChars;  // DELETE
   ```

3. **Remove dead methods**:
   If any methods from the old implementation remain unused (like standalone helper functions), remove them.

4. **Clean up commented code**:
   Remove any `// TODO: migrate` or `// OLD:` comments.

5. **Format code**:
   ```bash
   cargo fmt
   ```

6. **Verify**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   ```

### Common Cleanups

#### Advanced Renderer

```rust
// REMOVE these imports if present:
use crate::display::components::table::BorderStyle;
use crate::display::components::table::BorderChars;
use crate::display::components::table::TableBuilder;

// KEEP these:
use tabled::{builder::Builder, settings::{Style, Alignment, ...}};
```

#### Standard Renderer

Same pattern as Advanced.

### Key Files to Modify

- `src/display/renderers/advanced.rs` - Clean imports
- `src/display/renderers/standard.rs` - Clean imports
- Any other file with warnings

### Pitfalls to Avoid

- ⚠️ Don't remove imports that are still used
- ⚠️ Run tests after each cleanup
- ⚠️ `cargo clippy` may find additional issues

## Testing Requirements

### Unit Tests

- [ ] `cargo test` passes
- [ ] No test uses removed code

### Integration Tests

- [ ] `cargo test --all` passes

### Static Analysis

- [ ] `cargo clippy -- -D warnings` passes

## Documentation Requirements

- [ ] Remove any outdated doc comments referencing old approach

## Dependencies

- **Blocked By**: TABLED-009
- **Blocks**: TABLED-011
- **Related**: None

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Mechanical cleanup, tooling assistance

## Definition of Done

- [ ] Zero compiler warnings
- [ ] Zero clippy warnings
- [ ] Code formatted
- [ ] Tests pass
