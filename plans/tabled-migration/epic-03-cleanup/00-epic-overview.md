# Epic 3: Cleanup and Removal

## Summary

Remove deprecated manual table code, clean up unused imports, and update documentation. This epic completes the migration by eliminating all traces of the old implementation.

## Scope

### Included

- Remove `src/display/components/table.rs` entirely
- Remove `table` from `components/mod.rs` exports
- Clean up unused imports in renderers
- Archive/update `TABLE_ALIGNMENT_ANALYSIS.md`
- Update `TABLED_MIGRATION_ANALYSIS.md` to reflect completion
- Final verification of all tests and functionality

### Excluded

- New features
- Performance optimization
- Additional refactoring beyond cleanup

## Dependencies

- **Requires**: Epic 2 (Renderer Migration) complete
- **Enables**: Future display enhancements with clean foundation

## Acceptance Criteria

- [ ] `src/display/components/table.rs` deleted
- [ ] No references to `components::table::*` in any renderer
- [ ] All dead code removed
- [ ] `cargo build` succeeds with no warnings
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean
- [ ] Documentation updated

## Technical Approach

### File Removal

1. Remove `src/display/components/table.rs`
2. Update `src/display/components/mod.rs` to remove `pub mod table;`
3. Search codebase for any remaining references

### Import Cleanup

In each renderer, remove:
```rust
// Remove these
use crate::display::components::table::BorderStyle;
use crate::display::components::table::BorderChars;
use crate::display::components::table::TableBuilder;
```

### Documentation Updates

1. **Archive `TABLE_ALIGNMENT_ANALYSIS.md`**:
   - Move to `docs/archive/` or delete
   - The alignment issue is resolved

2. **Update `TABLED_MIGRATION_ANALYSIS.md`**:
   - Mark as complete
   - Document final outcome

3. **Update module docs**:
   - Remove references to manual table construction
   - Add note about `tabled` usage

## Estimated Effort

**Sprints**: 0.25 (1 day)
**Story Points**: 3

## Sprint Breakdown

### Sprint 1: Final Cleanup

| Ticket | Title | Points |
|--------|-------|--------|
| TABLED-009 | Remove table.rs and update module exports | 1 |
| TABLED-010 | Clean up unused imports in renderers | 1 |
| TABLED-011 | Update documentation | 1 |

---

**Epic Status**: Ready for implementation (blocked by Epic 2)
