# Sprint 1: Full Migration

## Goals

- Migrate all table rendering in `AdvancedRenderer` to use `tabled`
- Migrate all table rendering in `StandardRenderer` to use `tabled`
- Update all affected tests
- Verify visual correctness in terminal

## Tickets

| ID | Title | Points | Assignable | Dependencies |
|----|-------|--------|------------|--------------|
| TABLED-004 | Migrate AdvancedRenderer table header | 2 | Yes | Epic 1 complete |
| TABLED-005 | Migrate AdvancedRenderer iteration rows | 3 | Yes | TABLED-004 |
| TABLED-006 | Migrate AdvancedRenderer training summary | 2 | Yes | TABLED-005 |
| TABLED-007 | Migrate StandardRenderer tables | 3 | Yes | Epic 1 complete |
| TABLED-008 | Update renderer tests for new output format | 3 | Yes | TABLED-006, TABLED-007 |

## Dependencies

- **From Previous Sprint**: Epic 1 (tabled dependency and utilities)
- **To Next Sprint**: Enables Epic 3 (cleanup)

## Risks

- **Test string matching**: Tests may fail due to minor whitespace/alignment differences
  - **Mitigation**: Update tests to check content presence, not exact format
  
- **Visual regression**: Tables may look different in terminal
  - **Mitigation**: Manual visual comparison before/after

- **Statistics row complexity**: Panel may not work exactly like merged columns
  - **Mitigation**: Prototype in TABLED-005 before committing

## Definition of Done

- [ ] All tickets complete
- [ ] `cargo test` passes
- [ ] `cargo clippy` has no warnings
- [ ] Manual terminal test shows correct alignment
- [ ] No imports from `components::table::BorderChars` in renderers
- [ ] Code reviewed and merged
