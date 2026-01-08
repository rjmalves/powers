# [TABLED-001] Add tabled dependency to Cargo.toml

> **Epic**: [Epic 1: Core Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TABLED-002](./ticket-002-style-mapping.md), [TABLED-003](./ticket-003-builder-helpers.md)

## Context

### Background

The POWE.RS display system currently uses manual table rendering with hardcoded column widths, causing persistent alignment issues (documented in `docs/TABLE_ALIGNMENT_ANALYSIS.md`). We are migrating to the `tabled` crate to solve these issues.

### Relation to Epic

This is the first step in Epic 1, establishing the crate dependency that all subsequent work relies on.

### Current State

No `tabled` dependency exists. Manual table code is in `src/display/components/table.rs`.

## Specification

### Inputs

None - this is a configuration change.

### Outputs

- `Cargo.toml` modified with `tabled` dependency
- `cargo build` succeeds with new dependency

### Behavior

After this ticket, developers can `use tabled::*` in any module.

### Error Handling

N/A - compilation errors indicate incorrect configuration.

## Acceptance Criteria

- [ ] `tabled` added to `[dependencies]` in `Cargo.toml`
- [ ] Version pinned to `0.20`
- [ ] `default-features = false` to minimize footprint
- [ ] Features `["std", "ansi"]` enabled
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes (no behavior changes)
- [ ] Verify `tabled` appears in `Cargo.lock`

## Implementation Guide

### Suggested Approach

1. Open `Cargo.toml`
2. Add dependency line in `[dependencies]` section (alphabetical order)
3. Run `cargo build` to fetch and compile dependency
4. Run `cargo test` to ensure no regressions

### Key Files to Modify

- `Cargo.toml`: Add dependency (line ~15-35 in dependencies section)

### Exact Change

Add this line to `[dependencies]` section:

```toml
tabled = { version = "0.20", default-features = false, features = ["std", "ansi"] }
```

### Patterns to Follow

- Other dependencies use `version = "X.Y"` format
- Feature flags follow the pattern `features = ["feature1", "feature2"]`

### Pitfalls to Avoid

- ⚠️ Don't use `version = "*"` - always pin version
- ⚠️ Don't enable `derive` feature (we don't use it, adds proc-macro compile time)
- ⚠️ Don't enable `macros` feature (we don't use `row!`/`col!` macros)

## Testing Requirements

### Unit Tests

None required - this is a configuration change.

### Integration Tests

- [ ] Run `cargo test` to verify no regressions

### Performance Tests

None required.

### Validation Tests

- [ ] `cargo build --all-features` succeeds
- [ ] `cargo doc --no-deps` succeeds

## Documentation Requirements

- [ ] No documentation changes required for this ticket

## Dependencies

- **Blocked By**: None
- **Blocks**: TABLED-002, TABLED-003, all Epic 2 tickets
- **Related**: None

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Simple Cargo.toml edit, minimal risk

## Definition of Done

- [x] Implementation complete
- [x] `cargo build` succeeds
- [x] `cargo test` passes
- [x] `Cargo.lock` updated

---

**Status**: ✅ COMPLETED
**Completed**: 2026-01-07
