# [TICKET-010] Remove Deprecated Code and Update Documentation

**Sprint:** 4  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** High  
**Priority:** P1 - High

## Context

With the migration complete and validated, we can now remove the deprecated unified structures (`lagged_state`, `lag_fixing_constraints`) and all associated migration/validation code. This final cleanup solidifies the refactoring and prevents future confusion about which structures to use.

Additionally, all documentation must be updated to reflect the new explicit separation architecture.

## Acceptance Criteria

- [ ] Given codebase after cleanup, when searching for `lagged_state`, then no references exist except in migration notes or CHANGELOG
- [ ] Given codebase after cleanup, when searching for `lag_fixing_constraints`, then no references exist except in documentation
- [ ] Given updated documentation, when reading about lag variables, then explicit separation is clearly explained
- [ ] Given all tests after cleanup, when running test suite, then 100% pass without deprecated code
- [ ] Given final code review, when checking for migration artifacts, then none remain

## Tasks

### Implementation

- [ ] Remove deprecated fields from `Variables` struct in `src/subproblem.rs`
  - Delete `pub lagged_state: Option<Vec<Vec<usize>>>`
  - Remove any `#[deprecated]` attributes
  
- [ ] Remove deprecated fields from `Constraints` struct
  - Delete `pub lag_fixing_constraints: Option<Vec<Vec<usize>>>`
  - Remove any `#[deprecated]` attributes
  
- [ ] Remove migration validation code
  - Delete validation module from `src/subproblem.rs` (if behind feature flag)
  - Remove `migration_validation` feature from `Cargo.toml`
  - Delete validation calls from subproblem creation
  
- [ ] Remove any migration helper functions
  - Search for functions marked as temporary migration helpers
  - Delete if no longer used
  
- [ ] Clean up imports
  - Remove unused imports related to old structures
  - Run `cargo clippy` to catch any dead code
  
- [ ] Update error messages
  - Remove references to old structures in error messages
  - Update to reference explicit structures
  
- [ ] Remove any commented-out old code
  - Search for `// OLD:` or similar comments
  - Delete if no longer needed for reference

### Testing

- [ ] Run full test suite after removals
  - All tests should pass
  - No warnings about unused code
  
- [ ] Run `cargo clippy` with strict settings
  - No warnings
  - No dead code detected
  
- [ ] Run `cargo doc` and verify documentation builds
  - No broken links
  - No references to removed structures
  
- [ ] Verify examples still compile and run
  - Test all example configurations
  - Ensure output is correct
  
- [ ] Check binary size
  - Compare before/after cleanup
  - Should be slightly smaller without deprecated code

### Documentation

- [ ] Update `README.md`
  - Add section on lag variable architecture
  - Explain explicit separation rationale
  - Update any examples that mention lag variables
  
- [ ] Update `docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md`
  - Add "Implementation Status: COMPLETE" section
  - Document final implementation details
  - Add lessons learned
  
- [ ] Update module-level documentation in affected files
  - `src/subproblem.rs` - explain Variables and Constraints structures
  - `src/state.rs` - explain state extraction with explicit structures
  
- [ ] Update CHANGELOG.md
  - Add entry for architecture refactoring
  - Note bug fix in cut generation
  - Document performance improvements
  - Mark as breaking change if public API affected
  
- [ ] Create migration guide (for external users if any)
  - Document what changed
  - Show before/after code examples
  - Explain how to update code using old API
  
- [ ] Update inline documentation
  - Add doc examples showing explicit structure usage
  - Document best practices
  
- [ ] Update `docs/architecture/` if it exists
  - Add diagram showing explicit separation
  - Update any architecture docs mentioning lag variables

- [ ] Update BUG_FIX_PAR_LOWER_BOUND.md
  - Mark issue as RESOLVED
  - Reference this epic and tickets
  - Document the fix and verification

## Technical Notes

### Files to Update

**Core Implementation:**
- `src/subproblem.rs` - remove deprecated fields
- `src/state.rs` - verify only explicit structures used
- `Cargo.toml` - remove migration_validation feature

**Documentation:**
- `README.md` - add architecture section
- `CHANGELOG.md` - document changes
- `BUG_FIX_PAR_LOWER_BOUND.md` - mark resolved
- `docs/ARCHITECTURE_ANALYSIS_EXPLICIT_SEPARATION.md` - add completion notes
- Module docs in affected source files

**Tests:**
- Verify all tests pass
- Remove any migration-specific tests

### CHANGELOG Entry Template

```markdown
## [Unreleased]

### Fixed
- **CRITICAL BUG**: Fixed incorrect cut generation in systems with mixed AR models
  for loads and inflows. Previously, a heuristic-based approach could confuse load
  and inflow lag variables, leading to invalid lower bounds. (#TICKET-004)

### Changed
- **BREAKING**: Refactored lag variable and constraint structures for explicit 
  load/inflow separation. The unified `lagged_state` and `lag_fixing_constraints` 
  fields have been replaced with separate `load_lags`/`inflow_lags` and 
  `load_lag_constraints`/`inflow_lag_constraints` structures. (#EPIC)
  
  **Migration**: If you access these fields directly:
  - Old: `variables.lagged_state[entity_idx]`
  - New: `variables.load_lags.lags_by_bus[bus_id]` or 
         `variables.inflow_lags.lags_by_hydro[hydro_id]`

### Improved
- Cut generation performance improved by ~50% through direct indexed access
- Dual extraction performance improved by ~60% by eliminating entity filtering
- Overall SDDP iteration time reduced by ~2%
- Memory usage unchanged (identical allocation patterns)

### Added
- Comprehensive integration tests for mixed AR order scenarios
- Performance benchmarks for lag variable operations
- Architecture documentation explaining explicit separation design
```

### Migration Guide Structure

```markdown
# Migration Guide: Explicit Lag Separation

## Overview
Version X.Y.Z introduces explicit separation of load and inflow lag variables
and constraints. This improves correctness, performance, and code clarity.

## What Changed

### Variables Structure
**Before:**
```rust
pub struct Variables {
    pub lagged_state: Option<Vec<Vec<usize>>>, // Mixed loads + inflows
}

// Access (entity_idx order not guaranteed)
let lag_var = variables.lagged_state[entity_idx][lag_idx];
```

**After:**
```rust
pub struct Variables {
    pub load_lags: Option<LoadLagVariables>,
    pub inflow_lags: Option<InflowLagVariables>,
}

// Access by explicit entity ID
let load_lag = variables.load_lags.lags_by_bus[bus_id][lag_idx];
let inflow_lag = variables.inflow_lags.lags_by_hydro[hydro_id][lag_idx];
```

### Constraints Structure
Similar pattern - see full documentation in docs/ARCHITECTURE_*.md

## Why This Change?

1. **Bug Fix**: Eliminates heuristic-based entity matching that caused incorrect cuts
2. **Performance**: Direct indexed access is 2-3x faster than filtering
3. **Type Safety**: Compiler prevents load/inflow confusion
4. **Clarity**: Code explicitly states which entity type it's processing

## Action Required

### If you don't directly access lag structures
No action needed - internal changes only.

### If you do access lag structures
Update code to use explicit structures with entity IDs instead of entity indices.

See examples in documentation for detailed migration patterns.
```

### Verification Checklist

Before closing ticket:
- [ ] `git grep "lagged_state"` returns only doc references
- [ ] `git grep "lag_fixing_constraints"` returns only doc references
- [ ] `git grep "#\[deprecated\]"` returns no results in src/
- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` has no warnings
- [ ] `cargo doc` builds without errors
- [ ] All examples run correctly
- [ ] Documentation is comprehensive and accurate

### Code Review Focus Areas

Reviewer should check:
1. No deprecated code remains in source
2. All documentation is updated and accurate
3. CHANGELOG entry is comprehensive
4. Migration guide is helpful (if needed)
5. No "TODO" or "FIXME" comments related to migration remain
6. Code follows project style guidelines
7. No dead code or unused imports

## Dependencies

- Blocked by: All previous tickets (TICKET-001 through TICKET-009)
- Blocks: None
- Related: Completes the epic

## Definition of Done

- [ ] All deprecated code removed
- [ ] All tests pass without deprecated structures
- [ ] No clippy warnings
- [ ] Documentation fully updated
- [ ] CHANGELOG entry complete
- [ ] Migration guide created (if needed)
- [ ] Code review completed and approved
- [ ] Epic marked as complete
- [ ] Team notified of changes
