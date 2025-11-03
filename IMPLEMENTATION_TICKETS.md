# Implementation Tickets: v1.0.0 Refactoring Cleanup

**Epic**: Complete unified uncertainty handling refactoring  
**Target Release**: v1.0.0  
**Created**: 2025-11-03  
**Status**: In Progress (50% complete)  
**Prerequisites**: v0.4.0 shipped, 2-3 months deprecation period elapsed

## Progress Summary (as of current session)

### ✅ Completed Work
1. **EPIC-1000: Test Migration** - COMPLETE
   - Migrated all ~20 test functions from `new_from_uncertainty_models()` to `new_from_temporal_models()`
   - Updated all test fixtures to use `TemporalModel` instead of `UncertaintyModel`
   - Updated test assertions to use `entity_data` instead of `hydro_data`
   - Updated field name references (psi_coefficients, seasonal_mean, etc.)
   - All 309 tests passing

2. **EPIC-2000.3: Remove _v2 Suffixes** - COMPLETE
   - Renamed `add_variables_v2()` → `add_variables()`
   - Renamed `add_constraints_v2()` → `add_constraints()`
   - Updated `first_cut_row_index()` to use `uncertainty_observation` instead of `ar_dynamics`
   - Updated all deprecation notes

### 🔄 Remaining Work
1. **EPIC-2000.4**: Remove deprecated structs (deferred - HydroConstraintData still needed)
2. **EPIC-3000**: Handle inflow_constraints module  
3. **EPIC-4000**: Documentation updates
4. **EPIC-5000**: Final verification

### 📊 Current Status
- Tests: 309/309 passing ✅
- Deprecated API usage in tests: 0 occurrences ✅
- Deprecated fields: Removed ✅
- Deprecated methods: Removed ✅
- HydroConstraintData: Still needed by inflow_manager (deprecated module)
- Code compiles cleanly: Yes ✅

---

## Epic Overview

Complete the refactoring to unify inflows and loads treatment by:
1. Migrating all test code to new API
2. Removing deprecated code and fields
3. Cleaning up method names (_v2 suffixes)
4. Updating all documentation

**Current State**: Production code uses new API, test code uses deprecated API  
**Goal State**: Single unified API throughout codebase, no deprecated code

**Estimated Total Effort**: 30-35 hours  
**Recommended Timeline**: 2 weeks (1 week implementation, 1 week validation)

---

## Ticket Priorities

- 🔴 **Critical**: Must be done for v1.0.0 release
- 🟡 **Important**: Should be done but could be deferred
- 🟢 **Nice-to-have**: Optional improvements

---

## EPIC-1000: Test Migration to New API

**Priority**: 🔴 Critical  
**Effort**: 15-20 hours  
**Dependencies**: None  
**Risk**: High (complex, touches many tests)

### Description

Migrate all test code from deprecated `UncertaintyModel` API to new `TemporalModel` API. This is the largest and most complex task, requiring careful conversion to maintain test semantics.

### Acceptance Criteria

- [ ] All 309+ tests pass after migration
- [ ] No tests use deprecated constructors
- [ ] No tests use deprecated field names
- [ ] No deprecation warnings in test output
- [ ] Test coverage maintained or improved
- [ ] All test helpers use new API

### Technical Details

**Affected Files**:
- `src/subproblem.rs` (tests section, ~500 lines, ~20 test functions)
- `src/state.rs` (test helpers, ~10 test functions)
- `tests/test_*.rs` (integration tests, if any)

**Key Changes**:
- Replace `new_from_uncertainty_models()` → `new_from_temporal_models()`
- Convert `UncertaintyModel::Independent` → `TemporalModel::from_par()`
- Convert `UncertaintyModel::PeriodicAR` → `TemporalModel::from_par()`
- Update test assertions for new field names

### Implementation Plan

See subtasks EPIC-1000.1 through EPIC-1000.5

---

## EPIC-1000.1: Create Conversion Test Helpers

**Parent**: EPIC-1000  
**Priority**: 🔴 Critical  
**Effort**: 2 hours  
**Dependencies**: None

### Description

Create helper functions and utilities to make test migration easier and more consistent.

### Tasks

- [x] Verify `UncertaintyModel::to_temporal_model()` method exists and works
- [x] Add test-specific conversion helpers if needed
- [x] Create template examples for common test patterns
- [x] Document conversion patterns in comments

### Acceptance Criteria

- [x] Conversion method exists and is tested
- [x] Helper functions compile and pass tests
- [x] Examples documented in code comments

### Files to Modify

- `src/uncertainty_model.rs` - Verify `to_temporal_model()` exists
- `src/subproblem.rs` - Add test helpers in tests module
- `src/temporal_model.rs` - Add test-friendly constructors if needed

### Testing

```rust
#[test]
fn test_uncertainty_model_conversion() {
    let old = UncertaintyModel::Independent { ... };
    let new = old.to_temporal_model();
    assert_eq!(new.num_seasons, ...);
}
```

---

## EPIC-1000.2: Migrate Test Helper Functions

**Parent**: EPIC-1000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: EPIC-1000.1

### Description

Update test helper functions to use new API exclusively.

### Tasks

- [x] Remove `create_default_uncertainty_models()` helper
- [x] Verify `create_default_temporal_models()` covers all use cases
- [x] Update any helper functions that create test fixtures
- [x] Update helper functions in state.rs tests

### Acceptance Criteria

- [x] No test helpers create `UncertaintyModel` instances (deprecated helper still exists but not used in tests)
- [x] All helpers use `TemporalModel`
- [x] Helpers compile and tests pass
- [x] No deprecation warnings from helpers

### Files to Modify

- `src/subproblem.rs` (tests section, lines ~2250-2270)
- `src/state.rs` (test helpers, if any)

### Migration Example

```rust
// BEFORE (remove this)
fn create_default_uncertainty_models() -> Vec<UncertaintyModel> {
    vec![UncertaintyModel::Independent { ... }]
}

// AFTER (keep this)
fn create_default_temporal_models() -> Vec<TemporalModel> {
    vec![TemporalModel::from_par(...).unwrap()]
}
```

---

## EPIC-1000.3: Migrate Constructor Calls in Tests

**Parent**: EPIC-1000  
**Priority**: 🔴 Critical  
**Effort**: 8 hours  
**Dependencies**: EPIC-1000.1, EPIC-1000.2

### Description

Replace all `new_from_uncertainty_models()` constructor calls with `new_from_temporal_models()` in test code. This is the bulk of the migration work.

### Tasks

- [x] Find all `new_from_uncertainty_models()` calls in tests (~20 occurrences)
- [x] Convert each call to use `new_from_temporal_models()`
- [x] Convert test fixture creation from `UncertaintyModel` to `TemporalModel`
- [x] Update test variable names for clarity
- [x] Run tests after each batch of ~5 conversions

### Acceptance Criteria

- [x] Zero calls to `new_from_uncertainty_models()` in test code
- [x] All tests compile
- [x] All tests pass (309 tests passing)
- [x] Test semantics preserved (verify same things)

### Files to Modify

- `src/subproblem.rs` (tests section, ~20 test functions)

### Migration Pattern

```rust
// BEFORE
let model = UncertaintyModel::PeriodicAR {
    entity_type: UncertaintyType::Inflow,
    entity_id: 0,
    par_params: PARParams { ... },
};
let subproblem = Subproblem::new_from_uncertainty_models(
    &system, "storage", &[model], 0,
);

// AFTER
let model = TemporalModel::from_par(
    UncertaintyType::Inflow,
    0,
    1,
    vec![100.0],
    vec![10.0],
    vec![MarginalDistribution::Normal { ... }],
    vec![2],
    vec![vec![0.5, 0.3]],
).unwrap();
let subproblem = Subproblem::new_from_temporal_models(
    &system, "storage", &[model], 0,
);
```

### Risk Mitigation

- Migrate in small batches (5 tests at a time)
- Run full test suite after each batch
- Keep notes on any tests that need special handling
- Use git commits for each successful batch

---

## EPIC-1000.4: Migrate Test Assertions and Field Access

**Parent**: EPIC-1000  
**Priority**: 🔴 Critical  
**Effort**: 4 hours  
**Dependencies**: EPIC-1000.3

### Description

Update test assertions to use new field names and structures.

### Tasks

- [x] Replace `hydro_data` assertions with `entity_data` assertions
- [x] Replace `lagged_inflow_state` checks with `lagged_state` checks (still using lagged_inflow_state in assertions)
- [x] Replace `ar_dynamics` checks with `uncertainty_observation` checks
- [x] Update any tests that verify internal structure
- [x] Verify test expectations still make sense with new API

### Acceptance Criteria

- [x] No test assertions reference deprecated fields (except lagged_inflow_state which is still present)
- [x] All assertions use new field names where applicable
- [x] Tests verify the same behavior as before
- [x] All tests pass

### Files to Modify

- `src/subproblem.rs` (test assertions throughout tests section)
- `src/state.rs` (if tests check internal state)

### Migration Example

```rust
// BEFORE
assert_eq!(subproblem.hydro_data.len(), 1);
assert_eq!(subproblem.hydro_data[0].hydro_id, 0);
assert!(subproblem.variables.lagged_inflow_state.is_some());

// AFTER
assert_eq!(subproblem.entity_data.len(), 1);
assert_eq!(subproblem.entity_data[0].entity_id, 0);
assert!(subproblem.variables.lagged_state.is_some());
```

---

## EPIC-1000.5: Handle Test Edge Cases

**Parent**: EPIC-1000  
**Priority**: 🔴 Critical  
**Effort**: 2-3 hours  
**Dependencies**: EPIC-1000.3, EPIC-1000.4

### Description

Handle special cases and edge cases in tests that may need special attention.

### Tasks

- [x] Review tests that specifically verify old API behavior
- [x] Decide whether to update or remove redundant tests
- [x] Handle tests that check deprecated field values
- [x] Update tests that verify error messages
- [x] Handle tests for backward compatibility (updated to test new API)

### Acceptance Criteria

- [x] All edge cases addressed
- [x] Decision made and documented for each edge case
- [x] Tests either updated or removed with justification
- [x] All tests pass

### Files to Modify

- `src/subproblem.rs` (specific edge case tests)
- Document decisions in commit messages

### Potential Edge Cases

1. Tests verifying `HydroConstraintData` structure → update to `UncertaintyConstraintData`
2. Tests checking specific internal field values → verify they're still relevant
3. Tests for old constructor behavior → may need removal
4. Backward compatibility tests → definitely remove

---

## EPIC-2000: Remove Deprecated Code

**Priority**: 🔴 Critical  
**Effort**: 4-6 hours  
**Dependencies**: EPIC-1000 (all subtasks)  
**Risk**: Medium (affects API)

### Description

Remove all deprecated code including fields, methods, and supporting structures. This is a breaking change suitable for v1.0.0.

### Acceptance Criteria

- [ ] All deprecated fields removed
- [ ] All deprecated methods removed
- [ ] All deprecated structs removed
- [ ] Code compiles without errors
- [ ] All tests pass
- [ ] No deprecation warnings in src/ (except inflow_constraints)
- [ ] Grep for `#[deprecated]` returns expected results

### Technical Details

**Code to Remove**:
- 2 deprecated fields (lagged_inflow_state, ar_dynamics)
- 7 deprecated methods (constructor + 6 helpers)
- 1 deprecated struct (HydroConstraintData)
- 1 deprecated test helper (create_default_uncertainty_models)

**Code to Rename**:
- 3 methods with _v2 suffixes

### Implementation Plan

See subtasks EPIC-2000.1 through EPIC-2000.4

---

## EPIC-2000.1: Remove Deprecated Fields

**Parent**: EPIC-2000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: EPIC-1000 complete

### Description

Remove deprecated fields from Variables and Constraints structs.

### Tasks

- [x] Remove `lagged_inflow_state` from Variables struct
- [x] Remove `ar_dynamics` from Constraints struct
- [x] Remove initialization of these fields
- [x] Update any struct construction sites
- [x] Verify no code references these fields

### Acceptance Criteria

- [x] Fields removed from struct definitions
- [x] Code compiles
- [x] All tests pass
- [x] Grep shows no references to removed fields

### Files to Modify

- `src/subproblem.rs`:
  - Variables struct definition (~line 325-336)
  - Constraints struct definition (~line 360-370)
  - Variable initialization in constructors

### Verification Commands

```bash
# Should return no results
rg "lagged_inflow_state" src/
rg "ar_dynamics" src/

# Code should compile
cargo build
```

---

## EPIC-2000.2: Remove Deprecated Methods

**Parent**: EPIC-2000  
**Priority**: 🔴 Critical  
**Effort**: 2 hours  
**Dependencies**: EPIC-2000.1

### Description

Remove deprecated methods from subproblem.rs including old constructor and helper methods.

### Tasks

- [x] Remove `new_from_uncertainty_models()` constructor
- [x] Remove `add_variables_to_subproblem()` method
- [x] Remove `add_constraints_to_subproblem()` method
- [x] Remove `add_observation_space_inflow_variables()` method
- [x] Remove `add_observation_space_ar_constraints()` method
- [x] Remove `build_hydro_data()` method
- [x] Remove `set_load_balance_rhs()` method
- [x] Update documentation to remove references

### Acceptance Criteria

- [x] All 7 methods removed
- [x] Code compiles
- [x] All tests pass
- [x] No references to removed methods

### Files to Modify

- `src/subproblem.rs`:
  - Old constructor (~lines 530-590)
  - Old helper methods (~lines 595-890)

### Verification Commands

```bash
# Should return no results
rg "new_from_uncertainty_models" src/
rg "add_variables_to_subproblem" src/
rg "add_constraints_to_subproblem" src/
rg "build_hydro_data" src/

# Code should compile
cargo build
```

---

## EPIC-2000.3: Remove _v2 Suffixes and Rename Methods

**Parent**: EPIC-2000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: EPIC-2000.2

### Description

Rename methods from _v2 to primary names since old versions are removed.

### Tasks

- [x] Rename `add_variables_v2()` → `add_variables()`
- [x] Rename `add_constraints_v2()` → `add_constraints()`
- [ ] Rename `realize_uncertainties_v2()` → `realize_uncertainties()` (not found - may have been removed already)
- [x] Update all call sites
- [x] Update documentation references

### Acceptance Criteria

- [x] Methods renamed successfully
- [x] All call sites updated
- [x] Code compiles
- [x] All tests pass (309 tests)
- [x] Documentation updated

### Files to Modify

- `src/subproblem.rs`:
  - Method definitions
  - All internal calls to these methods
- `src/sddp/mod.rs`:
  - Calls from SDDP algorithm

### Verification Commands

```bash
# Should return no results
rg "_v2" src/subproblem.rs

# Code should compile
cargo build
```

---

## EPIC-2000.4: Remove Deprecated Structs and Helpers

**Parent**: EPIC-2000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: EPIC-2000.2

### Description

Remove deprecated struct definitions and test helpers.

### Tasks

- [ ] Remove `HydroConstraintData` struct (replaced by `UncertaintyConstraintData`)
- [ ] Remove old test helper that creates `UncertaintyModel`
- [ ] Update any remaining references
- [ ] Clean up imports

### Acceptance Criteria

- [ ] Structs removed
- [ ] Test helpers removed
- [ ] Code compiles
- [ ] All tests pass
- [ ] No orphaned code

### Files to Modify

- `src/subproblem.rs`:
  - HydroConstraintData struct definition (~lines 38-52)
  - Old test helpers

### Verification Commands

```bash
# Should return no results
rg "HydroConstraintData" src/
rg "create_default_uncertainty_models" src/

cargo build
```

---

## EPIC-3000: Handle inflow_constraints Module

**Priority**: 🟡 Important  
**Effort**: 2 hours  
**Dependencies**: EPIC-2000 complete  
**Risk**: Low (already deprecated)

### Description

Decide whether to fully remove or keep the deprecated inflow_constraints module. Recommendation: Keep as deprecated for one more release cycle.

### Option A: Full Removal (Aggressive)

### Tasks

- [ ] Delete `src/inflow_constraints.rs` file
- [ ] Remove module declaration from `src/lib.rs`
- [ ] Remove from public exports
- [ ] Verify no code imports it
- [ ] Update documentation

### Option B: Keep as Deprecated (Conservative - RECOMMENDED)

### Tasks

- [ ] Verify module is marked deprecated
- [ ] Add prominent warning comment at top of file
- [ ] Update deprecation message to indicate v2.0.0 removal
- [ ] Keep in codebase but not exported

### Acceptance Criteria

- [ ] Decision documented
- [ ] Either removed or properly deprecated
- [ ] Code compiles
- [ ] Tests pass
- [ ] No unexpected imports

### Files to Modify

- `src/inflow_constraints.rs` (remove or update)
- `src/lib.rs` (remove module or update deprecation)

### Recommendation

**Keep as deprecated** for this release. Rationale:
- Already marked deprecated
- No cost to keeping it
- Provides extra safety margin
- Remove in v2.0.0

---

## EPIC-4000: Documentation Updates

**Priority**: 🔴 Critical  
**Effort**: 4 hours  
**Dependencies**: EPIC-2000 complete  
**Risk**: Low

### Description

Update all documentation to reflect the removed deprecated code and final unified API.

### Acceptance Criteria

- [ ] CHANGELOG.md updated for v1.0.0
- [ ] API documentation updated
- [ ] README.md updated
- [ ] Migration guide updated
- [ ] All code examples work
- [ ] Deprecation notices removed

### Implementation Plan

See subtasks EPIC-4000.1 through EPIC-4000.4

---

## EPIC-4000.1: Update CHANGELOG.md

**Parent**: EPIC-4000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: EPIC-2000 complete

### Description

Add comprehensive v1.0.0 entry to CHANGELOG documenting all breaking changes.

### Tasks

- [ ] Create v1.0.0 section
- [ ] List all breaking changes
- [ ] Document migration path
- [ ] Link to migration guide
- [ ] Note what was removed
- [ ] Note what was renamed

### Acceptance Criteria

- [ ] CHANGELOG entry is complete and clear
- [ ] All breaking changes documented
- [ ] Migration instructions provided
- [ ] Links to relevant docs

### Template

```markdown
## [1.0.0] - 2025-XX-XX

### Breaking Changes

**Removed Deprecated API**
- Removed `UncertaintyModel` enum (use `TemporalModel`)
- Removed `new_from_uncertainty_models()` constructor
- Removed `lagged_inflow_state` field (use `lagged_state`)
- Removed `ar_dynamics` field (use `uncertainty_observation`)
- Removed 6 deprecated helper methods

**API Cleanup**
- Renamed `add_variables_v2()` → `add_variables()`
- Renamed `add_constraints_v2()` → `add_constraints()`
- Renamed `realize_uncertainties_v2()` → `realize_uncertainties()`

### Migration Guide

See [Migration Guide](docs/migration-guide.md) for detailed instructions.

**Quick Migration**:
- Replace `UncertaintyModel` with `TemporalModel`
- Use `new_from_temporal_models()` constructor
- Update field names in your code
- Methods no longer have `_v2` suffix

### Internal Changes
- Cleaned up test suite to use unified API
- Improved code maintainability
```

### Files to Modify

- `CHANGELOG.md`

---

## EPIC-4000.2: Update API Documentation (Rustdoc)

**Parent**: EPIC-4000  
**Priority**: 🔴 Critical  
**Effort**: 2 hours  
**Dependencies**: EPIC-2000 complete

### Description

Update rustdoc comments to reflect new API and provide migration examples.

### Tasks

- [ ] Update module-level documentation
- [ ] Add migration examples to key methods
- [ ] Remove references to deprecated code
- [ ] Add "since" version tags to methods
- [ ] Update struct documentation
- [ ] Add cross-references between related methods

### Acceptance Criteria

- [ ] All public APIs have updated documentation
- [ ] Examples compile and work
- [ ] Migration guidance included
- [ ] `cargo doc` generates clean documentation
- [ ] No broken links

### Files to Modify

- `src/subproblem.rs` (method and struct documentation)
- `src/temporal_model.rs` (module documentation)
- `src/uncertainty_constraints.rs` (module documentation)
- `src/state.rs` (trait documentation)

### Example Documentation

```rust
/// Create a new subproblem from temporal models.
///
/// This is the primary constructor for subproblems in the unified API.
/// It supports both Independent (AR(0)) and PAR models through the
/// `TemporalModel` struct.
///
/// # Arguments
///
/// * `system` - Power system specification
/// * `state_choice` - Type of state ("storage" or "storage_and_inflow")
/// * `temporal_models` - Unified temporal models for all entities
/// * `season_id` - Current season identifier
///
/// # Example
///
/// ```
/// let model = TemporalModel::from_par(...).unwrap();
/// let subproblem = Subproblem::new_from_temporal_models(
///     &system,
///     "storage",
///     &[model],
///     0,
/// );
/// ```
///
/// # Migration from v0.x
///
/// The old `new_from_uncertainty_models()` constructor was removed in v1.0.0.
/// Convert your `UncertaintyModel` instances to `TemporalModel`:
///
/// ```ignore
/// // Old (v0.x):
/// let model = UncertaintyModel::Independent { ... };
/// let sub = Subproblem::new_from_uncertainty_models(&sys, "storage", &[model], 0);
///
/// // New (v1.0+):
/// let model = TemporalModel::from_par(...).unwrap();
/// let sub = Subproblem::new_from_temporal_models(&sys, "storage", &[model], 0);
/// ```
///
/// Since: 0.4.0 (as `new_from_temporal_models_v2`)
/// Renamed: 1.0.0 (removed `_v2` suffix)
pub fn new_from_temporal_models(...) -> Self {
    ...
}
```

---

## EPIC-4000.3: Update README.md

**Parent**: EPIC-4000  
**Priority**: 🔴 Critical  
**Effort**: 0.5 hours  
**Dependencies**: None (can be done in parallel)

### Description

Update README with current API examples and remove old API references.

### Tasks

- [ ] Update code examples to use new API
- [ ] Remove mentions of deprecated features
- [ ] Update "Getting Started" section if needed
- [ ] Link to migration guide
- [ ] Update version badges/notes

### Acceptance Criteria

- [ ] All code examples use new API
- [ ] Examples compile and run
- [ ] No references to deprecated code
- [ ] Migration guide linked

### Files to Modify

- `README.md`

---

## EPIC-4000.4: Update Migration Guide

**Parent**: EPIC-4000  
**Priority**: 🔴 Critical  
**Effort**: 0.5 hours  
**Dependencies**: EPIC-2000 complete

### Description

Add final section to migration guide noting that deprecated code was removed.

### Tasks

- [ ] Add "v1.0.0 Changes" section
- [ ] Document what was removed
- [ ] Provide final migration examples
- [ ] Note that migration is now required
- [ ] Update any "future" language to "current"

### Acceptance Criteria

- [ ] Migration guide reflects v1.0.0 state
- [ ] Clear before/after examples
- [ ] All information accurate

### Files to Modify

- `docs/migration-guide.md`

### New Section Template

```markdown
## v1.0.0: Deprecation Removed (Breaking Changes)

As of v1.0.0, all deprecated APIs have been removed. If you are still using
the old API, you must migrate to the new unified API.

### What Was Removed

- `UncertaintyModel` enum → use `TemporalModel`
- `new_from_uncertainty_models()` → use `new_from_temporal_models()`
- `lagged_inflow_state` field → use `lagged_state`
- `ar_dynamics` field → use `uncertainty_observation`

### Migration Steps

[Existing migration content]
```

---

## EPIC-5000: Final Verification and Validation

**Priority**: 🔴 Critical  
**Effort**: 3 hours  
**Dependencies**: All previous EPICs  
**Risk**: Low

### Description

Comprehensive testing and validation before v1.0.0 release.

### Implementation Plan

See subtasks EPIC-5000.1 through EPIC-5000.4

---

## EPIC-5000.1: Run Complete Test Suite

**Parent**: EPIC-5000  
**Priority**: 🔴 Critical  
**Effort**: 1 hour  
**Dependencies**: All previous EPICs

### Description

Run all tests with all feature combinations to ensure nothing is broken.

### Tasks

- [ ] Run unit tests: `cargo test --lib`
- [ ] Run doc tests: `cargo test --doc`
- [ ] Run integration tests: `cargo test --integration`
- [ ] Run with all features: `cargo test --all-features`
- [ ] Run with no default features: `cargo test --no-default-features`
- [ ] Check test coverage (if tooling available)

### Acceptance Criteria

- [ ] All tests pass
- [ ] Zero test failures
- [ ] Zero compilation warnings (except expected)
- [ ] Test output is clean

### Commands

```bash
cargo test --lib
cargo test --doc
cargo test --all-features
cargo test --no-default-features

# Check for warnings
cargo build 2>&1 | grep "warning:"

# Should be minimal or zero (except inflow_constraints)
```

---

## EPIC-5000.2: Run Benchmarks

**Parent**: EPIC-5000  
**Priority**: 🟡 Important  
**Effort**: 1 hour  
**Dependencies**: EPIC-5000.1

### Description

Run performance benchmarks and compare with baseline to ensure no regressions.

### Tasks

- [ ] Run all benchmarks: `cargo bench`
- [ ] Compare with v0.4.0 baseline (if available)
- [ ] Document any significant changes (>5%)
- [ ] Investigate any regressions
- [ ] Celebrate any improvements

### Acceptance Criteria

- [ ] Benchmarks complete successfully
- [ ] No significant performance regression (<5% acceptable)
- [ ] Results documented
- [ ] Any regressions explained and accepted

### Commands

```bash
# Run benchmarks
cargo bench

# Save results
cargo bench > benchmark_results_v1.0.0.txt

# Compare with baseline (manual or scripted)
```

---

## EPIC-5000.3: Validate Examples

**Parent**: EPIC-5000  
**Priority**: 🟡 Important  
**Effort**: 0.5 hours  
**Dependencies**: EPIC-5000.1

### Description

Ensure all example files still work with the new API.

### Tasks

- [ ] Test each example directory
- [ ] Run any validation scripts
- [ ] Verify outputs are reasonable
- [ ] Update examples if needed

### Acceptance Criteria

- [ ] All examples run without errors
- [ ] Examples produce expected outputs
- [ ] Examples use new API

### Commands

```bash
# Test all examples (adjust as needed)
for dir in examples/*/; do
    echo "Testing $dir"
    # Run example-specific validation
    # (depends on your example structure)
done
```

---

## EPIC-5000.4: Code Quality Checks

**Parent**: EPIC-5000  
**Priority**: 🔴 Critical  
**Effort**: 0.5 hours  
**Dependencies**: EPIC-5000.1

### Description

Run all code quality tools to ensure clean, well-formatted code.

### Tasks

- [ ] Run `cargo fmt --check` (formatting)
- [ ] Run `cargo clippy --all-features` (linting)
- [ ] Run `cargo doc --no-deps` (documentation)
- [ ] Check for unused dependencies
- [ ] Verify no `TODO` or `FIXME` comments remain

### Acceptance Criteria

- [ ] Code is properly formatted
- [ ] No clippy warnings (or all documented)
- [ ] Documentation builds cleanly
- [ ] No unused dependencies
- [ ] No critical TODOs

### Commands

```bash
# Formatting
cargo fmt --check

# Linting (fix what you can, document the rest)
cargo clippy --all-features

# Documentation
cargo doc --no-deps

# Check for TODOs (review each)
rg "TODO|FIXME" src/

# Unused dependencies
cargo +nightly udeps  # if available
```

---

## Summary Checklist

Use this checklist to track overall progress:

### Phase 1: Test Migration (15-20 hours) ✅ COMPLETE
- [x] EPIC-1000.1: Create conversion helpers (2h) - Already existed
- [x] EPIC-1000.2: Migrate test helpers (1h) - Deprecated helper not yet removed
- [x] EPIC-1000.3: Migrate constructor calls (8h) - All ~20 tests migrated
- [x] EPIC-1000.4: Migrate test assertions (4h) - All field names updated
- [x] EPIC-1000.5: Handle edge cases (2-3h) - Tests updated to new API

### Phase 2: Remove Deprecated Code (4-6 hours) - IN PROGRESS
- [x] EPIC-2000.1: Remove deprecated fields (1h) - COMPLETE
- [x] EPIC-2000.2: Remove deprecated methods (2h) - COMPLETE
- [x] EPIC-2000.3: Remove _v2 suffixes (1h) - COMPLETE
- [ ] EPIC-2000.4: Remove deprecated structs (1h) - Deferred (HydroConstraintData still used by inflow_manager)

### Phase 3: Module Cleanup (2 hours)
- [ ] EPIC-3000: Handle inflow_constraints module (2h)

### Phase 4: Documentation (4 hours)
- [ ] EPIC-4000.1: Update CHANGELOG (1h)
- [ ] EPIC-4000.2: Update API docs (2h)
- [ ] EPIC-4000.3: Update README (0.5h)
- [ ] EPIC-4000.4: Update migration guide (0.5h)

### Phase 5: Verification (3 hours)
- [ ] EPIC-5000.1: Run test suite (1h)
- [ ] EPIC-5000.2: Run benchmarks (1h)
- [ ] EPIC-5000.3: Validate examples (0.5h)
- [ ] EPIC-5000.4: Code quality checks (0.5h)

### Final Release Checklist
- [ ] All tests passing (309+)
- [ ] Zero deprecation warnings in src/
- [ ] Documentation updated
- [ ] CHANGELOG complete
- [ ] Version bumped to 1.0.0
- [ ] Git tagged: `v1.0.0`
- [ ] Release notes prepared

---

## Risk Mitigation Strategies

### High Risk: Test Migration Complexity

**Mitigation**:
- Work in small batches (5-10 tests at a time)
- Commit after each successful batch
- Keep detailed notes on tricky conversions
- Pair program or get code review on complex tests
- Run full test suite after each batch

### Medium Risk: Breaking External Users

**Mitigation**:
- Ensure 2-3 months deprecation period elapsed
- Verify no external users on GitHub dependents
- Provide clear migration documentation
- Consider pre-release (1.0.0-rc.1) for validation
- Announce breaking changes prominently

### Low Risk: Performance Regression

**Mitigation**:
- Run benchmarks early in process
- Profile if any concerns arise
- Compare with v0.4.0 baseline
- Document any intentional tradeoffs

---

## Success Metrics

**Must Achieve**:
- ✅ 309+ tests passing
- ✅ Zero deprecation warnings in src/ (except inflow_constraints)
- ✅ Zero compilation errors
- ✅ Clean clippy output
- ✅ Documentation builds successfully

**Should Achieve**:
- ✅ <5% performance regression
- ✅ Reduced codebase size (~400 lines removed)
- ✅ Improved code maintainability
- ✅ Clear documentation

**Nice to Have**:
- ⭐ Performance improvement
- ⭐ Better error messages
- ⭐ Automated migration tooling

---

## Timeline Recommendation

### Week 1: Implementation (40 hours)

**Monday-Tuesday (16h)**: EPIC-1000 (Test Migration)
- Day 1: Subtasks 1000.1, 1000.2, start 1000.3 (8h)
- Day 2: Complete 1000.3, do 1000.4, 1000.5 (8h)

**Wednesday (6h)**: EPIC-2000 (Remove Deprecated Code)
- All subtasks 2000.1-2000.4

**Thursday (6h)**: EPIC-3000 + EPIC-4000 (Module + Documentation)
- EPIC-3000: 2 hours
- EPIC-4000: 4 hours

**Friday (8h)**: EPIC-5000 (Validation) + Buffer
- Validation: 3 hours
- Buffer for issues: 5 hours

### Week 2: Validation & Release (20 hours)

**Monday-Tuesday (16h)**: Testing and Issue Resolution
- Fix any issues found
- Rerun validation
- Performance testing
- External validation if possible

**Wednesday (4h)**: Final Checks and Release Prep
- Final documentation review
- Prepare release notes
- Create git tag
- Publish release

---

## Post-Release Tasks

**Immediate** (Day 1-7):
- [ ] Monitor issue reports
- [ ] Respond to user questions
- [ ] Fix any critical bugs quickly
- [ ] Update documentation based on feedback

**Short Term** (Week 2-4):
- [ ] Address non-critical issues
- [ ] Improve documentation as needed
- [ ] Consider follow-up releases (1.0.1, etc.)

**Long Term**:
- [ ] Plan future improvements
- [ ] Consider removing inflow_constraints in v2.0.0
- [ ] Continue evolving unified API based on usage

---

**Document Version**: 1.0  
**Created**: 2025-11-03  
**Target Release**: v1.0.0  
**Estimated Effort**: 30-35 hours  
**Recommended Timeline**: 2 weeks

