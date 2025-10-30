# CLEANUP-005: Audit and Remove HighsBasisStatus Enum

## Context

The `HighsBasisStatus` enum in `src/solver.rs` (line 74) is marked with `#[allow(dead_code)]` and appears to be completely unused. This suggests it may be leftover from an abandoned warm-start feature or similar solver enhancement.

**Current State**:
```rust
#[allow(dead_code)]
pub enum HighsBasisStatus {
    Lower = 0_isize,
    Basic = 1_isize,
    Upper = 2_isize,
    Zero = 3_isize,
    NonBasic = 4_isize,
}
```

**Risk Level**: LOW (appears unused, but need to verify)

## Acceptance Criteria

- [ ] Git history examined to understand when and why enum was added
- [ ] Codebase searched for any references to `HighsBasisStatus`
- [ ] Decision made: remove entirely OR move to separate module with documentation
- [ ] If removing: Enum deleted and `#[allow(dead_code)]` removed
- [ ] If keeping: Moved to `basis.rs` with clear documentation of future intent
- [ ] All tests pass after changes
- [ ] No compiler warnings introduced

## Tasks

### Investigation
- [ ] Search codebase for all references to `HighsBasisStatus`:
  ```bash
  rg "HighsBasisStatus" src/ tests/
  ```
- [ ] Check git history to see when enum was introduced:
  ```bash
  git log --all -p -- src/solver.rs | grep -A 10 -B 10 "HighsBasisStatus"
  ```
- [ ] Check git log for mentions of "warm start" or "basis":
  ```bash
  git log --all --grep="warm.start\|basis" --oneline
  ```
- [ ] Review unused helper functions (lines 381, 451) to see if they reference basis status
- [ ] Check if enum matches HiGHS C API definition (may be needed for future feature)
- [ ] Check issue tracker for any planned warm-start features

### Decision Making
- [ ] **If completely unused and no planned feature**: Remove entirely
- [ ] **If part of incomplete feature**: 
  - Check if feature is planned (issue tracker)
  - If planned soon (next 3 months): Keep with better documentation
  - If planned far future: Remove and recreate when needed
- [ ] **If needed for future compatibility**: Move to separate module with clear docs

### Option A: Remove Entirely
- [ ] Delete `HighsBasisStatus` enum from `src/solver.rs`
- [ ] Remove related `#[allow(dead_code)]` attribute
- [ ] Search for and remove any related unused helper functions
- [ ] Run full test suite to verify no breakage

### Option B: Keep with Better Documentation
- [ ] Create new file `src/basis.rs`
- [ ] Move `HighsBasisStatus` enum to `basis.rs`
- [ ] Add comprehensive module documentation:
  ```rust
  //! Basis status types for advanced solver features (warm-start, sensitivity analysis).
  //!
  //! Currently unused but defined to match HiGHS C API for future implementation.
  //! See issue #XXX for planned warm-start feature.
  ```
- [ ] Add doc comments explaining each variant
- [ ] Keep `#[allow(dead_code)]` but add explanation:
  ```rust
  #[allow(dead_code)] // Reserved for warm-start feature (issue #XXX)
  ```
- [ ] Add `mod basis;` to `src/lib.rs` if making it a module

### Testing
- [ ] Run `cargo test --workspace` to ensure no breakage
- [ ] Run `cargo clippy --all-targets -- -D warnings` to verify no new warnings
- [ ] Run `cargo build --release` to ensure clean build
- [ ] If enum is kept: Verify it matches HiGHS C API definition

### Documentation
- [ ] Update CHANGELOG.md if removing: "Removed unused HighsBasisStatus enum"
- [ ] If keeping: Update docs/architecture/SOLVER.md with note about basis status types
- [ ] Remove from cleanup backlog

## Technical Notes

**Location**: `src/solver.rs:74`

**Related Code to Check**:
- Lines 381, 451: Unused helper functions that may relate to basis
- `highs-sys` crate: Check if this enum should match C API types
- HiGHS documentation: Verify if basis status is needed for any current features

**HiGHS C API Context**:
The enum values (0-4) match the HiGHS basis status constants:
- `kHighsBasisStatusLower = 0`
- `kHighsBasisStatusBasic = 1`
- `kHighsBasisStatusUpper = 2`
- `kHighsBasisStatusZero = 3`
- `kHighsBasisStatusNonbasic = 4`

This suggests the enum was added to match the C API, likely for a warm-start feature that was never completed.

**Warm-Start Context**:
Warm-starting allows providing an initial basis to the solver, which can significantly speed up re-optimization when solving similar problems (e.g., in SDDP forward passes). However, this requires storing and managing basis information, which adds complexity.

**Recommendation**: Unless warm-start is planned for next quarter, remove the enum. It can easily be recreated when actually needed.

## Dependencies

- Blocked by: None
- Blocks: None
- Related: CLEANUP-006 (reviews other `#[allow(dead_code)]` in solver.rs)

## Estimated Effort

**0.5 story points** (2-4 hours, confidence: high)

Straightforward audit and removal/documentation work. Time includes investigation of git history and decision making.
