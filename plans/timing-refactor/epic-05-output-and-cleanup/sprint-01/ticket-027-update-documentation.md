# [T-027] Update documentation

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-026](./ticket-026-dead-code-cleanup.md)  
> **Blocks**: [T-028](./ticket-028-final-verification.md)

## Files to Read Before Starting

- `src/timing/mod.rs` - Module documentation
- `src/timing/iteration.rs` - Main entry point documentation
- `TIMING.md` - If exists, timing documentation
- `CHANGELOG.md` - For documenting changes

## Context

### Background

With the timing refactor complete, documentation needs to be updated to reflect the new architecture. This includes module docs, CHANGELOG, and any user-facing documentation.

### Current State

- Module docs may reference old types
- CHANGELOG doesn't document the refactor
- Examples may use old API

### Target State

- All timing module docs reference new types
- CHANGELOG documents breaking changes
- Examples updated (if any exist)

## Specification

### Documentation Updates

1. **Module documentation** (`timing/mod.rs`)
   - Update overview to describe new hierarchy
   - Update example code to use `NewIterationTiming`
   - Document the `to_output()` pattern

2. **CHANGELOG.md**
   - Document breaking changes in output schema
   - Document removed timing struct fields
   - Document new timing fields

3. **API documentation**
   - Ensure all public types have doc comments
   - Ensure all public methods have doc comments

### CHANGELOG Entry

Add to CHANGELOG.md:

```markdown
## [Unreleased]

### Changed

- **BREAKING**: Refactored timing infrastructure
  - `ForwardPassTiming` replaced with `ForwardTimingOutput`
  - `BackwardPassTiming` replaced with `BackwardTimingOutput`
  - `IterationResult.forward_timing` and `IterationResult.backward_timing` 
    consolidated into `IterationResult.timing`

- **BREAKING**: CSV/Parquet output schema changes
  - Removed: `backward_preprocessing_ms`
  - Removed: `backward_cut_cloning_ms`
  - Removed: `backward_handler_application_ms`
  - Renamed: `backward_fcf_state_update_ms` → `backward_problem_update_ms`
  - Added: `model_allocation_ms`, `model_cleanup_ms` (optional)
  - Added: `forward_parallel_overhead_ms`, `forward_solver_max_ms` (optional)

### Internal

- Consolidated ~15 timing structs into unified hierarchy
- Introduced RAII `TimingGuard` pattern for timing collection
- Per-trajectory timing stored for statistical analysis
```

## Acceptance Criteria

- [ ] Module docs in `timing/mod.rs` updated
- [ ] CHANGELOG.md updated with breaking changes
- [ ] All public types have doc comments
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] Examples compile (if any exist)

## Implementation Guide

### Step 1: Update timing/mod.rs docs

Update the module-level documentation:

```rust
//! Zero-pollution timing infrastructure for performance measurement.
//!
//! # Architecture
//!
//! The timing system uses a hierarchical structure:
//!
//! ```text
//! NewIterationTiming
//! ├── model_allocation
//! ├── forward: NewForwardTiming
//! │   ├── preprocessing.saa_sampling
//! │   ├── parallel.wall, parallel.trajectories[]
//! │   └── postprocessing.detail_capturing
//! ├── backward: NewBackwardTiming
//! │   ├── phase1 (model_preprocessing, solver, model_postprocessing)
//! │   ├── phase2 (cut_selection)
//! │   └── phase3 (problem_update)
//! └── model_cleanup
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let timing = NewIterationTiming::new(num_forward_passes);
//! {
//!     let _guard = TimingGuard::new(&timing.forward.preprocessing.saa_sampling);
//!     sample_scenarios();
//! }
//! timing.compute_total();
//! let output = timing.to_output();
//! ```
```

### Step 2: Update CHANGELOG.md

Add the entry shown in the specification.

### Step 3: Build docs and fix warnings

```bash
cargo doc -p powers-rs --no-deps 2>&1 | grep warning
```

Fix any documentation warnings.

### Pitfalls to Avoid

- ⚠️ Don't forget to update examples if they exist
- ⚠️ Ensure breaking changes are clearly documented
- ⚠️ Check for stale doc comments that reference old types

## Testing Requirements

### Documentation Tests

```bash
cargo doc -p powers-rs --no-deps
```

### Doc Tests

```bash
cargo test -p powers-rs --doc
```

## Documentation Requirements

This ticket IS the documentation task.

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Documentation updates are straightforward but need thoroughness
