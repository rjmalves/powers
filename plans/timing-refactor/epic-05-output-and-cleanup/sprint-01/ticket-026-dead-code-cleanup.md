# [T-026] Dead code cleanup

> **Epic**: [Epic 5: Output & Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-025](./ticket-025-remove-old-metrics.md)  
> **Blocks**: [T-027](./ticket-027-update-documentation.md)

## Files to Read Before Starting

- All files in `src/timing/` - check for dead code
- `src/sddp/mod.rs` - verify no dead timing code
- `src/algorithm/` - verify no dead timing code
- `src/output/` - verify no dead timing code

## Context

### Background

After removing legacy timing types, there may be unused imports, dead code, and orphaned helper functions throughout the codebase. This ticket uses `cargo clippy` and manual inspection to find and remove all dead code.

### Current State

Unknown - need to audit for:
- Unused imports
- Dead functions
- Orphaned type aliases
- Unused constants

### Target State

- Zero `#[allow(dead_code)]` attributes added for this refactor
- Zero unused import warnings
- Clean `cargo clippy` output

## Specification

### Cleanup Targets

1. **Unused imports**: Remove `use` statements for types that no longer exist
2. **Dead functions**: Remove helper functions that are no longer called
3. **Orphaned type aliases**: Remove type aliases that reference removed types
4. **Unused constants**: Remove constants only used by removed code
5. **Empty modules**: Remove or document empty modules

### Common Patterns to Find

```rust
// Unused import
use std::time::Instant;  // Remove if no Instant::now() calls remain

// Dead helper function
fn aggregate_timing(...) { ... }  // Remove if no longer called

// Orphaned type alias
type OldTimingAccumulator = ...;  // Remove
```

## Acceptance Criteria

- [ ] `cargo clippy -p powers-rs --all-targets` has no warnings
- [ ] `cargo build -p powers-rs --all-targets` has no warnings
- [ ] No `#[allow(dead_code)]` added for this refactor
- [ ] Manual audit confirms no obvious dead code
- [ ] `cargo test` passes

## Implementation Guide

### Step 1: Run clippy

```bash
cargo clippy -p powers-rs --all-targets 2>&1 | tee clippy_output.txt
```

### Step 2: Fix all warnings

Address each warning:
- `unused_imports`: Remove the import
- `dead_code`: Remove the function/struct
- `unused_variables`: Remove or prefix with `_`
- `unreachable_code`: Investigate and fix

### Step 3: Manual audit of timing-related code

Check these locations:
```bash
# Find any remaining old timing patterns
grep -rn "ForwardPassTiming\|BackwardPassTiming" src/
grep -rn "TimingAccumulator" src/
grep -rn "timing_start\|timing_end" src/
```

### Step 4: Check for TODO/FIXME comments

```bash
grep -rn "TODO\|FIXME" src/timing/ src/sddp/mod.rs
```

Address any timing-related TODOs.

### Step 5: Run full test suite

```bash
cargo test -p powers-rs
```

### Pitfalls to Avoid

- ⚠️ Don't remove code that appears dead but is used via macros
- ⚠️ Don't remove `pub` items that are part of public API
- ⚠️ Check if any removed items are used in examples/

## Testing Requirements

### Compilation Tests

```bash
cargo build -p powers-rs --all-targets
cargo clippy -p powers-rs --all-targets
```

### Unit Tests

```bash
cargo test -p powers-rs
```

### Integration Tests

Run full test suite to ensure nothing was accidentally removed.

## Documentation Requirements

- [ ] No documentation changes needed

## Effort Estimate

**Points**: 1  
**Confidence**: High  
**Rationale**: Automated tooling (clippy) does most of the work
