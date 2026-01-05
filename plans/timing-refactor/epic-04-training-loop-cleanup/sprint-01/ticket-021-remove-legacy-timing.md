# [T-021] Remove legacy timing structs from sddp

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-020](./ticket-020-update-iteration-result.md)  
> **Blocks**: [T-022](./ticket-022-verify-zero-pollution.md)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Current legacy timing struct definitions
- `src/timing/mod.rs` - Verify new types are exported

## Context

### Background

After T-017 through T-020, the training loop uses the new timing system. The legacy timing structs in `sddp/mod.rs` are now dead code and should be removed to complete the cleanup.

### Current State

`sddp/mod.rs` contains these legacy types (approximate line numbers):
- `ForwardPassTiming` (line ~36)
- `BackwardPassTiming` (line ~46)
- `ForwardPassTimingAccumulator` (line ~58)
- `BackwardPassTimingAccumulator` (line ~138)

### Target State

All legacy timing structs removed from `sddp/mod.rs`. The module should only have:
- `BranchingsTiming` - kept (used by handler)
- `BackwardPhase1Timing` - kept (now in timing module)
- `IterationLifecycleConfig` - kept (not timing-related)

## Specification

### Structs to Remove

| Struct | Location | Reason |
|--------|----------|--------|
| `ForwardPassTiming` | `sddp/mod.rs:36` | Replaced by `ForwardTimingOutput` |
| `BackwardPassTiming` | `sddp/mod.rs:46` | Replaced by `BackwardTimingOutput` |
| `ForwardPassTimingAccumulator` | `sddp/mod.rs:58` | Replaced by `NewForwardTiming` |
| `BackwardPassTimingAccumulator` | `sddp/mod.rs:138` | Replaced by `NewBackwardTiming` |

### Associated Code to Remove

- `ForwardPassTimingAccumulator::aggregate()` method
- `BackwardPassTimingAccumulator::into_timing()` method
- Any unused imports (`Duration`, `Instant` if now unused)

### Code to Keep

| Item | Reason |
|------|--------|
| `BranchingsTiming` | Still used by `eval_first_stage_bound()` |
| `BackwardPhase1Timing` | Used by `compute_cut_into_staging()` - but check if moved to timing module |
| `IterationLifecycleConfig` | Configuration struct, not timing |

## Acceptance Criteria

- [ ] `ForwardPassTiming` removed from `sddp/mod.rs`
- [ ] `BackwardPassTiming` removed from `sddp/mod.rs`
- [ ] `ForwardPassTimingAccumulator` removed from `sddp/mod.rs`
- [ ] `BackwardPassTimingAccumulator` removed from `sddp/mod.rs`
- [ ] `cargo build` succeeds
- [ ] No unused code warnings
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Identify struct locations

```bash
grep -n "^pub struct.*Timing" src/sddp/mod.rs
```

### Step 2: Check for usages

Before deleting, verify no usages remain:
```bash
grep -rn "ForwardPassTiming" src/
grep -rn "BackwardPassTiming" src/
grep -rn "ForwardPassTimingAccumulator" src/
grep -rn "BackwardPassTimingAccumulator" src/
```

### Step 3: Delete structs

Remove the struct definitions and their impl blocks.

### Step 4: Clean up imports

Remove any now-unused imports at the top of `sddp/mod.rs`:
```rust
// Remove if unused
use std::time::{Duration, Instant};
```

### Step 5: Run cargo check

```bash
cargo check -p powers-rs
```

Fix any remaining usage errors.

### Pitfalls to Avoid

- ⚠️ Don't remove `BranchingsTiming` - still in use
- ⚠️ Check if `BackwardPhase1Timing` in sddp is different from timing module version
- ⚠️ Some tests may reference old types - update or remove

## Testing Requirements

### Unit Tests

- Verify removed types are not referenced in tests
- Update or remove tests that used old types

### Integration Tests

```bash
cargo test -p powers-rs sddp
cargo test -p powers-rs
```

### Compilation Tests

```bash
cargo build -p powers-rs --all-targets
```

## Documentation Requirements

- [ ] Remove any doc comments that reference removed types

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Need to carefully identify all usages, may find unexpected dependencies
