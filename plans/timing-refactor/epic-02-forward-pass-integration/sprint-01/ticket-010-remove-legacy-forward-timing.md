# [T-010] Remove legacy forward timing code

> **Epic**: [Epic 2: Forward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-009](./ticket-009-update-sddp-handler-forward-methods.md)  
> **Blocks**: [T-011](./ticket-011-update-forward-pass-tests.md)

## Files to Read Before Starting

- `src/algorithm/context.rs` - Contains `TrajectoryTiming` to be removed
- `src/sddp/mod.rs` - Contains legacy conversion and rescaling code
- `plans/timing-refactor/00-master-plan.md` - See "Current Timing Structs" table

## Context

### Background

After T-008 and T-009, the forward pass uses `timing::TrajectoryTiming`. Now we can remove:
1. The duplicate `TrajectoryTiming` definition in `algorithm/context.rs`
2. The `ForwardPassTimingAccumulator` struct (if present)
3. The "recalibrate" timing rescaling code in the training loop
4. Any `legacy_timing` conversion code

### Removal Targets

From the master plan:

| Location | Struct | Action |
|----------|--------|--------|
| `algorithm/context.rs` | `TrajectoryTiming` | Remove (now in `timing/`) |
| `sddp/mod.rs` | `ForwardPassTimingAccumulator` | Remove |
| `sddp/mod.rs` | Rescaling code (lines ~1984-2005) | Remove |

## Specification

### Phase 1: Remove TrajectoryTiming from context.rs

1. Delete the `TrajectoryTiming` struct definition (lines 144-190)
2. Delete the `impl TrajectoryTiming` block
3. Delete associated tests

### Phase 2: Remove ForwardPassTimingAccumulator (if exists)

Search for and remove any `ForwardPassTimingAccumulator` struct and usage.

### Phase 3: Remove rescaling code

The master plan mentions "recalibrate" scaling code around lines 1984-2005 in `sddp/mod.rs`. This code proportionally redistributes timing values - remove it entirely.

Look for patterns like:
```rust
// Recalibrate timing
let total = x + y + z;
let ratio = measured / total;
x = x * ratio;
y = y * ratio;
// etc.
```

### Phase 4: Remove legacy_timing conversion

Search for `legacy_timing` or timing conversion code that transforms new timing to old format.

## Acceptance Criteria

- [ ] No `TrajectoryTiming` struct in `algorithm/context.rs`
- [ ] No `ForwardPassTimingAccumulator` anywhere in codebase
- [ ] No timing rescaling/redistribution code
- [ ] All imports updated to use `timing::TrajectoryTiming`
- [ ] `cargo build` compiles
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Find all TrajectoryTiming references

```bash
grep -rn "TrajectoryTiming" src/
```

Update each file to import from `timing::` instead of `algorithm::context`.

### Step 2: Remove from context.rs

In `src/algorithm/context.rs`:
- Delete lines 144-190 (TrajectoryTiming struct and impl)
- Delete associated tests in the test module

### Step 3: Search for ForwardPassTimingAccumulator

```bash
grep -rn "ForwardPassTimingAccumulator" src/
```

Remove the struct and all usages.

### Step 4: Find and remove rescaling code

```bash
grep -rn "recalibrate\|rescale\|redistribute" src/sddp/mod.rs
```

Also search for patterns that multiply timing by a ratio:
```bash
grep -n "\* ratio\|\* scale\|/ total" src/sddp/mod.rs
```

### Step 5: Verify no orphaned code

After removal, run:
```bash
cargo build 2>&1 | head -50
```

Fix any compilation errors from removed types.

### Pitfalls to Avoid

- ⚠️ Don't remove `BackwardStageTiming` from context.rs - it's still used (Epic 3)
- ⚠️ Ensure all callers are updated before removing definitions
- ⚠️ Keep the tests that use `timing::TrajectoryTiming` - only remove tests for the old version

## Testing Requirements

### Compilation Test

```bash
cargo build --all-targets
```

### Unit Tests

```bash
cargo test -p powers-rs context -- --nocapture
cargo test -p powers-rs forward_pass -- --nocapture
```

### Full Test Suite

```bash
cargo test
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward deletion once usages are updated
