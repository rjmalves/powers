# [T-015] Remove legacy backward timing types

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-014](./ticket-014-update-coordinator-phase-timing.md)  
> **Blocks**: [T-016](./ticket-016-update-backward-pass-tests.md)

## Files to Read Before Starting

- `src/algorithm/backward_pass.rs` - `BackwardPassTimingAccumulator` and `BackwardPassTimingSnapshot`
- `src/algorithm/context.rs` - `BackwardStageTiming`
- `src/sddp/mod.rs` - Look for backward timing types
- `plans/timing-refactor/00-master-plan.md` - See "Current Timing Structs" removal table

## Context

### Background

After T-012 through T-014, backward pass uses new timing types. Now we remove legacy types that are no longer needed.

### Types to Remove

| Location | Type | Reason |
|----------|------|--------|
| `backward_pass.rs` | `BackwardPassTimingAccumulator` | Replaced by `timing::NewBackwardTiming` |
| `backward_pass.rs` | `BackwardPassTimingSnapshot` | Replaced by `timing::BackwardTimingOutput` |
| `context.rs` | `BackwardStageTiming` | No longer used |
| `sddp/mod.rs` | `BackwardPassTimingAccumulator` (if duplicate) | Remove duplicate |

### Types to KEEP

| Location | Type | Reason |
|----------|------|--------|
| `processor.rs` | `CutComputationTiming` | Still used as return type |
| `processor.rs` | `FirstStageTiming` | Still used as return type |
| `sddp/mod.rs` | `BackwardPhase1Timing` | Used by handlers (if needed) |

## Specification

### Removal Steps

1. **Remove BackwardPassTimingAccumulator** from `backward_pass.rs`
2. **Remove BackwardPassTimingSnapshot** from `backward_pass.rs`
3. **Remove BackwardStageTiming** from `context.rs`
4. **Update imports** anywhere that imported these types
5. **Remove associated tests**

### Before Removal Checklist

Before deleting each type, verify:
- [ ] No other file imports it
- [ ] No runtime code references it
- [ ] Tests updated to use new types

## Acceptance Criteria

- [ ] `BackwardPassTimingAccumulator` removed from `backward_pass.rs`
- [ ] `BackwardPassTimingSnapshot` removed from `backward_pass.rs`
- [ ] `BackwardStageTiming` removed from `context.rs`
- [ ] No orphaned imports
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Find all references

```bash
grep -rn "BackwardPassTimingAccumulator" src/
grep -rn "BackwardPassTimingSnapshot" src/
grep -rn "BackwardStageTiming" src/
```

### Step 2: Verify no external usage

Check that no code outside the module uses these types. If they're in `pub` exports, check callers.

### Step 3: Remove from backward_pass.rs

Delete:
- Lines 35-119: `BackwardPassTimingAccumulator` struct and impl
- Lines 121-153: `BackwardPassTimingSnapshot` struct

### Step 4: Remove from context.rs

Delete:
- `BackwardStageTiming` struct and impl (lines 487-515)
- Associated tests

### Step 5: Update mod.rs exports if needed

If any timing types were re-exported from `algorithm/mod.rs`, remove those exports.

### Step 6: Verify compilation

```bash
cargo build --all-targets
```

Fix any errors from removed types.

### Pitfalls to Avoid

- ⚠️ Check `sddp/mod.rs` for duplicates before removing
- ⚠️ Don't remove types still used by coordinator
- ⚠️ Some types may be used in trait bounds - check carefully

## Testing Requirements

### Compilation Test

```bash
cargo build --all-targets
```

### Unit Tests

Run remaining tests:
```bash
cargo test -p powers-rs backward_pass
cargo test -p powers-rs context
```

### Full Suite

```bash
cargo test
```

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward deletion after previous tickets ensure no usage
