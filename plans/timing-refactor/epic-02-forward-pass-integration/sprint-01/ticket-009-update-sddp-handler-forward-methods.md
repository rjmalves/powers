# [T-009] Update SddpHandler forward methods

> **Epic**: [Epic 2: Forward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-008](./ticket-008-update-forward-pass-timing-types.md)  
> **Blocks**: [T-010](./ticket-010-remove-legacy-forward-timing.md)

## Files to Read Before Starting

- `src/sddp/mod.rs` - Main SDDP module with `SddpTrainHandler`
- `src/timing/trajectory.rs` - New `TrajectoryTiming`
- `src/timing/forward.rs` - `NewForwardTiming` and `ForwardParallelTiming`
- `plans/timing-refactor/00-master-plan.md` - Forward pass execution flow

## Context

### Background

The `SddpTrainHandler` struct manages per-handler state including forward pass execution. The forward pass methods need to use the new timing types from `timing/`. This ticket updates the handler to accept and work with new timing types.

### Current State

The handler stores and uses timing internally. The training loop in `sddp/mod.rs` has code that:
1. Creates `TrajectoryTiming` instances for each forward pass
2. Calls handler forward methods with timing
3. Aggregates timing after parallel forward passes
4. Converts timing to output format

## Specification

### Changes Required

1. **Update imports in sddp/mod.rs**: Add import for `timing::TrajectoryTiming`
2. **Update handler forward pass calls**: Ensure timing parameters use new types
3. **Keep backward compatibility**: The handler should work with both old and new timing during transition

### Key Code Locations

In `src/sddp/mod.rs`, find:
- Forward pass timing creation (look for `TrajectoryTiming::default()`)
- Forward pass execution calls (look for `forward_pass::execute`)
- Timing aggregation after parallel section

### Behavior Preservation

The forward pass flow must continue to:
1. Create N trajectory timings (one per forward pass)
2. Execute forward passes in parallel
3. Aggregate timing after completion
4. Report timing in output

## Acceptance Criteria

- [ ] `SddpTrainHandler` forward methods work with `timing::TrajectoryTiming`
- [ ] Parallel forward pass timing collection still works
- [ ] Timing is correctly aggregated after parallel section
- [ ] No duplicate timing definitions
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Locate forward pass timing creation

Search in `src/sddp/mod.rs` for where `TrajectoryTiming` instances are created:

```bash
grep -n "TrajectoryTiming" src/sddp/mod.rs
```

### Step 2: Update to use timing module version

If the code creates timing with:
```rust
use crate::algorithm::context::TrajectoryTiming;
```

Change to:
```rust
use crate::timing::TrajectoryTiming;
```

### Step 3: Verify parallel timing collection

Ensure the parallel forward pass section correctly:
1. Creates one `TrajectoryTiming` per handler
2. Passes timing to `forward_pass::execute()`
3. Collects timing after parallel execution

### Step 4: Verify aggregation

Ensure `aggregate_trajectory_timings()` is called correctly with the collected timings.

### Pitfalls to Avoid

- ⚠️ The training loop may use `ForwardPassTimingAccumulator` - don't remove it yet (T-010)
- ⚠️ There may be multiple places that create `TrajectoryTiming` - update all of them
- ⚠️ Watch for `legacy_timing` conversion code - note it for removal in T-010

## Testing Requirements

### Unit Tests

- Existing forward pass tests should pass

### Integration Tests

```bash
cargo test --test integration_tests
```

### Manual Verification

Run a simple case to verify timing is still collected:
```bash
cargo run --example [simple_example] 2>&1 | grep -i timing
```

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Need to trace through training loop code; may have multiple update points
