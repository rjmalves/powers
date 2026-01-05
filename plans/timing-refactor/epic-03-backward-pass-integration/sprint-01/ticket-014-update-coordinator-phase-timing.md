# [T-014] Update coordinator.rs phase timing

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-013](./ticket-013-update-processor-timing.md)  
> **Blocks**: [T-015](./ticket-015-remove-legacy-backward-timing.md)

## Files to Read Before Starting

- `src/algorithm/coordinator.rs` - `ParallelHandlerCoordinator` implementation
- `src/sddp/mod.rs` - `BackwardPhase1Timing` struct (currently used by coordinator)
- `src/timing/backward.rs` - New `BackwardPhase1Timing` from timing module

## Context

### Background

The `ParallelHandlerCoordinator` implements `BackwardStageProcessor` and collects Phase 1 timing from parallel handler operations. Currently it uses `sddp::BackwardPhase1Timing` which is separate from the new timing module's `BackwardPhase1Timing`.

### Current State

```rust
// coordinator.rs line 38
use crate::sddp::{BackwardPhase1Timing, SddpTrainHandler};
```

The coordinator collects `BackwardPhase1Timing` from each handler and aggregates them.

### Problem

There are TWO `BackwardPhase1Timing` types:
1. `sddp::BackwardPhase1Timing` - currently used
2. `timing::BackwardPhase1Timing` - new version (Cell<Duration>)

We need to consolidate on one.

## Specification

### Decision: Keep sddp::BackwardPhase1Timing for internal use

The `sddp::BackwardPhase1Timing` (if it exists) is used internally by handlers. The new `timing::BackwardPhase1Timing` uses `Cell<Duration>` for accumulation.

For coordinator phase timing:
1. Keep using handler's internal timing type
2. Convert to `CutComputationTiming` (plain Duration) for return
3. Caller accumulates into `timing::NewBackwardTiming`

### Changes Required

1. **Check sddp::BackwardPhase1Timing**: Verify it exists and how it's used
2. **Update aggregation**: Ensure coordinator correctly sums handler timings
3. **Return CutComputationTiming**: Continue returning plain Duration struct

### Verification Steps

```bash
# Find BackwardPhase1Timing definitions
grep -rn "struct BackwardPhase1Timing" src/

# Find usages in coordinator
grep -n "BackwardPhase1Timing" src/algorithm/coordinator.rs
```

## Acceptance Criteria

- [ ] Coordinator aggregates handler phase timing correctly
- [ ] Returns `CutComputationTiming` to backward pass
- [ ] No duplicate timing types visible in public API
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Understand current flow

Trace the timing flow:
1. Handler executes branching solves → returns `BackwardPhase1Timing`
2. Coordinator aggregates across handlers → builds `CutComputationTiming`
3. Backward pass receives `Phase1Result` with `CutComputationTiming`
4. Backward pass accumulates into `NewBackwardTiming`

### Step 2: Verify aggregation logic

In `coordinator.rs`, find where handler timings are summed:

```rust
// Look for pattern like:
let total_timing = CutComputationTiming {
    model_preprocessing: handlers_timing.iter().map(|t| t.model_preprocessing).sum(),
    solver: handlers_timing.iter().map(|t| t.solver).sum(),
    // ...
};
```

### Step 3: Add Instant::now() removal (if applicable)

If coordinator has manual `Instant::now()` calls, consider whether they can be replaced with `TimingGuard`. However, for parallel phases this may not be possible.

### Step 4: Document timing flow

Add doc comments explaining the timing data flow:

```rust
impl BackwardStageProcessor for ParallelHandlerCoordinator {
    /// Phase 1: Compute cuts in parallel.
    ///
    /// # Timing Flow
    ///
    /// 1. Each handler reports its own `BackwardPhase1Timing`
    /// 2. Coordinator sums timings across handlers
    /// 3. Returns `CutComputationTiming` (plain Duration) in `Phase1Result`
    /// 4. Caller (`backward_pass::execute`) accumulates into `NewBackwardTiming`
    fn compute_cuts_parallel_into_slots(...) -> Result<Phase1Result, String> {
```

### Pitfalls to Avoid

- ⚠️ Don't confuse `sddp::BackwardPhase1Timing` with `timing::BackwardPhase1Timing`
- ⚠️ Parallel timing can't use `TimingGuard` easily - keep Instant::now() for wall time
- ⚠️ Sum of parallel times != wall time (that's the overhead calculation)

## Testing Requirements

### Unit Tests

```bash
cargo test -p powers-rs coordinator
```

### Integration Tests

Run backward pass integration tests to verify timing is still collected.

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Need to trace timing flow through coordinator; may find complexity
