# [T-017] Create IterationTiming in training loop

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 2, Epic 3 complete  
> **Blocks**: [T-018](./ticket-018-replace-forward-timing.md), [T-019](./ticket-019-replace-backward-timing.md)

## Files to Read Before Starting

- `src/timing/iteration.rs` - `NewIterationTiming` struct definition
- `src/sddp/mod.rs` - Current training loop (look for iteration execution)
- `plans/timing-refactor/00-master-plan.md` - Algorithm hot path mapping

## Context

### Background

The training loop currently creates timing structs locally and accumulates timing manually. This ticket introduces `NewIterationTiming` as the single timing struct for each iteration, created at iteration start and passed to forward/backward pass functions.

### Current State

The training loop in `sddp/mod.rs` has patterns like:
```rust
let iteration_start = Instant::now();
let forward_timing_accumulator = ForwardPassTimingAccumulator::default();
let backward_timing_accumulator = BackwardPassTimingAccumulator::default();
// ... forward pass ...
// ... backward pass ...
let iteration_time = iteration_start.elapsed();
```

### Target State

```rust
let timing = NewIterationTiming::new(num_forward_passes);
// ... forward pass uses &timing.forward ...
// ... backward pass uses &timing.backward ...
timing.compute_total();
let output = timing.to_output();
```

## Specification

### Changes Required

1. **Add imports**: Import `NewIterationTiming` from `crate::timing`
2. **Create timing at iteration start**: `let timing = NewIterationTiming::new(num_forward_passes);`
3. **Use total guard**: Wrap entire iteration with `TimingGuard::new(&timing.total)` OR set total manually
4. **Keep forward/backward timing separate**: Don't integrate fully yet (T-018, T-019 do that)

### Important Considerations

- `NewIterationTiming::new()` preallocates trajectory timing Vec
- The timing struct is NOT Send/Sync (uses Cell), but this is fine for single-threaded iteration loop
- Forward/backward passes will receive references to sub-timing structs

## Acceptance Criteria

- [ ] `NewIterationTiming` created at each iteration start
- [ ] `timing.compute_total()` called at iteration end
- [ ] No regression in existing timing values (compare test outputs)
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Add imports

At top of `sddp/mod.rs`:
```rust
use crate::timing::{NewIterationTiming, TimingGuard};
```

### Step 2: Locate iteration loop

Find the main iteration loop (likely `for iteration in 0..num_iterations`).

### Step 3: Create timing struct

At iteration start:
```rust
let timing = NewIterationTiming::new(num_forward_passes);
```

### Step 4: Keep old code working

Don't remove old timing code yet. Keep both old and new timing running in parallel to verify correctness.

### Step 5: Test

Run the full test suite to ensure no regressions.

### Pitfalls to Avoid

- ⚠️ Don't try to make `NewIterationTiming` thread-safe - it uses `Cell` for single-threaded access
- ⚠️ Don't delete old timing code yet - that's T-021
- ⚠️ Ensure `num_forward_passes` matches the actual number of forward passes

## Testing Requirements

### Unit Tests

- Verify `NewIterationTiming::new(n)` creates correct structure
- Verify `compute_total()` sums components correctly

### Integration Tests

```bash
cargo test -p powers-rs sddp
```

### Performance Tests

- Run timing benchmark to ensure no overhead from new struct
- Compare iteration times with baseline

## Documentation Requirements

- [ ] No new documentation needed for this internal change

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple struct instantiation, low risk
