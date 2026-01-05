# [T-018] Replace forward timing code with guards

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-017](./ticket-017-create-iteration-timing.md)  
> **Blocks**: [T-020](./ticket-020-update-iteration-result.md)

## Files to Read Before Starting

- `src/timing/forward.rs` - `NewForwardTiming` and sub-structs
- `src/timing/guard.rs` - `TimingGuard` RAII pattern
- `src/sddp/mod.rs` - Current forward pass timing code
- `plans/timing-refactor/00-master-plan.md` - Forward pass execution flow

## Context

### Background

The forward pass has timing code scattered through the training loop: SAA sampling, model preprocessing, solver, model postprocessing, and postprocessing phases. This ticket replaces all manual `Instant::now()` / `.elapsed()` patterns with `TimingGuard` using the new timing hierarchy.

### Current State

Forward pass timing in `sddp/mod.rs`:
```rust
let saa_start = Instant::now();
sample_scenarios();
forward_timing.saa_sampling_time = saa_start.elapsed();

// In parallel section
let model_pre_start = Instant::now();
// ... work ...
timing_accum.model_preprocessing_time += model_pre_start.elapsed();
```

### Target State

```rust
{
    let _guard = TimingGuard::new(&timing.forward.preprocessing.saa_sampling);
    sample_scenarios();
}

// In parallel section, accumulate into trajectory timing
timing.forward.parallel.trajectories[traj_idx].model_preprocessing.set(
    timing.forward.parallel.trajectories[traj_idx].model_preprocessing.get() + elapsed
);
```

## Specification

### Changes Required

1. **SAA sampling**: Use `TimingGuard` with `timing.forward.preprocessing.saa_sampling`
2. **Parallel wall time**: Use `TimingGuard` with `timing.forward.parallel.wall`
3. **Per-trajectory timing**: Accumulate into `timing.forward.parallel.trajectories[idx]`
4. **Postprocessing**: Use `TimingGuard` with `timing.forward.postprocessing.detail_capturing`
5. **Compute aggregates**: Call `timing.forward.parallel.compute_aggregates()` after parallel section
6. **Compute total**: Call `timing.forward.compute_total()` after all forward work

### Timing Field Mapping

| Old Code | New Code |
|----------|----------|
| `saa_start.elapsed()` → `saa_sampling_time` | `TimingGuard::new(&timing.forward.preprocessing.saa_sampling)` |
| Parallel section wall time | `TimingGuard::new(&timing.forward.parallel.wall)` |
| Per-handler model preprocessing | `trajectories[idx].model_preprocessing` |
| Per-handler solver | `trajectories[idx].solver` |
| Per-handler model postprocessing | `trajectories[idx].model_postprocessing` |
| `forward_postprocessing_time` | `TimingGuard::new(&timing.forward.postprocessing.detail_capturing)` |

## Acceptance Criteria

- [ ] No `Instant::now()` for forward pass timing in training loop
- [ ] All forward timing uses `TimingGuard` or accumulates into new types
- [ ] `compute_aggregates()` called after parallel section
- [ ] `compute_total()` called after forward pass complete
- [ ] Forward timing values match previous implementation (within tolerance)
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Replace SAA sampling timing

```rust
// OLD
let saa_start = Instant::now();
sample_saa_scenarios(...);
let saa_time = saa_start.elapsed();

// NEW
{
    let _guard = TimingGuard::new(&timing.forward.preprocessing.saa_sampling);
    sample_saa_scenarios(...);
}
```

### Step 2: Replace parallel wall timing

```rust
// OLD
let parallel_start = Instant::now();
handlers.par_iter_mut().for_each(...);
let parallel_wall = parallel_start.elapsed();

// NEW
{
    let _guard = TimingGuard::new(&timing.forward.parallel.wall);
    handlers.par_iter_mut()
        .zip(timing.forward.parallel.trajectories.par_iter())
        .for_each(|(handler, traj_timing)| {
            forward_trajectory(handler, traj_timing);
        });
}
```

### Step 3: Update per-trajectory timing

Inside the parallel forward trajectory:
```rust
// OLD
accum.model_preprocessing_time += elapsed;

// NEW
traj_timing.model_preprocessing.set(
    traj_timing.model_preprocessing.get() + elapsed
);
```

### Step 4: Compute aggregates and total

After parallel section:
```rust
timing.forward.parallel.compute_aggregates();
timing.forward.compute_total();
```

### Pitfalls to Avoid

- ⚠️ `TrajectoryTiming` uses `Cell<Duration>` - use `.get()` and `.set()`, not `+=`
- ⚠️ Parallel iteration must zip with trajectory timing slice
- ⚠️ Don't forget to call `compute_aggregates()` - output will be wrong otherwise

## Testing Requirements

### Unit Tests

- Test `compute_aggregates()` with known values
- Test that avg/max are computed correctly

### Integration Tests

```bash
cargo test -p powers-rs forward_pass
cargo test -p powers-rs training
```

### Validation Tests

- Compare timing output with baseline run
- Ensure solver_avg, solver_max are reasonable

## Documentation Requirements

- [ ] No external documentation needed

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Multiple timing points to update, parallel section requires careful handling
