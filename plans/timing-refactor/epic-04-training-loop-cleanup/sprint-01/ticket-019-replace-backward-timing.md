# [T-019] Replace backward timing code with guards

> **Epic**: [Epic 4: Training Loop Cleanup](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-017](./ticket-017-create-iteration-timing.md)  
> **Blocks**: [T-020](./ticket-020-update-iteration-result.md)

## Files to Read Before Starting

- `src/timing/backward.rs` - `NewBackwardTiming` and phase structs
- `src/timing/guard.rs` - `TimingGuard` RAII pattern
- `src/sddp/mod.rs` - Current backward pass timing code
- `plans/timing-refactor/00-master-plan.md` - Backward pass execution flow

## Context

### Background

The backward pass timing is currently scattered between `sddp/mod.rs` and `algorithm/backward_pass.rs`. Epic 3 already updated the algorithm layer to use new types. This ticket updates the training loop to use `TimingGuard` for backward pass total time and ensures timing flows correctly from the algorithm layer.

### Current State

Backward pass timing in training loop:
```rust
let backward_start = Instant::now();
let backward_result = backward_pass::execute(...);
backward_timing.total_time = backward_start.elapsed();
```

### Target State

```rust
{
    let _guard = TimingGuard::new(&timing.backward.total);
    let backward_result = backward_pass::execute(&timing.backward, ...);
}
// Phase timing already accumulated inside execute()
```

## Specification

### Changes Required

1. **Total time guard**: Use `TimingGuard` with `timing.backward.total`
2. **Pass timing reference**: Pass `&timing.backward` to `backward_pass::execute()`
3. **Remove manual timing**: Remove `Instant::now()` / `.elapsed()` for backward pass
4. **Verify phase accumulation**: Ensure phase1/2/3 timing accumulates correctly from Epic 3

### Timing Field Mapping

| Old Code | New Code |
|----------|----------|
| `backward_start.elapsed()` | `TimingGuard::new(&timing.backward.total)` |
| `backward_preprocessing_time` | Removed (was misnamed) |
| `model_preprocessing_time` | `timing.backward.phase1.model_preprocessing` |
| `solver_time` | `timing.backward.phase1.solver` |
| `model_postprocessing_time` | `timing.backward.phase1.model_postprocessing` |
| `cut_selection_time` | `timing.backward.phase2.cut_selection` |
| `fcf_state_update_time` | `timing.backward.phase3.problem_update` (combined) |
| `cut_cloning_time` | `timing.backward.phase3.problem_update` (combined) |
| `handler_application_time` | `timing.backward.phase3.problem_update` (combined) |

## Acceptance Criteria

- [ ] No `Instant::now()` for backward pass timing in training loop
- [ ] Backward pass total time uses `TimingGuard`
- [ ] `&timing.backward` passed to `backward_pass::execute()` (or equivalent)
- [ ] Phase timing accumulated correctly
- [ ] Backward timing values match previous (within tolerance for combined fields)
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Replace backward total timing

```rust
// OLD
let backward_start = Instant::now();
let backward_result = backward_pass::execute(processor, ctx, &backward_accum, fcf_graph)?;
backward_timing.total_time = backward_start.elapsed();

// NEW
{
    let _guard = TimingGuard::new(&timing.backward.total);
    let backward_result = backward_pass::execute(
        processor,
        ctx,
        &timing.backward,  // Pass new timing type
        fcf_graph
    )?;
}
```

### Step 2: Verify execute() signature

Confirm that `backward_pass::execute()` now accepts `&NewBackwardTiming` (from Epic 3).

If not yet updated, add compatibility:
```rust
// In sddp/mod.rs, temporarily adapt
let backward_result = backward_pass::execute(...);
// Copy timing from result into timing.backward
timing.backward.phase1.model_preprocessing.set(backward_result.timing.model_preprocessing);
// etc.
```

### Step 3: Remove old accumulator

Remove creation of `BackwardPassTimingAccumulator` from training loop.

### Pitfalls to Avoid

- ⚠️ The phase3 fields are combined - don't expect separate fcf_state_update, cut_cloning, etc.
- ⚠️ Verify Epic 3 is complete before this ticket
- ⚠️ `backward_preprocessing_time` is removed entirely (was misnamed)

## Testing Requirements

### Unit Tests

- Test that phase timing accumulates correctly
- Test that total equals sum of phases

### Integration Tests

```bash
cargo test -p powers-rs backward_pass
cargo test -p powers-rs training
```

### Validation Tests

- Run full training and compare backward timing output
- Verify solver time is in expected range

## Documentation Requirements

- [ ] No external documentation needed

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Depends on Epic 3 completion, phase timing consolidation requires verification
