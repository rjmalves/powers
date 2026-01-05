# [T-013] Update processor.rs timing

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: [T-012](./ticket-012-update-backward-pass-timing-types.md)  
> **Blocks**: [T-014](./ticket-014-update-coordinator-phase-timing.md)

## Files to Read Before Starting

- `src/algorithm/processor.rs` - `CutComputationTiming` and trait definitions
- `src/timing/backward.rs` - `BackwardPhase1Timing` 
- `src/algorithm/coordinator.rs` - Implements `BackwardStageProcessor`

## Context

### Background

The `processor.rs` module defines timing types used by the `BackwardStageProcessor` trait:
- `CutComputationTiming` - returned from Phase 1
- `FirstStageTiming` - returned from first stage evaluation
- `Phase1Result` and `Phase2Result` - contain timing

This ticket aligns these types with the new timing hierarchy.

### Current State

```rust
// processor.rs lines 38-49
pub struct CutComputationTiming {
    pub model_preprocessing: Duration,
    pub solver: Duration,
    pub model_postprocessing: Duration,
    pub solver_calls: usize,
}
```

### Target Alignment

The new `BackwardPhase1Timing` has similar fields but uses `Cell<Duration>`. We have two options:

1. **Keep `CutComputationTiming`** as a plain Duration struct for return values, convert to `BackwardPhase1Timing` at accumulation
2. **Replace with `BackwardPhase1Timing`** and extract values when building return

Option 1 is cleaner - keep plain Duration for return values (matches output pattern).

## Specification

### Decision: Keep CutComputationTiming

Keep `CutComputationTiming` as-is (uses plain `Duration`). It's used for returning timing from Phase 1 methods. The caller (`backward_pass::execute`) converts it to `BackwardPhase1Timing` for accumulation.

### Changes Required

1. **No changes to CutComputationTiming** - keep as plain Duration struct
2. **Add conversion helper** in backward_pass.rs if needed
3. **Document the pattern**: Return types use Duration, accumulators use Cell<Duration>

### Conversion Pattern

In `backward_pass.rs::execute_stage()`:

```rust
// Phase 1 returns CutComputationTiming (plain Duration)
let phase1 = processor.compute_cuts_parallel_into_slots(stage_ctx, fcf_graph)?;

// Accumulate into NewBackwardTiming (Cell<Duration>)
timing.phase1.model_preprocessing.set(
    timing.phase1.model_preprocessing.get() + phase1.timing.model_preprocessing
);
timing.phase1.solver.set(
    timing.phase1.solver.get() + phase1.timing.solver
);
timing.phase1.model_postprocessing.set(
    timing.phase1.model_postprocessing.get() + phase1.timing.model_postprocessing
);
timing.add_solver_calls(phase1.timing.solver_calls);
```

## Acceptance Criteria

- [ ] `CutComputationTiming` remains as plain Duration struct (no Cell)
- [ ] Conversion from `CutComputationTiming` to `BackwardPhase1Timing` documented
- [ ] `Phase1Result` and `Phase2Result` unchanged (return types)
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Verify current usage

```bash
grep -n "CutComputationTiming" src/algorithm/
```

### Step 2: Document the pattern

Add a doc comment to `CutComputationTiming`:

```rust
/// Timing from Phase 1 cut computation.
///
/// This struct uses plain `Duration` (not `Cell<Duration>`) because it's
/// a return value from processor methods. The caller accumulates these
/// values into `timing::NewBackwardTiming` which uses `Cell<Duration>`.
///
/// # Conversion
///
/// To accumulate into `NewBackwardTiming`:
/// ```ignore
/// timing.phase1.model_preprocessing.set(
///     timing.phase1.model_preprocessing.get() + cut_timing.model_preprocessing
/// );
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CutComputationTiming { ... }
```

### Step 3: Keep FirstStageTiming

`FirstStageTiming` is simple and local - keep it as-is:

```rust
pub struct FirstStageTiming {
    pub solver: Duration,
    pub state_extraction: Duration,
}
```

### Pitfalls to Avoid

- ⚠️ Don't convert `CutComputationTiming` to use `Cell<Duration>` - it's a return type
- ⚠️ The trait `BackwardStageProcessor` returns `CutComputationTiming` - changing it affects all implementors

## Testing Requirements

### Compilation Test

```bash
cargo build -p powers-rs
```

### Trait Verification

Ensure `ParallelHandlerCoordinator` still implements `BackwardStageProcessor` correctly.

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Mostly documentation; no structural changes needed
