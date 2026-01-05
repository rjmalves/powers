# [T-012] Update backward_pass.rs timing types

> **Epic**: [Epic 3: Backward Pass Integration](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: Epic 1 complete  
> **Blocks**: [T-013](./ticket-013-update-processor-timing.md)

## Files to Read Before Starting

- `src/timing/backward.rs` - New `NewBackwardTiming` definition
- `src/algorithm/backward_pass.rs` - Current backward pass with `BackwardPassTimingAccumulator`
- `plans/timing-refactor/00-master-plan.md` - Architecture overview

## Context

### Background

The backward pass currently defines `BackwardPassTimingAccumulator` locally (lines 35-119 in `backward_pass.rs`). Epic 1 created `NewBackwardTiming` in `src/timing/backward.rs` with a cleaner hierarchical structure. This ticket replaces the local accumulator with the new timing types.

### Current State

```rust
// src/algorithm/backward_pass.rs
pub struct BackwardPassTimingAccumulator {
    pub preprocessing: Cell<Duration>,
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub model_postprocessing: Cell<Duration>,
    pub cut_selection: Cell<Duration>,
    pub fcf_state_update: Cell<Duration>,
    pub cut_cloning: Cell<Duration>,
    pub handler_application: Cell<Duration>,
    pub solver_calls: Cell<usize>,
}
```

### Target State

Replace with `timing::NewBackwardTiming` which has:
- `phase1.model_preprocessing`, `phase1.solver`, `phase1.model_postprocessing`
- `phase2.cut_selection`
- `phase3.problem_update` (consolidates fcf_state_update + cut_cloning + handler_application)
- `total`, `solver_calls`

## Specification

### Changes Required

1. **Add import**: `use crate::timing::NewBackwardTiming;`
2. **Update execute() signature**: Change `timing: &BackwardPassTimingAccumulator` to `timing: &NewBackwardTiming`
3. **Update timing accumulation**: Map old field accesses to new hierarchy
4. **Remove BackwardPassTimingAccumulator**: After all usages are updated
5. **Remove BackwardPassTimingSnapshot**: Replace with `BackwardTimingOutput`

### Field Mapping

| Old Field | New Field |
|-----------|-----------|
| `preprocessing` | Remove (not needed) |
| `model_preprocessing` | `phase1.model_preprocessing` |
| `solver` | `phase1.solver` |
| `model_postprocessing` | `phase1.model_postprocessing` |
| `cut_selection` | `phase2.cut_selection` |
| `fcf_state_update` | `phase3.problem_update` (combined) |
| `cut_cloning` | `phase3.problem_update` (combined) |
| `handler_application` | `phase3.problem_update` (combined) |
| `solver_calls` | `solver_calls` |

## Acceptance Criteria

- [ ] `backward_pass.rs` imports `NewBackwardTiming` from `timing/`
- [ ] `execute()` accepts `&NewBackwardTiming`
- [ ] Timing accumulation uses new field paths
- [ ] No compilation errors
- [ ] Existing tests adapted or removed
- [ ] `cargo test` passes
- [ ] `cargo clippy` clean

## Implementation Guide

### Step 1: Keep old types temporarily

Don't delete `BackwardPassTimingAccumulator` yet. First, update the execute function to work with both.

### Step 2: Update imports

Add at top of `backward_pass.rs`:
```rust
use crate::timing::{NewBackwardTiming, BackwardTimingOutput};
```

### Step 3: Create adapter in execute()

Initially, you can keep the old signature and adapt internally:
```rust
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,
    timing: &BackwardPassTimingAccumulator,  // Keep for now
    fcf_graph: &mut DirectedGraph<FutureCostFunction>,
) -> Result<BackwardPassResult, String>
```

### Step 4: Update timing accumulation

Replace:
```rust
BackwardPassTimingAccumulator::add_duration(&timing.solver, phase1.timing.solver);
```

With:
```rust
timing.phase1.solver.set(timing.phase1.solver.get() + phase1.timing.solver);
```

Or use the helper from `BackwardPhase1Timing`:
```rust
// If converting phase1.timing to BackwardPhase1Timing
timing.phase1.add(&converted_timing);
```

### Step 5: Combine Phase 3 timing

The old code has separate fields for `fcf_state_update`, `cut_cloning`, `handler_application`. Combine them into `phase3.problem_update`:

```rust
// Old
BackwardPassTimingAccumulator::add_duration(&timing.fcf_state_update, phase2.fcf_update_time);
BackwardPassTimingAccumulator::add_duration(&timing.handler_application, handler_time);

// New
let phase3_total = phase2.fcf_update_time + handler_time;
timing.phase3.problem_update.set(timing.phase3.problem_update.get() + phase3_total);
```

### Pitfalls to Avoid

- ⚠️ Don't remove `BackwardPassTimingAccumulator` until T-015
- ⚠️ The new `BackwardPhase1Timing` has a `cut_computation` field - decide if you need it
- ⚠️ Watch for callers of `execute()` - they need to pass correct type

## Testing Requirements

### Unit Tests

Update tests in `backward_pass.rs` to use new types:
- `test_timing_accumulator_default` → test `NewBackwardTiming::new()`
- `test_timing_accumulator_increment_solver_calls` → test `add_solver_calls()`
- `test_timing_accumulator_add_duration` → test phase1 accumulation

### Integration Tests

```bash
cargo test -p powers-rs backward_pass
```

## Effort Estimate

**Points**: 3  
**Confidence**: Medium  
**Rationale**: Need to trace all timing accumulation points and update mappings
