# [T-027] Update sddp/mod.rs to Use backward_pass Module

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-026](./ticket-026-backward-timing-integration.md)
> **Blocks**: Epic 4

---

## ⚠️ CRITICAL: Facade Pattern - Preserve Exact Behavior

This ticket updates `sddp/mod.rs` to use the new `backward_pass` module. We use the **Facade Pattern**: existing backward pass methods become thin wrappers that delegate to the new module.

**The algorithm behavior must remain EXACTLY unchanged.** Cut generation, selection, and application must produce identical results.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:680-1100` - Current backward pass implementation
- `src/sddp/mod.rs:1500-1800` - Training loop backward calls
- `src/algorithm/backward_pass.rs` - New module from T-024
- `src/algorithm/cut_computation.rs` - Cut computation from T-025
- `src/algorithm/context.rs` - Context structs

---

## Context

### Background

The backward pass logic has been extracted to `src/algorithm/backward_pass.rs`. Now we need to update `sddp/mod.rs` to use this new module while maintaining the existing API and parallel execution pattern.

### Strategy: Facade Pattern

The existing backward pass methods will delegate to `algorithm::backward_pass::execute()`:

```rust
// Before: 400+ lines of implementation scattered across methods
// After: Thin wrapper creating context and delegating

fn backward_pass_iteration(...) -> Result<BackwardPassResult, String> {
    let mut ctx = BackwardPassContext::new(...);
    let result = algorithm::backward_pass::execute(&mut ctx)?;
    // Convert result to legacy format if needed
    Ok(result)
}
```

### Parallel Execution Pattern

The current backward pass has complex parallel coordination:

```rust
// Phase 1: Parallel cut computation (per forward pass)
for forward_pass in forward_passes.par_iter() {
    let cuts = compute_cuts_for_trajectory(forward_pass)?;
    cut_buffer.push(cuts);
}

// Phase 2: Sequential FCF updates
for cut in cut_buffer {
    fcf.add_cut(cut)?;
}

// Phase 3: Handler application
apply_handlers()?;
```

**This pattern must be preserved exactly.**

---

## Specification

### Modify `src/sddp/mod.rs`

#### 1. Add Import

```rust
use crate::algorithm::{
    backward_pass, BackwardPassContext, BackwardPassResult as AlgoBackwardResult,
};
```

#### 2. Create Facade for Backward Pass

The exact integration depends on how the backward pass is structured in the training loop. The general pattern is:

```rust
/// Execute backward pass using the new algorithm module.
fn execute_backward_pass_iteration(
    &mut self,
    iteration: usize,
    forward_trajectory_ids: &[Vec<usize>],
    // ... other parameters ...
) -> Result<BackwardPassIterationResult, String> {
    use crate::algorithm::{BackwardPassContext, backward_pass};
    use crate::timing::BackwardTiming;

    // Create timing storage
    let timing = BackwardTiming::default();

    // Prepare backward stage order (reverse of forward)
    let backward_stage_ids: Vec<usize> = self.study_period_ids.iter()
        .rev()
        .copied()
        .collect();

    // Process each forward trajectory
    let mut total_cuts_added = 0usize;
    let mut total_cuts_removed = 0usize;
    let mut lower_bound = 0.0f64;

    for (fwd_idx, trajectory) in self.trajectory_engines.iter_mut().enumerate() {
        // Create context for this trajectory's backward pass
        let mut ctx = BackwardPassContext::new(
            &mut trajectory.subproblem_graph,
            &mut trajectory.realization_graph,
            &self.node_data_graph,
            &self.saa,
            &mut self.fcf,
            self.risk_measure.as_ref(),
            iteration,
            fwd_idx,
            &timing,
            &backward_stage_ids,
        );

        // Execute backward pass
        let result = backward_pass::execute(&mut ctx)?;

        total_cuts_added += result.cuts_added;
        total_cuts_removed += result.cuts_removed;

        // First trajectory gives the lower bound
        if fwd_idx == 0 {
            lower_bound = result.lower_bound;
        }
    }

    // Convert to legacy result format
    let legacy_timing = BackwardPassTimingAccumulator {
        backward_preprocessing_time: timing.preprocessing.get(),
        model_preprocessing_time: timing.model_preprocessing.get(),
        solver_time: timing.solver.get(),
        model_postprocessing_time: timing.model_postprocessing.get(),
        cut_selection_time: timing.cut_selection.get(),
        fcf_state_update_time: timing.fcf_update.get(),
        cut_cloning_time: Duration::ZERO, // Deprecated metric
        handler_application_time: timing.handler_application.get(),
        // ... other fields ...
    };

    Ok(BackwardPassIterationResult {
        lower_bound,
        cuts_added: total_cuts_added,
        cuts_removed: total_cuts_removed,
        timing: legacy_timing,
    })
}
```

#### 3. Preserve Parallel Execution Pattern

If the current code uses `par_iter()` for parallel backward passes, preserve that:

```rust
// If current pattern uses parallel execution:
let results: Vec<_> = self.trajectory_engines
    .par_iter_mut()
    .enumerate()
    .map(|(fwd_idx, trajectory)| {
        let ctx = BackwardPassContext::new(/* ... */);
        backward_pass::execute(&mut ctx)
    })
    .collect::<Result<Vec<_>, _>>()?;

// Aggregate results sequentially
let total_cuts_added: usize = results.iter().map(|r| r.cuts_added).sum();
// ... etc.
```

#### 4. Remove Duplicate Code

After the facade is verified working:

1. Mark old helper functions as `#[deprecated]`
2. Remove them after confirming golden tests pass
3. Clean up unused timing structs

---

## Acceptance Criteria

- [ ] Backward pass delegates to `algorithm::backward_pass::execute()`
- [ ] Context created correctly with all required fields
- [ ] Timing converted from new format to legacy format
- [ ] Parallel execution pattern preserved exactly
- [ ] All existing call sites work without modification
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (CRITICAL)

### Critical Verification

- [ ] Same cuts generated
- [ ] Same lower bounds computed
- [ ] Same FCF state after training
- [ ] Parallel behavior unchanged

---

## Implementation Guide

### Suggested Approach

1. **Understand current backward pass structure**:
   ```bash
   grep -n "backward\|cut\|fcf" src/sddp/mod.rs | head -50
   ```

2. **Identify the main backward pass entry point** in training loop

3. **Create backup** of original implementation

4. **Implement facade** incrementally:
   - Start with a single trajectory (no parallelism)
   - Verify golden tests
   - Add parallel execution
   - Verify golden tests again

5. **Handle timing conversion**:
   - Map new timing fields to old accumulator fields
   - Ensure no timing values are lost

6. **Run golden tests** after every change

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Update backward pass to use facade |

### Timing Conversion

| New (`BackwardTiming`) | Old (`BackwardPassTimingAccumulator`) |
|------------------------|---------------------------------------|
| `preprocessing` | `backward_preprocessing_time` |
| `model_preprocessing` | `model_preprocessing_time` |
| `solver` | `solver_time` |
| `model_postprocessing` | `model_postprocessing_time` |
| `cut_computation` | (part of model_postprocessing) |
| `cut_selection` | `cut_selection_time` |
| `fcf_update` | `fcf_state_update_time` |
| `handler_application` | `handler_application_time` |
| N/A | `cut_cloning_time` (deprecated) |

### Pitfalls to Avoid

- ⚠️ Do NOT change the parallel execution pattern
- ⚠️ Do NOT change cut ordering
- ⚠️ Do NOT change FCF update sequence
- ⚠️ Preserve exact error handling
- ⚠️ Keep original implementation available until verified

---

## Testing Requirements

### Integration Tests

- [ ] Training loop works correctly
- [ ] Multiple forward passes handled correctly
- [ ] Parallel execution works correctly

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Compare cuts generated before/after
- [ ] Compare lower bounds before/after

### Performance Tests

- [ ] `cargo bench` shows no significant regression
- [ ] Parallel scaling unchanged

---

## Documentation Requirements

- [ ] Update method docs to note delegation
- [ ] Add `#[deprecated]` notices to old helper functions
- [ ] Document timing field mapping

---

## Rollback Plan

If golden tests fail and you can't fix the issue:

1. **Revert the facade**:
   - Uncomment original implementation
   - Remove facade code

2. **Report the issue**:
   - Document what failed
   - Include error messages or diff output
   - Ask for clarification

3. **Do NOT proceed** with broken golden tests

---

## Effort Estimate

**Points**: 3
**Confidence**: Low-Medium
**Rationale**: Complex integration with parallel execution; many moving parts

---

## Definition of Done

- [ ] Backward pass uses facade pattern
- [ ] Context creation correct
- [ ] Timing conversion correct
- [ ] Parallel execution preserved
- [ ] All tests pass
- [ ] **Golden tests pass**
- [ ] No API changes
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Old code cleaned up
