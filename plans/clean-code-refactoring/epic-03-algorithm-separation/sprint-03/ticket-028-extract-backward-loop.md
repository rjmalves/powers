# [T-028] Extract Backward Pass Loop

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: Sprint 2 complete
> **Blocks**: [T-029](./ticket-029-extract-cut-computation.md)

---

## ⚠️ CRITICAL: Algorithm Logic Unchanged

This ticket extracts the backward pass loop. **The algorithm behavior must remain IDENTICAL.** Run golden tests after EVERY change.

---

## ⚠️ ARCHITECTURAL DECISION: Timing Separation

Per the pattern established in T-021:

**Timing is passed as a separate parameter, not inside context structs.**

```rust
// CORRECT:
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,           // No timing inside
    timing: &BackwardPassTimingAccumulator,  // Separate parameter
) -> Result<BackwardPassResult, String>
```

This enables `TimingGuard` to work without borrow conflicts.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:1762-2054` - Current backward pass loop
- `src/algorithm/coordinator.rs` - `ParallelHandlerCoordinator`
- `src/algorithm/processor.rs` - `BackwardStageProcessor` trait
- `src/algorithm/context.rs` - `BackwardPassContext`, `BackwardStageContext`
- [T-021](../sprint-01/ticket-021-forward-timing-integration.md) - Timing separation pattern

---

## Specification

### Create `src/algorithm/backward_pass.rs`

```rust
//! Backward pass execution for SDDP algorithm.
//!
//! The backward pass iterates through stages in reverse order, computing
//! Benders cuts at each stage and updating the future cost function.
//!
//! # Timing Separation
//!
//! Timing is passed as a separate parameter (not inside context) to enable
//! `TimingGuard` usage without borrow conflicts. See T-021 for rationale.

use crate::algorithm::context::{BackwardPassContext, BackwardPassResult};
use crate::algorithm::processor::BackwardStageProcessor;
use crate::timing::TimingGuard;
use std::time::Duration;
use std::cell::Cell;

/// Timing accumulator for backward pass.
///
/// Uses `Cell<Duration>` for interior mutability, enabling `TimingGuard`.
#[derive(Debug, Clone, Default)]
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

impl BackwardPassTimingAccumulator {
    pub fn increment_solver_calls(&self, count: usize) {
        self.solver_calls.set(self.solver_calls.get() + count);
    }
    
    /// Add duration to a timing field.
    pub fn add_duration(field: &Cell<Duration>, duration: Duration) {
        field.set(field.get() + duration);
    }
}

/// Execute the backward pass using the provided processor.
///
/// # Arguments
///
/// * `processor` - Implementation of `BackwardStageProcessor`
/// * `ctx` - Backward pass context (no timing inside)
/// * `timing` - Timing accumulator (separate from context for TimingGuard)
///
/// # Returns
///
/// * `Ok(BackwardPassResult)` - Lower bound, cut statistics
/// * `Err(String)` - If any stage fails
pub fn execute<P: BackwardStageProcessor>(
    processor: &mut P,
    ctx: &BackwardPassContext,
    timing: &BackwardPassTimingAccumulator,
) -> Result<BackwardPassResult, String> {
    let mut result = BackwardPassResult::new(0.0, 0, 0, 0, 0);
    
    for stage_idx in ctx.backward_stage_indices() {
        let stage_ctx = ctx.stage_context(stage_idx)
            .ok_or_else(|| format!("Invalid stage index {}", stage_idx))?;
        
        if stage_ctx.is_first_stage() {
            // First stage: evaluate bound only
            let (lb, first_timing) = processor.eval_first_stage_bound(&stage_ctx)?;
            result.lower_bound = lb;
            
            BackwardPassTimingAccumulator::add_duration(
                &timing.solver, 
                first_timing.solver
            );
            BackwardPassTimingAccumulator::add_duration(
                &timing.model_postprocessing,
                first_timing.state_extraction
            );
            timing.increment_solver_calls(
                stage_ctx.branching_count().unwrap_or(1) * processor.num_forward_passes()
            );
        } else {
            execute_stage(processor, &stage_ctx, &mut result, timing)?;
        }
    }
    
    Ok(result)
}

/// Execute a single stage of the backward pass.
fn execute_stage<P: BackwardStageProcessor>(
    processor: &mut P,
    stage_ctx: &BackwardStageContext,
    result: &mut BackwardPassResult,
    timing: &BackwardPassTimingAccumulator,
) -> Result<(), String> {
    // Phase 1: Parallel cut computation
    let phase1 = {
        // Note: TimingGuard works because timing is separate from processor/stage_ctx
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        processor.compute_cuts_parallel(stage_ctx)?
    };
    
    // Accumulate phase 1 timing (returned from parallel computation)
    BackwardPassTimingAccumulator::add_duration(&timing.solver, phase1.timing.solver);
    BackwardPassTimingAccumulator::add_duration(
        &timing.model_postprocessing, 
        phase1.timing.model_postprocessing
    );
    timing.increment_solver_calls(phase1.timing.solver_calls);
    
    // Phase 2: Sequential cut selection
    let phase2 = {
        let _guard = TimingGuard::new(&timing.cut_selection);
        processor.select_cuts_batch(phase1.cut_data, stage_ctx)?
    };
    
    // Accumulate phase 2 timing
    BackwardPassTimingAccumulator::add_duration(&timing.fcf_state_update, phase2.fcf_update_time);
    BackwardPassTimingAccumulator::add_duration(&timing.cut_cloning, phase2.cut_cloning_time);
    
    // Update result counts
    result.cuts_added += phase2.batch_result.new_cut_ids.len();
    result.cuts_removed += phase2.batch_result.removing_cut_ids.len();
    result.cuts_returned += phase2.batch_result.returning_cut_ids.len();
    
    // Phase 3: Parallel cut application
    {
        let _guard = TimingGuard::new(&timing.handler_application);
        processor.apply_cuts_parallel(&phase2, stage_ctx)?;
    }
    
    Ok(())
}
```

---

## Acceptance Criteria

- [x] `src/algorithm/backward_pass.rs` created
- [x] `BackwardPassTimingAccumulator` uses `Cell<Duration>` for all fields
- [x] `execute()` takes timing as separate parameter (not inside context)
- [x] Timing accumulation pattern (coordinator returns, module accumulates)
- [x] Module exported in `src/algorithm/mod.rs`
- [x] `cargo build -j1` succeeds
- [x] Function compiles with `ParallelHandlerCoordinator`

---

## Key Files to Create/Modify

| File | Action | Status |
|------|--------|--------|
| `src/algorithm/backward_pass.rs` | CREATE | ✅ |
| `src/algorithm/mod.rs` | UPDATE exports | ✅ |

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Core extraction with timing integration

---

## Definition of Done

- [x] Backward pass module created
- [x] Timing uses `Cell<Duration>` for accumulation
- [x] Timing passed as separate parameter
- [x] Module exported
- [ ] Code reviewed
