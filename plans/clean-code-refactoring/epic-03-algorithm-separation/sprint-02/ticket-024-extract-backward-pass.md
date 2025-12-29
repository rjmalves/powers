# [T-024] Extract Backward Pass Logic

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-023](./ticket-023-backward-pass-context.md)
> **Blocks**: [T-025](./ticket-025-extract-cut-computation.md)

---

## ⚠️ CRITICAL: Exact Behavioral Preservation

This ticket extracts the backward pass logic from `sddp/mod.rs` to `algorithm/backward_pass.rs`. **The algorithm logic must remain EXACTLY unchanged.**

The backward pass is the **most critical** part of the SDDP algorithm—it generates the Benders cuts that drive convergence. Any change to cut computation, selection, or application can cause:
- Different cuts generated
- Different convergence behavior
- Different final solutions

Run golden tests after EVERY significant change. If ANY test fails or output differs, **STOP IMMEDIATELY**.

---

## Files to Read Before Starting

- `src/sddp/mod.rs:680-1100` - Current backward pass implementation
- `src/sddp/mod.rs:688-820` - `compute_cut_data_for_backward_step()`
- `src/sddp/mod.rs:2540-2600` - Helper functions (`reuse_forward_basis`, etc.)
- `src/algorithm/context.rs` - `BackwardPassContext` from T-023
- `src/fcf.rs` - FCF and CutData types

---

## Context

### Background

The backward pass iterates through stages in reverse order, computing Benders cuts at each stage. The cut computation involves:

1. **Branching**: Generating multiple scenarios at each stage
2. **Solving**: Solving the subproblem for each branching
3. **Cut generation**: Computing cut coefficients from dual values
4. **Cut selection**: Choosing which cuts to add
5. **FCF update**: Adding selected cuts to the future cost function

### Current Structure

The backward pass is distributed across several methods in `sddp/mod.rs`:

```rust
// Main backward logic (in training loop)
for stage in stages.iter().rev() {
    // Phase 1: Compute cuts (can be parallel)
    let cut_data = compute_cut_data_for_backward_step(...)?;
    
    // Phase 2: Update FCF (sequential)
    fcf.add_cut(cut_data)?;
    
    // Phase 3: Apply handlers
    apply_handlers(...)?;
}

// Helper method
fn compute_cut_data_for_backward_step(...) -> Result<(CutData, Timing), String>
```

### Target Structure

```rust
// src/algorithm/backward_pass.rs
pub fn execute(ctx: &mut BackwardPassContext) -> Result<BackwardPassResult, String>

// Extracted functions
fn execute_stage_backward(...) -> Result<StageCutResult, String>
fn compute_branching_solutions(...) -> Result<Vec<Realization>, String>
```

---

## Specification

### Create `src/algorithm/backward_pass.rs`

```rust
//! Backward pass execution for SDDP algorithm.
//!
//! The backward pass iterates through stages in reverse order, computing
//! Benders cuts that approximate the expected future cost function.
//!
//! # Algorithm
//!
//! For each stage (in reverse order):
//! 1. **Branching**: Generate scenarios for cut computation
//! 2. **Solve**: Solve subproblem for each branching scenario
//! 3. **Cut computation**: Compute cut coefficients from duals
//! 4. **FCF update**: Add cut to future cost function
//!
//! # Thread Safety
//!
//! Phase 1 (branching solves) can be parallelized within a stage.
//! Phase 2 (FCF update) must be sequential (critical section).
//!
//! # Current Status
//!
//! This module extracts logic from `sddp/mod.rs`. The parallel execution
//! pattern is preserved exactly as in the original.

use crate::algorithm::context::{BackwardPassContext, BackwardPassResult, BackwardStageTiming};
use crate::fcf::CutData;
use crate::subproblem::Realization;
use std::time::Instant;

/// Execute a backward pass using the provided context.
///
/// Iterates through stages in reverse order, computing and adding Benders cuts.
///
/// # Arguments
///
/// * `ctx` - Mutable reference to backward pass context
///
/// # Returns
///
/// * `Ok(BackwardPassResult)` - Lower bound, cuts added, timing
/// * `Err(String)` - Error if any stage fails
///
/// # Errors
///
/// Returns an error if:
/// - A subproblem node is not found
/// - Branching solve fails
/// - Cut computation fails
/// - FCF update fails
pub fn execute(ctx: &mut BackwardPassContext) -> Result<BackwardPassResult, String> {
    let mut total_timing = BackwardStageTiming::default();
    let mut cuts_added = 0usize;
    let mut cuts_removed = 0usize;
    let mut lower_bound = 0.0f64;

    // Process stages in backward order
    for (stage_idx, &stage_id) in ctx.backward_stage_ids.iter().enumerate() {
        let stage_result = execute_stage_backward(
            ctx,
            stage_idx,
            stage_id,
            &mut total_timing,
        )?;

        cuts_added += stage_result.cuts_added;
        cuts_removed += stage_result.cuts_removed;

        // First stage gives the lower bound
        if stage_idx == ctx.backward_stage_ids.len() - 1 {
            lower_bound = stage_result.lower_bound;
        }
    }

    Ok(BackwardPassResult::new(
        lower_bound,
        cuts_added,
        cuts_removed,
        total_timing.solver_calls,
    ))
}

/// Result of processing a single stage in backward pass.
#[derive(Debug, Clone)]
struct StageResult {
    /// Lower bound contribution from this stage (only valid for first stage).
    lower_bound: f64,
    /// Number of cuts added at this stage.
    cuts_added: usize,
    /// Number of cuts removed at this stage.
    cuts_removed: usize,
}

/// Execute backward pass for a single stage.
///
/// This function handles:
/// 1. Getting the forward trajectory state
/// 2. Solving branching scenarios
/// 3. Computing the cut
/// 4. Updating the FCF
fn execute_stage_backward(
    ctx: &mut BackwardPassContext,
    stage_idx: usize,
    stage_id: usize,
    timing: &mut BackwardStageTiming,
) -> Result<StageResult, String> {
    // NOTE: This is a simplified skeleton. The actual implementation
    // must match the logic in sddp/mod.rs exactly.
    
    let prep_start = Instant::now();

    // Get node data for this stage
    let node_data = ctx
        .node_data_graph
        .get_node(stage_id)
        .ok_or_else(|| format!("Could not find node data for stage {}", stage_id))?;

    // Get past node IDs for trajectory lookup
    let past_node_ids = &node_data.data.past_ids;

    timing.model_preprocessing += prep_start.elapsed();

    // Compute cut data (this includes branching solves)
    let (cut_data, solve_timing) = compute_cut_data(
        ctx,
        stage_id,
        past_node_ids,
    )?;

    timing.solver += solve_timing.solver;
    timing.model_postprocessing += solve_timing.model_postprocessing;
    timing.cut_computation += solve_timing.cut_computation;
    timing.solver_calls += solve_timing.solver_calls;

    // Update FCF with the new cut
    let fcf_start = Instant::now();
    let (cuts_added, cuts_removed) = ctx.fcf.add_cut(cut_data)?;
    // Note: Actual FCF API may differ - adjust as needed
    timing.cut_computation += fcf_start.elapsed();

    // Compute lower bound if first stage
    let lower_bound = if stage_idx == ctx.backward_stage_ids.len() - 1 {
        // First stage evaluation
        // TODO: Match exact logic from sddp/mod.rs
        0.0
    } else {
        0.0
    };

    Ok(StageResult {
        lower_bound,
        cuts_added,
        cuts_removed,
    })
}

/// Compute cut data for a backward step.
///
/// This is extracted from `sddp/mod.rs::compute_cut_data_for_backward_step`.
fn compute_cut_data(
    ctx: &mut BackwardPassContext,
    stage_id: usize,
    past_node_ids: &[usize],
) -> Result<(CutData, BackwardStageTiming), String> {
    // TODO: Extract exact logic from sddp/mod.rs:688-820
    // This is a placeholder - must match original exactly
    
    todo!("Extract compute_cut_data_for_backward_step from sddp/mod.rs")
}

/// Reuse forward basis for warm-starting backward solves.
///
/// Extracted from `sddp/mod.rs::reuse_forward_basis`.
fn reuse_forward_basis(
    subproblem: &mut crate::subproblem::Subproblem,
    forward_realization: &Realization,
) -> Result<(), String> {
    // TODO: Copy exact logic from sddp/mod.rs:2543-2565
    todo!("Extract reuse_forward_basis from sddp/mod.rs")
}

/// Evaluate first stage bound from branching results.
///
/// Extracted from `sddp/mod.rs::eval_first_stage_bound`.
fn eval_first_stage_bound(
    branching_realizations: &[Realization],
    risk_measure: &dyn crate::risk_measure::RiskMeasure,
) -> Result<f64, String> {
    // TODO: Copy exact logic from sddp/mod.rs:2567-2580
    todo!("Extract eval_first_stage_bound from sddp/mod.rs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_result_default() {
        let result = StageResult {
            lower_bound: 0.0,
            cuts_added: 0,
            cuts_removed: 0,
        };
        assert_eq!(result.cuts_added, 0);
    }
}
```

### Update `src/algorithm/mod.rs`

```rust
pub mod backward_pass;

// Add to exports if needed
```

---

## Acceptance Criteria

- [ ] `src/algorithm/backward_pass.rs` created
- [ ] `execute()` function structure matches original backward loop
- [ ] Helper functions extracted: `compute_cut_data`, `reuse_forward_basis`, `eval_first_stage_bound`
- [ ] Module exported from `algorithm/mod.rs`
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (CRITICAL)

### Critical Verification

- [ ] Cut computation produces IDENTICAL cuts
- [ ] Lower bound calculation IDENTICAL
- [ ] Parallel execution pattern UNCHANGED
- [ ] FCF updates in same order

---

## Implementation Guide

### Suggested Approach

1. **Study the original code carefully**:
   ```bash
   sed -n '680,1100p' src/sddp/mod.rs > /tmp/backward_original.rs
   ```

2. **Create skeleton** with `todo!()` placeholders

3. **Extract helper functions first**:
   - `reuse_forward_basis()` - Simple, copy exactly
   - `eval_first_stage_bound()` - Simple, copy exactly
   - `compute_cut_data()` - Complex, be very careful

4. **Run golden tests** after each function extraction

5. **Implement `execute()`** last, calling helpers

6. **Verify golden tests** at every step

### Key Files to Modify/Create

| File | Action |
|------|--------|
| `src/algorithm/backward_pass.rs` | CREATE |
| `src/algorithm/mod.rs` | MODIFY (add export) |

### Critical Patterns to Preserve

1. **Stage ordering**: Reverse iteration order
2. **Branching pattern**: Same scenario generation
3. **Dual value extraction**: Same indices
4. **Cut coefficient computation**: Same formula
5. **Risk measure application**: Same API calls

### Pitfalls to Avoid

- ⚠️ Do NOT change cut computation formula
- ⚠️ Do NOT change branching scenario order
- ⚠️ Do NOT change dual value indices
- ⚠️ Preserve exact error messages
- ⚠️ Keep parallel execution pattern unchanged
- ⚠️ Do NOT modify FCF API calls

---

## Testing Requirements

### Unit Tests

- [ ] Test helper functions with mock data where feasible

### Golden Tests (CRITICAL)

- [ ] Run after EVERY function extraction
- [ ] `./scripts/golden-tests.sh verify` must pass

### Comparison Tests

- [ ] Compare cut coefficients before/after extraction
- [ ] Compare lower bounds before/after extraction

---

## Documentation Requirements

- [ ] Module-level docs explaining backward pass algorithm
- [ ] Function-level docs on all public functions
- [ ] Document the three phases (branching, cut, FCF)
- [ ] Note thread safety requirements

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Complex logic with many interdependencies; careful extraction required

---

## Definition of Done

- [ ] `backward_pass.rs` created with structure matching original
- [ ] Helper functions extracted with IDENTICAL logic
- [ ] Module exports updated
- [ ] All todos replaced with actual code
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Code reviewed for behavioral equivalence
