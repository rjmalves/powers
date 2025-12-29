# [T-026] Integrate Backward Timing Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Backward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-025](./ticket-025-extract-cut-computation.md)
> **Blocks**: [T-027](./ticket-027-update-sddp-backward.md)

---

## ⚠️ CRITICAL: Timing is Observational Only

This ticket integrates the new timing infrastructure into the backward pass. **Timing changes must NOT affect algorithm behavior.** Timing is measurement only.

If ANY test fails or numerical output differs, **STOP IMMEDIATELY**—timing should have zero semantic impact.

---

## Files to Read Before Starting

- `src/algorithm/backward_pass.rs` - Backward pass from T-024
- `src/algorithm/cut_computation.rs` - Cut computation from T-025
- `src/timing/metrics.rs` - `BackwardTiming` struct
- `src/timing/guard.rs` - `TimingGuard` RAII implementation
- `src/sddp/mod.rs:30-100` - Current timing accumulator structs

---

## Context

### Background

The backward pass has timing at multiple levels:
- **Preprocessing**: Setup for each stage
- **Solver**: LP solves for branching scenarios
- **Cut computation**: Computing cut coefficients
- **FCF update**: Adding cuts to future cost function
- **Handler application**: Post-processing handlers

### Current Pattern

```rust
let prep_start = Instant::now();
// ... preprocessing ...
timing.backward_preprocessing_time += prep_start.elapsed();

let solve_start = Instant::now();
// ... solve branchings ...
timing.solver_time += solve_start.elapsed();
```

### Target Pattern

```rust
{
    let _guard = TimingGuard::new(&ctx.timing.preprocessing);
    // ... preprocessing ...
}

{
    let _guard = TimingGuard::new(&ctx.timing.solver);
    // ... solve branchings ...
}
```

---

## Specification

### Update `src/algorithm/backward_pass.rs`

Add timing guards at each measurement point:

```rust
use crate::timing::{TimingGuard, BackwardTiming};

pub fn execute(ctx: &mut BackwardPassContext) -> Result<BackwardPassResult, String> {
    // Record preprocessing start
    {
        let _guard = TimingGuard::new(&ctx.timing.preprocessing);
        // ... any preprocessing before stage loop ...
    }

    for (stage_idx, &stage_id) in ctx.backward_stage_ids.iter().enumerate() {
        execute_stage_backward(ctx, stage_idx, stage_id)?;
    }

    Ok(BackwardPassResult::new(...))
}

fn execute_stage_backward(
    ctx: &mut BackwardPassContext,
    stage_idx: usize,
    stage_id: usize,
) -> Result<StageResult, String> {
    // Model preprocessing
    {
        let _guard = TimingGuard::new(&ctx.timing.model_preprocessing);
        // ... get node data, past IDs, etc. ...
    }

    // Solver (branching solves)
    let solve_start = Instant::now();
    let branching_results = solve_branchings(ctx, stage_id)?;
    ctx.timing.solver.set(
        ctx.timing.solver.get() + solve_start.elapsed()
    );

    // Cut computation
    {
        let _guard = TimingGuard::new(&ctx.timing.cut_computation);
        let cut_data = cut_computation::compute_cut(&branching_results, ...)?;
    }

    // FCF update
    {
        let _guard = TimingGuard::new(&ctx.timing.fcf_update);
        ctx.fcf.add_cut(cut_data)?;
    }

    // Cut selection (if applicable)
    {
        let _guard = TimingGuard::new(&ctx.timing.cut_selection);
        // ... cut selection logic ...
    }

    Ok(StageResult { ... })
}
```

### Timing Points to Instrument

Based on `BackwardTiming` in `src/timing/metrics.rs`:

| Timing Field | Where to Measure |
|--------------|------------------|
| `preprocessing` | Before stage loop starts |
| `model_preprocessing` | Node data lookup, trajectory retrieval |
| `solver` | Branching LP solves |
| `model_postprocessing` | Solution extraction from branchings |
| `cut_computation` | `compute_cut()` call |
| `cut_selection` | Cut filtering/selection logic |
| `fcf_update` | `fcf.add_cut()` call |
| `handler_application` | Post-cut handlers |

### Aggregation for Parallel Branchings

If branching solves are parallel, aggregate timing carefully:

```rust
/// Aggregate timing from parallel branching solves.
///
/// CRITICAL: Preserve precise values, do not redistribute.
pub fn aggregate_branching_timings(
    branching_timings: &[BranchingSolveTiming],
    target: &BackwardTiming,
) {
    // Sum all timings (or average for per-branching metrics)
    let total_solver: Duration = branching_timings
        .iter()
        .map(|t| t.solver)
        .sum();
    
    // Add to existing timing (Cell allows interior mutability)
    target.solver.set(target.solver.get() + total_solver);
}
```

---

## Acceptance Criteria

- [ ] All backward pass timing points instrumented
- [ ] `TimingGuard` used where practical
- [ ] Timing values preserved (no redistribution)
- [ ] Parallel branching timing aggregated correctly
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (timing changes have zero semantic impact)

### Correctness Verification

- [ ] Numerical output is IDENTICAL to before timing changes
- [ ] Timing values are reasonable (sanity check)
- [ ] No algorithm behavior changes

---

## Implementation Guide

### Suggested Approach

1. **Review `BackwardTiming` struct**:
   ```bash
   grep -A 30 "struct BackwardTiming" src/timing/metrics.rs
   ```

2. **Identify all timing points** in backward pass

3. **Add timing imports**:
   ```rust
   use crate::timing::{TimingGuard, BackwardTiming};
   ```

4. **Instrument each phase** with guards or manual timing

5. **Handle parallel branchings** carefully

6. **Verify golden tests** after each change

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/algorithm/backward_pass.rs` | Add timing instrumentation |
| `src/algorithm/cut_computation.rs` | Add timing if needed |

### Timing Patterns

**Pattern 1: Simple guard (when scope matches timing)**
```rust
{
    let _guard = TimingGuard::new(&ctx.timing.cut_computation);
    let cut = compute_cut(...)?;
}
```

**Pattern 2: Manual timing (when scope doesn't match)**
```rust
let start = Instant::now();
// ... work that may early return ...
ctx.timing.solver.set(ctx.timing.solver.get() + start.elapsed());
```

**Pattern 3: Aggregation (for parallel work)**
```rust
let timings: Vec<Duration> = parallel_results
    .iter()
    .map(|r| r.timing)
    .collect();
let total = timings.iter().sum();
ctx.timing.solver.set(ctx.timing.solver.get() + total);
```

### Pitfalls to Avoid

- ⚠️ Do NOT change any algorithm logic—timing only
- ⚠️ Be careful with borrow checker when using guards
- ⚠️ Ensure timing from parallel work is aggregated correctly
- ⚠️ Test with `timing` feature enabled and disabled

---

## Testing Requirements

### Unit Tests

- [ ] Test timing aggregation functions

### Feature Tests

- [ ] `cargo build --features timing` succeeds
- [ ] `cargo build` (without timing) succeeds

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Numerical output unchanged by timing changes

---

## Documentation Requirements

- [ ] Document timing points in backward pass
- [ ] Document aggregation strategy for parallel branchings
- [ ] Update module docs if timing approach is notable

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Similar to forward pass timing; parallel branching adds complexity

---

## Definition of Done

- [ ] All timing points instrumented
- [ ] Timing guards used where practical
- [ ] Parallel timing aggregated correctly
- [ ] Feature-gated compilation works
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Documentation updated
- [ ] Code reviewed
