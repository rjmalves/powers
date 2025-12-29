# [T-021] Integrate Forward Timing Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-020](./ticket-020-extract-forward-step.md)
> **Blocks**: [T-022](./ticket-022-update-sddp-forward.md)

---

## ⚠️ CRITICAL: Timing is Observational Only

This ticket integrates the new timing infrastructure from Epic 1 into the forward pass. **Timing changes must NOT affect algorithm behavior.** Timing is measurement only.

If ANY test fails or numerical output differs, **STOP IMMEDIATELY**—timing should have zero semantic impact.

---

## Files to Read Before Starting

- `src/algorithm/forward_pass.rs` - Forward pass from T-020
- `src/timing/mod.rs` - Timing module from Epic 1
- `src/timing/guard.rs` - `TimingGuard` RAII implementation
- `src/timing/metrics.rs` - `ForwardTiming`, `IterationTiming`
- `src/sddp/mod.rs:30-100` - Current timing accumulator structs
- `plans/clean-code-refactoring/00-master-plan.md` - Timing architecture section

---

## Context

### Background

The current forward pass uses scattered `Instant::now()` / `.elapsed()` calls for timing. The new timing infrastructure from Epic 1 uses RAII guards (`TimingGuard`) for cleaner, safer timing.

### Current Pattern (to replace)

```rust
let prep_start = Instant::now();
// ... work ...
timing.model_preprocessing += prep_start.elapsed();
```

### Target Pattern

```rust
{
    let _guard = TimingGuard::new(&ctx.timing.model_preprocessing);
    // ... work ...
} // Timing recorded automatically on drop
```

### Key Requirements from Master Plan

1. **Preserve precise values** - NEVER overwrite or redistribute timing
2. **Track parallel overhead explicitly** - Compute as `wall_time - avg(cpu_time)`
3. **Use RAII guards** - Eliminate scattered `Instant::now()` calls
4. **Feature-gated** - When `timing` feature disabled, compiles to no-op

---

## Specification

### Update `src/algorithm/forward_pass.rs`

Replace manual timing with `TimingGuard`:

```rust
use crate::timing::TimingGuard;

/// Execute a single stage of the forward pass.
fn execute_stage(
    ctx: &mut ForwardPassContext,
    stage_idx: usize,
    node_id: usize,
    timing: &mut TrajectoryTiming,
) -> Result<(), String> {
    // Model preparation - use timing guard
    let prep_elapsed = {
        let start = std::time::Instant::now();
        
        // Get subproblem node
        let subproblem_node = ctx
            .subproblem_graph
            .get_node_mut(node_id)
            .ok_or_else(|| format!("Could not find subproblem for node {}", node_id))?;

        // Get past realizations for this stage
        let past_node_ids = ctx
            .graph_bfs_table
            .get(stage_idx)
            .ok_or_else(|| format!("Could not find past node ids for node {}", node_id))?;

        let past_realizations: Vec<&Realization> = past_node_ids
            .iter()
            .map(|&past_id| {
                ctx.realization_graph
                    .get_node(past_id)
                    .map(|node| &node.data)
                    .ok_or_else(|| {
                        format!(
                            "Could not find realization for past_node {} (current_id {})",
                            past_id, node_id
                        )
                    })
            })
            .collect::<Result<_, _>>()?;

        // Prepare subproblem from trajectory
        subproblem_node.data.prepare_from_trajectory(&past_realizations)?;

        // Get realization node
        let realization_node = ctx
            .realization_graph
            .get_node_mut(node_id)
            .ok_or_else(|| format!("Could not find realization for node {}", node_id))?;

        // Get noises for this stage
        let current_stage_noises = ctx
            .sampled_noises
            .get(node_id)
            .ok_or_else(|| format!("Could not find noises for node {}", node_id))?;

        start.elapsed()
    };
    timing.model_preprocessing += prep_elapsed;

    // Need to re-get mutable references after prep block
    let subproblem_node = ctx
        .subproblem_graph
        .get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find subproblem for node {}", node_id))?;

    let realization_node = ctx
        .realization_graph
        .get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find realization for node {}", node_id))?;

    let current_stage_noises = ctx
        .sampled_noises
        .get(node_id)
        .ok_or_else(|| format!("Could not find noises for node {}", node_id))?;

    // Execute step (realize uncertainties and solve)
    let step_timing = step(
        &mut subproblem_node.data,
        &mut realization_node.data,
        current_stage_noises,
    )?;

    timing.solver += step_timing.solver_time;
    timing.model_postprocessing += step_timing.state_update_time;
    timing.solver_calls += 1;

    Ok(())
}
```

**Note**: Due to borrow checker constraints, we may need to keep the manual timing pattern in some places. The key is to use `TimingGuard` where possible and ensure timing values are **never overwritten**.

### Alternative: Use `time_scope!` macro where feasible

For simple cases:

```rust
use crate::time_scope;

fn some_function(timing: &ForwardTiming) {
    {
        time_scope!(timing.forward, model_preprocessing);
        // ... work ...
    }
}
```

### Update Timing Aggregation

When aggregating timing from parallel trajectories, **preserve precise values**:

```rust
/// Aggregate timing from multiple trajectories into ForwardTiming.
///
/// CRITICAL: This function preserves precise values and does NOT redistribute.
/// Parallel overhead is computed separately.
pub fn aggregate_trajectory_timings(
    trajectory_timings: &[TrajectoryTiming],
    target: &ForwardTiming,
) {
    if trajectory_timings.is_empty() {
        return;
    }

    let n = trajectory_timings.len() as u32;

    // Sum all timings (precise values preserved)
    let total_prep: Duration = trajectory_timings
        .iter()
        .map(|t| t.model_preprocessing)
        .sum();
    let total_solver: Duration = trajectory_timings.iter().map(|t| t.solver).sum();
    let total_post: Duration = trajectory_timings
        .iter()
        .map(|t| t.model_postprocessing)
        .sum();

    // Store averages (for representative per-trajectory metrics)
    target.model_preprocessing.set(total_prep / n);
    target.solver.set(total_solver / n);
    target.model_postprocessing.set(total_post / n);
}
```

---

## Acceptance Criteria

- [ ] Forward pass uses `TimingGuard` where practical
- [ ] Timing accumulation preserves precise values
- [ ] No timing value redistribution or overwriting
- [ ] `aggregate_trajectory_timings()` function added
- [ ] Parallel overhead computed separately (in training loop)
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

1. **Add timing imports**:
   ```rust
   use crate::timing::{TimingGuard, ForwardTiming};
   ```

2. **Identify timing points** in `execute_stage()`:
   - Model preprocessing (before step)
   - Solver time (from step)
   - Model postprocessing (from step)

3. **Replace manual timing** with guards where possible:
   - If guard works with borrow checker, use it
   - If not, keep manual timing but ensure pattern is clean

4. **Add aggregation function**:
   ```rust
   pub fn aggregate_trajectory_timings(...) { ... }
   ```

5. **Test timing feature**:
   ```bash
   cargo build --features timing
   cargo build --no-default-features
   ```

6. **Verify golden tests**:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/algorithm/forward_pass.rs` | Add timing guards, aggregation |
| `src/algorithm/context.rs` | May need timing field adjustments |

### Timing Guard Usage Patterns

**Pattern 1: Guard with block** (when no return needed):
```rust
{
    let _guard = TimingGuard::new(&timing.model_preprocessing);
    // ... work that doesn't return early ...
}
```

**Pattern 2: Manual timing** (when guard doesn't work):
```rust
let start = Instant::now();
// ... work with early returns or complex borrows ...
let elapsed = start.elapsed();
timing.model_preprocessing.set(timing.model_preprocessing.get() + elapsed);
```

**Pattern 3: time_scope! macro** (for Cell<Duration> fields):
```rust
{
    time_scope!(timing, model_preprocessing);
    // ... work ...
}
```

### Pitfalls to Avoid

- ⚠️ Do NOT change any algorithm logic—timing only
- ⚠️ Do NOT redistribute timing values after measurement
- ⚠️ Do NOT overwrite precise timing with computed values
- ⚠️ Be careful with borrow checker—timing guards borrow the Cell
- ⚠️ Test with and without `timing` feature

---

## Testing Requirements

### Unit Tests

- [ ] Test `aggregate_trajectory_timings()` with sample data
- [ ] Test timing guard accumulation (if adding new tests)

### Feature Tests

- [ ] `cargo build --features timing` succeeds
- [ ] `cargo build` (without timing) succeeds
- [ ] Timing values are collected when feature enabled

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Numerical output unchanged by timing changes

---

## Documentation Requirements

- [ ] Document timing points in `execute_stage()`
- [ ] Document aggregation function
- [ ] Update module docs if timing approach changes
- [ ] Note any places where manual timing was needed (and why)

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Timing integration may require borrow checker workarounds; feature testing adds complexity

---

## Definition of Done

- [ ] Timing guards used where practical
- [ ] Precise values preserved (no redistribution)
- [ ] Aggregation function implemented
- [ ] Feature-gated compilation works
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Documentation updated
- [ ] Code reviewed
