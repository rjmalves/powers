# [T-021] Integrate Forward Timing Infrastructure

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-020](./ticket-020-extract-forward-step.md)
> **Blocks**: [T-022](./ticket-022-update-sddp-forward.md)
> **Status**: ❌ Incomplete - Requires Rework (see below)

---

## ⚠️ STATUS: REWORK REQUIRED

**Problem Identified (2025-12-29)**: The initial implementation used `Instant::now()` instead of `TimingGuard` due to borrow checker conflicts. This was an unacceptable compromise that should have been escalated.

**Root Cause**: Timing was embedded inside `ForwardPassContext`, creating borrow conflicts:
```rust
// PROBLEMATIC: timing inside context
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    pub timing: &'a ForwardTiming,  // ❌ Borrowing this prevents mutable access to graph
}

fn execute_stage(ctx: &mut ForwardPassContext) {
    let _guard = TimingGuard::new(&ctx.timing.model_preprocessing); // borrows ctx
    let node = ctx.subproblem_graph.get_node_mut(id)?; // ❌ CONFLICT: ctx already borrowed
}
```

**Solution**: Remove timing from context, pass as separate parameter.

---

## Architectural Decision: Separate Timing from Context

### Rationale

Rust's borrow checker enforces that you cannot:
1. Immutably borrow part of a struct (for timing)
2. While mutably borrowing another part (for graph access)

When timing is inside the context, using `TimingGuard` creates a borrow that prevents mutable access to other context fields.

### Solution

**Pass timing as a separate parameter**, not inside the context:

```rust
// CORRECT: timing separate from context
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    pub realization_graph: &'a mut DirectedGraph<Realization>,
    // NO timing field
}

pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,  // ✅ Separate parameter
) -> Result<ForwardPassResult, String> {
    for (idx, &id) in ctx.study_period_ids.iter().enumerate() {
        execute_stage(ctx, idx, id, timing)?;
    }
    Ok(...)
}

fn execute_stage(
    ctx: &mut ForwardPassContext,
    stage_idx: usize,
    node_id: usize,
    timing: &TrajectoryTiming,  // ✅ Separate parameter
) -> Result<(), String> {
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        // ✅ Now we can mutably access ctx.subproblem_graph!
        let node = ctx.subproblem_graph.get_node_mut(node_id)?;
        // ... work ...
    }
    Ok(())
}
```

---

## Files to Read Before Starting

- `src/algorithm/context.rs` - Current `ForwardPassContext` (has timing inside)
- `src/algorithm/forward_pass.rs` - Current implementation (uses `Instant::now()`)
- `src/timing/guard.rs` - `TimingGuard` RAII implementation
- `src/timing/metrics.rs` - `ForwardTiming` struct
- `src/sddp/mod.rs` - Call sites for forward pass

---

## Specification

### Step 1: Update `ForwardPassContext` in `context.rs`

**Remove timing field from ForwardPassContext:**

```rust
/// Context for forward pass execution.
///
/// # Design Note: Timing Separation
///
/// Timing is NOT included in this context to avoid borrow checker conflicts.
/// When using `TimingGuard`, the guard borrows the timing struct. If timing
/// were inside this context, we couldn't mutably access graph fields while
/// timing is active.
///
/// Pass timing as a separate parameter to `forward_pass::execute()`.
pub struct ForwardPassContext<'a> {
    pub subproblem_graph: &'a mut DirectedGraph<Subproblem>,
    pub realization_graph: &'a mut DirectedGraph<Realization>,
    pub sampled_noises: &'a HashMap<usize, &'a OptimizedSampledBranchingNoises>,
    pub graph_bfs_table: &'a [Vec<usize>],
    pub study_period_ids: &'a [usize],
    // NO timing field - passed separately to avoid borrow conflicts
}
```

### Step 2: Update `TrajectoryTiming` to use `Cell<Duration>`

**Ensure `TrajectoryTiming` uses `Cell` for interior mutability:**

```rust
/// Timing data for a single forward pass trajectory.
///
/// Uses `Cell<Duration>` to allow `TimingGuard` to accumulate without &mut.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryTiming {
    pub model_preprocessing: Cell<Duration>,
    pub solver: Cell<Duration>,
    pub model_postprocessing: Cell<Duration>,
    pub solver_calls: Cell<usize>,
}

impl TrajectoryTiming {
    pub fn increment_solver_calls(&self) {
        self.solver_calls.set(self.solver_calls.get() + 1);
    }
}
```

### Step 3: Update `forward_pass.rs` to use `TimingGuard`

**Update function signatures:**

```rust
use crate::timing::TimingGuard;

/// Execute a forward pass using the provided context.
pub fn execute(
    ctx: &mut ForwardPassContext,
    timing: &TrajectoryTiming,  // Separate parameter
) -> Result<ForwardPassResult, String> {
    for (idx, &id) in ctx.study_period_ids.iter().enumerate() {
        execute_stage(ctx, idx, id, timing)?;
    }

    // Cost calculation with timing guard
    let trajectory_cost = {
        let _guard = TimingGuard::new(&timing.model_postprocessing);
        compute_trajectory_cost(ctx)?
    };

    Ok(ForwardPassResult::new(trajectory_cost, timing.solver_calls.get()))
}

fn execute_stage(
    ctx: &mut ForwardPassContext,
    stage_idx: usize,
    node_id: usize,
    timing: &TrajectoryTiming,
) -> Result<(), String> {
    // Model preparation with timing guard
    {
        let _guard = TimingGuard::new(&timing.model_preprocessing);
        
        let subproblem_node = ctx.subproblem_graph.get_node_mut(node_id)
            .ok_or_else(|| format!("Could not find subproblem for node {}", node_id))?;
        
        let past_node_ids = ctx.graph_bfs_table.get(stage_idx)
            .ok_or_else(|| format!("Could not find past node ids for node {}", node_id))?;
        
        let past_realizations: Vec<&Realization> = past_node_ids
            .iter()
            .map(|&past_id| {
                ctx.realization_graph
                    .get_node(past_id)
                    .map(|node| &node.data)
                    .ok_or_else(|| format!("Could not find realization for past_node {}", past_id))
            })
            .collect::<Result<_, _>>()?;
        
        subproblem_node.data.prepare_from_trajectory(&past_realizations)?;
    }
    
    // Solver execution with timing guard
    let realization_node = ctx.realization_graph.get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find realization for node {}", node_id))?;
    
    let current_stage_noises = ctx.sampled_noises.get(&node_id)
        .ok_or_else(|| format!("Could not find noises for node {}", node_id))?;
    
    let subproblem_node = ctx.subproblem_graph.get_node_mut(node_id)
        .ok_or_else(|| format!("Could not find subproblem for node {}", node_id))?;
    
    let step_timing = {
        let _guard = TimingGuard::new(&timing.solver);
        step(&mut subproblem_node.data, &mut realization_node.data, current_stage_noises)?
    };
    
    // Post-processing timing
    {
        let _guard = TimingGuard::new(&timing.model_postprocessing);
        // Any post-processing work
    }
    
    timing.increment_solver_calls();
    
    Ok(())
}
```

### Step 4: Update call sites in `sddp/mod.rs`

**Update forward pass calls:**

```rust
// Before:
let (result, trajectory_timing) = forward_pass::execute(&mut ctx)?;

// After:
let timing = TrajectoryTiming::default();
let result = forward_pass::execute(&mut ctx, &timing)?;
// timing now contains the accumulated values
```

### Step 5: Feature-gate the timing

**Ensure timing compiles to no-op when feature disabled:**

The `TimingGuard` already handles this. Verify that:
1. `cargo build --features timing` works
2. `cargo build` (without timing) works
3. When timing disabled, guards compile to no-ops

---

## Acceptance Criteria

- [ ] `ForwardPassContext` does NOT contain timing field
- [ ] `TrajectoryTiming` uses `Cell<Duration>` for all fields
- [ ] `forward_pass::execute()` takes timing as separate parameter
- [ ] `execute_stage()` takes timing as separate parameter
- [ ] All timing uses `TimingGuard` (no raw `Instant::now()`)
- [ ] `cargo build --features timing` succeeds
- [ ] `cargo build` (no timing feature) succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (timing changes have zero semantic impact)
- [ ] Feature-gated: timing compiles to no-op when disabled

---

## Testing Requirements

### Unit Tests

- [ ] Test `TrajectoryTiming` accumulation with `TimingGuard`
- [ ] Test `forward_pass::execute()` with timing parameter

### Feature Tests

- [ ] `cargo build --features timing` succeeds
- [ ] `cargo build` (without timing) succeeds
- [ ] Verify timing values are collected when feature enabled
- [ ] Verify no overhead when feature disabled

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Numerical output unchanged by timing changes

---

## Key Files to Modify

| File | Changes |
|------|---------|
| `src/algorithm/context.rs` | REMOVE timing from `ForwardPassContext`, UPDATE `TrajectoryTiming` to use `Cell` |
| `src/algorithm/forward_pass.rs` | ADD timing parameter, REPLACE `Instant::now()` with `TimingGuard` |
| `src/sddp/mod.rs` | UPDATE call sites to pass timing separately |

---

## Pitfalls to Avoid

- ⚠️ Do NOT change any algorithm logic—timing only
- ⚠️ Ensure `TrajectoryTiming` uses `Cell` for interior mutability
- ⚠️ Pass timing as `&TrajectoryTiming`, not `&mut` (Cell provides interior mutability)
- ⚠️ Test with and without `timing` feature
- ⚠️ Golden tests MUST pass

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear architectural fix, straightforward implementation

---

## Definition of Done

- [ ] Timing separated from context
- [ ] All `Instant::now()` replaced with `TimingGuard`
- [ ] Feature-gating verified
- [ ] `cargo build` succeeds (both with and without timing feature)
- [ ] `cargo test` passes
- [ ] **Golden tests pass**
- [ ] Documentation updated
- [ ] Code reviewed
