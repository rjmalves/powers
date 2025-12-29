# [T-022] Update sddp/mod.rs to Use forward_pass Module

> **Epic**: [Epic 3: Algorithm Separation](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Forward Pass Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-021](./ticket-021-forward-timing-integration.md)
> **Blocks**: Sprint 2

---

## ⚠️ CRITICAL: Facade Pattern Only

This ticket updates `sddp/mod.rs` to use the new `forward_pass` module. We use the **Facade Pattern**: the existing `forward()` method becomes a thin wrapper that delegates to the new module. This minimizes risk and allows easy rollback.

**The algorithm behavior must remain EXACTLY unchanged.**

---

## Files to Read Before Starting

- `src/sddp/mod.rs:596-680` - Current `forward()` to wrap
- `src/sddp/mod.rs:1310-1500` - Training loop that calls `forward()`
- `src/algorithm/forward_pass.rs` - New module from T-020
- `src/algorithm/context.rs` - Context structs from T-019

---

## Context

### Background

The forward pass logic has been extracted to `src/algorithm/forward_pass.rs`. Now we need to update `sddp/mod.rs` to use this new module while maintaining the existing API.

### Strategy: Facade Pattern

The existing `TrajectoryEngine::forward()` method will delegate to `algorithm::forward_pass::execute()`:

```rust
// Before: 80+ lines of implementation
pub fn forward(...) -> Result<(f64, ForwardPassTimingAccumulator), String> {
    // ... implementation ...
}

// After: Thin wrapper
pub fn forward(...) -> Result<(f64, ForwardPassTimingAccumulator), String> {
    let mut ctx = ForwardPassContext::new(...);
    let result = algorithm::forward_pass::execute(&mut ctx)?;
    Ok((result.trajectory_cost, convert_timing(result)))
}
```

### Benefits

1. **Minimal risk**: Call sites unchanged
2. **Easy rollback**: Can revert to inline implementation
3. **Incremental migration**: Can be done one method at a time
4. **Maintains compatibility**: No API changes

---

## Specification

### Modify `src/sddp/mod.rs`

#### 1. Add Import

```rust
use crate::algorithm;
```

#### 2. Update `TrajectoryEngine::forward()`

Replace the implementation with a facade:

```rust
impl TrajectoryEngine {
    pub fn forward(
        &mut self,
        sampled_noises: Vec<&scenario::OptimizedSampledBranchingNoises>,
        graph_bfs_table: &[Vec<usize>],
        study_period_ids: &[usize],
    ) -> Result<(f64, ForwardPassTimingAccumulator), String> {
        use crate::algorithm::{ForwardPassContext, forward_pass};
        use crate::timing::ForwardTiming;

        // Create timing storage
        let timing = ForwardTiming::default();

        // Create context with all necessary data
        let mut ctx = ForwardPassContext::new(
            &mut self.subproblem_graph,
            &mut self.realization_graph,
            &sampled_noises,
            graph_bfs_table,
            study_period_ids,
            &timing,
        );

        // Execute forward pass
        let result = forward_pass::execute(&mut ctx)?;

        // Convert to legacy timing format for backward compatibility
        let legacy_timing = ForwardPassTimingAccumulator {
            model_preprocessing_time: timing.model_preprocessing.get(),
            solver_time: timing.solver.get(),
            model_postprocessing_time: timing.model_postprocessing.get(),
            solver_calls: result.solver_calls,
        };

        Ok((result.trajectory_cost, legacy_timing))
    }
}
```

#### 3. Remove Duplicate Code

After the facade is working, the old `step()` function at the module level can be marked as `#[deprecated]` or removed if it's no longer used:

```rust
// Old function - can be removed once facade is verified
#[deprecated(note = "Use algorithm::forward_pass::step instead")]
fn step(
    subproblem: &mut subproblem::Subproblem,
    realization_container: &mut subproblem::Realization,
    noises: &scenario::OptimizedSampledBranchingNoises,
) -> Result<StepTiming, String> {
    // ... original implementation ...
}
```

**Note**: Only remove the old code after verifying golden tests pass with the facade.

---

## Acceptance Criteria

- [ ] `TrajectoryEngine::forward()` delegates to `algorithm::forward_pass::execute()`
- [ ] Context created correctly with all required fields
- [ ] Timing converted from new format to legacy `ForwardPassTimingAccumulator`
- [ ] Return type unchanged: `Result<(f64, ForwardPassTimingAccumulator), String>`
- [ ] All existing call sites work without modification
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] **Golden tests pass** (CRITICAL)

### Correctness Verification

- [ ] Numerical output is IDENTICAL to before the change
- [ ] Timing values are reasonable (sanity check)
- [ ] No API changes visible to callers

---

## Implementation Guide

### Suggested Approach

1. **Add import** at top of `sddp/mod.rs`:
   ```rust
   use crate::algorithm;
   ```

2. **Identify the exact function signature** of current `forward()`:
   ```bash
   sed -n '596,600p' src/sddp/mod.rs
   ```

3. **Create backup** of original implementation:
   - Comment out original body (don't delete yet)
   - Or copy to a `forward_original()` function

4. **Implement facade**:
   - Create `ForwardPassContext` from method parameters
   - Call `forward_pass::execute()`
   - Convert result to expected return type

5. **Handle timing conversion**:
   - New timing uses `ForwardTiming` with `Cell<Duration>`
   - Old timing uses `ForwardPassTimingAccumulator` with `Duration`
   - Convert after execution

6. **Run golden tests** IMMEDIATELY:
   ```bash
   ./scripts/golden-tests.sh verify
   ```

7. **If tests fail**:
   - Revert to original implementation
   - Debug the difference
   - Ask for help if needed

8. **If tests pass**:
   - Clean up commented code
   - Mark old `step()` as deprecated (if applicable)

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Update `forward()` to use facade |

### Timing Conversion

The new and old timing formats differ:

| New (`ForwardTiming`) | Old (`ForwardPassTimingAccumulator`) |
|----------------------|-------------------------------------|
| `model_preprocessing: Cell<Duration>` | `model_preprocessing_time: Duration` |
| `solver: Cell<Duration>` | `solver_time: Duration` |
| `model_postprocessing: Cell<Duration>` | `model_postprocessing_time: Duration` |
| N/A | `solver_calls: usize` |

Conversion:
```rust
ForwardPassTimingAccumulator {
    model_preprocessing_time: timing.model_preprocessing.get(),
    solver_time: timing.solver.get(),
    model_postprocessing_time: timing.model_postprocessing.get(),
    solver_calls: result.solver_calls,
}
```

### Pitfalls to Avoid

- ⚠️ Do NOT change the function signature
- ⚠️ Do NOT change the return type
- ⚠️ Do NOT modify call sites in the training loop yet
- ⚠️ Keep original implementation available until golden tests pass
- ⚠️ Be careful with lifetime annotations on context
- ⚠️ Ensure `sampled_noises` slice lives long enough

---

## Testing Requirements

### Unit Tests

- [ ] Existing unit tests for `forward()` still pass

### Integration Tests

- [ ] SDDP training loop works correctly

### Golden Tests (CRITICAL)

- [ ] `./scripts/golden-tests.sh verify` passes
- [ ] Run BEFORE and AFTER to compare

### Performance Tests

- [ ] `cargo bench` shows no significant regression

---

## Documentation Requirements

- [ ] Update `TrajectoryEngine::forward()` docs to note it delegates
- [ ] Add `#[deprecated]` notice to old `step()` if applicable
- [ ] Update module docs if structure changed

---

## Rollback Plan

If golden tests fail and you can't fix the issue:

1. **Revert the facade**:
   - Uncomment the original implementation
   - Remove the facade code

2. **Report the issue**:
   - Document what failed
   - Include error messages or diff output
   - Ask for clarification

3. **Do NOT proceed** with broken golden tests

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Facade is straightforward but timing conversion and lifetime management may require iteration

---

## Definition of Done

- [ ] `forward()` uses facade pattern
- [ ] Context creation correct
- [ ] Timing conversion correct
- [ ] Return type unchanged
- [ ] All tests pass
- [ ] **Golden tests pass**
- [ ] No API changes
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Old code cleaned up (deprecated or removed)
