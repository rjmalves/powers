# [T-097] Implement State Staging Buffer for Cut Computation

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

In `compute_cut_data()`, the state is cloned to create a working copy:

```rust
// src/subproblem.rs:1739
let mut visited_state = self.state.clone();
```

This allocates ~1.3 KB per cut computation × 60 stages × 4 forward passes = ~312 KB per iteration.

### Relation to Epic

Eliminates per-cut-computation state cloning.

### Current State

```rust
// src/subproblem.rs
fn compute_cut_data(&self, ...) -> CutData {
    let mut visited_state = self.state.clone();  // ALLOCATION
    // ... use visited_state ...
}
```

## Specification

### Changes Required

1. **Add staging state buffer** to subproblem or handler
2. **Replace clone with copy** into staging buffer
3. **Reuse staging buffer** across cut computations

### Approach Options

**Option A: Handler-level staging buffer**
- Each `SddpTrainHandler` has a staging `StateData`
- Copied from subproblem state before cut computation
- Cleared/reset after use

**Option B: Thread-local staging buffer**
- Thread-local buffer sized for max state dimension
- Copy state coefficients into buffer
- No per-subproblem overhead

### Inputs

- Original state (read-only)
- Staging buffer (mutable)

### Outputs

- Staging buffer contains copy of state coefficients
- No heap allocation for state

### Behavior

- Copy is shallow (just coefficients, not layout)
- Layout is shared/referenced, not copied
- State changes in staging buffer don't affect original

## Acceptance Criteria

- [ ] Staging buffer implemented
- [ ] `state.clone()` removed from cut computation
- [ ] State layout shared, not duplicated
- [ ] All tests pass
- [ ] Golden tests pass

## Implementation Guide

### Suggested Approach

1. **Identify what's actually cloned**:
   ```bash
   rg "\.clone\(\)" src/subproblem.rs | head -20
   ```

2. **Analyze state structure**:
   - What fields are in `State`?
   - Which need to be copied vs. referenced?

3. **Add staging buffer to handler**:
   ```rust
   // In SddpTrainHandler (if using handler approach)
   struct SddpTrainHandler {
       // ...existing fields...
       state_staging: StateStagingBuffer,
   }
   
   struct StateStagingBuffer {
       coefficients: Vec<f64>,
       // Other mutable state fields
   }
   ```

4. **Create copy method**:
   ```rust
   impl StateStagingBuffer {
       fn copy_from(&mut self, state: &impl State) {
           self.coefficients.clear();
           self.coefficients.extend_from_slice(state.coefficients());
       }
   }
   ```

5. **Update `compute_cut_data()`**:
   ```rust
   fn compute_cut_data(
       &self,
       staging: &mut StateStagingBuffer,  // Pass in staging buffer
       ...
   ) -> CutData {
       staging.copy_from(&self.state);
       // ... use staging instead of cloned state ...
   }
   ```

6. **If state is complex, consider partial copy**:
   - Only copy mutable parts
   - Keep layout reference shared

### Key Files to Modify

- `src/subproblem.rs` - `compute_cut_data()` and related
- `src/state.rs` - May need staging buffer types
- `src/sddp/mod.rs` - Handler initialization

### Patterns to Follow

- See `CutStagingBuffer` pattern from Sprint 1
- Separation of data (coefficients) from metadata (layout)

### Pitfalls to Avoid

- ⚠️ Ensure staging buffer is large enough
- ⚠️ Don't accidentally modify original state
- ⚠️ Clear staging buffer between uses if needed
- ⚠️ Consider trait object complexity in `State`

## Testing Requirements

### Unit Tests

- [ ] Staging buffer correctly copies state
- [ ] Cut computation produces same results

### Integration Tests

- [ ] Full training works with staging buffers
- [ ] Golden tests pass

### Performance Tests

- [ ] DHAT shows no `state.clone()` allocations

## Documentation Requirements

- [ ] Document staging buffer lifecycle
- [ ] Update `docs/MEMORY_BEHAVIOR.md`

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101 (DHAT verification)
- **Related**: Sprint 1 staging buffer pattern

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: State complexity may require careful design

## Definition of Done

- [ ] Staging buffer implemented
- [ ] No state cloning in cut computation
- [ ] Tests passing
- [ ] Golden tests pass
- [ ] Code reviewed
- [ ] PR merged
