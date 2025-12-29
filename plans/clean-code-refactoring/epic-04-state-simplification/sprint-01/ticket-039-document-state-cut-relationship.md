# [T-039] Document State-Cut 1:1 Relationship and Slot Indexing

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-038](./ticket-038-analyze-state-structure.md)
> **Blocks**: [T-043](./ticket-043-pool-compatible-extensions.md)

## Files to Read Before Starting

- `src/fcf.rs` - FutureCostFunction, CutStatePair, CutData
- `src/state.rs` - VisitedStatePool, State trait
- `src/cut.rs` - BendersCut, BendersCutPool
- `src/algorithm/coordinator.rs` - How cuts and states are created together

---

## Context

### Background

Each Benders cut has exactly one originating state—the state that was visited when the cut was computed. This 1:1 relationship is fundamental to:
1. **Cut selection**: Checking if a cut dominates a visited state
2. **Pool storage**: States and cuts share the same slot index
3. **Memory optimization**: (Epic 5) Storing cut-state pairs together

This ticket documents the relationship formally to enable pool-based allocation in Epic 5.

### Current State

The relationship exists but is implicit:
- `BendersCutPool` and `VisitedStatePool` are separate in `FutureCostFunction`
- Both use `(iteration, forward_pass_idx)` as logical slot key
- `CutData` carries both cut and state coefficients together

---

## Specification

### Outputs

Document the following:

1. **Lifecycle Diagram**
   ```
   Forward Pass → Visit State → Store in StatePool[slot]
        ↓
   Backward Pass → Compute Cut → Store in CutPool[slot]
        ↓
   Cut Selection → Pair Cut[slot] with State[slot] for domination
   ```

2. **Slot Index Calculation**
   - Formula: `slot = (iteration - 1) * num_forward_passes + forward_pass_idx`
   - Where this calculation happens in code
   - Any edge cases or off-by-one considerations

3. **Data Flow Table**
   | Stage | Function | State Action | Cut Action |
   |-------|----------|--------------|------------|
   | Forward | `forward()` | Extract from trajectory | - |
   | Backward | `compute_cut_data()` | Package coefficients | Compute RHS |
   | Selection | `add_cuts_batch_from_data()` | Store in pool[slot] | Store in pool[slot] |
   | Domination | `eval_new_cut_domination()` | Iterate pool | Update non_dominated_count |

4. **Key Code References**
   - Where slot index is computed
   - Where state-cut pair is created
   - Where they are accessed together

---

## Acceptance Criteria

- [ ] State-Cut 1:1 relationship documented with lifecycle
- [ ] Slot index formula documented with code references
- [ ] Data flow from forward pass → backward pass → selection documented
- [ ] Edge cases (first iteration, cut selection disabled) noted
- [ ] Ready for T-043 pool interface design

---

## Implementation Guide

### Suggested Approach

1. **Trace state creation path**:
   ```bash
   grep -n "forward_pass_idx\|iteration" src/state.rs src/fcf.rs src/cut.rs
   ```

2. **Trace cut creation path**:
   ```bash
   grep -n "add_cuts_batch\|CutData" src/algorithm/coordinator.rs src/fcf.rs
   ```

3. **Document slot calculation**:
   - Find where `slot = (iteration - 1) * num_forward_passes + forward_pass_idx`
   - Or equivalent calculation

4. **Verify 1:1 invariant**:
   - Confirm cuts and states always have matching iteration/forward_pass_idx
   - Check if this invariant is enforced or just convention

### Key Files

- `src/fcf.rs:400-500` - `add_cuts_batch_from_data()` and pool updates
- `src/state.rs:422-434` - `VisitedStatePool::update_state()`
- `src/algorithm/coordinator.rs:200-250` - `select_cuts_batch()`

---

## Testing Requirements

- [ ] No code changes in this ticket (documentation only)

---

## Documentation Requirements

- [ ] Complete lifecycle diagram
- [ ] Document slot indexing formula with references
- [ ] Create data flow table

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Tracing existing code paths, documentation only

---

## Definition of Done

- [ ] 1:1 relationship formally documented
- [ ] Slot indexing explained with code references
- [ ] Data flow documented
- [ ] Edge cases noted
- [ ] Ready for Epic 5 pool design

---

## Documentation Results (To Be Completed)

### State-Cut 1:1 Relationship

```
[Diagram to be added]
```

### Slot Index Formula

```rust
// Location: [file:line]
// Formula:
let slot = ...;
```

### Data Flow

| Stage | Function | State Action | Cut Action |
|-------|----------|--------------|------------|
| | | | |

### Code References

- Slot calculation: `file.rs:line`
- State-cut pairing: `file.rs:line`
- Pool storage: `file.rs:line`
