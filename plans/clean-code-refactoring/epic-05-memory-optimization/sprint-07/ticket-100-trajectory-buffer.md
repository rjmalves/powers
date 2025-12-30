# [T-100] Preallocate past_realizations Trajectory Buffer

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

In forward pass execution, `past_realizations` is collected into a Vec every stage:

```rust
// src/algorithm/forward_pass.rs:123-136
let past_realizations: Vec<&Realization> = past_node_ids
    .iter()
    .map(...)
    .collect::<Result<_, _>>()?;
```

With 60 stages × 4 forward passes = 240 allocations per iteration.

### Relation to Epic

Eliminates per-stage allocations in forward pass.

### Current State

- New Vec created for each stage
- Contains references to past realizations
- Used only within stage processing

## Specification

### Changes Required

1. **Add trajectory buffer to ForwardPassContext**
2. **Reuse buffer across stages** (clear and refill)
3. **Pass buffer reference to stage processing**

### Behavior

- Buffer cleared at start of each stage
- Filled with references to past realizations
- Reused for all stages in forward pass

## Acceptance Criteria

- [ ] Trajectory buffer preallocated in context
- [ ] No per-stage allocation for past_realizations
- [ ] All tests pass
- [ ] Golden tests pass

## Implementation Guide

### Suggested Approach

1. **Add buffer to ForwardPassContext**:
   ```rust
   pub struct ForwardPassContext<'a> {
       // ...existing fields...
       past_realizations_buffer: Vec<&'a Realization>,
   }
   
   impl<'a> ForwardPassContext<'a> {
       pub fn new(..., max_stages: usize) -> Self {
           Self {
               // ...
               past_realizations_buffer: Vec::with_capacity(max_stages),
           }
       }
   }
   ```

2. **Update forward pass to use buffer**:
   ```rust
   // Before:
   let past_realizations: Vec<&Realization> = past_node_ids
       .iter()
       .map(...)
       .collect::<Result<_, _>>()?;
   
   // After:
   ctx.past_realizations_buffer.clear();
   for node_id in past_node_ids {
       let realization = get_realization(node_id)?;
       ctx.past_realizations_buffer.push(realization);
   }
   let past_realizations = &ctx.past_realizations_buffer;
   ```

3. **Alternative: Use indices instead of references**:
   ```rust
   // Store indices into realization storage
   let past_indices: &mut Vec<usize> = &mut ctx.past_indices_buffer;
   past_indices.clear();
   for node_id in past_node_ids {
       past_indices.push(node_id_to_index(node_id));
   }
   ```

### Key Files to Modify

- `src/algorithm/forward_pass.rs` - Stage processing
- `src/algorithm/mod.rs` - ForwardPassContext if exists

### Patterns to Follow

- See existing context buffer patterns
- Clear-and-refill pattern for reuse

### Pitfalls to Avoid

- ⚠️ Lifetime of references must be valid for stage
- ⚠️ Buffer must be cleared between stages
- ⚠️ Consider borrow checker with mutable context

## Testing Requirements

### Unit Tests

- [ ] Buffer reuse works correctly
- [ ] Past realizations collected properly

### Integration Tests

- [ ] Forward pass produces correct results
- [ ] Golden tests pass

## Documentation Requirements

- [ ] Document buffer lifecycle in context

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101 (DHAT verification)
- **Related**: T-094, T-095 (other buffer optimizations)

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Lifetime management needs care

## Definition of Done

- [ ] Buffer preallocated
- [ ] No per-stage allocation
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
