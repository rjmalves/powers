# [T-098] Replace forward_costs.clone() with Move

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 7: Rust Application Allocation Optimization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-101

---

## Context

### Background

In the training loop, `forward_costs` is cloned when creating `IterationResult`:

```rust
// src/sddp/mod.rs:2080 (approximate)
iterations.push(IterationResult {
    forward_costs: forward_costs.clone(),
    ...
});
```

Since `forward_costs` is not used after this point, we can move instead of clone.

### Relation to Epic

Eliminates a per-iteration Vec clone.

### Current State

```rust
let (forward_costs, forward_timings): (Vec<f64>, Vec<...>) = 
    forward_results.into_iter().unzip();

// ... forward_costs used for convergence check ...

iterations.push(IterationResult {
    forward_costs: forward_costs.clone(),  // CLONE instead of move
    ...
});
```

## Specification

### Changes Required

1. **Use `std::mem::take()` or direct move** instead of clone
2. **Or reorder code** so forward_costs can be moved directly

### Expected Behavior

- No allocation for cloning forward costs
- Same `IterationResult` content

## Acceptance Criteria

- [ ] `forward_costs.clone()` removed
- [ ] Move or take used instead
- [ ] All tests pass

## Implementation Guide

### Suggested Approach

1. **Find the clone location**:
   ```bash
   rg "forward_costs.clone\(\)" src/sddp/mod.rs
   ```

2. **Check if forward_costs is used after**:
   - If not used: simply remove `.clone()`
   - If used before: reorder to use first, then move

3. **Use move**:
   ```rust
   iterations.push(IterationResult {
       forward_costs,  // Move instead of clone
       ...
   });
   ```

4. **Or use take if needed earlier**:
   ```rust
   // If forward_costs needed in multiple places:
   let costs_for_result = std::mem::take(&mut forward_costs);
   // forward_costs is now empty Vec
   
   iterations.push(IterationResult {
       forward_costs: costs_for_result,
       ...
   });
   ```

### Key Files to Modify

- `src/sddp/mod.rs` - Training loop

### Pitfalls to Avoid

- ⚠️ Ensure forward_costs isn't used after move
- ⚠️ Check for other clones in same pattern

## Testing Requirements

### Integration Tests

- [ ] Full training works
- [ ] IterationResult contains correct costs

## Dependencies

- **Blocked By**: None
- **Blocks**: T-101
- **Related**: Other clone elimination tickets

## Effort Estimate

**Points**: 1
**Confidence**: High
**Rationale**: Simple ownership change

## Definition of Done

- [ ] Clone removed
- [ ] Tests passing
- [ ] PR merged
