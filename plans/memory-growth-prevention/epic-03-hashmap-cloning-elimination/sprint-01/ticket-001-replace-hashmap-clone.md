# [TICKET-001] Replace HashMap clone with key snapshot

> **Epic**: [Epic 3: HashMap Cloning Elimination](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: None

## Context

### Background

During backward pass, the `active_cut_indices` HashMap is cloned to track pre-batch state. This clones ~250 KB per stage, totaling ~118 MB over 8 iterations.

### Key Discovery

**The parameter is UNUSED!** In `subproblem.rs:1729`:
```rust
_active_cut_indices_before: &std::collections::HashMap<usize, usize>,
```

The underscore prefix indicates it's not used. We may be able to simply remove it.

### Files to Read Before Starting

- `src/sddp/mod.rs:2077-2091` - HashMap clone site
- `src/subproblem.rs:1725-1768` - Unused parameter

## Specification

### Investigation Steps

1. Verify `_active_cut_indices_before` is truly unused in subproblem.rs
2. Check if removing it breaks any callers
3. Remove the parameter and clone if safe
4. If needed elsewhere, use minimal snapshot (Vec or HashSet of keys)

### Behavior

- Remove unnecessary HashMap clone
- Maintain identical cut selection behavior

## Acceptance Criteria

- [ ] No HashMap clone in hot path
- [ ] Parameter removed or replaced with minimal snapshot
- [ ] Examples produce identical results
- [ ] No functional change

## Implementation Guide

### Option A: Remove Unused Parameter (Preferred if truly unused)

```rust
// src/subproblem.rs - Remove parameter
pub fn apply_aggregated_cut_selection_result(
    &mut self,
    aggregated_result: &fcf::AggregatedCutSelectionResult,
    // REMOVED: _active_cut_indices_before
    cuts_to_add: &[(usize, cut::BendersCut)],
) -> Result<(), String> {
    // ...
}

// src/sddp/mod.rs - Update call site
parent_subproblem_node
    .data
    .apply_aggregated_cut_selection_result(
        aggregated_result,
        // REMOVED: active_cut_indices_before,
        cuts_to_add,
    )?;

// src/sddp/mod.rs - Remove clone
// REMOVED: let active_cut_indices_before = ...clone()
```

### Option B: Use HashSet<usize> if keys are needed

```rust
let active_cut_ids_before: std::collections::HashSet<usize> = {
    let fcf_locked = parent_fcf_node.data.lock().unwrap();
    fcf_locked.cut_pool.active_cut_indices.keys().copied().collect()
};
```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/sddp/mod.rs` | Remove clone, update handler call |
| `src/subproblem.rs` | Remove unused parameter |

### Verification Steps

1. Remove `_active_cut_indices_before` parameter
2. Update all callers
3. Remove HashMap clone at line 2077-2091
4. Run `cargo build` - should compile
5. Run examples - should produce identical output

### Pitfalls to Avoid

- ⚠️ Check ALL callers of `apply_aggregated_cut_selection_result`
- ⚠️ Verify underscore isn't hiding actual usage in debug/logging

## Testing Requirements

### Unit Tests

- [ ] `cargo build` succeeds after removal

### Integration Tests

- [ ] Example 01-deterministic produces identical output
- [ ] Example 07-par-model produces identical output

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Parameter appears unused, straightforward removal

## Definition of Done

- [ ] HashMap clone removed
- [ ] No functional change
- [ ] Tests passing
- [ ] Code reviewed
- [ ] PR merged
