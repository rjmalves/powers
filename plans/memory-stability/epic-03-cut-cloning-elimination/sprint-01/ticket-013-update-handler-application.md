# [TICKET-013] Update handler cut application for Arc

> **Epic**: [Epic 3: Cut Cloning Elimination](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-012](./ticket-012-change-cutpool-to-arc.md)
> **Blocks**: Epic 4 tickets

## Context

### Background

Handler cut application currently clones full BendersCut data (~1 KB per cut) to pass to parallel handlers. With Arc-based pool, we can clone the Arc (~16 bytes) instead.

### Relation to Epic

Completes the cut cloning elimination by updating the usage site.

### Current State

```rust
// src/sddp/mod.rs:2166-2180
let cuts: Vec<(usize, crate::cut::BendersCut)> =
    aggregated_result
        .new_cut_ids
        .iter()
        .chain(aggregated_result.returning_cut_ids.iter())
        .filter_map(|&cut_id| {
            fcf_locked.cut_pool.pool.get(cut_id)
                .map(|cut| (cut_id, cut.clone()))  // FULL CLONE per cut!
        })
        .collect();
```

## Files to Read Before Starting

- `src/sddp/mod.rs:2166-2180` - Current cut cloning location
- `src/cut.rs` - Arc-based BendersCutPool (after TICKET-012)
- Handler application code in sddp module

## Specification

### Code Change

```rust
// Before: Clone full data
let cuts: Vec<(usize, crate::cut::BendersCut)> = ...
    .map(|cut| (cut_id, cut.clone()))

// After: Clone Arc
let cuts: Vec<(usize, Arc<crate::cut::BendersCut>)> = 
    aggregated_result
        .new_cut_ids
        .iter()
        .chain(aggregated_result.returning_cut_ids.iter())
        .filter_map(|&cut_id| {
            fcf_locked.cut_pool.pool.get(cut_id)
                .map(|cut| (cut_id, Arc::clone(cut)))  // Cheap Arc clone!
        })
        .collect();
```

### Handler Interface Changes

Handlers receiving cuts need to accept `Arc<BendersCut>` instead of owned `BendersCut`:

```rust
// Handler method signature change
fn apply_cuts(&mut self, cuts: &[(usize, Arc<BendersCut>)]) {
    for (cut_id, cut) in cuts {
        // Read-only access to cut data
        let coeffs = &cut.coefficients;
        let rhs = cut.rhs;
        // ...
    }
}
```

### Memory Savings

- Before: ~80 MB transient allocations (5521 cuts × 59 FCFs × ~1.3 KB × clones)
- After: ~1 MB (Arc clones are 16 bytes)

## Acceptance Criteria

- [ ] Handler cut application uses Arc::clone()
- [ ] Handler methods accept Arc<BendersCut>
- [ ] ~80 MB transient allocation eliminated
- [ ] All existing tests pass
- [ ] Training produces identical results

## Implementation Guide

### Suggested Approach

1. Update cut collection to use Arc::clone()
2. Update handler method signatures
3. Update all handler implementation sites
4. Run tests to find any missed updates

### Key Files to Modify

- `src/sddp/mod.rs`: Update cut collection and handler calls
- Handler implementation files (if separate)

### Pitfalls to Avoid

- ⚠️ Don't try to mutate cuts through Arc (use atomic methods)
- ⚠️ Ensure handlers only need read access to cut data

## Testing Requirements

### Unit Tests

- [ ] Test Arc clone is used (not data clone)
- [ ] Test handler receives correct cut data

### Integration Tests

- [ ] Run example-01 and verify identical results
- [ ] Run example-05 and verify identical results

### Performance Tests

- [ ] Memory profiling shows reduced transient allocations
- [ ] Runtime not significantly affected

## Documentation Requirements

- [ ] Update handler method doc comments
- [ ] Note that cuts are shared via Arc (read-only)

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward change once pool uses Arc
