# Epic 2: FCF Full Preallocation

## Summary

Ensure `FutureCostFunction::with_capacity()` is consistently used everywhere, eliminating Vec and HashMap reallocations during training.

## Scope

### Included

- Audit all FCF instantiation sites
- Replace `FutureCostFunction::new()` with `with_capacity()` where applicable
- Ensure BendersCutPool and VisitedStatePool use preallocation
- Validate memory profile

### Excluded

- New preallocation infrastructure (already exists)
- HiGHS changes (covered by Epic 1)

## Dependencies

- **Requires**: Epic 1 (HiGHS preallocation for full memory determinism)
- **Enables**: Epic 3 (SoA blocks, though independent)

## Acceptance Criteria

- [ ] All FCF instances use `with_capacity()` from SizingInfo
- [ ] No Vec reallocations during training
- [ ] No HashMap rehashing during training
- [ ] 1-3% performance improvement
- [ ] Examples produce identical results

## Technical Approach

### Phase 1: Audit (Day 1)

Find all places where `FutureCostFunction` is created:
- `src/sddp/mod.rs`
- `src/sddp/builder.rs`
- Test files (if any)

### Phase 2: Update (Days 2-3)

Replace `new()` with `with_capacity()`:

```rust
// Before
let fcf = FutureCostFunction::new();

// After
let fcf = FutureCostFunction::with_capacity(
    sizing.num_forward_passes,
    sizing.max_iterations,
    sizing.max_state_dimension,
);
```

### Phase 3: Validate (Days 4-5)

- Run examples
- Profile memory
- Benchmark performance

## Estimated Effort

- **Sprint 1**: 5 story points (1 week)

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Missing instantiation sites | Low | Grep for "FutureCostFunction::" |
| Incorrect capacity calculation | Low | Use existing SizingInfo fields |
