# Epic 1: HiGHS Constraint Preallocation

## Summary

Pre-allocate cut constraint slots in HiGHS solver at training start, then modify coefficients and bounds during training instead of adding new rows. This eliminates all HiGHS memory allocations during the training hot path.

## Scope

### Included

- Add `Highs_addRows` batch API binding to `src/solver.rs`
- Add `Highs_changeCoeff` API binding to `src/solver.rs`
- Extend `SizingInfo` with `estimate_max_cuts_per_node()` method
- Add cut slot tracking fields to `Subproblem`
- Implement `preallocate_cut_constraints()` method
- Modify `add_cut_constraint_to_model()` to use coefficient updates
- Implement cut removal via bound relaxation
- Implement cut slot reuse for cut selection

### Excluded

- HiGHS internal optimization (out of scope)
- Model reconstruction (defeats warm-starting purpose)
- Presolve optimization (test behavior, may disable)

## Dependencies

- **Requires**: None (first epic)
- **Enables**: Epic 2 (FCF), Epic 3 (SoA blocks)

## Acceptance Criteria

- [ ] Zero `Highs_addRow` calls during training iterations
- [ ] Memory profile flat (±1%) during training
- [ ] All examples produce identical results
- [ ] Performance improvement ≥3%
- [ ] Cut slot exhaustion handled gracefully

## Technical Approach

### Phase 1: Core Infrastructure (Sprint 1)

1. **Add HiGHS API Bindings**:
   - `add_rows_batch()`: Batch add rows via `Highs_addRows`
   - `change_coefficient()`: Update single coefficient via `Highs_changeCoeff`

2. **Extend SizingInfo**:
   - `estimate_max_cuts_per_node()`: Conservative estimate of max cuts
   - `estimate_lp_dimensions()`: Total rows/cols including cut slots

3. **Subproblem Cut Slot Infrastructure**:
   - `first_preallocated_cut_row`: Index of first cut row in solver
   - `num_preallocated_cuts`: Total preallocated slots
   - `next_available_cut_slot`: Next sequential slot
   - `cut_slot_to_id`: Mapping from slot to cut ID
   - `free_cut_slots`: Stack of freed slots for reuse

### Phase 2: Integration (Sprint 2)

4. **Update Cut Addition**:
   - Allocate slot from free list or sequential
   - Update coefficients via `Highs_changeCoeff`
   - Activate constraint via `Highs_changeRowBounds(row, rhs, ∞)`

5. **Update Cut Removal**:
   - Deactivate via `Highs_changeRowBounds(row, -∞, ∞)`
   - Add slot to free list for reuse

6. **Validation**:
   - Run examples 01 and 07
   - Compare convergence with baseline
   - Profile memory with massif

## Estimated Effort

- **Sprint 1**: 5 story points (1 week)
- **Sprint 2**: 5 story points (1 week)
- **Total**: 10 story points (2 weeks)

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| HiGHS presolve removes inactive constraints | Medium | Test with presolve disabled |
| Coefficient change slower than expected | Low | Benchmark, fallback to current approach |
| Slot exhaustion in pathological cases | Low | Graceful fallback with warning log |
