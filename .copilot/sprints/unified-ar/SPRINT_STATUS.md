# Unified AR Model Refactoring - Sprint Status

**Epic:** Simplify AR Model Representation in Subproblems  
**Start Date:** TBD  
**Target Completion:** TBD (estimated 22 working days across 4 sprints)  
**Status:** Not Started

---

## Executive Summary

This sprint plan implements the architectural refactoring described in `UNIFIED_AR_ROADMAP_REVISED.md`. The core goal is to **eliminate conditional logic** in hot paths by representing all inflow models (independent and autoregressive) through a unified, constraint-based formulation **while preserving both state type choices** (StorageState and StorageAndInflowState).

### Key Insights

- **AR(0) = Independent** — Allows a single code path with variable coefficients
- **Keep Both State Types** — StorageState and StorageAndInflowState are valid modeling choices
- **Decouple State from AR** — AR dynamics via UnifiedInflowModel, independent of state choice

### Expected Benefits

- **Simplicity:** Single code path for AR dynamics (no state-type conditionals)
- **Flexibility:** Users can choose StorageState (simpler) or StorageAndInflowState (richer cuts)
- **Performance:** 15-30% speedup from eliminated branches and optimizations
- **Maintainability:** Clear separation: state choice vs AR representation
- **Correctness:** Explicit space transformations reduce numerical bugs

---

## Sprint Overview

### Sprint 1: Foundation (8 days)

Build the UnifiedInflowModel infrastructure without breaking existing code.

**Tickets:**

- [TICKET-001](TICKET-001-create-unified-inflow-model-struct.md): Create UnifiedInflowModel struct (2 days)
- [TICKET-002](TICKET-002-implement-constraint-generation.md): Implement LP constraint generation (3 days)
- [TICKET-003](TICKET-003-implement-lag-buffer-management.md): Implement lag buffer management (3 days)

**Goals:**

- ✅ UnifiedInflowModel handles both independent and AR cases
- ✅ Constraint generation creates explicit AR dynamics in LP
- ✅ Lag buffer maintains historical residuals correctly

**Success Criteria:**

- All unit tests pass
- No integration with Subproblem yet (isolated development)
- Performance baseline established

---

### Sprint 2: Subproblem Refactor (10 days)

Integrate UnifiedInflowModel into Subproblem and eliminate conditional logic.

**Tickets:**

- [TICKET-004](TICKET-004-refactor-variables-struct.md): Refactor Variables struct (2 days)
- [TICKET-005](TICKET-005-refactor-constraints-struct.md): Refactor Constraints struct (1 day)
- [TICKET-006](TICKET-006-update-realization-struct.md): Update Realization struct (1 day)
- [TICKET-007](TICKET-007-integrate-unified-model-into-subproblem.md): Integrate UnifiedInflowModel into Subproblem (2 days)
- [TICKET-008](TICKET-008-simplify-realize-uncertainties.md): Simplify realize_uncertainties() (3 days) **[CRITICAL]**
- [TICKET-009](TICKET-009-implement-update-from-trajectory.md): Implement update_from_trajectory() (2 days)

**Goals:**

- ✅ Subproblem uses UnifiedInflowModel for all inflow operations
- ✅ realize_uncertainties() has zero conditionals on state type
- ✅ Dual space representation (observation + residual) consistently maintained

**Success Criteria:**

- Examples 06 and 07 compile and run
- realize_uncertainties() cyclomatic complexity < 5
- 15-25% speedup measured on realize_uncertainties()

---

### Sprint 3: Cleanup (5 days)

Remove duplicate AR logic and update all tests/examples.

**Tickets:**

- [TICKET-010](TICKET-010-remove-storage-and-inflow-state.md): Refactor State trait interface (2 days)
- [TICKET-011](TICKET-011-remove-observation-space-par.md): Remove observation space PAR code (2 days)
- [TICKET-012](TICKET-012-update-integration-tests.md): Update integration tests and examples (2 days) [REVISED]

**Goals:**

- ✅ State trait simplified to focus on state variable definition
- ✅ Both StorageState and StorageAndInflowState work with unified AR model
- ✅ PAR generator works exclusively in residual space
- ✅ All examples and tests updated and passing

**Success Criteria:**

- State trait has no AR-specific methods
- All integration tests pass with both state types
- Test coverage maintained or improved

---

### Sprint 4: Optimization (4 days)

Performance optimizations to achieve full speedup potential.

**Tickets:**

- [TICKET-013](TICKET-013-precompute-seasonal-cache.md): Precompute seasonal transformations (2 days)
- [TICKET-014](TICKET-014-optimize-batch-rhs-updates.md): Optimize constraint RHS updates (2 days)

**Goals:**

- ✅ Seasonal parameters cached for O(1) access
- ✅ RHS updates batched to reduce FFI overhead
- ✅ Full 20-30% speedup achieved

**Success Criteria:**

- Subproblem construction 5-10% faster
- realize_uncertainties 15-25% faster
- Overall algorithm 15-25% faster
- Memory overhead < 1MB

---

## Current Status

### Sprint 1: Foundation

- [x] TICKET-001: Create UnifiedInflowModel struct
- [x] TICKET-002: Implement constraint generation
- [ ] TICKET-003: Implement lag buffer management

### Sprint 2: Subproblem Refactor

- [ ] TICKET-004: Refactor Variables struct
- [ ] TICKET-005: Refactor Constraints struct
- [ ] TICKET-006: Update Realization struct
- [ ] TICKET-007: Integrate UnifiedInflowModel into Subproblem
- [ ] TICKET-008: Simplify realize_uncertainties() **[CRITICAL]**
- [ ] TICKET-009: Implement update_from_trajectory()

### Sprint 3: Cleanup

- [ ] TICKET-010: Refactor State trait interface
- [ ] TICKET-011: Remove observation space PAR code
- [ ] TICKET-012: Update integration tests and examples

### Sprint 4: Optimization

- [ ] TICKET-013: Precompute seasonal transformations
- [ ] TICKET-014: Optimize batch RHS updates

---

## Progress Tracking

| Sprint    | Tickets | Completed | In Progress | Blocked | Total Days | Status          |
| --------- | ------- | --------- | ----------- | ------- | ---------- | --------------- |
| Sprint 1  | 3       | 2         | 0           | 0       | 8          | In Progress     |
| Sprint 2  | 6       | 0         | 0           | 0       | 10         | Not Started     |
| Sprint 3  | 3       | 0         | 0           | 0       | 5          | Not Started     |
| Sprint 4  | 2       | 0         | 0           | 0       | 4          | Not Started     |
| **Total** | **14**  | **2**     | **0**       | **0**   | **27**     | **In Progress** |

---

## Dependencies Graph

```
Sprint 1 (Foundation)
├── TICKET-001 (UnifiedInflowModel struct)
│   ├── TICKET-002 (Constraint generation) → needs struct
│   └── TICKET-003 (Lag buffer) → needs struct
│
Sprint 2 (Subproblem Refactor)
├── TICKET-004 (Variables struct) → independent
├── TICKET-005 (Constraints struct) → independent
├── TICKET-006 (Realization struct) → independent
├── TICKET-007 (Subproblem integration) → needs 001, 002, 004, 005
├── TICKET-008 (realize_uncertainties) → needs 001-007 [CRITICAL]
└── TICKET-009 (update_from_trajectory) → needs 003, 007, 008
│
Sprint 3 (Cleanup)
├── TICKET-010 (Refactor State trait interface) → needs 008, 009
├── TICKET-011 (Remove observation PAR) → needs 001
└── TICKET-012 (Update tests/examples) → needs 008, 010, 011
│
Sprint 4 (Optimization)
├── TICKET-013 (Seasonal cache) → needs 001, 002
└── TICKET-014 (Batch RHS updates) → independent
```

---

## Critical Path

The critical path determines the minimum time to complete the epic:

1. **TICKET-001** (2 days) → UnifiedInflowModel struct
2. **TICKET-002** (3 days) → Constraint generation
3. **TICKET-004** (2 days, parallel) → Variables struct
4. **TICKET-005** (1 day, parallel) → Constraints struct
5. **TICKET-006** (1 day, parallel) → Realization struct
6. **TICKET-007** (2 days) → Subproblem integration
7. **TICKET-008** (3 days) → realize_uncertainties [CRITICAL]
8. **TICKET-009** (2 days) → update_from_trajectory
9. **TICKET-010** (1 day) → Refactor State trait interface
10. **TICKET-011** (2 days, parallel) → Remove observation PAR
11. **TICKET-012** (2 days) → Update tests
12. **TICKET-013** (2 days, parallel) → Seasonal cache
13. **TICKET-014** (2 days, parallel) → Batch RHS updates

**Minimum completion time: 22 days** (with some parallelization)

---

## Risk Management

### High-Risk Areas

1. **TICKET-008 (realize_uncertainties)** - Core algorithm change

   - **Risk:** Numerical differences, correctness issues
   - **Mitigation:** Extensive testing, comparison with baseline
   - **Fallback:** Keep old implementation behind feature flag

2. **TICKET-010 (Refactor State trait interface)** - Architectural change
   - **Risk:** Existing code incorrectly assumes removal of StorageAndInflowState
   - **Mitigation:** Thorough validation that both state types work with UnifiedInflowModel
   - **Fallback:** Document clear examples for when to use each state type

### Medium-Risk Areas

3. **TICKET-012 (Update tests/examples)** - Many files to change
   - **Risk:** Missed test cases, broken examples
   - **Mitigation:** Systematic checklist, CI validation
4. **TICKET-011 (PAR generator)** - Algorithm change
   - **Risk:** Estimated coefficients differ
   - **Mitigation:** Validate with known test cases

### Low-Risk Areas

5. **Struct refactoring tickets (004-006)** - Data structure changes only
6. **Optimization tickets (013-014)** - Performance only, correctness unchanged

---

## Quality Gates

Each sprint must pass these gates before proceeding:

### After Sprint 1

- [ ] All Sprint 1 unit tests pass
- [ ] `cargo clippy` shows no warnings
- [ ] `cargo fmt` applied
- [ ] Documentation builds successfully
- [ ] Code review completed

### After Sprint 2

- [ ] All unit and integration tests pass
- [ ] Examples 06 and 07 run successfully
- [ ] realize_uncertainties speedup measured (expect 15-25%)
- [ ] No regressions in other examples
- [ ] Code review completed

### After Sprint 3

- [ ] All tests pass with cleanup completed
- [ ] Grep searches confirm code removed
- [ ] Test coverage maintained (≥ previous level)
- [ ] All examples produce correct output
- [ ] Performance maintained or improved
- [ ] Code review completed

### After Sprint 4

- [ ] Performance targets achieved (15-30% overall speedup)
- [ ] Memory overhead acceptable (< 1MB)
- [ ] All benchmarks updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md complete
- [ ] Code review completed
- [ ] Ready for release

---

## Performance Targets

| Metric                                        | Baseline | Target                   | Measured |
| --------------------------------------------- | -------- | ------------------------ | -------- |
| realize_uncertainties                         | -        | 15-25% faster            | -        |
| Subproblem construction                       | -        | 5-10% faster             | -        |
| Overall algorithm                             | -        | 15-25% faster            | -        |
| Memory usage                                  | -        | ±20% (reduce redundancy) | -        |
| Cyclomatic complexity (realize_uncertainties) | ~15      | ≤ 5                      | -        |

---

## Testing Strategy

### Unit Tests (Continuous)

- Test each new component in isolation
- Test edge cases (AR(0), AR(1), AR(2), mixed)
- Test space transformations explicitly

### Integration Tests (End of each sprint)

- Full forward/backward pass
- Multi-stage problems
- PAR models with multi-node PreStudy

### Regression Tests (End of Sprint 3)

- Compare output with baseline on all examples
- Numerical tolerance: ± 0.1% on lower bound
- Performance regression: no slowdowns

### Performance Tests (End of Sprint 4)

- Benchmark realize_uncertainties
- Benchmark full algorithm
- Memory profiling
- Compare before/after for each optimization

---

## Communication Plan

### Daily Updates

- Update ticket status in sprint status file
- Log any blockers or issues
- Update estimated completion dates

### End of Sprint Reviews

- Review sprint goals achievement
- Update roadmap if needed
- Identify lessons learned
- Plan next sprint

### Stakeholder Updates

- Weekly progress summary
- Performance measurement results
- Risk assessment updates

---

## Definition of Done

A ticket is "Done" when:

- [ ] Code compiles without warnings
- [ ] All unit tests pass
- [ ] Integration tests pass (if applicable)
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Documentation updated (inline and module-level)
- [ ] CHANGELOG.md updated
- [ ] Code reviewed by at least one team member
- [ ] Performance benchmarked (if applicable)
- [ ] Examples tested (if applicable)

---

## Resources

- **Architecture Doc:** `UNIFIED_AR_ROADMAP.md`
- **Current Code:** `src/subproblem.rs`, `src/state.rs`, `src/stochastic_process.rs`
- **Examples:** `examples/06-par-model/`, `examples/07-par-model-with-inflow-state/`
- **Tests:** `tests/test_par_*.rs`, `tests/test_sddp_*.rs`

---

## Completed Work Log

### TICKET-002: Implement LP constraint generation (Completed)

**Date:** 2024-01-XX  
**Files Modified:**

- `src/unified_inflow_model.rs` (constraint generation implementation)

**Key Deliverables:**

1. ✅ `ConstraintIndices` struct with `ar_dynamics` and `observation_transform` vectors
2. ✅ `add_constraints_to_lp()` method generates two constraints per hydro:
   - AR dynamics: Z'\_t - Σ(φ_k \* Z'\_{t-k}) = ε_t
   - Observation transform: Y_t - σ_s\*Z'\_t = μ_s
3. ✅ 8 comprehensive unit tests covering:
   - AR(1) models
   - AR(2) models
   - Independent (AR(0)) models
   - Mixed models (different orders)
   - Constraint indices structure
   - Seasonal parameter application

**Quality Gates:**

- ✅ All 16 unit tests pass (10 from TICKET-001 + 8 from TICKET-002)
- ✅ `cargo fmt --all` clean
- ✅ `cargo clippy --all-targets --all-features -- -D warnings` zero warnings
- ✅ No regressions in existing tests (432/434 tests pass, 2 pre-existing failures in test_scenario_validation.rs)

**Performance Characteristics:**

- Constraint generation is O(n·p) where n = number of hydros, p = max_lag
- Pre-allocates vectors to avoid runtime allocations
- Uses inline hints for hot path methods

**Notes:**

- Fixed clippy warning about needless borrow in `add_row()` call
- Updated test helper `create_mock_variables()` to properly allocate variables in solver::Problem
- Independent models (AR(0)) get simplified AR dynamics constraint: Z'\_t = ε_t

---

## Notes

### Key Design Decisions

1. **AR(0) = Independent:** Single code path for all inflow types
2. **Dual Space Representation:** Maintain both Y_t (observation) and Z'\_t (residual)
3. **Explicit Constraints:** AR dynamics as LP constraints (not implicit in state)
4. **Residual Space Native:** All AR coefficients and dynamics in residual space
5. **Keep Both State Types:** StorageState and StorageAndInflowState are both valid, UnifiedInflowModel supports both

### Lessons Learned

(To be filled in during implementation)

---

## Quick Start Guide

### For Developers Starting a Ticket

1. Read the ticket markdown file in detail
2. Review dependencies (are they complete?)
3. Create a feature branch: `git checkout -b unified-ar/ticket-XXX`
4. Implement following the ticket's task checklist
5. Run pre-checks: `cargo fmt -- --check && cargo clippy`
6. Run tests: `cargo test --workspace`
7. Update this status file with progress
8. Submit PR when validation checklist complete

### For Reviewers

1. Check that all checkboxes in ticket validation are complete
2. Verify code follows architectural patterns in roadmap
3. Test locally with examples
4. Check performance impact (if applicable)
5. Review documentation updates
6. Approve or request changes

---

**Last Updated:** 2024-01-XX  
**Updated By:** GitHub Copilot  
**Next Review:** After TICKET-003 completion
