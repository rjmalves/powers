# Sprint 2 Index

**Duration**: 2 weeks  
**Total Effort**: 52 hours (estimated)  
**Focus**: Convergence Tracking + Coverage Improvements + Numerical Validation

---

## Sprint 2 Overview

Building on Sprint 1's strong foundation (312 tests, 69.93% coverage), Sprint 2 focuses on:

1. **Convergence Tracking**: Enable tests to validate numerical correctness
2. **Coverage Improvements**: Address Sprint 1 gaps (FCF 57%→90%, Stochastic 57%→80%)
3. **Numerical Validation**: Create benchmarks with known solutions
4. **Comprehensive Testing**: Solver interface, subproblem construction

**Key Deliverable**: TrainingResult infrastructure enabling rigorous numerical validation.

---

## Tickets

### Phase 1: Convergence Infrastructure (Week 1, Part 1) - 15h

**T2.1: TrainingResult Struct** - 6h  
**Status**: Not Started  
**Priority**: Critical  
**Dependencies**: None

Create `TrainingResult` and `IterationResult` structs to capture convergence history.

**Deliverables**:

- TrainingResult struct with iteration history
- IterationResult with bounds, costs, timings
- Helper methods: final_gap(), relative_gap(), converged()
- Comprehensive tests

**Blocks**: T2.2

---

**T2.2: Update train() Return Type** - 4h  
**Status**: Not Started  
**Priority**: Critical  
**Dependencies**: T2.1

Change `train()` to return `Result<TrainingResult, String>` (breaking change).

**Deliverables**:

- Updated train() signature
- Population of TrainingResult in training loop
- All existing tests updated
- Migration guide in CHANGELOG.md

**Blocks**: T2.3, T2.6

---

**T2.3: Integration Test Convergence Validation** - 5h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: T2.2

Enhance integration tests with convergence assertions.

**Deliverables**:

- All integration tests use TrainingResult
- Convergence assertions (monotonicity, gap reduction, bounds validity)
- 4 new convergence validation tests
- Helper functions for assertions
- TESTING.md convergence section

**Related**: T2.7

---

### Phase 2: Coverage Improvements (Week 1, Part 2) - 9h

**T2.4: FCF Coverage Improvement (57% → 90%)** - 6h  
**Status**: Not Started  
**Priority**: Critical  
**Dependencies**: None

Comprehensive testing of Future Cost Function module.

**Deliverables**:

- Cut domination logic tests
- Active cut selection tests
- Cut pool management tests
- Edge case tests
- Integration tests
- FCF coverage ≥90%

**Context**: Sprint 1 gap - lowest coverage module

---

**T2.5: Stochastic Process Coverage (57% → 80%)** - 3h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: None

Testing of stochastic process module.

**Deliverables**:

- Distribution realization tests (Uniform, Normal, Discrete)
- Multi-dimensional process tests
- Edge case tests
- Integration with SAA tests
- Coverage ≥80%

**Context**: Sprint 1 gap - tied for lowest coverage

---

### Phase 3: Numerical Validation (Week 2, Part 1) - 11h

**T2.6: Benchmark Problems** - 6h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: T2.2

Create benchmark problems with known solutions for validation.

**Deliverables**:

- Newsvendor benchmark (known solution)
- Hydrothermal benchmark (known solution)
- Benchmark module with factory functions
- tests/BENCHMARKS.md documentation
- Validation tests

**Enables**: T2.7

---

**T2.7: Numerical Validation Tests** - 5h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: T2.6

Comprehensive numerical correctness tests using benchmarks.

**Deliverables**:

- Convergence to known solutions tests
- Bound property validation tests
- Stability across seeds tests
- Robustness to parameters tests
- Edge case tests (deterministic, high variance)
- TESTING.md validation section

**Context**: Validates algorithmic correctness, not just code execution

---

### Phase 4: Comprehensive Testing (Week 2, Part 2) - 10h

**T2.8: Solver Interface Tests** - 5h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: None

Thorough testing of solver interface with mock and real solvers.

**Deliverables**:

- Mock solver for controlled testing
- Real solver integration tests
- Error handling tests (infeasible, unbounded)
- Edge case tests
- Performance tests
- Thread safety tests (if applicable)
- docs/SOLVER_INTERFACE.md

**Context**: Critical integration point - failures can crash algorithm

---

**T2.9: Subproblem Tests** - 5h  
**Status**: Not Started  
**Priority**: High  
**Dependencies**: None

Comprehensive subproblem construction and validation tests.

**Deliverables**:

- Construction tests (first/intermediate/last stage)
- Constraint generation tests
- Cut integration tests
- State transition tests
- Uncertainty realization tests
- Edge case tests
- Integration with solver tests
- docs/SUBPROBLEM_STRUCTURE.md

**Context**: Complex logic requiring thorough validation

---

### Phase 5: Review and Documentation (Week 2, End) - 5h

**T2.10: Sprint Review** - 5h  
**Status**: Not Started  
**Priority**: Medium  
**Dependencies**: T2.1-T2.9

Sprint 2 review, metrics, and documentation.

**Deliverables**:

- REVIEW.md (comprehensive assessment)
- RETROSPECTIVE.md (learnings)
- CHANGELOG.md updated
- TESTING.md updated
- README.md updated (if needed)
- Master roadmap updated
- Sprint 3 prep notes

**Context**: Final Sprint 2 closure similar to T1.10

---

## Execution Order

### Recommended Sequence

1. **T2.1** → T2.2 → T2.3 (Convergence infrastructure - sequential)
2. **T2.4** ‖ T2.5 (Coverage improvements - parallel)
3. **T2.6** → T2.7 (Benchmarks then validation - sequential)
4. **T2.8** ‖ T2.9 (Solver and subproblem - parallel)
5. **T2.10** (Review - depends on all)

### Critical Path

T2.1 → T2.2 → T2.6 → T2.7 → T2.10 (23 hours)

**Parallel work opportunities**:

- T2.4 & T2.5 can run in parallel (9h → ~6h with concurrency)
- T2.8 & T2.9 can run in parallel (10h → ~6h with concurrency)

**Realistic timeline**: ~44-48 hours with some parallelization

---

## Dependencies Graph

```
T2.1 (TrainingResult)
  ↓
T2.2 (train() update)
  ↓ ↘
T2.3 (Test updates)  T2.6 (Benchmarks)
                       ↓
T2.4 (FCF)           T2.7 (Validation)
  ⊗                    ↓
T2.5 (Stochastic)    T2.10 (Review)
  ↓                    ↑
T2.8 (Solver) -------'
  ⊗
T2.9 (Subproblem) ---'

Legend:
  ↓  = blocks
  ⊗ = can be done in parallel
  ' = contributes to
```

---

## Metrics Targets

### Coverage

| Module     | Sprint 1 | Sprint 2 Target | Delta  |
| ---------- | -------- | --------------- | ------ |
| Overall    | 69.93%   | 75%             | +5.07% |
| FCF        | 57%      | 90%             | +33%   |
| Stochastic | 57%      | 80%             | +23%   |
| SDDP       | 85%      | 85-90%          | +0-5%  |

### Tests

- **Sprint 1**: 312 tests (40 unit, 267 integration, 5 doc)
- **Sprint 2 Target**: ~400+ tests
- **New Tests**: ~90+ tests
  - Convergence tests: ~15
  - FCF tests: ~20
  - Stochastic tests: ~10
  - Benchmark tests: ~5
  - Validation tests: ~15
  - Solver tests: ~15
  - Subproblem tests: ~15

### Quality

- **Clippy**: Zero warnings (maintained)
- **Format**: 100% formatted (maintained)
- **Doc coverage**: Improve from Sprint 1
- **Build time**: Keep under 10 minutes

---

## Risk Assessment

### High Risk

1. **Convergence tracking breaking change** (T2.2)

   - Mitigation: Clear migration guide, update all tests
   - Impact: All existing code calling train()

2. **Coverage targets ambitious** (T2.4, T2.5)
   - Mitigation: Prioritize critical paths over 100% coverage
   - Impact: May not hit 90% FCF if very complex

### Medium Risk

1. **Benchmark solutions accuracy** (T2.6)

   - Mitigation: Validate with external references or DP
   - Impact: Tests may have wrong expected values

2. **Test flakiness with probabilistic tests** (T2.5, T2.7)
   - Mitigation: Large sample sizes, appropriate tolerances
   - Impact: CI may have intermittent failures

### Low Risk

1. **Parallel ticket execution** (T2.4 ‖ T2.5, T2.8 ‖ T2.9)
   - Mitigation: Clear separation of concerns
   - Impact: Minimal if tickets are truly independent

---

## Success Criteria

### Must Have (Critical)

- [ ] T2.1-T2.3: Convergence tracking complete and working
- [ ] T2.4: FCF coverage ≥85% (90% stretch goal)
- [ ] Overall coverage ≥73% (75% stretch goal)
- [ ] T2.6-T2.7: At least 2 benchmarks with validation
- [ ] All tests pass, zero clippy warnings

### Should Have (High Priority)

- [ ] T2.5: Stochastic coverage ≥75% (80% stretch goal)
- [ ] T2.8: Comprehensive solver tests
- [ ] T2.9: Comprehensive subproblem tests
- [ ] Documentation fully updated

### Nice to Have (Medium Priority)

- [ ] 3rd benchmark problem
- [ ] Performance regression tests
- [ ] Advanced numerical validation tests

---

## Sprint 2 Philosophy

**Focus**: **Numerical correctness validation**

Unlike Sprint 1 (foundation), Sprint 2 enables **validation that SDDP produces correct results**, not just that it runs without crashing.

**Key principle**: "Tests should validate correctness, not just coverage"

**Approach**:

1. Enable convergence inspection (TrainingResult)
2. Improve coverage of critical modules (FCF, Stochastic)
3. Create ground truth (benchmarks)
4. Validate against ground truth (numerical tests)
5. Comprehensive edge case testing (solver, subproblem)

---

## Links

- **Sprint 2 Overview**: `overview.md`
- **Sprint 1 Review**: `../sprint-01/REVIEW.md`
- **Master Roadmap**: `../../roadmap/MASTER_ROADMAP.md`
- **Testing Guide**: `../../TESTING.md` (project root)

---

## Status Legend

- **Not Started**: Ticket not begun
- **In Progress**: Currently being worked on
- **Blocked**: Waiting on dependencies
- **Review**: Implementation complete, under review
- **Complete**: Fully done, tested, documented

---

## Notes

**Sprint 2 builds on Sprint 1's success**:

- Sprint 1: Foundation (tests, coverage, CI/CD) - Grade: A (9.5/10)
- Sprint 2: Correctness (convergence, validation, benchmarks)

**Breaking change in Sprint 2**:

- T2.2 changes train() return type
- Requires update to all calling code
- Migration guide provided

**Sprint 2 → Sprint 3 transition**:

- Sprint 2 completes testing foundation
- Sprint 3 can focus on features, performance, or advanced topics
- Decision informed by Sprint 2 retrospective
