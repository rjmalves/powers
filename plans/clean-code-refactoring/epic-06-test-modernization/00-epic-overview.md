# Epic 6: Test Modernization

> **Master Plan**: [00-master-plan.md](../00-master-plan.md)
> **Duration**: 2 weeks (1 sprint)
> **Status**: ⬜ Not Started

---

## ⚠️ CRITICAL REMINDER

Test modernization focuses on **test infrastructure**, not algorithm behavior.

The algorithm remains unchanged. New tests must validate existing behavior, not impose new behavior.

---

## Summary

This epic replaces brittle, implementation-coupled tests with behavior-focused test suites. The goal is test infrastructure that:

1. **Validates behavior**, not implementation details
2. **Enables refactoring** without breaking tests
3. **Provides confidence** in algorithm correctness

---

## Scope

### Included

1. **Test Audit**
   - Identify brittle tests coupled to implementation
   - Document tests that should be behavior-based
   - Identify missing test coverage

2. **Behavior-Focused Tests**
   - Create integration tests based on observable outcomes
   - Property-based tests for numerical invariants

3. **Test Utilities**
   - Helper functions for common test patterns
   - Fixtures for test data

### Excluded

- Algorithm changes
- Performance optimization
- New features

---

## Dependencies

- **Requires**:
  - Epic 1 complete (golden test baseline)
  - Epics 2-5 complete (refactored code to test)
- **Enables**:
  - Epic 7: Performance Validation (confidence in correctness)

---

## Acceptance Criteria

- [ ] Brittle tests identified and documented
- [ ] Key integration tests created for major workflows
- [ ] Property-based tests for numerical invariants
- [ ] Test coverage maintained ≥85%
- [ ] Golden tests still passing
- [ ] Refactored code has clean test coverage

---

## Sprints

### [Sprint 1: Test Infrastructure](./sprint-01/00-sprint-overview.md)

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| T-037 | Audit existing tests for brittleness | 3 | ⬜ |
| T-038 | Create behavior-focused integration tests | 5 | ⬜ |
| T-039 | Add property-based tests for numerical invariants | 5 | ⬜ |
| T-040 | Create test utilities and fixtures | 3 | ⬜ |

**Sprint Points**: 16

---

## Estimated Effort

- **Duration**: 1 sprint (2 weeks)
- **Story Points**: 16
- **Risk Level**: Low (tests don't affect algorithm)

---

## Definition of Done

- [ ] All tickets complete
- [ ] Test suite modernized
- [ ] Coverage ≥85%
- [ ] Golden tests pass
- [ ] All tests pass
