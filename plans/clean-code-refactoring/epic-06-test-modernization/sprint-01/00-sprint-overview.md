# Sprint 1: Test Infrastructure

> **Epic**: [Epic 6: Test Modernization](../00-epic-overview.md)
> **Duration**: 2 weeks
> **Status**: ⬜ Not Started

---

## Goals

1. **Primary**: Audit and document brittle tests
2. **Primary**: Create behavior-focused integration tests
3. **Secondary**: Add property-based tests
4. **Validation**: Maintain test coverage

---

## Tickets

| ID | Title | Points | Assignable | Dependencies | Status |
|----|-------|--------|------------|--------------|--------|
| [T-037](./ticket-037-audit-tests.md) | Audit existing tests for brittleness | 3 | Yes | Epics 2-5 | ⬜ |
| [T-038](./ticket-038-integration-tests.md) | Create behavior-focused integration tests | 5 | Yes | T-037 | ⬜ |
| [T-039](./ticket-039-property-tests.md) | Add property-based tests for numerical invariants | 5 | Yes | T-037 | ⬜ |
| [T-040](./ticket-040-test-utilities.md) | Create test utilities and fixtures | 3 | Yes | None | ⬜ |

**Total Points**: 16

---

## Parallelization

```
T-040 (Utilities) ────────────────────────────→
T-037 (Audit) ──→ T-038 (Integration) ────────→
             └──→ T-039 (Property-based) ─────→
```

T-038 and T-039 can run in parallel after T-037. T-040 is independent.

---

## Definition of Done

- [ ] All 4 tickets complete
- [ ] Test coverage ≥85%
- [ ] Golden tests pass
- [ ] All tests pass
