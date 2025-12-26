# Sprint 1: Core Infrastructure

## Goals

- Add HiGHS batch row and coefficient change API bindings
- Extend SizingInfo with cut estimation methods
- Implement Subproblem cut slot infrastructure

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| TICKET-001 | Add HiGHS batch row API bindings | 2 | None |
| TICKET-002 | Extend SizingInfo with cut estimation | 2 | None |
| TICKET-003 | Implement Subproblem cut slot infrastructure | 3 | TICKET-001, TICKET-002 |

## Dependencies

- **From Previous Sprint**: None (first sprint)
- **To Next Sprint**: TICKET-003 provides infrastructure for cut addition/removal

## Risks

- HiGHS API binding may require unsafe code review
- Cut estimation formula may need tuning

## Definition of Done

- [ ] All tickets complete
- [ ] Code compiles without warnings
- [ ] Examples 01 and 07 still work
- [ ] New API methods have doc comments
