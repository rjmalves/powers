# Sprint 1: Complete FCF Preallocation

## Goals

- Audit all FCF instantiation sites
- Ensure all instances use `with_capacity()`
- Validate memory profile and performance

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| TICKET-008 | Audit FCF instantiation sites | 1 | None |
| TICKET-009 | Ensure with_capacity usage everywhere | 2 | TICKET-008 |
| TICKET-010 | Validate memory profile | 2 | TICKET-009 |

## Dependencies

- **From Previous Epic**: Epic 1 complete (HiGHS preallocation)
- **To Next Epic**: Epic 3 (Handler SoA blocks)

## Risks

- May find many instantiation sites requiring SizingInfo threading

## Definition of Done

- [ ] All FCF instances use with_capacity()
- [ ] Memory profile flat during training
- [ ] 1-3% performance improvement
- [ ] Examples produce identical results
