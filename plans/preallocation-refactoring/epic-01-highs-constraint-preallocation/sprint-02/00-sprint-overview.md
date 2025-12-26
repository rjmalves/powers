# Sprint 2: Integration & Validation

## Goals

- Update cut addition to use preallocated slots and coefficient changes
- Implement cut removal via bound relaxation
- Add slot reuse for cut selection
- Integrate preallocation at training start
- Validate correctness and memory profile

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| TICKET-004 | Update cut addition to use coefficient changes | 3 | TICKET-003 |
| TICKET-005 | Implement cut removal via bound relaxation | 2 | TICKET-003 |
| TICKET-006 | Add cut slot reuse for selection | 2 | TICKET-004, TICKET-005 |
| TICKET-007 | Integration and validation | 2 | TICKET-006 |

## Dependencies

- **From Previous Sprint**: TICKET-003 provides cut slot infrastructure
- **To Next Epic**: Epic 2 (FCF preallocation)

## Risks

- Presolve may remove inactive constraints (test behavior)
- Cut selection integration may require careful slot tracking

## Definition of Done

- [ ] All tickets complete
- [ ] Zero `Highs_addRow` calls during training
- [ ] Memory profile flat during training (±1%)
- [ ] Examples 01 and 07 produce identical results
- [ ] Performance improvement ≥3%
