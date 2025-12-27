# Sprint 1: Core Cut Pool Preallocation

## Status: ✅ COMPLETE

## Goals

- Implement slot computation utility
- Add in-place cut update methods
- Implement full cut pool preallocation
- Integrate with FCF initialization
- Update cut addition to use slot-based access

## Tickets

| ID | Title | Points | Status |
|----|-------|--------|--------|
| TICKET-001 | Add slot computation utility function | 2 | ✅ Done |
| TICKET-002 | Add BendersCut::update() for in-place modification | 3 | ✅ Done |
| TICKET-003 | Implement BendersCutPool::preallocate() | 5 | ✅ Done |
| TICKET-004 | Modify FCF initialization to use preallocation | 5 | ✅ Done |
| TICKET-005 | Update add_cuts_batch() to use slot-based access | 5 | ✅ Done |
| TICKET-006 | Update cut domination for preallocated pool | 3 | ✅ Done |

**Total Points**: 23

## Dependencies

- **From Previous Sprint**: None (first sprint)
- **To Next Sprint**: Enables Epic 2 (State Pool Preallocation)

## Risks

- Slot computation off-by-one errors → mitigated with debug_assert!
- Cut domination logic changes may affect algorithm → validated with tests

## Definition of Done

- [x] All 6 tickets complete
- [x] All existing tests pass (`cargo test`)
- [x] New unit tests for slot computation and update methods
- [ ] Lower bounds match baseline on examples 01, 05 (needs validation)
- [ ] Code reviewed and merged
