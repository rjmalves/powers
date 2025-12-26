# Sprint 1: Deterministic Cut Slot Management

## Goals

- Remove all dynamic allocation fallbacks from cut management
- Implement deterministic slot calculation based on (iteration, forward_pass_idx)
- Simplify codebase by removing slot tracking data structures

## Tickets

| ID | Title | Points | Dependencies |
|----|-------|--------|--------------|
| TICKET-001b | Add slot_index to BendersCut and num_forward_passes to Subproblem | 1 | None |
| TICKET-002b | Implement deterministic slot calculation | 2 | TICKET-001b |
| TICKET-003b | Update cut addition flow to use deterministic slots | 2 | TICKET-002b |
| TICKET-004b | Update cut deactivation to use stored slot | 1 | TICKET-003b |
| TICKET-005b | Remove slot tracking data structures | 1 | TICKET-004b |
| TICKET-006b | Update SDDP call sites | 2 | TICKET-003b |
| TICKET-007b | Validation and testing | 1 | All above |

**Total**: 10 story points (~1 week)

## Dependencies

- **From Previous Epic**: Epic 1 complete (HiGHS preallocation working)
- **To Next Epic**: Enables Epic 2 (FCF) with simplified cut management

## Risks

- **Low Risk**: Cut returning logic - cuts that return should use same slot (same iter/fp)
- **Mitigation**: Returning cuts maintain their original slot_index

## Definition of Done

- [x] All tickets complete
- [x] No `allocate_cut_slot()` or `free_cut_slots` in codebase
- [x] All examples pass with identical results
- [x] Valgrind shows zero allocations during training
- [x] Performance ≥23% improvement maintained
- [x] Code reviewed and merged
