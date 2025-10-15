# Sprint 1: Foundation (Week 1-2)

**Goal**: Build the foundational data structures and state management for PAR models.

**Duration**: 2 weeks  
**Total Story Points**: 13

---

## Overview

This sprint establishes the core infrastructure needed for PAR model state representation. We create the `StorageAndInflowState` struct that extends storage-only state with lag buffers for autoregressive dynamics.

**Key Deliverable**: Fully functional state representation that can store and manage lagged inflow values.

---

## Sprint Tickets

| ID | Title | Points | Status |
|----|-------|--------|--------|
| PAR-V2-001 | Create StorageAndInflowState struct | 3 | 🔵 Not Started |
| PAR-V2-002 | Implement lag buffer management | 3 | 🔵 Not Started |
| PAR-V2-003 | Implement constructor | 3 | 🔵 Not Started |
| PAR-V2-004 | Implement State trait methods | 2 | 🔵 Not Started |
| PAR-V2-005 | Extract AR parameters from config | 2 | 🔵 Not Started |

---

## Critical Path

```
PAR-V2-001 (Struct) → PAR-V2-002 (Buffer) → PAR-V2-003 (Constructor) → PAR-V2-004 (Trait)
                                                    ↑
                                                    |
                                            PAR-V2-005 (Parameters)
```

**Parallelization**: PAR-V2-005 can be developed in parallel with PAR-V2-002/003 if API contract is defined upfront.

---

## Success Criteria

- [ ] `StorageAndInflowState` struct compiles and passes all tests
- [ ] Lag buffers correctly manage circular history
- [ ] State trait methods work for mixed PAR/naive hydros
- [ ] AR parameters extracted and validated
- [ ] All unit tests pass
- [ ] No regressions in existing storage-only state tests

---

## Dependencies

**External**: None (foundational work)

**Blocks**: Sprint 2 (LP Integration) - can't add lag variables without state struct

---

## Notes

- Take time to get the foundation right - everything else builds on this
- Consider edge cases: single hydro, all PAR, all naive, mixed
- Test incrementally as each component is built
- Pay attention to memory efficiency (circular buffer is key)

---

## Files to Edit

- `src/state.rs` - Add StorageAndInflowState
- `src/input.rs` - AR parameter extraction
- `src/initial_condition.rs` - Add lag initial values
- `tests/test_state.rs` - Add PAR state tests
