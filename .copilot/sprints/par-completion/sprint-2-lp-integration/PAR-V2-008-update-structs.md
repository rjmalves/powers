# PAR-V2-008: Update Variables and Constraints Structs

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 2 (LP Integration)  
**Story Points**: 1  
**Priority**: Medium  
**Status**: 🔵 Not Started

---

## Context

The `Variables` and `Constraints` structs in `src/subproblem.rs` need minor updates to accommodate PAR state variables and AR dynamics constraints. This is largely done in PAR-V2-006 and PAR-V2-007, but needs finalization and testing.

---

## Acceptance Criteria

- [ ] Variables struct includes `lag_inflow: HashMap<usize, Vec<usize>>`
- [ ] Constraints struct includes `ar_dynamics: HashMap<usize, usize>`
- [ ] Struct documentation updated
- [ ] No breaking changes to existing code

---

## Tasks

- [ ] Verify struct extensions from PAR-V2-006/007
- [ ] Add doc comments
- [ ] Update any Display/Debug implementations
- [ ] Test serialization if applicable

---

## Dependencies

### Blocked By

- ✅ PAR-V2-006: Lag variables
- ✅ PAR-V2-007: AR constraints

---

## Estimated Effort

**1 story point** (half day)
