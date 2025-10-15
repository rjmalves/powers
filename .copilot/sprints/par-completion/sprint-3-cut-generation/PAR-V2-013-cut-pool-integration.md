# PAR-V2-013: Integrate PAR into Cut Pool

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 3 (Cut Generation)  
**Story Points**: 2  
**Priority**: Medium  
**Status**: 🔵 Not Started

---

## Context

Cut pool manages cuts: storage, selection, aggregation, pruning. With lag coefficients, need to verify cut pool operations handle extended dimension.

**Key Operations**:
- Cut storage: serialize cuts with lag coefficients
- Cut selection: evaluate cuts at states with lags
- Cut dominance: compare cuts in extended space
- Cut pruning: remove dominated cuts

**References**:
- Code: `src/sddp/cut_pool.rs` or equivalent
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 2

---

## Acceptance Criteria

- [ ] Cut pool stores cuts with lag coefficients
- [ ] Cut selection evaluates cuts at extended states
- [ ] Cut dominance checks consider lag dimension
- [ ] No panics with PAR cuts
- [ ] Memory usage acceptable

---

## Tasks

### Implementation

- [ ] Verify cut pool stores extended Cut struct
- [ ] Update cut selection if needed
- [ ] Update cut dominance checks if needed
- [ ] Test cut pool operations with PAR cuts

### Testing

- [ ] Test storing/retrieving PAR cuts
- [ ] Test cut selection with lags
- [ ] Test cut pruning with lags

---

## Dependencies

### Blocked By

- ✅ PAR-V2-011: Cuts with lag coefficients

---

## Estimated Effort

**2 story points** (1 day)
