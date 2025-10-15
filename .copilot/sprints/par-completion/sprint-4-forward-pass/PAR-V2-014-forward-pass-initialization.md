# PAR-V2-014: Integrate PAR into Forward Pass Initialization

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 2  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

Forward pass starts from initial state and simulates forward through time. For PAR models:

**Initial State**: Must include initial lag values
- storage_0 (existing)
- lag_{j,0} for each PAR hydro j (NEW)

These come from `InitialCondition` and are used to construct the initial `StorageAndInflowState`.

**References**:
- Code: `src/sddp/forward_pass.rs`
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.1

---

## Acceptance Criteria

- [ ] Forward pass initializes state with lag values
- [ ] Initial lags taken from InitialCondition
- [ ] First subproblem solved with correct initial lags
- [ ] No regressions for non-PAR models

---

## Tasks

### Implementation

- [ ] Update forward pass initialization
- [ ] Construct StorageAndInflowState with initial lags
- [ ] Verify first stage uses correct lag values

### Testing

- [ ] Test forward pass initialization with PAR
- [ ] Verify initial lag values propagated correctly

---

## Dependencies

### Blocked By

- ✅ PAR-V2-003: Constructor with initial lags

---

## Estimated Effort

**2 story points** (1 day)
