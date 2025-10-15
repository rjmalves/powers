# PAR-V2-012: Integrate PAR into Backward Pass

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 3 (Cut Generation)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

The backward pass generates cuts by:
1. Solving subproblems at sampled states
2. Extracting dual variables
3. Generating cuts
4. Adding cuts to previous stage

For PAR models, steps 2-3 now include lag states. Need to ensure backward pass orchestration handles the extended state space correctly.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 2, Section 2.3
- Code: `src/sddp/backward_pass.rs`

---

## Acceptance Criteria

- [ ] Backward pass handles extended state space
- [ ] Lag duals extracted and passed to cut generation
- [ ] Cuts with lag coefficients added to cut pool
- [ ] No regressions for non-PAR models
- [ ] Integration tests pass

---

## Tasks

### Implementation

- [ ] Update backward_pass.rs to use extended state
- [ ] Ensure dual extraction includes lags (PAR-V2-010)
- [ ] Ensure cut generation includes lags (PAR-V2-011)
- [ ] Test backward pass produces valid cuts

### Testing

- [ ] Test backward pass with PAR state
- [ ] Verify cuts added to pool correctly
- [ ] Regression test for non-PAR models

---

## Dependencies

### Blocked By

- ✅ PAR-V2-010: Lag dual extraction
- ✅ PAR-V2-011: Lag in cuts

---

## Estimated Effort

**3 story points** (1-2 days)
