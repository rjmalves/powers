# PAR-V2-019: Implement Scenario Tree Construction for PAR

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 2  
**Priority**: Medium  
**Status**: 🔵 Not Started

---

## Context

SDDP backward pass needs scenario trees (bushiness). For PAR:

**Scenario** = vector of innovations, one per stage
**Tree Node** = specific combination of innovations up to stage t

Ensure scenario tree structure handles innovations correctly.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.5
- Code: `src/scenario.rs`

---

## Acceptance Criteria

- [ ] Scenario tree built with innovations
- [ ] Tree structure correct (branching)
- [ ] Works with existing SDDP loop
- [ ] Tested with multiple scenarios

---

## Tasks

- [ ] Verify scenario tree works with innovation-based scenarios
- [ ] Test tree construction
- [ ] Validate scenario independence

---

## Estimated Effort

**2 story points** (1 day)
