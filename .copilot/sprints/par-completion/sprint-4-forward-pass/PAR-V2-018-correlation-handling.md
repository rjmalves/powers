# PAR-V2-018: Handle Innovation Correlation

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 4 (Forward Pass & Scenario Handling)  
**Story Points**: 3  
**Priority**: Medium  
**Status**: 🔵 Not Started

---

## Context

Innovations across hydros may be correlated. After sampling independent ε ~ N(0,1), apply correlation:

```
ε_correlated = L · ε_independent
```

where L is Cholesky decomposition of correlation matrix.

**Existing Infrastructure**: `src/correlation.rs` and `src/correlation_applicator.rs` handle this for absolute values. Adapt for innovations.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 3, Section 3.4
- Code: `src/correlation_applicator.rs`

---

## Acceptance Criteria

- [ ] Correlation applied to innovations
- [ ] Maintains N(0,1) marginals
- [ ] Induces correct correlation structure
- [ ] Tested numerically
- [ ] Works with existing correlation config

---

## Tasks

### Implementation

- [ ] Adapt CorrelationApplicator for innovations
- [ ] Test correlation preservation

---

## Estimated Effort

**3 story points** (1-2 days)
