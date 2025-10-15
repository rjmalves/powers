# Sprint 2: LP Integration (Week 3) ⚠️ CRITICAL WEEK

**Goal**: Integrate AR dynamics into the linear programming formulation as state variables and constraints.

**Duration**: 1 week  
**Total Story Points**: 11  
**Risk Level**: 🔴 EXTREME

---

## ⚠️ THIS IS THE MOST CRITICAL SPRINT

**Why Critical**: This is where the state-space formulation actually enters the mathematical optimization. Everything before this builds infrastructure. Everything after depends on getting this right.

**Key Insight**: AR dynamics are **constraints in the LP**, not external scenario generation. This is what makes the approach mathematically correct for SDDP.

---

## Sprint Tickets

| ID | Title | Points | Status | Priority |
|----|-------|--------|--------|----------|
| PAR-V2-006 | Add lag state variables to subproblem | 3 | 🔵 Not Started | 🔥 Critical |
| PAR-V2-007 | Add AR dynamics as LP constraints | 5 | 🔵 Not Started | 🔥🔥🔥 MOST CRITICAL |
| PAR-V2-008 | Update Variables/Constraints structs | 1 | 🔵 Not Started | Medium |
| PAR-V2-009 | Validate AR constraint structure | 2 | 🔵 Not Started | 🔥 Critical |

---

## Critical Path

```
PAR-V2-006 (Lag Variables) → PAR-V2-007 (AR Constraints) → PAR-V2-009 (Validation)
                                          ↓
                                   PAR-V2-008 (Struct Updates)
```

**NO PARALLELIZATION**: These tickets must be done in strict sequence.

---

## The Make-or-Break Ticket: PAR-V2-007

**PAR-V2-007 (Add AR Constraints)** is the single most important ticket in the entire epic.

**What it does**:
```
Adds constraint: inflow_t - σ·(Σ φ_k·lag_{t-k}) = μ + ε_t
```

**Why it matters**:
- AR coefficients (φ) enter the constraint matrix → LP generates correct dual variables
- Dual variables from this constraint → enter Benders cuts
- Wrong implementation → entire PAR policy will be incorrect

**Before proceeding with PAR-V2-007**:
1. ✅ Complete PAR-V2-006 (lag variables must exist)
2. ✅ Review mathematical formulation in plan
3. ✅ Hand-calculate expected constraint structure
4. ✅ Plan comprehensive validation tests
5. ✅ Have architect available for review

**After PAR-V2-007**:
1. ✅ Run PAR-V2-009 validation immediately
2. ✅ Inspect constraint matrix manually
3. ✅ Verify coefficients match theory
4. ✅ Get architect sign-off before proceeding

---

## Success Criteria

- [ ] Lag state variables added to all subproblems
- [ ] AR dynamics constraints correctly formulated
- [ ] Constraint coefficients match mathematical theory
- [ ] Hand-calculated validation passes
- [ ] LP solves successfully with AR constraints
- [ ] Dual variables extracted successfully
- [ ] No numerical instabilities

---

## Validation Checklist

After Sprint 2, verify:

- [ ] Constraint count = number of PAR hydros
- [ ] Coefficient for inflow_t = 1.0
- [ ] Coefficient for lag[k] = -σ·φ_k
- [ ] RHS initialized to μ (innovations added later)
- [ ] Constraint is equality (lower = upper = μ)
- [ ] LP remains feasible after adding constraints
- [ ] Solver produces non-zero duals for AR constraints

---

## Dependencies

**Blocked By**: Sprint 1 (Foundation) - needs state struct

**Blocks**: Sprint 3 (Cut Generation) - needs AR constraints for duals

---

## Risk Mitigation

**If things go wrong**:
1. ⛔ STOP immediately - don't proceed to Sprint 3
2. 🔍 Debug with simple test case (PAR(1), round numbers)
3. 📊 Print constraint matrix and verify structure
4. 🧪 Hand-calculate expected coefficients
5. 👥 Escalate to architect

**Don't rush this sprint.** Better to take an extra week to get it right than discover issues later.

---

## Files to Edit

- `src/subproblem.rs` - Add Variables, Constraints extensions
- `src/state.rs` - Implement add_constraints_to_subproblem for PAR
- `tests/test_subproblem_construction.rs` - Add constraint validation tests
- `tests/test_ar_constraints.rs` - New file for AR-specific tests

---

## Notes for Implementer

- Use existing `StorageState::add_constraints_to_subproblem` as pattern
- Season ID lookup mechanism TBD (decide in PAR-V2-007)
- Constraint indices must be stored for later RHS updates
- Consider adding debug prints during development
- Verify each step before proceeding

**Remember**: This sprint determines whether the state-space approach works. Everything depends on correctness here.
