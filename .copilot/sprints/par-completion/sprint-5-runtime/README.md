# Sprint 5: Runtime & Optimization (Week 8-9)

**Goal**: Complete SDDP integration, enable end-to-end training, and validate algorithm correctness.

**Duration**: 2 weeks  
**Total Story Points**: 16  
**Risk Level**: 🔴 EXTREME

---

## Overview

This sprint brings everything together. All infrastructure from Sprints 1-4 gets integrated into the main SDDP loop. This is where we find out if the state-space approach actually works.

**Key Milestone**: **PAR-V2-021** (End-to-End SDDP Integration) is the culmination of the entire epic.

---

## Sprint Tickets

| ID | Title | Points | Status | Priority |
|----|-------|--------|--------|----------|
| PAR-V2-020 | Update RHS with innovation-based realizations | 3 | 🔵 Not Started | 🔥🔥 Critical |
| PAR-V2-021 | End-to-end SDDP integration with PAR | 5 | 🔵 Not Started | 🔥🔥🔥 CRITICAL MILESTONE |
| PAR-V2-022 | Add numerical validation tests | 3 | 🔵 Not Started | 🔥 High |
| PAR-V2-023 | Performance benchmarking | 2 | 🔵 Not Started | Medium |

---

## Critical Path

```
PAR-V2-020 (RHS Updates) → PAR-V2-021 (End-to-End) → PAR-V2-022 (Validation)
                                                            ↓
                                                     PAR-V2-023 (Benchmark)
```

---

## The Critical Moment: PAR-V2-021

**PAR-V2-021** is the make-or-break ticket for the entire PAR implementation.

### What Makes It Critical

1. **Integration Complexity**: All components must work together
2. **Convergence Uncertainty**: Will the algorithm converge?
3. **Policy Quality**: Will the policy be reasonable?
4. **Numerical Stability**: Will the solver behave?
5. **Performance**: Will it be usably fast?

### Success Definition

**Minimal Success**:
- SDDP runs without crashing
- Gap decreases (doesn't have to converge fully)
- Solutions are feasible

**Full Success**:
- Algorithm converges to tolerance
- Policy produces rational decisions
- No numerical issues
- Performance acceptable (< 50% slower than non-PAR)

### If PAR-V2-021 Fails

**Failure Modes**:
1. **Non-convergence**: Gap doesn't decrease
   - **Debug**: Check cut validity (PAR-V2-022)
   - **Check**: Are lag coefficients in cuts correct?
   - **Verify**: Dual extraction from AR constraints

2. **Numerical instability**: inf/nan, solver errors
   - **Debug**: Print constraint matrix
   - **Check**: Are coefficients reasonable magnitude?
   - **Verify**: Constraint matrix is well-conditioned

3. **Infeasibility**: LP becomes infeasible
   - **Debug**: Which constraints are conflicting?
   - **Check**: RHS updates correct (PAR-V2-020)?
   - **Verify**: Innovation values reasonable?

4. **Poor policy**: Converges but decisions are irrational
   - **Debug**: Evaluate policy at boundary states
   - **Check**: Are cuts approximating value function correctly?
   - **Verify**: State transitions preserve feasibility?

**Escalation Path**:
1. ⛔ Stop all work on Sprint 6
2. 🔍 Debug systematically (component by component)
3. 📊 Compare with theoretical expectations
4. 🧪 Test with simplest possible case (2-stage, PAR(1), deterministic)
5. 👥 Architect review

---

## RHS Updates (PAR-V2-020)

Before PAR-V2-021 can succeed, PAR-V2-020 must be correct.

**What it does**: Updates AR constraint RHS with scenario realizations:

```
Initial RHS: μ
Updated RHS: μ + σ·ε_t
```

**Why critical**: This is where stochastic realizations enter the subproblem. Without this, all subproblems are deterministic (no scenarios).

**Timing**:
```
1. Sample innovation ε_t
2. Update AR constraint RHS = μ + σ·ε
3. Solve subproblem
4. Extract solution
5. Update state
```

**Must happen**: Before each subproblem solve in both forward and backward passes.

---

## Numerical Validation (PAR-V2-022)

Critical validation tests to ensure correctness:

### Test 1: AR Equation Satisfied
```
Verify: inflow_realized = μ + σ·(Σ φ·lag) + σ·ε
```
Track inflows and lags through forward pass, verify equation holds.

### Test 2: State Transition Correct
```
Verify: lag_{t+1}[k] = lag_t[k-1] for k > 0
Verify: lag_{t+1}[0] = inflow_t
```

### Test 3: Cut Validity
```
Verify: cut_value(state) ≤ true_future_cost(state)
```
Cuts must be valid lower bounds.

### Test 4: Policy Rationality
```
Test boundary conditions:
- Empty reservoir → turbine less
- Full reservoir → turbine more or spill
- High lag → anticipate higher future inflows
```

---

## Success Criteria

- [ ] RHS updates implemented correctly
- [ ] End-to-end SDDP runs successfully
- [ ] Algorithm converges (or gap decreases)
- [ ] All validation tests pass
- [ ] Policy produces rational decisions
- [ ] No numerical instabilities
- [ ] Performance benchmarked and acceptable

---

## Validation Checklist

After Sprint 5, must verify:

- [ ] SDDP trains to convergence
- [ ] Lower bound ≤ Upper bound (always)
- [ ] Gap decreases monotonically (usually)
- [ ] Cuts have reasonable coefficients (not extreme values)
- [ ] State transitions are feasible
- [ ] AR equation satisfied in all simulations
- [ ] Policy evaluation produces reasonable costs
- [ ] Out-of-sample simulation succeeds
- [ ] Memory usage acceptable
- [ ] Training time < 2x non-PAR (ideally)

---

## Dependencies

**Blocked By**: All previous sprints (cumulative integration)

**Blocks**: Sprint 6 (Documentation) - can't document until it works

---

## Files to Edit

- `src/sddp/mod.rs` - Main SDDP loop integration
- `src/lib.rs` - Factory API integration
- `src/subproblem.rs` - RHS update methods
- `src/sddp/forward_pass.rs` - Integrate RHS updates
- `src/sddp/backward_pass.rs` - Integrate RHS updates
- `tests/test_sddp_algorithm.rs` - End-to-end tests
- `tests/test_numerical_validation.rs` - New validation tests
- `benches/par_benchmarks.rs` - Performance benchmarks

---

## Sprint Goal

**By end of Sprint 5**: PAR implementation is functionally complete and validated.

**Deliverable**: Working SDDP algorithm that trains policies for PAR models, with numerical validation confirming correctness.

**Decision Point**: If PAR-V2-021 fails, Sprint 6 is blocked until issues are resolved. Better to take extra time here than ship broken code.

---

## Notes

### Take Your Time

This sprint determines whether months of work have produced a correct implementation. Don't rush.

### Test Incrementally

Don't jump straight to complex systems. Start with:
1. 2-stage, PAR(1), deterministic (ε=0)
2. 2-stage, PAR(1), stochastic
3. 3-stage, PAR(2), stochastic
4. Multi-season, multiple hydros

### Debug Systematically

If something doesn't work:
1. Identify which component is failing
2. Test component in isolation
3. Compare with theoretical expectation
4. Fix and re-test integration

### Celebrate When It Works

If PAR-V2-021 succeeds and validation passes, this is a major achievement. The state-space formulation works and PAR models are now supported.

**Then proceed to Sprint 6 to polish and document.**
