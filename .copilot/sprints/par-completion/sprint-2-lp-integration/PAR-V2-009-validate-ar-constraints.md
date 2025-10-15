# PAR-V2-009: Validate AR Constraint Structure

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 2 (LP Integration)  
**Story Points**: 2  
**Priority**: 🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

After PAR-V2-007 adds AR dynamics constraints, we need comprehensive validation to ensure mathematical correctness. This ticket focuses on **verification**, not implementation.

**What We're Validating**:
1. Constraint matrix structure matches theory
2. Coefficients are exactly correct (φ, σ, signs)
3. RHS values initialized correctly
4. Constraint count matches expectation

**Why Critical**: This is our safety net. If PAR-V2-007 has subtle bugs, this ticket must catch them before they propagate.

---

## Acceptance Criteria

- [ ] Hand-calculated test case validates constraint structure
- [ ] Coefficient extraction and comparison tests pass
- [ ] Constraint matrix inspection utility implemented
- [ ] All validation tests pass
- [ ] Edge cases covered (PAR(1), PAR(p), mixed systems)

---

## Tasks

### Implementation

- [ ] **Implement constraint matrix inspector**
  ```rust
  pub fn inspect_ar_constraint(
      model: &solver::Model,
      constraint_idx: usize,
  ) -> ConstraintInfo {
      // Extract constraint row from LP
      // Return structure: variable indices, coefficients, bounds
  }
  ```

- [ ] **Hand-calculated validation test**
  ```rust
  #[test]
  fn test_ar_constraint_matches_theory() {
      // PAR(2): φ_1=0.6, φ_2=0.3, σ=10, μ=100
      // Expected: inflow - 6.0·lag[0] - 3.0·lag[1] = 100
      let state = create_test_state();
      let subproblem = Subproblem::new(&state, ...);
      
      let constraint = inspect_ar_constraint(&subproblem.model, ar_constraint_idx);
      
      // Verify coefficients
      assert_eq!(constraint.coef(inflow_var), 1.0);
      assert_eq!(constraint.coef(lag0_var), -6.0);
      assert_eq!(constraint.coef(lag1_var), -3.0);
      assert_eq!(constraint.rhs(), 100.0);
  }
  ```

- [ ] **Coefficient precision test**
  ```rust
  #[test]
  fn test_coefficient_precision() {
      // Use irrational φ values to test floating-point precision
      // φ = sqrt(2)/10 ≈ 0.1414213562...
      // Verify we maintain sufficient precision
  }
  ```

- [ ] **Constraint count test**
  ```rust
  #[test]
  fn test_constraint_count() {
      // System: 3 hydros (2 PAR, 1 naive)
      // Verify exactly 2 AR constraints
      let constraints = count_constraints_by_type(&subproblem);
      assert_eq!(constraints.ar_dynamics, 2);
  }
  ```

### Testing

- [ ] Test PAR(1) constraint structure
- [ ] Test PAR(3) constraint structure
- [ ] Test mixed PAR and naive hydros
- [ ] Test seasonal parameter variation
- [ ] Test coefficient signs (all negative except inflow)

### Documentation

- [ ] Document validation methodology
- [ ] Add comments explaining test cases
- [ ] Create debugging guide for constraint issues

---

## Dependencies

### Blocked By

- ✅ PAR-V2-007: AR constraints implemented

### Blocks

- PAR-V2-010: Dual extraction (depends on constraint correctness)

---

## Estimated Effort

**2 story points** (1 day)
