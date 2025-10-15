# PAR-V2-022: Add Numerical Validation Tests

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 5 (Runtime & Optimization)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

Need rigorous numerical validation to ensure PAR implementation is mathematically correct. Tests should verify:

1. **AR equation satisfied**: Realized inflows match AR dynamics
2. **State transition correct**: Lag evolution matches theory
3. **Cut validity**: Cuts are valid lower bounds
4. **Policy optimality**: Decisions are rational given state

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 4, Section 4.3

---

## Acceptance Criteria

- [ ] AR equation validation test
- [ ] State transition validation test
- [ ] Cut validity test
- [ ] Policy rationality test
- [ ] All tests pass with tight tolerances

---

## Tasks

### Implementation

- [ ] **AR equation test**
  ```rust
  #[test]
  fn test_ar_equation_satisfied() {
      // Run forward pass, track inflows and lags
      // For each realization: verify inflow = μ + σ·(Σ φ·lag) + σ·ε
      // Allow numerical tolerance but verify equation holds
  }
  ```

- [ ] **State transition test**
  ```rust
  #[test]
  fn test_state_transition_correctness() {
      // Track state through multiple stages
      // Verify: lag_{t+1}[0] == inflow_t
      // Verify: lag_{t+1}[k] == lag_t[k-1] for k>0
  }
  ```

- [ ] **Cut validity test**
  ```rust
  #[test]
  fn test_cut_lower_bound_property() {
      // Generate cuts in backward pass
      // Evaluate cuts at multiple states
      // Compare with true value (from forward simulation)
      // Verify: cut_value ≤ true_value (with tolerance)
  }
  ```

- [ ] **Policy rationality test**
  ```rust
  #[test]
  fn test_policy_rationality() {
      // Train policy
      // Test at boundary conditions:
      //   - Empty reservoir → should turbine less
      //   - Full reservoir → should turbine more or spill
      //   - High lag → should anticipate higher inflows
      // Verify decisions are economically rational
  }
  ```

### Testing

- [ ] Run validation tests on example systems
- [ ] Test with different AR orders (PAR(1), PAR(2), PAR(3))
- [ ] Test with seasonal parameter variation
- [ ] Test with multiple hydros

### Documentation

- [ ] Document validation methodology
- [ ] Explain test tolerances
- [ ] Add comments to test cases

---

## Dependencies

### Blocked By

- ✅ PAR-V2-021: End-to-end integration

---

## Estimated Effort

**3 story points** (1-2 days)
