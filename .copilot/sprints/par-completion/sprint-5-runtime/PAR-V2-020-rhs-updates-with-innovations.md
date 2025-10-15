# PAR-V2-020: Update RHS with Innovation-Based Realizations

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 5 (Runtime & Optimization)  
**Story Points**: 3  
**Priority**: 🔥🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

In PAR-V2-007, AR dynamics constraints were added with RHS = μ. During forward/backward passes, RHS must be updated with scenario realizations:

**Initial**: `RHS = μ`
**Updated**: `RHS = μ + σ·ε_t`

where ε_t is the sampled innovation.

This update happens:
1. **Before solving each subproblem**
2. **For each scenario realization**

**Why Critical**: This is where stochastic realizations enter the LP. Without this, all subproblems have deterministic RHS (no randomness).

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 4, Section 4.1
- Constraint indices: PAR-V2-007 (stored in Constraints struct)

---

## Acceptance Criteria

- [ ] RHS updated correctly before each subproblem solve
- [ ] Update uses innovation (ε) not absolute inflow
- [ ] Seasonal parameters (μ, σ) applied correctly
- [ ] Constraint indices used for efficient update
- [ ] Tested numerically

---

## Tasks

### Implementation

- [ ] **Implement RHS update function**
  ```rust
  impl Subproblem {
      pub fn update_ar_rhs_with_innovation(
          &mut self,
          hydro_id: usize,
          season_id: usize,
          innovation: f64,
          state: &StorageAndInflowState,
      ) {
          let constraint_idx = self.constraints.ar_dynamics[&hydro_id];
          let params = &state.ar_params[&(hydro_id, season_id)];
          let p = state.ar_orders[&hydro_id];
          
          let mu = params[p];
          let sigma = params[p + 1];
          
          let new_rhs = mu + sigma * innovation;
          self.model.set_rhs(constraint_idx, new_rhs);
      }
  }
  ```

- [ ] **Integrate into forward pass**
  ```rust
  // Before solving subproblem in forward pass
  for (hydro_id, innovation) in scenario.innovations.iter() {
      subproblem.update_ar_rhs_with_innovation(*hydro_id, season_id, *innovation, &state);
  }
  subproblem.solve();
  ```

- [ ] **Integrate into backward pass**
  ```rust
  // Before solving subproblem in backward pass
  for (hydro_id, innovation) in scenario.innovations.iter() {
      subproblem.update_ar_rhs_with_innovation(*hydro_id, season_id, *innovation, &state);
  }
  subproblem.solve();
  ```

### Testing

- [ ] **Test RHS update**
  ```rust
  #[test]
  fn test_rhs_update_with_innovation() {
      // μ=100, σ=10, ε=0.5
      // Expected RHS: 100 + 10·0.5 = 105
      let mut subproblem = create_subproblem();
      subproblem.update_ar_rhs_with_innovation(hydro_id, season_id, 0.5, &state);
      let rhs = subproblem.model.get_rhs(constraint_idx);
      assert!((rhs - 105.0).abs() < 1e-10);
  }
  ```

- [ ] **Test multiple hydros**
  ```rust
  #[test]
  fn test_rhs_update_multiple_hydros() {
      // Update RHS for 3 PAR hydros with different innovations
      // Verify each constraint updated correctly
  }
  ```

- [ ] **Integration test**
  ```rust
  #[test]
  fn test_forward_pass_with_rhs_updates() {
      // Run forward pass with innovation scenarios
      // Verify subproblems solved correctly
      // Check solution feasibility
  }
  ```

### Documentation

- [ ] Document RHS update timing
- [ ] Explain innovation-to-RHS transformation
- [ ] Add inline comments

---

## Technical Notes

### RHS Update Timing

**Critical**: Update RHS **before** solving, **after** state transition.

**Order**:
1. Transition to stage t (state now has lag_{t-1})
2. Sample innovation ε_t
3. Update AR constraint RHS with ε_t
4. Solve subproblem
5. Extract solution and update state for next stage

### Efficiency Considerations

**Constraint Index Lookup**: O(1) via HashMap

**Alternative**: Batch RHS updates using solver API (set_rhs_batch) if available

---

## Dependencies

### Blocked By

- ✅ PAR-V2-007: AR constraints exist
- ✅ PAR-V2-016: Innovation sampling
- ✅ PAR-V2-017: Innovation transformation

### Blocks

- PAR-V2-021: End-to-end SDDP integration

---

## Estimated Effort

**3 story points** (1-2 days)
