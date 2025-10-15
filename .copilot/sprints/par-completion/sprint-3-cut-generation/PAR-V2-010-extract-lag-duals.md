# PAR-V2-010: Extract Dual Variables from Lag States

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 3 (Cut Generation)  
**Story Points**: 3  
**Priority**: 🔥🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

In SDDP, Benders cuts include dual variables (shadow prices) for all state variables. For PAR models:

**States**: [storage_1, ..., storage_n, lag_{1,0}, lag_{1,1}, ..., lag_{m,p-1}]

**Duals Needed**:
- Storage duals (existing): from hydro balance constraints
- **Lag duals (NEW)**: from AR dynamics constraints

The lag duals are the shadow prices of the AR constraint, representing the marginal value of having higher lagged inflow.

**Why This Matters**: These duals enter the Benders cuts with correct coefficients, propagating lag state values backward through time.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 2, Section 2.1
- Existing pattern: `src/sddp/backward_pass.rs` → dual extraction
- Theory: Benders decomposition with extended state space

---

## Acceptance Criteria

- [ ] Dual variables extracted from AR dynamics constraints
- [ ] Duals organized by (hydro_id, lag_index)
- [ ] Dual extraction integrated into backward pass
- [ ] Duals passed to cut generation function
- [ ] Test verifies duals are non-zero and meaningful

---

## Tasks

### Implementation

- [ ] **Step 1**: Extend DualVariables struct
  ```rust
  // In src/sddp/backward_pass.rs or src/cut.rs
  pub struct DualVariables {
      pub storage: Vec<f64>,
      pub lag_inflow: HashMap<usize, Vec<f64>>,  // NEW: hydro_id → [π_lag0, π_lag1, ...]
  }
  ```

- [ ] **Step 2**: Extract lag duals after LP solve
  ```rust
  fn extract_dual_variables(
      solution: &solver::Solution,
      constraints: &Constraints,
      state: &dyn State,
  ) -> DualVariables {
      // Extract storage duals (existing)
      let storage_duals = extract_storage_duals(solution, constraints);
      
      // Extract lag duals (NEW)
      let mut lag_duals = HashMap::new();
      if let Some(storage_inflow_state) = state.as_any().downcast_ref::<StorageAndInflowState>() {
          for (hydro_id, p) in &storage_inflow_state.ar_orders {
              let ar_constraint_idx = constraints.ar_dynamics[hydro_id];
              let dual = solution.get_dual(ar_constraint_idx);
              
              // For PAR(p), we have 1 AR constraint but p lag states
              // The dual of the AR constraint affects all lags
              // Store appropriately
              lag_duals.insert(*hydro_id, vec![dual; *p]);  // Simplified; may need refinement
          }
      }
      
      DualVariables {
          storage: storage_duals,
          lag_inflow: lag_duals,
      }
  }
  ```

- [ ] **Step 3**: Pass duals to cut generation
  ```rust
  // In backward_pass.rs
  let dual_vars = extract_dual_variables(&solution, &constraints, state);
  let cut = generate_cut(state_value, &dual_vars, ...);
  ```

### Testing

- [ ] **Test dual extraction**
  ```rust
  #[test]
  fn test_extract_lag_duals() {
      let subproblem = create_subproblem_with_par();
      let solution = subproblem.solve();
      let duals = extract_dual_variables(&solution, &constraints, &state);
      
      // Verify lag duals exist
      assert!(duals.lag_inflow.contains_key(&hydro_id));
      assert_eq!(duals.lag_inflow[&hydro_id].len(), p);
      
      // Verify duals are non-zero (in most cases)
      // Note: duals can be zero if constraint is not binding
  }
  ```

- [ ] **Test dual propagation to cut**
  ```rust
  #[test]
  fn test_dual_in_cut_generation() {
      // Generate cut with lag duals
      // Verify cut includes lag dual terms
  }
  ```

- [ ] **Numerical test: verify dual meaning**
  ```rust
  #[test]
  fn test_lag_dual_interpretation() {
      // Solve subproblem with lag[0] = 100
      // Resolve with lag[0] = 101
      // Verify ΔObjective ≈ π_lag0 · Δlag[0]
      // This validates the shadow price interpretation
  }
  ```

### Documentation

- [ ] Document dual variable meaning for lag states
- [ ] Explain why AR constraint dual affects all lags
- [ ] Add inline comments in extraction code

---

## Technical Notes

### Dual Interpretation

The dual variable π_AR of the AR dynamics constraint:

```
inflow - σ·(Σ φ_k·lag[k]) = μ + ε
```

Represents: "Marginal value of relaxing the AR relationship by one unit."

**For cut generation**: Each lag[k] contributes -σ·φ_k to the constraint, so the sensitivity of the objective to lag[k] is:

```
∂f/∂lag[k] = -σ·φ_k · π_AR
```

**Thus**: lag_dual[k] = -σ·φ_k · π_AR

**Refinement Needed**: The simplified implementation above may need adjustment based on how duals map to individual lag states.

### Alternative: Dual per Lag

If we add constraints that explicitly define lag relationships (e.g., lag[1] = lag[0]_prev), each would have its own dual. Current formulation has one AR constraint, so duals must be computed from that single shadow price.

**Decision Point**: Choose between:
1. One AR constraint → compute lag duals from φ coefficients
2. Multiple constraints → extract duals directly

Recommend Option 1 for now (matches PAR-V2-007).

---

## Dependencies

### Blocked By

- ✅ PAR-V2-007: AR constraints exist
- ✅ PAR-V2-009: AR constraints validated

### Blocks

- PAR-V2-011: Include lag duals in cut generation

---

## Estimated Effort

**3 story points** (1-2 days)

**Complexity**: Understanding dual-to-lag mapping
