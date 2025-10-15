# PAR-V2-011: Include Lag Duals in Cut Generation

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 3 (Cut Generation)  
**Story Points**: 5  
**Priority**: 🔥🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

Benders cuts in SDDP have the form:

```
θ_{t+1} ≥ α + Σ β_storage_i · storage_i + Σ β_lag_{j,k} · lag_{j,k}
```

Currently, cuts only include storage state coefficients. For PAR models, we must add lag state coefficients.

**Mathematical Foundation**:

The cut coefficient for each state variable equals its dual variable (shadow price). With extended state space:

```
β_storage_i = π_storage_i  (from hydro balance constraint)
β_lag_{j,k} = π_lag_{j,k}  (from AR dynamics constraint)
```

**Why Critical**: Without lag coefficients, cuts won't correctly represent future costs as a function of lag states, breaking the Bellman recursion for PAR models.

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 2, Section 2.2
- Existing code: `src/cut.rs` → Cut struct and generation
- Theory: Bellman equation with extended state space

---

## Acceptance Criteria

- [ ] Cut struct extended to include lag coefficients
- [ ] Cut generation function includes lag duals
- [ ] Cut evaluation includes lag state terms
- [ ] Cuts tested numerically for correctness
- [ ] Backward pass integration complete

---

## Tasks

### Implementation

- [ ] **Step 1**: Extend Cut struct
  ```rust
  // In src/cut.rs
  pub struct Cut {
      pub intercept: f64,
      pub storage_coefficients: Vec<f64>,
      pub lag_coefficients: HashMap<usize, Vec<f64>>,  // NEW: hydro_id → [β_lag0, β_lag1, ...]
  }
  ```

- [ ] **Step 2**: Update cut generation
  ```rust
  pub fn generate_cut(
      obj_value: f64,
      dual_vars: &DualVariables,
      current_state: &dyn State,
  ) -> Cut {
      // Compute intercept (existing logic)
      let intercept = compute_intercept(obj_value, dual_vars, current_state);
      
      // Storage coefficients (existing)
      let storage_coefficients = dual_vars.storage.clone();
      
      // Lag coefficients (NEW)
      let lag_coefficients = dual_vars.lag_inflow.clone();
      
      Cut {
          intercept,
          storage_coefficients,
          lag_coefficients,
      }
  }
  ```

- [ ] **Step 3**: Update cut evaluation
  ```rust
  impl Cut {
      pub fn evaluate(&self, state: &dyn State) -> f64 {
          let mut value = self.intercept;
          
          // Storage terms (existing)
          let storage_values = state.get_storage_values();
          for (i, &coef) in self.storage_coefficients.iter().enumerate() {
              value += coef * storage_values[i];
          }
          
          // Lag terms (NEW)
          if let Some(storage_inflow_state) = state.as_any().downcast_ref::<StorageAndInflowState>() {
              for (hydro_id, lag_coefs) in &self.lag_coefficients {
                  for (k, &coef) in lag_coefs.iter().enumerate() {
                      let lag_value = storage_inflow_state.get_lag(*hydro_id, k);
                      value += coef * lag_value;
                  }
              }
          }
          
          value
      }
  }
  ```

- [ ] **Step 4**: Update intercept calculation
  ```rust
  fn compute_intercept(
      obj_value: f64,
      dual_vars: &DualVariables,
      current_state: &dyn State,
  ) -> f64 {
      let mut intercept = obj_value;
      
      // Subtract storage dual contributions (existing)
      for (i, &dual) in dual_vars.storage.iter().enumerate() {
          intercept -= dual * current_state.get_storage_value(i);
      }
      
      // Subtract lag dual contributions (NEW)
      if let Some(storage_inflow_state) = current_state.as_any().downcast_ref::<StorageAndInflowState>() {
          for (hydro_id, lag_duals) in &dual_vars.lag_inflow {
              for (k, &dual) in lag_duals.iter().enumerate() {
                  intercept -= dual * storage_inflow_state.get_lag(*hydro_id, k);
              }
          }
      }
      
      intercept
  }
  ```

### Testing

- [ ] **Test cut structure**
  ```rust
  #[test]
  fn test_cut_includes_lag_coefficients() {
      let cut = generate_cut_with_par_state();
      assert!(!cut.lag_coefficients.is_empty());
      assert_eq!(cut.lag_coefficients[&hydro_id].len(), p);
  }
  ```

- [ ] **Test cut evaluation**
  ```rust
  #[test]
  fn test_cut_evaluation_with_lags() {
      // Create state with known storage and lag values
      // Create cut with known coefficients
      // Verify evaluation matches hand calculation
      let state = create_test_state(); // storage=[100], lag=[50, 40]
      let cut = create_test_cut(); // β_storage=[0.5], β_lag=[0.1, 0.2], α=10
      let value = cut.evaluate(&state);
      // Expected: 10 + 0.5*100 + 0.1*50 + 0.2*40 = 10 + 50 + 5 + 8 = 73
      assert!((value - 73.0).abs() < 1e-10);
  }
  ```

- [ ] **Test intercept calculation**
  ```rust
  #[test]
  fn test_intercept_with_lags() {
      // Verify: α = f(x) - Σ π_i · x_i (including lag terms)
  }
  ```

- [ ] **Numerical validation test**
  ```rust
  #[test]
  fn test_cut_approximates_future_cost() {
      // Run backward pass to generate cuts
      // Evaluate cuts at various states
      // Compare with true future cost (from forward simulation)
      // Verify cut is a lower bound
  }
  ```

### Documentation

- [ ] Document extended cut structure
- [ ] Explain lag coefficient interpretation
- [ ] Add examples in doc comments

---

## Technical Notes

### Cut Intercept Formula

For standard Benders cut:

```
α = f(x*) - Σ π_i · x_i*
```

With extended state x = [storage, lags]:

```
α = f(x*) - Σ π_storage_i · storage_i* - Σ π_lag_{j,k} · lag_{j,k}*
```

This ensures: `θ ≥ α + π^T x` is tight at x*.

### State Downcasting Pattern

Need to safely access lag values from generic `State` trait:

```rust
if let Some(storage_inflow_state) = state.as_any().downcast_ref::<StorageAndInflowState>() {
    // Access lag values
}
```

**Alternative**: Add `get_lag_values()` to State trait (cleaner but more invasive).

### Cut Pool Compatibility

Ensure cuts with lag coefficients work with existing cut selection/aggregation:
- Cut dominance check must consider lag coefficients
- Cut pruning must handle extended dimension
- Level-1 cuts (cuts at boundaries) need lag handling

---

## Dependencies

### Blocked By

- ✅ PAR-V2-010: Lag dual extraction

### Blocks

- PAR-V2-012: Backward pass integration
- PAR-V2-013: Cut pool integration

---

## Estimated Effort

**5 story points** (2-3 days)

**Confidence**: Medium (complex integration, testing critical)

---

## Definition of Done

- [x] Cut struct extended
- [x] Cut generation updated
- [x] Cut evaluation updated
- [x] Intercept calculation updated
- [x] All tests pass
- [x] Numerical validation passes
- [x] Code reviewed
- [x] Documentation complete
- [x] Merged to branch
