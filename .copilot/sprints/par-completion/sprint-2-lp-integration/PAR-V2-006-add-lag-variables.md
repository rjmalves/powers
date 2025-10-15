# PAR-V2-006: Add Lag State Variables to Subproblem

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation → 2 LP Integration)  
**Story Points**: 3  
**Priority**: 🔥 Critical  
**Status**: 🔵 Not Started

---

## Context

Lag states (inflow_{t-1}, ..., inflow_{t-p}) must be added as **LP variables** to each subproblem. These are:
- **State variables** (passed between stages via Bellman recursion)
- **Continuous variables** (lower_bound, upper_bound)
- **Part of the state vector** for cut generation

**Why Variables Not Constants**: Lagged values must be LP variables so that:
1. They appear in the constraint matrix (for AR dynamics)
2. Their dual variables enter Benders cuts
3. The Bellman recursion correctly propagates their values

**References**:
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.3
- Existing pattern: `src/subproblem.rs` → Variables struct, add_variables_to_problem
- State vector: PAR-V2-004

---

## Acceptance Criteria

- [ ] Lag variables added to LP for each PAR hydro
- [ ] Variable bounds set appropriately (e.g., non-negative)
- [ ] Variables indexed in `Variables` struct for later access
- [ ] Variable count: p variables per PAR(p) hydro
- [ ] Variables initialized with current lag values from state

---

## Tasks

### Implementation

- [ ] **Step 1**: Extend Variables struct
  ```rust
  // In src/subproblem.rs
  pub struct Variables {
      pub load: Vec<usize>,
      pub generation: Vec<Vec<usize>>,
      pub storage: Vec<usize>,
      pub inflow: Vec<usize>,
      pub lag_inflow: HashMap<usize, Vec<usize>>,  // NEW: hydro_id → [lag_0, lag_1, ..., lag_{p-1}]
      pub future_cost: usize,
  }
  ```

- [ ] **Step 2**: Add variables in add_variables_to_problem
  ```rust
  // In subproblem construction
  let mut lag_inflow_vars = HashMap::new();
  
  if let Some(storage_inflow_state) = state.as_any().downcast_ref::<StorageAndInflowState>() {
      for (hydro_id, p) in &storage_inflow_state.ar_orders {
          let mut lag_vars = Vec::with_capacity(*p);
          for k in 0..*p {
              let lag_value = storage_inflow_state.get_lag(*hydro_id, k);
              let bounds = (0.0, f64::INFINITY);  // Non-negative inflows
              let var_idx = problem.add_variable(lag_value, bounds);
              lag_vars.push(var_idx);
          }
          lag_inflow_vars.insert(*hydro_id, lag_vars);
      }
  }
  
  variables.lag_inflow = lag_inflow_vars;
  ```

- [ ] **Step 3**: Initialize variables with current lag values
  - Variable initial value = current lag from state
  - This sets the starting point for LP solver

### Testing

- [ ] **Test lag variables added**
  ```rust
  #[test]
  fn test_lag_variables_added() {
      let state = create_par_state(); // PAR(2) hydro
      let subproblem = Subproblem::new(&state, ...);
      assert_eq!(subproblem.variables.lag_inflow[&0].len(), 2);
  }
  ```

- [ ] **Test variable initialization**
  ```rust
  #[test]
  fn test_lag_variable_initialization() {
      // State with lag[0]=100, lag[1]=90
      // Verify LP variables initialized to these values
  }
  ```

- [ ] **Test variable bounds**
  ```rust
  #[test]
  fn test_lag_variable_bounds() {
      // Verify bounds are (0, +inf)
  }
  ```

### Documentation

- [ ] Document lag variable indexing
- [ ] Explain why lags are variables not constants
- [ ] Add inline comments

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: Struct exists
- ✅ PAR-V2-002: Buffer management
- ✅ PAR-V2-004: State trait methods

### Blocks

- PAR-V2-007: AR dynamics constraints (needs lag variables)
- PAR-V2-010: Extract duals (needs lag variables)

---

## Estimated Effort

**3 story points** (1-2 days)
