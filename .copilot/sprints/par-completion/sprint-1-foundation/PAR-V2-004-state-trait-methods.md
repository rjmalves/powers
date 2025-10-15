# PAR-V2-004: Implement State Trait Methods

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation)  
**Story Points**: 2  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

`StorageAndInflowState` must implement the `State` trait to integrate with the SDDP framework. Key methods to implement:

1. **`initial_value()`**: Return initial state vector (storage + lags)
2. **`update_from_solution()`**: Update state after subproblem solution
3. **`get_value()`**: Return current state values
4. **`dimension()`**: Return state space dimension

Most logic delegates to base `StorageState`, with additions for lag buffers.

**References**:
- State trait: `src/state.rs`
- Existing pattern: `StorageState` implementation
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.2

---

## Acceptance Criteria

- [ ] All State trait methods implemented
- [ ] Methods delegate to base `StorageState` where appropriate
- [ ] Lag buffer updates integrated into `update_from_solution`
- [ ] State dimension includes storage + lag states
- [ ] No panics in trait methods

---

## Tasks

### Implementation

- [ ] **Implement `initial_value()`**
  ```rust
  fn initial_value(&self) -> Vec<f64> {
      let mut values = self.storage.initial_value();
      for (hydro_id, buffer) in &self.lag_buffers {
          values.extend(buffer.iter());
      }
      values
  }
  ```

- [ ] **Implement `update_from_solution()`**
  ```rust
  fn update_from_solution(&mut self, solution: &Solution, variables: &Variables) {
      self.storage.update_from_solution(solution, variables);
      for (hydro_id, _) in &self.ar_orders {
          let inflow = solution.get_value(variables.inflow[*hydro_id]);
          self.update_lag(*hydro_id, inflow);
      }
  }
  ```

- [ ] **Implement `get_value()`**
  ```rust
  fn get_value(&self) -> Vec<f64> {
      let mut values = self.storage.get_value();
      for (hydro_id, buffer) in &self.lag_buffers {
          values.extend(buffer.iter());
      }
      values
  }
  ```

- [ ] **Implement `dimension()`**
  ```rust
  fn dimension(&self) -> usize {
      let storage_dim = self.storage.dimension();
      let lag_dim: usize = self.ar_orders.values().sum();
      storage_dim + lag_dim
  }
  ```

### Testing

- [ ] **Test dimension calculation**
- [ ] **Test initial_value construction**
- [ ] **Test update_from_solution**
- [ ] **Test get_value after updates**

### Documentation

- [ ] Add doc comments to all trait methods
- [ ] Explain state vector structure

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: Struct exists
- ✅ PAR-V2-002: Buffer management
- ✅ PAR-V2-003: Constructor

### Blocks

- PAR-V2-006: Add lag variables to subproblem

---

## Estimated Effort

**2 story points** (1 day)
