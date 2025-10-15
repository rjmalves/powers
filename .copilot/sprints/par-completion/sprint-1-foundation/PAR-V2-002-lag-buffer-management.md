# PAR-V2-002: Implement Lag Buffer Management

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

The `StorageAndInflowState` struct includes a `CircularBuffer<f64>` to hold lagged inflow values for each hydro with PAR dynamics. We need to implement the logic that manages this buffer:

1. **Initialization**: Set initial lagged values from `InitialCondition`
2. **Update**: Roll buffer forward and insert new inflow observation
3. **Access**: Retrieve lag values for constraint/cut generation

**Why Circular Buffer**: Efficient O(1) insertion/access for fixed-size lag history. No memory allocations during simulation.

**References**:
- Struct definition: PAR-V2-001
- Similar pattern: `src/state.rs` → `StorageState::update_from_solution`
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.2

---

## Acceptance Criteria

### Functional Requirements

- [ ] Buffer initializes with correct lag values from `InitialCondition`
- [ ] Buffer updates correctly when new inflow realized
- [ ] Buffer maintains fixed size (p values for PAR(p))
- [ ] Latest value at lag[0], oldest at lag[p-1]
- [ ] Access methods return correct lag values by index

### Technical Requirements

- [ ] No panics on buffer operations
- [ ] No memory allocations during update
- [ ] Works for all AR orders (p=1 to p=max)
- [ ] Thread-safe if accessed during parallel operations

### Validation Requirements

- [ ] Unit tests verify update logic
- [ ] Unit tests verify indexing
- [ ] Edge case: PAR(1) with single lag
- [ ] Edge case: Initialization with zero lags

---

## Tasks

### Implementation

- [ ] **Step 1**: Implement initialization in constructor
  ```rust
  // In PAR-V2-003 constructor
  // For each PAR hydro:
  let initial_lags = initial_condition.get_initial_lags(hydro_id);
  let buffer = CircularBuffer::with_capacity(p);
  for &lag_value in initial_lags.iter() {
      buffer.push(lag_value);
  }
  ```

- [ ] **Step 2**: Implement buffer access method
  ```rust
  impl StorageAndInflowState {
      pub fn get_lag(&self, hydro_id: usize, k: usize) -> f64 {
          self.lag_buffers.get(&hydro_id)
              .and_then(|buffer| buffer.get(k))
              .expect("Lag buffer must exist for PAR hydros")
      }
  }
  ```

- [ ] **Step 3**: Implement buffer update method
  ```rust
  impl StorageAndInflowState {
      pub fn update_lag(&mut self, hydro_id: usize, new_inflow: f64) {
          if let Some(buffer) = self.lag_buffers.get_mut(&hydro_id) {
              buffer.push(new_inflow);  // Circular buffer handles rotation
          }
      }
  }
  ```

- [ ] **Step 4**: Integrate with State trait update method
  ```rust
  impl State for StorageAndInflowState {
      fn update_from_solution(&mut self, solution: &Solution, variables: &Variables) {
          // Update storage (delegate to base)
          self.storage.update_from_solution(solution, variables);
          
          // Update lag buffers
          for (hydro_id, _) in self.ar_orders.iter() {
              let inflow_value = solution.get_variable_value(variables.inflow[*hydro_id]);
              self.update_lag(*hydro_id, inflow_value);
          }
      }
  }
  ```

### Testing

- [ ] **Unit Test**: Buffer initialization
  ```rust
  #[test]
  fn test_lag_buffer_initialization() {
      // Create PAR(3) hydro with initial lags [100, 90, 80]
      // Verify buffer contains correct values
      assert_eq!(state.get_lag(hydro_id, 0), 100.0);
      assert_eq!(state.get_lag(hydro_id, 2), 80.0);
  }
  ```

- [ ] **Unit Test**: Buffer update
  ```rust
  #[test]
  fn test_lag_buffer_update() {
      // Initialize buffer with [100, 90, 80]
      // Update with new_inflow = 110
      // Verify buffer is now [110, 100, 90]
      state.update_lag(hydro_id, 110.0);
      assert_eq!(state.get_lag(hydro_id, 0), 110.0);
      assert_eq!(state.get_lag(hydro_id, 2), 90.0);
  }
  ```

- [ ] **Unit Test**: Multiple updates
  ```rust
  #[test]
  fn test_multiple_updates() {
      // PAR(2) with initial [50, 40]
      // Update sequence: 60, 70, 80
      // Verify final state is [80, 70]
  }
  ```

- [ ] **Unit Test**: Edge case - PAR(1)
  ```rust
  #[test]
  fn test_par1_single_lag() {
      // PAR(1) with single lag value
      // Verify updates work correctly
  }
  ```

### Documentation

- [ ] Document buffer semantics (lag[0] = most recent)
- [ ] Add inline comments explaining circular buffer behavior
- [ ] Document initialization requirements

---

## Technical Notes

### Circular Buffer Semantics

**Convention**: `lag[0]` = most recent observation, `lag[p-1]` = oldest

**Example** (PAR(3)):
```
Time t-3:  buffer = [50, 40, 30]  (oldest to newest)
Update(60):  buffer = [60, 50, 40]  (oldest value dropped)
Access: lag[0]=60, lag[1]=50, lag[2]=40
```

### InitialCondition Integration

Need to extend `InitialCondition` struct to include initial lag values:

```rust
// In src/initial_condition.rs
pub struct InitialCondition {
    pub storage: Vec<f64>,
    pub lag_inflows: HashMap<usize, Vec<f64>>,  // NEW
}
```

**TODO**: Coordinate with PAR-V2-003 (Constructor) for API design

### Performance Considerations

- Buffer capacity fixed at construction → no allocations
- Circular buffer: O(1) push, O(1) access
- HashMap lookup: O(1) average case

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: StorageAndInflowState struct exists

### Blocks

- PAR-V2-003: Constructor (needs buffer initialization logic)
- PAR-V2-004: State trait methods (needs buffer update logic)

### Related

- None

---

## Implementation Hints

**Start Simple**: Test with PAR(1) first, then generalize to PAR(p)

**Use Existing CircularBuffer**: Check if `circular_buffer` crate has the API we need, or implement simple version

**Verify Indexing**: Print buffer state after each update during development

---

## Estimated Effort

**3 story points** (1-2 days)

**Confidence**: High (straightforward data structure operations)

---

## Definition of Done

- [x] Buffer initialization implemented
- [x] Buffer update implemented
- [x] Buffer access methods implemented
- [x] All unit tests pass
- [x] Edge cases covered
- [x] Documentation complete
- [x] Code reviewed and merged
