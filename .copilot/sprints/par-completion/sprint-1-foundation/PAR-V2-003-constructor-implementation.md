# PAR-V2-003: Implement StorageAndInflowState Constructor

**Epic**: PAR Model Completion (State-Space Approach)  
**Sprint**: 1 (Foundation)  
**Story Points**: 3  
**Priority**: 🔥 High  
**Status**: 🔵 Not Started

---

## Context

We need to implement the constructor for `StorageAndInflowState` that:

1. Initializes the base `StorageState` component
2. Extracts AR orders and parameters from system/graph config
3. Allocates and initializes lag buffers
4. Sets up parameter lookup structures

This constructor is the entry point for creating PAR-enabled state representations.

**References**:
- Struct definition: PAR-V2-001
- Parameter extraction: PAR-V2-005 (can implement in parallel)
- Plan: `PAR-MODEL-COMPLETION-PLAN-V2.md` → Phase 1, Section 1.1

---

## Acceptance Criteria

### Functional Requirements

- [ ] Constructor accepts `SystemInput`, `GraphInput`, `InitialCondition`
- [ ] Base `StorageState` initialized correctly
- [ ] AR orders extracted for all hydros
- [ ] AR parameters extracted and stored
- [ ] Lag buffers initialized with initial conditions
- [ ] Non-PAR hydros handled correctly (no lag buffers)

### Technical Requirements

- [ ] No panics during construction
- [ ] Validates AR parameter completeness
- [ ] Efficient parameter storage (HashMap with (hydro_id, season_id) keys)
- [ ] Memory allocated only for PAR hydros

### Validation Requirements

- [ ] Unit tests verify construction succeeds
- [ ] Unit tests verify AR orders extracted correctly
- [ ] Unit tests verify lag buffers initialized
- [ ] Edge case: System with no PAR hydros
- [ ] Edge case: Mixed PAR and naive hydros

---

## Tasks

### Implementation

- [ ] **Step 1**: Define constructor signature
  ```rust
  impl StorageAndInflowState {
      pub fn new(
          system: &system::SystemInput,
          graph: &graph::GraphInput,
          initial_condition: &initial_condition::InitialCondition,
      ) -> Result<Self, crate::Error> {
          // Implementation
      }
  }
  ```

- [ ] **Step 2**: Initialize base storage
  ```rust
  let storage = state::StorageState::new(system, initial_condition)?;
  ```

- [ ] **Step 3**: Extract AR orders
  ```rust
  let mut ar_orders = HashMap::new();
  for hydro in &system.hydros {
      if let Some(process) = &hydro.inflow_process {
          if let StochasticProcessType::PAR { order } = process.process_type {
              ar_orders.insert(hydro.id, order);
          }
      }
  }
  ```

- [ ] **Step 4**: Extract AR parameters
  ```rust
  let ar_params = extract_ar_parameters(system, graph)?;
  // See PAR-V2-005 for extraction logic
  ```

- [ ] **Step 5**: Initialize lag buffers
  ```rust
  let mut lag_buffers = HashMap::new();
  for (hydro_id, p) in &ar_orders {
      let initial_lags = initial_condition.get_initial_lags(*hydro_id)?;
      let mut buffer = CircularBuffer::with_capacity(*p);
      for &lag in initial_lags {
          buffer.push(lag);
      }
      lag_buffers.insert(*hydro_id, buffer);
  }
  ```

- [ ] **Step 6**: Return constructed state
  ```rust
  Ok(StorageAndInflowState {
      storage,
      ar_orders,
      ar_params,
      lag_buffers,
  })
  ```

### Testing

- [ ] **Unit Test**: Construct state with PAR hydros
  ```rust
  #[test]
  fn test_construct_with_par_hydros() {
      let system = create_system_with_par_hydros();
      let initial_condition = create_initial_condition();
      let state = StorageAndInflowState::new(&system, &graph, &initial_condition);
      assert!(state.is_ok());
      assert_eq!(state.unwrap().ar_orders.len(), 2); // 2 PAR hydros
  }
  ```

- [ ] **Unit Test**: Construct state with no PAR hydros
  ```rust
  #[test]
  fn test_construct_with_naive_hydros_only() {
      let system = create_system_naive_only();
      let state = StorageAndInflowState::new(&system, &graph, &initial_condition);
      assert!(state.is_ok());
      assert_eq!(state.unwrap().ar_orders.len(), 0);
  }
  ```

- [ ] **Unit Test**: Lag buffers initialized correctly
  ```rust
  #[test]
  fn test_lag_buffers_initialized() {
      let state = StorageAndInflowState::new(&system, &graph, &initial_condition).unwrap();
      for (hydro_id, p) in &state.ar_orders {
          assert_eq!(state.lag_buffers[hydro_id].len(), *p);
      }
  }
  ```

- [ ] **Integration Test**: End-to-end construction from JSON
  ```rust
  #[test]
  fn test_construct_from_json() {
      let system = SystemInput::from_file("tests/fixtures/par_system.json").unwrap();
      let graph = GraphInput::from_file("tests/fixtures/par_graph.json").unwrap();
      let initial_condition = InitialCondition::default_for_system(&system);
      let state = StorageAndInflowState::new(&system, &graph, &initial_condition);
      assert!(state.is_ok());
  }
  ```

### Documentation

- [ ] Add doc comments explaining constructor parameters
- [ ] Document error cases (missing parameters, invalid config)
- [ ] Add usage example in module docs

---

## Technical Notes

### InitialCondition Extension

Need to add lag initial values to `InitialCondition`:

```rust
// In src/initial_condition.rs
pub struct InitialCondition {
    pub storage: Vec<f64>,
    pub lag_inflows: HashMap<usize, Vec<f64>>,  // NEW
}

impl InitialCondition {
    pub fn get_initial_lags(&self, hydro_id: usize) -> Result<&[f64], Error> {
        self.lag_inflows.get(&hydro_id)
            .map(|v| v.as_slice())
            .ok_or_else(|| Error::MissingInitialLags(hydro_id))
    }
}
```

**Default Strategy**: If no initial lags provided, use zeros or historical mean.

### Error Handling

Constructor should fail gracefully if:
- AR parameters missing for declared PAR hydro
- Initial lag count doesn't match AR order
- Invalid AR order (p < 1 or p > max)

```rust
if initial_lags.len() != p {
    return Err(Error::InvalidInitialLagCount {
        hydro_id,
        expected: p,
        found: initial_lags.len(),
    });
}
```

### Parameter Extraction Coordination

Constructor calls `extract_ar_parameters` from PAR-V2-005. These tickets can be developed in parallel if we define the API contract upfront:

```rust
fn extract_ar_parameters(
    system: &SystemInput,
    graph: &GraphInput,
) -> Result<HashMap<(usize, usize), Vec<f64>>, Error> {
    // Implementation in PAR-V2-005
}
```

---

## Dependencies

### Blocked By

- ✅ PAR-V2-001: StorageAndInflowState struct exists
- ✅ PAR-V2-002: Lag buffer management logic (can use stubs initially)

### Blocks

- PAR-V2-004: State trait methods (needs constructor to create instances)

### Related

- PAR-V2-005: AR parameter extraction (parallel work, API contract needed)

---

## Implementation Hints

**Start with Stub**: If PAR-V2-005 not ready, use stub that returns empty parameters:
```rust
fn extract_ar_parameters(...) -> Result<HashMap<...>, Error> {
    Ok(HashMap::new())  // Stub
}
```

**Test Incrementally**: Test construction with minimal config first, add complexity gradually

**Use Builder Pattern**: Consider implementing a builder for tests:
```rust
StorageAndInflowStateBuilder::new()
    .add_par_hydro(0, 2, vec![0.6, 0.3])
    .build()
```

---

## Estimated Effort

**3 story points** (1-2 days)

**Confidence**: High (standard constructor pattern)

---

## Definition of Done

- [x] Constructor implemented
- [x] Base storage initialized
- [x] AR parameters extracted and stored
- [x] Lag buffers initialized
- [x] All unit tests pass
- [x] Integration test from JSON passes
- [x] Error handling complete
- [x] Documentation complete
- [x] Code reviewed and merged
