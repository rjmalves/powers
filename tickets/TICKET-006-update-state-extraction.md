# [TICKET-006] Update State Lag Extraction Methods

**Sprint:** 3  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** Medium-High  
**Priority:** P1 - High

## Context

State extraction methods read lag values from trajectories to set initial conditions for subproblems. Currently, these methods iterate through a unified entity list and filter by type. With explicit lag structures, we can simplify to direct indexed access.

**Key methods affected:**
- `extract_lags_from_trajectory` - extracts lag values from scenario realizations
- Lag buffer update logic - maintains historical lag values
- State initialization from previous solutions

## Acceptance Criteria

- [ ] Given trajectory with load and inflow realizations, when extracting lags, then load lags are stored in `load_lags` by bus_id and inflow lags in `inflow_lags` by hydro_id
- [ ] Given lag values to set in subproblem, when updating constraint RHS, then values are matched to constraints using explicit structures
- [ ] Given state with partial lag information (some entities have lags, others don't), when extracting, then empty vectors are correctly handled
- [ ] Given existing state extraction tests, when using new implementation, then all tests produce identical results
- [ ] Performance: State extraction should not regress (target: < 5% overhead)

## Tasks

### Implementation

- [ ] Identify all state extraction methods in `src/state.rs`
  - `extract_lags_from_trajectory`
  - Methods that set lag constraint RHS values
  - Lag buffer initialization/update methods
  
- [ ] Update `extract_lags_from_trajectory` to use explicit structures
  - Separate loops for load realizations and inflow realizations
  - Direct indexing by bus_id and hydro_id
  - Remove entity type checking
  
- [ ] Update lag constraint RHS setting logic
  - Use `load_lag_constraints.get_constraint(bus_id, lag_idx)` for loads
  - Use `inflow_lag_constraints.get_constraint(hydro_id, lag_idx)` for inflows
  - Eliminate unified constraint iteration
  
- [ ] Update lag buffer management
  - Maintain separate buffers for loads and inflows if needed
  - Or use explicit structures to organize buffer access
  
- [ ] Simplify state initialization methods
  - Direct access patterns for setting up subproblem state
  - Clear separation of load vs inflow state

- [ ] Remove helper functions for entity filtering if no longer needed

### Testing

- [ ] Unit test: Extract lags from trajectory with 3 buses, 2 hydros
  - Trajectory: [(load[b0]=10, load[b1]=20, load[b2]=30), (inflow[h0]=5, inflow[h1]=8)]
  - Verify load_lags[0] = [10], load_lags[1] = [20], etc.
  - Verify inflow_lags[0] = [5], inflow_lags[1] = [8]

- [ ] Unit test: Extract lags for entities with different AR orders
  - Bus 0: AR(2), Bus 1: AR(0), Bus 2: AR(1)
  - Hydro 0: AR(1), Hydro 1: AR(3)
  - Verify correct number of lags extracted for each entity

- [ ] Unit test: Set lag constraint RHS values from state
  - Create subproblem with lag constraints
  - Set RHS using extracted lag values
  - Verify constraint bounds are correctly updated in solver model

- [ ] Integration test: Multi-stage trajectory with state transitions
  - Run 5-stage problem
  - Extract state at each stage
  - Verify lag values propagate correctly through stages

- [ ] Regression test: Compare state values before/after migration
  - Same system, same seed, same trajectory
  - Extract states using old and new methods
  - Verify bitwise identical results

- [ ] Performance test: State extraction in hot path
  - Measure time for 10,000 state extractions
  - Compare before/after
  - Ensure < 5% regression

- [ ] Edge case: First stage (no previous lags)
  - Verify initial lag values handled correctly
  - Usually initialized to zero or mean

- [ ] Edge case: Trajectory with missing values
  - Handle NaN or missing realizations gracefully

### Documentation

- [ ] Update doc comments for state extraction methods
- [ ] Explain the separation of load and inflow lag extraction
- [ ] Document lag buffer structure and organization
- [ ] Add examples showing state initialization flow
- [ ] Update any architecture docs mentioning state extraction

## Technical Notes

### Current Approach (Conceptual)

```rust
fn extract_lags_from_trajectory(
    &self,
    trajectory: &Trajectory,
    temporal_models: &[TemporalModel],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut all_lags = Vec::new();
    
    // Extract lag values for all entities
    for (entity_idx, model) in temporal_models.iter().enumerate() {
        let mut entity_lags = Vec::new();
        for lag_order in 1..=model.max_ar_order {
            let value = trajectory.get_realization_at_lag(entity_idx, lag_order);
            entity_lags.push(value);
        }
        all_lags.push(entity_lags);
    }
    
    // Now filter by type (requires entity metadata)
    let load_lags = filter_by_type(all_lags, Load);
    let inflow_lags = filter_by_type(all_lags, Inflow);
    
    (load_lags, inflow_lags)
}
```

### Proposed Approach

```rust
fn extract_lags_from_trajectory(
    &self,
    trajectory: &Trajectory,
    system: &System,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut load_lags = vec![Vec::new(); system.buses.len()];
    let mut inflow_lags = vec![Vec::new(); system.hydros.len()];
    
    // Extract load lags directly by bus_id
    for bus_id in 0..system.buses.len() {
        let ar_order = self.get_load_ar_order(bus_id);
        let mut bus_lags = Vec::with_capacity(ar_order);
        
        for lag_order in 1..=ar_order {
            let value = trajectory.get_load_realization_at_lag(bus_id, lag_order);
            bus_lags.push(value);
        }
        
        load_lags[bus_id] = bus_lags;
    }
    
    // Extract inflow lags directly by hydro_id
    for hydro_id in 0..system.hydros.len() {
        let ar_order = self.get_inflow_ar_order(hydro_id);
        let mut hydro_lags = Vec::with_capacity(ar_order);
        
        for lag_order in 1..=ar_order {
            let value = trajectory.get_inflow_realization_at_lag(hydro_id, lag_order);
            hydro_lags.push(value);
        }
        
        inflow_lags[hydro_id] = hydro_lags;
    }
    
    (load_lags, inflow_lags)
}
```

### Setting Lag Constraint RHS

**Before:**
```rust
// Unified constraint iteration with filtering
for (entity_idx, entity_lag_values) in lag_values.iter().enumerate() {
    let entity = &self.entities[entity_idx];
    let constraints = &self.constraints.lag_fixing_constraints[entity_idx];
    
    for (lag_idx, &value) in entity_lag_values.iter().enumerate() {
        let constraint = constraints[lag_idx];
        model.set_rhs(constraint, value..=value); // Fix Y_{t-k} = value
    }
}
```

**After:**
```rust
// Separate explicit loops
if let Some(load_constraints) = &self.constraints.load_lag_constraints {
    for bus_id in 0..system.buses.len() {
        let lag_values = &load_lag_values[bus_id];
        let constraints = load_constraints.get_constraints(bus_id);
        
        for (lag_idx, &value) in lag_values.iter().enumerate() {
            let constraint = constraints[lag_idx];
            model.set_rhs(constraint, value..=value);
        }
    }
}

if let Some(inflow_constraints) = &self.constraints.inflow_lag_constraints {
    for hydro_id in 0..system.hydros.len() {
        let lag_values = &inflow_lag_values[hydro_id];
        let constraints = inflow_constraints.get_constraints(hydro_id);
        
        for (lag_idx, &value) in lag_values.iter().enumerate() {
            let constraint = constraints[lag_idx];
            model.set_rhs(constraint, value..=value);
        }
    }
}
```

### Trajectory Structure Considerations

May need to update `Trajectory` or `Realization` structures to provide:
- `get_load_realization(bus_id, time_offset)` 
- `get_inflow_realization(hydro_id, time_offset)`

Currently, `Realization` already has separated `load_lag_duals` and `inflow_lag_duals`, so this aligns well.

### Lag Buffer Management

If lag buffers exist for temporal state management:

```rust
struct LagBuffer {
    // Old: unified
    // values: Vec<VecDeque<f64>>,
    
    // New: explicit
    load_buffers: Vec<VecDeque<f64>>,   // indexed by bus_id
    inflow_buffers: Vec<VecDeque<f64>>, // indexed by hydro_id
}

impl LagBuffer {
    fn push_load_value(&mut self, bus_id: usize, value: f64) {
        self.load_buffers[bus_id].push_back(value);
        if self.load_buffers[bus_id].len() > self.max_ar_order {
            self.load_buffers[bus_id].pop_front();
        }
    }
    
    fn get_load_lags(&self, bus_id: usize) -> &VecDeque<f64> {
        &self.load_buffers[bus_id]
    }
}
```

### Edge Cases

- Stage 0 with no prior history → Initialize with zeros or means
- Trajectory shorter than AR order → Pad with zeros or extrapolate
- Entity with AR(0) → Empty lag vector
- Very high AR order (>10) → Ensure buffer management is efficient
- Stochastic process with missing data → Default to mean or fail gracefully

### Performance Considerations

- Pre-allocate vectors with correct capacity
- Avoid unnecessary cloning
- Cache AR orders to avoid repeated lookups
- Consider using iterator adapters for cleaner code without overhead

## Dependencies

- Blocked by: TICKET-001, TICKET-002 (need explicit structures)
- Blocks: None
- Related: TICKET-005 (similar extraction pattern), TICKET-007 (uses extracted lags)

## Definition of Done

- [ ] All state extraction uses explicit structures
- [ ] No entity filtering in extraction logic
- [ ] All tests pass with identical results to old implementation
- [ ] Performance within 5% of baseline
- [ ] Code reviewed with focus on state correctness
- [ ] Documentation updated for state management flow
