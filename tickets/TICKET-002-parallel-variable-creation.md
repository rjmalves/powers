# [TICKET-002] Add Parallel Lag Variable Creation in Subproblem

**Sprint:** 1  
**Estimated Effort:** 5 story points (3 days)  
**Confidence:** High  
**Priority:** P0 - Critical path

## Context

With the new lag variable structures defined (TICKET-001), we need to populate them during subproblem variable creation. This ticket implements the variable creation logic that populates both the old unified structure (for backward compatibility during migration) and the new explicit structures (for new code).

The key challenge is correctly routing variables to the appropriate structure based on `TemporalModel::entity_type` (Load vs Inflow) and `entity_id`.

## Acceptance Criteria

- [ ] Given temporal models with both loads and inflows, when creating subproblem variables, then both old and new structures are populated identically
- [ ] Given a load with bus_id=2 and AR(2), when variables are created, then `load_lags.lags_by_bus[2]` contains exactly 2 variables
- [ ] Given an inflow with hydro_id=1 and AR(3), when variables are created, then `inflow_lags.lags_by_hydro[1]` contains exactly 3 variables
- [ ] Given a system with no AR models (all order 0), when variables are created, then both `load_lags` and `inflow_lags` are None
- [ ] Performance: Variable creation time should not increase by more than 5%

## Tasks

### Implementation

- [ ] Update `add_variables` method in `src/subproblem.rs` to create both structures
  - Initialize `LoadLagVariables` with `system.buses.len()`
  - Initialize `InflowLagVariables` with `system.hydros.len()`
  - Initialize old unified structure as before
  
- [ ] Modify lag variable creation loop to populate both structures
  - For each temporal model, create lag variables as before
  - Match on `model.entity_type`:
    - `UncertaintyType::Load` → store in `load_lags.lags_by_bus[model.entity_id]`
    - `UncertaintyType::Inflow` → store in `inflow_lags.lags_by_hydro[model.entity_id]`
  - Also store in old `lagged_state` structure
  
- [ ] Set fields to None if no lags exist
  - If `load_lags.total_lag_count() == 0` → `variables.load_lags = None`
  - If `inflow_lags.total_lag_count() == 0` → `variables.inflow_lags = None`
  
- [ ] Update `add_constraints` method for lag-fixing constraints
  - Create `LoadLagConstraints` and `InflowLagConstraints`
  - Populate in parallel with existing `lag_fixing_constraints`
  - Match on entity type to route to correct structure
  
- [ ] Ensure entity metadata is accessible for matching
  - Extract entity type and entity id from `TemporalModel`
  - Validate that entity_id is within bounds for respective entity type

### Testing

- [ ] Integration test: System with 2 buses (AR(0), AR(1)) and 3 hydros (AR(2), AR(0), AR(1))
  - Verify `load_lags.lags_by_bus[0]` is empty
  - Verify `load_lags.lags_by_bus[1]` has 1 variable
  - Verify `inflow_lags.lags_by_hydro[0]` has 2 variables
  - Verify `inflow_lags.lags_by_hydro[1]` is empty
  - Verify `inflow_lags.lags_by_hydro[2]` has 1 variable
  
- [ ] Integration test: System with only loads (no hydros with AR models)
  - Verify `load_lags` is Some and populated
  - Verify `inflow_lags` is None
  
- [ ] Integration test: System with only inflows (no loads with AR models)
  - Verify `inflow_lags` is Some and populated
  - Verify `load_lags` is None
  
- [ ] Integration test: System with no AR models (all entities AR(0))
  - Verify both `load_lags` and `inflow_lags` are None
  - Verify old `lagged_state` is also None
  
- [ ] Integration test: Large system (50 buses, 30 hydros)
  - Measure variable creation time vs baseline
  - Verify no memory leaks
  
- [ ] Regression test: All existing subproblem tests still pass

### Documentation

- [ ] Update doc comments in `add_variables` explaining dual population
- [ ] Add comment explaining backward compatibility approach
- [ ] Document entity type matching logic
- [ ] Update README.md if variable creation behavior is documented there

## Technical Notes

### Implementation Approach

The core logic looks like:

```rust
pub fn add_variables(&mut self, model: &mut solver::Model, system: &System) {
    // ... existing variable creation for alpha, stored_volume, etc. ...
    
    // Create both old and new lag variable structures
    let mut old_lagged_state = Vec::new();
    let mut load_lags = LoadLagVariables::new(system.buses.len());
    let mut inflow_lags = InflowLagVariables::new(system.hydros.len());
    
    for temporal_model in &self.temporal_models {
        let mut entity_lag_vars = Vec::new();
        
        // Create lag variables for this entity
        for lag_order in 0..temporal_model.max_ar_order {
            let var = model.add_column(
                0.0,  // objective coefficient
                0.0.., // lower bound 0, no upper bound
            );
            entity_lag_vars.push(var);
        }
        
        // Store in old structure
        old_lagged_state.push(entity_lag_vars.clone());
        
        // Store in new structure based on entity type
        match temporal_model.entity_type {
            UncertaintyType::Load => {
                let bus_id = temporal_model.entity_id;
                load_lags.lags_by_bus[bus_id] = entity_lag_vars;
            }
            UncertaintyType::Inflow => {
                let hydro_id = temporal_model.entity_id;
                inflow_lags.lags_by_hydro[hydro_id] = entity_lag_vars;
            }
        }
    }
    
    // Set fields (None if empty)
    self.variables.lagged_state = if old_lagged_state.is_empty() {
        None
    } else {
        Some(old_lagged_state)
    };
    
    self.variables.load_lags = if load_lags.total_lag_count() > 0 {
        Some(load_lags)
    } else {
        None
    };
    
    self.variables.inflow_lags = if inflow_lags.total_lag_count() > 0 {
        Some(inflow_lags)
    } else {
        None
    };
}
```

### Constraint Creation

Similar approach for `add_constraints`:

```rust
// Create lag-fixing constraints
let mut old_lag_constraints = Vec::new();
let mut load_lag_constraints = LoadLagConstraints::new(system.buses.len());
let mut inflow_lag_constraints = InflowLagConstraints::new(system.hydros.len());

for temporal_model in &self.temporal_models {
    let mut entity_constraints = Vec::new();
    
    for lag_order in 0..temporal_model.max_ar_order {
        let lag_var = /* get from variables */;
        let constraint = model.add_row(
            lag_var..=lag_var,  // Y_t-k = value (fixed by state)
            vec![(lag_var, 1.0)],
        );
        entity_constraints.push(constraint);
    }
    
    old_lag_constraints.push(entity_constraints.clone());
    
    match temporal_model.entity_type {
        UncertaintyType::Load => {
            load_lag_constraints.constraints_by_bus[temporal_model.entity_id] = entity_constraints;
        }
        UncertaintyType::Inflow => {
            inflow_lag_constraints.constraints_by_hydro[temporal_model.entity_id] = entity_constraints;
        }
    }
}
```

### Edge Cases

- Empty temporal_models list
- Temporal model with entity_id out of bounds for entity type
- Mixed AR orders (some 0, some >0)
- Very high AR orders (e.g., AR(10))

### Validation Points

- Total variable count should match between old and new structures
- Entity IDs must be valid for their respective entity counts
- Variable indices must be unique across all structures

### Performance Considerations

- `clone()` operations on `Vec<usize>` are cheap (Copy types)
- No additional solver calls (just organizing existing variables)
- Memory overhead is temporary during migration

## Dependencies

- Blocked by: TICKET-001 (requires structures to exist)
- Blocks: TICKET-003, TICKET-004, TICKET-005, TICKET-006, TICKET-007
- Related: None

## Definition of Done

- [ ] Both old and new structures populated identically
- [ ] All integration tests pass
- [ ] Performance overhead < 5%
- [ ] Code reviewed and approved
- [ ] No warnings or clippy issues
- [ ] Documentation complete
