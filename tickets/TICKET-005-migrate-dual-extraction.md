# [TICKET-005] Migrate Dual Extraction to Use Explicit Structures

**Sprint:** 2  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** High  
**Priority:** P1 - High

## Context

The dual extraction methods in various State implementations currently iterate through the unified `lag_fixing_constraints` structure and filter by entity type to separate load and inflow duals. With explicit constraint structures now available, we can simplify this to direct access by entity ID.

This change improves performance and code clarity, and eliminates the need for entity type filtering logic.

**Affected functions:**
- `get_lag_duals_from_solution` in State trait implementations
- Any methods that extract dual values for state transitions

## Acceptance Criteria

- [ ] Given subproblem solution with dual values, when extracting load lag duals, then values are retrieved directly from `load_lag_constraints` by bus_id
- [ ] Given subproblem solution with dual values, when extracting inflow lag duals, then values are retrieved directly from `inflow_lag_constraints` by hydro_id
- [ ] Given system with mixed entities, when extracting duals, then no entity type filtering logic is used
- [ ] Given existing tests with dual extraction, when using new implementation, then all tests produce identical results
- [ ] Performance: Dual extraction should be 30-50% faster due to direct access

## Tasks

### Implementation

- [ ] Update `StorageAndInflowState::get_lag_duals_from_solution` in `src/state.rs`
  - Replace unified entity iteration with separate load/inflow loops
  - Use `load_lag_constraints.get_constraints(bus_id)` for loads
  - Use `inflow_lag_constraints.get_constraints(hydro_id)` for inflows
  - Remove entity type checking logic
  
- [ ] Update similar methods in other State implementations if they exist
  - Search for `get_lag_duals` pattern in state.rs
  - Apply same refactoring pattern
  
- [ ] Simplify dual value collection
  - Direct iteration over hydros/buses instead of entities
  - Use `map()` for cleaner collection
  
- [ ] Remove any helper functions that existed solely for entity filtering
  - Search for entity type matching helpers
  - Delete if no longer used

### Testing

- [ ] Unit test: Extract duals from system with 3 hydros, 5 buses
  - Populate constraint dual values with known values
  - Verify extracted load_lag_duals[bus_id] matches expected
  - Verify extracted inflow_lag_duals[hydro_id] matches expected

- [ ] Unit test: System where some entities have no lags (AR=0)
  - Verify empty vectors for entities with no lags
  - Verify non-empty vectors have correct length

- [ ] Regression test: Run existing dual extraction tests
  - Compare results before/after migration
  - Should be bitwise identical

- [ ] Integration test: Full SDDP iteration with dual extraction
  - Verify state transitions use correct dual values
  - Verify convergence not affected

- [ ] Performance benchmark: Dual extraction timing
  - Before: unified structure with filtering
  - After: explicit structures with direct access
  - Measure on system with 50 entities (20 hydros, 30 buses)

- [ ] Edge case: All constraints have zero dual values
  - Verify correct structure returned with zeros

- [ ] Edge case: Very large system (100+ entities)
  - Verify no performance regression
  - Check memory usage

### Documentation

- [ ] Update doc comments for dual extraction methods
- [ ] Explain the direct access pattern
- [ ] Document the relationship between entity_id and dual indices
- [ ] Add example showing how to access specific entity's duals

## Technical Notes

### Before (Current Implementation)

```rust
fn get_lag_duals_from_solution(
    &self,
    solution: &solver::Solution,
    system: &System,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut load_lag_duals = vec![Vec::new(); system.buses.len()];
    let mut inflow_lag_duals = vec![Vec::new(); system.hydros.len()];
    
    if let Some(lag_constraints) = &self.constraints.lag_fixing_constraints {
        // Need to iterate with entity metadata to know type
        for (entity_idx, entity_constraints) in lag_constraints.iter().enumerate() {
            let entity = &self.entity_metadata[entity_idx]; // Need to maintain this!
            let mut duals = Vec::new();
            
            for &constraint_idx in entity_constraints {
                duals.push(solution.rowdual[constraint_idx]);
            }
            
            // Filter by entity type
            match entity.entity_type {
                UncertaintyType::Load => {
                    load_lag_duals[entity.entity_id] = duals;
                }
                UncertaintyType::Inflow => {
                    inflow_lag_duals[entity.entity_id] = duals;
                }
            }
        }
    }
    
    (load_lag_duals, inflow_lag_duals)
}
```

**Problems:**
- Requires maintaining `entity_metadata` parallel structure
- O(n_entities) iteration even if only need inflows
- Entity type matching logic in every extraction
- Harder to parallelize (single loop over mixed types)

### After (Proposed Implementation)

```rust
fn get_lag_duals_from_solution(
    &self,
    solution: &solver::Solution,
    system: &System,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut load_lag_duals = vec![Vec::new(); system.buses.len()];
    let mut inflow_lag_duals = vec![Vec::new(); system.hydros.len()];
    
    // Extract load lag duals directly by bus_id
    if let Some(load_constraints) = &self.constraints.load_lag_constraints {
        for bus_id in 0..system.buses.len() {
            let constraints = load_constraints.get_constraints(bus_id);
            load_lag_duals[bus_id] = constraints.iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }
    
    // Extract inflow lag duals directly by hydro_id
    if let Some(inflow_constraints) = &self.constraints.inflow_lag_constraints {
        for hydro_id in 0..system.hydros.len() {
            let constraints = inflow_constraints.get_constraints(hydro_id);
            inflow_lag_duals[hydro_id] = constraints.iter()
                .map(|&idx| solution.rowdual[idx])
                .collect();
        }
    }
    
    (load_lag_duals, inflow_lag_duals)
}
```

**Benefits:**
- No entity metadata needed
- Can process loads and inflows independently (parallelizable)
- O(n_hydros) + O(n_buses) instead of O(n_entities) with filtering
- Clearer intent - explicitly processing each entity type
- Simpler code - no match/filter logic

### Performance Analysis

**Typical system:** 30 entities (20 buses, 10 hydros)

**Before:**
```
for entity in 0..30:
    get entity_type (lookup)
    match entity_type:
        Load: process
        Inflow: process
```
Operations: 30 iterations, 30 lookups, 30 branches

**After:**
```
for bus in 0..20:
    process load
for hydro in 0..10:
    process inflow
```
Operations: 30 iterations, 0 lookups, 0 branches

**Expected improvement:**
- 30% faster from eliminated lookups and branches
- Better cache locality from sequential access
- Potential for parallel processing (independent loops)

### Edge Cases

- System with no buses (only hydros)
- System with no hydros (only buses)
- Entity with AR(0) → empty constraint vector → empty dual vector
- Entity with AR(10) → 10 constraints → 10 dual values
- Solution where all duals are zero
- Solution where some duals are NaN (infeasible subproblem)

### Validation Strategy

During migration, add temporary validation:

```rust
#[cfg(feature = "migration_validation")]
{
    // Extract using old method
    let (old_load_duals, old_inflow_duals) = self.get_lag_duals_old(solution, system);
    
    // Extract using new method
    let (new_load_duals, new_inflow_duals) = self.get_lag_duals_new(solution, system);
    
    // Verify identical
    assert_eq!(old_load_duals, new_load_duals, "Load dual mismatch");
    assert_eq!(old_inflow_duals, new_inflow_duals, "Inflow dual mismatch");
}
```

### Alternative: Parallel Extraction

For very large systems, could parallelize:

```rust
use rayon::prelude::*;

let load_lag_duals: Vec<Vec<f64>> = (0..system.buses.len())
    .into_par_iter()
    .map(|bus_id| {
        let constraints = load_constraints.get_constraints(bus_id);
        constraints.iter()
            .map(|&idx| solution.rowdual[idx])
            .collect()
    })
    .collect();
```

However, probably not worth the overhead unless system has 1000+ buses.

## Dependencies

- Blocked by: TICKET-001, TICKET-002 (need explicit structures)
- Blocks: None
- Related: TICKET-004 (similar pattern), TICKET-003 (validation)

## Definition of Done

- [ ] All dual extraction uses explicit structures
- [ ] No entity type filtering remains in dual extraction code
- [ ] All existing tests pass with identical results
- [ ] Performance improvement measured and documented
- [ ] Code reviewed with focus on correctness
- [ ] No entity metadata required for dual extraction
