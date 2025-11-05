# [TICKET-001] Design and Implement Lag Variable Data Structures

**Sprint:** 1  
**Estimated Effort:** 3 story points (2 days)  
**Confidence:** High  
**Priority:** P0 - Blocker for all other tickets

## Context

This is the foundational ticket for the explicit lag separation epic. Currently, lag variables are stored in a unified `Vec<Vec<usize>>` that mixes loads and inflows, erasing type information. We need to create separate, type-safe structures for load and inflow lag variables that enable direct indexed access by bus_id and hydro_id respectively.

This follows the successful pattern used in `Realization` where `lag_duals` was split into `load_lag_duals` and `inflow_lag_duals`.

## Acceptance Criteria

- [ ] Given a system with N buses and M hydros, when creating `LoadLagVariables`, then it allocates N empty vectors indexed by bus_id
- [ ] Given a system with N buses and M hydros, when creating `InflowLagVariables`, then it allocates M empty vectors indexed by hydro_id
- [ ] Given lag variables populated for a specific entity, when accessing by entity_id, then retrieval is O(1) without type checking
- [ ] Given `LoadLagVariables` and `InflowLagVariables`, when comparing memory usage to current unified approach, then total memory is identical or less
- [ ] Performance: Structure access should be zero-cost abstraction (no runtime overhead)

## Tasks

### Implementation

- [ ] Create `LoadLagVariables` struct in `src/subproblem.rs`
  - Field: `lags_by_bus: Vec<Vec<usize>>`
  - Constructor: `new(buses_count: usize)`
  - Methods: `get_lags(bus_id)`, `get_lag_var(bus_id, lag_idx)`, `total_lag_count()`
  
- [ ] Create `InflowLagVariables` struct in `src/subproblem.rs`
  - Field: `lags_by_hydro: Vec<Vec<usize>>`
  - Constructor: `new(hydros_count: usize)`
  - Methods: `get_lags(hydro_id)`, `get_lag_var(hydro_id, lag_idx)`, `total_lag_count()`
  
- [ ] Create `LoadLagConstraints` struct in `src/subproblem.rs`
  - Field: `constraints_by_bus: Vec<Vec<usize>>`
  - Constructor: `new(buses_count: usize)`
  - Methods: `get_constraints(bus_id)`, `get_constraint(bus_id, lag_idx)`
  
- [ ] Create `InflowLagConstraints` struct in `src/subproblem.rs`
  - Field: `constraints_by_hydro: Vec<Vec<usize>>`
  - Constructor: `new(hydros_count: usize)`
  - Methods: `get_constraints(hydro_id)`, `get_constraint(hydro_id, lag_idx)`
  
- [ ] Add new fields to `Variables` struct (parallel to existing)
  - `pub load_lags: Option<LoadLagVariables>`
  - `pub inflow_lags: Option<InflowLagVariables>`
  - Keep existing `lagged_state` field temporarily
  
- [ ] Add new fields to `Constraints` struct (parallel to existing)
  - `pub load_lag_constraints: Option<LoadLagConstraints>`
  - `pub inflow_lag_constraints: Option<InflowLagConstraints>`
  - Keep existing `lag_fixing_constraints` field temporarily

### Testing

- [ ] Unit test: Create `LoadLagVariables` with 5 buses, verify capacity and empty state
- [ ] Unit test: Create `InflowLagVariables` with 3 hydros, verify capacity and empty state
- [ ] Unit test: Populate load lag variables for bus 0 with 2 lags, verify retrieval
- [ ] Unit test: Populate inflow lag variables for hydro 1 with 3 lags, verify retrieval
- [ ] Unit test: Test `total_lag_count()` with mixed populated/empty entities
- [ ] Unit test: Test bounds checking (out of range entity_id should panic in debug mode)
- [ ] Unit test: Verify `Clone` and `Debug` traits work correctly
- [ ] Memory test: Compare memory footprint with equivalent unified structure

### Documentation

- [ ] Add doc comments to all new structs explaining purpose and usage
- [ ] Add doc comments to all public methods with examples
- [ ] Document the relationship between entity_id and vector indices
- [ ] Add module-level documentation explaining the separation rationale
- [ ] Add inline comments for any non-obvious design decisions

## Technical Notes

### Design Decisions

1. **Direct Vec indexing:** Use `Vec<Vec<usize>>` indexed by entity_id rather than HashMap for O(1) access and cache locality
2. **Zero-cost abstraction:** Thin wrapper around Vec with inline methods
3. **Separate types:** Distinct types for loads vs inflows prevents accidental confusion at compile time
4. **Option<T> for optionality:** If no lags exist, field is None to avoid allocating empty structures

### Implementation Hints

```rust
/// Load lag variables indexed by bus ID
#[derive(Clone, Debug)]
pub struct LoadLagVariables {
    /// lags_by_bus[bus_id] = [Y_{t-1}, Y_{t-2}, ..., Y_{t-p}]
    /// where p = AR order for that bus
    pub lags_by_bus: Vec<Vec<usize>>,
}

impl LoadLagVariables {
    /// Create new load lag variables structure for given number of buses
    pub fn new(buses_count: usize) -> Self {
        Self {
            lags_by_bus: vec![Vec::new(); buses_count],
        }
    }
    
    /// Get lag variables for a specific bus
    #[inline]
    pub fn get_lags(&self, bus_id: usize) -> &[usize] {
        &self.lags_by_bus[bus_id]
    }
    
    /// Get specific lag variable for a bus
    #[inline]
    pub fn get_lag_var(&self, bus_id: usize, lag_idx: usize) -> usize {
        self.lags_by_bus[bus_id][lag_idx]
    }
    
    /// Count total number of lag variables across all buses
    pub fn total_lag_count(&self) -> usize {
        self.lags_by_bus.iter().map(|lags| lags.len()).sum()
    }
}
```

### Edge Cases

- Empty systems (no buses or no hydros)
- Systems where all entities have zero AR order
- Systems where some entities have lags and others don't
- Very large systems (1000+ entities) - ensure no performance regression

### Performance Considerations

- Keep structures simple to enable compiler optimizations
- Use `#[inline]` for simple accessor methods
- Avoid heap allocations in hot paths
- Measure memory overhead vs unified approach

## Dependencies

- Blocked by: None
- Blocks: TICKET-002, TICKET-003, TICKET-004, TICKET-005, TICKET-006, TICKET-007
- Related: None

## Definition of Done

- [ ] All code implemented and compiles without warnings
- [ ] All unit tests pass
- [ ] Memory usage verified equivalent to current approach
- [ ] Code reviewed and approved
- [ ] Documentation complete and reviewed
- [ ] No regression in existing tests
