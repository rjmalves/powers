# TICKET-006: Update Realization Struct for Residual Space

**Sprint:** 2 - Subproblem Refactor  
**Phase:** 2 - Refactor Subproblem  
**Estimated Effort:** 1 day (3 story points)  
**Confidence:** High  
**Status:** Not Started

## Context

The `Realization` struct represents the solution of a subproblem. For the unified AR model, it needs to store both observation space inflows (for output/reporting) and residual space values (for lag buffer updates and cut generation). This ticket adds the necessary fields while maintaining backward compatibility.

## Acceptance Criteria

- [ ] Given a Realization struct, when accessed, then it has `inflow_residual: Vec<f64>` field for residual space values
- [ ] Given a Realization struct, when accessed, then it has `lag_duals: Vec<Vec<f64>>` field for AR lag constraint duals
- [ ] Given existing code using Realization, when refactored struct is used, then existing fields (inflows, stored_volume, etc.) are unchanged
- [ ] Given a solved subproblem, when Realization is created, then both observation and residual space values are extracted
- [ ] Given Default implementation, when Realization::default() is called, then all fields are properly initialized
- [ ] Performance: Realization struct should remain efficiently clonable (all Vec fields)

## Acceptance Criteria

- [ ] Given a Realization struct, when accessed, then it has `inflow_residual: Vec<f64>` field for residual space values
- [ ] Given a Realization struct, when accessed, then it has `lag_duals: Vec<Vec<f64>>` field for AR lag constraint duals
- [ ] Given existing code using Realization, when refactored struct is used, then existing fields (inflows, stored_volume, etc.) are unchanged
- [ ] Given a solved subproblem, when Realization is created, then both observation and residual space values are extracted
- [ ] Given Default implementation, when Realization::default() is called, then all fields are properly initialized
- [ ] Performance: Realization struct should remain efficiently clonable (all Vec fields)

## Tasks

### Implementation

- [ ] Update `Realization` struct in `src/subproblem.rs`:
  - Keep all existing fields: inflows, load, stored_volume, turbined_flow, spillage, etc.
  - Add `inflow_residual: Vec<f64>` field with comment `// Z'_t values (residual space)`
  - Update `lag_duals: Vec<Vec<f64>>` field comment to clarify `// Duals on AR lag constraints [hydro][lag]`
  - Ensure field is already present (verify in current code)
- [ ] Update `Default` implementation for Realization:
  - Initialize `inflow_residual` to empty Vec
  - Verify `lag_duals` initialization is correct
- [ ] Update `Realization::new()` constructor if it exists:
  - Include `inflow_residual` parameter
  - Handle both observation and residual space values
- [ ] Add helper methods:
  - `has_residuals(&self) -> bool` - checks if inflow_residual is populated
  - `num_lag_duals(&self, hydro: usize) -> usize` - returns lag_duals[hydro].len()
- [ ] Ensure Realization remains `Clone` and `Debug` derivable

### Testing

- [ ] Unit test: Create Realization with observation and residual space values
- [ ] Unit test: Verify Default implementation initializes all fields correctly
- [ ] Unit test: Verify has_residuals() returns correct boolean
- [ ] Unit test: Verify num_lag_duals() returns correct count per hydro
- [ ] Unit test: Clone Realization, verify all fields copied correctly
- [ ] Unit test: Realization with mixed lag duals ([0, 2, 1] for 3 hydros)
- [ ] Integration test: Extract Realization from solved subproblem

### Documentation

- [ ] Add doc comment to Realization struct explaining dual space representation
- [ ] Document `inflow_residual` field with clear explanation of residual space
- [ ] Update module-level docs with example showing both observation and residual values
- [ ] Add inline comment explaining when residuals are populated vs observation values
- [ ] Update CHANGELOG.md with "Changed: Realization struct includes residual space values"

## Technical Notes

### Realization Structure Design

```rust
#[derive(Debug, Clone)]
pub struct Realization {
    // Physical variables (observation space)
    pub inflows: Vec<f64>,
    pub load: Vec<f64>,
    pub stored_volume: Vec<f64>,
    pub turbined_flow: Vec<f64>,
    pub spillage: Vec<f64>,
    pub deficit: Vec<f64>,
    pub thermal_gen: Vec<f64>,
    pub direct_exchange: Vec<f64>,
    pub reverse_exchange: Vec<f64>,

    // Inflow residual space (for AR dynamics)
    pub inflow_residual: Vec<f64>,  // Z'_t values

    // Cost components
    pub immediate_cost: f64,
    pub future_cost: f64,

    // Dual values
    pub load_balance_duals: Vec<f64>,
    pub hydro_balance_duals: Vec<f64>,
    pub lag_duals: Vec<Vec<f64>>,  // [hydro][lag] - AR lag constraint duals

    // Study period metadata
    pub study_period_kind: StudyPeriodKind,
    pub study_period_id: usize,

    // Timing information
    pub timing: RealizeUncertaintiesTiming,
}
```

### When to Populate Fields

**inflows (observation space):**

- Always populated after subproblem solve
- Extracted from solver solution: `solution.colvalue[vars.inflow[h]]`
- Used for output, reporting, hydro balance

**inflow_residual (residual space):**

- Always populated after subproblem solve (unified model)
- Extracted from solver solution: `solution.colvalue[vars.inflow_residual[h]]`
- Used for lag buffer updates, cut generation

**lag_duals:**

- Only populated for hydros with AR dynamics (lag_order > 0)
- Extracted from solver duals: `solution.rowdual[constraints.ar_dynamics[h]]` (if lags present)
- Empty for independent hydros
- Used for cut generation and gradient calculations

### Migration Notes

**Field Addition:**

- `inflow_residual: Vec<f64>` - NEW field for unified model

**Field Already Present:**

- `lag_duals: Vec<Vec<f64>>` - Should already exist from recent PAR work

**Backward Compatibility:**

- Existing code using `inflows` continues to work
- New code can use `inflow_residual` for AR dynamics
- Old realizations can be migrated by setting `inflow_residual = vec![]`

### Edge Cases

- **Independent hydros**: inflow_residual still populated (from AR(0) formulation)
- **No lag duals**: lag_duals[hydro] is empty Vec for independent hydros
- **Mixed models**: Some hydros have lag_duals, some don't
- **PreStudy realizations**: May have residuals from initialization

### Memory Considerations

**Additional Memory per Realization:**

```
inflow_residual: Vec<f64>  // n floats (n = number of hydros)
lag_duals: Vec<Vec<f64>>   // ~(n * p) floats (p = avg lag order)
```

For typical system (30 hydros, AR(2)):

- inflow_residual: 30 \* 8 bytes = 240 bytes
- lag_duals: 30 _ 2 _ 8 bytes = 480 bytes
- **Total additional: ~720 bytes per realization**

This is acceptable given the clarity and correctness benefits.

## Dependencies

- **Blocked by**: None (pure data structure change)
- **Blocks**: TICKET-003 (lag buffer needs inflow_residual field)
- **Related**: TICKET-007 (solution extraction populates these fields)

## References

- Current `src/subproblem.rs` lines 1035-1057 - Realization struct definition
- UNIFIED_AR_ROADMAP.md - Section 1.2 (Space Consistency)

## Validation Checklist

Before marking this ticket as done:

- [ ] Code compiles without warnings
- [ ] All existing tests still pass
- [ ] New unit tests for Realization pass
- [ ] `cargo clippy` shows no issues
- [ ] `cargo fmt` applied
- [ ] Memory overhead is acceptable (<1KB per realization)
- [ ] Documentation builds without warnings
- [ ] Code reviewed by at least one team member
