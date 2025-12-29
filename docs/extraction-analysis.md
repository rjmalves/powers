# Extraction Analysis: subproblem.rs

> **Created for**: [T-006](../plans/clean-code-refactoring/epic-02-core-extraction/sprint-01/ticket-006-analyze-extraction-points.md)
> **Last Updated**: 2025-12-29

## Overview

| Metric | Value |
|--------|-------|
| Total lines | ~6,631 |
| Solution extraction functions | 12 |
| Constraint building functions | 3 |
| Key structs | Variables, Constraints, Realization |

---

## Solution Extraction Functions

All functions extract from `solver::Solution` into `Realization`.

### Primal Variables (from `solution.colvalue`)

| Function | Lines | Pattern | Target Field | Source Variables |
|----------|-------|---------|--------------|------------------|
| `get_deficit_from_solution` | 1862-1872 | Contiguous range | `realization.deficit` | `variables.deficit` |
| `get_net_exchange_from_solution` | 1874-1896 | Two ranges, subtraction | `realization.exchange` | `variables.direct_exchange`, `variables.reverse_exchange` |
| `get_thermal_gen_from_solution` | 1898-1910 | Optional contiguous | `realization.thermal_generation` | `variables.thermal_gen` |
| `get_spillage_from_solution` | 1912-1922 | Contiguous range | `realization.spillage` | `variables.spillage` |
| `get_turbined_flow_from_solution` | 1924-1934 | Contiguous range | `realization.turbined_flow` | `variables.turbined_flow` |
| `get_final_storage_from_solution` | 1936-1946 | Contiguous range | `realization.final_storage` | `variables.stored_volume` |
| `get_load_from_solution` | 1948-1957 | Index-by-index | `realization.loads` | `variables.load` |
| `get_inflow_from_solution` | 1959-1968 | Index-by-index | `realization.inflow` | `variables.inflow` |

### Dual Variables (from `solution.rowdual`)

| Function | Lines | Pattern | Target Field | Source Constraints |
|----------|-------|---------|--------------|-------------------|
| `get_water_values_from_solution` | 1970-1980 | Contiguous range | `realization.water_value` | `constraints.hydro_balance` |
| `get_marginal_cost_from_solution` | 2036-2046 | Contiguous range | `realization.marginal_cost` | `constraints.load_balance` |
| `get_lag_duals_from_solution` | 1992-2034 | Nested loops | `realization.load_lag_duals`, `realization.inflow_lag_duals` | `constraints.load_lag_constraints`, `constraints.inflow_lag_constraints` |

### State/Objective Functions

| Function | Lines | Pattern | Target Field | Source |
|----------|-------|---------|--------------|--------|
| `get_current_stage_objective` | 130-136 | Free function | `realization.current_stage_objective` | `solution.colvalue.last()` |
| `populate_initial_state_fields` | 2065-2092 | State extraction | `realization.initial_storage`, `realization.inflow_lags` | `self.state.coefficients()`, `self.inflow_lag_data` |

---

## Function Dependency Analysis

### Pure Extraction Functions (no side effects, `&self`)

These can be extracted as free functions with index parameters:

1. `get_deficit_from_solution` - Uses `variables.deficit`
2. `get_thermal_gen_from_solution` - Uses `variables.thermal_gen`
3. `get_spillage_from_solution` - Uses `variables.spillage`
4. `get_turbined_flow_from_solution` - Uses `variables.turbined_flow`
5. `get_final_storage_from_solution` - Uses `variables.stored_volume`
6. `get_water_values_from_solution` - Uses `constraints.hydro_balance`
7. `get_marginal_cost_from_solution` - Uses `constraints.load_balance`
8. `get_load_from_solution` - Uses `variables.load` (index-by-index)
9. `get_net_exchange_from_solution` - Uses `variables.direct_exchange`, `variables.reverse_exchange`

### Functions with `&mut self`

1. `get_inflow_from_solution` - Takes `&mut self` but only reads `self.variables.inflow`
   - **Note**: `&mut self` appears unnecessary; likely can be `&self`

### Functions with Complex Dependencies

1. `get_lag_duals_from_solution` - Uses:
   - `self.constraints.load_lag_constraints`
   - `self.constraints.inflow_lag_constraints`
   - Allocates inside (`resize`, `collect`)
   - **Complexity**: Nested structure access

2. `populate_initial_state_fields` - Uses:
   - `self.state.coefficients()` - Dynamic dispatch
   - `self.inflow_lag_data`
   - **Complexity**: State trait dependency

---

## Data Structures

### Variables Struct (line 700-733)

```rust
pub struct Variables {
    pub deficit: Vec<usize>,           // Always present
    pub direct_exchange: Vec<usize>,   // Optional (may be empty)
    pub reverse_exchange: Vec<usize>,  // Optional (may be empty)
    pub thermal_gen: Vec<usize>,       // Optional (may be empty)
    pub turbined_flow: Vec<usize>,     // Always present
    pub spillage: Vec<usize>,          // Always present
    pub stored_volume: Vec<usize>,     // Always present
    pub load: Vec<usize>,              // Non-contiguous indices
    pub inflow: Vec<usize>,            // Non-contiguous indices
    pub innovation: Vec<usize>,
    pub lagged_state: Option<Vec<Vec<usize>>>,
    pub load_lags: Option<LoadLagVariables>,
    pub inflow_lags: Option<InflowLagVariables>,
    pub alpha: usize,
}
```

### Constraints Struct (line 740-748)

```rust
pub struct Constraints {
    pub load_balance: Vec<usize>,          // Contiguous
    pub hydro_balance: Vec<usize>,         // Contiguous
    pub uncertainty_observation: Vec<usize>,
    pub load_lag_constraints: Option<LoadLagConstraints>,
    pub inflow_lag_constraints: Option<InflowLagConstraints>,
}
```

### Realization Struct (line 2721-2826)

Located in `src/subproblem.rs`, not `src/sddp/mod.rs`.

Key fields populated by extraction:
- `loads`, `deficit`, `exchange`, `inflow` (primal observations)
- `turbined_flow`, `spillage`, `thermal_generation` (physical)
- `water_value`, `marginal_cost` (duals)
- `load_lag_duals`, `inflow_lag_duals` (lag duals)
- `current_stage_objective`, `total_stage_objective`
- `initial_storage`, `final_storage`, `inflow_lags`
- `basis`

---

## Constraint Building Functions

| Function | Lines | Purpose | Dependencies |
|----------|-------|---------|--------------|
| `add_constraints` | 2355-2494 | Add load/hydro balance constraints | `system`, `model`, `variables`, `constraints` |
| `add_uncertainty_observation_constraints` | 2495-2566 | Add AR observation constraints | `system`, `model`, `variables`, `constraints` |
| `build_uncertainty_observation_data` | 2567-? | Precompute uncertainty data | `system`, `config` |

### `add_constraints` Analysis (lines 2355-2494)

Adds two types of constraints:
1. **Load balance** (power balance at buses)
2. **Hydro balance** (water balance at hydros)

Dependencies:
- `system.buses`, `system.thermals`, `system.hydros`
- `model.add_row_*`
- `variables.*` for coefficient mapping

### `add_uncertainty_observation_constraints` Analysis

Adds AR dynamics constraints:
- Y_t = base + σ·η + sum(φ_k · Y_{t-k})
- Lag fixing: Y_{t-k} = value

---

## Data Dependency Graph

```
                     Variables                        Constraints
                        │                                  │
    ┌───────────────────┼──────────────────┐   ┌──────────┼─────────────┐
    ▼                   ▼                  ▼   ▼          ▼             ▼
 deficit        direct_exchange      thermal_gen  load_balance  hydro_balance  lag_constraints
    │              reverse_exchange        │          │              │              │
    │                   │                  │          │              │              │
    ▼                   ▼                  ▼          ▼              ▼              ▼
get_deficit    get_net_exchange    get_thermal_gen  get_marginal  get_water_val  get_lag_duals
    │                   │                  │          │              │              │
    └───────────────────┴──────────────────┴──────────┴──────────────┴──────────────┘
                                           │
                                           ▼
                                      Realization
```

---

## Recommended Extraction Order

### Phase 1: Simple Contiguous Extractions

1. `get_deficit_from_solution` - Simplest pattern
2. `get_spillage_from_solution` - Same pattern
3. `get_turbined_flow_from_solution` - Same pattern
4. `get_final_storage_from_solution` - Same pattern (uses `stored_volume`)
5. `get_thermal_gen_from_solution` - Optional variant

### Phase 2: Dual Extractions

6. `get_water_values_from_solution` - Contiguous from rowdual
7. `get_marginal_cost_from_solution` - Same pattern

### Phase 3: Exchange (special logic)

8. `get_net_exchange_from_solution` - Two ranges with subtraction

### Phase 4: Non-contiguous Extractions

9. `get_load_from_solution` - Index-by-index
10. `get_inflow_from_solution` - Index-by-index (change to `&self`)

### Phase 5: Complex Extractions

11. `get_lag_duals_from_solution` - Nested structure
12. `populate_initial_state_fields` - State dependency

---

## Risks and Concerns

### Risk 1: `get_inflow_from_solution` uses `&mut self`

**Issue**: Function signature is `fn get_inflow_from_solution(&mut self, ...)` but only reads data.

**Mitigation**: Check if `&mut` is actually needed. Likely can be changed to `&self`.

### Risk 2: Lag duals allocation in hot path

**Issue**: `get_lag_duals_from_solution` uses `clear()`, `resize()`, `collect()` - allocates.

**Mitigation**: Future optimization in Epic 5. For now, extract as-is.

### Risk 3: State trait dynamic dispatch in `populate_initial_state_fields`

**Issue**: Calls `self.state.coefficients()` which is dynamic dispatch.

**Mitigation**: Keep `&self` access; cannot extract as free function.

### Risk 4: Realization in subproblem.rs, not separate module

**Issue**: `Realization` is defined in `subproblem.rs`, not a separate module.

**Mitigation**: Import from `crate::subproblem::Realization`. Consider moving in future.

### Risk 5: Hidden dependencies in constraint building

**Issue**: Constraint building has extensive dependencies on `system` struct.

**Mitigation**: Careful analysis in Sprint 2. May need ConstraintContext struct.

---

## Summary

| Category | Count | Notes |
|----------|-------|-------|
| Simple contiguous extractions | 6 | Easy to extract |
| Dual extractions | 2 | Easy |
| Exchange extraction | 1 | Special subtraction logic |
| Non-contiguous extractions | 2 | Index-by-index loops |
| Complex extractions | 2 | State deps, nested structures |
| **Total** | **12** | |

All extraction functions are self-contained with clear inputs/outputs. The main complexity is in `get_lag_duals_from_solution` and `populate_initial_state_fields`, which have nested data structures and state dependencies.

Recommended approach: Create `VariableIndices` and `ConstraintIndices` structs that precompute ranges, then implement `SolutionExtractor` with dual API pattern.
