# [T-015] Extract AR Dynamics Constraints

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-012](./ticket-012-constraints-module-structure.md)
> **Blocks**: [T-017](./ticket-017-refactor-subproblem-facade.md)

---

## ⚠️ CRITICAL: Behavioral Equivalence

This ticket extracts the AR (auto-regressive) dynamics constraint building, including uncertainty observation constraints and lag fixing constraints. These are mathematically complex—ensure **exact behavioral equivalence**.

Run golden tests after implementation.

---

## Files to Read Before Starting

- `src/model/constraints/ar_dynamics.rs` - Scaffold from T-012
- `src/subproblem.rs:2495-2542` - Uncertainty observation constraints
- `src/subproblem.rs:2422-2475` - Lag fixing constraints
- `src/temporal_model.rs` - TemporalModel struct
- `src/subproblem.rs` - LoadLagConstraints, InflowLagConstraints structs

---

## Context

### Background

AR dynamics constraints handle the auto-regressive nature of uncertain quantities (loads, inflows):

1. **Uncertainty Observation Constraints**: Link observation variable Y to innovation η and lags
   ```
   Y[i] - Σ(ψ_k * Y_{t-k}[i]) = deterministic_base + σ * η
   ```

2. **Lag Fixing Constraints**: Fix lag variables to specific values
   ```
   Y_{t-k}[i] = value  (updated during realize_uncertainties)
   ```

### Current Implementation

**Uncertainty Observation (subproblem.rs:2495-2542)**:
```rust
fn add_uncertainty_observation_constraints(
    pb: &mut solver::Problem,
    variables: &Variables,
    temporal_models: &[temporal_model::TemporalModel],
    season_id: usize,
) -> Vec<usize> {
    let mut constraint_indices = Vec::new();
    let mut load_idx = 0;
    let mut inflow_idx = 0;

    for (global_idx, model) in temporal_models.iter().enumerate() {
        let observation_var = match model.entity_type {
            UncertaintyType::Load => {
                let var = variables.load[load_idx];
                load_idx += 1;
                var
            }
            UncertaintyType::Inflow => {
                let var = variables.inflow[inflow_idx];
                inflow_idx += 1;
                var
            }
        };

        let mut factors = vec![(observation_var, 1.0)];

        // Add lag variables with negative psi coefficients
        if let Some(ref lag_vars) = variables.lagged_state {
            let entity_lag_vars = &lag_vars[global_idx];
            let psi_coeffs = &model.psi_coefficients[season_id];

            for (lag_idx, &lag_var) in entity_lag_vars.iter().enumerate() {
                if lag_idx < psi_coeffs.len() {
                    let psi = psi_coeffs[lag_idx];
                    factors.push((lag_var, -psi));
                }
            }
        }

        let row = pb.add_row(0.0..=0.0, &factors);
        constraint_indices.push(row);
    }

    constraint_indices
}
```

**Lag Fixing (subproblem.rs:2422-2475)**:
```rust
// Inside add_constraints, after uncertainty_observation
let (load_lag_constraints, inflow_lag_constraints) =
    if let Some(ref lag_vars) = variables.lagged_state {
        let mut new_load_constraints = LoadLagConstraints::new(system.buses.len());
        let mut new_inflow_constraints = InflowLagConstraints::new(system.hydros.len());

        for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            let mut entity_constraints = Vec::new();

            for &var in entity_lags {
                let constraint = pb.add_row(0.0..=0.0, vec![(var, 1.0)]);
                entity_constraints.push(constraint);
            }

            let model = &temporal_models[entity_idx];
            match model.entity_type {
                UncertaintyType::Load => {
                    let bus_id = model.entity_id;
                    new_load_constraints.constraints_by_bus[bus_id] = entity_constraints;
                }
                UncertaintyType::Inflow => {
                    let hydro_id = model.entity_id;
                    new_inflow_constraints.constraints_by_hydro[hydro_id] = entity_constraints;
                }
            }
        }

        let load_opt = if new_load_constraints.total_constraint_count() > 0 {
            Some(new_load_constraints)
        } else { None };
        let inflow_opt = if new_inflow_constraints.total_constraint_count() > 0 {
            Some(new_inflow_constraints)
        } else { None };

        (load_opt, inflow_opt)
    } else {
        (None, None)
    };
```

---

## Specification

### Implementation

Replace the `todo!()` methods in `ArDynamicsBuilder`:

```rust
// src/model/constraints/ar_dynamics.rs

use super::ConstraintContext;
use crate::input::UncertaintyType;
use crate::subproblem::{LoadLagConstraints, InflowLagConstraints};

/// Builder for AR dynamics constraints.
///
/// Builds two types of constraints:
/// 1. **Observation constraints**: Y = base + σ·η + Σ(ψ·lags)
/// 2. **Lag fixing constraints**: Y_{t-k} = value
pub struct ArDynamicsBuilder;

impl ArDynamicsBuilder {
    /// Build uncertainty observation constraints.
    ///
    /// Creates one constraint per uncertain entity (loads then inflows).
    /// The constraint form is:
    /// ```text
    /// Y[i] - Σ(ψ_k · Y_{t-k}[i]) = 0  (RHS updated in realize_uncertainties)
    /// ```
    ///
    /// # Returns
    ///
    /// Vector of constraint indices in temporal_model order
    pub fn build_observation(ctx: &mut ConstraintContext) -> Vec<usize> {
        let mut constraint_indices = Vec::new();
        let mut load_idx = 0;
        let mut inflow_idx = 0;
        
        for (global_idx, model) in ctx.temporal_models.iter().enumerate() {
            // Get observation variable for this entity
            let observation_var = match model.entity_type {
                UncertaintyType::Load => {
                    let var = ctx.variables.load[load_idx];
                    load_idx += 1;
                    var
                }
                UncertaintyType::Inflow => {
                    let var = ctx.variables.inflow[inflow_idx];
                    inflow_idx += 1;
                    var
                }
            };
            
            let mut factors = vec![(observation_var, 1.0)];
            
            // Add lag variables with negative psi coefficients
            if let Some(ref lag_vars) = ctx.variables.lagged_state {
                let entity_lag_vars = &lag_vars[global_idx];
                let psi_coeffs = &model.psi_coefficients[ctx.season_id];
                
                for (lag_idx, &lag_var) in entity_lag_vars.iter().enumerate() {
                    if lag_idx < psi_coeffs.len() {
                        let psi = psi_coeffs[lag_idx];
                        factors.push((lag_var, -psi));
                    }
                }
            }
            
            // Add constraint with RHS = 0 (updated in realize_uncertainties)
            let row = ctx.problem.add_row(0.0..=0.0, &factors);
            constraint_indices.push(row);
        }
        
        constraint_indices
    }
    
    /// Build lag fixing constraints.
    ///
    /// Creates constraints that fix lag variables to specific values:
    /// ```text
    /// Y_{t-k}[i] = 0  (RHS updated in realize_uncertainties)
    /// ```
    ///
    /// # Returns
    ///
    /// Tuple of (load_lag_constraints, inflow_lag_constraints)
    pub fn build_lag_fixing(
        ctx: &mut ConstraintContext,
    ) -> (Option<LoadLagConstraints>, Option<InflowLagConstraints>) {
        let Some(ref lag_vars) = ctx.variables.lagged_state else {
            return (None, None);
        };
        
        let mut load_constraints = LoadLagConstraints::new(ctx.system.buses.len());
        let mut inflow_constraints = InflowLagConstraints::new(ctx.system.hydros.len());
        
        for (entity_idx, entity_lags) in lag_vars.iter().enumerate() {
            let mut entity_constraints = Vec::new();
            
            for &var in entity_lags {
                // Simple fixing constraint: var = 0 (RHS updated later)
                let constraint = ctx.problem.add_row(0.0..=0.0, vec![(var, 1.0)]);
                entity_constraints.push(constraint);
            }
            
            // Route to appropriate structure based on entity type
            let model = &ctx.temporal_models[entity_idx];
            match model.entity_type {
                UncertaintyType::Load => {
                    let bus_id = model.entity_id;
                    load_constraints.constraints_by_bus[bus_id] = entity_constraints;
                }
                UncertaintyType::Inflow => {
                    let hydro_id = model.entity_id;
                    inflow_constraints.constraints_by_hydro[hydro_id] = entity_constraints;
                }
            }
        }
        
        // Convert to Option (None if empty)
        let load_opt = if load_constraints.total_constraint_count() > 0 {
            Some(load_constraints)
        } else {
            None
        };
        
        let inflow_opt = if inflow_constraints.total_constraint_count() > 0 {
            Some(inflow_constraints)
        } else {
            None
        };
        
        (load_opt, inflow_opt)
    }
}
```

### Behavior

- **Observation constraints**: Y - Σ(ψ·lags) = 0 (RHS updated during realize)
- **Lag constraints**: Y_{t-k} = 0 (RHS updated during realize)
- **Entity routing**: Loads go to LoadLagConstraints, inflows to InflowLagConstraints
- **Empty handling**: Return None if no constraints of that type

---

## Acceptance Criteria

- [ ] `ArDynamicsBuilder::build_observation` implemented
- [ ] `ArDynamicsBuilder::build_lag_fixing` implemented
- [ ] Observation constraints have correct psi coefficients (negative)
- [ ] Lag constraints correctly route to load vs inflow structures
- [ ] Empty cases handled (no lags → return None)
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Correctness Verification

- [ ] Psi coefficients are negated in constraint
- [ ] Entity type routing is correct
- [ ] Constraint indices match original order

---

## Implementation Guide

### Suggested Approach

1. **Check imports needed**:
   ```rust
   use crate::input::UncertaintyType;
   use crate::subproblem::{LoadLagConstraints, InflowLagConstraints};
   ```

2. **Implement `build_observation`** - copy logic from subproblem.rs:2495-2542

3. **Implement `build_lag_fixing`** - copy logic from subproblem.rs:2422-2475

4. **Verify with tests**:
   ```bash
   cargo build
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/constraints/ar_dynamics.rs` | Implement both methods |

### Patterns to Follow

- Use `let Some(ref ...) else { return }` for early return
- Keep entity routing logic identical to original
- Use `total_constraint_count()` for empty check

### Pitfalls to Avoid

- ⚠️ Psi coefficients must be NEGATED in the constraint
- ⚠️ Entity type matching must use correct variant names
- ⚠️ Load/inflow index counters are separate
- ⚠️ Don't confuse global_idx with entity_id

---

## Testing Requirements

### Unit Tests

If mock types are available:
- [ ] Test observation constraint with no lags
- [ ] Test observation constraint with lags and psi coefficients
- [ ] Test lag fixing with load entities
- [ ] Test lag fixing with inflow entities
- [ ] Test empty lag case (returns None)

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Doc comments on both methods
- [ ] Explain psi coefficient negation
- [ ] Document RHS update timing

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: Complex logic with entity routing and coefficient handling

---

## Definition of Done

- [ ] Both methods implemented
- [ ] Logic matches original exactly
- [ ] Psi negation verified
- [ ] Entity routing verified
- [ ] Tests passing
- [ ] Golden tests passing
- [ ] Documented
