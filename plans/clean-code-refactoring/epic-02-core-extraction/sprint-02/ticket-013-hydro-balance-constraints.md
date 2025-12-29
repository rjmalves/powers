# [T-013] Extract Hydro Balance Constraints

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-012](./ticket-012-constraints-module-structure.md)
> **Blocks**: [T-017](./ticket-017-refactor-subproblem-facade.md)

---

## ⚠️ CRITICAL: Behavioral Equivalence

This ticket extracts the hydro balance constraint building logic. The extracted code **must produce identical constraints** to the current implementation.

Run golden tests after implementation.

---

## Files to Read Before Starting

- `src/model/constraints/hydro_balance.rs` - Scaffold from T-012
- `src/subproblem.rs:2394-2412` - Current hydro balance implementation
- `src/system.rs` - Hydro struct definition
- `src/solver.rs` - Problem::add_row API

---

## Context

### Background

Hydro balance constraints ensure water conservation at each reservoir:

```
final_storage + turbined_flow + spillage = initial_storage + inflow + upstream_outflow
```

Rearranged as equality constraint (RHS updated with initial_storage):
```
final_storage + turbined_flow + spillage - inflow - Σ(upstream_turbined + upstream_spillage) = initial_storage
```

### Current Implementation (from subproblem.rs)

```rust
let mut hydro_balance: Vec<usize> = vec![0; system.meta.hydros_count];
for hydro in system.hydros.iter() {
    let mut factors: Vec<(usize, f64)> = vec![
        (variables.stored_volume[hydro.id], 1.0),
        (variables.turbined_flow[hydro.id], 1.0),
        (variables.spillage[hydro.id], 1.0),
    ];

    if hydro.id < variables.inflow.len() {
        factors.push((variables.inflow[hydro.id], -1.0));
    }

    for upstream_hydro_id in hydro.upstream_hydro_ids.iter() {
        factors.push((variables.turbined_flow[*upstream_hydro_id], -1.0));
        factors.push((variables.spillage[*upstream_hydro_id], -1.0));
    }
    hydro_balance[hydro.id] = pb.add_row(0.0..0.0, &factors);
}
```

---

## Specification

### Implementation

Replace the `todo!()` in `HydroBalanceBuilder::build`:

```rust
// src/model/constraints/hydro_balance.rs

use super::ConstraintContext;

/// Builder for hydro balance constraints.
///
/// Creates water balance constraints for each hydro plant:
/// ```text
/// stored_volume + turbined_flow + spillage - inflow - upstream_outflow = 0
/// ```
///
/// The RHS (initially 0) is updated during `prepare_from_trajectory` to reflect
/// the initial storage from the incoming state.
pub struct HydroBalanceBuilder;

impl HydroBalanceBuilder {
    /// Build hydro balance constraints.
    ///
    /// Creates one constraint per hydro plant. Returns indices in hydro_id order.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Constraint context with problem, variables, and system
    ///
    /// # Returns
    ///
    /// Vector of constraint indices, indexed by hydro_id
    pub fn build(ctx: &mut ConstraintContext) -> Vec<usize> {
        let mut hydro_balance = vec![0usize; ctx.system.meta.hydros_count];
        
        for hydro in ctx.system.hydros.iter() {
            let mut factors: Vec<(usize, f64)> = vec![
                (ctx.variables.stored_volume[hydro.id], 1.0),
                (ctx.variables.turbined_flow[hydro.id], 1.0),
                (ctx.variables.spillage[hydro.id], 1.0),
            ];
            
            // Inflow variable (if present for this hydro)
            if hydro.id < ctx.variables.inflow.len() {
                factors.push((ctx.variables.inflow[hydro.id], -1.0));
            }
            
            // Upstream contributions (cascading hydrology)
            for &upstream_hydro_id in &hydro.upstream_hydro_ids {
                factors.push((ctx.variables.turbined_flow[upstream_hydro_id], -1.0));
                factors.push((ctx.variables.spillage[upstream_hydro_id], -1.0));
            }
            
            // Add constraint: LHS = 0 (RHS updated during realize)
            hydro_balance[hydro.id] = ctx.problem.add_row(0.0..0.0, &factors);
        }
        
        hydro_balance
    }
}
```

### Behavior

- **Input**: ConstraintContext with problem, variables, system
- **Output**: Vector of constraint indices indexed by hydro_id
- **Constraint form**: stored_volume + turbined_flow + spillage - inflow - upstream = 0
- **RHS**: Initially 0, updated during `prepare_from_trajectory`

---

## Acceptance Criteria

- [ ] `HydroBalanceBuilder::build` implemented
- [ ] Logic exactly matches `subproblem.rs:2394-2412`
- [ ] Returns correct constraint indices
- [ ] Handles upstream hydro correctly
- [ ] Handles missing inflow variable correctly
- [ ] Unit tests verify constraint structure
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Correctness Verification

- [ ] Constraint coefficients match original
- [ ] Constraint ordering matches original (indexed by hydro_id)
- [ ] Upstream hydro contributions are negative

---

## Implementation Guide

### Suggested Approach

1. **Open `src/model/constraints/hydro_balance.rs`**

2. **Replace `todo!()` with implementation** (copy logic from subproblem.rs)

3. **Ensure imports are correct**:
   ```rust
   use super::ConstraintContext;
   ```

4. **Add unit tests** (if possible with mock types, otherwise integration-only)

5. **Verify compilation**:
   ```bash
   cargo build
   ```

6. **Run tests**:
   ```bash
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/constraints/hydro_balance.rs` | Implement build method |

### Patterns to Follow

- Use `ctx.problem`, `ctx.variables`, `ctx.system` from context
- Build factor vector, then call `add_row`
- Return indices in order

### Pitfalls to Avoid

- ⚠️ Don't change constraint coefficients (must be identical)
- ⚠️ Handle the `hydro.id < variables.inflow.len()` check correctly
- ⚠️ Iterate `upstream_hydro_ids` correctly (it's a Vec or slice)
- ⚠️ Ensure RHS is `0.0..0.0` (equality constraint)

---

## Testing Requirements

### Unit Tests

If mock types are available:
- [ ] Test constraint factors are correct for simple case
- [ ] Test upstream hydro contributions
- [ ] Test missing inflow case

### Integration Tests

- [ ] Extraction produces same indices as original

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Doc comment on `build` method
- [ ] Explain constraint form in module docs
- [ ] Document that RHS is updated later

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear extraction with well-defined logic

---

## Definition of Done

- [ ] Method implemented
- [ ] Logic matches original
- [ ] Tests passing
- [ ] Golden tests passing
- [ ] Documented
