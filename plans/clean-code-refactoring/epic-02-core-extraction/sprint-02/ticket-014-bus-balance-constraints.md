# [T-014] Extract Bus Balance Constraints

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-012](./ticket-012-constraints-module-structure.md)
> **Blocks**: [T-017](./ticket-017-refactor-subproblem-facade.md)

---

## ⚠️ CRITICAL: Behavioral Equivalence

This ticket extracts the bus (load) balance constraint building logic. The extracted code **must produce identical constraints** to the current implementation.

Run golden tests after implementation.

---

## Files to Read Before Starting

- `src/model/constraints/bus_balance.rs` - Scaffold from T-012
- `src/subproblem.rs:2363-2392` - Current bus balance implementation
- `src/system.rs` - Bus struct definition
- `src/solver.rs` - Problem::add_row API

---

## Context

### Background

Bus balance constraints ensure power balance at each bus (node) in the network:

```
generation + imports - exports + deficit = load
```

Rearranged as equality constraint:
```
deficit - load + Σ(thermal_gen) + Σ(hydro_productivity * turbined_flow) + exchange_flows = 0
```

### Current Implementation (from subproblem.rs)

```rust
let mut load_balance: Vec<usize> = vec![0; system.meta.buses_count];
for bus in system.buses.iter() {
    let mut factors = vec![
        (variables.deficit[bus.id], 1.0),
        (variables.load[bus.id], -1.0),
    ];

    // Add generators
    for thermal_id in bus.thermal_ids.iter() {
        factors.push((variables.thermal_gen[*thermal_id], 1.0));
    }
    for hydro_id in bus.hydro_ids.iter() {
        factors.push((
            variables.turbined_flow[*hydro_id],
            system.hydros.get(*hydro_id).unwrap().productivity,
        ));
    }

    // Add transmission lines
    for line_id in bus.source_line_ids.iter() {
        factors.push((variables.reverse_exchange[*line_id], 1.0));
        factors.push((variables.direct_exchange[*line_id], -1.0));
    }
    for line_id in bus.target_line_ids.iter() {
        factors.push((variables.direct_exchange[*line_id], 1.0));
        factors.push((variables.reverse_exchange[*line_id], -1.0));
    }

    load_balance[bus.id] = pb.add_row(0.0..0.0, &factors);
}
```

---

## Specification

### Implementation

Replace the `todo!()` in `BusBalanceBuilder::build`:

```rust
// src/model/constraints/bus_balance.rs

use super::ConstraintContext;

/// Builder for bus (load) balance constraints.
///
/// Creates power balance constraints for each bus:
/// ```text
/// deficit - load + generation + exchange_net = 0
/// ```
///
/// Generation includes:
/// - Thermal plants: coefficient = 1.0
/// - Hydro plants: coefficient = productivity (MW/m³/s)
///
/// Exchange flows:
/// - Source lines: +reverse, -direct (power leaving)
/// - Target lines: +direct, -reverse (power arriving)
pub struct BusBalanceBuilder;

impl BusBalanceBuilder {
    /// Build bus balance constraints.
    ///
    /// Creates one constraint per bus. Returns indices in bus_id order.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Constraint context with problem, variables, and system
    ///
    /// # Returns
    ///
    /// Vector of constraint indices, indexed by bus_id
    pub fn build(ctx: &mut ConstraintContext) -> Vec<usize> {
        let mut load_balance = vec![0usize; ctx.system.meta.buses_count];
        
        for bus in ctx.system.buses.iter() {
            let mut factors = vec![
                (ctx.variables.deficit[bus.id], 1.0),
                (ctx.variables.load[bus.id], -1.0),
            ];
            
            // Thermal generators at this bus
            for &thermal_id in &bus.thermal_ids {
                factors.push((ctx.variables.thermal_gen[thermal_id], 1.0));
            }
            
            // Hydro generators at this bus (with productivity coefficient)
            for &hydro_id in &bus.hydro_ids {
                let productivity = ctx.system.hydros.get(hydro_id)
                    .expect("hydro_id should be valid")
                    .productivity;
                factors.push((ctx.variables.turbined_flow[hydro_id], productivity));
            }
            
            // Transmission lines where this bus is the source
            // Power flows OUT via direct_exchange, IN via reverse_exchange
            for &line_id in &bus.source_line_ids {
                factors.push((ctx.variables.reverse_exchange[line_id], 1.0));
                factors.push((ctx.variables.direct_exchange[line_id], -1.0));
            }
            
            // Transmission lines where this bus is the target
            // Power flows IN via direct_exchange, OUT via reverse_exchange
            for &line_id in &bus.target_line_ids {
                factors.push((ctx.variables.direct_exchange[line_id], 1.0));
                factors.push((ctx.variables.reverse_exchange[line_id], -1.0));
            }
            
            // Add constraint: LHS = 0
            load_balance[bus.id] = ctx.problem.add_row(0.0..0.0, &factors);
        }
        
        load_balance
    }
}
```

### Behavior

- **Input**: ConstraintContext with problem, variables, system
- **Output**: Vector of constraint indices indexed by bus_id
- **Constraint form**: deficit - load + generation + exchange = 0
- **Productivity**: Hydro contribution uses productivity coefficient

---

## Acceptance Criteria

- [ ] `BusBalanceBuilder::build` implemented
- [ ] Logic exactly matches `subproblem.rs:2363-2392`
- [ ] Returns correct constraint indices
- [ ] Thermal generators included with coefficient 1.0
- [ ] Hydro generators included with productivity coefficient
- [ ] Exchange flows have correct signs
- [ ] Unit tests verify constraint structure
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass

### Correctness Verification

- [ ] Constraint coefficients match original exactly
- [ ] Exchange flow signs are correct (source vs target)
- [ ] Productivity coefficients fetched correctly

---

## Implementation Guide

### Suggested Approach

1. **Open `src/model/constraints/bus_balance.rs`**

2. **Replace `todo!()` with implementation**

3. **Verify exchange flow signs**:
   - Source lines: this bus sends power
     - Direct exchange sends power OUT (coefficient -1.0)
     - Reverse exchange brings power IN (coefficient +1.0)
   - Target lines: this bus receives power
     - Direct exchange brings power IN (coefficient +1.0)
     - Reverse exchange sends power OUT (coefficient -1.0)

4. **Compile and test**:
   ```bash
   cargo build
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/constraints/bus_balance.rs` | Implement build method |

### Patterns to Follow

- Use iterator patterns for cleaner code
- Use `expect()` with descriptive message for unwrap
- Keep factor ordering consistent with original

### Pitfalls to Avoid

- ⚠️ Exchange signs are tricky—verify carefully
- ⚠️ Productivity is per-hydro, not constant
- ⚠️ Don't forget `iter()` or reference patterns for IDs
- ⚠️ Ensure thermal_gen, direct_exchange, reverse_exchange indices are correct

---

## Testing Requirements

### Unit Tests

If mock types are available:
- [ ] Test basic bus with deficit and load only
- [ ] Test bus with thermal generators
- [ ] Test bus with hydro generators (productivity)
- [ ] Test bus with exchange lines (source and target)

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Doc comment on `build` method
- [ ] Explain exchange flow conventions
- [ ] Document productivity usage

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Clear extraction, moderate complexity with exchange flows

---

## Definition of Done

- [ ] Method implemented
- [ ] Logic matches original
- [ ] Exchange signs verified
- [ ] Tests passing
- [ ] Golden tests passing
- [ ] Documented
