# [T-012] Create Constraints Module Structure

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Constraint Extraction](./00-sprint-overview.md)
> **Dependencies**: Sprint 1 complete
> **Blocks**: [T-013](./ticket-013-hydro-balance-constraints.md), [T-014](./ticket-014-bus-balance-constraints.md), [T-015](./ticket-015-ar-dynamics-constraints.md), [T-016](./ticket-016-bound-constraints.md)

---

## ⚠️ CRITICAL: Scaffold Only

This ticket creates the module structure and common abstractions for constraint building. No actual constraint logic is migrated—just the framework for subsequent tickets.

---

## Files to Read Before Starting

- `src/subproblem.rs:2355-2584` - Current constraint building functions
- `src/model/mod.rs` - Current model module structure
- `src/solver.rs` - Problem and Row types
- `plans/clean-code-refactoring/epic-02-core-extraction/00-epic-overview.md` - Future-ready design

---

## Context

### Background

Constraint building in `subproblem.rs` is currently monolithic—all constraint types are built in a single large function `add_constraints()`. This sprint extracts constraint building into focused modules:

- `hydro_balance.rs` - Water balance constraints
- `bus_balance.rs` - Power balance at buses
- `ar_dynamics.rs` - Auto-regressive lag constraints
- `bounds.rs` - Variable bounds (if needed)

### Current Structure

```rust
// In add_constraints() - ~130 lines
fn add_constraints(...) -> Constraints {
    // Load balance constraints (~30 lines)
    for bus in system.buses.iter() { ... }
    
    // Hydro balance constraints (~20 lines)
    for hydro in system.hydros.iter() { ... }
    
    // Uncertainty observation constraints (delegated)
    let uncertainty_observation = Self::add_uncertainty_observation_constraints(...);
    
    // Lag fixing constraints (~50 lines)
    if let Some(ref lag_vars) = variables.lagged_state { ... }
    
    Constraints { ... }
}
```

### Target Structure

```
src/model/constraints/
├── mod.rs           # Common traits and re-exports
├── hydro_balance.rs # HydroBalanceBuilder
├── bus_balance.rs   # BusBalanceBuilder  
├── ar_dynamics.rs   # ArDynamicsBuilder (uncertainty + lags)
└── bounds.rs        # Variable bounds (optional)
```

---

## Specification

### Module Structure to Create

#### 1. `src/model/constraints/mod.rs`

```rust
//! Constraint building for LP subproblems.
//!
//! This module contains builders for different constraint types used in SDDP subproblems.
//! Each builder follows the pattern:
//!
//! 1. Create builder from system data
//! 2. Build constraints into a Problem
//! 3. Return constraint indices for later reference
//!
//! # Future: Preallocation Support
//!
//! Builders are designed to support future preallocation patterns where
//! constraint indices and values are written to preallocated buffers.

pub mod hydro_balance;
pub mod bus_balance;
pub mod ar_dynamics;

pub use hydro_balance::HydroBalanceBuilder;
pub use bus_balance::BusBalanceBuilder;
pub use ar_dynamics::ArDynamicsBuilder;

use crate::solver;

/// Context passed to constraint builders.
///
/// Contains all system and variable information needed for constraint generation.
#[derive(Debug)]
pub struct ConstraintContext<'a> {
    /// The LP problem being built
    pub problem: &'a mut solver::Problem,
    /// Variable indices from subproblem
    pub variables: &'a crate::subproblem::Variables,
    /// System data
    pub system: &'a crate::system::System,
    /// Temporal models (for uncertainty constraints)
    pub temporal_models: &'a [crate::temporal_model::TemporalModel],
    /// Season ID for seasonal parameters
    pub season_id: usize,
}

/// Result of building a constraint group.
///
/// Returns the constraint indices for later reference (dual extraction, updates).
pub struct ConstraintResult {
    /// Indices of the constraints in the LP model
    pub indices: Vec<usize>,
}
```

#### 2. `src/model/constraints/hydro_balance.rs` (Empty scaffold)

```rust
//! Hydro balance constraint builder.
//!
//! Builds water balance constraints of the form:
//! ```text
//! final_storage + turbined_flow + spillage - inflow - upstream_outflow = initial_storage
//! ```

use super::ConstraintContext;

/// Builder for hydro balance constraints.
pub struct HydroBalanceBuilder;

impl HydroBalanceBuilder {
    /// Build hydro balance constraints.
    ///
    /// Creates one constraint per hydro plant.
    pub fn build(_ctx: &mut ConstraintContext) -> Vec<usize> {
        // TODO: Implement in T-013
        todo!("HydroBalanceBuilder::build")
    }
}
```

#### 3. `src/model/constraints/bus_balance.rs` (Empty scaffold)

```rust
//! Bus (load) balance constraint builder.
//!
//! Builds power balance constraints at each bus:
//! ```text
//! generation + imports - exports + deficit = load
//! ```

use super::ConstraintContext;

/// Builder for bus/load balance constraints.
pub struct BusBalanceBuilder;

impl BusBalanceBuilder {
    /// Build bus balance constraints.
    ///
    /// Creates one constraint per bus.
    pub fn build(_ctx: &mut ConstraintContext) -> Vec<usize> {
        // TODO: Implement in T-014
        todo!("BusBalanceBuilder::build")
    }
}
```

#### 4. `src/model/constraints/ar_dynamics.rs` (Empty scaffold)

```rust
//! AR dynamics constraint builders.
//!
//! Builds constraints for auto-regressive uncertainty models:
//! - Uncertainty observation constraints: Y = base + σ·innovation + Σ(ψ·lags)
//! - Lag fixing constraints: Y_{t-k} = value

use super::ConstraintContext;

/// Builder for AR dynamics constraints.
pub struct ArDynamicsBuilder;

impl ArDynamicsBuilder {
    /// Build uncertainty observation constraints.
    ///
    /// Creates one constraint per uncertain entity (load + inflow).
    pub fn build_observation(_ctx: &mut ConstraintContext) -> Vec<usize> {
        // TODO: Implement in T-015
        todo!("ArDynamicsBuilder::build_observation")
    }
    
    /// Build lag fixing constraints.
    ///
    /// Creates constraints that fix lag variables to specific values.
    pub fn build_lag_fixing(
        _ctx: &mut ConstraintContext,
    ) -> (Option<crate::subproblem::LoadLagConstraints>, 
          Option<crate::subproblem::InflowLagConstraints>) {
        // TODO: Implement in T-015
        todo!("ArDynamicsBuilder::build_lag_fixing")
    }
}
```

### Update `src/model/mod.rs`

```rust
//! LP Model Operations
//! ...existing docs...

pub mod variable_indices;
pub mod constraint_indices;
pub mod solution_extract;
pub mod constraints;  // NEW

pub use variable_indices::VariableIndices;
pub use constraint_indices::ConstraintIndices;
pub use solution_extract::SolutionExtractor;
pub use constraints::{ConstraintContext, HydroBalanceBuilder, BusBalanceBuilder, ArDynamicsBuilder};
```

---

## Acceptance Criteria

- [ ] `src/model/constraints/` directory created
- [ ] `mod.rs` with `ConstraintContext` struct
- [ ] `hydro_balance.rs` with `HydroBalanceBuilder` scaffold
- [ ] `bus_balance.rs` with `BusBalanceBuilder` scaffold
- [ ] `ar_dynamics.rs` with `ArDynamicsBuilder` scaffold
- [ ] All modules exported from `src/model/mod.rs`
- [ ] `cargo build` succeeds (todo! warnings acceptable)
- [ ] All existing tests pass
- [ ] Golden tests pass (no behavioral changes)

### Correctness Verification

- [ ] No changes to `subproblem.rs` add_constraints function
- [ ] Only new files created
- [ ] Imports compile correctly

---

## Implementation Guide

### Suggested Approach

1. **Create directory**:
   ```bash
   mkdir -p src/model/constraints
   ```

2. **Create `src/model/constraints/mod.rs`** with ConstraintContext and re-exports

3. **Create scaffold files**:
   - `hydro_balance.rs`
   - `bus_balance.rs`
   - `ar_dynamics.rs`

4. **Update `src/model/mod.rs`** to include constraints module

5. **Verify compilation**:
   ```bash
   cargo build
   ```

6. **Run tests**:
   ```bash
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Create

| File | Purpose |
|------|---------|
| `src/model/constraints/mod.rs` | Module root, ConstraintContext |
| `src/model/constraints/hydro_balance.rs` | Hydro constraint scaffold |
| `src/model/constraints/bus_balance.rs` | Bus constraint scaffold |
| `src/model/constraints/ar_dynamics.rs` | AR constraint scaffold |

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/mod.rs` | Add `pub mod constraints;` and re-exports |

### Patterns to Follow

- Use `todo!()` for unimplemented methods
- Document the purpose of each builder
- Use `ConstraintContext` to pass all needed data
- Return constraint indices as `Vec<usize>`

### Pitfalls to Avoid

- ⚠️ Don't implement actual constraint logic—just scaffolds
- ⚠️ Don't modify `subproblem.rs`
- ⚠️ Ensure imports resolve correctly for all types
- ⚠️ Verify `LoadLagConstraints` and `InflowLagConstraints` are importable

---

## Testing Requirements

### Compilation Test

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes (no new tests needed)

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Module-level documentation on `constraints/mod.rs`
- [ ] Doc comments on `ConstraintContext`
- [ ] Doc comments on each builder scaffold

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward directory/file creation with clear structure

---

## Definition of Done

- [ ] Directory structure created
- [ ] All scaffold files created
- [ ] `ConstraintContext` struct defined
- [ ] Module exports working
- [ ] `cargo build` succeeds
- [ ] Tests pass
- [ ] Golden tests pass
- [ ] Ready for T-013, T-014, T-015
