# [T-008] Create SolutionExtractor Scaffold with Dual API

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-007](./ticket-007-variable-indices-struct.md)
> **Blocks**: [T-009](./ticket-009-extract-hydro-solution.md), [T-010](./ticket-010-extract-thermal-solution.md)

---

## ⚠️ CRITICAL: Scaffold Only

This ticket creates the `SolutionExtractor` struct and its dual API pattern. The actual extraction methods will be implemented in subsequent tickets (T-009, T-010, T-011). Focus on establishing the correct structure and API design.

---

## Files to Read Before Starting

- `src/model/variable_indices.rs` - VariableIndices from T-007
- `src/model/constraint_indices.rs` - ConstraintIndices from T-007
- `src/subproblem.rs:1862-2100` - Current extraction functions
- `src/solver.rs` - Solution struct definition
- `src/sddp/mod.rs` - Realization struct location
- `plans/clean-code-refactoring/epic-02-core-extraction/00-epic-overview.md` - Dual API design

---

## Context

### Background

The `SolutionExtractor` is the central abstraction for extracting LP solution values into domain types. It implements a **dual API pattern**:

1. **`extract_X_into(&self, solution, target_slice)`** - Low-level, works with raw slices (SoA-ready)
2. **`extract_X(&self, solution, realization)`** - High-level, works with Realization (current pattern)

This design enables future SoA migration while preserving current functionality.

### Current Pattern

```rust
// In subproblem.rs
fn get_deficit_from_solution(&self, solution: &Solution, realization: &mut Realization) {
    let first = *self.variables.deficit.first().unwrap();
    let last = *self.variables.deficit.last().unwrap() + 1;
    realization.deficit.clone_from_slice(&solution.colvalue[first..last]);
}
```

### Target Pattern

```rust
impl SolutionExtractor {
    /// Low-level: extract into any slice (SoA-ready)
    #[inline]
    pub fn extract_deficit_into(&self, solution: &Solution, target: &mut [f64]) {
        let range = self.var_indices.deficit_range();
        target.copy_from_slice(&solution.colvalue[range]);
    }
    
    /// High-level: extract into Realization (current API)
    #[inline]
    pub fn extract_deficit(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_deficit_into(solution, &mut realization.deficit);
    }
}
```

---

## Specification

### Struct to Create

```rust
// src/model/solution_extract.rs

use crate::model::{VariableIndices, ConstraintIndices};
use crate::solver::Solution;
use crate::sddp::Realization;  // Verify actual location

/// Extracts LP solution values into domain types.
///
/// This struct encapsulates all solution extraction logic, providing both
/// low-level slice-based APIs (for future SoA layouts) and high-level
/// Realization-based APIs (for current usage).
///
/// # Design Principles
///
/// 1. **Zero allocation**: All extraction uses preallocated buffers
/// 2. **Dual API**: `extract_X_into()` for slices, `extract_X()` for Realization
/// 3. **Inlined hot paths**: All extraction methods are `#[inline]`
/// 4. **Single responsibility**: Only extraction logic, no constraint building
///
/// # Example
///
/// ```ignore
/// let extractor = SolutionExtractor::new(var_indices, con_indices);
/// 
/// // Low-level API (SoA-ready)
/// extractor.extract_deficit_into(&solution, &mut deficit_buffer);
/// 
/// // High-level API (current pattern)
/// extractor.extract_deficit(&solution, &mut realization);
/// ```
#[derive(Clone, Debug)]
pub struct SolutionExtractor {
    var_indices: VariableIndices,
    con_indices: ConstraintIndices,
}

impl SolutionExtractor {
    /// Create a new SolutionExtractor from index structs.
    pub fn new(var_indices: VariableIndices, con_indices: ConstraintIndices) -> Self {
        Self { var_indices, con_indices }
    }
    
    /// Create from existing Variables and Constraints.
    ///
    /// This is the bridge for migrating existing code.
    pub fn from_subproblem_types(
        variables: &crate::subproblem::Variables,
        constraints: &crate::subproblem::Constraints,
    ) -> Self {
        Self::new(
            VariableIndices::from_variables(variables),
            ConstraintIndices::from_constraints(constraints),
        )
    }
    
    // === Primal Variable Extraction (from solution.colvalue) ===
    
    // Deficit extraction
    #[inline]
    pub fn extract_deficit_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-009 or T-011
        todo!("extract_deficit_into")
    }
    
    #[inline]
    pub fn extract_deficit(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_deficit_into(solution, &mut realization.deficit);
    }
    
    // Exchange extraction (direct - reverse)
    #[inline]
    pub fn extract_exchange_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-009 or T-011
        todo!("extract_exchange_into")
    }
    
    #[inline]
    pub fn extract_exchange(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_exchange_into(solution, &mut realization.exchange);
    }
    
    // Thermal generation extraction
    #[inline]
    pub fn extract_thermal_gen_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-010
        todo!("extract_thermal_gen_into")
    }
    
    #[inline]
    pub fn extract_thermal_gen(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_thermal_gen_into(solution, &mut realization.thermal_generation);
    }
    
    // Spillage extraction
    #[inline]
    pub fn extract_spillage_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-009
        todo!("extract_spillage_into")
    }
    
    #[inline]
    pub fn extract_spillage(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_spillage_into(solution, &mut realization.spillage);
    }
    
    // Turbined flow extraction
    #[inline]
    pub fn extract_turbined_flow_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-009
        todo!("extract_turbined_flow_into")
    }
    
    #[inline]
    pub fn extract_turbined_flow(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_turbined_flow_into(solution, &mut realization.turbined_flow);
    }
    
    // Final storage extraction
    #[inline]
    pub fn extract_final_storage_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-009
        todo!("extract_final_storage_into")
    }
    
    #[inline]
    pub fn extract_final_storage(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_final_storage_into(solution, &mut realization.final_storage);
    }
    
    // Load extraction
    #[inline]
    pub fn extract_load_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-011
        todo!("extract_load_into")
    }
    
    #[inline]
    pub fn extract_load(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_load_into(solution, &mut realization.loads);
    }
    
    // Inflow extraction
    #[inline]
    pub fn extract_inflow_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-011
        todo!("extract_inflow_into")
    }
    
    #[inline]
    pub fn extract_inflow(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_inflow_into(solution, &mut realization.inflow);
    }
    
    // === Dual Variable Extraction (from solution.rowdual) ===
    
    // Water values (hydro balance duals)
    #[inline]
    pub fn extract_water_values_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-011
        todo!("extract_water_values_into")
    }
    
    #[inline]
    pub fn extract_water_values(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_water_values_into(solution, &mut realization.water_value);
    }
    
    // Marginal costs (load balance duals)
    #[inline]
    pub fn extract_marginal_costs_into(&self, solution: &Solution, target: &mut [f64]) {
        // TODO: Implement in T-011
        todo!("extract_marginal_costs_into")
    }
    
    #[inline]
    pub fn extract_marginal_costs(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_marginal_costs_into(solution, &mut realization.marginal_cost);
    }
    
    // Lag duals (complex extraction - may not have _into variant)
    pub fn extract_lag_duals(&self, solution: &Solution, realization: &mut Realization) {
        // TODO: Implement in T-011
        todo!("extract_lag_duals")
    }
    
    // === Convenience Methods ===
    
    /// Check if exchange variables exist
    #[inline]
    pub fn has_exchange(&self) -> bool {
        self.var_indices.has_exchange()
    }
    
    /// Check if thermal variables exist
    #[inline]
    pub fn has_thermal(&self) -> bool {
        self.var_indices.has_thermal()
    }
    
    /// Extract all primal variables into realization.
    ///
    /// This replaces multiple individual calls in the hot path.
    pub fn extract_all_primals(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_deficit(solution, realization);
        if self.has_exchange() {
            self.extract_exchange(solution, realization);
        }
        if self.has_thermal() {
            self.extract_thermal_gen(solution, realization);
        }
        self.extract_spillage(solution, realization);
        self.extract_turbined_flow(solution, realization);
        self.extract_final_storage(solution, realization);
        self.extract_load(solution, realization);
        self.extract_inflow(solution, realization);
    }
    
    /// Extract all dual variables into realization.
    pub fn extract_all_duals(&self, solution: &Solution, realization: &mut Realization) {
        self.extract_water_values(solution, realization);
        self.extract_marginal_costs(solution, realization);
        self.extract_lag_duals(solution, realization);
    }
}
```

### Module Updates

Update `src/model/mod.rs`:
```rust
pub mod variable_indices;
pub mod constraint_indices;
pub mod solution_extract;

pub use variable_indices::VariableIndices;
pub use constraint_indices::ConstraintIndices;
pub use solution_extract::SolutionExtractor;
```

---

## Acceptance Criteria

- [ ] `SolutionExtractor` struct created in `src/model/solution_extract.rs`
- [ ] All method signatures defined (with `todo!()` bodies)
- [ ] `new()` and `from_subproblem_types()` constructors implemented
- [ ] Dual API pattern established (`extract_X_into` + `extract_X`)
- [ ] `extract_all_primals` and `extract_all_duals` convenience methods defined
- [ ] `has_exchange()` and `has_thermal()` helpers implemented
- [ ] Module exported from `src/model/mod.rs`
- [ ] Doc comments on all public items
- [ ] `cargo build` succeeds (with `todo!()` warnings acceptable)
- [ ] Existing tests still pass

### Correctness Verification

- [ ] No changes to existing extraction logic in `subproblem.rs`
- [ ] Types align with existing `Solution` and `Realization` structs
- [ ] Golden tests still pass

---

## Implementation Guide

### Suggested Approach

1. **Find the Realization struct location**:
   ```bash
   grep -rn "struct Realization" src/
   ```

2. **Create `src/model/solution_extract.rs`**:
   - Copy the struct definition from Specification above
   - Replace `todo!()` with actual `todo!("method_name")` for clarity
   - Ensure correct import paths

3. **Handle Realization import**:
   - The `Realization` struct is likely in `src/sddp/mod.rs` or similar
   - Add appropriate `use` statement
   - May need to make `Realization` public if not already

4. **Update `src/model/mod.rs`**:
   ```rust
   pub mod variable_indices;
   pub mod constraint_indices;
   pub mod solution_extract;
   
   pub use variable_indices::VariableIndices;
   pub use constraint_indices::ConstraintIndices;
   pub use solution_extract::SolutionExtractor;
   ```

5. **Verify compilation**:
   ```bash
   cargo build 2>&1 | head -50
   ```
   - `todo!()` warnings are expected and acceptable
   - Fix any actual errors (import paths, visibility)

6. **Run tests**:
   ```bash
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify/Create

| File | Changes |
|------|---------|
| `src/model/solution_extract.rs` | NEW: SolutionExtractor struct |
| `src/model/mod.rs` | Add module and re-export |

### Patterns to Follow

- Use `#[inline]` on all extraction methods
- Use `todo!("method_name")` for unimplemented methods
- High-level methods call low-level `_into` variants
- Keep method signatures consistent across all extraction types
- Document the dual API pattern in struct-level docs

### Pitfalls to Avoid

- ⚠️ Don't implement the actual extraction logic yet—just scaffolding
- ⚠️ Don't modify `subproblem.rs`—this is additive only
- ⚠️ Ensure `Realization` is importable (may need visibility changes)
- ⚠️ Check that `Solution` import path is correct (`crate::solver::Solution`)
- ⚠️ Don't add tests for `todo!()` methods—they'll panic

---

## Testing Requirements

### Compilation Test

- [ ] `cargo build` succeeds with only `todo!()` warnings
- [ ] `cargo test` passes (no new tests for todo methods)

### No Functional Tests Yet

The actual extraction tests will be added in T-009, T-010, T-011 when methods are implemented.

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes (no behavioral changes)

---

## Documentation Requirements

- [ ] Struct-level documentation explaining dual API pattern
- [ ] Doc comments on all public methods
- [ ] Example usage in struct documentation
- [ ] Document which methods are SoA-ready

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Mostly boilerplate struct creation; main complexity is ensuring correct imports and visibility

---

## Definition of Done

- [ ] `SolutionExtractor` struct created
- [ ] All method signatures defined
- [ ] Constructors implemented
- [ ] Module exported
- [ ] Documentation complete
- [ ] `cargo build` succeeds
- [ ] Existing tests pass
- [ ] Golden tests pass
- [ ] Ready for T-009/T-010 to implement methods
