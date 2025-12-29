# [T-007] Create VariableIndices and ConstraintIndices Structs

> **Epic**: [Epic 2: Core Extraction](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Solution Extraction](./00-sprint-overview.md)
> **Dependencies**: [T-006](./ticket-006-analyze-extraction-points.md)
> **Blocks**: [T-008](./ticket-008-solution-extractor-scaffold.md)

---

## ⚠️ CRITICAL: Structural Change Only

This ticket creates new structs that will be used by `SolutionExtractor`. The existing `Variables` and `Constraints` structs in `subproblem.rs` remain unchanged—we are creating **extraction-focused** versions that provide cleaner APIs.

Run golden tests after implementation to verify no behavioral changes.

---

## Files to Read Before Starting

- `src/subproblem.rs:700-750` - Current `Variables` and `Constraints` structs
- `src/model/mod.rs` - Where new structs will live
- `docs/extraction-analysis.md` - Analysis from T-006
- `plans/clean-code-refactoring/epic-02-core-extraction/00-epic-overview.md` - SoA-ready API design

---

## Context

### Background

The current `Variables` struct stores individual indices in `Vec<usize>`. For solution extraction, we need to efficiently compute ranges like `first..last+1`. The new `VariableIndices` struct will precompute these ranges for O(1) access during hot-path extraction.

### Current Pattern (Inefficient)

```rust
// Current: Compute range every time
let first = *self.variables.deficit.first().unwrap();
let last = *self.variables.deficit.last().unwrap() + 1;
realization.deficit.clone_from_slice(&solution.colvalue[first..last]);
```

### Target Pattern (Efficient)

```rust
// Target: Precomputed range
let range = indices.deficit_range();
target.copy_from_slice(&solution.colvalue[range]);
```

---

## Specification

### Structs to Create

#### 1. `VariableIndices` - Primal Variable Ranges

```rust
// src/model/variable_indices.rs

use std::ops::Range;

/// Precomputed variable index ranges for efficient solution extraction.
///
/// These ranges are computed once during subproblem construction and
/// used repeatedly during SDDP forward/backward passes.
///
/// # Design for SoA Migration
///
/// The range-based API enables future SoA layouts where data is extracted
/// directly into contiguous arrays rather than per-realization structs.
#[derive(Clone, Debug)]
pub struct VariableIndices {
    /// Deficit variable range in LP solution
    deficit: Range<usize>,
    /// Direct exchange variable range
    direct_exchange: Option<Range<usize>>,
    /// Reverse exchange variable range
    reverse_exchange: Option<Range<usize>>,
    /// Thermal generation variable range
    thermal_gen: Option<Range<usize>>,
    /// Turbined flow variable range
    turbined_flow: Range<usize>,
    /// Spillage variable range
    spillage: Range<usize>,
    /// Stored volume (final storage) variable range
    stored_volume: Range<usize>,
    /// Load observation variable indices (may not be contiguous)
    load: Vec<usize>,
    /// Inflow observation variable indices (may not be contiguous)
    inflow: Vec<usize>,
}

impl VariableIndices {
    /// Create from existing Variables struct.
    ///
    /// This is the bridge between old and new code.
    pub fn from_variables(vars: &crate::subproblem::Variables) -> Self {
        // Implementation computes ranges from Vec<usize>
    }

    // Range accessors - all return Range<usize> for slice operations
    #[inline]
    pub fn deficit_range(&self) -> Range<usize> { self.deficit.clone() }
    
    #[inline]
    pub fn direct_exchange_range(&self) -> Option<Range<usize>> { 
        self.direct_exchange.clone() 
    }
    
    // ... etc for all fields
    
    /// Check if exchange variables exist
    #[inline]
    pub fn has_exchange(&self) -> bool { self.direct_exchange.is_some() }
    
    /// Check if thermal variables exist
    #[inline]
    pub fn has_thermal(&self) -> bool { self.thermal_gen.is_some() }
}
```

#### 2. `ConstraintIndices` - Dual Variable Ranges

```rust
// src/model/constraint_indices.rs

use std::ops::Range;

/// Precomputed constraint index ranges for efficient dual extraction.
///
/// Used to extract marginal costs, water values, and lag duals.
#[derive(Clone, Debug)]
pub struct ConstraintIndices {
    /// Load balance constraint range (for marginal costs)
    load_balance: Range<usize>,
    /// Hydro balance constraint range (for water values)
    hydro_balance: Range<usize>,
    /// Load lag constraint indices by bus_id
    load_lag: Option<Vec<Vec<usize>>>,
    /// Inflow lag constraint indices by hydro_id
    inflow_lag: Option<Vec<Vec<usize>>>,
}

impl ConstraintIndices {
    /// Create from existing Constraints struct.
    pub fn from_constraints(cons: &crate::subproblem::Constraints) -> Self {
        // Implementation
    }
    
    #[inline]
    pub fn load_balance_range(&self) -> Range<usize> { 
        self.load_balance.clone() 
    }
    
    #[inline]
    pub fn hydro_balance_range(&self) -> Range<usize> { 
        self.hydro_balance.clone() 
    }
    
    /// Get lag constraints for a specific bus
    pub fn load_lag_constraints(&self, bus_id: usize) -> &[usize] {
        // Return empty slice if None or out of bounds
    }
    
    /// Get lag constraints for a specific hydro
    pub fn inflow_lag_constraints(&self, hydro_id: usize) -> &[usize] {
        // Return empty slice if None or out of bounds
    }
}
```

### Module Structure

```
src/model/
├── mod.rs                  # Add: pub mod variable_indices; pub mod constraint_indices;
├── variable_indices.rs     # NEW: VariableIndices struct
└── constraint_indices.rs   # NEW: ConstraintIndices struct
```

### Behavior

- `from_variables()` and `from_constraints()` convert from existing structs
- All range methods return `Range<usize>` for direct use with slice indexing
- Optional fields use `Option<Range<usize>>` pattern
- Non-contiguous indices (load, inflow) stay as `Vec<usize>`

### Error Handling

- Panic if input Vec is empty when range is required (caller ensures valid data)
- Return `None` for optional ranges when not present
- Return empty slice for missing lag constraints

---

## Acceptance Criteria

- [ ] `VariableIndices` struct created in `src/model/variable_indices.rs`
- [ ] `ConstraintIndices` struct created in `src/model/constraint_indices.rs`
- [ ] Both structs have `from_*` constructors for existing types
- [ ] All range accessor methods implemented with `#[inline]`
- [ ] Unit tests verify range computation correctness
- [ ] `src/model/mod.rs` exports both structs
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass (no behavioral changes)

### Correctness Verification

- [ ] Ranges computed from existing Variables match expected first..last+1 pattern
- [ ] Optional fields correctly handle empty/missing data
- [ ] No changes to existing `Variables` or `Constraints` structs

---

## Implementation Guide

### Suggested Approach

1. **Create `variable_indices.rs`**:
   ```rust
   // src/model/variable_indices.rs
   use std::ops::Range;
   
   #[derive(Clone, Debug)]
   pub struct VariableIndices {
       deficit: Range<usize>,
       direct_exchange: Option<Range<usize>>,
       reverse_exchange: Option<Range<usize>>,
       thermal_gen: Option<Range<usize>>,
       turbined_flow: Range<usize>,
       spillage: Range<usize>,
       stored_volume: Range<usize>,
       load: Vec<usize>,
       inflow: Vec<usize>,
   }
   
   impl VariableIndices {
       /// Create from existing Variables struct
       pub fn from_variables(vars: &crate::subproblem::Variables) -> Self {
           Self {
               deficit: Self::vec_to_range(&vars.deficit),
               direct_exchange: Self::vec_to_optional_range(&vars.direct_exchange),
               reverse_exchange: Self::vec_to_optional_range(&vars.reverse_exchange),
               thermal_gen: Self::vec_to_optional_range(&vars.thermal_gen),
               turbined_flow: Self::vec_to_range(&vars.turbined_flow),
               spillage: Self::vec_to_range(&vars.spillage),
               stored_volume: Self::vec_to_range(&vars.stored_volume),
               load: vars.load.clone(),
               inflow: vars.inflow.clone(),
           }
       }
       
       /// Convert contiguous Vec<usize> to Range
       fn vec_to_range(indices: &[usize]) -> Range<usize> {
           let first = *indices.first().expect("indices must not be empty");
           let last = *indices.last().expect("indices must not be empty");
           first..last + 1
       }
       
       /// Convert optional Vec<usize> to Option<Range>
       fn vec_to_optional_range(indices: &[usize]) -> Option<Range<usize>> {
           if indices.is_empty() {
               None
           } else {
               Some(Self::vec_to_range(indices))
           }
       }
       
       // Accessors
       #[inline]
       pub fn deficit_range(&self) -> Range<usize> { self.deficit.clone() }
       
       #[inline]
       pub fn direct_exchange_range(&self) -> Option<Range<usize>> { 
           self.direct_exchange.clone() 
       }
       
       #[inline]
       pub fn reverse_exchange_range(&self) -> Option<Range<usize>> { 
           self.reverse_exchange.clone() 
       }
       
       #[inline]
       pub fn thermal_gen_range(&self) -> Option<Range<usize>> { 
           self.thermal_gen.clone() 
       }
       
       #[inline]
       pub fn turbined_flow_range(&self) -> Range<usize> { 
           self.turbined_flow.clone() 
       }
       
       #[inline]
       pub fn spillage_range(&self) -> Range<usize> { 
           self.spillage.clone() 
       }
       
       #[inline]
       pub fn stored_volume_range(&self) -> Range<usize> { 
           self.stored_volume.clone() 
       }
       
       #[inline]
       pub fn load_indices(&self) -> &[usize] { &self.load }
       
       #[inline]
       pub fn inflow_indices(&self) -> &[usize] { &self.inflow }
       
       #[inline]
       pub fn has_exchange(&self) -> bool { self.direct_exchange.is_some() }
       
       #[inline]
       pub fn has_thermal(&self) -> bool { self.thermal_gen.is_some() }
   }
   ```

2. **Create `constraint_indices.rs`**:
   ```rust
   // src/model/constraint_indices.rs
   use std::ops::Range;
   
   #[derive(Clone, Debug)]
   pub struct ConstraintIndices {
       load_balance: Range<usize>,
       hydro_balance: Range<usize>,
       load_lag: Option<Vec<Vec<usize>>>,
       inflow_lag: Option<Vec<Vec<usize>>>,
   }
   
   impl ConstraintIndices {
       pub fn from_constraints(cons: &crate::subproblem::Constraints) -> Self {
           let load_balance = Self::vec_to_range(&cons.load_balance);
           let hydro_balance = Self::vec_to_range(&cons.hydro_balance);
           
           let load_lag = cons.load_lag_constraints.as_ref().map(|lc| {
               lc.constraints_by_bus.clone()
           });
           
           let inflow_lag = cons.inflow_lag_constraints.as_ref().map(|ic| {
               ic.constraints_by_hydro.clone()
           });
           
           Self { load_balance, hydro_balance, load_lag, inflow_lag }
       }
       
       fn vec_to_range(indices: &[usize]) -> Range<usize> {
           let first = *indices.first().expect("indices must not be empty");
           let last = *indices.last().expect("indices must not be empty");
           first..last + 1
       }
       
       #[inline]
       pub fn load_balance_range(&self) -> Range<usize> { 
           self.load_balance.clone() 
       }
       
       #[inline]
       pub fn hydro_balance_range(&self) -> Range<usize> { 
           self.hydro_balance.clone() 
       }
       
       pub fn load_lag_constraints(&self, bus_id: usize) -> &[usize] {
           self.load_lag.as_ref()
               .and_then(|lags| lags.get(bus_id))
               .map(|v| v.as_slice())
               .unwrap_or(&[])
       }
       
       pub fn inflow_lag_constraints(&self, hydro_id: usize) -> &[usize] {
           self.inflow_lag.as_ref()
               .and_then(|lags| lags.get(hydro_id))
               .map(|v| v.as_slice())
               .unwrap_or(&[])
       }
       
       pub fn num_buses(&self) -> usize {
           self.load_lag.as_ref().map(|v| v.len()).unwrap_or(0)
       }
       
       pub fn num_hydros(&self) -> usize {
           self.inflow_lag.as_ref().map(|v| v.len()).unwrap_or(0)
       }
   }
   ```

3. **Update `src/model/mod.rs`**:
   ```rust
   //! LP Model Operations
   //! ...existing docs...
   
   pub mod variable_indices;
   pub mod constraint_indices;
   
   pub use variable_indices::VariableIndices;
   pub use constraint_indices::ConstraintIndices;
   ```

4. **Add unit tests** (at bottom of each file):
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       
       #[test]
       fn test_vec_to_range() {
           let indices = vec![5, 6, 7, 8, 9];
           let range = VariableIndices::vec_to_range(&indices);
           assert_eq!(range, 5..10);
       }
       
       #[test]
       fn test_optional_range_empty() {
           let indices: Vec<usize> = vec![];
           let range = VariableIndices::vec_to_optional_range(&indices);
           assert!(range.is_none());
       }
       
       #[test]
       fn test_optional_range_present() {
           let indices = vec![10, 11, 12];
           let range = VariableIndices::vec_to_optional_range(&indices);
           assert_eq!(range, Some(10..13));
       }
   }
   ```

5. **Verify build and tests**:
   ```bash
   cargo build
   cargo test
   ./scripts/golden-tests.sh verify
   ```

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/model/mod.rs` | Add module declarations and re-exports |
| `src/model/variable_indices.rs` | NEW: VariableIndices struct |
| `src/model/constraint_indices.rs` | NEW: ConstraintIndices struct |

### Patterns to Follow

- Use `#[inline]` on all accessor methods
- Use `Range<usize>` for contiguous indices
- Use `Option<Range<usize>>` for optional fields
- Keep `Vec<usize>` for non-contiguous indices (load, inflow)
- Clone ranges in accessors (Range is Copy-like, cheap to clone)

### Pitfalls to Avoid

- ⚠️ Don't modify existing `Variables` or `Constraints` structs
- ⚠️ Don't assume all fields have data—handle empty vecs for optional fields
- ⚠️ Don't forget the `+1` in range computation (exclusive end)
- ⚠️ Don't expose internal fields directly—use accessor methods
- ⚠️ Verify `LoadLagConstraints` and `InflowLagConstraints` field names in subproblem.rs

---

## Testing Requirements

### Unit Tests

- [ ] Test `vec_to_range` with normal input
- [ ] Test `vec_to_range` with single element
- [ ] Test `vec_to_optional_range` with empty input
- [ ] Test `vec_to_optional_range` with data
- [ ] Test `has_exchange()` when present/absent
- [ ] Test `has_thermal()` when present/absent
- [ ] Test lag constraint accessors with valid/invalid indices

### Integration Tests

No integration tests needed—these are data structures only.

### Golden Tests

- [ ] Run `./scripts/golden-tests.sh verify` to ensure no behavioral changes

---

## Documentation Requirements

- [ ] Doc comments on all public structs and methods
- [ ] Explain the range computation (first..last+1)
- [ ] Note which fields are optional and why
- [ ] Document the SoA migration design intent

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward struct creation with clear specifications; some complexity in handling optional fields and lag constraints

---

## Definition of Done

- [ ] Both structs created and compile
- [ ] All accessor methods implemented
- [ ] Unit tests pass
- [ ] `cargo build` succeeds
- [ ] `cargo test` passes
- [ ] Golden tests pass
- [ ] Code documented
- [ ] No existing code modified (only additions)
