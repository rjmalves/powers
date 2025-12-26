# [TICKET-002] Extend SizingInfo with cut estimation methods

> **Epic**: [Epic 1: HiGHS Constraint Preallocation](../00-epic-overview.md)  
> **Sprint**: [Sprint 1](./00-sprint-overview.md)  
> **Dependencies**: None  
> **Blocks**: [TICKET-003](./ticket-003-subproblem-cut-slots.md)

## Context

### Background

To preallocate HiGHS constraint slots, we need to estimate the maximum number of cuts per node at training start. `SizingInfo` already computes buffer dimensions from input; we need to add cut estimation methods.

### Relation to Epic

This ticket provides the sizing calculations needed to determine how many cut slots to preallocate.

### Current State

`SizingInfo` in `src/memory/sizing.rs` has:
- `estimate_cuts_for_node()`: Heuristic for stabilized cut count ✅
- `estimate_cuts_without_selection()`: Max cuts without selection ✅
- `estimate_max_cuts_per_node()`: **Missing** ❌ (for preallocation)
- `estimate_lp_dimensions()`: **Missing** ❌ (total rows/cols)

## Files to Read Before Starting

- `src/memory/sizing.rs` - SizingInfo implementation (lines 155-750)
- `PREALLOCATION_STATUS_2025_12.md` - Required additions (lines 375-404)

## Specification

### New Methods

**`estimate_max_cuts_per_node()`**:
- Returns conservative estimate of maximum cuts per node
- Used for HiGHS preallocation (must not underestimate)
- Formula: `num_iterations * num_forward_passes` (worst case: no cut selection)

**`estimate_lp_dimensions()`**:
- Returns `(total_rows, total_cols)` including preallocated cut slots
- `total_rows = base_constraints + max_cuts`
- `total_cols = 3*num_hydros + num_thermals + num_buses + 1`

### Outputs

- `estimate_max_cuts_per_node() -> usize`
- `estimate_lp_dimensions() -> (usize, usize)`

### Behavior

- Conservative estimates to avoid slot exhaustion
- Use existing `num_iterations` and `num_forward_passes` from config

## Acceptance Criteria

- [ ] `estimate_max_cuts_per_node()` method added
- [ ] `estimate_lp_dimensions()` method added
- [ ] Both methods have doc comments
- [ ] Unit tests added for new methods
- [ ] Examples 01 and 07 still pass

## Implementation Guide

### Suggested Approach

1. Add `estimate_max_cuts_per_node()` (simple multiplication)
2. Add `estimate_lp_dimensions()` using existing field values
3. Add unit tests following existing test patterns

### Key Files to Modify

- `src/memory/sizing.rs`: Add methods after `estimate_cuts_without_selection()` (~line 747)

### Code Template

```rust
// src/memory/sizing.rs - Add after estimate_cuts_without_selection()

/// Estimate maximum cuts per node for preallocation.
///
/// Returns a **conservative estimate** (worst case) assuming cut selection
/// is disabled or ineffective. This ensures preallocation never runs out
/// of slots.
///
/// # Formula
///
/// ```text
/// max_cuts = num_iterations * num_forward_passes
/// ```
///
/// # Example
///
/// For 20 iterations with 10 forward passes:
/// - Max cuts = 20 × 10 = 200 cuts per node
///
/// # Note
///
/// This is intentionally conservative. In practice, cut selection reduces
/// active cuts to 10-30% of this maximum. Use `estimate_cuts_for_node()`
/// for realistic memory estimation.
pub fn estimate_max_cuts_per_node(&self) -> usize {
    self.max_iterations * self.num_forward_passes
}

/// Estimate LP dimensions including preallocated cut slots.
///
/// Returns `(num_rows, num_cols)` for the complete LP with maximum cuts.
///
/// # Row Composition
///
/// - Hydro balance constraints: `num_hydros`
/// - Bus balance constraints: `num_buses`  
/// - Line limit constraints: `2 * num_lines`
/// - Preallocated cut slots: `max_cuts_per_node`
///
/// # Column Composition
///
/// - Hydro variables: `3 * num_hydros` (generation, spillage, storage)
/// - Thermal variables: `num_thermals`
/// - Deficit variables: `num_buses`
/// - Future cost (alpha): `1`
///
/// # Example
///
/// ```ignore
/// let (rows, cols) = sizing.estimate_lp_dimensions();
/// println!("LP size: {} rows × {} cols", rows, cols);
/// ```
pub fn estimate_lp_dimensions(&self) -> (usize, usize) {
    // Base constraints (excluding cuts)
    let base_rows = self.num_hydros  // Hydro balance
                  + self.num_buses   // Bus balance
                  + 2 * self.num_lines;  // Line limits (forward + reverse)
    
    // Add preallocated cut slots
    let max_cuts = self.estimate_max_cuts_per_node();
    let total_rows = base_rows + max_cuts;
    
    // Variables
    let total_cols = 3 * self.num_hydros  // generation, spillage, storage
                   + self.num_thermals     // thermal generation
                   + self.num_buses        // deficit
                   + 1;                    // alpha (future cost)
    
    (total_rows, total_cols)
}
```

### Test Template

```rust
// Add to mod tests in src/memory/sizing.rs

#[test]
fn test_estimate_max_cuts_per_node() {
    let system = make_test_system(3, 2, 4, 5);
    let graph = make_test_graph(8, "storage");
    let config = make_test_config(20, 10, None, None);

    let sizing = SizingInfo::from_input(&system, &graph, &config);
    
    assert_eq!(sizing.estimate_max_cuts_per_node(), 200);
}

#[test]
fn test_estimate_lp_dimensions() {
    let system = make_test_system(3, 2, 4, 5);
    let graph = make_test_graph(8, "storage");
    let config = make_test_config(10, 4, None, None);

    let sizing = SizingInfo::from_input(&system, &graph, &config);
    
    let (rows, cols) = sizing.estimate_lp_dimensions();
    
    // Base rows: 3 (hydro) + 4 (bus) + 2*5 (lines) = 17
    // Max cuts: 10 * 4 = 40
    // Total rows: 17 + 40 = 57
    assert_eq!(rows, 57);
    
    // Cols: 3*3 (hydro) + 2 (thermal) + 4 (deficit) + 1 (alpha) = 16
    assert_eq!(cols, 16);
}
```

### Pitfalls to Avoid

- ⚠️ Don't use `estimate_cuts_for_node()` for preallocation (underestimates)
- ⚠️ Remember line limits are bidirectional (2× lines)

## Testing Requirements

### Unit Tests

- [ ] `test_estimate_max_cuts_per_node()` with various configs
- [ ] `test_estimate_lp_dimensions()` validates row/col calculation

### Integration Tests

- [ ] Examples 01 and 07 still produce correct results

## Documentation Requirements

- [ ] Doc comments with formulas and examples
- [ ] Note difference from `estimate_cuts_for_node()`

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Simple arithmetic methods following existing patterns

## Definition of Done

- [ ] Implementation complete
- [ ] Unit tests pass
- [ ] Doc comments added
- [ ] Examples still work
