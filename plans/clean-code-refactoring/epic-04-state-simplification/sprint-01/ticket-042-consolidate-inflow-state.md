# [T-042] Consolidate StorageAndInflowState Methods

> **Epic**: [Epic 4: State Simplification](../00-epic-overview.md)
> **Sprint**: [Sprint 1: State Consolidation](./00-sprint-overview.md)
> **Dependencies**: [T-040](./ticket-040-extract-state-utilities.md)
> **Blocks**: [T-043](./ticket-043-pool-compatible-extensions.md)

## Files to Read Before Starting

- `src/state.rs` - StorageAndInflowState implementation (lines 995-1560)
- `src/state/common.rs` (or equivalent) - StateCore from T-040
- T-038 analysis results
- T-041 for pattern to follow

---

## Context

### Background

Similar to T-041 for `StorageState`, this ticket refactors `StorageAndInflowState` to use `StateCore` composition. This type is more complex due to:
- `StateLayout` for heterogeneous AR orders
- Additional lag extraction methods
- More complex coefficient structure

### Current State

```rust
pub struct StorageAndInflowState {
    dimension: usize,                    // num_hydros (not state dimension!)
    layout: StateLayout,                 // Per-hydro dimensions and offsets
    state_coefficients: Vec<f64>,        // [storage + lags interleaved]
    dominating_objective: f64,
    dominating_cut_id: usize,
    iteration: usize,
    forward_pass_idx: usize,
}
```

### Target State

```rust
pub struct StorageAndInflowState {
    core: StateCore,      // Common fields (state_coefficients, tracking)
    dimension: usize,     // num_hydros - needed for iteration
    layout: StateLayout,  // Per-hydro structure
}
```

**Note**: `dimension` in `StorageAndInflowState` is `num_hydros`, while `core.dimension` is `layout.total_dim` (total state size). This distinction must be preserved.

---

## Specification

### Changes Required

1. **Replace common fields with `StateCore` embed**
2. **Keep `layout` and `dimension` (num_hydros)**
3. **Update constructor** to initialize `StateCore` with `layout.total_dim`
4. **Delegate common trait methods** to `self.core`
5. **Preserve all lag-related methods**

### Behavioral Invariant

**All numerical outputs must remain bit-for-bit identical.** This is a structural refactoring only.

---

## Acceptance Criteria

- [ ] `StorageAndInflowState` uses `StateCore` composition
- [ ] All common methods delegate to `core`
- [ ] `layout` and `dimension` (num_hydros) preserved
- [ ] All lag-related methods preserved and working
- [ ] All existing tests pass
- [ ] Golden tests pass
- [ ] No performance regression

---

## Implementation Guide

### Suggested Approach

1. **Update struct definition**:
   ```rust
   pub struct StorageAndInflowState {
       core: StateCore,
       dimension: usize,     // num_hydros for iteration
       layout: StateLayout,  // Per-hydro structure
   }
   ```

2. **Update constructor**:
   ```rust
   impl StorageAndInflowState {
       pub fn new(
           system: &system::System,
           uncertainty_models: &[TemporalModel],
       ) -> Self {
           let dimension = system.meta.hydros_count;
           
           let per_hydro_dims = per_hydro_state_dims(system, uncertainty_models, 0);
           
           let mut offsets = Vec::with_capacity(dimension + 1);
           offsets.push(0);
           let mut cumsum = 0;
           for &dim in &per_hydro_dims {
               cumsum += dim;
               offsets.push(cumsum);
           }
           
           let layout = StateLayout {
               per_hydro_dims,
               offsets,
               total_dim: cumsum,
           };
           
           Self {
               core: StateCore::with_capacity(cumsum),  // Or StateCore::new(cumsum)
               dimension,
               layout,
           }
       }
   }
   ```

3. **Delegate common methods**:
   ```rust
   impl State for StorageAndInflowState {
       fn coefficients(&self) -> &[f64] {
           self.core.coefficients()
       }
       
       fn dimension(&self) -> usize {
           self.core.dimension  // This is layout.total_dim
       }
       
       // ... other delegations
   }
   ```

4. **Preserve type-specific methods**:
   - `get_lag_order()` - uses `self.layout`
   - `get_total_dimension()` - uses `self.layout.total_dim`
   - `has_lagged_observation_state()` - returns `true`
   - `get_lagged_observations()` - accesses `self.core.state_coefficients`
   - `extract_storage_from_trajectory()` - uses `self.layout`
   - `extract_lags_from_trajectory()` - complex lag extraction
   - `evaluate_cut()` - StorageAndInflowState-specific structure

### Key Dimension Distinction

```rust
// IMPORTANT: Two different "dimensions"
self.dimension      // = num_hydros (for iteration in evaluate_cut)
self.core.dimension // = layout.total_dim (total state coefficients)

// In trait impl:
fn dimension(&self) -> usize {
    self.layout.total_dim  // Return total state dimension
}
```

### Pitfalls to Avoid

- ⚠️ **Don't confuse the two dimensions** - num_hydros vs total_dim
- ⚠️ **Layout access patterns** - Some methods use `self.layout.hydro_slice()`
- ⚠️ **State coefficient access** - Now via `self.core.state_coefficients`

---

## Testing Requirements

### Unit Tests

- [ ] Constructor creates correct layout
- [ ] All trait methods work via delegation
- [ ] `get_lagged_observations()` returns correct slices
- [ ] `extract_storage_from_trajectory()` works correctly
- [ ] `evaluate_cut()` produces identical results

### Regression Tests

- [ ] All existing `StorageAndInflowState` tests pass
- [ ] AR(p) model tests still work

### Golden Tests

- [ ] `./scripts/golden-tests.sh verify` passes

---

## Documentation Requirements

- [ ] Update struct doc comments
- [ ] Document dimension vs core.dimension distinction
- [ ] Update evaluate_cut documentation

---

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: More complex than T-041 due to layout and dimension distinction

---

## Definition of Done

- [ ] `StorageAndInflowState` uses `StateCore` composition
- [ ] Dimension distinction correctly handled
- [ ] All lag methods preserved
- [ ] All tests pass
- [ ] Golden tests pass
- [ ] No performance regression
- [ ] Ready for T-043
