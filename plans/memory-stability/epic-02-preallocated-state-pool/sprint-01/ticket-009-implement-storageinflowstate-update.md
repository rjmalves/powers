# [TICKET-009] Implement update methods for StorageAndInflowState

> **Epic**: [Epic 2: Preallocated State Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-007](./ticket-007-add-state-trait-methods.md)
> **Blocks**: [TICKET-010](./ticket-010-implement-statepool-preallocate.md)

## Context

### Background

`StorageAndInflowState` is more complex with heterogeneous AR orders per hydro. The coefficient vector has a layout defined by `StateLayout`. Update methods must respect this structure.

### Relation to Epic

Completes the in-place update implementation for all state types.

### Current State

`StorageAndInflowState` has:
- `state_coefficients: Vec<f64>` - Flattened state vector
- `layout: StateLayout` - Offsets and dimensions per hydro

## Files to Read Before Starting

- `src/state.rs` - `StorageAndInflowState` implementation (lines 721-1165)
- `src/state.rs` - `StateLayout` structure (lines 375-486)
- [TICKET-007](./ticket-007-add-state-trait-methods.md) - Trait method signatures

## Specification

### Implementation

```rust
impl State for StorageAndInflowState {
    fn update_coefficients(&mut self, coefficients: &[f64]) {
        debug_assert_eq!(
            self.state_coefficients.len(),
            coefficients.len(),
            "Coefficient dimension mismatch: expected {}, got {}",
            self.state_coefficients.len(),
            coefficients.len()
        );
        self.state_coefficients.copy_from_slice(coefficients);
    }
    
    fn reset_to_zero(&mut self) {
        self.state_coefficients.fill(0.0);
    }
    
    fn dimension(&self) -> usize {
        self.layout.total_dim
    }
}
```

### Behavior

- `update_coefficients()`: Direct copy of flattened vector, O(total_dim)
- `reset_to_zero()`: Fill with zeros, O(total_dim)
- `dimension()`: Returns `layout.total_dim` (sum of all per-hydro dimensions)

### Complexity Note

The coefficient layout is:
```
[storage₀, lag₀₁, lag₀₂, storage₁, lag₁₁, storage₂]
 ←─ hydro 0 ──→  ←── hydro 1 ──→  ← hydro 2 →
```

The update method copies the entire flattened vector, so it doesn't need to know the per-hydro layout.

## Acceptance Criteria

- [ ] All three methods implemented correctly
- [ ] Handles heterogeneous AR orders correctly
- [ ] No memory allocation during update
- [ ] Unit tests with various AR order configurations
- [ ] Existing tests still pass

## Implementation Guide

### Suggested Approach

1. Add implementations to `impl State for StorageAndInflowState`
2. Use `layout.total_dim` for dimension
3. Add tests with heterogeneous AR orders (AR(0), AR(1), AR(2))

### Key Files to Modify

- `src/state.rs`: Implement methods in `impl State for StorageAndInflowState`

### Pitfalls to Avoid

- ⚠️ Don't iterate per-hydro for coefficient copy (just copy entire Vec)
- ⚠️ Ensure `dimension()` uses `layout.total_dim` not `self.dimension` (which is num_hydros)

## Testing Requirements

### Unit Tests

- [ ] Test with homogeneous AR orders (all AR(1))
- [ ] Test with heterogeneous AR orders (AR(0), AR(1), AR(2))
- [ ] Test dimension matches layout.total_dim
- [ ] Test coefficients correctly copied including lags

```rust
#[test]
fn test_storage_inflow_state_update_coefficients_heterogeneous() {
    let system = create_test_system_with_hydros(3);
    let temporal_models = vec![
        create_independent_model(0),        // AR(0) - 1 coef
        create_par_model_uniform_sigma(1, vec![0.5]),     // AR(1) - 2 coefs
        create_par_model_uniform_sigma(2, vec![0.5, 0.3]), // AR(2) - 3 coefs
    ];
    
    let mut state = StorageAndInflowState::new(&system, &temporal_models);
    
    // Total dimension: 1 + 2 + 3 = 6
    assert_eq!(state.dimension(), 6);
    
    let new_coeffs = [1.0, 2.0, 2.1, 3.0, 3.1, 3.2];
    state.update_coefficients(&new_coeffs);
    
    assert_eq!(state.coefficients(), &new_coeffs);
}
```

## Documentation Requirements

- [ ] Doc comments noting layout preservation

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: More complex structure, needs thorough testing with heterogeneous orders
