# Epic 2: Preallocated State Pool

## Status: ✅ COMPLETE

## Summary

Implement full preallocation of `State` instances (visited states) at training initialization. This mirrors Epic 1's approach but for the state pool, using the same slot-based access pattern with `(iteration, forward_pass_idx)`.

## Scope

### Included

- Add `update_coefficients()` and `reset_to_zero()` methods to `State` trait
- Implement these methods for `StorageState` and `StorageAndInflowState`
- `VisitedStatePool::preallocate()` method for full preallocation
- Modified state addition to use slot-based access
- State pool integration with `add_cuts_batch()`

### Excluded

- Cut pool changes (Epic 1)
- Arc-based sharing (Epic 3)
- New state implementations

## Dependencies

- **Requires**: Epic 1 (establishes slot computation pattern)
- **Enables**: Epic 4 (validation)

## Acceptance Criteria

- [x] All states preallocated at training start
- [x] Zero `Box<dyn State>` allocations during training
- [x] State trait extended with in-place update methods
- [x] Both `StorageState` and `StorageAndInflowState` implementations complete
- [x] All existing lib tests pass
- [ ] Lower bounds identical to baseline (needs validation)

## Technical Approach

### Trait Extension

```rust
pub trait State: Send + Sync {
    // Existing methods...
    
    /// Update coefficient values in place (no allocation).
    fn update_coefficients(&mut self, coefficients: &[f64]);
    
    /// Reset coefficients to zero while preserving capacity.
    fn reset_to_zero(&mut self);
    
    /// Get mutable reference to internal coefficients for direct update.
    fn coefficients_mut(&mut self) -> &mut [f64];
}
```

### State Pool Preallocation Challenge

Unlike cuts, states use `Box<dyn State>` for dynamic dispatch. Preallocation requires:

1. **Template-based approach**: Use a template state to create preallocated instances
2. **Concrete type knowledge**: Know whether `StorageState` or `StorageAndInflowState` at preallocation time

```rust
impl VisitedStatePool {
    pub fn preallocate<S: State + Clone + 'static>(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &S,
    ) -> Self {
        let total_states = num_iterations * num_forward_passes;
        
        let pool: Vec<Box<dyn State>> = (0..total_states)
            .map(|_| {
                let mut state = template_state.clone();
                state.reset_to_zero();
                Box::new(state) as Box<dyn State>
            })
            .collect();
        
        Self { pool }
    }
}
```

### Slot-Based State Update

```rust
impl VisitedStatePool {
    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut Box<dyn State> {
        let state = &mut self.pool[slot];
        state.update_coefficients(coefficients);
        state.set_iteration(iteration);
        state.set_forward_pass_idx(forward_pass_idx);
        state
    }
}
```

## Estimated Effort

**1 Sprint (2 weeks)** / **18 story points**

## Files to Modify

| File | Changes |
|------|---------|
| `src/state.rs` | Add trait methods, implement for both state types, add `preallocate()` |
| `src/fcf.rs` | Update state addition to use slot-based access |
| `src/sddp/mod.rs` | Modify state pool initialization |
