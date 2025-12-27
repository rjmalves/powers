# [TICKET-010] Implement VisitedStatePool::preallocate()

> **Epic**: [Epic 2: Preallocated State Pool](../00-epic-overview.md)
> **Sprint**: [Sprint 1](./00-sprint-overview.md)
> **Dependencies**: [TICKET-008](./ticket-008-implement-storagestate-update.md), [TICKET-009](./ticket-009-implement-storageinflowstate-update.md)
> **Blocks**: [TICKET-011](./ticket-011-update-state-addition.md)

## Context

### Background

The state pool needs full preallocation similar to the cut pool. However, states use `Box<dyn State>` for dynamic dispatch, which complicates preallocation. We use a template-based approach where a sample state is cloned for all preallocated slots.

### Relation to Epic

Core implementation enabling zero state allocation during training.

### Current State

```rust
pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}

impl VisitedStatePool {
    pub fn with_capacity(num_states: usize) -> Self {
        Self {
            pool: Vec::with_capacity(num_states),  // Only reserves pointers!
        }
    }
}
```

## Files to Read Before Starting

- `src/state.rs` - VisitedStatePool (lines 247-291)
- `src/state.rs` - `clone_dyn()` method for Box<dyn State> cloning
- [TICKET-007](./ticket-007-add-state-trait-methods.md) - `reset_to_zero()` method

## Specification

### New Method

```rust
impl VisitedStatePool {
    /// Preallocate all states for the entire training run.
    ///
    /// Uses the template state to create preallocated instances with the same
    /// structure but zeroed coefficients. All states start as inactive.
    ///
    /// # Arguments
    /// * `num_iterations` - Number of training iterations
    /// * `num_forward_passes` - Forward passes per iteration
    /// * `template_state` - Template with correct dimension (will be cloned and reset)
    ///
    /// # Performance
    /// Allocates `num_iterations * num_forward_passes` states upfront.
    /// Each state has preallocated coefficient vector.
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &Box<dyn State>,
    ) -> Self {
        let total_states = num_iterations * num_forward_passes;
        
        let pool: Vec<Box<dyn State>> = (0..total_states)
            .map(|_| {
                let mut state = template_state.clone_dyn();
                state.reset_to_zero();
                state.set_iteration(0);
                state.set_forward_pass_idx(0);
                state.set_dominating_cut_id(0);
                state.set_dominating_objective(0.0);
                state
            })
            .collect();
        
        Self { pool }
    }
    
    /// Update state at the given slot index.
    /// No allocation - modifies preallocated state in place.
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

### Behavior

1. `preallocate()`:
   - Clone template state `total_states` times
   - Reset each clone to zero
   - Clear metadata (iteration, forward_pass_idx, dominating_*)

2. `update_state()`:
   - Access slot directly (no bounds check in release)
   - Update coefficients in place
   - Set metadata
   - Return mutable reference for further updates

## Acceptance Criteria

- [ ] All states preallocated with correct dimension
- [ ] States start with zeroed coefficients
- [ ] `update_state()` works correctly with slot access
- [ ] Works for both StorageState and StorageAndInflowState
- [ ] No allocation during `update_state()`
- [ ] All existing tests pass

## Implementation Guide

### Suggested Approach

1. Add `preallocate()` method to `impl VisitedStatePool`
2. Add `update_state()` method
3. Store `num_forward_passes` if needed (or pass slot directly)
4. Test with both state types

### Key Files to Modify

- `src/state.rs`: Add methods to `VisitedStatePool`

### Pitfalls to Avoid

- ⚠️ Use `clone_dyn()` not `clone()` for Box<dyn State>
- ⚠️ Remember to reset dominating_objective after clone
- ⚠️ Template state must have correct dimension before preallocation

## Testing Requirements

### Unit Tests

- [ ] Test preallocate creates correct number of states
- [ ] Test each state has correct dimension
- [ ] Test states start zeroed
- [ ] Test update_state modifies correctly
- [ ] Test with StorageState template
- [ ] Test with StorageAndInflowState template (heterogeneous AR)

```rust
#[test]
fn test_state_pool_preallocate_storage() {
    let system = create_test_system_with_hydros(3);
    let template: Box<dyn State> = Box::new(StorageState::new(&system));
    
    let pool = VisitedStatePool::preallocate(8, 16, &template);
    
    assert_eq!(pool.pool.len(), 128); // 8 * 16
    assert_eq!(pool.pool[0].dimension(), 3);
    assert_eq!(pool.pool[0].coefficients(), &[0.0, 0.0, 0.0]);
}
```

## Documentation Requirements

- [ ] Doc comments on both methods
- [ ] Note about template state requirement

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Dynamic dispatch adds complexity, needs thorough testing
