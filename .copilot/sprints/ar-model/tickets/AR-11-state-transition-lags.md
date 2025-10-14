# AR-11: State Transition with Lag Update

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 1 day  
**Priority**: P1 (High)  
**Assignee**: TBD

---

## Context

In the forward pass, when transitioning from stage t to t+1, we must update both:

# AR-11: State Transition with Lag Update

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 1 day  
**Priority**: P1 (High)  
**Assignee**: TBD

---

## Context

In AR models, the state at each stage must be updated to reflect the most recent inflow realization and the lag history for each resource. This requires shifting the lag vector and inserting the new inflow value, ensuring the correct order and memory safety. The transition logic must be robust, efficient, and compatible with both AR and independent noise models.

**Why this matters**: If lag updates are incorrect, the AR process will be broken, leading to invalid scenario generation, incorrect cut coefficients, and ultimately, unreliable optimization results.

---

## Objective

Implement robust state transition logic for `StorageWithInflowState`:

- Shifts lag inflows for each resource
- Inserts new inflow realization at the front of the lag vector
- Ensures correct order and memory safety
- Maintains compatibility with legacy `StorageState`
- Updates the cached coefficient vector for cut generation

---

## Acceptance Criteria

### Must Have

- [ ] `StorageWithInflowState::update_lags(resource, new_inflow)` shifts lag vector and inserts new value
- [ ] State transition logic in forward pass uses this method for all AR resources
- [ ] Unit tests for lag update logic (AR(1), AR(2), AR(p))
- [ ] Edge case: independent noise (no lag update needed)
- [ ] Updates cached coefficient vector after mutation

### Should Have

- [ ] Efficient implementation (in-place shift, no unnecessary allocations)
- [ ] Documentation for method and usage
- [ ] Example usage in doc comments

### Won't Have (Yet)

- Integration with subproblem/cut generation (AR-14)

---

## Implementation Tasks

### 1. Implement Lag Update Method (1 hour)

- In `src/state.rs`, implement `update_lags(resource: &str, new_inflow: f64)`
- Shift lag inflows left, insert new value at front
- Update cached coefficient vector

### 2. Integrate with Forward Pass (1 hour)

- In forward pass logic, call `update_lags` for each resource with AR model
- Ensure correct order of updates (all resources, all lags)

### 3. Unit Tests (1 hour)

- Test AR(1): single lag update
- Test AR(2): shift and insert
- Test AR(p): generic test for p > 2
- Test edge cases (empty lags, missing resource, zero lags)

### 4. Documentation (30 min)

- Rustdoc for method
- Example usage in doc comments

---

## Testing Requirements

### Unit Tests

- [ ] Lag update logic for AR(1), AR(2), AR(p)
- [ ] Edge cases: empty lags, missing resource, zero lags

### Integration Tests

- [ ] Used in forward pass (AR-15)

---

## Documentation Requirements

### Code Documentation

- [ ] Rustdoc for method
- [ ] Example usage
- [ ] Explanation of lag update logic

### User Documentation

- [ ] Update state transition documentation in user guide

---

## Files to Modify

- `src/state.rs`: Implement lag update method
- `src/forward_pass.rs`: Integrate with forward pass
- `tests/test_state.rs`: Add unit tests

---

## Dependencies

- AR-7 (StorageWithInflowState implementation)
- AR-8 (State trait refactoring)
- AR-10 (Pre-study nodes for lag initialization)

---

## Blocks

- AR-15 (Forward pass AR integration)
- AR-14 (Cut generation with extended state)

---

## Technical Notes

### Lag Update Logic

- For AR(p), lag vector is `[ξₜ₋₁, ξₜ₋₂, ..., ξₜ₋ₚ]`
- On new inflow ξₜ, shift left and insert ξₜ at front: `[ξₜ, ξₜ₋₁, ..., ξₜ₋₍ₚ₋₁₎]`
- Drop oldest lag

### Performance Considerations

- Use in-place shift to avoid allocations
- Update cached coefficient vector after mutation

### Edge Cases

- Resource with zero lags: skip update
- Missing resource: error or skip (documented behavior)

---

## Validation Checklist

- [ ] All unit tests passing
- [ ] `cargo clippy` clean
- [ ] `cargo fmt` applied
- [ ] Documentation complete

---

**Created**: 2025-10-10
**Last Updated**: 2025-10-10
**Previous Ticket**: AR-10 (Pre-study nodes for lag initialization)
**Next Ticket**: AR-12 (Subproblem AR constraints)
}

        // Get new inflow realization
        let new_inflow = inflow_realizations.get(&resource.name)
            .copied()
            .ok_or_else(|| StateError::MissingInflowRealization {
                resource: resource.name.clone(),
            })?;

        // Shift lags: [ξₜ₋₁, ξₜ₋₂, ..., ξₜ₋ₚ] → [ξₜ, ξₜ₋₁, ..., ξₜ₋₍ₚ₋₁₎]
        let mut new_lags = Vec::with_capacity(lag_order);
        new_lags.push(new_inflow); // Most recent

        // Copy older lags (drop the oldest)
        for i in 0..(lag_order - 1) {
            new_lags.push(current_lags[i]);
        }

        new_lag_inflows.push(new_lags);
    }

    Ok(new_lag_inflows)

}

````

### 2. Integration with AR Sampling (1 hour)
```rust
// In src/sddp/forward_pass.rs

/// Sample inflows using conditional AR sampling
pub fn sample_inflows(
    current_state: &dyn State,
    node: &Node,
    process_map: &HashMap<String, Arc<dyn StochasticProcess>>,
    system: &System,
    rng: &mut impl Rng,
) -> HashMap<String, f64> {
    let mut inflow_realizations = HashMap::new();

    for (r, resource) in system.resources.iter().enumerate() {
        let process = match process_map.get(&resource.name) {
            Some(p) => p,
            None => continue, // No uncertainty for this resource
        };

        let realization = if process.is_conditional() {
            // AR process: need lag state
            if let Some(extended_state) = current_state.as_any().downcast_ref::<StorageWithInflowState>() {
                let lag_state = extended_state.lags(r);
                process.sample_conditional(lag_state, rng)
            } else {
                log::warn!(
                    "Conditional process for '{}' but state has no lags, using marginal",
                    resource.name
                );
                sample_from_distribution(process.distribution(), rng)
            }
        } else {
            // Independent process: unconditional sampling
            process.sample(rng)
        };

        inflow_realizations.insert(resource.name.clone(), realization);
    }

    inflow_realizations
}
````

### 3. Add Error Types (15 min)

```rust
// In src/error.rs

#[error("Unknown state type (cannot downcast)")]
UnknownStateType,

#[error("Failed to downcast state to expected type")]
DowncastFailed,

#[error("Missing inflow realization for resource '{resource}'")]
MissingInflowRealization {
    resource: String,
},
```

### 4. Tests (3 hours)

```rust
#[test]
fn test_state_transition_storage_only() {
    let state = StorageState::new(vec![500.0, 300.0]);
    let decisions = create_test_decisions();
    let inflows = HashMap::from([
        ("res1".to_string(), 100.0),
        ("res2".to_string(), 80.0),
    ]);
    let system = create_test_system();

    let next_state = transition_state(&state, &decisions, &inflows, &system).unwrap();

    // Verify volumes updated correctly
    // (depends on decisions and mass balance)
}

#[test]
fn test_state_transition_with_lags() {
    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0, 115.0], vec![80.0]], // AR(2) and AR(1)
    ).unwrap();

    let decisions = create_test_decisions();
    let inflows = HashMap::from([
        ("res1".to_string(), 130.0), // New realization
        ("res2".to_string(), 85.0),
    ]);
    let system = create_test_system();

    let next_state = transition_state(&state, &decisions, &inflows, &system).unwrap();

    // Verify lags shifted correctly
    let extended = next_state.as_any().downcast_ref::<StorageWithInflowState>().unwrap();
    assert_eq!(extended.lag_inflow(0, 0), 130.0); // New
    assert_eq!(extended.lag_inflow(0, 1), 120.0); // Shifted from [0]
    // 115.0 dropped (was [1], now gone)

    assert_eq!(extended.lag_inflow(1, 0), 85.0); // New
    // 80.0 dropped
}

#[test]
fn test_lag_shift_preserves_order() {
    let state = StorageWithInflowState::new(
        vec![500.0],
        vec![vec![120.0, 115.0, 110.0]], // AR(3)
    ).unwrap();

    let decisions = create_test_decisions();
    let inflows = HashMap::from([("res1".to_string(), 130.0)]);
    let system = create_test_system();

    let next_state = transition_state(&state, &decisions, &inflows, &system).unwrap();
    let extended = next_state.as_any().downcast_ref::<StorageWithInflowState>().unwrap();

    // [120, 115, 110] → [130, 120, 115]
    assert_eq!(extended.lags(0), &[130.0, 120.0, 115.0]);
}

#[test]
fn test_sample_inflows_ar_conditional() {
    let state = StorageWithInflowState::new(
        vec![500.0],
        vec![vec![100.0]], // AR(1) with lag = 100
    ).unwrap();

    let ar_process = Arc::new(
        AutoRegressive::new(
            vec![0.7],
            Distribution::Normal { mean: 0.0, std_dev: 15.0 },
            "res1".to_string(),
        ).unwrap()
    ) as Arc<dyn StochasticProcess>;

    let process_map = HashMap::from([("res1".to_string(), ar_process)]);
    let node = create_test_node();
    let system = create_test_system();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let inflows = sample_inflows(&state, &node, &process_map, &system, &mut rng);

    // Should use conditional sampling: E[ξₜ | ξₜ₋₁ = 100] ≈ 0.7 * 100 = 70
    let realization = inflows.get("res1").unwrap();
    assert!((realization - 70.0).abs() < 50.0); // Tolerance for noise
}
```

---

## Documentation Requirements

### Code Documentation

- [ ] Rustdoc for transition functions
- [ ] Explain lag shift operation
- [ ] Document mass balance equation

### User Documentation

- [ ] Explain state transition concept
- [ ] Show lag propagation example

---

## Files to Modify

### Core Implementation

- `src/state.rs` or `src/sddp/forward_pass.rs`: Add transition logic
- `src/error.rs`: Add new error types

### Tests

- `tests/test_state.rs`: Add transition tests
- `tests/test_sddp_algorithm.rs`: Add AR transition integration test

---

## Dependencies

### Depends On

- AR-7 (StorageWithInflowState)
- AR-9 (AR stochastic process)
- AR-10 (Pre-study nodes)

### Blocks

- AR-15 (Forward pass AR integration)

---

## Technical Notes

### Lag Shift Operation

**Critical**: Maintain correct temporal order.

Before: `[ξₜ₋₁, ξₜ₋₂, ξₜ₋₃]` (index 0 = most recent)
After: `[ξₜ, ξₜ₋₁, ξₜ₋₂]` (oldest dropped)

**Implementation**: Insert at front, drop last.

### Mass Balance

Standard hydro equation:

```
v_{t+1} = v_t + inflow - generation - spillage
```

For AR models, `inflow` comes from AR sampling (not deterministic).

### Performance Considerations

- Lag shift: O(p) per resource
- Volume update: O(R)
- Total: O(R(1 + p)) per transition

---

## Success Metrics

- ✅ Lags shift correctly (oldest dropped, newest added)
- ✅ Volumes updated correctly (mass balance)
- ✅ Both state types handled
- ✅ AR conditional sampling integrated

---

**Created**: 2025-01-10  
**Last Updated**: 2025-01-10  
**Previous Ticket**: AR-10 (Pre-study nodes)  
**Next Ticket**: AR-12 (Subproblem AR constraints)
