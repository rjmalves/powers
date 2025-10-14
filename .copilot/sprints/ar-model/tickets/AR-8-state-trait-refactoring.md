# AR-8: State Trait Refactoring

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 2 days  
**Priority**: P1 (High)  
**Assignee**: TBD

---

## Context

With `StorageWithInflowState` implemented (AR-7), we need to ensure the existing `State` trait and its usage throughout the codebase works seamlessly with extended states. This includes:

1. Cut evaluation with extended state
2. State factory updates
3. Type-safe downcasting where needed
4. Performance validation

**Why this matters**: The State trait is used throughout SDDP (cuts, FCF, subproblems). Any issues here cascade to the entire algorithm.

---

## Objective

Refactor State trait usage to support both StorageState and StorageWithInflowState seamlessly, ensuring all existing code works with extended states.

---

## Acceptance Criteria

### Must Have

- [ ] State trait works with both StorageState and StorageWithInflowState
- [ ] Cut evaluation handles extended state correctly
- [ ] FCF domination logic works with different state dimensions
- [ ] Type-safe downcasting mechanism for concrete types
- [ ] All existing tests pass with both state types

### Should Have

- [ ] Performance benchmarks (extended state vs storage-only)
- [ ] Helper functions for state type detection
- [ ] Clear documentation on when to use each state type

### Won't Have (Yet)

- State factory integration (AR-17)
- Automatic state type selection (done in factory)

---

## Implementation Tasks

### 1. Audit State Trait Usage (1 hour)

```bash
# Find all State trait usage
grep -r "Box<dyn State>" src/
grep -r "impl State" src/
grep -r "state.coefficients()" src/

# Key locations:
# - src/cut.rs: BendersCut evaluation
# - src/fcf.rs: FCF cut selection, domination
# - src/sddp/*.rs: Forward/backward pass
# - src/subproblem.rs: Variable creation
```

### 2. Add Type Detection Helpers (30 min)

```rust
// In src/state.rs

pub trait State: Send + Sync + std::fmt::Debug {
    // Existing methods...
    fn coefficients(&self) -> &[f64];
    fn dimension(&self) -> usize;
    fn clone_box(&self) -> Box<dyn State>;

    // New: Type identification
    fn as_any(&self) -> &dyn std::any::Any;

    /// Check if state has AR lag components
    fn has_lags(&self) -> bool {
        false // Default for StorageState
    }

    /// Get number of storage variables
    fn num_volumes(&self) -> usize {
        self.dimension() // Default assumes storage-only
    }

    /// Get number of lag variables
    fn num_lags(&self) -> usize {
        0 // Default for StorageState
    }
}

impl State for StorageState {
    // Existing implementation...

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // Use defaults for has_lags(), num_lags()

    fn num_volumes(&self) -> usize {
        self.volumes.len()
    }
}

impl State for StorageWithInflowState {
    // From AR-7...

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn has_lags(&self) -> bool {
        true
    }

    fn num_volumes(&self) -> usize {
        self.volumes.len()
    }

    fn num_lags(&self) -> usize {
        self.lag_inflows.iter().map(|l| l.len()).sum()
    }
}

// Helper for downcasting
pub fn downcast_state<T: State + 'static>(state: &dyn State) -> Option<&T> {
    state.as_any().downcast_ref::<T>()
}
```

### 3. Update Cut Evaluation (1 hour)

```rust
// In src/cut.rs

impl BendersCut {
    /// Evaluate cut at given state
    ///
    /// Returns: α + ⟨π, x⟩ where x = state.coefficients()
    pub fn evaluate(&self, state: &dyn State) -> f64 {
        let state_vec = state.coefficients();

        // Dimension check
        if state_vec.len() != self.coefficients.len() {
            log::warn!(
                "State dimension {} doesn't match cut dimension {}",
                state_vec.len(),
                self.coefficients.len()
            );
            // Fall back to partial evaluation (use available dimensions)
            let min_dim = state_vec.len().min(self.coefficients.len());
            return self.intercept
                + self.coefficients[..min_dim]
                    .iter()
                    .zip(state_vec[..min_dim].iter())
                    .map(|(c, s)| c * s)
                    .sum::<f64>();
        }

        // Standard evaluation: α + ⟨π, x⟩
        self.intercept
            + self.coefficients
                .iter()
                .zip(state_vec.iter())
                .map(|(c, s)| c * s)
                .sum::<f64>()
    }

    /// Check if cut is compatible with state
    pub fn is_compatible(&self, state: &dyn State) -> bool {
        self.coefficients.len() == state.dimension()
    }
}
```

### 4. Update FCF Domination Logic (1.5 hours)

```rust
// In src/fcf.rs

impl FutureCostFunction {
    /// Check if cut1 dominates cut2 over feasible region
    ///
    /// For extended states: must check domination in full state space
    pub fn dominates(&self, cut1: &BendersCut, cut2: &BendersCut, state_type: StateType) -> bool {
        // Dimension must match for comparison
        if cut1.coefficients.len() != cut2.coefficients.len() {
            return false; // Different dimensions, no domination
        }

        match state_type {
            StateType::Storage => {
                // Original logic: check over storage bounds
                self.dominates_storage(cut1, cut2)
            },
            StateType::StorageWithInflow => {
                // Extended logic: check over storage × lag bounds
                self.dominates_extended(cut1, cut2)
            },
        }
    }

    fn dominates_extended(&self, cut1: &BendersCut, cut2: &BendersCut) -> bool {
        // For now: conservative check (only if cut1 ≥ cut2 everywhere)
        // More sophisticated: sample state space

        // Check at extreme points of feasible region
        let num_volumes = self.num_resources;
        let num_lags = cut1.coefficients.len() - num_volumes;

        // Sample corner points (2^dim is expensive, use smart sampling)
        let samples = self.generate_sample_states(num_volumes, num_lags);

        for sample_state in samples {
            let val1 = cut1.evaluate(&*sample_state);
            let val2 = cut2.evaluate(&*sample_state);

            if val1 < val2 - 1e-6 {
                return false; // cut1 not everywhere ≥ cut2
            }
        }

        true // cut1 dominates cut2
    }

    fn generate_sample_states(&self, num_volumes: usize, num_lags: usize) -> Vec<Box<dyn State>> {
        // Sample at corners and center of feasible region
        // For now: simple uniform sampling

        let mut samples = Vec::new();

        // Center point
        samples.push(self.create_sample_state(num_volumes, num_lags, 0.5));

        // Corners (simplified: just min/max extremes)
        samples.push(self.create_sample_state(num_volumes, num_lags, 0.0));
        samples.push(self.create_sample_state(num_volumes, num_lags, 1.0));

        // TODO: More sophisticated sampling for high-dimensional states

        samples
    }

    fn create_sample_state(&self, num_volumes: usize, num_lags: usize, fraction: f64) -> Box<dyn State> {
        if num_lags == 0 {
            // Storage-only
            let volumes = self.volume_bounds.iter()
                .map(|(min, max)| min + fraction * (max - min))
                .collect();
            Box::new(StorageState::new(volumes))
        } else {
            // Extended state
            let volumes = self.volume_bounds.iter()
                .map(|(min, max)| min + fraction * (max - min))
                .collect();

            // For lags: use reasonable range based on volumes
            let lag_inflows = vec![vec![100.0; num_lags / num_volumes]; num_volumes];

            Box::new(StorageWithInflowState::new(volumes, lag_inflows).unwrap())
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum StateType {
    Storage,
    StorageWithInflow,
}
```

### 5. Update Tests (2 hours)

Add tests ensuring both state types work throughout the pipeline.

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_cut_eval_storage_state() {
    let cut = BendersCut {
        intercept: 100.0,
        coefficients: vec![0.5, 0.3],
    };

    let state = StorageState::new(vec![500.0, 300.0]);
    let value = cut.evaluate(&state);

    assert_relative_eq!(value, 100.0 + 0.5 * 500.0 + 0.3 * 300.0, epsilon = 1e-9);
}

#[test]
fn test_cut_eval_extended_state() {
    let cut = BendersCut {
        intercept: 100.0,
        coefficients: vec![0.5, 0.3, 0.1], // volume0, volume1, lag0
    };

    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0], vec![]],
    ).unwrap();

    let value = cut.evaluate(&state);

    assert_relative_eq!(
        value,
        100.0 + 0.5 * 500.0 + 0.3 * 300.0 + 0.1 * 120.0,
        epsilon = 1e-9
    );
}

#[test]
fn test_cut_dimension_mismatch_warning() {
    let cut = BendersCut {
        intercept: 100.0,
        coefficients: vec![0.5, 0.3], // Expects 2 dimensions
    };

    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0], vec![]],
    ).unwrap(); // Has 3 dimensions

    // Should warn but not panic (use partial evaluation)
    let value = cut.evaluate(&state);
    assert!(value.is_finite());
}

#[test]
fn test_state_type_detection() {
    let storage_state: Box<dyn State> = Box::new(StorageState::new(vec![500.0]));
    assert!(!storage_state.has_lags());
    assert_eq!(storage_state.num_lags(), 0);

    let extended_state: Box<dyn State> = Box::new(
        StorageWithInflowState::new(vec![500.0], vec![vec![120.0]]).unwrap()
    );
    assert!(extended_state.has_lags());
    assert_eq!(extended_state.num_lags(), 1);
}

#[test]
fn test_downcast_state() {
    let state: Box<dyn State> = Box::new(
        StorageWithInflowState::new(vec![500.0], vec![vec![120.0]]).unwrap()
    );

    // Can downcast to concrete type
    let concrete = downcast_state::<StorageWithInflowState>(&*state).unwrap();
    assert_eq!(concrete.volume(0), 500.0);
    assert_eq!(concrete.lag_inflow(0, 0), 120.0);

    // Cannot downcast to wrong type
    assert!(downcast_state::<StorageState>(&*state).is_none());
}
```

### Integration Tests

```rust
#[test]
fn test_fcf_with_extended_state() {
    // Create FCF with cuts for extended state
    let mut fcf = FutureCostFunction::new(2);

    let cut1 = BendersCut {
        intercept: 100.0,
        coefficients: vec![0.5, 0.3, 0.1],
    };

    let cut2 = BendersCut {
        intercept: 90.0,
        coefficients: vec![0.6, 0.2, 0.15],
    };

    fcf.add_cut(cut1);
    fcf.add_cut(cut2);

    // Evaluate at extended state
    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0], vec![]],
    ).unwrap();

    let value = fcf.evaluate(&state);
    assert!(value > 0.0);
}
```

---

## Documentation Requirements

### Code Documentation

- [ ] Document state type detection methods
- [ ] Explain domination logic for extended states
- [ ] Provide examples of both state types

### User Documentation

- [ ] Explain when each state type is used
- [ ] Document performance characteristics

---

## Files to Modify

### Core Implementation

- `src/state.rs`: Add type detection methods
- `src/cut.rs`: Update evaluation for extended states
- `src/fcf.rs`: Update domination logic

### Tests

- `tests/test_state.rs`: Add state type tests
- `tests/test_cut.rs`: Add extended state cut evaluation
- `tests/test_fcf.rs`: Add extended state FCF tests

---

## Dependencies

### Depends On

- AR-7 (StorageWithInflowState implementation)

### Blocks

- AR-12 (Subproblem AR constraints)
- AR-14 (Cut generation with extended state)
- AR-15 (Forward pass AR integration)

---

## Technical Notes

### Dimension Compatibility

**Key Insight**: Cuts from different state types are incompatible. We must track which state type generated each cut.

Options:

1. Store state type in BendersCut (chosen for simplicity)
2. Separate FCF instances per state type
3. Runtime dimension checking (error-prone)

### Domination Logic Complexity

For extended states, domination checking is exponential in dimension. Strategies:

1. **Sampling**: Check at representative points
2. **Conservative**: Only dominate if everywhere better
3. **Heuristic**: Use bounds on coefficients

**Implementation**: Start with sampling (good enough for p ≤ 3).

### Performance Considerations

- Cut evaluation: O(d) where d = dimension
- Extended state: d = R + Σpᵣ (typically 2-3× storage-only)
- Impact: ~2-3× slower cut evaluation (acceptable)

### Edge Cases

1. **Mixed cuts**: Some storage-only, some extended (handle via dimension check)
2. **Zero lags**: Extended state with all pᵣ=0 (degenerate to storage)
3. **Very high dimension**: d > 20 (domination checking expensive)

### Alternative Considered

**Separate traits**: StorageState and ExtendedState with different traits. Rejected for excessive code duplication.

---

## Validation Checklist

Before marking this ticket complete:

- [ ] State trait works with both types
- [ ] Cut evaluation correct for both types
- [ ] Dimension mismatch handled gracefully
- [ ] FCF domination logic works
- [ ] All tests passing
- [ ] No performance regression (< 5% overhead)
- [ ] `cargo clippy` clean
- [ ] Documentation complete

---

## Success Metrics

- ✅ Both state types work seamlessly throughout codebase
- ✅ Cut evaluation correct for extended states
- ✅ Type-safe downcasting works
- ✅ Performance overhead <5% for storage-only (backward compatibility)
- ✅ All existing tests pass unchanged

---

**Created**: 2025-01-10  
**Last Updated**: 2025-01-10  
**Previous Ticket**: AR-7 (StorageWithInflowState implementation)  
**Next Ticket**: AR-9 (AR stochastic process implementation)
