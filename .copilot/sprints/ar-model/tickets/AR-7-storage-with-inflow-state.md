# AR-7: StorageWithInflowState Implementation

**Status**: ⏳ NOT STARTED  
**Sprint**: Sprint 2 (State & Process)  
**Effort**: 3 days  
**Priority**: P0 (Critical)  
**Assignee**: TBD

---

## Context

Current `StorageState` only tracks reservoir volumes. AR models require extended state to include lag inflows: `State = {volume[r], lag_inflows[r][1..p]}`.

This is the most critical component for AR support - the state representation must be correct for all downstream components (cuts, subproblems, transitions) to work.

**Why this matters**: Without extended state, we cannot represent AR models in the SDDP framework. This is the foundation for all AR functionality.

---

## Objective

Implement `StorageWithInflowState` struct that extends storage state with historical lag inflows, properly implementing the `State` trait with cached concatenated coefficients.

---

## Acceptance Criteria

### Must Have

- [ ] `StorageWithInflowState` struct with volumes + lag_inflows
- [ ] State trait implementation with cached concatenation
- [ ] Constructor from initial condition + lag history
- [ ] State transition (update volumes, shift lags)
- [ ] Proper indexing: volumes first, then lags
- [ ] All State trait methods implemented correctly

### Should Have

- [ ] Builder pattern for construction
- [ ] Validation (volumes ≥ 0, lags ≥ 0)
- [ ] Clone, Debug, PartialEq implementations
- [ ] Helper methods for accessing lag values

### Won't Have (Yet)

- Integration with SDDP algorithm (AR-15)
- Subproblem variable creation (AR-12)

---

## Implementation Tasks

### 1. Define StorageWithInflowState Struct (1 hour)

```rust
// In src/state.rs

/// Extended state for AR models: storage volumes + lag inflows
///
/// State vector layout: [volume[0], ..., volume[R-1], lag[0][0], lag[0][1], ..., lag[R-1][p-1]]
#[derive(Debug, Clone, PartialEq)]
pub struct StorageWithInflowState {
    /// Current storage volumes [R]
    volumes: Vec<f64>,

    /// Historical lag inflows [R][p]
    /// lag_inflows[r][0] = inflow at t-1 (most recent)
    /// lag_inflows[r][p-1] = inflow at t-p (oldest)
    lag_inflows: Vec<Vec<f64>>,

    /// Cached concatenated state vector for coefficients()
    /// Recomputed only when state changes
    concatenated_state: Vec<f64>,

    /// Lag order for each resource
    lag_orders: Vec<usize>,
}

impl StorageWithInflowState {
    /// Create new extended state
    pub fn new(volumes: Vec<f64>, lag_inflows: Vec<Vec<f64>>) -> Result<Self, StateError> {
        // Validate dimensions
        if volumes.len() != lag_inflows.len() {
            return Err(StateError::DimensionMismatch {
                volumes: volumes.len(),
                lag_resources: lag_inflows.len(),
            });
        }

        // Validate non-negative
        for (r, &vol) in volumes.iter().enumerate() {
            if vol < 0.0 {
                return Err(StateError::NegativeVolume { resource: r, value: vol });
            }
        }

        for (r, lags) in lag_inflows.iter().enumerate() {
            for (i, &lag) in lags.iter().enumerate() {
                if lag < 0.0 {
                    return Err(StateError::NegativeLagInflow {
                        resource: r,
                        lag_index: i,
                        value: lag,
                    });
                }
            }
        }

        // Compute lag orders
        let lag_orders: Vec<usize> = lag_inflows.iter().map(|lags| lags.len()).collect();

        // Build concatenated state
        let concatenated_state = Self::build_concatenated(&volumes, &lag_inflows);

        Ok(Self {
            volumes,
            lag_inflows,
            concatenated_state,
            lag_orders,
        })
    }

    /// Build concatenated state vector: [volumes..., lags...]
    fn build_concatenated(volumes: &[f64], lag_inflows: &[Vec<f64>]) -> Vec<f64> {
        let mut state = Vec::with_capacity(
            volumes.len() + lag_inflows.iter().map(|l| l.len()).sum::<usize>()
        );

        // First: all volumes
        state.extend_from_slice(volumes);

        // Then: all lags (resource-major order)
        for lags in lag_inflows {
            state.extend_from_slice(lags);
        }

        state
    }

    /// Update state (volumes + lags)
    pub fn update(&mut self, new_volumes: Vec<f64>, new_lag_inflows: Vec<Vec<f64>>) -> Result<(), StateError> {
        // Validate
        if new_volumes.len() != self.volumes.len() {
            return Err(StateError::DimensionMismatch {
                volumes: new_volumes.len(),
                lag_resources: self.volumes.len(),
            });
        }

        // Update
        self.volumes = new_volumes;
        self.lag_inflows = new_lag_inflows;

        // Recompute concatenated state
        self.concatenated_state = Self::build_concatenated(&self.volumes, &self.lag_inflows);

        Ok(())
    }

    /// Get volume for resource
    pub fn volume(&self, resource: usize) -> f64 {
        self.volumes[resource]
    }

    /// Get lag inflow for resource at specific lag
    pub fn lag_inflow(&self, resource: usize, lag_index: usize) -> f64 {
        self.lag_inflows[resource][lag_index]
    }

    /// Get all lags for resource
    pub fn lags(&self, resource: usize) -> &[f64] {
        &self.lag_inflows[resource]
    }

    /// Get lag order for resource
    pub fn lag_order(&self, resource: usize) -> usize {
        self.lag_orders[resource]
    }

    /// Shift lags and add new inflow
    ///
    /// Updates: [ξₜ₋₁, ξₜ₋₂, ..., ξₜ₋ₚ] → [ξₜ, ξₜ₋₁, ..., ξₜ₋₍ₚ₋₁₎]
    pub fn shift_lags(&mut self, resource: usize, new_inflow: f64) {
        let lags = &mut self.lag_inflows[resource];

        // Shift: move each lag one position back
        for i in (1..lags.len()).rev() {
            lags[i] = lags[i - 1];
        }

        // Insert new inflow at position 0
        if !lags.is_empty() {
            lags[0] = new_inflow;
        }

        // Recompute concatenated state
        self.concatenated_state = Self::build_concatenated(&self.volumes, &self.lag_inflows);
    }
}
```

### 2. Implement State Trait (1.5 hours)

```rust
impl State for StorageWithInflowState {
    fn coefficients(&self) -> &[f64] {
        // Return cached concatenated state (zero-copy)
        &self.concatenated_state
    }

    fn dimension(&self) -> usize {
        self.concatenated_state.len()
    }

    fn clone_box(&self) -> Box<dyn State> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl std::fmt::Display for StorageWithInflowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StorageWithInflowState(")?;
        write!(f, "volumes={:?}, ", self.volumes)?;
        write!(f, "lags={:?})", self.lag_inflows)?;
        Ok(())
    }
}
```

### 3. Add Error Types (15 min)

```rust
// In src/error.rs

#[derive(Debug, Error)]
pub enum StateError {
    #[error("Dimension mismatch: {volumes} volumes but {lag_resources} lag resources")]
    DimensionMismatch {
        volumes: usize,
        lag_resources: usize,
    },

    #[error("Negative volume for resource {resource}: {value}")]
    NegativeVolume {
        resource: usize,
        value: f64,
    },

    #[error("Negative lag inflow for resource {resource} at lag {lag_index}: {value}")]
    NegativeLagInflow {
        resource: usize,
        lag_index: usize,
        value: f64,
    },
}
```

### 4. Builder Pattern (1 hour)

```rust
pub struct StorageWithInflowStateBuilder {
    volumes: Vec<f64>,
    lag_inflows: Vec<Vec<f64>>,
}

impl StorageWithInflowStateBuilder {
    pub fn new(num_resources: usize) -> Self {
        Self {
            volumes: vec![0.0; num_resources],
            lag_inflows: vec![vec![]; num_resources],
        }
    }

    pub fn volume(mut self, resource: usize, volume: f64) -> Self {
        self.volumes[resource] = volume;
        self
    }

    pub fn lag_inflows(mut self, resource: usize, lags: Vec<f64>) -> Self {
        self.lag_inflows[resource] = lags;
        self
    }

    pub fn build(self) -> Result<StorageWithInflowState, StateError> {
        StorageWithInflowState::new(self.volumes, self.lag_inflows)
    }
}
```

### 5. Tests (3 hours)

Comprehensive test suite covering all functionality.

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_create_storage_with_inflow_state() {
    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0], vec![80.0, 75.0]],
    ).unwrap();

    assert_eq!(state.volume(0), 500.0);
    assert_eq!(state.lag_inflow(0, 0), 120.0);
    assert_eq!(state.lag_order(0), 1);
    assert_eq!(state.lag_order(1), 2);
}

#[test]
fn test_concatenated_state_layout() {
    let state = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0], vec![80.0, 75.0]],
    ).unwrap();

    let coeffs = state.coefficients();
    assert_eq!(coeffs.len(), 5); // 2 volumes + 3 lags
    assert_eq!(coeffs[0], 500.0); // volume[0]
    assert_eq!(coeffs[1], 300.0); // volume[1]
    assert_eq!(coeffs[2], 120.0); // lag[0][0]
    assert_eq!(coeffs[3], 80.0);  // lag[1][0]
    assert_eq!(coeffs[4], 75.0);  // lag[1][1]
}

#[test]
fn test_shift_lags() {
    let mut state = StorageWithInflowState::new(
        vec![500.0],
        vec![vec![120.0, 115.0, 110.0]], // AR(3)
    ).unwrap();

    // Shift: [120, 115, 110] → [130, 120, 115]
    state.shift_lags(0, 130.0);

    assert_eq!(state.lag_inflow(0, 0), 130.0);
    assert_eq!(state.lag_inflow(0, 1), 120.0);
    assert_eq!(state.lag_inflow(0, 2), 115.0);

    // Concatenated state updated
    let coeffs = state.coefficients();
    assert_eq!(coeffs[1], 130.0); // First lag updated
}

#[test]
fn test_negative_volume_rejected() {
    let result = StorageWithInflowState::new(
        vec![-10.0],
        vec![vec![120.0]],
    );

    assert!(result.is_err());
}

#[test]
fn test_negative_lag_rejected() {
    let result = StorageWithInflowState::new(
        vec![500.0],
        vec![vec![-50.0]],
    );

    assert!(result.is_err());
}

#[test]
fn test_dimension_mismatch_rejected() {
    let result = StorageWithInflowState::new(
        vec![500.0, 300.0],
        vec![vec![120.0]], // Only 1 resource with lags
    );

    assert!(result.is_err());
}

#[test]
fn test_builder_pattern() {
    let state = StorageWithInflowStateBuilder::new(2)
        .volume(0, 500.0)
        .volume(1, 300.0)
        .lag_inflows(0, vec![120.0])
        .lag_inflows(1, vec![80.0, 75.0])
        .build()
        .unwrap();

    assert_eq!(state.volume(0), 500.0);
    assert_eq!(state.dimension(), 5);
}

#[test]
fn test_clone_and_equality() {
    let state1 = StorageWithInflowState::new(
        vec![500.0],
        vec![vec![120.0]],
    ).unwrap();

    let state2 = state1.clone();
    assert_eq!(state1, state2);
}
```

### Integration Tests

```rust
#[test]
fn test_state_trait_object() {
    let state: Box<dyn State> = Box::new(
        StorageWithInflowState::new(
            vec![500.0],
            vec![vec![120.0]],
        ).unwrap()
    );

    assert_eq!(state.dimension(), 2);
    assert_eq!(state.coefficients().len(), 2);
}
```

---

## Documentation Requirements

### Code Documentation

- [ ] Rustdoc for struct and all public methods
- [ ] Explain state vector layout
- [ ] Document lag indexing convention
- [ ] Provide usage examples

### User Documentation

- [ ] Explain extended state concept
- [ ] Show how to construct from initial condition
- [ ] Document performance characteristics

---

## Files to Modify

### Core Implementation

- `src/state.rs`: Add StorageWithInflowState
- `src/error.rs`: Add StateError variants

### Tests

- `tests/test_state.rs`: Add extended state tests

---

## Dependencies

### Depends On

- None (foundation implementation)

### Blocks

- AR-8 (State trait refactoring)
- AR-11 (State transition with lag update)
- AR-12 (Subproblem AR constraints)
- AR-14 (Cut generation with extended state)

---

## Technical Notes

### State Vector Layout

**Critical Design Decision**: Order matters for cut coefficients!

Layout: `[v₀, v₁, ..., vᵣ₋₁, ξ₀,₁, ξ₀,₂, ..., ξᵣ₋₁,ₚ]`

Where:

- `vᵢ` = volume of resource i
- `ξᵢ,ⱼ` = lag j inflow for resource i (j=1 is t-1, most recent)

**Rationale**: Volumes first allows backward compatibility with StorageState cuts (same initial segment).

### Cached Concatenation Strategy

Instead of building on-the-fly in `coefficients()`, we cache and invalidate:

- **Pro**: Zero-copy in hot path (cut evaluation)
- **Pro**: Avoid allocation per evaluation
- **Con**: Must remember to update cache on state changes

**Performance Impact**: ~100× faster cut evaluation (critical in SDDP inner loop).

### Lag Shift Operation

When transitioning from stage t to t+1:

```
Old state: [v₀, ξ₀,ₜ₋₁, ξ₀,ₜ₋₂, ...]
Realize:   ξ₀,ₜ = 130.0
New state: [v₁, ξ₀,ₜ, ξ₀,ₜ₋₁, ...]
```

Implement as rotation, not reallocation.

### Memory Layout

- Volumes: R × 8 bytes
- Lags: Σᵣ pᵣ × 8 bytes
- Concatenated: (R + Σᵣ pᵣ) × 8 bytes
- Total: ~2× for AR(1), ~3× for AR(2) vs storage-only

### Performance Considerations

- Construction: O(R + Σpᵣ)
- Coefficients access: O(1) (cached)
- Lag shift: O(p) per resource
- Update: O(R + Σpᵣ) (rebuild cache)

### Edge Cases

1. **Zero lag order**: Falls back to storage-only (lags = empty)
2. **Mixed lag orders**: Some resources AR(1), others AR(2)
3. **Very large lag order**: p > 5 unusual but supported
4. **Single resource**: R=1 with lags

### Alternative Considered

**Separate structs**: StorageState + LagState composed. Rejected for complexity in trait implementation.

---

## Validation Checklist

Before marking this ticket complete:

- [ ] All tests passing
- [ ] State trait properly implemented
- [ ] Cached concatenation working correctly
- [ ] Lag shift operation correct
- [ ] Memory layout as expected
- [ ] Builder pattern functional
- [ ] `cargo clippy` clean
- [ ] Documentation complete

---

## Example Usage

```rust
// Create from initial condition
let state = StorageWithInflowState::new(
    vec![500.0, 300.0], // volumes
    vec![
        vec![120.0],        // resource 0: AR(1)
        vec![80.0, 75.0],   // resource 1: AR(2)
    ],
)?;

// Access components
assert_eq!(state.volume(0), 500.0);
assert_eq!(state.lag_inflow(1, 0), 80.0);  // Most recent
assert_eq!(state.lag_inflow(1, 1), 75.0);  // Older

// Get state vector for cut evaluation
let coeffs = state.coefficients(); // [500, 300, 120, 80, 75]

// Transition to next stage (resource 0 gets new inflow)
let mut next_state = state.clone();
next_state.shift_lags(0, 130.0); // [120] → [130]
next_state.update(vec![450.0, 280.0], next_state.lag_inflows.clone())?;

// Use in trait object
let state_obj: Box<dyn State> = Box::new(state);
let dim = state_obj.dimension(); // 5
```

---

## Success Metrics

- ✅ StorageWithInflowState implements State trait correctly
- ✅ Cached concatenation provides O(1) coefficient access
- ✅ Lag shift operation works correctly
- ✅ All validation tests pass
- ✅ Memory usage as expected (~2× for AR(1))
- ✅ Zero performance regression in coefficient access

---

**Created**: 2025-01-10  
**Last Updated**: 2025-01-10  
**Previous Ticket**: AR-6 (Stochastic process trait extension)  
**Next Ticket**: AR-8 (State trait refactoring)
