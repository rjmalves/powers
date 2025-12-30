# [T-076] Refactor VisitedStatePool to Shared Layout

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 4: Pool Architecture Refinement](./00-sprint-overview.md)
> **Dependencies**: [T-075](./ticket-075-create-state-data.md)
> **Blocks**: [T-077](./ticket-077-update-fcf-state-access.md)

## Files to Read Before Starting

- `src/state.rs:871-1020` - Current `VisitedStatePool` implementation
- `src/state.rs:789-869` - `StateConfig` enum
- `src/state.rs:1100-1200` - `StateLayout` struct
- `src/fcf.rs:52-70` - FCF state pool usage
- `src/fcf.rs:180-220` - FCF domination evaluation

---

## Context

### Background

The pool currently stores `Vec<ConcreteState>`, where each `StorageAndInflow` variant contains a full copy of `StateLayout`. This wastes memory and pollutes the cache with duplicated metadata.

### Current State

```rust
pub struct VisitedStatePool {
    pub pool: Vec<ConcreteState>,  // Each state has its own layout copy
}
```

### Target State

```rust
pub struct VisitedStatePool {
    /// Pure state data (no metadata)
    pub pool: Vec<StateData>,
    
    /// Shared layout (stored once, not per state)
    pub layout: Option<StateLayout>,
    
    /// State type identifier
    pub state_type: StateTypeId,
    
    /// Number of hydros
    pub num_hydros: usize,
}
```

---

## Specification

### Updated VisitedStatePool

```rust
/// Pool of visited states with shared metadata.
///
/// # Architecture (Epic 5 - T-076)
///
/// Separates state data from layout metadata:
/// - `pool`: Pure coefficient data (`Vec<StateData>`)
/// - `layout`: Shared layout stored once (not per state)
///
/// This eliminates ~160 bytes per state of duplicated metadata.
pub struct VisitedStatePool {
    /// State coefficient data
    pub pool: Vec<StateData>,
    
    /// Shared layout for StorageAndInflow states (None for Storage)
    pub layout: Option<StateLayout>,
    
    /// State type for all states in this pool
    pub state_type: StateTypeId,
    
    /// Number of hydros
    pub num_hydros: usize,
}
```

### Updated Methods

```rust
impl VisitedStatePool {
    pub fn with_capacity(num_states: usize) -> Self;
    
    /// Preallocate using legacy template (backward compat)
    pub fn preallocate(
        num_iterations: usize,
        num_forward_passes: usize,
        template_state: &dyn State,
    ) -> Self;
    
    /// Preallocate using StateConfig (preferred)
    pub fn preallocate_concrete(
        num_iterations: usize,
        num_forward_passes: usize,
        config: &StateConfig,
    ) -> Self;
    
    pub fn is_preallocated(&self) -> bool;
    
    /// Update state at slot - returns &mut StateData
    pub fn update_state(
        &mut self,
        slot: usize,
        coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
    ) -> &mut StateData;
    
    /// Get shared layout (for StorageAndInflow operations)
    pub fn get_layout(&self) -> Option<&StateLayout>;
    
    /// Check if this is a StorageAndInflow pool
    pub fn has_layout(&self) -> bool;
}
```

---

## Acceptance Criteria

- [x] `VisitedStatePool` uses `Vec<StateData>` internally
- [x] Layout stored once in pool, not per state
- [x] All existing tests pass without modification
- [x] New tests for shared layout functionality
- [x] Memory usage reduced (verified by size comparison)

---

## Implementation Guide

### Step 1: Update struct definition

```rust
pub struct VisitedStatePool {
    /// State coefficient data (no layout duplication)
    pub pool: Vec<StateData>,
    
    /// Shared layout for all states (None for Storage type)
    pub layout: Option<StateLayout>,
    
    /// State type identifier
    pub state_type: StateTypeId,
    
    /// Number of hydros
    pub num_hydros: usize,
}
```

### Step 2: Update with_capacity

```rust
pub fn with_capacity(num_states: usize) -> Self {
    Self {
        pool: Vec::with_capacity(num_states),
        layout: None,
        state_type: StateTypeId::Storage,
        num_hydros: 0,
    }
}
```

### Step 3: Update preallocate_concrete

```rust
pub fn preallocate_concrete(
    num_iterations: usize,
    num_forward_passes: usize,
    config: &StateConfig,
) -> Self {
    let total_states = num_iterations * num_forward_passes;
    
    match config {
        StateConfig::Storage { num_hydros } => {
            let pool = (0..total_states)
                .map(|_| StateData::new(*num_hydros))
                .collect();
            
            Self {
                pool,
                layout: None,
                state_type: StateTypeId::Storage,
                num_hydros: *num_hydros,
            }
        }
        StateConfig::StorageAndInflow { num_hydros, per_hydro_state_dims } => {
            // Build layout ONCE
            let mut offsets = Vec::with_capacity(num_hydros + 1);
            offsets.push(0);
            let mut cumsum = 0;
            for &dim in per_hydro_state_dims {
                cumsum += dim;
                offsets.push(cumsum);
            }
            
            let layout = StateLayout {
                per_hydro_dims: per_hydro_state_dims.clone(),
                offsets,
                total_dim: cumsum,
            };
            
            let pool = (0..total_states)
                .map(|_| StateData::new(cumsum))
                .collect();
            
            Self {
                pool,
                layout: Some(layout),
                state_type: StateTypeId::StorageAndInflow,
                num_hydros: *num_hydros,
            }
        }
    }
}
```

### Step 4: Update preallocate (legacy wrapper)

```rust
pub fn preallocate(
    num_iterations: usize,
    num_forward_passes: usize,
    template_state: &dyn State,
) -> Self {
    let config = StateConfig::from_dyn(template_state);
    Self::preallocate_concrete(num_iterations, num_forward_passes, &config)
}
```

### Step 5: Update is_preallocated

```rust
#[inline]
pub fn is_preallocated(&self) -> bool {
    !self.pool.is_empty()
        && self.pool[0].get_iteration() == 0
        && self.pool[0].get_forward_pass_idx() == 0
}
```

### Step 6: Update update_state

```rust
pub fn update_state(
    &mut self,
    slot: usize,
    coefficients: &[f64],
    iteration: usize,
    forward_pass_idx: usize,
) -> &mut StateData {
    let state = &mut self.pool[slot];
    state.update_coefficients(coefficients);
    state.set_iteration(iteration);
    state.set_forward_pass_idx(forward_pass_idx);
    state
}
```

### Step 7: Add layout accessors

```rust
/// Get the shared layout (for StorageAndInflow states)
#[inline]
pub fn get_layout(&self) -> Option<&StateLayout> {
    self.layout.as_ref()
}

/// Check if this pool has a layout (StorageAndInflow type)
#[inline]
pub fn has_layout(&self) -> bool {
    self.layout.is_some()
}

/// Get state type
#[inline]
pub fn state_type(&self) -> StateTypeId {
    self.state_type
}

/// Get number of hydros
#[inline]
pub fn num_hydros(&self) -> usize {
    self.num_hydros
}
```

---

## Callsite Updates Required

The pool is accessed in these patterns that need updating:

### Pattern 1: Direct index access

```rust
// Before: pool.pool[slot] returns ConcreteState
// After:  pool.pool[slot] returns StateData

// The method names are the same, so most code works:
let coeffs = pool.pool[slot].coefficients();  // ✅ Works
pool.pool[slot].set_iteration(5);             // ✅ Works
```

### Pattern 2: update_state return type

```rust
// Before: returns &mut ConcreteState
// After:  returns &mut StateData

// Callers using .coefficients(), .set_iteration(), etc. still work
```

### Pattern 3: Domination evaluation (in FCF)

```rust
// May need layout context for certain operations
// Check src/fcf.rs:180-220 for any layout-dependent code
```

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_state_pool_shared_layout_storage() {
    let config = StateConfig::Storage { num_hydros: 5 };
    let pool = VisitedStatePool::preallocate_concrete(4, 8, &config);
    
    assert_eq!(pool.pool.len(), 32);
    assert_eq!(pool.state_type, StateTypeId::Storage);
    assert_eq!(pool.num_hydros, 5);
    assert!(pool.layout.is_none());  // No layout for Storage
}

#[test]
fn test_state_pool_shared_layout_storage_and_inflow() {
    let config = StateConfig::StorageAndInflow {
        num_hydros: 2,
        per_hydro_state_dims: vec![2, 3],  // AR(1), AR(2)
    };
    let pool = VisitedStatePool::preallocate_concrete(4, 8, &config);
    
    assert_eq!(pool.pool.len(), 32);
    assert_eq!(pool.state_type, StateTypeId::StorageAndInflow);
    assert_eq!(pool.num_hydros, 2);
    
    // Layout stored ONCE
    let layout = pool.layout.as_ref().expect("should have layout");
    assert_eq!(layout.total_dim, 5);
    assert_eq!(layout.per_hydro_dims, vec![2, 3]);
}

#[test]
fn test_state_pool_layout_not_duplicated() {
    let config = StateConfig::StorageAndInflow {
        num_hydros: 10,
        per_hydro_state_dims: vec![2; 10],  // AR(1) for all
    };
    let pool = VisitedStatePool::preallocate_concrete(10, 50, &config);
    
    // 500 states, but only ONE layout
    assert_eq!(pool.pool.len(), 500);
    assert!(pool.layout.is_some());
    
    // Layout is not in each state - states are just StateData
    // (This is verified by the type system)
}

#[test]
fn test_state_pool_update_state_returns_state_data() {
    let config = StateConfig::Storage { num_hydros: 3 };
    let mut pool = VisitedStatePool::preallocate_concrete(2, 4, &config);
    
    let state = pool.update_state(3, &[10.0, 20.0, 30.0], 1, 3);
    
    assert_eq!(state.coefficients(), &[10.0, 20.0, 30.0]);
    assert_eq!(state.get_iteration(), 1);
}
```

---

## Pitfalls to Avoid

- ⚠️ **Check all pool.pool[slot] usages** - return type changes from ConcreteState to StateData
- ⚠️ **update_state return type** - changes from `&mut ConcreteState` to `&mut StateData`
- ⚠️ **Layout-dependent operations** - some code may need the layout; add get_layout() accessor

---

## Documentation Requirements

- [ ] Update struct documentation
- [ ] Document the data/metadata separation
- [ ] Update CHANGELOG

---

## Effort Estimate

**Points**: 5
**Confidence**: Medium
**Rationale**: Significant struct change with multiple callsite updates. Core logic is straightforward but need to audit all usages.

---

## Definition of Done

- [x] VisitedStatePool uses Vec<StateData>
- [x] Layout stored once in pool
- [x] All 573+ tests pass (578 tests, 3 pre-existing failures in unrelated test file)
- [x] No new deprecation warnings
- [x] Code is clippy-clean
