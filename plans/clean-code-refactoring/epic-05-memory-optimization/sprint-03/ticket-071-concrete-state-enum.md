# [T-071] Create ConcreteState Enum

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 3: Pool Memory Model Optimization](./00-sprint-overview.md)
> **Dependencies**: None (can run in parallel with T-069, T-070)
> **Blocks**: [T-072](./ticket-072-migrate-state-pool.md)

## Files to Read Before Starting

- `src/state.rs:60-200` - `State` trait definition
- `src/state.rs:700-1160` - `StorageState` implementation
- `src/state.rs:1300-1760` - `StorageAndInflowState` implementation

---

## Context

### Background

`VisitedStatePool` currently uses `Vec<Box<dyn State>>` for dynamic dispatch. This adds:
- Vtable pointer lookup per method call (~1-3ns)
- Heap indirection (cache miss potential)
- Clone complexity

In a given problem, all states are the same concrete type. We can use an enum for static dispatch.

### Current State

```rust
pub trait State: Debug + Send + Sync {
    fn coefficients(&self) -> &[f64];
    fn update_coefficients(&mut self, new_values: &[f64]);
    // ... many more methods
}

pub struct VisitedStatePool {
    pub pool: Vec<Box<dyn State>>,
}
```

### Target State

```rust
pub enum ConcreteState {
    Storage(StorageStateCore),
    StorageAndInflow(StorageAndInflowStateCore),
}

impl ConcreteState {
    fn coefficients(&self) -> &[f64] {
        match self {
            Self::Storage(s) => s.coefficients(),
            Self::StorageAndInflow(s) => s.coefficients(),
        }
    }
    // ... delegate all methods
}
```

---

## Specification

### Enum Definition

```rust
/// Concrete state representation without dynamic dispatch.
///
/// Used in `VisitedStatePool` for cache-friendly storage and
/// static dispatch. Match statements compile to efficient jump tables.
#[derive(Debug, Clone)]
pub enum ConcreteState {
    /// Storage-only state (no AR inflows)
    Storage(StorageStateCore),
    /// Storage + lagged inflows state (for AR models)
    StorageAndInflow(StorageAndInflowStateCore),
}
```

### Required Methods

Implement all methods needed by domination evaluation and pool operations:

```rust
impl ConcreteState {
    /// Get state coefficients slice.
    pub fn coefficients(&self) -> &[f64];
    
    /// Update coefficients in place.
    pub fn update_coefficients(&mut self, new_values: &[f64]);
    
    /// Get iteration that created this state.
    pub fn get_iteration(&self) -> usize;
    
    /// Set iteration.
    pub fn set_iteration(&mut self, iteration: usize);
    
    /// Get forward pass index.
    pub fn get_forward_pass_idx(&self) -> usize;
    
    /// Set forward pass index.
    pub fn set_forward_pass_idx(&mut self, idx: usize);
    
    /// Get dominating cut ID.
    pub fn get_dominating_cut_id(&self) -> usize;
    
    /// Set dominating cut ID.
    pub fn set_dominating_cut_id(&mut self, id: usize);
    
    /// Get dominating objective.
    pub fn get_dominating_objective(&self) -> f64;
    
    /// Set dominating objective.
    pub fn set_dominating_objective(&mut self, obj: f64);
    
    /// Reset state to zero values.
    pub fn reset_to_zero(&mut self);
    
    /// Clone from another concrete state (same variant).
    pub fn clone_from_concrete(&mut self, other: &ConcreteState);
}
```

### Core Structs

The enum wraps the core data, not the full trait objects:

```rust
/// Core data for storage-only state.
#[derive(Debug, Clone)]
pub struct StorageStateCore {
    pub state_coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}

/// Core data for storage + inflow state.
#[derive(Debug, Clone)]
pub struct StorageAndInflowStateCore {
    pub state_coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
    // Additional fields specific to inflow state
    pub num_hydros: usize,
    pub per_hydro_state_dims: Vec<usize>,
}
```

---

## Acceptance Criteria

- [ ] `ConcreteState` enum defined
- [ ] All required methods implemented via match dispatch
- [ ] Unit tests verify functionality matches trait implementations
- [ ] Performance: match dispatch is not slower than vtable

---

## Implementation Guide

### Step 1: Create core structs

In `src/state.rs`, add after existing structs:

```rust
/// Core data for storage-only state (enum variant).
#[derive(Debug, Clone)]
pub struct StorageStateCore {
    /// Unified state coefficients (storage volumes)
    pub state_coefficients: Vec<f64>,
    /// Best cut height at this state
    pub dominating_objective: f64,
    /// ID of best cut
    pub dominating_cut_id: usize,
    /// Iteration that produced this state
    pub iteration: usize,
    /// Forward pass index
    pub forward_pass_idx: usize,
}

impl StorageStateCore {
    pub fn new(num_hydros: usize) -> Self {
        Self {
            state_coefficients: vec![0.0; num_hydros],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
    
    pub fn coefficients(&self) -> &[f64] {
        &self.state_coefficients
    }
    
    pub fn update_coefficients(&mut self, new_values: &[f64]) {
        self.state_coefficients[..new_values.len()].copy_from_slice(new_values);
    }
    
    pub fn reset_to_zero(&mut self) {
        self.state_coefficients.fill(0.0);
        self.dominating_objective = 0.0;
        self.dominating_cut_id = 0;
        self.iteration = 0;
        self.forward_pass_idx = 0;
    }
}
```

### Step 2: Create enum

```rust
/// Concrete state without dynamic dispatch.
#[derive(Debug, Clone)]
pub enum ConcreteState {
    Storage(StorageStateCore),
    StorageAndInflow(StorageAndInflowStateCore),
}

impl ConcreteState {
    /// Create storage-only variant.
    pub fn storage(num_hydros: usize) -> Self {
        Self::Storage(StorageStateCore::new(num_hydros))
    }
    
    /// Create storage + inflow variant.
    pub fn storage_and_inflow(num_hydros: usize, per_hydro_dims: Vec<usize>) -> Self {
        let total_dim = per_hydro_dims.iter().sum();
        Self::StorageAndInflow(StorageAndInflowStateCore {
            state_coefficients: vec![0.0; total_dim],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
            num_hydros,
            per_hydro_state_dims: per_hydro_dims,
        })
    }
    
    pub fn coefficients(&self) -> &[f64] {
        match self {
            Self::Storage(s) => s.coefficients(),
            Self::StorageAndInflow(s) => &s.state_coefficients,
        }
    }
    
    pub fn update_coefficients(&mut self, new_values: &[f64]) {
        match self {
            Self::Storage(s) => s.update_coefficients(new_values),
            Self::StorageAndInflow(s) => {
                s.state_coefficients[..new_values.len()].copy_from_slice(new_values);
            }
        }
    }
    
    // ... implement all other methods with match dispatch
}
```

### Step 3: Implement remaining methods

Each method follows the pattern:
```rust
pub fn get_iteration(&self) -> usize {
    match self {
        Self::Storage(s) => s.iteration,
        Self::StorageAndInflow(s) => s.iteration,
    }
}

pub fn set_iteration(&mut self, iteration: usize) {
    match self {
        Self::Storage(s) => s.iteration = iteration,
        Self::StorageAndInflow(s) => s.iteration = iteration,
    }
}
```

### Step 4: Add conversion from trait object

```rust
impl ConcreteState {
    /// Create from existing dyn State (for migration).
    pub fn from_dyn(state: &dyn State) -> Self {
        // Determine variant based on state dimension or type
        // This may require adding a method to State trait
        // or using downcast if State: Any
        todo!("Implement based on actual state detection")
    }
}
```

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_concrete_state_storage() {
    let mut state = ConcreteState::storage(3);
    assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);
    
    state.update_coefficients(&[1.0, 2.0, 3.0]);
    assert_eq!(state.coefficients(), &[1.0, 2.0, 3.0]);
    
    state.set_iteration(5);
    assert_eq!(state.get_iteration(), 5);
}

#[test]
fn test_concrete_state_storage_and_inflow() {
    let mut state = ConcreteState::storage_and_inflow(2, vec![2, 3]);
    assert_eq!(state.coefficients().len(), 5);  // 2 + 3
}

#[test]
fn test_concrete_state_reset() {
    let mut state = ConcreteState::storage(3);
    state.update_coefficients(&[1.0, 2.0, 3.0]);
    state.set_iteration(5);
    
    state.reset_to_zero();
    
    assert_eq!(state.coefficients(), &[0.0, 0.0, 0.0]);
    assert_eq!(state.get_iteration(), 0);
}
```

---

## Pitfalls to Avoid

- ⚠️ **Variant detection**: Need a way to determine which variant to create. May need to add enum tag to State trait.
- ⚠️ **Missing methods**: Audit all State trait methods used in pool operations.
- ⚠️ **Memory layout**: Ensure enum is not larger than Box<dyn State>.

---

## Effort Estimate

**Points**: 5  
**Confidence**: Medium  
**Rationale**: Significant boilerplate for match dispatch. Core structs need careful design.

---

## Definition of Done

- [ ] `ConcreteState` enum implemented
- [ ] All methods delegated via match
- [ ] Unit tests pass
- [ ] No dependency on Box<dyn State> for enum operations
