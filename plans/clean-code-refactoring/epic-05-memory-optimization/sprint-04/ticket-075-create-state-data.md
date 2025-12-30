# [T-075] Create StateData Struct (Pure Coefficient Data)

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 4: Pool Architecture Refinement](./00-sprint-overview.md)
> **Dependencies**: [T-072](../sprint-03/ticket-072-migrate-state-pool.md)
> **Blocks**: [T-076](./ticket-076-refactor-state-pool.md)

## Files to Read Before Starting

- `src/state.rs:401-478` - `StateCore` struct (similar purpose)
- `src/state.rs:480-787` - `ConcreteState` enum (to be replaced)
- `src/state.rs:871-1020` - `VisitedStatePool` (consumer)

---

## Context

### Background

The current `ConcreteState` enum embeds both state **data** and state **metadata** (layout). This causes:
1. Layout duplication across all states in a pool
2. Unnecessary enum variant dispatch
3. Memory overhead

### Current State

```rust
pub enum ConcreteState {
    Storage {
        core: StateCore,        // Data
        num_hydros: usize,      // Metadata
    },
    StorageAndInflow {
        core: StateCore,        // Data
        num_hydros: usize,      // Metadata
        layout: StateLayout,    // Metadata (DUPLICATED!)
    },
}
```

### Target State

```rust
/// Pure state data - no metadata
#[derive(Debug, Clone)]
pub struct StateData {
    pub coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}
```

The metadata (num_hydros, layout) moves to the pool level in T-076.

---

## Specification

### StateData Struct

```rust
/// Pure state coefficient data without layout metadata.
///
/// # Architecture (Epic 5 - T-075)
///
/// Separates state **data** from state **metadata**. The layout information
/// is stored once at the pool level, not duplicated per state.
///
/// # Memory Layout
///
/// - Stack: 56 bytes (5 fields)
/// - Heap: 8 × dimension bytes (coefficients Vec)
///
/// This is ~160 bytes smaller than `ConcreteState::StorageAndInflow` which
/// duplicates `StateLayout` in every instance.
#[derive(Debug, Clone)]
pub struct StateData {
    /// State coefficients (storage volumes + optional lag values)
    pub coefficients: Vec<f64>,
    
    /// Best cut height observed at this state
    pub dominating_objective: f64,
    
    /// ID of the cut that achieves dominating_objective
    pub dominating_cut_id: usize,
    
    /// Training iteration when this state was visited (1-based)
    pub iteration: usize,
    
    /// Forward pass index that visited this state (0-based)
    pub forward_pass_idx: usize,
}
```

### Required Methods

```rust
impl StateData {
    /// Create a new StateData with the specified dimension.
    pub fn new(dimension: usize) -> Self;
    
    /// Create with existing coefficients.
    pub fn with_coefficients(coefficients: Vec<f64>) -> Self;
    
    /// Get coefficients slice.
    pub fn coefficients(&self) -> &[f64];
    
    /// Update coefficients in place.
    pub fn update_coefficients(&mut self, values: &[f64]);
    
    /// Get dimension.
    pub fn dimension(&self) -> usize;
    
    /// Reset to zero values.
    pub fn reset_to_zero(&mut self);
    
    /// Clone data from another StateData.
    pub fn clone_from_data(&mut self, other: &StateData);
    
    // Getters/setters for tracking fields
    pub fn get_iteration(&self) -> usize;
    pub fn set_iteration(&mut self, iteration: usize);
    pub fn get_forward_pass_idx(&self) -> usize;
    pub fn set_forward_pass_idx(&mut self, idx: usize);
    pub fn get_dominating_cut_id(&self) -> usize;
    pub fn set_dominating_cut_id(&mut self, id: usize);
    pub fn get_dominating_objective(&self) -> f64;
    pub fn set_dominating_objective(&mut self, obj: f64);
    
    /// Update dominating cut (convenience method)
    pub fn update_dominating_cut(&mut self, cut: &BendersCut, height: f64);
}
```

---

## Acceptance Criteria

- [x] `StateData` struct implemented in `src/state.rs`
- [x] All methods from specification implemented
- [x] Unit tests for all methods
- [x] Memory size is smaller than `ConcreteState::StorageAndInflow`
- [x] No compilation errors

---

## Implementation Guide

### Step 1: Add StateData struct

Add after `StateCore` definition (around line 478):

```rust
/// Pure state coefficient data without layout metadata.
#[derive(Debug, Clone)]
pub struct StateData {
    pub coefficients: Vec<f64>,
    pub dominating_objective: f64,
    pub dominating_cut_id: usize,
    pub iteration: usize,
    pub forward_pass_idx: usize,
}
```

### Step 2: Implement basic methods

```rust
impl StateData {
    #[inline]
    pub fn new(dimension: usize) -> Self {
        Self {
            coefficients: vec![0.0; dimension],
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
    
    #[inline]
    pub fn with_coefficients(coefficients: Vec<f64>) -> Self {
        Self {
            coefficients,
            dominating_objective: 0.0,
            dominating_cut_id: 0,
            iteration: 0,
            forward_pass_idx: 0,
        }
    }
    
    #[inline]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
    
    #[inline]
    pub fn dimension(&self) -> usize {
        self.coefficients.len()
    }
    
    #[inline]
    pub fn update_coefficients(&mut self, values: &[f64]) {
        debug_assert_eq!(self.coefficients.len(), values.len());
        self.coefficients.copy_from_slice(values);
    }
    
    pub fn reset_to_zero(&mut self) {
        self.coefficients.fill(0.0);
        self.dominating_objective = 0.0;
        self.dominating_cut_id = 0;
        self.iteration = 0;
        self.forward_pass_idx = 0;
    }
}
```

### Step 3: Add tracking field accessors

```rust
impl StateData {
    #[inline]
    pub fn get_iteration(&self) -> usize { self.iteration }
    
    #[inline]
    pub fn set_iteration(&mut self, iteration: usize) { self.iteration = iteration; }
    
    #[inline]
    pub fn get_forward_pass_idx(&self) -> usize { self.forward_pass_idx }
    
    #[inline]
    pub fn set_forward_pass_idx(&mut self, idx: usize) { self.forward_pass_idx = idx; }
    
    #[inline]
    pub fn get_dominating_cut_id(&self) -> usize { self.dominating_cut_id }
    
    #[inline]
    pub fn set_dominating_cut_id(&mut self, id: usize) { self.dominating_cut_id = id; }
    
    #[inline]
    pub fn get_dominating_objective(&self) -> f64 { self.dominating_objective }
    
    #[inline]
    pub fn set_dominating_objective(&mut self, obj: f64) { self.dominating_objective = obj; }
    
    #[inline]
    pub fn update_dominating_cut(&mut self, cut: &cut::BendersCut, height: f64) {
        self.dominating_cut_id = cut.id;
        self.dominating_objective = height;
    }
    
    pub fn clone_from_data(&mut self, other: &StateData) {
        self.coefficients.copy_from_slice(&other.coefficients);
        self.dominating_objective = other.dominating_objective;
        self.dominating_cut_id = other.dominating_cut_id;
        self.iteration = other.iteration;
        self.forward_pass_idx = other.forward_pass_idx;
    }
}
```

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_state_data_new() {
    let data = StateData::new(5);
    assert_eq!(data.dimension(), 5);
    assert_eq!(data.coefficients(), &[0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(data.get_iteration(), 0);
}

#[test]
fn test_state_data_with_coefficients() {
    let data = StateData::with_coefficients(vec![1.0, 2.0, 3.0]);
    assert_eq!(data.coefficients(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_state_data_update_coefficients() {
    let mut data = StateData::new(3);
    data.update_coefficients(&[1.0, 2.0, 3.0]);
    assert_eq!(data.coefficients(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_state_data_tracking() {
    let mut data = StateData::new(3);
    data.set_iteration(5);
    data.set_forward_pass_idx(10);
    data.set_dominating_cut_id(42);
    data.set_dominating_objective(123.456);
    
    assert_eq!(data.get_iteration(), 5);
    assert_eq!(data.get_forward_pass_idx(), 10);
    assert_eq!(data.get_dominating_cut_id(), 42);
    assert!((data.get_dominating_objective() - 123.456).abs() < 1e-10);
}

#[test]
fn test_state_data_reset() {
    let mut data = StateData::new(3);
    data.update_coefficients(&[1.0, 2.0, 3.0]);
    data.set_iteration(5);
    
    data.reset_to_zero();
    
    assert_eq!(data.coefficients(), &[0.0, 0.0, 0.0]);
    assert_eq!(data.get_iteration(), 0);
}

#[test]
fn test_state_data_clone_from() {
    let mut target = StateData::new(3);
    let source = {
        let mut s = StateData::new(3);
        s.update_coefficients(&[1.0, 2.0, 3.0]);
        s.set_iteration(5);
        s
    };
    
    target.clone_from_data(&source);
    
    assert_eq!(target.coefficients(), &[1.0, 2.0, 3.0]);
    assert_eq!(target.get_iteration(), 5);
}

#[test]
fn test_state_data_size_smaller_than_concrete_state() {
    use std::mem::size_of;
    
    // StateData should be smaller than ConcreteState
    // StateData: Vec (24) + f64 (8) + 3*usize (24) = 56 bytes
    // ConcreteState::StorageAndInflow has StateCore + num_hydros + StateLayout
    assert!(size_of::<StateData>() < size_of::<ConcreteState>());
}
```

---

## Pitfalls to Avoid

- ⚠️ **Don't remove ConcreteState yet** - That's T-078
- ⚠️ **Match StateCore interface** - Keep method names consistent
- ⚠️ **Keep it simple** - StateData is just data, no behavior

---

## Documentation Requirements

- [ ] Doc comments on struct and all public methods
- [ ] Add to module documentation explaining data/metadata separation

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple struct with straightforward methods, similar to existing StateCore.

---

## Definition of Done

- [x] StateData struct implemented
- [x] All methods implemented with #[inline] where appropriate
- [x] Unit tests pass
- [x] Size assertion test passes
- [x] Code is clippy-clean
