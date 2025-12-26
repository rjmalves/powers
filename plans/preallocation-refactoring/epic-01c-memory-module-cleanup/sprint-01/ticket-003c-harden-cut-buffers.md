# TICKET-003c: Harden CutComputationBuffers with Capacity Enforcement

## Status: ✅ COMPLETE (2025-12-26)

## Context

### Background

The `CutComputationBuffers` currently allows **silent reallocation** when the preallocated capacity is insufficient:

```rust
pub fn reset_for_cut(&mut self, state_dim: usize, num_scenarios: usize) {
    self.coefficients.clear();
    self.coefficients.resize(state_dim, 0.0);  // May reallocate!

    while self.contributions_outer.len() < num_scenarios {
        self.contributions_outer.push(Vec::with_capacity(state_dim));  // ALLOCATION!
    }
    // ...
}
```

This violates our goal of 100% memory determinism. If the initialization uses wrong dimensions (e.g., `num_hydros` instead of `num_hydros + inflow_lags`), allocations happen silently in the hot path.

### Relation to Epic

Part of Epic: [Epic 01c: Memory Module Cleanup](../00-epic-overview.md)  
Sprint: [Sprint 1](./00-sprint-overview.md)

### Current State

- `CutComputationBuffers::new(max_state_dim, max_scenarios)` - Preallocates
- `reset_for_cut(state_dim, num_scenarios)` - Silently grows if needed
- No way to detect capacity violations

## Specification

### Changes to CutComputationBuffers

1. **Add capacity tracking fields**:
```rust
pub struct CutComputationBuffers {
    pub coefficients: Vec<f64>,
    pub contributions_outer: Vec<Vec<f64>>,
    // NEW:
    max_state_dim: usize,
    max_scenarios: usize,
}
```

2. **Panic on capacity overflow**:
```rust
pub fn reset_for_cut(&mut self, state_dim: usize, num_scenarios: usize) {
    // ENFORCE: No dynamic allocation
    assert!(
        state_dim <= self.max_state_dim,
        "Cut buffer capacity overflow: state_dim {} > max_state_dim {}. \
         This indicates incorrect initialization. Check that max_state_dim \
         accounts for inflow lags in StorageAndInflowState.",
        state_dim,
        self.max_state_dim
    );
    assert!(
        num_scenarios <= self.max_scenarios,
        "Cut buffer capacity overflow: num_scenarios {} > max_scenarios {}",
        num_scenarios,
        self.max_scenarios
    );
    
    // Safe: capacity is guaranteed sufficient
    self.coefficients.clear();
    self.coefficients.resize(state_dim, 0.0);
    
    for contrib in self.contributions_outer.iter_mut().take(num_scenarios) {
        contrib.clear();
    }
}
```

3. **Remove lazy initialization fallback**:
```rust
pub fn with_cut_buffers<F, R>(f: F) -> R {
    CUT_BUFFERS.with(|buffers| {
        let mut buffers = buffers.borrow_mut();
        
        // REMOVE: No more lazy init with arbitrary defaults
        // if buffers.is_none() {
        //     *buffers = Some(CutComputationBuffers::new(50, 20));
        // }
        
        match buffers.as_mut() {
            Some(b) => f(b),
            None => panic!(
                "Cut computation buffers not initialized! \
                 Call initialize_cut_buffers() before training."
            ),
        }
    })
}
```

4. **Add capacity getter for debugging**:
```rust
impl CutComputationBuffers {
    pub fn capacity(&self) -> (usize, usize) {
        (self.max_state_dim, self.max_scenarios)
    }
}
```

### Behavior

- `reset_for_cut()` panics if requested dimensions exceed preallocated capacity
- `with_cut_buffers()` panics if buffers not initialized
- Error messages clearly indicate the problem and solution

## Acceptance Criteria

- [ ] `CutComputationBuffers` has `max_state_dim` and `max_scenarios` fields
- [ ] `reset_for_cut()` asserts dimensions are within capacity
- [ ] `with_cut_buffers()` panics if not initialized (no lazy fallback)
- [ ] Clear error messages with debugging information
- [ ] Code compiles
- [ ] All tests pass (may need to update tests that relied on lazy init)

## Implementation Guide

### Step 1: Add Capacity Fields

```rust
pub struct CutComputationBuffers {
    pub coefficients: Vec<f64>,
    pub contributions_outer: Vec<Vec<f64>>,
    max_state_dim: usize,
    max_scenarios: usize,
}
```

### Step 2: Update Constructor

```rust
pub fn new(max_state_dim: usize, max_scenarios: usize) -> Self {
    let mut contributions_outer = Vec::with_capacity(max_scenarios);
    for _ in 0..max_scenarios {
        contributions_outer.push(Vec::with_capacity(max_state_dim));
    }
    
    Self {
        coefficients: Vec::with_capacity(max_state_dim),
        contributions_outer,
        max_state_dim,
        max_scenarios,
    }
}
```

### Step 3: Add Capacity Assertions to reset_for_cut

See specification above.

### Step 4: Remove Lazy Initialization

In `with_cut_buffers()`, replace the lazy init with a panic.

### Step 5: Fix Tests

Tests that relied on auto-initialization need to call `initialize_cut_buffers()` first:

```rust
#[test]
fn test_cut_buffers_auto_initialization() {
    // BEFORE: Relied on lazy init
    // with_cut_buffers(|buffers| { ... });
    
    // AFTER: Explicit init
    initialize_cut_buffers(10, 4);
    with_cut_buffers(|buffers| { ... });
}
```

### Key Files to Modify

- `src/memory/buffers.rs` - Core changes

### Pitfalls to Avoid

- ⚠️ Don't remove `with_cut_buffers` - it's used in hot path
- ⚠️ Ensure the panic messages are clear and actionable
- ⚠️ Update tests before running them

## Testing Requirements

### Unit Tests

```rust
#[test]
#[should_panic(expected = "state_dim")]
fn test_cut_buffers_overflow_state_dim() {
    initialize_cut_buffers(10, 4);
    with_cut_buffers(|buffers| {
        buffers.reset_for_cut(100, 4);  // 100 > 10, should panic
    });
}

#[test]
#[should_panic(expected = "num_scenarios")]
fn test_cut_buffers_overflow_scenarios() {
    initialize_cut_buffers(10, 4);
    with_cut_buffers(|buffers| {
        buffers.reset_for_cut(10, 100);  // 100 > 4, should panic
    });
}

#[test]
#[should_panic(expected = "not initialized")]
fn test_cut_buffers_uninitialized_panics() {
    CUT_BUFFERS.with(|b| *b.borrow_mut() = None);
    with_cut_buffers(|_| {});  // Should panic
}
```

### Integration Test

```bash
cargo run --release -- run examples/07-par-model-with-inflow-state
# Should work without panics (correct dimensions)
```

## Documentation Requirements

- [ ] Update doc comments for `CutComputationBuffers`
- [ ] Document panic conditions
- [ ] Update module docs

## Dependencies

- **Blocked By**: TICKET-002c (clean module state)
- **Blocks**: TICKET-004c (correct initialization)

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Straightforward changes but need to update tests

## Definition of Done

- [ ] Capacity fields added
- [ ] Assertions in place
- [ ] Lazy init removed
- [ ] All tests pass
- [ ] Examples run without panics
