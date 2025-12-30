# [T-060] Create CutStagingBuffer Struct

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Handler Staging Buffers](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: [T-061](./ticket-061-add-staging-to-handler.md), [T-063](./ticket-063-update-from-staging.md)

## Files to Read Before Starting

- `src/memory/buffers.rs` - Existing `CutComputationBuffers` pattern
- `src/cut.rs:67-105` - `CutEvalResult` struct (similar lightweight result)
- `src/sddp/mod.rs:95-110` - `BackwardPhase1Timing` struct
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Full architecture context

---

## Context

### Background

Each `SddpTrainHandler` needs a staging area to hold one computed cut and its associated state. This enables parallel cut computation where each handler writes to its own buffer, followed by sequential copy to global pools.

### Current State

Currently, cut computation produces `CutData` which allocates two `Vec<f64>`:
```rust
pub struct CutData {
    pub cut_coefficients: Vec<f64>,      // ALLOCATES via .to_vec()
    pub state_coefficients: Vec<f64>,    // ALLOCATES via .to_vec()
    ...
}
```

### Target State

A preallocated staging buffer that holds results without allocation:
```rust
pub struct CutStagingBuffer {
    pub cut_coefficients: Vec<f64>,      // Preallocated, reused
    pub state_coefficients: Vec<f64>,    // Preallocated, reused
    ...
}
```

---

## Specification

### Struct Definition

```rust
/// Staging area for one cut + one state computation.
///
/// Lives in `SddpTrainHandler`, reused across all stages within an iteration.
/// Enables parallel cut computation by giving each handler its own buffer.
///
/// # Memory
///
/// Size: ~2 × state_dim × 8 bytes ≈ 1.6 KB for 100-dimension state
///
/// # Usage
///
/// 1. Handler computes cut into thread-local `CutComputationBuffers`
/// 2. Results copied into this staging buffer via `stage_from()`
/// 3. Sequential loop copies from staging to global pools
/// 4. Buffer reused for next stage
#[derive(Debug)]
pub struct CutStagingBuffer {
    /// Computed cut coefficients (water values, lag duals)
    pub cut_coefficients: Vec<f64>,
    /// Computed cut RHS
    pub cut_rhs: f64,
    /// State coefficients at which cut was computed
    pub state_coefficients: Vec<f64>,
    /// Iteration that produced this cut (1-based)
    pub iteration: usize,
    /// Forward pass index (0-based)
    pub forward_pass_idx: usize,
    /// Timing from computation
    pub timing: BackwardPhase1Timing,
    /// Whether buffer contains valid data
    pub populated: bool,
}
```

### Methods

```rust
impl CutStagingBuffer {
    /// Create staging buffer with preallocated capacity.
    ///
    /// # Arguments
    /// * `state_dim` - Maximum state dimension for this handler
    pub fn new(state_dim: usize) -> Self;

    /// Copy computed results into staging area.
    ///
    /// Called at end of parallel cut computation, while still holding
    /// the thread-local buffer reference.
    ///
    /// # Arguments
    /// * `eval_result` - Reference to cut evaluation result
    /// * `state_coefficients` - State coefficients slice
    /// * `iteration` - Current iteration (1-based)
    /// * `forward_pass_idx` - Forward pass index (0-based)
    /// * `timing` - Computation timing
    pub fn stage_from(
        &mut self,
        eval_result: &CutEvalResult,
        state_coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
        timing: BackwardPhase1Timing,
    );

    /// Reset buffer for next iteration (clear populated flag).
    pub fn reset(&mut self);

    /// Get actual cut coefficient length (may be less than capacity).
    pub fn cut_len(&self) -> usize;

    /// Get actual state coefficient length.
    pub fn state_len(&self) -> usize;
}
```

### Behavior

- `new()`: Preallocates vectors to `state_dim` capacity, all zeros
- `stage_from()`: Copies slices via `copy_from_slice`, sets populated = true
- `reset()`: Sets populated = false, does NOT clear vectors (reuse capacity)

---

## Acceptance Criteria

- [ ] `CutStagingBuffer` struct defined in `src/memory/buffers.rs`
- [ ] All methods implemented with doc comments
- [ ] Unit tests for:
  - [ ] `new()` creates buffer with correct capacity
  - [ ] `stage_from()` copies data correctly
  - [ ] `reset()` clears populated flag
  - [ ] Multiple `stage_from()` calls reuse same memory
- [ ] No heap allocations after `new()` (verify with test)
- [ ] All existing tests pass

---

## Implementation Guide

### Step 1: Add struct to buffers.rs

Add after `CutComputationBuffers` (~line 180):

```rust
/// Staging area for one cut + one state computation.
/// [full doc comment from spec]
#[derive(Debug)]
pub struct CutStagingBuffer {
    // fields...
}
```

### Step 2: Implement methods

```rust
impl CutStagingBuffer {
    pub fn new(state_dim: usize) -> Self {
        Self {
            cut_coefficients: vec![0.0; state_dim],
            state_coefficients: vec![0.0; state_dim],
            cut_rhs: 0.0,
            iteration: 0,
            forward_pass_idx: 0,
            timing: BackwardPhase1Timing::default(),
            populated: false,
        }
    }

    #[inline]
    pub fn stage_from(
        &mut self,
        eval_result: &CutEvalResult,
        state_coefficients: &[f64],
        iteration: usize,
        forward_pass_idx: usize,
        timing: BackwardPhase1Timing,
    ) {
        let cut_len = eval_result.coefficients.len();
        let state_len = state_coefficients.len();
        
        debug_assert!(cut_len <= self.cut_coefficients.len());
        debug_assert!(state_len <= self.state_coefficients.len());
        
        self.cut_coefficients[..cut_len].copy_from_slice(eval_result.coefficients);
        self.state_coefficients[..state_len].copy_from_slice(state_coefficients);
        self.cut_rhs = eval_result.rhs;
        self.iteration = iteration;
        self.forward_pass_idx = forward_pass_idx;
        self.timing = timing;
        self.populated = true;
    }

    #[inline]
    pub fn reset(&mut self) {
        self.populated = false;
    }

    #[inline]
    pub fn cut_len(&self) -> usize {
        // In practice, use actual dimension tracking
        self.cut_coefficients.len()
    }

    #[inline]
    pub fn state_len(&self) -> usize {
        self.state_coefficients.len()
    }
}
```

### Step 3: Export from module

In `src/memory/mod.rs`, add to exports:
```rust
pub use buffers::CutStagingBuffer;
```

### Step 4: Add imports

```rust
use crate::cut::CutEvalResult;
use crate::sddp::BackwardPhase1Timing;
```

Note: May need to move `BackwardPhase1Timing` or import differently to avoid circular deps.

---

## Testing Requirements

### Unit Tests

Add to `src/memory/buffers.rs` tests module:

```rust
#[test]
fn test_staging_buffer_new() {
    let buf = CutStagingBuffer::new(100);
    assert_eq!(buf.cut_coefficients.len(), 100);
    assert_eq!(buf.state_coefficients.len(), 100);
    assert!(!buf.populated);
}

#[test]
fn test_staging_buffer_stage_from() {
    let mut buf = CutStagingBuffer::new(10);
    
    let coeffs = [1.0, 2.0, 3.0];
    let eval_result = CutEvalResult::new(&coeffs, 42.0, 1, 0);
    let state = [4.0, 5.0, 6.0];
    let timing = BackwardPhase1Timing::default();
    
    buf.stage_from(&eval_result, &state, 1, 0, timing);
    
    assert!(buf.populated);
    assert_eq!(buf.cut_coefficients[..3], [1.0, 2.0, 3.0]);
    assert_eq!(buf.state_coefficients[..3], [4.0, 5.0, 6.0]);
    assert_eq!(buf.cut_rhs, 42.0);
    assert_eq!(buf.iteration, 1);
    assert_eq!(buf.forward_pass_idx, 0);
}

#[test]
fn test_staging_buffer_reuse() {
    let mut buf = CutStagingBuffer::new(10);
    
    // First stage
    let coeffs1 = [1.0, 2.0];
    let eval1 = CutEvalResult::new(&coeffs1, 10.0, 1, 0);
    buf.stage_from(&eval1, &[3.0, 4.0], 1, 0, BackwardPhase1Timing::default());
    
    // Get pointer to verify no reallocation
    let ptr1 = buf.cut_coefficients.as_ptr();
    
    // Second stage
    let coeffs2 = [5.0, 6.0, 7.0];
    let eval2 = CutEvalResult::new(&coeffs2, 20.0, 1, 1);
    buf.stage_from(&eval2, &[8.0, 9.0, 10.0], 1, 1, BackwardPhase1Timing::default());
    
    // Verify same memory (no reallocation)
    let ptr2 = buf.cut_coefficients.as_ptr();
    assert_eq!(ptr1, ptr2);
    
    // Verify new data
    assert_eq!(buf.cut_coefficients[..3], [5.0, 6.0, 7.0]);
}

#[test]
fn test_staging_buffer_reset() {
    let mut buf = CutStagingBuffer::new(10);
    let coeffs = [1.0];
    let eval = CutEvalResult::new(&coeffs, 1.0, 1, 0);
    buf.stage_from(&eval, &[2.0], 1, 0, BackwardPhase1Timing::default());
    
    assert!(buf.populated);
    buf.reset();
    assert!(!buf.populated);
    
    // Data still there (not cleared, just marked invalid)
    assert_eq!(buf.cut_coefficients[0], 1.0);
}
```

---

## Pitfalls to Avoid

- ⚠️ **Circular dependencies**: `BackwardPhase1Timing` is in `sddp/mod.rs`. May need to move it to a shared location or use a simpler timing struct.
- ⚠️ **Dimension tracking**: The buffer is preallocated to max size, but actual data may be smaller. Consider adding `actual_cut_len` and `actual_state_len` fields.
- ⚠️ **Debug vs Release**: Use `debug_assert!` for bounds checks (zero cost in release).

---

## Documentation Requirements

- [ ] Doc comments on struct and all methods
- [ ] Add to module-level documentation in `src/memory/mod.rs`
- [ ] Reference in architecture doc if needed

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward struct with simple copy operations. Similar to existing `CutComputationBuffers`.

---

## Definition of Done

- [ ] Struct implemented with all methods
- [ ] All unit tests pass
- [ ] Exported from `memory` module
- [ ] Doc comments complete
- [ ] No new warnings
- [ ] All existing tests pass (549+)
