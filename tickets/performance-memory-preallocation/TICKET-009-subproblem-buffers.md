# TICKET-009: Implement SubproblemBuffers and integrate with Subproblem

## Context

This ticket adds buffer pre-allocation to Subproblem operations, eliminating allocations during uncertainty realization, state extraction, and constraint updates. Each subproblem solve (thousands per training) currently allocates multiple temporary buffers. By pre-allocating and reusing these buffers, we target 3-5% additional performance improvement.

**Why this matters**: Subproblems are solved thousands of times during training. While each solve's allocations are small, they accumulate significantly. This optimization complements the forward/backward pass optimizations by eliminating allocations in the LP solving layer.

**Part of**: Performance Implementation Plan - Phase 3: Forward Pass and Subproblem Optimization

**Depends on**: Memory module (TICKET-001, 002, 003)

## Acceptance Criteria

- [ ] Given SizingInfo, when SubproblemBuffers is created, then buffers are pre-allocated for all subproblem operations
- [ ] Given subproblem solve, when profiled, then zero allocations occur in realization and state extraction
- [ ] Given subproblem with buffers, when compared to baseline, then solve time overhead is reduced by 3-5%
- [ ] Given buffer reuse across solves, when validated, then no data leakage occurs
- [ ] Performance: Subproblem-related malloc overhead reduced by >40%
- [ ] Correctness: LP solutions identical to baseline (within solver tolerance)

## Tasks

### Implementation

- [ ] Create `src/subproblem/buffers.rs` file
- [ ] Implement `SubproblemBuffers` struct:
  - [ ] Add `realization_buffer: Buffer<f64>` field
  - [ ] Add `state_buffer: Buffer<f64>` field
  - [ ] Add `gradient_buffer: Buffer<f64>` field
  - [ ] Add `lag_buffer: Buffer<f64>` field
  - [ ] Add `constraint_rhs_buffer: Buffer<f64>` field
  - [ ] Add `sizing: SizingInfo` field
  - [ ] Implement `new(sizing: &SizingInfo)` constructor
  - [ ] Implement `reset()` method to clear all buffers
- [ ] Add `SubproblemBuffers` field to `Subproblem` struct
- [ ] Update `Subproblem::new()` to accept and store buffers:
  - [ ] Add `buffers: SubproblemBuffers` parameter
  - [ ] Or create buffers internally from SizingInfo
  - [ ] Document buffer lifecycle
- [ ] Refactor `realize_uncertainties()` to use realization_buffer:
  - [ ] Replace `Vec::new()` with buffer acquisition
  - [ ] Write realization values to buffer
  - [ ] Clear buffer at start of each call
  - [ ] Add PERFORMANCE comment
- [ ] Refactor `extract_state()` to use state_buffer:
  - [ ] Replace allocation with buffer write
  - [ ] Return buffer reference or copy
  - [ ] Clear buffer at start
- [ ] Refactor `compute_cut_gradient()` to use gradient_buffer:
  - [ ] Write dual values to buffer
  - [ ] Apply chain rule in-place
  - [ ] Return buffer contents
- [ ] Refactor AR lag tracking to use lag_buffer:
  - [ ] Pre-size buffer for max AR order
  - [ ] Update lag values in-place
  - [ ] Avoid reallocations
- [ ] Update constraint RHS updates to use constraint_rhs_buffer:
  - [ ] Pre-allocate buffer for constraint count
  - [ ] Update RHS values in-place
  - [ ] Pass buffer to solver interface
- [ ] Add buffer reset calls at appropriate points:
  - [ ] At start of `solve_forward_step()`
  - [ ] At start of `solve_backward_step()`
  - [ ] Before each realization

### Testing

- [ ] Unit test: Create SubproblemBuffers with realistic SizingInfo
  - [ ] Verify all buffers have correct capacity
  - [ ] Verify buffer sizes match SizingInfo dimensions
- [ ] Unit test: Buffer reset clears all buffers
  - [ ] Fill buffers with non-zero values
  - [ ] Call reset()
  - [ ] Verify all buffers are cleared
- [ ] Integration test: Subproblem solve with buffers on 03-multistage
  - [ ] Run forward and backward steps
  - [ ] Verify LP solutions match baseline
  - [ ] Verify state extraction correct
- [ ] Integration test: Subproblem solve on large-scale example
  - [ ] Run multiple iterations
  - [ ] Verify no memory leaks
  - [ ] Verify numerical correctness
- [ ] Correctness test: Uncertainty realization
  - [ ] Compare realized values with baseline
  - [ ] Verify AR lag tracking correct
  - [ ] Test with various AR orders (0, 1, 3, 5)
- [ ] Correctness test: State extraction
  - [ ] Compare extracted states with baseline
  - [ ] Test storage-only state space
  - [ ] Test storage+inflow state space
  - [ ] Verify all state components match
- [ ] Correctness test: Cut gradient computation
  - [ ] Compare gradients with baseline
  - [ ] Verify chain rule application correct
  - [ ] Test with various state dimensions
- [ ] Performance test: Allocation tracking
  - [ ] Profile subproblem solve with massif
  - [ ] Verify zero allocations in solve loop
  - [ ] Compare allocation count with baseline
- [ ] Performance test: Solve overhead
  - [ ] Benchmark subproblem solve time
  - [ ] Measure overhead of buffer management
  - [ ] Verify overhead is <1% of solve time
- [ ] Stress test: 10000 consecutive solves
  - [ ] Verify buffer reuse works correctly
  - [ ] Verify no data leakage between solves
  - [ ] Verify no memory growth

### Documentation

- [ ] Add comprehensive doc comment for `SubproblemBuffers`:
  - [ ] Purpose: Pre-allocated buffers for subproblem operations
  - [ ] Buffer organization and purposes
  - [ ] Usage pattern example
  - [ ] Lifecycle notes (create once, reset per solve)
- [ ] Update `Subproblem` struct doc comment:
  - [ ] Note about integrated buffer management
  - [ ] Performance characteristics with buffers
- [ ] Add doc comments for buffer-using methods:
  - [ ] `realize_uncertainties()` - note buffer reuse
  - [ ] `extract_state()` - note buffer output
  - [ ] `compute_cut_gradient()` - note buffer computation
- [ ] Add PERFORMANCE comments in code:
  - [ ] Explain why each buffer is needed
  - [ ] Note allocation elimination
  - [ ] Document buffer lifecycle
- [ ] Update module-level docs for `subproblem/`:
  - [ ] Add performance optimization notes
  - [ ] Link to memory module
- [ ] Add inline comments for non-obvious buffer usage:
  - [ ] AR lag buffer indexing
  - [ ] Constraint RHS buffer layout
  - [ ] Gradient buffer chain rule application

## Technical Notes

### Buffer Organization

**Structure**:
```rust
pub struct SubproblemBuffers {
    /// Buffer for realized uncertainty values (inflows, loads, etc.)
    pub realization_buffer: Buffer<f64>,
    
    /// Buffer for extracted state values
    pub state_buffer: Buffer<f64>,
    
    /// Buffer for cut gradient computation (dual values → coefficients)
    pub gradient_buffer: Buffer<f64>,
    
    /// Buffer for AR lag tracking
    pub lag_buffer: Buffer<f64>,
    
    /// Buffer for constraint RHS updates
    pub constraint_rhs_buffer: Buffer<f64>,
}
```

**Sizing Logic**:
- `realization_buffer`: Size = number of uncertain parameters (inflows + loads)
- `state_buffer`: Size = state dimension
- `gradient_buffer`: Size = state dimension (for cut coefficients)
- `lag_buffer`: Size = sum of AR orders for all inflows
- `constraint_rhs_buffer`: Size = number of constraints

### Uncertainty Realization Pattern

**Before** (allocating):
```rust
pub fn realize_uncertainties(&mut self, innovations: &[f64]) -> Result<Vec<f64>> {
    let mut realizations = Vec::new();  // ALLOCATION
    
    for (i, &innovation) in innovations.iter().enumerate() {
        let realized_value = self.apply_ar_model(i, innovation)?;
        realizations.push(realized_value);
    }
    
    Ok(realizations)
}
```

**After** (buffer reuse):
```rust
pub fn realize_uncertainties(
    &mut self,
    innovations: &[f64],
) -> Result<&[f64]> {
    // PERFORMANCE: Use pre-allocated buffer instead of allocating Vec.
    // This buffer is reused for every subproblem solve (thousands of times).
    self.buffers.realization_buffer.clear();
    
    for (i, &innovation) in innovations.iter().enumerate() {
        let realized_value = self.apply_ar_model(i, innovation)?;
        self.buffers.realization_buffer.push(realized_value);
    }
    
    Ok(self.buffers.realization_buffer.as_slice())
}
```

### State Extraction Pattern

**Before** (allocating):
```rust
pub fn extract_state(&self, solution: &SolveResult) -> Result<Vec<f64>> {
    let mut state = Vec::new();  // ALLOCATION
    
    // Extract storage values
    for hydro_idx in 0..self.system.hydro.len() {
        let storage = solution.get_variable_value(self.storage_vars[hydro_idx])?;
        state.push(storage);
    }
    
    // Extract AR lag values if needed
    if self.state_space.includes_inflow_lags() {
        state.extend(&self.current_lags);
    }
    
    Ok(state)
}
```

**After** (buffer reuse):
```rust
pub fn extract_state(&self, solution: &SolveResult) -> Result<&[f64]> {
    // PERFORMANCE: Write directly to pre-allocated state buffer
    self.buffers.state_buffer.clear();
    
    // Extract storage values
    for hydro_idx in 0..self.system.hydro.len() {
        let storage = solution.get_variable_value(self.storage_vars[hydro_idx])?;
        self.buffers.state_buffer.push(storage);
    }
    
    // Append AR lag values if needed
    if self.state_space.includes_inflow_lags() {
        self.buffers.state_buffer.extend_from_slice(&self.current_lags);
    }
    
    Ok(self.buffers.state_buffer.as_slice())
}
```

### Memory Footprint Estimation

For typical subproblem:
- Realization buffer: 156 hydros + 100 buses = 256 values × 8 bytes = 2KB
- State buffer: 200-dim state × 8 bytes = 1.6KB
- Gradient buffer: 200-dim × 8 bytes = 1.6KB
- Lag buffer: 156 hydros × max 5 lags × 8 bytes = 6.2KB
- Constraint RHS buffer: 500 constraints × 8 bytes = 4KB

**Total**: ~15KB per subproblem instance

For parallel execution with 8 threads: ~120KB total (negligible)

### Integration with Solver

**Key consideration**: Solver interface should accept buffer references, not take ownership:

```rust
// ✅ Good: Accepts reference
fn update_constraint_rhs(&mut self, rhs_values: &[f64]) -> Result<()>;

// ❌ Bad: Takes ownership
fn update_constraint_rhs(&mut self, rhs_values: Vec<f64>) -> Result<()>;
```

If solver interface takes ownership, we need to clone from buffer (acceptable overhead).

### Thread Safety for Parallel Execution

Each subproblem instance has its own buffers:
- Forward pass: Multiple subproblems, each with own buffers ✅
- Backward pass: Multiple subproblems, each with own buffers ✅
- No shared mutable state between subproblems ✅

### Buffer Lifecycle

```
Subproblem Creation
    ↓
SubproblemBuffers allocated (once)
    ↓
Training Loop
    ↓
For each iteration:
    ↓
    solve_forward_step()
        ↓
        buffers.reset()  ← Clear buffers
        ↓
        realize_uncertainties() → uses realization_buffer
        ↓
        solver.solve()
        ↓
        extract_state() → uses state_buffer
    ↓
    solve_backward_step()
        ↓
        buffers.reset()  ← Clear buffers
        ↓
        compute_cut_gradient() → uses gradient_buffer
    ↓
End Loop
```

### Edge Cases

- [ ] Empty AR orders (lag_buffer size = 0)
- [ ] Single-hydro system (minimal buffer sizes)
- [ ] Very large system (156+ hydros, verify no overflow)
- [ ] Storage-only state space (no lag buffer needed)
- [ ] Buffer capacity exceeded (should not happen, but verify panic)

### Performance Validation Strategy

1. **Profile baseline**: Subproblem allocation overhead
2. **Implement buffers**: Add SubproblemBuffers to Subproblem
3. **Profile optimized**: Measure allocation reduction
4. **Benchmark**: Compare solve+extraction time
5. **Target**: 3-5% overhead reduction, 40% malloc reduction in subproblem layer

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 3.2
- Current subproblem: `src/subproblem.rs`
- Buffer infrastructure: TICKET-002

## Dependencies

- Blocked by: TICKET-001, 002, 003 (memory module)
- Blocks: TICKET-010 (Vec::with_capacity audit)
- Related: TICKET-008 (forward pass buffers, can work in parallel)

## Estimated Effort

**5 story points** (3 days)

**Confidence**: Medium (subproblem is complex, many integration points)

**Breakdown**:
- Implementation: 1.5 days (multiple methods to refactor)
- Testing: 1 day (correctness critical for LP solving)
- Documentation: 0.5 day (many integration points to document)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests
- [ ] LP solutions verified identical to baseline (within solver tolerance)
- [ ] State extraction verified correct (all test cases pass)
- [ ] Cut gradients verified correct (numerical tests pass)
- [ ] Allocation profiling shows zero allocations in solve operations
- [ ] Performance benchmarks show 3-5% overhead reduction
- [ ] Stress test (10000 solves) passes with no leaks
- [ ] `cargo clippy` produces no new warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc` builds without warnings
- [ ] Code reviewed by team member
- [ ] Integration with forward/backward passes verified

## Notes

**Parallel Work**: This ticket can be worked on in parallel with TICKET-008 (ForwardPassBuffers) since they touch different modules.

**Correctness Critical**: Subproblem is at the core of the algorithm. Extensive testing is essential to ensure buffer reuse doesn't introduce bugs.

**Solver Interface**: Be careful with solver API. If solver expects ownership of data, we may need to clone from buffers (acceptable cost compared to repeated allocations).

**AR Lag Tracking**: The lag buffer management is non-trivial. Pay special attention to indexing and ensure lag updates are correct across solves.

**Return Type Changes**: Methods that currently return `Vec<T>` will return `&[T]` (buffer reference). Callers may need updates, but benefits outweigh the refactoring cost.
