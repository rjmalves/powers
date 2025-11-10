# TICKET-006b: Eliminate Nested Allocations in Backward Pass with Pre-allocated Buffers

## Status: 📋 READY TO START (Priority: HIGH)

**Created**: 2025-11-10  
**Priority**: P1 - HIGH (Completes backward pass optimization)  
**Estimated Effort**: 10 story points (2 days)  
**Confidence**: High

---

## Context

TICKET-006 successfully eliminated **outer allocations** (the unzip operation), saving ~30,720 reallocations at production scale. However, this optimization only addressed the tip of the iceberg.

### The Remaining Problem: Nested Allocations

**Inside each cut computation**, there are multiple allocations that happen in the hot path:

```rust
fn compute_cut_for_backward_step(...) -> CutStatePair {
    // ALLOCATION #1: coefficients vector (156 floats = 1,248 bytes)
    let mut coefficients = Vec::new();
    
    for hydro in hydros {
        coefficients.push(compute_coeff(hydro));  // Reallocations as vector grows
    }
    
    // ALLOCATION #2: state vector (156 floats = 1,248 bytes)
    let state = extract_state(...);  // Also allocates internally
    
    CutStatePair {
        cut: BendersCut { coefficients, ... },
        state,
    }
}
```

**At production scale (192 forward passes, 32 iterations, 5 stages)**:
- Cut coefficient allocations: 192 × 32 × 5 = **30,720 allocations** (122 KB each)
- State vector allocations: 192 × 32 × 5 = **30,720 allocations** (122 KB each)
- **Total nested allocations: ~61,440 per training run** (vs 30,720 outer allocations fixed in TICKET-006)

### Current vs Target State

| Metric | After TICKET-006 | After TICKET-006b | Improvement |
|--------|------------------|-------------------|-------------|
| Outer allocations | 0 ✅ | 0 ✅ | — |
| Nested allocations | ~61,440 ❌ | ~50 ✅ | **99.9% reduction** |
| Malloc overhead | ~8-10% | <2% | **75% reduction** |
| Backward pass time | Baseline | -10-15% | **Significant** |

### Why Now?

TICKET-000 provides the deep memory estimation needed to:
1. **Measure accurately**: Know true allocation costs
2. **Size correctly**: Pre-allocate with exact capacity
3. **Validate properly**: Confirm actual reduction vs estimate

**See**: `MEMORY_OPTIMIZATION_STRATEGY.md` Section 4 for complete pattern design

---

## Acceptance Criteria

- [ ] Given backward pass execution, when computing cuts, then zero allocations occur in nested structures (verified by profiler)
- [ ] Given 192 forward passes, when running backward pass, then ~61,440 nested allocations are eliminated
- [ ] Given production workload, when profiling with perf, then malloc overhead reduced from ~8% to <2%
- [ ] Given backward pass time measurement, when comparing to baseline, then improvement is 10-15% faster
- [ ] Given numerical results, when comparing to TICKET-006 baseline, then results are identical (within 1e-10)
- [ ] Given parallel execution, when multiple threads compute cuts, then thread-safety is maintained
- [ ] Given thread-local buffers, when acquired and released, then no contention or deadlocks occur
- [ ] All existing tests pass (486 tests, no behavior changes)
- [ ] Performance improvement scales with num_forward_passes (negligible at 4, significant at 192+)

---

## Tasks

### Implementation - Phase 1: Thread-Local Buffer Infrastructure

- [ ] Add `CoefficientBufferPool` to `src/memory/buffers.rs`:
  ```rust
  pub struct CoefficientBufferPool {
      buffers: Vec<Vec<f64>>,  // One per thread
  }
  ```
  - [ ] Create with sizing: `num_threads × max_state_dimension`
  - [ ] Implement `acquire(thread_id)` to get mutable buffer reference
  - [ ] Add thread-safety documentation (each thread has unique buffer)
- [ ] Add `StateBufferPool` to `src/memory/buffers.rs`:
  ```rust
  pub struct StateBufferPool {
      buffers: Vec<Vec<f64>>,  // One per thread
  }
  ```
  - [ ] Similar structure to CoefficientBufferPool
  - [ ] Documentation explaining separation (cut coeffs vs state values)
- [ ] Update `ThreadLocalBuffers` struct to include new pools:
  - [ ] Add `coeff_buffers: CoefficientBufferPool`
  - [ ] Add `state_buffers: StateBufferPool`
  - [ ] Update `new(sizing: &SizingInfo)` constructor
  - [ ] Update `DeepSizeEstimate` implementation to include new buffers
- [ ] Add unit tests for buffer pools:
  - [ ] Test: Create pool with correct capacity per thread
  - [ ] Test: Acquire returns buffer with expected capacity
  - [ ] Test: Acquired buffer can be filled and cleared
  - [ ] Test: Different thread IDs get different buffers

### Implementation - Phase 2: Cut Computation Refactoring

- [ ] Identify current cut computation function in `src/sddp/mod.rs`:
  - [ ] Locate `compute_cut_for_backward_step` or equivalent
  - [ ] Document current allocation pattern
  - [ ] Measure current allocations with profiler (baseline)
- [ ] Add new method signature with buffer parameter:
  ```rust
  fn compute_cut_with_buffers(
      &self,
      thread_id: usize,
      coeff_buffer: &mut [f64],
      state_buffer: &mut [f64],
      ...
  ) -> CutStatePair
  ```
- [ ] Refactor coefficient computation:
  - [ ] Replace `let mut coefficients = Vec::new()` with buffer writes
  - [ ] Use `coeff_buffer.fill(0.0)` at start
  - [ ] Write coefficients directly to buffer: `coeff_buffer[i] = ...`
  - [ ] Clone exact slice to create final Vec: `coeff_buffer[..len].to_vec()`
  - [ ] Add PERFORMANCE comment explaining one allocation at exact size
- [ ] Refactor state extraction:
  - [ ] Replace state allocation with buffer writes
  - [ ] Use `state_buffer.fill(0.0)` at start
  - [ ] Extract state values directly to buffer
  - [ ] Clone exact slice to create final state vector
- [ ] Update call sites to pass buffers:
  - [ ] Get thread_id from Rayon: `rayon::current_thread_index().unwrap()`
  - [ ] Acquire buffers from pool
  - [ ] Pass to refactored computation method
- [ ] Add safety checks:
  - [ ] Assert buffer capacity >= required size
  - [ ] Add helpful error messages if assertion fails
  - [ ] Document buffer size requirements

### Implementation - Phase 3: Integration with Backward Pass

- [ ] Add buffer pools to `SddpAlgorithm` struct:
  - [ ] Field: `thread_local_buffers: ThreadLocalBuffers`
  - [ ] Initialize in constructor from `SizingInfo`
  - [ ] Update `DeepSizeEstimate` to include buffers
- [ ] Update backward pass Phase 1 to use buffers:
  ```rust
  let phase1_results: Vec<(CutStatePair, Timing)> = train_handlers
      .par_iter_mut()
      .enumerate()
      .map(|(idx, handler)| {
          let thread_id = rayon::current_thread_index().unwrap();
          let coeff_buf = &mut self.thread_local_buffers.coeff_buffers.acquire(thread_id);
          let state_buf = &mut self.thread_local_buffers.state_buffers.acquire(thread_id);
          
          handler.compute_cut_with_buffers(thread_id, coeff_buf, state_buf, ...)
      })
      .collect()?;
  ```
- [ ] Verify thread-safety:
  - [ ] Document that each thread has unique buffer index
  - [ ] Confirm no shared mutable state
  - [ ] Add test with parallel execution (4+ threads)
- [ ] Add PERFORMANCE comments explaining optimization:
  - [ ] Why thread-local buffers (avoid contention)
  - [ ] What allocations are eliminated (nested vecs)
  - [ ] Expected impact at production scale (61K → ~50)
  - [ ] Scaling characteristics (negligible small, significant large)

### Testing - Correctness Tests

- [ ] Unit test: Buffer-based cut computation produces identical result:
  - [ ] Create test with known inputs
  - [ ] Compute with old allocation method
  - [ ] Compute with new buffer method
  - [ ] Assert results identical (within 1e-15)
- [ ] Unit test: State extraction with buffers matches original:
  - [ ] Extract state with old method
  - [ ] Extract state with new buffer method
  - [ ] Assert vectors are identical
- [ ] Integration test: Full backward pass numerically identical:
  - [ ] Run backward pass on 03-multistage with buffers
  - [ ] Compare cuts to baseline (TICKET-006 results)
  - [ ] Assert all cuts identical within 1e-10
  - [ ] Verify training convergence unchanged
- [ ] Integration test: All 4 examples produce identical results:
  - [ ] Example 01: Verify objective value unchanged
  - [ ] Example 02: Verify objective value unchanged
  - [ ] Example 03: Verify objective value unchanged
  - [ ] Example 05: Verify objective value unchanged

### Testing - Thread Safety Tests

- [ ] Unit test: Parallel buffer acquisition is safe:
  - [ ] Spawn 4 threads
  - [ ] Each thread acquires buffer with unique ID
  - [ ] Verify no panics or contention
  - [ ] Verify each thread gets different buffer
- [ ] Integration test: Parallel backward pass with multiple threads:
  - [ ] Run backward pass with 4 forward passes (4 threads)
  - [ ] Run backward pass with 10 forward passes (10 threads)
  - [ ] Verify no race conditions or deadlocks
  - [ ] Verify results are deterministic (same inputs → same outputs)
- [ ] Stress test: 100 iterations with parallel execution:
  - [ ] Run training to completion
  - [ ] Verify no memory corruption
  - [ ] Verify consistent results across iterations

### Testing - Performance Tests

- [ ] Benchmark: Backward pass on 03-multistage (4 forward passes):
  - [ ] Measure time before optimization (TICKET-006 baseline)
  - [ ] Measure time after optimization (TICKET-006b)
  - [ ] Document result (expected: negligible difference)
  - [ ] Reason: Small problem, allocation overhead already low
- [ ] Benchmark: Backward pass with 50 forward passes:
  - [ ] Create test configuration with 50 FPs
  - [ ] Measure time before vs after
  - [ ] Expected: 5-8% improvement
- [ ] Benchmark: Backward pass with 192 forward passes (production):
  - [ ] Create test configuration with 192 FPs
  - [ ] Measure time before vs after
  - [ ] Expected: 10-15% improvement
  - [ ] Document scaling characteristic
- [ ] Profile: Measure malloc overhead reduction:
  - [ ] Profile with perf before optimization
  - [ ] Extract malloc overhead percentage
  - [ ] Profile with perf after optimization
  - [ ] Verify malloc overhead: 8-10% → <2%
- [ ] Profile: Count actual allocations:
  - [ ] Use allocation tracker or strace
  - [ ] Count allocations in backward pass loop
  - [ ] Verify ~61,440 allocations eliminated
  - [ ] Document results in ticket

### Testing - Regression Tests

- [ ] Run full test suite: `cargo test`
  - [ ] All 486 existing tests must pass
  - [ ] No behavior changes in other modules
- [ ] Run all 4 examples:
  - [ ] Verify they complete successfully
  - [ ] Verify output is numerically identical
  - [ ] Verify convergence behavior unchanged
- [ ] Check for memory leaks:
  - [ ] Run valgrind on 03-multistage
  - [ ] Verify zero memory leaks
  - [ ] Verify buffer cleanup on algorithm drop

### Documentation

- [ ] Add inline PERFORMANCE comments to refactored code:
  - [ ] Explain nested allocation elimination
  - [ ] Document thread-local buffer pattern
  - [ ] Show allocation reduction calculation
  - [ ] Explain scaling with num_forward_passes
- [ ] Update `BackwardPassBuffers` documentation:
  - [ ] Note that this struct is for different optimization pattern
  - [ ] Explain why we use thread-local buffers instead
  - [ ] Cross-reference to TICKET-006b implementation
- [ ] Update `src/memory/buffers.rs` module documentation:
  - [ ] Add section on thread-local buffer pattern
  - [ ] Explain when to use buffer pools
  - [ ] Provide usage example for backward pass
- [ ] Update `MEMORY_OPTIMIZATION_STRATEGY.md`:
  - [ ] Mark TICKET-006b as complete
  - [ ] Document actual performance results
  - [ ] Compare estimate vs measured allocation reduction
- [ ] Add section to `PERFORMANCE_REFACTORING_PLAN.md`:
  - [ ] Document Phase 2 completion
  - [ ] Include profiling results
  - [ ] Update metrics table with actual improvements
- [ ] Update `CHANGELOG.md`:
  - [ ] Add entry under "Performance" section
  - [ ] Note: "Eliminated nested allocations in backward pass, 10-15% faster at production scale"

---

## Technical Notes

### Thread-Local Buffer Pattern

**Design**:
```rust
// At algorithm initialization
let thread_local_buffers = ThreadLocalBuffers::new(&sizing);
// Allocates: num_threads × (coeff_buffer + state_buffer)
// Total: 8 threads × (156 floats + 156 floats) × 8 bytes = ~20 KB
// Compare to: 61,440 allocations × 1,248 bytes = ~77 MB per training run

// In hot path
let thread_id = rayon::current_thread_index().unwrap();
let coeff_buf = buffers.coeff_buffers.acquire(thread_id);  // Zero cost
coeff_buf.fill(0.0);  // Reuse buffer
// ... compute into buffer ...
let coefficients = coeff_buf[..len].to_vec();  // One allocation, exact size
```

**Why Thread-Local?**
- **No contention**: Each thread has dedicated buffer
- **No locks**: No mutex or atomic operations needed
- **Cache-friendly**: Same thread reuses same buffer (cache locality)
- **Simple**: Thread ID from Rayon is guaranteed unique

**Why Not BackwardPassBuffers from TICKET-005?**

The `BackwardPassBuffers` struct was designed for a different pattern (per-trajectory buffers) that doesn't fit the actual 3-phase architecture:
- Phase 1 uses Rayon parallel map (can't mutate external buffers)
- Thread-local pattern fits Rayon's execution model
- Simpler and more efficient than per-trajectory allocation

### Memory Impact Analysis

**Before TICKET-006b** (After TICKET-006):
```
Per iteration at production scale (192 FPs, 5 stages):
- Outer allocations: 0 (fixed by TICKET-006)
- Cut coefficient vecs: 192 × 5 = 960 allocations × 1,248 bytes = 1.2 MB
- State vecs: 192 × 5 = 960 allocations × 1,248 bytes = 1.2 MB
- Total per iteration: ~2.4 MB allocated

Over 32 iterations:
- Total nested allocations: 61,440
- Total memory allocated and freed: ~77 MB
- Malloc overhead: ~8-10% CPU time
```

**After TICKET-006b**:
```
At initialization:
- Thread-local buffers: 8 threads × 2 buffers × 1,248 bytes = ~20 KB (one-time)

Per iteration:
- Buffer reuse: 0 allocations (just fill and clear)
- Final Vec clones: 960 allocations × 1,248 bytes = 1.2 MB (one allocation per cut at exact size)

Over 32 iterations:
- Total nested allocations: ~30,720 (50% reduction from pre-allocation)
- Compare to before: 61,440 → 30,720 (50% reduction)
- Malloc overhead: ~8-10% → ~4-5%
```

**Note**: We still need one final allocation per cut to create the owned `Vec<f64>` that escapes to the result. This is unavoidable unless we change the API to use buffer references (much larger refactoring). The win is:
1. No reallocations during coefficient computation (exact capacity upfront)
2. Buffer reuse across iterations (zero allocations except final clone)

### Performance Scaling

| num_forward_passes | Allocations Eliminated | Expected Improvement |
|--------------------|------------------------|---------------------|
| 4 (small) | ~640 | Negligible (~1-2%) |
| 10 (typical) | ~1,600 | Small (~3-5%) |
| 50 (large) | ~8,000 | Moderate (~5-8%) |
| 192 (production) | ~30,720 | Significant (~10-15%) |

**Why improvement is less than allocation reduction**: Allocation is only one component of backward pass time. Other costs:
- Solver solve time (dominant)
- Cut computation math
- State extraction
- Data structure updates

At 192 forward passes, allocation becomes a larger fraction of non-solver time, hence the improvement.

### Edge Cases

1. **Thread count mismatch**: Rayon pool size might not match sizing.num_threads
   - **Solution**: Use `rayon::current_num_threads()` in sizing
2. **Buffer too small**: State dimension varies per node
   - **Solution**: Use `max_state_dimension` for buffer capacity
   - **Safety**: Assert buffer capacity in computation function
3. **No parallel execution**: Sequential backward pass (num_forward_passes = 1)
   - **Solution**: Still use buffers, just thread_id = 0
   - **Behavior**: Same optimization, single thread

### Validation Strategy

**Three-Level Validation**:

1. **Correctness**: All tests pass, numerical results identical
2. **Performance**: Profiling confirms allocation reduction
3. **Production**: Real workload shows expected improvement

**If Performance Target Not Met**:
- Acceptable if >5% at production scale
- Document actual results honestly
- Identify next bottleneck with profiler
- Don't over-optimize without data

### References

- See `MEMORY_OPTIMIZATION_STRATEGY.md` Section 4 for pattern design
- See TICKET-006 for outer allocation optimization (completed)
- See TICKET-000 for deep memory estimation (prerequisite)
- See profiling analysis showing 8-10% malloc overhead (not 2%)

---

## Dependencies

### Blocked By

- **TICKET-000**: Deep memory estimation (need accurate buffer sizing) ✅ Must complete first
- **TICKET-006**: Outer allocation optimization (baseline for comparison) ✅ Complete

### Blocks

- **TICKET-007**: Performance validation (needs complete optimization for measurement)

### Related

- **TICKET-005**: BackwardPassBuffers (different pattern, not used here)
- **TICKET-008**: Forward pass optimization (will use same pattern)

---

## Estimated Effort

**10 story points (2 days)**

**Breakdown**:
- Buffer infrastructure: 3 hours
- Cut computation refactoring: 4 hours
- Integration with backward pass: 3 hours
- Correctness tests: 4 hours
- Thread safety tests: 2 hours
- Performance tests: 4 hours
- Profiling and validation: 3 hours
- Documentation: 3 hours
- **Total**: ~26 hours (3 days with buffer for testing)

**Confidence**: High
- Pattern proven in similar optimizations
- Implementation is straightforward refactoring
- Thread-safety is guaranteed by design (unique buffers)
- Risk: Low (can validate correctness with existing tests)

---

## Validation Checklist

Before marking this ticket complete:

- [ ] All 486 unit tests pass
- [ ] All 4 examples produce identical numerical results
- [ ] Thread safety tests pass with 4+ threads
- [ ] Performance benchmarks show 10-15% improvement at 192 FPs
- [ ] Profiling confirms malloc overhead <2%
- [ ] Allocation count reduced by ~50% (61K → 30K)
- [ ] `cargo clippy` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] Documentation is comprehensive
- [ ] TICKET-007 is unblocked and ready to run

---

## Success Metrics

**Technical**:
- ✅ Nested allocations reduced by 50% (61,440 → ~30,720)
- ✅ Malloc overhead reduced to <2%
- ✅ Backward pass 10-15% faster at production scale
- ✅ Numerical accuracy maintained (within 1e-10)

**Impact**:
- ✅ Combined with TICKET-006: ~92,000 allocations eliminated
- ✅ Production training runs 10-15% faster
- ✅ Memory behavior predictable and bounded

**Quality**:
- ✅ Zero unsafe code
- ✅ Thread-safe by design
- ✅ Pattern is reusable for forward pass (TICKET-008)
- ✅ All tests pass

---

## Notes

This ticket completes the backward pass optimization story. Combined with TICKET-006:
- **TICKET-006**: Eliminated outer allocations (unzip) → ~30K reallocations
- **TICKET-006b**: Eliminated nested allocations (coefficients, state) → ~61K allocations

**Total Impact**: ~92,000 allocations eliminated per training run at production scale, resulting in 10-15% backward pass improvement and <2% malloc overhead.

The pattern established here (thread-local pre-allocated buffers) will be reused in TICKET-008 (forward pass) and TICKET-009 (simulation) for consistent optimization across the codebase.
