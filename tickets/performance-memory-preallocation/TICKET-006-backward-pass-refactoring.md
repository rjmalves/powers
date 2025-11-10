# TICKET-006: Refactor backward_pass() to use pre-allocated buffers

## Context

This is the critical ticket that delivers the 8-10% performance improvement by eliminating allocations in the backward pass. The current `backward_pass()` method allocates vectors for each trajectory and cut-state pair. This ticket refactors it to write directly to pre-allocated buffers created in TICKET-005.

**Why this matters**: This is where we actually realize the performance gains. The backward pass runs thousands of times during training, and each run currently triggers ~60 allocations. By reusing pre-allocated buffers, we eliminate this overhead entirely.

**Part of**: Performance Implementation Plan - Phase 2: Backward Pass Optimization

**Depends on**: TICKET-005 (BackwardPassBuffers infrastructure must exist)

## Acceptance Criteria

- [ ] Given training execution, when backward pass runs, then zero allocations occur (verified by profiler)
- [ ] Given backward pass results, when compared to baseline, then numerical results are identical (within 1e-10)
- [ ] Given 8-iteration training run, when profiled, then malloc overhead is reduced by >50% (from ~2% to <1%)
- [ ] Given 8-iteration training run, when timed, then execution time is 8-10% faster than baseline
- [ ] Given parallel backward execution, when multiple trajectories processed, then thread-safety is maintained
- [ ] All existing tests pass with no behavior changes
- [ ] No performance regressions in other parts of code

## Tasks

### Implementation

- [ ] Backup current `backward_pass()` implementation:
  - [ ] Copy to `backward_pass_original()` for A/B testing
  - [ ] Or save in git branch
- [ ] Refactor `backward_pass()` method in `src/sddp/mod.rs`:
  - [ ] Change return type if needed (or keep Vec<Cut>)
  - [ ] Add code to acquire buffers from `self.backward_buffers`
  - [ ] Replace `Vec::new()` allocations with buffer acquisition
  - [ ] Clear buffers at start of each iteration
  - [ ] Write results directly to buffers instead of temporary vectors
  - [ ] Extract cuts from buffers at end (minimal final allocation)
- [ ] Create new method `backward_step_to_buffer()`:
  - [ ] Takes trajectory and mutable buffer reference
  - [ ] Writes cut-state pairs directly to buffer
  - [ ] Returns Result<()> instead of Vec
- [ ] Update parallel backward step execution:
  - [ ] Each parallel task gets its own buffer (via trajectory index)
  - [ ] Write results to buffer instead of collecting into Vec
  - [ ] Use buffer index matching trajectory index
- [ ] Add PERFORMANCE comments explaining optimization:
  - [ ] Why buffers are used
  - [ ] What allocations were eliminated
  - [ ] Expected performance impact
- [ ] Verify all edge cases still handled:
  - [ ] Empty trajectories
  - [ ] Single-stage problems
  - [ ] Error handling paths

### Testing

- [ ] Correctness test: Compare results with baseline
  - [ ] Run training on 03-multistage example
  - [ ] Record cuts from original implementation
  - [ ] Record cuts from optimized implementation
  - [ ] Verify cuts are numerically identical (diff < 1e-10)
  - [ ] Verify convergence behavior is identical
- [ ] Correctness test: Run full test suite
  - [ ] All existing tests should pass
  - [ ] No changes to test expectations needed
- [ ] Correctness test: Backward pass with various configurations
  - [ ] Different numbers of forward passes (1, 10, 50)
  - [ ] Different stage counts (3, 5, 10)
  - [ ] Different system sizes (small, medium, large)
- [ ] Integration test: Multi-iteration training
  - [ ] Run 20 iterations
  - [ ] Verify no data leakage between iterations
  - [ ] Verify convergence path is unchanged
- [ ] Performance test: Memory allocation tracking
  - [ ] Run with allocation profiler (massif or custom)
  - [ ] Measure allocations in backward pass
  - [ ] Verify zero allocations in hot path
  - [ ] Compare allocation count before/after
- [ ] Performance test: Execution time measurement
  - [ ] Benchmark backward pass on realistic example
  - [ ] Measure time for 100 backward passes
  - [ ] Compare with original implementation
  - [ ] Verify >8% improvement
- [ ] Stress test: Buffer reuse over many iterations
  - [ ] Run 1000 iterations
  - [ ] Verify no memory leaks
  - [ ] Verify results remain correct
- [ ] Parallel test: Thread-safety verification
  - [ ] Run with ThreadSanitizer if available
  - [ ] Run with 100 forward passes (stress parallel execution)
  - [ ] Verify no data races

### Documentation

- [ ] Update backward_pass() doc comment:
  - [ ] Note about pre-allocated buffers
  - [ ] Performance characteristics
  - [ ] Mention zero-allocation guarantee
- [ ] Add inline PERFORMANCE comments:
  - [ ] Before buffer acquisition: explain why
  - [ ] Before buffer clear: explain reuse
  - [ ] At result collection: explain minimal allocation
- [ ] Update module-level docs for `sddp/` module:
  - [ ] Mention buffer-based optimization
  - [ ] Link to memory module
- [ ] Add entry to CHANGELOG.md:
  - [ ] "Performance: Backward pass now uses pre-allocated buffers"
  - [ ] "Improvement: 8-10% faster training execution"
- [ ] Update PERFORMANCE_REFACTORING_PLAN.md:
  - [ ] Mark Phase 2.1 as complete
  - [ ] Record actual performance improvements
  - [ ] Update metrics table

## Technical Notes

### Refactoring Pattern

**Before** (allocating):
```rust
pub fn backward_pass(&mut self) -> Result<Vec<Cut>> {
    let trajectories = self.sample_trajectories()?;
    
    let results: Vec<CutStatePair> = trajectories  // ALLOCATION
        .par_iter()
        .flat_map(|trajectory| {
            self.backward_step(trajectory) // Returns Vec (ALLOCATION)
        })
        .collect();
    
    let mut cuts = Vec::new();  // ALLOCATION
    for (cut, _state) in results {
        cuts.push(cut);
    }
    Ok(cuts)
}

fn backward_step(&self, trajectory: &Trajectory) -> Vec<CutStatePair> {
    let mut results = Vec::new();  // ALLOCATION per trajectory
    for stage in (1..self.num_stages).rev() {
        let cut = self.compute_cut_at_stage(trajectory, stage)?;
        let state = self.extract_state(trajectory, stage)?;
        results.push((cut, state));  // Growing vector
    }
    results
}
```

**After** (buffer reuse):
```rust
pub fn backward_pass(&mut self) -> Result<Vec<Cut>> {
    let trajectories = self.sample_trajectories()?;
    
    // PERFORMANCE: Use pre-allocated buffers instead of allocating per trajectory.
    // This eliminates ~60 allocations per iteration for typical problems (10 forward
    // passes × 5 stages). Profiling showed this reduced malloc overhead from 2% to <1%.
    let results: Vec<&[CutStatePair]> = trajectories
        .par_iter()
        .enumerate()
        .map(|(idx, trajectory)| {
            // Acquire pre-allocated buffer for this trajectory
            let buffer = &mut self.backward_buffers.acquire_result_buffer(idx);
            buffer.clear(); // Reset from previous iteration
            
            // Compute cuts and write directly to buffer (zero allocation)
            self.backward_step_to_buffer(trajectory, buffer)?;
            
            Ok(buffer.as_slice())
        })
        .collect::<Result<Vec<_>>>()?;
    
    // Extract cuts from buffers (minimal final allocation with known size)
    let total_cuts: usize = results.iter().map(|r| r.len()).sum();
    let mut cuts = Vec::with_capacity(total_cuts);  // Single allocation, correct size
    
    for result in results {
        for cut_state_pair in result {
            cuts.push(cut_state_pair.cut.clone());
        }
    }
    
    Ok(cuts)
}

/// Backward step that writes to pre-allocated buffer (zero allocation).
fn backward_step_to_buffer(
    &self,
    trajectory: &Trajectory,
    buffer: &mut Buffer<CutStatePair>,
) -> Result<()> {
    // Write cuts directly to buffer instead of allocating Vec
    for stage in (1..self.num_stages).rev() {
        let cut = self.compute_cut_at_stage(trajectory, stage)?;
        let state = self.extract_state(trajectory, stage)?;
        
        buffer.push(CutStatePair { cut, state });
    }
    
    Ok(())
}
```

### Key Changes

1. **Buffer acquisition**: Get pre-allocated buffer by trajectory index
2. **Direct writes**: Write to buffer instead of building temporary Vec
3. **Clear between uses**: Reset buffer at start of each iteration
4. **Minimal final allocation**: Only allocate result Vec with exact size

### Performance Impact Breakdown

**Allocations eliminated per iteration**:
- 1 trajectory results vector
- `num_forward_passes` backward step vectors (e.g., 10)
- `num_forward_passes × num_stages` cut allocations (e.g., 50)
- Growing vector reallocations (unknown count)

**Total**: ~60+ allocations per iteration → 1 final allocation

**Expected speedup**: 8-10% overall (backward pass is ~25% of runtime, 30% faster backward = 7.5% overall)

### Correctness Validation Strategy

**Critical**: Results must be numerically identical to baseline.

1. **Golden test**: Save baseline results for 03-multistage example
2. **Numerical comparison**: Compare cut coefficients element-wise (tolerance 1e-10)
3. **Convergence test**: Verify convergence path unchanged (same number of iterations)
4. **Statistical test**: Compare final bounds (should be identical)

### Error Handling

Maintain existing error handling:
- If `compute_cut_at_stage()` fails, propagate error
- If buffer is full (shouldn't happen), panic with clear message
- If parallel task fails, propagate through Result

### Parallel Execution

**Thread safety considerations**:
- Each trajectory gets its own buffer (by index)
- No shared mutable state
- Buffers owned by SddpAlgorithm (not moved)
- Safe parallel access via buffer index

### Memory Safety

**Lifetime considerations**:
- Buffers outlive backward pass execution
- Buffer references don't escape function
- Clear buffers before use (no stale data)

### Rollback Plan

If optimization causes issues:
1. Keep original implementation as `backward_pass_original()`
2. Add feature flag to switch implementations
3. Can revert easily via git

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 2.2
- Current implementation: `src/sddp/mod.rs` backward_pass()
- Buffer infrastructure: TICKET-005

## Dependencies

- Blocked by: TICKET-005 (BackwardPassBuffers must exist)
- Blocks: TICKET-007 (performance validation needs this complete)
- Related: TICKET-006 (testing validates correctness)

## Estimated Effort

**5 story points** (3 days)

**Confidence**: Medium (hot path modification, needs careful validation)

**Breakdown**:
- Implementation: 1 day (refactoring is straightforward but needs care)
- Testing: 1.5 days (extensive correctness and performance validation)
- Documentation: 0.5 day (performance comments and changelog)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all existing tests
- [ ] Numerical results verified identical to baseline
- [ ] Allocation profiling shows zero allocations in hot path
- [ ] Performance benchmarks show >8% improvement
- [ ] Parallel execution verified safe (ThreadSanitizer)
- [ ] Stress test (1000 iterations) passes
- [ ] No memory leaks detected
- [ ] `cargo clippy` produces no new warnings
- [ ] `cargo fmt --check` passes
- [ ] Code reviewed by team member
- [ ] Performance improvement documented in CHANGELOG.md
- [ ] Metrics updated in PERFORMANCE_REFACTORING_PLAN.md

## Notes

**Critical**: This is a hot path optimization. Test thoroughly before merging:
1. Correctness is paramount (verify numerical results)
2. Performance gains must be measured (don't trust assumptions)
3. Thread safety must be verified (parallel execution)
4. Memory safety is guaranteed by Rust, but logic bugs are possible

**Review Focus**:
- Buffer lifecycle (acquire → use → clear → reuse)
- Parallel execution safety
- Numerical correctness
- Error handling preservation

**Success Criteria**:
- Zero allocations in backward pass (profiler confirms)
- 8-10% faster training (benchmark confirms)
- Numerically identical results (tests confirm)
