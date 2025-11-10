# TICKET-008: Implement ForwardPassBuffers and optimize forward pass execution

## Context

This ticket applies buffer pre-allocation to the forward pass, eliminating allocations in trajectory construction and state storage. The forward pass executes multiple times per iteration (typically 10-50 forward passes), and each currently allocates trajectory and state vectors. By pre-allocating and reusing buffers, we target 5-7% additional performance improvement on top of Phase 2's gains.

**Why this matters**: The forward pass accounts for ~20% of training time. While not as hot as the backward pass, it still executes thousands of times and contributes to allocation overhead. This is the second-highest impact optimization after backward pass.

**Part of**: Performance Implementation Plan - Phase 3: Forward Pass and Subproblem Optimization

**Depends on**: Memory module (TICKET-001, 002, 003) and Phase 2 completion (for validation)

## Acceptance Criteria

- [ ] Given SizingInfo, when ForwardPassBuffers is created, then trajectory buffers are pre-allocated for all forward passes
- [ ] Given forward pass execution, when profiled, then zero allocations occur in trajectory construction
- [ ] Given forward pass with pre-allocated buffers, when compared to baseline, then execution time is 5-8% faster
- [ ] Given buffer reuse across iterations, when validated, then no data leakage occurs
- [ ] Performance: Forward pass malloc overhead reduced by >50%
- [ ] Correctness: Numerical results identical to baseline (within 1e-10)

## Tasks

### Implementation

- [ ] Create `src/sddp/forward_pass/` module directory
- [ ] Create `src/sddp/forward_pass/mod.rs` with module documentation
- [ ] Create `src/sddp/forward_pass/buffers.rs`:
  - [ ] Implement `ForwardPassBuffers` struct
  - [ ] Add `trajectory_pool: BufferPool<Trajectory>` field
  - [ ] Add `stage_states: BufferPool<Vec<f64>>` field
  - [ ] Add `sizing: SizingInfo` field
  - [ ] Implement `new(sizing: &SizingInfo)` constructor
  - [ ] Implement `acquire_trajectory_buffer(pass_idx: usize)` method
  - [ ] Implement `acquire_state_buffer(pass_idx: usize, stage: usize)` method
- [ ] Add `ForwardPassBuffers` field to `SddpAlgorithm` struct
- [ ] Update `SddpAlgorithm::new()` to initialize ForwardPassBuffers:
  - [ ] Use existing SizingInfo
  - [ ] Create ForwardPassBuffers instance
  - [ ] Store in algorithm struct
- [ ] Refactor `forward_pass()` method in `src/sddp/mod.rs`:
  - [ ] Acquire trajectory buffer for each pass
  - [ ] Clear buffer at start of each pass
  - [ ] Write trajectory data directly to buffer
  - [ ] Return buffer references instead of allocating new Vec
  - [ ] Add PERFORMANCE comments explaining optimization
- [ ] Create `build_trajectory_to_buffer()` helper method:
  - [ ] Takes trajectory buffer as mutable reference
  - [ ] Writes stages directly to buffer
  - [ ] Returns Result<()> instead of Trajectory
- [ ] Update stage-by-stage forward simulation:
  - [ ] Acquire state buffer for current stage
  - [ ] Write state directly to buffer
  - [ ] Reuse buffer for next stage

### Testing

- [ ] Unit test: Create ForwardPassBuffers with realistic SizingInfo
  - [ ] Verify trajectory pool has num_forward_passes buffers
  - [ ] Verify each buffer has capacity for num_stages
- [ ] Unit test: Acquire trajectory and state buffers
  - [ ] Verify correct buffer returned for given index
  - [ ] Verify cycling behavior
- [ ] Integration test: Forward pass with buffers on 03-multistage
  - [ ] Run multiple forward passes
  - [ ] Verify trajectories are constructed correctly
  - [ ] Compare with baseline results
- [ ] Integration test: Forward pass with buffers on large-scale example
  - [ ] Verify memory footprint is reasonable
  - [ ] Verify no memory leaks
- [ ] Correctness test: Numerical validation
  - [ ] Run 20 iterations with buffers
  - [ ] Compare trajectories with baseline
  - [ ] Verify states match exactly (tolerance 1e-10)
- [ ] Performance test: Allocation tracking
  - [ ] Profile forward pass with massif
  - [ ] Verify zero allocations in trajectory loop
  - [ ] Compare allocation count with baseline
- [ ] Performance test: Execution time
  - [ ] Benchmark forward pass on realistic example
  - [ ] Measure time for 100 forward passes
  - [ ] Verify >5% improvement over baseline
- [ ] Stress test: Buffer reuse over 1000 iterations
  - [ ] Verify no data leakage between passes
  - [ ] Verify results remain correct

### Documentation

- [ ] Add comprehensive module-level doc for `forward_pass/`:
  - [ ] Purpose: Forward pass with pre-allocated buffers
  - [ ] Architecture: Buffer organization and reuse
  - [ ] Usage pattern example
  - [ ] Performance impact notes
- [ ] Add doc comment for `ForwardPassBuffers`:
  - [ ] Purpose and motivation
  - [ ] Buffer organization (trajectory and state)
  - [ ] Usage example
  - [ ] Memory layout notes
- [ ] Update `forward_pass()` doc comment:
  - [ ] Note about pre-allocated buffers
  - [ ] Performance characteristics
  - [ ] Zero-allocation guarantee
- [ ] Add PERFORMANCE comments in code:
  - [ ] Explain buffer acquisition
  - [ ] Explain trajectory construction to buffer
  - [ ] Explain state buffer reuse
- [ ] Update module-level docs for `sddp/`:
  - [ ] Add forward pass optimization notes
  - [ ] Link to memory module

## Technical Notes

### Buffer Organization

**Structure**:
```
ForwardPassBuffers
├── trajectory_pool: BufferPool<Trajectory>
│   ├── Buffer 0: Trajectory with num_stages capacity
│   ├── Buffer 1: Trajectory with num_stages capacity
│   └── ...
├── stage_states: BufferPool<Vec<f64>>
│   ├── Buffer 0: [f64; state_dimension]
│   ├── Buffer 1: [f64; state_dimension]
│   └── ...
└── sizing: SizingInfo
```

**Sizing Logic**:
- Trajectory buffers: `num_forward_passes` buffers, each with `num_stages` capacity
- State buffers: `num_forward_passes * num_stages` buffers, each with `state_dimension` capacity
- Total memory: ~`num_forward_passes × num_stages × state_dimension × 8` bytes

### Trajectory Construction Pattern

**Before** (allocating):
```rust
pub fn forward_pass(&mut self) -> Result<Vec<Trajectory>> {
    let mut trajectories = Vec::new();  // ALLOCATION
    
    for pass_idx in 0..self.num_forward_passes {
        let mut trajectory = Trajectory::new();  // ALLOCATION
        
        for stage in 0..self.num_stages {
            let state = self.simulate_stage(stage)?;  // ALLOCATION
            trajectory.push_stage(state);  // Growing vector
        }
        
        trajectories.push(trajectory);
    }
    
    Ok(trajectories)
}
```

**After** (buffer reuse):
```rust
pub fn forward_pass(&mut self) -> Result<Vec<Trajectory>> {
    // PERFORMANCE: Use pre-allocated trajectory buffers instead of allocating
    // per pass. Eliminates ~50 allocations per iteration (10 passes × 5 stages).
    let mut trajectories = Vec::with_capacity(self.num_forward_passes);
    
    for pass_idx in 0..self.num_forward_passes {
        // Acquire pre-allocated trajectory buffer
        let traj_buffer = self.forward_buffers.trajectory_pool.acquire(pass_idx);
        traj_buffer.clear();
        
        // Build trajectory in buffer (zero allocation)
        self.build_trajectory_to_buffer(pass_idx, traj_buffer)?;
        
        trajectories.push(traj_buffer.clone());  // Minimal clone for return
    }
    
    Ok(trajectories)
}

fn build_trajectory_to_buffer(
    &mut self,
    pass_idx: usize,
    trajectory: &mut Buffer<TrajectoryStage>,
) -> Result<()> {
    for stage in 0..self.num_stages {
        // Acquire state buffer
        let state_buffer = self.forward_buffers.acquire_state_buffer(pass_idx, stage);
        state_buffer.clear();
        
        // Simulate stage, write to buffer
        self.simulate_stage_to_buffer(stage, state_buffer)?;
        
        // Add stage to trajectory (no allocation, buffer already sized)
        trajectory.push(TrajectoryStage {
            stage,
            state: state_buffer.as_slice().to_vec(),  // Minimal copy
            objective: self.current_objective,
        });
    }
    
    Ok(())
}
```

### Memory Footprint Estimation

For typical system (10 forward passes, 5 stages, 200-dim state):
- Trajectory buffers: 10 × 5 × (stage metadata) ≈ 10KB
- State buffers: 10 × 5 × 200 × 8 = 80KB
- Total: ~90KB (negligible)

### Alternative Approach: Trajectory Pool

Consider if `Trajectory` should be stored by reference vs cloned:
- **Option A**: Return `Vec<Trajectory>` (requires final clone)
- **Option B**: Return `Vec<&Trajectory>` (requires lifetime management)
- **Recommendation**: Start with Option A (simpler), optimize to Option B if needed

### Integration with Backward Pass

Forward pass produces trajectories consumed by backward pass:
- Forward pass returns `Vec<Trajectory>`
- Backward pass takes `&[Trajectory]`
- Buffer management is internal to each pass (no cross-pass buffer sharing)

### Edge Cases

- [ ] Single forward pass (pool size = 1)
- [ ] Very large forward pass count (>100)
- [ ] Single-stage problem (trajectory buffer size = 1)
- [ ] Buffer reuse across many iterations (10000+)

### Performance Validation Strategy

1. **Profile before**: Baseline forward pass malloc overhead
2. **Implement optimization**: Add ForwardPassBuffers
3. **Profile after**: Measure malloc reduction
4. **Benchmark**: Compare forward pass time
5. **Target**: 5-8% improvement + 50% malloc reduction

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 3.1
- Current forward pass: `src/sddp/mod.rs`
- Buffer infrastructure: TICKET-002

## Dependencies

- Blocked by: TICKET-001, 002, 003 (memory module)
- Blocked by: TICKET-007 (Phase 2 validation complete)
- Blocks: TICKET-010 (Vec::with_capacity audit)
- Related: TICKET-009 (subproblem buffers, can work in parallel)

## Estimated Effort

**5 story points** (3 days)

**Confidence**: High

**Breakdown**:
- Implementation: 1.5 days (similar to backward pass pattern)
- Testing: 1 day (correctness + performance validation)
- Documentation: 0.5 day (module and API docs)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests
- [ ] Numerical results verified identical to baseline
- [ ] Allocation profiling shows zero allocations in forward pass loop
- [ ] Performance benchmarks show >5% improvement
- [ ] Stress test (1000 iterations) passes
- [ ] No memory leaks detected
- [ ] `cargo clippy` produces no new warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc` builds without warnings
- [ ] Code reviewed by team member
- [ ] Performance improvement measured and documented

## Notes

**Parallel Work**: This ticket can be worked on in parallel with TICKET-009 (SubproblemBuffers) since they touch different parts of the codebase.

**Trajectory Cloning**: The final `trajectories.push(traj_buffer.clone())` is acceptable—it's one allocation per forward pass (10 total), compared to ~50 allocations before optimization. If this becomes a bottleneck, we can optimize further with lifetime management.

**Testing Emphasis**: Since forward pass is less hot than backward pass, testing is even more critical to ensure the optimization is correct and worth the added complexity.
