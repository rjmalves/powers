# TICKET-005: Implement BackwardPassBuffers for pre-allocated backward pass execution

## Context

This is the first high-impact optimization ticket that applies the buffer management infrastructure to eliminate allocations in the backward pass hot path. The backward pass currently allocates ~60 vectors per iteration (for 10 forward passes × 5 stages). By pre-allocating result buffers and reusing them, we target 8-10% overall performance improvement.

**Why this matters**: Profiling shows the backward pass is executed thousands of times during training. Each execution allocates multiple vectors for cut-state pairs. These allocations account for a significant portion of the 5.28% malloc overhead. This ticket eliminates those allocations.

**Part of**: Performance Implementation Plan - Phase 2: Backward Pass Optimization

**Depends on**: Memory module infrastructure (TICKET-001, 002, 003)

## Acceptance Criteria

- [ ] Given SizingInfo, when BackwardPassBuffers is created, then buffers are pre-allocated for all forward passes
- [ ] Given pre-allocated buffers, when backward pass executes, then no new allocations occur (verified by profiler)
- [ ] Given backward pass buffers, when reused across iterations, then no data leakage occurs between iterations
- [ ] Given parallel backward step execution, when using buffers, then thread-safety is maintained
- [ ] Performance: Backward pass malloc overhead reduced by >50% (from ~2% to <1%)
- [ ] Performance: Backward pass execution time reduced by 8-10%
- [ ] Correctness: Numerical results match baseline (within 1e-10 tolerance)

## Tasks

### Implementation

- [ ] Create directory `src/sddp/backward_pass/` (new module structure)
- [ ] Create `src/sddp/backward_pass/mod.rs` with module documentation
- [ ] Create `src/sddp/backward_pass/buffers.rs`:
  - [ ] Implement `BackwardPassBuffers` struct
  - [ ] Add `results: BufferPool<CutStatePair>` field
  - [ ] Add `cuts_buffer: BufferPool<Cut>` field (if needed)
  - [ ] Add `states_buffer: BufferPool<Vec<f64>>` field (if needed)
  - [ ] Add `sizing: SizingInfo` field
  - [ ] Implement `new(sizing: &SizingInfo)` constructor
  - [ ] Implement `acquire_result_buffer(trajectory_idx)` method
- [ ] Define `CutStatePair` struct in buffers.rs:
  - [ ] Add `cut: Cut` field
  - [ ] Add `state: Vec<f64>` field
  - [ ] Implement necessary traits (Clone, Debug, Default)
- [ ] Update `src/sddp/mod.rs` to include backward_pass module:
  - [ ] Add `mod backward_pass;` declaration
  - [ ] Re-export if needed
- [ ] Add `BackwardPassBuffers` field to `SddpAlgorithm` struct in `src/sddp/mod.rs`
- [ ] Update `SddpAlgorithm::new()` to initialize BackwardPassBuffers:
  - [ ] Compute SizingInfo from configuration
  - [ ] Create BackwardPassBuffers instance
  - [ ] Store in struct
- [ ] Document buffer sizing calculations in constructor

### Testing

- [ ] Unit test: Create BackwardPassBuffers with realistic SizingInfo
  - [ ] Verify results pool has num_forward_passes buffers
  - [ ] Verify each buffer has capacity for num_stages-1 cuts
- [ ] Unit test: Acquire result buffer by index
  - [ ] Verify cycling behavior (index % buffer_count)
- [ ] Unit test: CutStatePair creation and manipulation
- [ ] Integration test: Create BackwardPassBuffers for 03-multistage example
  - [ ] Verify buffer dimensions match expected values
- [ ] Integration test: Create BackwardPassBuffers for large-scale example
  - [ ] Verify memory footprint is reasonable (<1GB for typical case)
- [ ] Integration test: Simulate backward pass buffer usage
  - [ ] Acquire buffers for 10 trajectories
  - [ ] Fill with dummy cut-state pairs
  - [ ] Verify independence (no cross-contamination)
- [ ] Memory test: Verify buffer pre-allocation
  - [ ] Track allocations during constructor
  - [ ] Verify all buffers allocated upfront
  - [ ] Track allocations during buffer acquisition
  - [ ] Verify zero allocations during acquire operations

### Documentation

- [ ] Add comprehensive module-level doc comment for `backward_pass/` module:
  - [ ] Purpose: Backward pass execution with pre-allocated buffers
  - [ ] Architecture: How buffers eliminate allocations
  - [ ] Usage pattern example
  - [ ] Performance impact
- [ ] Add doc comment for `BackwardPassBuffers`:
  - [ ] Purpose and motivation
  - [ ] Buffer organization (one per trajectory)
  - [ ] Usage example
  - [ ] Thread-safety notes
- [ ] Add doc comment for `CutStatePair`:
  - [ ] What it represents
  - [ ] Why it's needed (result of backward step)
- [ ] Document sizing calculations with inline comments
- [ ] Add performance notes about buffer reuse benefits
- [ ] Document relationship to SizingInfo

## Technical Notes

### Buffer Organization

**Structure**:
```
BackwardPassBuffers
├── results: BufferPool<CutStatePair>  // One buffer per forward pass
│   ├── Buffer 0: [CutStatePair; num_stages-1]
│   ├── Buffer 1: [CutStatePair; num_stages-1]
│   └── ...
└── sizing: SizingInfo
```

**Sizing Logic**:
- Number of buffers = `num_forward_passes` (one per trajectory)
- Buffer capacity = `num_stages - 1` (no cut at final stage)
- Total memory ≈ `num_forward_passes × num_stages × (cut_size + state_size)`

### CutStatePair Design

```rust
pub struct CutStatePair {
    pub cut: Cut,
    pub state: Vec<f64>,
}

impl Default for CutStatePair {
    fn default() -> Self {
        Self {
            cut: Cut::default(),
            state: Vec::new(),
        }
    }
}
```

**Note**: May need to implement `Clone` depending on usage pattern.

### Memory Footprint Estimation

For typical system (10 forward passes, 5 stages, 200-dim state):
- Cut size: ~2KB (intercept + 200 coefficients)
- State size: ~1.6KB (200 f64 values)
- Per buffer: ~(2KB + 1.6KB) × 4 cuts = ~15KB
- Total: ~15KB × 10 buffers = ~150KB

This is tiny compared to solver memory, so no concern about memory usage.

### Integration with SddpAlgorithm

**Current state** (conceptual):
```rust
pub struct SddpAlgorithm {
    // ... existing fields ...
}
```

**After this ticket**:
```rust
pub struct SddpAlgorithm {
    // ... existing fields ...
    backward_buffers: BackwardPassBuffers,
}
```

### Constructor Pattern

```rust
impl SddpAlgorithm {
    pub fn new(/* existing params */) -> Self {
        // Compute sizing info
        let sizing = SizingInfo::from_input(&system, &graph, &recourse, &config);
        
        // Pre-allocate buffers
        let backward_buffers = BackwardPassBuffers::new(&sizing);
        
        Self {
            // ... existing initialization ...
            backward_buffers,
        }
    }
}
```

### Buffer Acquisition Pattern

```rust
// In backward pass
for (idx, trajectory) in trajectories.iter().enumerate() {
    let buffer = self.backward_buffers.acquire_result_buffer(idx);
    buffer.clear(); // Reset from previous iteration
    
    // Compute cuts, write to buffer
    self.backward_step_to_buffer(trajectory, buffer)?;
}
```

### Edge Cases

- [ ] Zero forward passes (shouldn't happen, but handle gracefully)
- [ ] Single-stage problem (no backward pass needed)
- [ ] Very large forward pass count (>100)
- [ ] Buffer reuse across many iterations (10000+)

### Performance Validation Strategy

1. **Profile before**: Run `perf record` on backward pass, measure malloc %
2. **Apply optimization**: Add BackwardPassBuffers (next ticket will use them)
3. **Profile after**: Re-run `perf record`, verify malloc % reduction
4. **Benchmark**: Use criterion to measure backward pass time improvement

### Integration Points

- **Uses**: `SizingInfo` from memory module
- **Uses**: `BufferPool` from memory module
- **Used by**: TICKET-006 (backward pass refactoring uses these buffers)
- **Related**: `src/sddp/mod.rs` (backward_pass method)

### References

- See `PERFORMANCE_IMPLEMENTATION_PLAN.md` Section 2.1
- Current backward pass: `src/sddp/mod.rs` (backward_pass method)
- Profiling data: `PROFILING_ANALYSIS.md`

## Dependencies

- Blocked by: TICKET-001 (needs SizingInfo)
- Blocked by: TICKET-002 (needs BufferPool)
- Blocked by: TICKET-003 (needs memory module integration)
- Blocks: TICKET-006 (backward pass refactoring uses these buffers)
- Related: TICKET-004 (testing infrastructure validates these buffers)

## Estimated Effort

**3 story points** (2 days)

**Confidence**: High

**Breakdown**:
- Implementation: 1 day (straightforward buffer pool usage)
- Testing: 0.5 day (unit + integration tests)
- Documentation: 0.5 day (module and API docs)

## Validation Checklist

Before marking this ticket complete:

- [ ] `cargo test` passes all tests
- [ ] `cargo clippy` produces no warnings
- [ ] `cargo fmt --check` passes
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] All acceptance criteria met
- [ ] BackwardPassBuffers created successfully for all examples
- [ ] Memory footprint validated (reasonable size)
- [ ] Buffer acquisition works correctly
- [ ] Code reviewed by team member
- [ ] Integration with SddpAlgorithm compiles and runs

## Notes

**Important**: This ticket only creates the buffer infrastructure. The next ticket (TICKET-006) will refactor the actual backward pass to USE these buffers. This separation reduces risk and makes changes easier to review.

**Memory Safety**: All buffers are owned by BackwardPassBuffers and borrowed for use. No manual memory management required.

**Next Step**: After this ticket, TICKET-006 will modify the backward_pass() method to acquire and use these pre-allocated buffers instead of allocating fresh vectors.
