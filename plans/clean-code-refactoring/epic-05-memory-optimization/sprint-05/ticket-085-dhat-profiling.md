# [T-085] DHAT Profiling to Verify Zero Allocations in Hot Path

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 5](./00-sprint-overview.md)
> **Dependencies**: [T-080](./ticket-080-audit-add-row.md), [T-081](./ticket-081-thread-local-row-buffers.md), [T-082](./ticket-082-highs-warmup.md), [T-083](./ticket-083-coordinator-buffers.md), [T-084](./ticket-084-trajectory-buffers.md)
> **Blocks**: [T-086](./ticket-086-benchmark-memory.md)
> **Status**: ⏳ Ready for manual verification

---

## Context

### Background

After implementing all allocation elimination changes in T-080 through T-084, we need to verify that the hot path truly has zero allocations. DHAT (Dynamic Heap Analysis Tool) is part of Valgrind and provides detailed allocation profiling.

### Goal

Prove that after the warmup phase, SDDP training iterations perform **zero heap allocations** in:
- Cut computation (`evaluate_cut`, `evaluate_cut_ref`)
- Subproblem solving (`realize_and_solve`)
- Cut addition (`add_cut_with_preallocation`)
- Coordinator result collection
- State updates

---

## Specification

### Inputs

- Optimized build of powers
- Example case (e.g., example 05)
- DHAT instrumentation

### Outputs

- DHAT report showing allocation sites
- Analysis of remaining allocations (if any)
- Documentation of expected vs. actual behavior

### DHAT Commands

```bash
# Build release with debug info
cargo build --release

# Run with DHAT
valgrind --tool=dhat ./target/release/powers run examples/05-linear-model 2>&1 | tee dhat_output.txt

# View DHAT output
dh_view.html  # Open dhat.out.* file in browser
```

### Expected Results

After warmup phase, the only acceptable allocations are:
- Logging/tracing output (I/O buffers)
- JSON serialization for results (at training end)
- System allocator overhead (minimal)

### Hot Path Functions to Verify (Zero Allocations)

| Function | Location | Expected Allocations |
|----------|----------|---------------------|
| `evaluate_cut` | `state.rs` | 0 (uses thread-local buffers) |
| `evaluate_cut_ref` | `state.rs` | 0 (uses caller-provided buffers) |
| `realize_and_solve` | `subproblem.rs` | 0 (uses thread-local solution buffer) |
| `add_cut_with_preallocation` | `subproblem.rs` | 0 (modifies preallocated slots) |
| `compute_cuts_parallel_into_slots` | `coordinator.rs` | 0 (uses preallocated buffers) |
| `try_add_row` | `solver.rs` | 0 (uses thread-local buffers) |

---

## Acceptance Criteria

- [ ] DHAT profiling run completed on example 05
- [ ] Analysis document created with findings
- [ ] All hot path functions verified to have zero allocations
- [ ] If allocations found, document location and propose fix
- [ ] Baseline DHAT report saved for future regression testing

---

## Implementation Guide

### Suggested Approach

1. **Build with debug info**:
   ```bash
   RUSTFLAGS="-g" cargo build --release
   ```

2. **Run DHAT on example 05**:
   ```bash
   valgrind --tool=dhat --dhat-out-file=dhat.out ./target/release/powers run examples/05-linear-model
   ```

3. **Analyze DHAT output**:
   - Look for allocations in `powers` namespace
   - Filter out allocations from initialization phase
   - Focus on allocations that occur N × iterations times

4. **Create allocation report**:
   ```markdown
   ## DHAT Analysis Report
   
   ### Summary
   - Total allocations: X
   - Training phase allocations: Y
   - Hot path allocations: Z (target: 0)
   
   ### Top Allocation Sites
   1. [location]: N allocations, M bytes
   2. ...
   
   ### Hot Path Verification
   | Function | Allocations | Status |
   |----------|-------------|--------|
   | evaluate_cut | 0 | ✅ |
   | ... | ... | ... |
   ```

5. **If allocations found**:
   - Identify root cause
   - Create follow-up ticket if fix is non-trivial
   - Document workaround if needed

### Alternative: Custom Allocator Tracking

If DHAT overhead is too high, use a tracking allocator:

```rust
#[cfg(feature = "track_allocs")]
use std::alloc::{GlobalAlloc, Layout, System};

#[cfg(feature = "track_allocs")]
struct TrackingAllocator {
    inner: System,
}

#[cfg(feature = "track_allocs")]
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // Increment counter
        // Log backtrace if in hot path
        self.inner.alloc(layout)
    }
    // ...
}
```

### Key Files to Review

- DHAT output file (`dhat.out.*`)
- Browser-based DHAT viewer for detailed analysis

### Pitfalls to Avoid

- ⚠️ DHAT adds significant overhead - use smaller examples if needed
- ⚠️ Debug builds may have different allocation patterns than release
- ⚠️ Thread-local initialization counts as allocation (expected, once per thread)

---

## Testing Requirements

### Verification Tests

- [ ] DHAT run completes without errors
- [ ] Hot path functions show zero allocations after warmup
- [ ] Allocation count matches expected behavior

### Regression Baseline

- [ ] Save DHAT report as baseline for future comparisons
- [ ] Document expected allocation count for reference

---

## Documentation Requirements

- [ ] Create `docs/MEMORY_PROFILING.md` with DHAT instructions
- [ ] Document expected allocation sites and their justification
- [ ] Add DHAT analysis results to sprint documentation

---

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Profiling and analysis work, straightforward execution.

---

## Definition of Done

- [ ] DHAT profiling completed
- [ ] Analysis report created
- [ ] Hot path verified to have zero allocations
- [ ] Any remaining allocations documented and justified
- [ ] Baseline saved for regression testing
- [ ] Documentation updated
