# Master Plan: Memory Stability via Index-Based Pool Preallocation

## Executive Summary

Implement complete memory preallocation for SDDP training to achieve zero allocation during the training loop. The key insight is that SDDP with cut selection is fundamentally a **bounded memory algorithm**—after reading input data, we know the exact number of cuts and states that will be created. By preallocating all `BendersCut` and `State` instances at training start and updating them in place using `(iteration, forward_pass_idx)` as the access key, we can eliminate ~290 MB of allocations during training.

## Goals & Non-Goals

### Goals

- **Zero cut allocation during training**: Preallocate all `BendersCut` instances with coefficient vectors
- **Zero state allocation during training**: Preallocate all `State` instances with coefficient vectors
- **Eliminate cut cloning**: Use `Arc<BendersCut>` for handler application instead of full clones
- **Memory growth per iteration**: < 10 MB (from ~375 MB currently)
- **Peak RSS (example 05)**: < 3 GB (from ~5 GB currently)
- **Algorithm correctness**: Identical lower bounds to baseline

### Non-Goals (Explicit Scope Exclusions)

- HiGHS internal memory optimization (external library, partially addressed via preallocation)
- HashMap entry allocation (minor impact, ~0.5 MB/iter)
- Allocator-level fragmentation (addressed separately in memory-growth-prevention plan)
- Modifying the cut selection algorithm logic

## Architecture Overview

### Current State

From `MEMORY_STABILITY_ANALYSIS.md`:

| Component | Per Iteration | Over 8 Iter | Type |
|-----------|---------------|-------------|------|
| BendersCut objects | ~1.2 MB | ~10 MB | Heap alloc |
| Box<dyn State> objects | ~1.4 MB | ~11 MB | Heap alloc |
| Cut coefficient Vecs | ~12 MB | ~96 MB | Nested alloc |
| State coefficient Vecs | ~12 MB | ~96 MB | Nested alloc |
| Cut cloning | ~10 MB | ~80 MB | Transient |
| **Total Rust allocations** | ~36 MB | ~293 MB | |

**Current behavior**:
- `Vec::reserve()` preallocates pointer arrays but not the actual objects
- Each `BendersCut::new()` allocates a new object with its coefficient Vec
- Each `Box::new(state)` allocates a new Box with its coefficient Vec
- Cuts are cloned when passed to handlers for parallel application

### Target State

| Component | At Init | During Training | Type |
|-----------|---------|-----------------|------|
| BendersCut objects | ~21 MB | 0 | Preallocated |
| State objects | ~22 MB | 0 | Preallocated |
| Cut cloning | 0 | ~1 MB (Arc) | Cheap |
| **Total** | ~43 MB | ~1 MB | |

**Key design change**: Use `(iteration, forward_pass_idx)` as deterministic slot index:

```rust
slot = (iteration - 1) * num_forward_passes + forward_pass_idx
```

### Key Design Decisions

1. **Slot-based access**: Use deterministic slot computation instead of `push()` to enable preallocation

2. **In-place updates**: Add `update()` methods to modify preallocated cuts/states without allocation

3. **Arc for cut sharing**: Store `Arc<BendersCut>` in pool for cheap cloning to handlers

4. **Trait extensions**: Add `update_coefficients()` and `reset_to_zero()` to `State` trait

## Technical Approach

### Core Abstraction: Slot Computation

```rust
/// Compute slot index for (iteration, forward_pass_idx) pair.
#[inline]
fn compute_slot(iteration: usize, forward_pass_idx: usize, num_forward_passes: usize) -> usize {
    debug_assert!(iteration >= 1);
    debug_assert!(forward_pass_idx < num_forward_passes);
    (iteration - 1) * num_forward_passes + forward_pass_idx
}
```

### Data Flow Changes

**Current flow**:
```
backward_pass() → BendersCut::new() → fcf.add_cut(cut)  // Allocation
```

**New flow**:
```
backward_pass() → fcf.cut_pool.update_cut(iter, fp_idx, coeffs, rhs)  // No allocation
```

### Parallelism Strategy

- Cuts/states are computed in parallel forward passes
- Each forward pass has unique `forward_pass_idx` → unique slot
- No lock contention on slot access (each slot accessed by one thread)
- FCF lock only held during `add_cuts_batch()` → minimal contention

### Performance Strategy

- All allocations moved to initialization phase
- `copy_from_slice()` for coefficient updates (cache-friendly)
- Arc cloning for handler application (~16 bytes vs ~1 KB per cut)

## Phases & Milestones

| Phase | Epic | Duration | Milestone | Status |
|-------|------|----------|-----------|--------|
| 1 | Preallocated Cut Pool | 1 week | Zero cut allocation during training | ✅ COMPLETE |
| 2 | Preallocated State Pool | 1 week | Zero state allocation during training | ✅ COMPLETE |
| 3 | Cut Cloning Elimination | 3 days | Arc-based cut sharing | ✅ COMPLETE |
| 4 | Validation & Profiling | 3 days | Memory stability verified | 🔲 Not Started |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Slot computation errors | Low | High | Extensive unit tests, debug_assert! |
| State trait changes break implementations | Medium | Medium | Careful trait extension, test coverage |
| Arc overhead in hot path | Low | Low | Benchmark before/after |
| Existing tests fail | Medium | Medium | Run full test suite after each change |

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Memory growth/iteration | ~375 MB | < 10 MB |
| Peak RSS (example 05) | ~5 GB | < 3 GB |
| Allocations during training | ~36 MB/iter | < 1 MB/iter |
| Algorithm correctness | Baseline | Identical lower bounds |

## Verification Commands

```bash
# Memory profile during training
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Compare lower bounds before/after
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | grep "lower"

# Run full test suite
cargo test --release
```

## References

1. `MEMORY_STABILITY_ANALYSIS.md` - Root cause analysis and proposed solution
2. `MEMORY_GROWTH_ANALYSIS.md` - Original memory growth analysis
3. `plans/memory-growth-prevention/` - Related buffer reuse implementation
4. `src/fcf.rs` - FutureCostFunction and cut pool implementation
5. `src/state.rs` - State trait and implementations
6. `src/cut.rs` - BendersCut structure
7. `src/sddp/mod.rs` - Training loop and cut addition
