# Master Plan: Memory Growth Prevention

## Status: 🟡 Partially Complete (2025-12-26)

### Implementation Status
- ✅ **Epic 1**: Solution/Basis buffer reuse - Complete
- ⬜ **Epic 2**: Cut cloning elimination - Not started (low impact ~9 MB)
- ✅ **Epic 3**: HashMap cloning elimination - Complete (dead code removed)
- ✅ **Epic 4**: Custom allocator (mimalloc) - Complete

### Key Findings
The original analysis overestimated transient allocation impact. After implementing buffer reuse:
- **Transient allocations eliminated**: Solution/Basis buffer-into pattern working
- **Dead code removed**: Unused HashMap clone eliminated
- **RSS still ~5 GB**: Due to legitimate memory usage, not transient allocations

### Root Cause Analysis (Revised)
The ~5 GB RSS is primarily due to:
1. **16 parallel handlers × 59 stages = 944 HiGHS models** (~2+ GB)
2. **Legitimate cut storage**: 5521 active cuts × 59 FCFs × ~1.3 KB (~420 MB)
3. **HiGHS internal structures**: Basis factorization, working memory
4. **Rust data structures**: Subproblems, realizations, state objects

This is architectural memory usage, NOT memory growth from transient allocations.

## Executive Summary

Implement the recommendations from `MEMORY_GROWTH_ANALYSIS.md` to eliminate memory allocation growth during SDDP training. The primary causes are transient allocations in the solver interface (~4.9 GB churn), FCF pool growth (~720 MB), and cut/HashMap cloning (~200 MB). This plan addresses all 6 priority areas to achieve near-zero memory growth per iteration.

## Goals & Non-Goals

### Goals

- **Eliminate Solution/Basis allocation churn**: Reuse buffers instead of allocating new vectors per solve (~4.9 GB → 0)
- **Eliminate transient cut cloning**: Use Arc or references instead of full clones (~80 MB → 0)
- **Eliminate HashMap cloning**: Restructure active cut indices tracking (~120 MB → 0)
- **Reduce RSS fragmentation**: Optional custom allocator to return memory to OS
- **Memory growth per iteration**: ≤5 MB (from ~375 MB currently)
- **Peak RSS**: <2 GB (from ~5 GB currently)

### Non-Goals (Explicit Scope Exclusions)

- HiGHS internal memory optimization (external library, out of scope)
- FCF cut pool growth (already optimized with reserve() in Epic 2)
- State pool preallocation (low impact, complex Box<dyn State> handling)
- Modifying HiGHS library source code

## Architecture Overview

### Current State

Per the MEMORY_GROWTH_ANALYSIS.md:

| Component | Per Iteration | Total (8 iter) | Type |
|-----------|--------------|----------------|------|
| Solution/Basis extractions | ~600 MB | ~4.9 GB | Transient (freed) |
| FCF Cut Pool growth | ~40 MB | ~320 MB | Retained |
| FCF State Pool growth | ~50 MB | ~400 MB | Retained |
| Cut cloning | ~10 MB | ~80 MB | Transient |
| HashMap cloning | ~15 MB | ~120 MB | Transient |

**Key Issue**: `Realization::with_capacity()` preallocates basis storage, but `realize_and_solve()` replaces it with newly allocated buffers, discarding the preallocation.

### Target State

- **Solution/Basis**: Buffers stored in `Realization`, passed as mutable references to `get_solution_into()` / `get_basis_into()`
- **Cut References**: Use `Arc<BendersCut>` for lock-free sharing without cloning
- **Active Cut Indices**: Store as `Vec<(cut_id, pool_index)>` or use snapshot instead of HashMap clone
- **Memory Profile**: Flat after initialization

### Key Design Decisions

1. **Buffer-into pattern for HiGHS**: Add `get_solution_into(&self, buf: &mut Solution)` that writes into existing buffer instead of allocating

2. **Arc-wrapped cuts in pool**: Store `Arc<BendersCut>` in cut pool, clone Arc (cheap) instead of cloning cut data (expensive)

3. **Snapshot-based cut indices**: Instead of cloning HashMap, capture cut IDs at iteration start as a simple Vec

4. **Optional global allocator**: mimalloc or jemalloc as feature flag for better memory return to OS

## Technical Approach

### Priority 1: Solution/Basis Buffer Preallocation (HIGH IMPACT)

**Location**: `src/solver.rs:622-672`

Add buffer-into variants:

```rust
pub fn get_solution_into(&self, solution: &mut Solution) {
    solution.ensure_capacity(self.num_cols(), self.num_rows());
    unsafe {
        Highs_getSolution(
            self.highs.unsafe_mut_ptr(),
            solution.colvalue.as_mut_ptr(),
            solution.coldual.as_mut_ptr(),
            solution.rowvalue.as_mut_ptr(),
            solution.rowdual.as_mut_ptr(),
        );
    }
}

pub fn get_basis_into(&self, basis: &mut Basis) {
    // Similar pattern with raw buffer reuse
}
```

**Fix in Realization**: Use stored buffers instead of replacing:

```rust
// Before (line 1583):
realization_container.basis = basis;  // Discards preallocated buffer!

// After:
solver.get_basis_into(&mut realization_container.basis);
```

### Priority 2: Cut Cloning Elimination

**Location**: `src/sddp/mod.rs:2184-2198`

Change cut pool to store Arc:

```rust
// Before:
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
}

// After:
pub struct BendersCutPool {
    pub pool: Vec<Arc<BendersCut>>,
}

// Usage - clone Arc, not data:
.map(|cut| (cut_id, Arc::clone(cut)))
```

### Priority 3: HashMap Cloning Elimination

**Location**: `src/sddp/mod.rs:2077-2091`

Option A - Use Vec snapshot:

```rust
// Before:
let active_cut_indices_before: HashMap<usize, usize> = 
    fcf_locked.cut_pool.active_cut_indices.clone();

// After - just collect keys:
let active_cut_ids_before: Vec<usize> = 
    fcf_locked.cut_pool.active_cut_indices.keys().copied().collect();
```

Option B - Add snapshot method to CutPool that returns owned Vec

### Priority 4-6: Allocator Optimizations (OPTIONAL)

Feature-gated custom allocator:

```toml
[features]
mimalloc = ["dep:mimalloc"]

[dependencies]
mimalloc = { version = "0.1", optional = true }
```

## Phases & Milestones

| Phase | Epic | Duration | Milestone |
|-------|------|----------|-----------|
| 1 | Solution/Basis Buffer Reuse | 1 week | ~4.9 GB allocation churn eliminated |
| 2 | Arc-Wrapped Cuts | 3-5 days | ~80 MB transient allocation eliminated |
| 3 | Active Cut Indices Optimization | 2-3 days | ~120 MB transient allocation eliminated |
| 4 | Custom Allocator (optional) | 1-2 days | Feature flag for mimalloc |

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Arc overhead impacts performance | Low | Medium | Benchmark before/after |
| Buffer sizing mismatch | Medium | Low | Add capacity checks, resize if needed |
| API breakage in solver.rs | Low | Medium | Add new methods, deprecate old |
| Numerical difference from buffer reuse | Very Low | High | Validate with examples |

## Success Metrics

- [ ] Peak RSS < 2 GB (currently ~5 GB)
- [ ] Memory growth per iteration < 5 MB (currently ~375 MB)
- [ ] Minor page faults < 500K (currently 3.4M)
- [ ] No regression in runtime performance
- [ ] Examples 01, 05, 07 produce identical results

## Validation Commands

```bash
# Memory profile during training
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep -E "Iteration|MEMORY"

# Peak RSS measurement
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Detailed allocation tracking
valgrind --tool=massif --detailed-freq=1 \
  ./target/release/powers run examples/01-deterministic
```

## References

1. `MEMORY_GROWTH_ANALYSIS.md` - Root cause analysis and recommendations
2. `src/solver.rs:622-672` - Solution/Basis allocation
3. `src/sddp/mod.rs:2184-2198` - Cut cloning
4. `src/sddp/mod.rs:2077-2091` - HashMap cloning
5. `plans/preallocation-refactoring/` - Related preallocation work
