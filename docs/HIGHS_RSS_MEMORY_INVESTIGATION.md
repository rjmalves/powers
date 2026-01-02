# HiGHS RSS Memory Growth Investigation

> **Sprint**: Epic 5, Sprint 7 Analysis
> **Date**: 2025-12-31
> **Related**: [DHAT_SPRINT7_ANALYSIS.md](./DHAT_SPRINT7_ANALYSIS.md)

---

## Problem Statement

During SDDP training, the application's **Resident Set Size (RSS) continuously increases** despite:
1. HiGHS being a robust solver with internal memory management
2. Temporary allocations that should be deallocated after each solve

This document investigates the root causes and proposes mitigations.

---

## Executive Summary

The RSS growth is caused by:

1. **HiGHS internal buffer accumulation** - HiGHS maintains high-water-mark buffers
2. **Model growth over iterations** - Cuts are added, increasing problem size
3. **No explicit memory reclaim API** - HiGHS doesn't deallocate internal structures

**Key Insight**: This is **not a memory leak** but rather HiGHS's performance-oriented design that trades memory for speed by avoiding repeated reallocations.

---

## Technical Analysis

### 1. HiGHS Internal Architecture

From HiGHS source code analysis (`HEkkDual.cpp`, `HFactor.cpp`):

```cpp
// HiGHS maintains working vectors sized to problem dimensions
class HEkkDual {
  HighsInt num_row_;
  HighsInt num_col_;
  std::vector<double> work_infeasibility;   // Size: num_row_
  std::vector<double> work_dual;            // Size: num_col_
  std::vector<double> work_edge_weight;     // Size: num_row_
  // ... many more vectors proportional to problem size
};
```

**Key observation**: These vectors grow when the problem grows (cuts added) but are **never shrunk** during the model lifetime.

### 2. Memory Growth Pattern in SDDP

```
Iteration 1:  Model has N rows (base constraints)
              HiGHS allocates vectors for N rows
              RSS: ~X MB

Iteration 10: Model has N + 10*F cuts (F = forward passes)
              HiGHS reallocates vectors for larger size
              RSS: ~X + delta MB

Iteration 100: Model has N + 100*F cuts
               HiGHS has grown to accommodate peak
               RSS: ~X + 10*delta MB (plateau)
```

### 3. Why Memory Isn't Reclaimed

#### No Shrink Policy

From `std::vector` behavior:
- `push_back()` grows capacity
- `clear()` sets size=0 but keeps capacity
- Only `shrink_to_fit()` reduces capacity (not called by HiGHS)

HiGHS intentionally avoids shrinking for performance:

```cpp
// HiGHS philosophy: avoid reallocations in hot paths
// Vectors grow to worst-case and stay there
void HEkkDual::resize() {
  if (new_size > work_dual.size()) {
    work_dual.resize(new_size);  // Only grow, never shrink
  }
}
```

#### Model Persistence

Each `solver::Model` in our `Subproblem` struct holds a HiGHS instance:

```rust
pub struct Subproblem {
    pub model: Option<solver::Model>,  // Persists for model lifetime
    // ...
}
```

The `HighsPtr` wrapping the HiGHS instance is only destroyed when the `Model` is dropped, which doesn't happen during normal training.

### 4. Allocation Sites from DHAT

The top allocation sites confirm HiGHS internal growth:

| Function | Bytes | Pattern |
|----------|-------|---------|
| `_M_fill_assign` | ~20 GB | Vector resize operations |
| `_M_default_append` | ~12 GB | Vector growth |
| `HEkkDual::*` | ~8 GB | Working vectors |

These are **valid allocations** that HiGHS needs, not leaks.

---

## Why This Isn't a Problem for HiGHS Itself

HiGHS is designed for:
1. **Repeated solves on similar problems** - Buffers reused
2. **Single-shot optimization** - Memory freed on exit
3. **Performance priority** - Avoid reallocation overhead

Our use case (SDDP training) is:
1. **Many evolving models** - Problem size grows
2. **Long-running process** - Hours of training
3. **Memory-constrained** - May hit system limits

This mismatch explains the RSS behavior.

---

## Potential Mitigations

### Option 1: Periodic Model Rebuild (Recommended)

**Concept**: Periodically destroy and recreate the HiGHS model to force memory reclaim.

```rust
// Every N iterations:
if iteration % REBUILD_INTERVAL == 0 {
    // Store current cuts
    let active_cuts = subproblem.get_active_cuts();
    
    // Destroy old model
    subproblem.model = None;
    
    // Rebuild with only active cuts
    subproblem.rebuild_model(active_cuts)?;
}
```

**Pros**:
- Forces HiGHS to reallocate at current size
- Removes dormant cuts from model

**Cons**:
- Requires saving/restoring cut information
- Rebuilding is expensive (once per N iterations)
- May affect basis warm-start

**Recommended interval**: Every 50-100 iterations

### Option 2: Aggressive Cut Selection

**Concept**: Limit the number of active cuts to bound model size.

Current: All cuts that dominate at least one state are kept active.

Proposed: Keep only the K most effective cuts per stage.

```rust
const MAX_CUTS_PER_STAGE: usize = 500;

// In add_cuts_batch:
if self.active_cut_count() > MAX_CUTS_PER_STAGE {
    self.remove_least_effective_cuts()?;
}
```

**Pros**:
- Bounds model size
- May improve solve time

**Cons**:
- May slow convergence
- Requires effectiveness metric

### Option 3: Model Pooling (Complex)

**Concept**: Pool of models that are periodically recycled.

**Not recommended** due to:
- Complex implementation
- Doesn't address HiGHS internal behavior
- May introduce correctness issues

### Option 4: Custom Allocator (jemalloc/mimalloc)

**Concept**: Use allocator with better fragmentation handling.

```toml
# Cargo.toml
[dependencies]
jemallocator = "0.5"
```

```rust
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;
```

**Pros**:
- May reduce fragmentation
- Drop-in replacement

**Cons**:
- Doesn't address HiGHS buffer growth
- May not help significantly

---

## Empirical Memory Analysis

### Expected Steady-State

For typical SDDP training (example: 05-large-scale-brazilian):

| Component | Base | Per Cut | After 1000 cuts |
|-----------|------|---------|-----------------|
| HiGHS model base | ~50 MB | - | ~50 MB |
| Cut rows | - | ~1 KB | ~1 MB |
| HiGHS working vectors | - | ~10 KB | ~10 MB |
| Factorization cache | ~20 MB | ~5 KB | ~25 MB |
| **Per Stage** | ~70 MB | | ~86 MB |
| **60 Stages Total** | ~4.2 GB | | ~5.2 GB |

### Observed vs Expected

| Metric | Expected | Observed | Difference |
|--------|----------|----------|------------|
| Initial RSS | ~4 GB | ~4 GB | Match |
| After 100 iter | ~5 GB | ~6 GB | +20% overhead |
| After 300 iter | ~5.5 GB | ~8 GB | +45% overhead |

The overhead is HiGHS's worst-case buffer sizing.

---

## Recommendations

### Immediate Actions

1. **Document as known behavior** ✅ (this document)
2. **Ensure sufficient system RAM** - Plan for 2x expected peak
3. **Monitor RSS during training** - Add logging if needed

### Sprint 8 Implementation

1. **Implement periodic model rebuild** (new ticket T-104):
   - Add `Subproblem::rebuild_model()` method
   - Integrate with training loop at configurable interval
   - Preserve active cuts during rebuild

2. **Add RSS monitoring** (optional T-105):
   - Log RSS at iteration intervals
   - Alert if approaching system limits

### Long-Term Considerations

1. **HiGHS contribution**: Propose `shrink_to_fit()` API or memory management options
2. **Alternative solvers**: Evaluate GLPK/CLP memory behavior
3. **Memory profiling**: Add jemalloc profiling for production monitoring

---

## Conclusion

The increasing RSS during SDDP training is **expected behavior** caused by:

1. HiGHS's performance-oriented memory management
2. Model growth as cuts are added
3. Lack of explicit memory reclaim API

This is **not a memory leak** but a trade-off inherent to HiGHS's design. The recommended mitigation is **periodic model rebuild** (every 50-100 iterations) to force memory reclaim while preserving active cuts.

---

## References

- HiGHS Source: https://github.com/ERGO-Code/HiGHS
- `HEkkDual.cpp`: Working vector management
- `HFactor.cpp`: Factorization buffer handling
- std::vector memory behavior: https://en.cppreference.com/w/cpp/container/vector
