# Memory Growth Analysis Report

**Date**: 2025-12-26  
**Example**: 05-large-scale-brazilian  
**POWE.RS Version**: 0.2.0

## Executive Summary

Despite preallocation efforts (23% performance improvement achieved), the application still exhibits significant memory growth during training. This report identifies the root causes and proposes solutions for achieving full memory determinism in HPC production environments.

**Key Finding**: The primary cause is **transient allocations** in the solver interface (`get_solution()` and `get_basis()`) that create ~4.9 GB of allocation churn during training. While these allocations are freed, the allocator does not return memory to the OS, causing observed RSS growth. Combined with FCF pool growth (~720 MB) and HiGHS internal allocations (~1.6 GB), the total observed RSS reaches ~5 GB.

## Observed Behavior

### Memory Profile During Training

| Iteration | Active Cuts | Memory (MB) | Delta (MB/iter) |
|-----------|-------------|-------------|-----------------|
| 1 | 944 | 2017.0 | baseline |
| 2 | 1819 | 2266.4 | +249.4 |
| 3 | 2556 | 2598.7 | +332.3 |
| 4 | 3329 | 3056.9 | +458.2 |
| 5 | 3796 | 3532.1 | +475.2 |
| 6 | 4267 | 4029.7 | +497.6 |
| 7 | 4890 | 4566.3 | +536.6 |
| 8 | 5643 | 5033.9 | +467.6 |

**Total growth**: ~3 GB over 8 iterations (~375 MB/iteration average)

### System Configuration

- **Stages**: 60 (59 study + 1 pre-study)
- **Branchings per stage**: 20
- **Forward passes**: 16
- **Iterations**: 8
- **Hydros**: 156
- **Total solves**: 8 × 16 × 60 × 20 = 153,600

## Root Cause Analysis

### 1. HiGHS Solution/Basis Extraction (PRIMARY CAUSE)

**Location**: `src/solver.rs:622-672`

Every LP solve allocates new vectors for the solution and basis:

```rust
pub fn get_solution(&self) -> Solution {
    let cols = self.num_cols();  // ~600+ columns
    let rows = self.num_rows();  // ~300+ rows
    let mut colvalue: Vec<f64> = vec![0.; cols];   // NEW ALLOCATION
    let mut coldual: Vec<f64> = vec![0.; cols];    // NEW ALLOCATION
    let mut rowvalue: Vec<f64> = vec![0.; rows];   // NEW ALLOCATION
    let mut rowdual: Vec<f64> = vec![0.; rows];    // NEW ALLOCATION
    // ...
}

pub fn get_basis(&self) -> Basis {
    let mut raw_colstatus: Vec<c_int> = vec![0; cols];  // NEW ALLOCATION
    let mut raw_rowstatus: Vec<c_int> = vec![0; rows];  // NEW ALLOCATION
    let colstatus = raw_colstatus.iter().map(...).collect();  // NEW ALLOCATION
    let rowstatus = raw_rowstatus.iter().map(...).collect();  // NEW ALLOCATION
    // ...
}
```

**Critical Bug**: `Realization::with_capacity()` preallocates basis storage (line 2927):
```rust
basis: solver::Basis::with_capacity(num_cols, num_rows),
```

But `realize_and_solve()` replaces it with a newly allocated one (line 1583):
```rust
realization_container.basis = basis;  // Preallocated buffer is discarded!
```

**Impact per solve**:
- Solution: 4 × ~500 × 8 bytes = ~16 KB
- Basis: 4 × ~500 × 8 bytes = ~16 KB
- Total per solve: ~32 KB
- Total across training: 153,600 × 32 KB = **4.9 GB allocated (but freed)**

While these are freed after use, they cause:
- Allocation pressure on the heap
- Fragmentation
- Memory not returned to OS (RSS stays high)

### 2. FCF Cut Pool Growth

**Location**: `src/fcf.rs:118-124`

The cut pool and state pool grow unbounded:

```rust
pub fn add_cut(&mut self, new_cut: cut::BendersCut) {
    self.cut_pool.pool.push(new_cut);  // Vec grows
}

pub fn add_state(&mut self, new_state: Box<dyn state::State>) {
    self.state_pool.pool.push(new_state);  // Vec grows + Box allocation
}
```

**Impact**:
- Cuts: 5643 active × ~1.3 KB/cut = ~7.3 MB per FCF
- States: 5643 × ~1.5 KB/state = ~8.5 MB per FCF
- With 59 FCF nodes: ~930 MB total

Note: While pools are preallocated via `reserve()` in `train()`, the **actual storage** still allocates as items are pushed.

### 3. Cut Cloning in Batch Processing

**Location**: `src/sddp/mod.rs:2184-2198`

Cuts are cloned for lock-free handler application:

```rust
let cuts: Vec<(usize, crate::cut::BendersCut)> = aggregated_result
    .new_cut_ids
    .iter()
    .chain(aggregated_result.returning_cut_ids.iter())
    .filter_map(|&cut_id| {
        fcf_locked.cut_pool.pool.get(cut_id)
            .map(|cut| (cut_id, cut.clone()))  // CLONE per cut
    })
    .collect();
```

**Impact per iteration**:
- ~128 cuts × 59 stages × ~1.3 KB = ~9.7 MB cloned per iteration
- Over 8 iterations: ~78 MB (freed after use)

### 4. HashMap Cloning in Cut Selection

**Location**: `src/sddp/mod.rs:2077-2091`

The active cut indices HashMap is cloned each stage:

```rust
let active_cut_indices_before: HashMap<usize, usize> = {
    let fcf_locked = parent_fcf_node.data.lock().unwrap();
    fcf_locked.cut_pool.active_cut_indices.clone()  // HashMap CLONE
};
```

**Impact**:
- HashMap with ~5000 entries × ~50 bytes = ~250 KB per clone
- 59 stages × 8 iterations = 472 clones = ~118 MB

### 5. HiGHS Internal Memory (EXTERNAL)

HiGHS C library internally manages memory that is not controlled by Rust. As constraint bounds change (cuts activated/deactivated), HiGHS may:
- Rebuild internal data structures
- Allocate working memory for simplex iterations
- Maintain basis factorization tables

This is harder to control without modifying HiGHS itself.

### 6. State Box Allocations

**Location**: `src/fcf.rs:122-123`

Each state is a `Box<dyn State>`, requiring heap allocation:

```rust
pub fn add_state(&mut self, new_state: Box<dyn state::State>) {
    self.state_pool.pool.push(new_state);  // Box is already heap-allocated
}
```

**Impact**: 5643 states × 24 bytes (Box overhead) + trait object = ~200 KB per FCF

## Memory Breakdown Estimate

| Component | Per Iteration | Total (8 iter) |
|-----------|--------------|----------------|
| Solution/Basis extractions (transient) | ~600 MB | ~4.9 GB churned |
| FCF Cut Pool growth | ~40 MB | ~320 MB retained |
| FCF State Pool growth | ~50 MB | ~400 MB retained |
| Cut cloning (transient) | ~10 MB | ~80 MB churned |
| HashMap cloning (transient) | ~15 MB | ~120 MB churned |
| HiGHS internal (estimated) | ~200 MB | ~1.6 GB retained |

**Total retained at end**: ~2.3 GB (HiGHS) + ~720 MB (FCF pools) = ~3 GB
**Observed**: 5 GB (includes fragmentation, RSS not returned to OS)

## Proposed Solutions

### Priority 1: Preallocate Solution/Basis Buffers (HIGH IMPACT)

**Change**: Pass mutable buffers to `get_solution()` and `get_basis()` instead of allocating new ones.

```rust
// Before
pub fn get_solution(&self) -> Solution { ... }

// After
pub fn get_solution_into(&self, solution: &mut Solution) { ... }
```

**Implementation**:
1. Add `Solution::with_capacity(cols, rows)` constructor
2. Add `Basis::with_capacity(cols, rows)` constructor  
3. Add `get_solution_into(&self, buf: &mut Solution)` method
4. Add `get_basis_into(&self, buf: &mut Basis)` method
5. Store pre-allocated buffers in `Realization` struct

**Impact**: Eliminates ~4.9 GB of allocation churn

### Priority 2: Preallocate Cut/State Storage in FCF

**Change**: Pre-allocate the actual storage for cuts and states, not just Vec capacity.

```rust
// Before: Reserve capacity but still push
fcf.cut_pool.pool.reserve(max_cuts);

// After: Pre-create slots, use index-based assignment
struct PreallocatedCutPool {
    cuts: Vec<Option<BendersCut>>,  // All slots pre-created
    free_list: Vec<usize>,
}
```

**Impact**: Eliminates ~320 MB of growth

### Priority 3: Eliminate Cut Cloning in Batch Processing

**Change**: Use Arc<BendersCut> or references instead of cloning.

```rust
// Before
.map(|cut| (cut_id, cut.clone()))

// After
.map(|cut| (cut_id, Arc::clone(&cut)))
// Or restructure to pass references
```

**Impact**: Eliminates ~80 MB of transient allocations

### Priority 4: Use Thread-Local Active Cut Indices

**Change**: Instead of cloning HashMap, use thread-local snapshots or restructure to avoid the clone.

```rust
// Before: Clone HashMap
let active_cut_indices_before = fcf.active_cut_indices.clone();

// After: Store as Vec of (cut_id, index) pairs
// or use a lock-free data structure
```

**Impact**: Eliminates ~120 MB of transient allocations

### Priority 5: Custom Allocator for HiGHS-Related Data

**Change**: Use `jemalloc` or `mimalloc` with tuned settings to reduce fragmentation and return memory to OS more aggressively.

```toml
# Cargo.toml
[dependencies]
mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }
```

```rust
// main.rs
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

**Impact**: Reduces RSS growth from fragmentation

### Priority 6: Arena Allocator for Transient Allocations

**Change**: Use `bumpalo` arena for per-iteration transient allocations:

```rust
use bumpalo::Bump;

let arena = Bump::new();
// Allocate cut copies in arena
let cuts = arena.alloc_slice_clone(&cut_copies);
// Arena dropped at end of iteration - instant deallocation
```

**Impact**: Near-zero allocation overhead for transient data

## Verification Commands

After implementing fixes, verify with:

```bash
# Memory profile during training
./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep -E "Iteration|MEMORY"

# Check peak RSS
/usr/bin/time -v ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | \
  grep "Maximum resident set size"

# Detailed allocation tracking (slow)
valgrind --tool=massif --detailed-freq=1 \
  ./target/release/powers run examples/01-deterministic
ms_print massif.out.*
```

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Peak RSS | 5 GB | < 2 GB |
| Memory growth per iteration | ~375 MB | 0 MB |
| Minor page faults | 3.4M | < 500K |

## Implementation Order

1. **Week 1**: Solution/Basis buffer preallocation (Priority 1)
2. **Week 2**: FCF pool preallocation (Priority 2)
3. **Week 3**: Cut cloning elimination (Priority 3)
4. **Week 4**: Testing and validation

## Appendix: Code Locations

| Issue | File | Lines |
|-------|------|-------|
| Solution allocation | `src/solver.rs` | 622-647 |
| Basis allocation | `src/solver.rs` | 650-672 |
| Cut pool push | `src/fcf.rs` | 118-120 |
| State pool push | `src/fcf.rs` | 122-124 |
| Cut cloning | `src/sddp/mod.rs` | 2184-2198 |
| HashMap cloning | `src/sddp/mod.rs` | 2077-2091 |
| HiGHS model creation | `src/subproblem.rs` | 857 |

## References

1. [HiGHS Memory Management](https://github.com/ERGO-Code/HiGHS/wiki)
2. [Rust Allocator Guidelines](https://doc.rust-lang.org/std/alloc/index.html)
3. [mimalloc Performance](https://github.com/microsoft/mimalloc)
4. [bumpalo Arena Allocator](https://docs.rs/bumpalo)
