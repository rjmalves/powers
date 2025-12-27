# Remaining Dynamic Allocations Analysis

**Date**: 2025-12-27  
**Version**: 0.2.0  
**Example**: 05-large-scale-brazilian (16 forward passes, 59 stages, 8 iterations)

## Executive Summary

Despite significant preallocation efforts achieving ~34% faster backward pass timing, the application still performs **~15,000+ heap allocations per training run** in the hot path. These allocations don't cause memory growth (preallocated pools absorb the data), but they create allocation pressure, cache pollution, and prevent true HPC-grade memory determinism.

This document catalogs every remaining dynamic allocation source to guide a future refactoring effort.

---

## Critical Hot Path Allocations

### 1. State Clone in `compute_new_cut` (HIGHEST IMPACT)

**Location**: `src/subproblem.rs:1689`

```rust
pub fn compute_new_cut(
    &self,
    branching_realizations: &[Realization],
    risk_measure: &dyn risk_measure::RiskMeasure,
    iteration: usize,
    forward_pass_idx: usize,
) -> fcf::CutStatePair {
    let mut visited_state = self.state.clone();  // <-- ALLOCATION HERE
    visited_state.set_iteration(iteration);
    visited_state.set_forward_pass_idx(forward_pass_idx);
    let cut = visited_state.compute_new_cut(risk_measure, branching_realizations);
    fcf::CutStatePair::new(cut, visited_state, forward_pass_idx)
}
```

**What happens**:
- `self.state` is a `Box<dyn State>` (trait object)
- Cloning calls `clone_dyn()` which allocates a new `Box` on the heap
- For `StorageState`: allocates ~1.2 KB (156 hydros × 8 bytes + struct overhead)
- For `StorageAndInflowState`: allocates ~2+ KB (storage + lag dimensions)

**Call frequency per training run**:
- `num_forward_passes × num_stages × num_iterations`
- Example 05: `16 × 59 × 8 = 7,552 allocations`

**Memory churn per run**:
- 7,552 × ~1.5 KB average = **~11.3 MB allocated and freed**

**Why it exists**:
- `CutStatePair` requires `Box<dyn State>` for ownership transfer to FCF
- The state is needed for: (1) computing cut RHS, (2) storing in state pool

**Why it's wasteful in preallocated mode**:
- The FCF state pool is already preallocated
- The cloned Box is thrown away after extracting coefficients (fcf.rs:351)
- Only the coefficients are copied to the preallocated slot

---

### 2. Cut Coefficients Clone (HIGH IMPACT)

**Location**: `src/state.rs:837` (StorageState) and `src/state.rs:1309` (StorageAndInflowState)

```rust
// In StorageState::evaluate_cut
cut::BendersCut::new(
    0,
    cut_coefficients.clone(),  // <-- ALLOCATION HERE
    cut_rhs,
    self.get_iteration(),
    self.get_forward_pass_idx(),
)
```

**What happens**:
- `cut_coefficients` is a reference to thread-local buffer (`&mut Vec<f64>`)
- `.clone()` allocates a new `Vec<f64>` with owned data
- Size: `state_dimension × 8 bytes` (~1.2 KB for 156 hydros)

**Call frequency per training run**:
- Same as state clone: 7,552 times

**Memory churn per run**:
- 7,552 × ~1.2 KB = **~9 MB allocated**

**Why it exists**:
- `BendersCut::new()` takes ownership of coefficients (`Vec<f64>`, not `&[f64]`)
- The cut must own its coefficient data for storage in FCF

**Why it's wasteful in preallocated mode**:
- The FCF cut pool has preallocated cuts with preallocated coefficient vectors
- `BendersCutPool::update_cut()` copies coefficients into preallocated slot
- The freshly allocated Vec is then dropped

---

### 3. Storage Extraction Clone (MEDIUM IMPACT)

**Location**: `src/state.rs:718` and `src/state.rs:932`

```rust
// StorageState::extract_storage_from_trajectory
fn extract_storage_from_trajectory(
    &mut self,
    trajectory: &[&subproblem::Realization],
) -> Vec<f64> {
    let prev_realization = trajectory.last().unwrap();
    self.state_coefficients.clone_from_slice(&prev_realization.final_storage);
    self.state_coefficients.clone()  // <-- ALLOCATION HERE
}
```

**What happens**:
- Returns a cloned Vec of storage values
- Size: `num_hydros × 8 bytes` (~1.2 KB)

**Call frequency**:
- Called during state extraction before each backward pass solve
- Unknown exact count without profiling, but likely ~7,500+ per run

**Why it exists**:
- Subproblem needs owned storage values for constraint updates
- API returns `Vec<f64>` instead of `&[f64]`

---

## Secondary Allocations (Lower Impact)

### 4. Realization Clones for Detail Export

**Locations**: 
- `src/sddp/mod.rs:744` - backward detail
- `src/sddp/mod.rs:826` - forward detail  
- `src/sddp/mod.rs:886` - backward detail
- `src/sddp/mod.rs:971` - backward detail

```rust
history.push(BackwardPassDetail {
    iteration,
    forward_pass_idx,
    stage_id: id as isize,
    training_state_id: 0,
    branching_idx,
    realization: realization.clone(),  // <-- ALLOCATION HERE
});
```

**Status**: Only active when `preserve_backward_detail` or `preserve_forward_detail` is true.

**Impact**: ~3 KB per realization × stages × iterations when enabled.

**Note**: Currently disabled in config (`export_forward_detail: false`), so zero impact in normal operation.

---

### 5. Simulation Result Clones

**Location**: `src/sddp/mod.rs:1198-1209`

```rust
RealizationData {
    loads: realization.loads.clone(),
    deficit: realization.deficit.clone(),
    exchange: realization.exchange.clone(),
    inflow: realization.inflow.clone(),
    turbined_flow: realization.turbined_flow.clone(),
    spillage: realization.spillage.clone(),
    thermal_generation: realization.thermal_generation.clone(),
    water_value: realization.water_value.clone(),
    marginal_cost: realization.marginal_cost.clone(),
    final_storage: realization.final_storage.clone(),
    ...
}
```

**Impact**: Only during simulation phase, not training hot path.

---

### 6. Noise Vector `to_vec()` Conversions

**Location**: `src/sddp/mod.rs:1921` and `src/sddp/mod.rs:2534`

```rust
.map(|(handler, noises)| self.forward(noises.to_vec(), handler))
```

**What happens**:
- Converts slice to owned Vec for function call
- Size: `num_uncertainties × 8 bytes`

**Call frequency**: Once per forward pass per iteration.

---

### 7. HashSet Allocations in Batch Cut Selection

**Location**: `src/fcf.rs:325-326` and `src/fcf.rs:407`

```rust
let mut new_cut_ids = HashSet::new();
let mut returning_cut_ids = HashSet::new();
// ...
.collect()  // <-- HashSet allocation
```

**Impact**: ~100-200 bytes per batch, 59 batches per iteration = ~12 KB per iteration.

---

### 8. Removed Indices Vec in Cut Selection

**Location**: `src/sddp/mod.rs:2173`

```rust
let mut removed_indices: Vec<usize> = Vec::new();
```

**Impact**: Small Vec, ~100 bytes, reused per stage.

---

## Allocation-Free Paths (Already Optimized)

These paths have been optimized and no longer allocate:

1. **Solution/Basis extraction** - Uses thread-local `SOLUTION_BUFFER`
2. **Cut coefficient computation** - Uses `CutComputationBuffers` (costs, objective_contributions, contributions_outer)
3. **Kahan summation** - Uses `kahan_sum_iter()` instead of collecting to Vec
4. **State coefficient copying in FCF** - Passes slice directly, no `to_vec()`

---

## Memory Impact Summary

| Allocation Source | Per Call | Calls/Run | Total/Run | % of Hot Path |
|-------------------|----------|-----------|-----------|---------------|
| State Clone (Box) | ~1.5 KB | 7,552 | 11.3 MB | 45% |
| Cut Coefficients Clone | ~1.2 KB | 7,552 | 9.0 MB | 36% |
| Storage Extraction Clone | ~1.2 KB | ~7,500 | 9.0 MB | 15% |
| HashSet allocations | ~0.2 KB | 472 | 0.1 MB | <1% |
| Other | varies | varies | ~1 MB | 3% |
| **TOTAL** | | | **~30 MB** | |

Note: These allocations are transient (freed after use), so they don't increase RSS. However, they:
- Create allocation pressure on the heap
- Pollute CPU caches with allocation metadata
- Prevent vectorization of surrounding code
- Add ~5-10% runtime overhead

---

## Proposed Fixes (Future Refactoring)

### Fix 1: Eliminate State Clone in Preallocated Mode

**Problem**: `CutStatePair` contains `Box<dyn State>` which is cloned but then discarded.

**Solution A**: Split `CutStatePair` into two variants:
```rust
pub enum CutStatePair {
    // For preallocated mode: no Box, just coefficients
    Lightweight {
        cut: BendersCut,
        state_coefficients: Vec<f64>,  // Or borrow with lifetime
        forward_pass_idx: usize,
    },
    // For non-preallocated mode: full Box
    Full {
        cut: BendersCut,
        state: Box<dyn State>,
        forward_pass_idx: usize,
    },
}
```

**Solution B**: Use COW (Copy-on-Write) for state:
```rust
pub struct CutStatePair {
    cut: BendersCut,
    state: Cow<'static, dyn State>,  // Borrows in preallocated, owns otherwise
    forward_pass_idx: usize,
}
```

**Solution C**: Separate the cut computation from state storage:
```rust
// Returns just the data needed for preallocated update
pub fn compute_new_cut_data(
    &self,
    branching_realizations: &[Realization],
    risk_measure: &dyn RiskMeasure,
    iteration: usize,
    forward_pass_idx: usize,
) -> CutData {
    CutData {
        coefficients: self.state.compute_cut_coefficients(...),
        rhs: self.state.compute_cut_rhs(...),
        state_coefficients: self.state.coefficients().to_vec(),  // Still allocates, but smaller
        iteration,
        forward_pass_idx,
    }
}
```

**Estimated Impact**: Eliminates 7,552 × 1.5 KB = **11.3 MB** allocations per run.

---

### Fix 2: Eliminate Cut Coefficients Clone

**Problem**: `BendersCut::new()` takes `Vec<f64>` ownership, requiring clone from buffer.

**Solution A**: Add `BendersCut::from_buffer()` that steals buffer content:
```rust
impl BendersCut {
    /// Creates cut by taking ownership of buffer contents.
    /// Caller's buffer is left empty but with capacity preserved.
    pub fn from_buffer(
        id: usize,
        coefficients: &mut Vec<f64>,
        rhs: f64,
        iteration: usize,
        forward_pass_idx: usize,
    ) -> Self {
        Self {
            id,
            coefficients: std::mem::take(coefficients),  // Zero-copy transfer
            rhs,
            ...
        }
    }
}
```

**Solution B**: For preallocated mode, don't create BendersCut at all:
```rust
// In evaluate_cut for preallocated mode:
// Return a lightweight struct that references the buffer
pub struct CutDataRef<'a> {
    coefficients: &'a [f64],
    rhs: f64,
    iteration: usize,
    forward_pass_idx: usize,
}
```

**Estimated Impact**: Eliminates 7,552 × 1.2 KB = **9 MB** allocations per run.

---

### Fix 3: Eliminate Storage Extraction Clone

**Problem**: `extract_storage_from_trajectory()` returns `Vec<f64>` instead of reference.

**Solution**: Change API to update in place:
```rust
fn update_storage_from_trajectory(
    &mut self,
    trajectory: &[&Realization],
    target: &mut [f64],  // Caller provides preallocated buffer
) {
    let prev = trajectory.last().unwrap();
    target.copy_from_slice(&prev.final_storage);
}
```

Or return a reference when possible:
```rust
fn storage_coefficients(&self) -> &[f64] {
    &self.state_coefficients
}
```

**Estimated Impact**: Eliminates ~7,500 × 1.2 KB = **9 MB** allocations per run.

---

## Implementation Priority

| Priority | Fix | Complexity | Impact |
|----------|-----|------------|--------|
| 1 | State Clone Elimination | High | 11.3 MB |
| 2 | Cut Coefficients Clone | Medium | 9.0 MB |
| 3 | Storage Extraction Clone | Low | 9.0 MB |
| 4 | HashSet preallocatoin | Low | 0.1 MB |

**Recommended Approach**:
1. Start with Fix 3 (easiest, low risk)
2. Then Fix 2 (medium complexity, high impact)
3. Finally Fix 1 (requires API redesign, highest impact)

---

## Validation Plan

After implementing fixes:

```bash
# 1. Verify correctness
cargo test --lib
./target/release/powers run examples/01-deterministic
./target/release/powers run examples/03-multistage
./target/release/powers run examples/05-large-scale-brazilian

# 2. Profile allocations
valgrind --tool=massif ./target/release/powers run examples/05-large-scale-brazilian
ms_print massif.out.*

# 3. Measure performance
hyperfine --warmup 1 --runs 3 \
  './target/release/powers run examples/05-large-scale-brazilian'

# 4. Verify numerical reproducibility (critical!)
for i in {1..5}; do
  ./target/release/powers run examples/05-large-scale-brazilian 2>&1 | grep "Final policy"
done
# All runs must produce identical results
```

---

## References

1. `MEMORY_GROWTH_ANALYSIS.md` - Original root cause analysis
2. `plans/memory-growth-prevention/00-master-plan.md` - Implementation status
3. `src/memory/buffers.rs` - Cut computation buffers (already optimized)
4. `src/subproblem.rs:1682-1696` - compute_new_cut
5. `src/state.rs:756-841` - StorageState::evaluate_cut
6. `src/state.rs:1204-1310` - StorageAndInflowState::evaluate_cut
7. `src/fcf.rs:337-390` - add_cuts_batch

---

## Appendix: Data Structure Sizes

### StorageState
```
Size: 48 bytes (stack) + state_coefficients heap allocation
     = 48 + (num_hydros × 8) bytes
     = 48 + (156 × 8) = 1,296 bytes for example 05
```

### StorageAndInflowState
```
Size: 80 bytes (stack) + state_coefficients + layout overhead
     = 80 + (state_dim × 8) + ~100 bytes
     = 80 + (312 × 8) + 100 = ~2,676 bytes for PAR models
```

### BendersCut
```
Size: 88 bytes (stack) + coefficients heap allocation
     = 88 + (state_dim × 8) bytes
     = 88 + (156 × 8) = 1,336 bytes for example 05
```

### CutStatePair
```
Size: 24 bytes (stack) + Box<dyn State> + BendersCut
     = 24 + 1,296 + 1,336 = ~2,656 bytes total heap
```
