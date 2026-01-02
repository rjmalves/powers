# Hot Path Allocation Audit Report

> **Date**: 2025-12-30
> **Purpose**: Identify potential memory allocation sources causing RSS growth during SDDP training
> **Scope**: Training hot path from `train()` → forward pass → backward pass

---

## Executive Summary

Despite Sprint 05 memory optimizations, RSS still grows during training. This report identifies **all potential allocation sites** in the training hot path by tracing code flow from `SddpAlgorithm::train()` through forward and backward passes.

### ⚠️ CRITICAL FINDING FROM DHAT PROFILING

**The HiGHS LP solver accounts for 94.7% of all heap allocations** (see [Appendix B](#appendix-b-dhat-heap-profile-analysis)). The Rust application code contributes only 2.0% of allocations. Optimization efforts should prioritize:

1. HiGHS warm-starting / factorization reuse
2. Batching `changeRowBounds` calls (3 million allocations)
3. Disabling HiGHS debug mode

### Key Finding Categories (Rust Code Only)

| Category | Severity | Count | Impact |
|----------|----------|-------|--------|
| **Per-iteration allocations** | 🔴 HIGH | 8 | RSS grows with each iteration |
| **Per-stage allocations** | 🟠 MEDIUM | 12 | Multiplied by ~60 stages |
| **Per-forward-pass allocations** | 🟡 LOW | 5 | Constant overhead |
| **Conditional allocations** | 🟢 OK | 6 | Only when features enabled |

---

## Training Loop Entry (`train()` at sddp/mod.rs:1743)

### 🟢 Initialization Phase (OK - runs once)

These allocations happen before training iterations start:

```rust
// Line 1823: Thread-local buffer initialization (once per thread)
crate::memory::initialize_cut_buffers(max_state_dim, max_scenarios);

// Line 1868: FCF pool preallocation (once per node)
fcf::FutureCostFunction::preallocate_pools(...)

// Line 1883: Iterations result vector
let mut iterations = Vec::with_capacity(num_iterations);

// Line 1906-1918: Handler creation (once)
let handlers: Vec<SddpTrainHandler> = (0..num_forward_passes).map(...).collect();
```

### 🔴 Per-Iteration Allocations (CRITICAL)

These happen **every iteration** in the training loop:

#### 1. SAA Scenario Sampling (Line 1942-1944)
```rust
let all_sampled_noises: Vec<_> = (0..num_forward_passes)
    .map(|_| saa.sample_scenario(&mut rng))
    .collect();
```
**Issue**: `sample_scenario()` returns `Vec<&OptimizedSampledBranchingNoises>`
- Allocates Vec of references per forward pass
- **Fix**: Use preallocated buffer for noise references

#### 2. `noises.to_vec()` in Forward Call (Line 1952)
```rust
.map(|(handler, noises)| self.forward(noises.to_vec(), handler))
```
**Issue**: Clones the noise reference vector for each forward pass
- **Fix**: Pass slice reference instead of owned Vec

#### 3. Forward Results Collection (Line 1948-1953)
```rust
let forward_results: Vec<(f64, ForwardPassTimingAccumulator)> = coordinator
    .handlers_mut()
    .par_iter_mut()
    ...
    .collect::<Result<Vec<...>, String>>()?;
```
**Issue**: Allocates result tuples and collects into Vec
- **Fix**: Use preallocated result buffer

#### 4. Forward Costs/Timings Unzip (Line 1957-1960)
```rust
let (forward_costs, forward_timings): (Vec<f64>, Vec<ForwardPassTimingAccumulator>) = 
    forward_results.into_iter().unzip();
```
**Issue**: Creates two new Vecs from the results
- **Fix**: Use preallocated buffers for costs and timings

#### 5. Forward Costs Clone for IterationResult (Line 2080)
```rust
iterations.push(IterationResult {
    forward_costs: forward_costs.clone(),
    ...
});
```
**Issue**: Clones forward costs Vec every iteration
- **Fix**: Move instead of clone, or use indices into shared buffer

#### 6. BackwardPassContext Creation (Line 2018)
```rust
let backward_ctx = BackwardPassContext::new(...)
```
**Issue**: May allocate internal buffers if `BackwardPassContext::new` allocates
- **Verify**: Check `BackwardPassContext::new` implementation

---

## Forward Pass Hot Path

### `SddpTrainHandler::forward()` (sddp/mod.rs:651)

#### 7. ForwardPassContext Creation (Line 665)
```rust
let mut ctx = ForwardPassContext::new(...)
```
**Issue**: Created fresh each forward pass - may allocate internal structures
- **Verify**: Check `ForwardPassContext` fields

### `forward_pass::execute()` (algorithm/forward_pass.rs:74)

#### 8. Past Realizations Collection (Line 123-136)
```rust
let past_realizations: Vec<&Realization> = past_node_ids
    .iter()
    .map(...)
    .collect::<Result<_, _>>()?;
```
**Issue**: Allocates Vec of references **every stage** (60 stages × 4 forward passes = 240 allocations/iteration)
- **Fix**: Use preallocated slice buffer in context

---

## Backward Pass Hot Path

### `backward_pass::execute()` (algorithm/backward_pass.rs:189)

#### 9. Coordinator Result Buffers (coordinator.rs:321-322) - **Partially Fixed**
```rust
let mut slots = Vec::with_capacity(self.num_forward_passes);
let mut timings = Vec::with_capacity(self.num_forward_passes);
```
**Status**: `compute_cuts_into_slots` uses preallocated buffers, but `compute_cuts_parallel_into_slots` still allocates
- **Fix**: Ensure ALL coordinator methods use `self.buffers`

#### 10. Slots Clone in compute_cuts_into_slots (coordinator.rs:234)
```rust
let mut slots = self.buffers.slots.clone();
```
**Issue**: Clones the buffer for return value
- **Fix**: Return reference or use output parameter

### Cut Selection Phase

#### 11. HashSet Allocations in FCF (fcf.rs:321-322)
```rust
let mut new_cut_ids = HashSet::new();
let mut returning_cut_ids = HashSet::new();
```
**Issue**: Creates new HashSets per batch - allocates hash table
- **Fix**: Use preallocated BitSet or Vec-based set for known max size

#### 12. HashSet Clone for AggregatedResult (coordinator.rs:384-386)
```rust
let aggregated = AggregatedCutSelectionResult {
    new_cut_ids: batch_result.new_cut_ids.clone(),
    returning_cut_ids: batch_result.returning_cut_ids.clone(),
    removing_cut_ids: batch_result.removing_cut_ids.clone(),
};
```
**Issue**: Clones all three HashSets every stage
- **Fix**: Move or use reference wrapper

#### 13. Cut IDs Collection (coordinator.rs:400-405)
```rust
let cut_ids: Vec<usize> = aggregated
    .new_cut_ids
    .iter()
    .chain(aggregated.returning_cut_ids.iter())
    .copied()
    .collect();
```
**Issue**: Allocates Vec for cut IDs every stage
- **Fix**: Use preallocated buffer

#### 14. Removing Cut IDs Collection (fcf.rs:381-391)
```rust
let removing_cut_ids: HashSet<usize> = if enable_cut_selection {
    self.cut_pool.pool.iter()
        .filter(...)
        .map(|c| c.id)
        .collect()
} else {
    HashSet::new()
};
```
**Issue**: Allocates HashSet every batch even if empty
- **Fix**: Use preallocated set or return iterator

---

## State/Cut Evaluation Hot Path

### `StorageState::evaluate_cut_ref()` (state.rs:1358-1419)

**Status**: ✅ Uses preallocated buffers correctly

### `StorageAndInflowState::evaluate_cut_ref()` (state.rs:1917-1995)

**Status**: ✅ Uses preallocated buffers correctly

### `uniform_prob_by_count()` (utils/mod.rs:269)
```rust
pub fn uniform_prob_by_count(count: usize) -> Vec<f64> {
    vec![p; count]
}
```
**Issue**: Allocates Vec **every cut evaluation** (60 stages × 4 FP × branchings)
- Called from: `state.rs:1298, 1375, 1842, 1935`
- **Fix**: Use preallocated probability buffer or compute in-place

### `compute_cut_data()` (subproblem.rs:1735-1762)

#### 15. State Clone (Line 1739)
```rust
let mut visited_state = self.state.clone();
```
**Issue**: Clones entire state object for each cut computation
- Contains: `state_coefficients` Vec and layout data
- **Fix**: Use state reference or staging buffer

#### 16. Coefficients to_vec() (Line 1755)
```rust
let cut = cut::BendersCut::new(
    0,
    eval_result.coefficients.to_vec(),  // ALLOCATION
    ...
);
```
**Issue**: Copies coefficients from buffer to owned Vec for BendersCut
- **Fix**: BendersCut should use slot-based coefficient storage

#### 17. Cuts to Process Collection (subproblem.rs:1779-1790)
```rust
let mut cuts_to_process: Vec<(usize, &cut::BendersCut)> = cut_ids
    .iter()
    .filter_map(...)
    .collect();
```
**Issue**: Allocates Vec for cuts to process in `apply_aggregated_cut_selection_result`
- **Fix**: Use preallocated buffer

---

## Scenario Sampling

### `sample_scenario()` (scenario.rs:452-466)

#### 18. Branching Indices Vec (Line 456-457)
```rust
let branching_indices: Vec<usize> =
    self.index_samplers.iter().map(|d| d.sample(rng)).collect();
```
**Issue**: Allocates Vec of indices every scenario sample
- **Fix**: Use preallocated buffer per thread

#### 19. Return Vec (Line 459-466)
```rust
branching_indices.iter()
    .enumerate()
    .map(...)
    .collect()
```
**Issue**: Allocates Vec of noise references
- **Fix**: Return iterator or use preallocated buffer

---

## Hidden Allocations

### Risk Measure `adjust_probabilities()` (risk_measure.rs:18-22)
```rust
fn adjust_probabilities<'a>(
    &self,
    probabilities: &'a [f64],
    _costs: &[f64],
) -> &'a [f64] {
    probabilities  // No allocation for Expectation
}
```
**Status**: ✅ OK for Expectation risk measure (returns reference)

### Solution Buffer (subproblem.rs:21-27)
```rust
thread_local! {
    static SOLUTION_BUFFER: RefCell<solver::Solution> = const { RefCell::new(solver::Solution {
        colvalue: Vec::new(),
        ...
    })};
}
```
**Status**: ✅ Uses thread-local buffer correctly

---

## Priority Fix List

### 🔴 HIGH Priority (Per-iteration, significant impact)

1. **`uniform_prob_by_count()`** - Allocates every cut evaluation
   - Impact: `num_stages × num_forward_passes × num_branchings` allocations/iteration
   - Fix: Preallocated probability buffer

2. **`sample_scenario()` Vecs** - Allocates every iteration
   - Impact: 2 Vecs per forward pass per iteration
   - Fix: Preallocated scenario buffer

3. **`noises.to_vec()`** - Clones noise references
   - Impact: 1 clone per forward pass per iteration
   - Fix: Pass slice reference

4. **`state.clone()` in `compute_cut_data()`** - Clones state per cut
   - Impact: `num_stages × num_forward_passes` clones/iteration
   - Fix: State staging buffer

5. **`forward_costs.clone()`** - Clones for IterationResult
   - Impact: 1 Vec clone per iteration
   - Fix: Move ownership

### 🟠 MEDIUM Priority (Per-stage)

6. **`past_realizations` Vec** - Allocates every stage
   - Fix: Preallocated trajectory buffer

7. **HashSet allocations in FCF** - 3 HashSets per stage
   - Fix: Preallocated BitSet

8. **`cut_ids` collection** - Vec per stage
   - Fix: Preallocated buffer

9. **`slots.clone()`** in coordinator
   - Fix: Return reference

### 🟡 LOW Priority (One-time or conditional)

10. **ForwardPassContext creation** - Per forward pass
    - Audit internal allocations

11. **BackwardPassContext creation** - Per iteration
    - Audit internal allocations

---

## Estimated Impact

Assuming example 05 (60 stages, 4 forward passes, 8 iterations, ~20 branchings avg):

| Allocation Site | Per-Iter Count | Estimated Bytes | Total/Iter |
|-----------------|----------------|-----------------|------------|
| `uniform_prob_by_count` | 60×4×20 = 4,800 | 160 bytes | 768 KB |
| `sample_scenario` | 4 | 480 bytes | 1.9 KB |
| `noises.to_vec()` | 4 | 480 bytes | 1.9 KB |
| `state.clone()` | 60×4 = 240 | ~1.3 KB | 312 KB |
| `past_realizations` | 60×4 = 240 | 240 bytes | 56 KB |
| HashSets (3×) | 60 | ~500 bytes | 30 KB |
| **Total per iteration** | | | **~1.2 MB** |
| **Total 8 iterations** | | | **~9.5 MB** |

This doesn't account for:
- HiGHS internal allocations (unknown)
- Fragmentation overhead
- OS page allocation granularity

---

## Recommendations

### Phase 1: Quick Wins
1. Replace `uniform_prob_by_count()` with computation into preallocated buffer
2. Remove `noises.to_vec()` - pass slice reference
3. Use `std::mem::take()` instead of `forward_costs.clone()`

### Phase 2: Structural Changes
4. Preallocate scenario sampling buffers
5. Preallocate trajectory reference buffer
6. Replace HashSets with BitVec or preallocated Vec

### Phase 3: Architecture
7. State staging buffer to avoid clone
8. Coefficient slot storage to avoid `coefficients.to_vec()`
9. Context object pooling

---

## Appendix: Code Locations

| Function | File | Line |
|----------|------|------|
| `train()` | src/sddp/mod.rs | 1743 |
| `forward()` (handler) | src/sddp/mod.rs | 651 |
| `forward_pass::execute()` | src/algorithm/forward_pass.rs | 74 |
| `backward_pass::execute()` | src/algorithm/backward_pass.rs | 189 |
| `compute_cuts_into_slots()` | src/algorithm/coordinator.rs | 191 |
| `compute_cuts_parallel_into_slots()` | src/algorithm/coordinator.rs | 306 |
| `evaluate_cut_ref()` (Storage) | src/state.rs | 1358 |
| `evaluate_cut_ref()` (StorageAndInflow) | src/state.rs | 1917 |
| `compute_cut_data()` | src/subproblem.rs | 1735 |
| `sample_scenario()` | src/scenario.rs | 452 |
| `uniform_prob_by_count()` | src/utils/mod.rs | 269 |
| `add_cuts_batch()` | src/fcf.rs | 311 |
| `finalize_cuts_batch()` | src/fcf.rs | 575 |

---

## Appendix B: DHAT Heap Profile Analysis

> **Profiling Date**: 2025-12-30
> **Command**: `./target/release/powers run examples/05-large-scale-brazilian`
> **Tool**: DHAT (Valgrind heap profiler)

### Executive Summary

The DHAT heap profiler reveals that **94.7% of all heap allocations come from the HiGHS solver library**, not from the Rust application code. This is a critical finding that redirects optimization focus.

| Component | Bytes Allocated | Percentage |
|-----------|-----------------|------------|
| HiGHS Solver (HEkk/HFactor) | 83.5 GB | 94.7% |
| HiGHS Presolve | 2.8 GB | 3.2% |
| Rust Application (powers) | 1.8 GB | 2.0% |
| Parquet I/O | 12.9 MB | 0.0% |

**Total allocations during run**: 88.19 GB across 159 million allocation blocks

**Peak heap size**: 416 MB (allocations are freed, but churn is high)

---

### HiGHS Solver Allocations (Dominant)

The HiGHS linear programming solver dominates heap activity. Key allocation sites:

#### 1. HFactor::setupGeneral (44.9% of all allocations)
- **Bytes**: 39.6 GB in 2.4 million blocks
- **Cause**: Factorization matrix setup for each LP solve
- **Pattern**: Allocates sparse matrix structures repeatedly
- **Impact**: ~88 KB average per factorization

#### 2. HEkk::computeDual (22.7% of all allocations)  
- **Bytes**: 20.0 GB in 3.8 million blocks
- **Cause**: Dual simplex operations allocate HVectorBase work arrays
- **Pattern**: Called ~119,000 times (once per solve + iterations)
- **Wasteful**: Many allocations have zero reads (written but never used)

#### 3. HEkkDual::initialiseInstance (9.6%)
- **Bytes**: 8.4 GB in 1.8 million blocks
- **Cause**: Dual simplex initialization per solve

#### 4. HEkkDual::solve/rebuild/cleanup (11.9% combined)
- **Bytes**: 10.6 GB
- **Cause**: Simplex iteration memory management

---

### High-Frequency Allocation Sites

Sorted by **block count** (frequency of allocation):

| Rank | Function | Blocks | Bytes | Avg Size | Source |
|------|----------|--------|-------|----------|--------|
| 1 | HFactor::setupGeneral | 37.8M | 22.7 GB | 600 B | Factorization |
| 2 | HEkk::computeDual | 3.8M | 20.0 GB | 5.3 KB | Dual ops |
| 3 | changeRowBounds (HiGHS) | 3.0M | 188 MB | 62 B | Bound updates |
| 4 | changeRowBoundsInterface | 3.0M | 176 MB | 58 B | Bound interface |
| 5 | debugDualSimplex | 2.9M | 50 MB | 17 B | Debug strings |

**Critical Finding**: `changeRowBounds` is called 3 million times, allocating small vectors and strings each call. This is triggered by `Model::change_rows_bounds()` in `solver.rs:572`.

---

### Wasteful Allocations (Write-Heavy, Low/No Reads)

These allocations are written but rarely or never read, indicating potential waste:

| Function | Writes | Reads | Ratio | Bytes |
|----------|--------|-------|-------|-------|
| HFactor::setupGeneral (vector reserve) | 6.4 GB | 0 | ∞ | 3.3 GB |
| HFactor::setupGeneral (int default_append) | 1.7 GB | 0 | ∞ | 1.1 GB |
| HFactor::setupGeneral (double reserve) | 1.1 GB | 0 | ∞ | 3.3 GB |
| HEkk::allocateWorkAndBaseArrays | 977 MB | 144 B | 6.8M:1 | 2.5 MB |
| HVectorBase::setup (double) | 882 MB | 0 | ∞ | 882 MB |
| HVectorBase::setup (char) | 873 MB | 0 | ∞ | 873 MB |
| HEkk::putBacktrackingBasis | 779 MB | 0 | ∞ | 12 MB |

**Insight**: HiGHS allocates large work arrays that are often not fully utilized.

---

### Long-Lived Allocations at End

Only 9.4 MB remains allocated at program end:

| Allocation | End Bytes | Blocks | Source |
|------------|-----------|--------|--------|
| HighsTaskExecutor threads | 8.4 MB | 16 | HiGHS thread pool |
| Rust Vec growth | 379 KB | 304 | Application data |
| Rayon deque | 24 KB | 16 | Thread work queues |

**No significant memory leaks detected** - all training allocations are freed.

---

### Rust Application Allocations (2.0%)

While small in comparison, the Rust code still allocates ~1.8 GB:

#### Powers/SDDP Specific Sites:
- `Model::try_set_basis` → triggers HiGHS basis factorization
- `solve_all_branchings` → reuses forward basis (triggers HiGHS)
- `retry_solve` → fallback solving path

#### Parquet I/O (12.9 MB total):
- Dictionary encoding for output files
- Column writer buffers
- Minimal impact on training hot path

---

### Key Insights

1. **HiGHS is the allocation bottleneck**, not Rust code
   - 94.7% of allocations from solver internals
   - Rust code cannot directly reduce these allocations

2. **Model warm-starting could reduce HiGHS allocations**
   - `HFactor::setupGeneral` runs on every solve (44.9%)
   - Reusing factorization basis could eliminate this

3. **changeRowBounds is called excessively**
   - 3 million calls for bound updates
   - Each call allocates 5 small vectors + 2 strings
   - Consider batching bound changes

4. **HiGHS debug mode may be enabled**
   - `debugDualSimplex` allocates 50 MB of strings
   - Ensure release mode disables debug output

5. **Parquet I/O is negligible** (0.01% of allocations)
   - No optimization needed for output path

---

### Recommendations (Updated with DHAT Findings)

#### Phase 1: HiGHS Configuration (Highest Impact)
1. **Investigate HiGHS warm-start API** - Avoid `HFactor::setupGeneral` on every solve
2. **Batch changeRowBounds calls** - Reduce 3M calls to fewer bulk updates
3. **Disable HiGHS debug mode** - Eliminate 50 MB of debug string allocations
4. **Consider HiGHS presolve settings** - 3.2% of allocations from presolve

#### Phase 2: Rust Allocations (Lower Priority)
5. Continue with original audit recommendations for Rust code
6. Focus on `uniform_prob_by_count()` and scenario sampling

#### Phase 3: Profile-Guided Optimization
7. Rebuild with HiGHS `NDEBUG` flag to remove debug allocations
8. Profile again after Phase 1 changes to measure impact

---

### Allocation Timeline Estimates

Based on 119,241 solver invocations detected:

| Metric | Value |
|--------|-------|
| Solves | 119,241 |
| Avg allocations per solve | 1,333 |
| Avg bytes per solve | 740 KB |
| HiGHS factorization overhead | ~330 KB/solve |
| HiGHS vector setup overhead | ~250 KB/solve |

---

### DHAT Output File Reference

Full DHAT output available at: `dhat.out` (115K lines)

Key JSON fields:
- `tb`: Total bytes allocated
- `tbk`: Total blocks allocated  
- `mb`: Maximum bytes at any point
- `rb`: Total bytes read
- `wb`: Total bytes written
- `eb`: Bytes remaining at end
- `fs`: Frame stack (indices into `ftbl`)

---

## Appendix C: Sprint 6 Results Summary

> **Date**: 2025-12-30
> **Full Analysis**: [DHAT_SPRINT6_ANALYSIS.md](./DHAT_SPRINT6_ANALYSIS.md)

Sprint 6 achieved exceptional results, far exceeding the 30% target:

| Metric | Before Sprint 6 | After Sprint 6 | Improvement |
|--------|-----------------|----------------|-------------|
| **Total Bytes Allocated** | 88.19 GB | 45.43 GB | **48.5% reduction** |
| **Total Allocation Blocks** | 159.0M | 43.4M | **72.7% reduction** |
| **HFactor::setupGeneral** | 39.58 GB | 2.00 GB | **95.0% reduction** |
| **changeRowBounds blocks** | 98.4M | 0.4M | **99.6% reduction** |

**Key Finding**: The `reuse_forward_basis()` function was triggering HiGHS "alien basis" handling, forcing full factorization rebuilds on every backward branching solve. Disabling it eliminated 95% of HFactor allocations.

This validates the hypothesis from [HIGHS_WARM_START_INVESTIGATION.md](./HIGHS_WARM_START_INVESTIGATION.md).
