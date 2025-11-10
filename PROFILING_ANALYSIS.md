# 🔍 Profiling Analysis - Example 05 (156 Hydros)

**Date**: 2025-11-09  
**Problem**: Large-Scale Brazilian System (156 hydros, 121 thermals, 60 stages)  
**Runtime**: 37.3 seconds (8 iterations, 16 forward passes)

---

## Executive Summary

### ✅ What Worked
1. **Perf Report** - Valid data showing CPU hotspots
2. **Massif Memory** - 2GB peak memory, allocation patterns identified
3. **Timing Data** - Detailed forward/backward breakdown

### ❌ What Failed
- **Flamegraph Generation** - Lost 80% of samples due to I/O overload
- **Cache Counters** - Not supported on this CPU (perf stat)

### 🎯 Key Finding

**Backward pass is the bottleneck** (3x slower than forward):
- Forward: ~0.15s average per iteration
- Backward: ~4.7s average per iteration (and growing!)
- **Ratio: 31x slower** at iteration 8!

---

## 📊 Performance Breakdown

### Timing Analysis (from training output)

| Iteration | Forward | Backward | Ratio | Active Cuts |
|-----------|---------|----------|-------|-------------|
| 1 | 0.17s | 2.63s | 15.5x | 944 |
| 2 | 0.08s | 2.28s | 28.5x | 1,821 |
| 3 | 0.09s | 3.04s | 33.8x | 2,529 |
| 4 | 0.11s | 4.08s | 37.1x | 3,312 |
| 5 | 0.13s | 4.83s | 37.2x | 3,899 |
| 6 | 0.15s | 5.66s | 37.7x | 4,655 |
| 7 | 0.17s | 6.51s | 38.3x | 5,310 |
| 8 | 0.21s | 7.26s | **34.6x** | 5,996 |

**Critical Observation**: Backward pass time **grows linearly** with active cuts!
- Iteration 1: 2.6s with 944 cuts
- Iteration 8: 7.3s with 5,996 cuts (6.3x more cuts, 2.8x slower)

**Conclusion**: Cut management overhead is significant but sublinear (good!).

---

## 🔥 CPU Hotspots (from perf report)

### Top 15 Functions by CPU Time

| % Time | Function | Component | Analysis |
|--------|----------|-----------|----------|
| 11.44% | `HFactor::ftranU` | **HiGHS Solver** | LP factorization (unavoidable) |
| 5.31% | `std::_Rb_tree_increment` | **std::map iterator** | ⚠️ **Red flag** - too much time in tree traversal |
| 3.85% | `_int_malloc` | **Allocator** | ⚠️ Memory allocations |
| 3.45% | `HVectorBase::reIndex` | **HiGHS** | Sparse vector ops |
| 3.28% | `HighsSparseMatrix::priceByColumn` | **HiGHS** | LP operations |
| 3.19% | `HFactor::updateFT` | **HiGHS** | Factorization update |
| 3.02% | `HighsSparseMatrix::update` | **HiGHS** | Matrix updates |
| 2.89% | `HFactor::btranL` | **HiGHS** | Backsolve |
| 2.86% | `HFactor::buildSimple` | **HiGHS** | Build factorization |
| 2.84% | `HFactor::ftranL` | **HiGHS** | Forward solve |
| 2.73% | `HFactor::btranU` | **HiGHS** | Backsolve upper |
| 2.64% | `HFactor::ftranFT` | **HiGHS** | Forwardsolve |
| 2.21% | `malloc` | **Allocator** | ⚠️ More allocations |
| 2.03% | `memset` | **Memory** | Zeroing memory |
| 2.01% | `solveHyper` | **HiGHS** | Hyper-graph solve |

### Analysis

**Good News** (60% of time):
- Most time spent in HiGHS solver (11.44% + others ≈ 60%)
- This is expected and largely unavoidable
- HiGHS is highly optimized C++ code

**Concerns** (15% of time):
1. **5.31%** in `std::_Rb_tree_increment` - **Red flag!**
   - This is std::map or std::set iteration
   - Likely: Cut pool management or state lookups
   - **Optimization opportunity**: Replace with Vec or HashMap
   
2. **6.06%** total in allocations (`_int_malloc` + `malloc`)
   - Allocating during hot path
   - **Optimization opportunity**: Pre-allocate buffers

3. **2.03%** in `memset` - Zeroing memory
   - Could be from `vec![0.0; n]` patterns
   - **Optimization opportunity**: Reuse buffers

---

## 💾 Memory Analysis (from massif)

### Peak Memory Usage

**Peak**: ~2.0 GB (at 353 seconds into execution)

### Top Allocation Sources

1. **25.35% (510 MB)** - `std::vector<double>::reserve`
   - **HiGHS internal** (6.33% = 127 MB)
   - **Our backward pass** (6.20% = 124 MB) ⚠️
   
2. **19.85% (400 MB)** - `std::vector::_M_realloc_insert`
   - Growing vectors incrementally
   - **Optimization**: Use `with_capacity`

3. **Growing allocation during backward pass**:
   - 62 MB in Rayon parallel backward
   - Growing with iteration count

### Key Finding

**6.20% (124 MB)** allocated in our backward pass parallel code:
```
rayon::iter::plumbing::bridge_producer_consumer
└─> powers_rs::sddp (backward pass)
    └─> Vec<(CutStatePair, BackwardPhase1Timing)>
```

**Problem**: Allocating result vectors in parallel workers.

**Optimization**: Pre-allocate or use memory pools.

---

## 🎯 Identified Bottlenecks

### Priority 1: Backward Pass Scaling (CRITICAL)

**Observation**: Time grows from 2.6s → 7.3s as cuts increase.

**Root Cause**: Likely cut selection overhead with growing pool.

**Evidence**:
- 5.31% time in `std::_Rb_tree_increment` (map/set iteration)
- Time scales with active cut count

**Recommended Fix**:
1. Profile backward pass in detail to confirm
2. If cut selection: Replace std::map with Vec + binary search
3. If state lookup: Use flat array with indexing

**Expected Impact**: 20-30% reduction in backward time.

---

### Priority 2: Memory Allocations (HIGH)

**Observation**: 6% of CPU time in malloc/memset.

**Root Cause**: Allocating in hot loops.

**Evidence from massif**:
- 124 MB allocated in backward pass per iteration
- Growing with thread count

**Recommended Fixes**:
1. **Pre-allocate buffers** in structs:
   ```rust
   struct BackwardPass {
       temp_buffer: Vec<f64>,  // Reuse across solves
       result_buffer: Vec<CutStatePair>,
   }
   ```

2. **Use `Vec::with_capacity`** when size known:
   ```rust
   let mut results = Vec::with_capacity(num_scenarios);
   ```

3. **Pool allocations** in Rayon workers:
   ```rust
   thread_local! {
       static BUFFER: RefCell<Vec<f64>> = ...;
   }
   ```

**Expected Impact**: 15-20% reduction in overall time.

---

### Priority 3: Forward Pass Optimization (MEDIUM)

**Observation**: Forward pass is fast (0.15s) but could be faster.

**Current Time**: ~150ms for 16 forward passes = 9.4ms per pass.

**Calculation**: 
- 16 forward passes × 60 stages = 960 solver calls
- 9.4ms / 60 stages = 0.16ms per stage
- With 156 hydros, this is acceptable

**Recommendation**: Focus on backward pass first. Forward is already efficient.

---

## 🚫 Why Flamegraph Failed

### The Problem

```
Warning: Processed 497762 samples and lost 80.08%!
Check IO/CPU overload!
```

### Root Cause

**Perf record couldn't keep up** with the data rate:
- CPU-intensive workload (20,160 solver calls)
- High sampling frequency
- Writing 3.8 GB of perf.data
- Likely I/O bottleneck on disk

### The Fix

**Option 1**: Reduce sampling frequency
```bash
perf record -F 99  # 99 Hz instead of default 1000 Hz
```

**Option 2**: Use perf report instead (we already have it!)
```bash
# We have valid perf report - just use that!
cat profiling_results/baseline_*/perf_report.txt
```

**Option 3**: Use cargo flamegraph with lower frequency
```bash
cargo flamegraph --freq 99 --bin powers -- example
```

**Option 4**: Profile shorter run
```bash
# Reduce iterations in config.json: 8 → 2
cargo flamegraph --bin powers -- examples/05-large-scale-brazilian
```

### What We Have

The **perf report is valid** and sufficient! It shows:
- Top 30 functions by CPU time
- Call graphs
- Allocation sources

**We don't need the flamegraph.** The data is in the perf report.

---

## 📋 Action Plan

### Phase 1: Validate Findings (~2 hours)

1. **Profile backward pass specifically**:
   ```bash
   # Add timing to confirm cut selection overhead
   perf record -e cpu-clock --call-graph dwarf \
     cargo run --release -- examples/05-large-scale-brazilian
   ```

2. **Check cut pool data structure**:
   ```bash
   grep -r "std::map\|BTreeMap\|BTreeSet" src/fcf.rs src/cut.rs
   ```

3. **Confirm allocation sites**:
   ```bash
   grep -r "vec!\[" src/sddp/mod.rs | grep "backward"
   ```

### Phase 2: Optimize Backward Pass (~1 week)

**Target**: Reduce backward time by 30% (7.3s → 5.1s)

1. **Replace std::map with Vec** (if confirmed)
   - Test: Does cut pool use BTreeMap?
   - Fix: Convert to `Vec<Cut>` with binary search
   - Validate: Benchmark + tests

2. **Pre-allocate buffers**
   - Identify allocations in backward loop
   - Add buffers to struct, reuse
   - Validate: Massif shows 80% reduction

3. **Optimize cut selection**
   - Current: O(n log n) per stage?
   - Target: O(k log n) with heap
   - Validate: Timing shows improvement

### Phase 3: Memory Optimizations (~3 days)

**Target**: Reduce allocations by 80%

1. **Buffer reuse in parallel**
   - Use thread-local storage
   - Pool common allocations

2. **Pre-size collections**
   - Replace `Vec::new()` with `Vec::with_capacity()`
   - Especially in loops

### Phase 4: Validate & Document (~1 day)

1. Run full benchmark suite
2. Verify correctness (all tests pass)
3. Document optimizations with data
4. Update profiling baseline

---

## 📊 Expected Results

| Metric | Baseline | Target | Strategy |
|--------|----------|--------|----------|
| **Backward time** | 7.3s | 5.1s (-30%) | Cut selection + buffers |
| **Memory allocs** | 510 MB | 100 MB (-80%) | Pre-allocation |
| **Total time** | 37.3s | 26-29s (-25-30%) | Combined |
| **Solver efficiency** | 540 calls/s | 700 calls/s (+30%) | Reduced overhead |

**Stretch goal**: If we can optimize forward pass too → **33% total improvement**.

---

## 🔧 Immediate Next Steps

1. **✅ DONE**: Understand profiling results (this document)

2. **TODO**: Examine cut pool implementation
   ```bash
   cat src/fcf.rs | grep -A 10 "struct.*Pool"
   cat src/cut.rs | grep -A 10 "struct.*Cut"
   ```

3. **TODO**: Profile backward pass in isolation
   ```bash
   # Create minimal test case that runs just backward pass
   # Profile that specifically
   ```

4. **TODO**: Create optimization branch
   ```bash
   git checkout -b perf/optimize-backward-pass
   ```

5. **TODO**: Implement first optimization (cut pool data structure)

---

## 💡 Key Insights

1. **Backward pass is 34x slower than forward** - This is THE bottleneck
2. **Time grows with cut count** - Cut management overhead is significant
3. **5.31% in tree iteration** - Likely suboptimal data structure (std::map)
4. **6% in allocations** - Low-hanging fruit for optimization
5. **HiGHS takes 60% of time** - Expected, can't optimize further
6. **Flamegraph failed but perf report worked** - We have the data we need!

---

## 🎯 Recommendation

**Start with Priority 1**: Backward pass cut pool optimization.

**Why?**
- Biggest impact (30% potential improvement)
- Clear evidence (5.31% in tree iteration)
- Isolated change (cut pool data structure)
- Low risk (well-defined interface)

**Steps**:
1. Confirm cut pool uses BTreeMap/BTreeSet
2. Benchmark current implementation
3. Replace with Vec + binary search
4. Validate correctness + performance
5. Document results

**Expected time**: 2-3 days for complete implementation and validation.

---

**Status**: ✅ Analysis complete - Ready to optimize
**Next**: Examine cut pool implementation
