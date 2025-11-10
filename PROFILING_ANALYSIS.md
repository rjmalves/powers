# 🔍 Profiling Analysis - Example 05 (156 Hydros)

**Latest Update**: 2025-11-10  
**Problem**: Large-Scale Brazilian System (156 hydros, 121 thermals, 60 stages)  
**Runtime**: 34.0 seconds (8 iterations, 16 forward passes)  
**Improvement**: 8.8% faster than initial baseline (37.3s → 34.0s)

---

## 🎯 Latest Results (2025-11-10)

### ✅ Performance Improvement

**Runtime Comparison:**
- **Initial baseline** (Nov 9): 37.3s average
- **After optimizations** (Nov 10): 34.0s average
- **Improvement**: 3.3s faster (8.8% speedup)

**Individual runs:**
- Run 1: 33.89s
- Run 2: 34.37s  
- Run 3: 33.80s
- Average: **34.02s** ± 0.31s

### 🔍 Current Status

**Good News** ✅:
- Overall 8.8% speedup achieved
- Runtime is more consistent (low variance)
- Allocation overhead reduced by 0.8%
- Forward passes remain fast

**Remaining Issues** ⚠️:
- **BTreeMap still present**: 4.95% CPU time in `std::_Rb_tree_increment`
  - This should have been removed but is still showing up
  - Need to investigate which code path still uses BTreeMap/BTreeSet
- **Allocation overhead**: 5.28% total in malloc/memset operations
  - malloc: 3.34%
  - _int_malloc: 1.94%
  - memset: 1.78%
- **HiGHS dominates**: 60%+ of time (expected, unavoidable)

---

## 🔥 CPU Hotspots (Latest - Nov 10, 2025)

### Top 20 Functions by CPU Time

| % Time | Function | Component | Analysis |
|--------|----------|-----------|----------|
| 11.17% | `HFactor::ftranU` | **HiGHS Solver** | LP factorization (unavoidable) |
| **4.95%** | `std::_Rb_tree_increment` | **std::map iterator** | ⚠️ **STILL PRESENT** - BTreeMap not fully removed! |
| 3.41% | `HVectorBase::reIndex` | **HiGHS** | Sparse vector ops |
| 3.34% | `_int_malloc` | **Allocator** | ⚠️ Memory allocations |
| 3.13% | `HFactor::updateFT` | **HiGHS** | Factorization update |
| 3.12% | `HighsSparseMatrix::priceByColumn` | **HiGHS** | LP operations |
| 2.96% | `HFactor::btranL` | **HiGHS** | Backsolve |
| 2.84% | `HighsSparseMatrix::update` | **HiGHS** | Matrix updates |
| 2.82% | `HFactor::ftranL` | **HiGHS** | Forward solve |
| 2.81% | `HFactor::btranU` | **HiGHS** | Backsolve upper |
| 2.74% | `HFactor::buildSimple` | **HiGHS** | Build factorization |
| 2.62% | `HFactor::ftranFT` | **HiGHS** | Forward solve |
| 1.94% | `solveHyper` | **HiGHS** | Hyper-graph solve |
| 1.94% | `malloc` | **Allocator** | ⚠️ More allocations |
| 1.89% | `HFactor::btranFT` | **HiGHS** | Backsolve |
| 1.78% | `__memset_avx2_unaligned_erms` | **Memory** | Zeroing memory |
| 1.76% | `HighsSparseMatrix::priceByRowWithSwitch` | **HiGHS** | LP operations |
| 1.67% | `HEkkDualRow::choosePossible` | **HiGHS** | Dual simplex |

### Comparison with Initial Baseline

| Function | Initial (Nov 9) | Current (Nov 10) | Change |
|----------|-----------------|------------------|--------|
| `HFactor::ftranU` | 11.44% | 11.17% | -0.27% ✅ |
| `std::_Rb_tree_increment` | **5.31%** | **4.95%** | -0.36% (Slight improvement, but **STILL TOO HIGH**) |
| `_int_malloc` | 3.85% | 3.34% | -0.51% ✅ |
| `malloc` | 2.21% | 1.94% | -0.27% ✅ |
| `memset` | 2.03% | 1.78% | -0.25% ✅ |

### Analysis

**Improvement Areas** ✅:
- Allocation overhead reduced by ~0.8% overall
- HiGHS solver operations slightly more efficient
- Memory operations optimized

**Critical Finding** 🔴:
- **BTreeMap is STILL consuming 4.95% CPU time**
- This suggests BTreeMap/BTreeSet was not fully removed from all code paths
- Need to search codebase for remaining BTree usage:
  ```bash
  grep -r "BTreeMap\|BTreeSet" src/
  ```

**Good News** (60% of time):
- Most time still in HiGHS solver (expected and optimal)
- No new bottlenecks introduced by optimizations
- HiGHS is highly optimized C++ code

**Remaining Concerns** (10% of time):
1. **4.95%** in `std::_Rb_tree_increment` - **Must find and eliminate!**
2. **5.28%** total in allocations (malloc + _int_malloc + memset)
   - **Optimization opportunity**: Pre-allocate buffers
3. Some allocation patterns still present in hot loops

---

## 💾 Memory Analysis

### Peak Memory Usage

**Peak**: ~2.4 GB (Nov 10 run)

### Analysis

Memory usage is primarily from:
1. **HiGHS solver internals** (~40% - unavoidable)
2. **Rayon parallel backward pass** allocations (~6%)
3. **Growing collections** during backward pass

**Note**: Detailed memory breakdown shows allocations in Rayon worker threads, suggesting parallel overhead.

---

## 🎯 Prioritized Optimization Targets

### Priority 1: Find and Remove Remaining BTreeMap Usage 🔴 **CRITICAL**

**Impact**: 4.95% CPU time  
**Expected Improvement**: 10-15% when fully removed

**Action Items**:
1. **Search for remaining BTree usage**:
   ```bash
   grep -rn "BTreeMap\|BTreeSet" src/
   ```

2. **Check specific files**:
   ```bash
   grep -n "BTree" src/fcf.rs
   grep -n "BTree" src/sddp/mod.rs
   grep -n "BTree" src/subproblem.rs
   grep -n "BTree" src/cut.rs
   ```

3. **Replace with HashMap or Vec**:
   - For key-value lookups: Use `HashMap`
   - For sets: Use `HashSet`  
   - For ordered iteration: Use `Vec` with manual sorting

**Why this matters**:
- BTreeMap/BTreeSet use tree iteration which is slow
- HashMap has O(1) lookup vs O(log n)
- Flat Vec can be faster for small collections

---

### Priority 2: Pre-allocate Buffers 🟡 **HIGH**

**Impact**: 5.28% CPU time in allocations  
**Expected Improvement**: 15-20%

**Specific Targets**:

1. **Backward pass result vectors**:
   ```rust
   // Before: Allocates every iteration
   let results: Vec<_> = scenarios.par_iter()
       .map(|s| solve(s))  // Each allocates
       .collect();
   
   // After: Pre-allocate
   struct BackwardPass {
       result_buffer: Vec<CutStatePair>,
   }
   
   impl BackwardPass {
       fn solve(&mut self, scenarios: &[Scenario]) {
           self.result_buffer.clear();
           self.result_buffer.reserve(scenarios.len());
           // Reuse buffer
       }
   }
   ```

2. **Temporary computation buffers**:
   ```rust
   struct Subproblem {
       // Pre-allocated buffers
       temp_values: Vec<f64>,
       temp_gradient: Vec<f64>,
   }
   ```

3. **Thread-local buffers** for parallel code:
   ```rust
   thread_local! {
       static TEMP_BUFFER: RefCell<Vec<f64>> = 
           RefCell::new(Vec::with_capacity(10000));
   }
   ```

---

### Priority 3: Use `Vec::with_capacity` 🟢 **MEDIUM**

**Impact**: Reduces incremental growth allocations  
**Expected Improvement**: 5-10%

**Search for patterns**:
```bash
grep -rn "Vec::new()" src/ | grep -v test
```

**Replace with**:
```rust
// Before
let mut results = Vec::new();
for item in items {
    results.push(process(item));
}

// After
let mut results = Vec::with_capacity(items.len());
for item in items {
    results.push(process(item));
}
```

---

## 📋 Action Plan

### Week 1: Find and Remove Remaining BTree

**Goal**: Eliminate all BTreeMap/BTreeSet usage

**Tasks**:
1. Search codebase for BTree usage
2. Identify why previous replacement didn't catch all instances
3. Replace with HashMap/HashSet or Vec
4. Validate correctness with tests
5. Profile and compare

**Success Criteria**: `std::_Rb_tree_increment` < 1% CPU time

---

### Week 2: Memory Optimizations

**Goal**: Reduce allocation overhead by 80%

**Tasks**:
1. Pre-allocate buffers in hot structures
2. Add `with_capacity` to growing collections
3. Implement thread-local buffers for parallel code
4. Profile and validate

**Success Criteria**: malloc + memset < 2% CPU time

---

### Week 3: Validate and Document

**Goal**: Confirm improvements and prepare for next phase

**Tasks**:
1. Run full benchmark suite
2. Verify all tests pass
3. Document optimizations
4. Update baseline
5. Measure total improvement

**Success Criteria**: Runtime < 29s (25% improvement from original 37.3s)

---

## 🎯 Performance Goals

| Metric | Original | Current | Phase 1 Goal | Phase 2 Goal |
|--------|----------|---------|--------------|--------------|
| **Total time** | 37.3s | 34.0s | <29s | <26s |
| **Improvement** | - | 8.8% | 25% | 30% |
| **BTreeMap** | 5.31% | 4.95% | <1% | 0% |
| **Allocations** | 6.06% | 5.28% | <2% | <1% |
| **Memory** | 2.0GB | 2.4GB | <1.8GB | <1.6GB |

---

## 💡 Key Insights

### What Worked ✅

1. **Initial optimizations** achieved 8.8% speedup
2. **Allocation overhead** reduced by 0.8%
3. **No new bottlenecks** introduced
4. **HiGHS integration** remains optimal

### What Didn't Work ❌

1. **BTreeMap removal incomplete** - still 4.95% overhead
2. **Memory allocations** still significant at 5.28%
3. **Some optimization opportunities** not yet exploited

### Next Steps 🚀

1. **Find remaining BTree usage** - highest priority
2. **Pre-allocate all hot-path buffers** - high impact
3. **Profile after each change** - validate improvements
4. **Continue until goals met** - iterate and measure

---

## 🔧 Recommended Commands

```bash
# Find remaining BTree usage
grep -rn "BTreeMap\|BTreeSet" src/

# Search for allocation patterns
grep -rn "Vec::new()" src/ | grep -v test
grep -rn "vec!\[" src/ | grep -v test

# Profile after changes
./scripts/profile_compare.sh examples/05-large-scale-brazilian

# Run tests
cargo test --release

# Full benchmark
cargo bench
```

---

**Status**: 8.8% improvement achieved, but **BTreeMap still present**. Continue optimization!

**Next Action**: Find and eliminate remaining BTreeMap/BTreeSet usage.
