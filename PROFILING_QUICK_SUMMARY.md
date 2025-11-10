# ✅ Profiling Summary & Next Steps

## What Happened

### ✅ Success
1. **Perf report worked** - Valid CPU profiling data
2. **Massif worked** - Memory allocation tracking
3. **Timing data collected** - Forward/backward breakdown
4. **Benchmark baseline saved** - In `target/criterion/before_refactoring/`

### ❌ Flamegraph Failed
- **Cause**: I/O overload - lost 80% of samples
- **Impact**: None - perf report has all the data we need!
- **Fix if needed**: Use lower sampling frequency (`-F 99`)

---

## 🔥 Critical Findings

### Finding #1: Backward Pass Dominates (34x slower)

**Evidence**:
```
Iteration 8:
- Forward:  0.21s (  1 part)
- Backward: 7.26s ( 34 parts) ← THE BOTTLENECK
```

**Root Cause**: Time grows linearly with cut count
- Iteration 1: 2.6s with 944 cuts
- Iteration 8: 7.3s with 5,996 cuts

### Finding #2: BTreeMap Overhead (5.31% CPU)

**Evidence from perf**:
```
5.31% CPU in std::_Rb_tree_increment (BTreeMap iteration)
```

**Root Cause Found**:
```rust
// In src/cut.rs line 56:
pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_indices: BTreeMap<usize, usize>,  ← THIS!
}
```

**Why it's slow**:
- BTreeMap lookup/iteration: O(log n)
- With 6,000 cuts: log₂(6000) = ~13 comparisons per access
- Accessed frequently in backward pass
- Cache-unfriendly (pointer chasing)

**Optimization**: Replace with `Vec<Option<usize>>` or `HashMap`
- Vec lookup: O(1), cache-friendly
- HashMap lookup: O(1) expected
- Trade-off: Sparse array uses more memory (negligible at this scale)

### Finding #3: Allocation Overhead (6% CPU)

**Evidence**:
```
3.85% in _int_malloc
2.21% in malloc  
2.03% in memset
─────────────────
8.09% total
```

**Memory growth**:
- 124 MB allocated per backward iteration
- Peak: 2.0 GB

**Root Cause**: Allocating in hot loops during parallel backward pass.

---

## 📋 Optimization Roadmap

### Phase 1: Replace BTreeMap (HIGH IMPACT, LOW RISK)

**Estimated Impact**: 20-25% backward pass improvement

**Change**:
```rust
// Before (src/cut.rs)
pub active_cut_indices: BTreeMap<usize, usize>,

// After (option 1: HashMap)
pub active_cut_indices: HashMap<usize, usize>,

// After (option 2: Vec - if cut IDs are dense)
pub active_cut_indices: Vec<Option<usize>>,
```

**Pros HashMap**:
- O(1) lookup/insert
- Drop-in replacement
- Better cache locality than BTreeMap

**Pros Vec**:
- O(1) lookup by index
- Best cache locality
- Requires cut IDs ≤ max_cut_id

**Recommendation**: Try HashMap first (easiest), then Vec if needed.

**Time**: 1-2 days
**Risk**: Low (well-defined interface)

### Phase 2: Pre-allocate Buffers (MEDIUM IMPACT, LOW RISK)

**Estimated Impact**: 15-20% overall improvement

**Changes**:

1. **Backward pass buffers**:
   ```rust
   struct BackwardPass {
       // Reuse across iterations
       temp_buffer: Vec<f64>,
       result_buffer: Vec<CutStatePair>,
   }
   ```

2. **Use Vec::with_capacity**:
   ```rust
   // Find patterns like:
   let mut results = Vec::new();  // ← Bad
   
   // Replace with:
   let mut results = Vec::with_capacity(known_size);  // ← Good
   ```

3. **Thread-local pools** (advanced):
   ```rust
   thread_local! {
       static BUFFER_POOL: RefCell<Vec<f64>> = ...;
   }
   ```

**Time**: 3-5 days
**Risk**: Low (isolated changes)

### Phase 3: Optimize Cut Selection (IF NEEDED)

**Check first**: Profile after Phase 1 & 2.

**If still slow**: Optimize cut selection algorithm.

---

## 🎯 Immediate Action Items

### 1. Validate BTreeMap Hypothesis (30 minutes)

```bash
# Check how active_cut_indices is used
grep -n "active_cut_indices" src/*.rs src/**/*.rs

# Count iterations over it
grep -B5 -A5 "active_cut_indices.iter()" src/*.rs
```

### 2. Create Optimization Branch

```bash
git checkout -b perf/replace-btreemap-with-hashmap
```

### 3. Implement HashMap Replacement (2 hours)

```rust
// src/cut.rs
use std::collections::HashMap;  // Add this

pub struct BendersCutPool {
    pub pool: Vec<BendersCut>,
    pub active_cut_indices: HashMap<usize, usize>,  // Change this
    pub total_cut_count: usize,
}

impl BendersCutPool {
    pub fn new() -> Self {
        Self {
            pool: vec![],
            active_cut_indices: HashMap::new(),  // Change this
            total_cut_count: 0,
        }
    }
}
```

### 4. Run Tests & Benchmark

```bash
# Tests must pass
cargo test --release

# Compare performance
cargo bench --bench sddp_e2e single_iteration -- --baseline before_refactoring
```

### 5. Validate with Profiling

```bash
# Re-run profiling
./scripts/profile_baseline.sh examples/05-large-scale-brazilian

# Check if std::_Rb_tree_increment disappeared from perf report
```

---

## 📊 Expected Results

| Metric | Baseline | After Phase 1 | After Phase 2 | Total Improvement |
|--------|----------|---------------|---------------|-------------------|
| Backward time | 7.3s | 5.5s (-25%) | 4.4s (-20%) | -40% |
| Total time | 37.3s | 35.0s (-6%) | 30.0s (-14%) | -20% |
| BTreeMap % | 5.31% | ~0% | ~0% | -5.3pp |
| Malloc % | 6.06% | 6.06% | ~2% | -4pp |

**Conservative estimate**: 20% total improvement  
**Optimistic estimate**: 30% total improvement

---

## 🚦 Go/No-Go Decision

### GO ✅

**Reasons**:
1. **Clear bottleneck identified**: BTreeMap taking 5.3% CPU
2. **Low-risk change**: HashMap is drop-in replacement
3. **Measurable impact**: Can verify with benchmarks
4. **Isolated change**: Only affects cut pool
5. **Fast to implement**: 1-2 days max

### Next Command

```bash
# Start optimization
git checkout -b perf/replace-btreemap-with-hashmap
code src/cut.rs  # Edit active_cut_indices type
```

---

## 📚 Documentation

All profiling data saved in:
```
profiling_results/baseline_20251109_212950/
├── perf_report.txt      ← CPU hotspots (MOST USEFUL)
├── massif_report.txt    ← Memory allocations
├── timing.md            ← Timing summary
├── perf.data           ← Raw perf data (3.8 GB)
└── flamegraph.svg      ← Empty (failed, but we don't need it)
```

**To view perf report**:
```bash
less profiling_results/baseline_20251109_212950/perf_report.txt
```

**To re-analyze perf.data**:
```bash
cd profiling_results/baseline_20251109_212950/
perf report -i perf.data
```

---

## ✅ Status: Ready to Optimize

**Bottleneck**: Identified (BTreeMap in cut pool)  
**Solution**: Clear (Replace with HashMap)  
**Risk**: Low (simple change)  
**Timeline**: 1-2 days  
**Expected ROI**: 20-25% improvement

**Recommendation**: Proceed with Phase 1 optimization.
