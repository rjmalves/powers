# TICKET-006b Performance Results

**Date**: 2025-11-10  
**Status**: ⚠️ UNEXPECTED REGRESSION  
**Implementation**: COMPLETE (Phases 1-3)

---

## Results Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Run 1** | 0.441s | 0.537s | **+21.8%** |
| **Run 2** | 0.443s | 0.547s | **+23.5%** |
| **Run 3** | 0.430s | 0.570s | **+32.6%** |
| **Average** | **0.437s** | **0.551s** | **+26.1% SLOWER** |
| **Max RSS** | 23,424 KB | 24,256 KB | +3.5% |

---

## Analysis

### Unexpected Regression

The optimization shows a **26% slowdown** instead of the expected 10-15% improvement.

### Possible Causes

1. **Thread-local overhead**: `thread_local!` access may have overhead
2. **Borrow checking**: `RefCell::borrow_mut()` has runtime cost
3. **Auto-initialization**: Lazy init check on every call
4. **Buffer management**: `reset_for_cut()` overhead
5. **Cache effects**: Different memory layout affects cache hits
6. **Small problem size**: Overhead > savings for tiny example (3 hydros)

### What We Did Right

✅ All 500 tests passing  
✅ Numerical correctness preserved  
✅ Thread-safe implementation  
✅ Clean API design  
✅ Comprehensive testing  

### What Went Wrong

The optimization added more overhead than it saved allocations:

**Added Overhead**:
- `thread_local!` access on every `with_cut_buffers` call
- `RefCell::borrow_mut()` runtime borrow checking
- Lazy initialization check
- `reset_for_cut()` clearing and resizing buffers
- Function call overhead (closure)

**Savings**:
- Eliminated ~2,048 allocations per run
- But for small vectors (3 elements), allocation is very fast!

### The Core Issue

**For small problems** (3 hydros, 4 scenarios):
- Allocation is fast (~10-20ns per small Vec)
- Thread-local + RefCell overhead is comparable
- Buffer management overhead negates savings

**For large problems** (156 hydros, 100+ scenarios):
- Allocation would be expensive
- Buffer reuse would dominate
- Should see expected 10-15% improvement

---

## Investigation Steps

### 1. Measure Thread-Local Overhead

Let me check if thread-local access is the bottleneck:

```bash
# Benchmark thread_local access
cargo bench --bench buffer_overhead
```

### 2. Test on Large Problem

The optimization should shine on large systems:

```bash
./scripts/profile_allocations_simple.sh examples/05-large-scale-brazilian
```

### 3. Profile with Flamegraph

See where time is actually spent:

```bash
cargo flamegraph --bin powers -- examples/03-multistage
```

### 4. Micro-benchmark Allocations

Compare allocation vs buffer reuse:

```rust
// Small vec allocation
let v: Vec<f64> = vec![0.0; 3];  // ~10-20ns

// Thread-local + RefCell
with_cut_buffers(|buf| {  // ~30-50ns?
    buf.reset_for_cut(3, 4);
});
```

---

## Hypothesis

**Small Problem Overhead Dominates**:
- 3 hydros × 4 scenarios = tiny vectors
- Allocating 12 f64s (96 bytes) is ~15ns
- Thread-local + RefCell + reset is ~40ns
- **Overhead > Savings for small problems**

**Expected for Large Problems**:
- 156 hydros × 100 scenarios = large vectors
- Allocating 15,600 f64s (125KB) is ~500ns
- Thread-local + RefCell + reset is still ~40ns
- **Savings > Overhead for large problems**

---

## Next Actions

### Immediate (Today)

1. ⏳ **Test on large problem** - Verify hypothesis
2. ⏳ **Profile to find bottleneck** - Flamegraph analysis
3. ⏳ **Micro-benchmark** - Measure actual overhead

### If Hypothesis Confirmed

**Option A**: Keep optimization (helps large problems)
- Document: "Optimized for large systems (50+ hydros)"
- Small problems may see slight regression
- Large problems see 10-15% improvement

**Option B**: Make optimization conditional
- Check problem size at runtime
- Use buffers only if `num_hydros * num_scenarios > threshold`
- Best of both worlds

**Option C**: Reduce overhead
- Use `static` instead of `thread_local!` where possible
- Remove `RefCell` if we can prove single-threaded access
- Optimize `reset_for_cut()` to be faster

### If Hypothesis Wrong

- Revert optimization
- Document learnings in QUICKWIN_RESULTS.md style
- Find different optimization target

---

## Performance Optimizer Notes

**What We Learned**:
1. ✅ **Measure actual impact** - Caught regression immediately
2. ✅ **Small != Large** - Optimization depends on problem size
3. ✅ **Overhead matters** - Thread-local + RefCell has cost
4. ⚠️ **Profile first** - Should have benchmarked overhead before implementing

**Process Was Correct**:
- ✅ Established baseline
- ✅ Implemented carefully
- ✅ All tests passing
- ✅ Measured impact
- ✅ Analyzing root cause

**The optimization is sound, execution may need refinement**

---

## Conclusion

Implementation is **technically correct** but shows **unexpected regression on small problems**.

**Status**: Need to test on large problem to validate hypothesis.

**Confidence**: Medium (implementation correct, performance unclear)

**Next**: Test on 156-hydro system to see if optimization helps where it matters.

---

**Last Updated**: 2025-11-10 (after initial profiling)
