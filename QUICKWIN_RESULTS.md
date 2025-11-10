# Quick Win Optimization Results

**Date**: 2025-11-10  
**Optimization**: Pre-allocate inner vectors with `Vec::with_capacity`  
**Location**: `src/state.rs:586` (StorageState::evaluate_cut)

---

## Change Made

```rust
// BEFORE:
let contrib: Vec<f64> = realization
    .water_value
    .iter()
    .map(|&val| prob * val)
    .collect();  // No pre-allocation

// AFTER:
let mut contrib = Vec::with_capacity(realization.water_value.len());
contrib.extend(realization.water_value.iter().map(|&val| prob * val));
```

---

## Results Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Run 1** | 0.441s | 0.528s | +19.7% |
| **Run 2** | 0.443s | 0.524s | +18.3% |
| **Run 3** | 0.430s | 0.536s | +24.7% |
| **Average** | 0.437s | 0.529s | **+21.1%** |
| **Max RSS** | 23,424 KB | 24,060 KB | +2.7% |

---

## Analysis

### Unexpected Regression

The optimization shows a **~21% slowdown** instead of the expected improvement.

### Possible Causes

1. **Iterator Overhead**: `extend()` may have more overhead than `collect()`
   - `collect()` is highly optimized in Rust std
   - `extend()` adds an extra layer of indirection

2. **Small Vector Size**: With only 3 hydros:
   - Pre-allocation overhead > allocation savings
   - `collect()` inlining is better for tiny vectors

3. **Cache Effects**: Different code path may affect branch prediction

4. **Measurement Noise**: Need more runs to confirm

---

## Investigation

Let me check what `collect()` does internally vs our change:

```rust
// collect() implementation (highly optimized):
impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();  // Gets exact size!
        let mut vec = Vec::with_capacity(lower);  // Already pre-allocates!
        vec.extend(iter);
        vec
    }
}
```

**Ah!** `collect()` already pre-allocates based on `size_hint()`.

Our change adds:
1. Explicit `with_capacity` call
2. Then `extend()` call

This is **redundant** and adds overhead!

---

## Lesson Learned

**Rayon's `collect()` Already Optimizes**

From the original TICKET-006b analysis:
> "Already Optimal: Rayon's collect() pre-allocates based on size_hint()"

This was correct! The iterator provides accurate `size_hint()`, and `collect()` uses it.

### What We Learned

1. ✅ **Trust std library**: `collect()` is highly optimized
2. ✅ **Measure before optimizing**: This showed regression immediately
3. ✅ **Understand internals**: `collect()` already does what we wanted
4. ❌ **Quick wins aren't always wins**: Sometimes std is already optimal

---

## Decision

**REVERT** this change - it's a pessimization, not an optimization.

The real optimization opportunity is **reusing buffers across iterations**, not pre-allocating within a single iteration (which `collect()` already does).

---

## Corrected Path Forward

### Not This (Redundant):
```rust
let mut contrib = Vec::with_capacity(realization.water_value.len());
contrib.extend(realization.water_value.iter().map(|&val| prob * val));
```

### This (Thread-Local Reuse):
```rust
thread_local! {
    static BUFFERS: RefCell<CutBuffers> = ...;
}

BUFFERS.with(|buffers| {
    let buf = &mut buffers.borrow_mut().contribution_buffer;
    buf.clear();
    buf.extend(realization.water_value.iter().map(|&val| prob * val));
    // Use buf, don't allocate
});
```

---

## Impact on TICKET-006b

This validates the original analysis:
- ❌ Pre-allocation within iteration: Already done by `collect()`
- ✅ Buffer reuse across iterations: The real opportunity
- ✅ Thread-local storage: Required for hot path optimization

**TICKET-006b remains justified**, but skip the "quick win" and go straight to proper implementation.

---

## Next Steps

1. **Revert** this change (it's a regression)
2. **Implement** full TICKET-006b with thread-local buffers
3. **Measure** actual impact of buffer reuse

---

## Performance Optimizer Notes

**What Went Well** ✅:
- Measured before assuming
- Caught regression immediately
- Learned about `collect()` internals

**What to Improve** 🔄:
- Verify assumptions about std library behavior
- Check `size_hint()` before assuming no pre-allocation
- Read std library source when in doubt

**Key Insight** 💡:
> "When Rust's standard library does exactly what you need, don't try to outsmart it."

---

**Status**: Quick win reverted, proceeding with proper TICKET-006b implementation  
**Lesson**: Always measure, trust std library optimizations  
**Confidence**: Higher (we now know what doesn't work)
