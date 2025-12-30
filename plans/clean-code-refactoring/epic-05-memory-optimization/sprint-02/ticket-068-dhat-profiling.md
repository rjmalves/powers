# [T-068] DHAT Profiling to Verify Zero Allocations

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Training Loop Integration](./00-sprint-overview.md)
> **Dependencies**: [T-065](./ticket-065-wire-backward-pass.md)
> **Blocks**: None

## Files to Read Before Starting

- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Architecture goals
- `src/fcf.rs:830-837` - Deprecated `CutData::from_refs()`

---

## Context

### Background

DHAT (Dynamic Heap Analysis Tool) from Valgrind tracks every heap allocation. We use it to verify that the new path produces zero allocations in the cut computation hot path.

### What We're Looking For

**Before (old path)**:
```
~7,500 allocations in CutData::from_refs()
```

**After (new path)**:
```
0 allocations in cut computation
```

---

## Specification

### DHAT Profiling Steps

1. **Install DHAT** (part of Valgrind):
   ```bash
   sudo apt-get install valgrind
   ```

2. **Build with debug symbols**:
   ```bash
   cargo build --release
   # or with debug info:
   RUSTFLAGS="-C debuginfo=2" cargo build --release
   ```

3. **Run with DHAT**:
   ```bash
   valgrind --tool=dhat ./target/release/powers run examples/02-stochastic \
     --seed 42 --iterations 10 2>&1 | tee dhat-output.txt
   ```

4. **Analyze output**:
   ```bash
   grep -i "CutData\|from_refs\|to_vec" dhat-output.txt
   ```

---

## Acceptance Criteria

- [ ] DHAT profile captured
- [ ] No allocations from `CutData::from_refs()`
- [ ] No allocations from `to_vec()` in cut computation
- [ ] Profile report saved for documentation
- [ ] Comparison with baseline (before T-065)

---

## Implementation Guide

### Step 1: Baseline profile (old path)

```bash
# Temporarily revert to old path
git stash

# Build
cargo build --release

# Profile
valgrind --tool=dhat --dhat-out-file=dhat-before.txt \
  ./target/release/powers run examples/02-stochastic --seed 42 --iterations 10

# Restore new path
git stash pop
```

### Step 2: New path profile

```bash
# Build new path
cargo build --release

# Profile
valgrind --tool=dhat --dhat-out-file=dhat-after.txt \
  ./target/release/powers run examples/02-stochastic --seed 42 --iterations 10
```

### Step 3: Compare allocations

```bash
# Count CutData allocations in baseline
grep -c "CutData" dhat-before.txt

# Should be 0 in new path
grep -c "CutData" dhat-after.txt
```

### Step 4: Detailed analysis

Use DHAT viewer for visual analysis:
```bash
# Install dhat-viewer if needed
cargo install dhat-viewer

# View results
dhat-viewer dhat-after.txt
```

Look for:
- Allocation site: `src/fcf.rs` `from_refs` → Should be 0
- Allocation site: `copy_from_slice` → Should be 0 (copies, not allocates)

### Step 5: Document findings

Create summary:

```markdown
## DHAT Profile Results

### Baseline (Old Path)
- Total allocations: X
- CutData allocations: Y
- Average per iteration: Z

### New Path
- Total allocations: X'
- CutData allocations: 0 ✅
- Reduction: (X - X') allocations eliminated

### Remaining Allocations
[List any remaining allocations and whether they're acceptable]
```

---

## Testing Requirements

### Verify Specific Functions

| Function | Expected Allocations |
|----------|---------------------|
| `CutData::from_refs()` | 0 |
| `CutData::new()` | 0 |
| `.to_vec()` in state.rs cut path | 0 |
| `CutStagingBuffer::new()` | 1 per handler (init only) |
| Thread-local buffer init | 1 per thread (init only) |

### Acceptable Allocations

- One-time initialization (buffers, pools)
- LP solver internal allocations
- Result collection (final output)

### Unacceptable Allocations

- Any allocation per cut
- Any allocation per stage
- Any allocation per iteration in hot path

---

## Alternative: Debug Assertions

If DHAT is unavailable, add debug assertions:

```rust
// In CutData::from_refs
#[cfg(debug_assertions)]
{
    static CALL_COUNT: std::sync::atomic::AtomicUsize = 
        std::sync::atomic::AtomicUsize::new(0);
    let count = CALL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if count > 0 {
        panic!("CutData::from_refs called {} times - should be 0 in production!", count);
    }
}
```

---

## Pitfalls to Avoid

- ⚠️ **DHAT overhead**: Profile runs ~10x slower
- ⚠️ **Symbol names**: Need debug symbols for readable output
- ⚠️ **False positives**: Distinguish hot-path allocations from init allocations

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Straightforward profiling if tools available. Analysis may take time.

---

## Definition of Done

- [ ] DHAT profile captured
- [ ] Zero CutData allocations confirmed
- [ ] Profile report documented
- [ ] Comparison with baseline included
