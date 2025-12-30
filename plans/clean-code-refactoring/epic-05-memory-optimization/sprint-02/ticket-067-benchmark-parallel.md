# [T-067] Benchmark Parallel vs Sequential

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Training Loop Integration](./00-sprint-overview.md)
> **Dependencies**: [T-065](./ticket-065-wire-backward-pass.md)
> **Blocks**: None

## Files to Read Before Starting

- `benches/sddp_e2e.rs` - Existing benchmark
- `src/algorithm/backward_pass.rs` - Training loop

---

## Context

### Background

The new parallel-then-sequential path must not regress performance. It should:
- Preserve parallelism in Phase 1a (cut computation)
- Add minimal overhead in Phase 1b (sequential copy)

This ticket benchmarks both paths to verify.

---

## Specification

### Benchmark Configuration

1. **Hardware**: Note CPU cores, memory
2. **Problem size**: Use `05-large-scale-brazilian` or similar
3. **Thread count**: Test with 1, 4, 8, 16 threads
4. **Iterations**: Sufficient for stable measurements (10-20)

### Metrics to Capture

- Total training time
- Backward pass time
- Phase 1 time (cut computation)
- Phase 2 time (cut selection)
- Cuts per second

---

## Acceptance Criteria

- [ ] Benchmark run with old path (baseline)
- [ ] Benchmark run with new path
- [ ] No regression (new path ≥ old path speed)
- [ ] Results documented
- [ ] Parallel execution confirmed via timing

---

## Implementation Guide

### Step 1: Run baseline benchmark

```bash
# Checkout code before T-065 or use old method
git stash  # Save changes

cargo bench --bench sddp_e2e -- --save-baseline before

git stash pop  # Restore changes
```

### Step 2: Run new path benchmark

```bash
cargo bench --bench sddp_e2e -- --baseline before
```

### Step 3: Compare results

Criterion will show comparison automatically:
```
training_loop/backward_pass  time: [X ms Y ms Z ms]
                             change: [-5.2% -3.1% -1.0%] (p = 0.01 < 0.05)
                             Performance has improved.
```

### Step 4: Vary thread count

```bash
for threads in 1 4 8 16; do
  echo "=== $threads threads ==="
  RAYON_NUM_THREADS=$threads cargo bench --bench sddp_e2e 2>&1 | grep -A2 "backward_pass"
done
```

### Step 5: Document results

Create summary:

```markdown
## Benchmark Results

**Hardware**: [CPU model], [cores], [RAM]
**Problem**: 05-large-scale-brazilian

| Threads | Old Path (ms) | New Path (ms) | Change |
|---------|---------------|---------------|--------|
| 1       | X             | Y             | -Z%    |
| 4       | X             | Y             | -Z%    |
| 8       | X             | Y             | -Z%    |
| 16      | X             | Y             | -Z%    |

**Conclusion**: New path shows [improvement/parity/regression].
```

---

## Testing Requirements

### Expected Results

- **1 thread**: New path should be faster (no allocation overhead)
- **N threads**: New path should maintain parallel speedup
- **Phase 1b overhead**: Should be <1% of Phase 1 time

### Red Flags

- New path slower than old path
- Parallel speedup worse with new path
- Phase 1b taking significant time

---

## Pitfalls to Avoid

- ⚠️ **Warm-up**: Run benchmarks multiple times for warm cache
- ⚠️ **System load**: Run on quiet system
- ⚠️ **Memory pressure**: Ensure sufficient RAM
- ⚠️ **Power management**: Disable CPU throttling if possible

---

## Effort Estimate

**Points**: 3  
**Confidence**: High  
**Rationale**: Benchmark infrastructure exists. Main work is running and documenting.

---

## Definition of Done

- [ ] Baseline captured
- [ ] New path benchmarked
- [ ] No regression confirmed
- [ ] Results documented
- [ ] PR includes benchmark summary
