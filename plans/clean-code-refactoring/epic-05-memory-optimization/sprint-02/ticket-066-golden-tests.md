# [T-066] Golden Tests Validation

> **Epic**: [Epic 5: Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 2: Training Loop Integration](./00-sprint-overview.md)
> **Dependencies**: [T-065](./ticket-065-wire-backward-pass.md)
> **Blocks**: None

## Files to Read Before Starting

- `scripts/golden-tests.sh` - Golden test script (if exists)
- `examples/` - Example problems used for golden tests
- `docs/PARALLEL_ZERO_ALLOCATION_ARCHITECTURE.md` - Context

---

## Context

### Background

Golden tests verify that training produces bit-for-bit identical results before and after refactoring. This is critical for SDDP where numerical reproducibility is required.

### Why This Matters

Any change to cut computation, even "equivalent" refactoring, could introduce:
- Floating-point ordering differences
- Numerical precision variations
- Timing-dependent behavior changes

Golden tests catch these issues.

---

## Specification

### Test Procedure

1. **Generate baseline** (before T-065 changes):
   ```bash
   git stash  # Save T-065 changes
   cargo build --release -j1
   for example in examples/0*; do
     ./target/release/powers run $example --seed 42 > golden/$(basename $example).txt
   done
   git stash pop  # Restore T-065 changes
   ```

2. **Build with new path**:
   ```bash
   cargo build --release -j1
   ```

3. **Compare outputs**:
   ```bash
   for example in examples/0*; do
     diff -u golden/$(basename $example).txt \
       <(./target/release/powers run $example --seed 42) \
       || echo "MISMATCH: $example"
   done
   ```

4. **Verify no differences**: All outputs must match exactly.

---

## Acceptance Criteria

- [ ] Baseline golden outputs captured
- [ ] All examples produce identical output with new path
- [ ] No numerical differences detected
- [ ] Test documented and repeatable

---

## Implementation Guide

### Step 1: Create golden directory

```bash
mkdir -p golden
```

### Step 2: Generate baselines (if not already done)

If baselines already exist from Epic 1, verify they're still valid:

```bash
# Verify existing baselines
./scripts/golden-tests.sh verify
```

If no baselines exist:
```bash
# Build with old path (revert T-065 temporarily)
git stash
cargo build --release -j1

# Generate baselines
for example in examples/0*; do
  ./target/release/powers run "$example" --seed 42 --iterations 10 \
    > "golden/$(basename $example).txt" 2>&1
done

# Restore changes
git stash pop
```

### Step 3: Verify with new path

```bash
# Build with new path
cargo build --release -j1

# Compare
for example in examples/0*; do
  echo "Testing: $example"
  diff -u "golden/$(basename $example).txt" \
    <(./target/release/powers run "$example" --seed 42 --iterations 10 2>&1) \
    && echo "  PASS" || echo "  FAIL"
done
```

### Step 4: Investigate any failures

If a test fails:
1. Check if failure is in timing output (acceptable difference)
2. Check if failure is in numerical values (NOT acceptable)
3. For numerical differences, bisect to find cause

---

## Testing Requirements

### Examples to Test

| Example | Description | Expected |
|---------|-------------|----------|
| `01-deterministic` | Single scenario | Must match |
| `02-stochastic` | Multiple scenarios | Must match |
| `03-with-inflow` | Inflow uncertainty | Must match |
| `04-autoregressive` | AR dynamics | Must match |
| `05-large-scale-brazilian` | Real-world case | Must match |

### Acceptable Differences

- Timing values (may vary between runs)
- Memory usage reports

### Unacceptable Differences

- Lower bounds
- Upper bounds
- Cut counts
- Convergence iterations
- Any numerical output

---

## Pitfalls to Avoid

- ⚠️ **Seed must match**: Always use `--seed 42`
- ⚠️ **Iteration count**: Use same iteration count for baseline and test
- ⚠️ **Build configuration**: Use same optimization level (release)
- ⚠️ **Thread count**: Consider setting `RAYON_NUM_THREADS=1` for maximum reproducibility

---

## Effort Estimate

**Points**: 2  
**Confidence**: High  
**Rationale**: Straightforward comparison if baselines exist. Main work is investigation if differences found.

---

## Definition of Done

- [ ] All examples produce identical output
- [ ] Test procedure documented
- [ ] No numerical regressions
