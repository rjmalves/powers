# [T-133] Test jemalloc Allocator RSS Behavior

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-130, T-132
> **Blocks**: T-135 (optional - see fallback)
> **Status**: ✅ Complete (NOT RECOMMENDED)

---

## ⚠️ CRITICAL: Known Compilation Issues

### Problem

jemalloc test compilation has caused **SIGBUS crashes** and **system resource exhaustion**:
- Crash during compilation of `test_infrastructure` test target
- I/O errors due to system instability
- Environment requires recovery after crash

### Root Cause

- jemalloc compiles native C code with many platform-specific configurations
- Parallel compilation compounds memory pressure
- Full test suite triggers recompilation of entire dependency tree with jemalloc

### This Ticket is OPTIONAL

If jemalloc continues to cause instability, **skip this ticket** and proceed with:
1. T-131 (mimalloc) results for allocator selection
2. T-134 (malloc_trim) as backup strategy

---

## Context

### Background

After adding jemalloc in T-132, this ticket tests its RSS behavior using the same harness from T-130. jemalloc is known for good memory release characteristics and is used in production by Firefox, Redis, and many other systems.

### Current State

- T-132 added jemalloc as optional dependency ✅
- RSS harness from T-130 is available ✅
- Build with `cargo build -j1 --release --features jemalloc` succeeded ✅
- Test compilation crashed with SIGBUS ❌

## Specification

### Safe Test Procedure (REVISED)

**CRITICAL**: Always use `-j1` to limit parallelism and prevent SIGBUS.

#### Step 1: Check System Resources

```bash
# Ensure sufficient resources before attempting build
free -h
df -h /tmp
# Should have >2GB free RAM and >1GB free disk
```

#### Step 2: Build Binary Only (Already Done)

```bash
# This succeeded in previous session
cargo build -j1 --release --features jemalloc
```

#### Step 3: Test with Specific Test Target Only

```bash
# Do NOT run `cargo test` - it compiles all test targets
# Instead, run only the specific RSS test
cargo test -j1 --release --features jemalloc \
    --test test_rss_stability test_rss_stability_full \
    -- --nocapture --ignored
```

#### Step 4: If Step 3 Fails - Use Binary Directly

```bash
# Run the binary directly to measure RSS manually
./target/release/powers run examples/05-large-scale-brazilian \
    --max-iterations 20 \
    --forward-count 4 \
    --threads 1 \
    --log-level debug 2>&1 | tee jemalloc_rss_manual.log

# Extract RSS from log output
grep -i "rss\|memory" jemalloc_rss_manual.log
```

#### Step 5: If All Else Fails - Skip This Ticket

Document the failure and proceed with mimalloc:
```markdown
## jemalloc Test Result: SKIPPED

**Reason**: Compilation causes SIGBUS/system instability
**Recommendation**: Use mimalloc as default allocator
**Future Work**: Investigate jemalloc issues in isolated environment
```

### Expected Behavior (If Successful)

jemalloc should return freed memory to the OS:
- RSS should decrease (or stabilize) after `finalize_iteration()`
- Inter-iteration delta should be near zero after warmup

### Metrics to Capture

| Metric | glibc | mimalloc | jemalloc (Target) |
|--------|-------|----------|-------------------|
| RSS after 20 iter | 1,045 MB | ? | < 600 MB |
| Avg inter-iteration delta | +25 MB | ? | ±5 MB |
| Is stable after warmup | No | ? | **Yes** |

## Acceptance Criteria

**SUCCESS PATH:**
- [ ] Build succeeds with `--features jemalloc` (using -j1)
- [ ] RSS stability test runs to completion
- [ ] RSS snapshots collected for all 20 iterations
- [ ] Analysis document created with comparison
- [ ] Result recorded: PASS (stable) or FAIL (still grows)

**FALLBACK PATH (if jemalloc unstable):**
- [ ] Attempts documented with error details
- [ ] Decision recorded: SKIP jemalloc, proceed with mimalloc
- [ ] T-135 updated to exclude jemalloc from comparison

## Implementation Guide

### Pre-Flight Checklist

Before attempting jemalloc tests:

```bash
# 1. Check available memory (need >2GB free)
free -h | grep Mem

# 2. Check disk space (need >1GB in /tmp)
df -h /tmp

# 3. Check no other heavy processes running
top -bn1 | head -20

# 4. Consider running in screen/tmux for recovery
screen -S jemalloc-test
```

### Safe Build Commands

```bash
# ALWAYS use -j1 to prevent resource exhaustion
cargo build -j1 --release --features jemalloc

# Test ONLY the specific RSS test, not the full suite
cargo test -j1 --release --features jemalloc \
    --test test_rss_stability test_rss_stability_full \
    -- --nocapture --ignored 2>&1 | tee jemalloc_rss.log
```

### Manual RSS Measurement (Fallback)

If test compilation fails, measure RSS manually:

```bash
#!/bin/bash
# run_jemalloc_rss_test.sh

# Run powers with jemalloc in background
./target/release/powers run examples/05-large-scale-brazilian \
    --max-iterations 20 \
    --forward-count 4 \
    --threads 1 \
    --log-level debug &
PID=$!

# Sample RSS every 5 seconds
while kill -0 $PID 2>/dev/null; do
    RSS=$(grep VmRSS /proc/$PID/status 2>/dev/null | awk '{print $2}')
    echo "$(date +%H:%M:%S) RSS: ${RSS:-N/A} kB"
    sleep 5
done

wait $PID
echo "Process completed with exit code: $?"
```

### Analysis Template

```markdown
# jemalloc RSS Analysis

## Test Status

- [ ] Compilation succeeded
- [ ] Test ran to completion
- [ ] No system instability

## Configuration
- Allocator: jemalloc (tikv-jemallocator 0.6.x)
- Example: 05-large-scale-brazilian
- Iterations: 20, Forward passes: 4, Threads: 1
- Build command: `cargo build -j1 --release --features jemalloc`

## Results

| Iteration | RSS Start (KB) | RSS End (KB) | Delta (KB) |
|-----------|----------------|--------------|------------|
| 1 | ... | ... | ... |
| ... | ... | ... | ... |

## Comparison

| Metric | glibc | mimalloc | jemalloc |
|--------|-------|----------|----------|
| Final RSS | 1,045 MB | X MB | Y MB |
| Stable | No | ? | ? |

## Conclusion

[PASS/FAIL/SKIP]: jemalloc [does/does not/could not be tested] solve RSS growth.

### If SKIP:
Reason: [Description of failure]
Recommendation: Proceed with mimalloc as default allocator
```

### Pitfalls to Avoid

- ⚠️ **NEVER use default parallelism** - always `-j1`
- ⚠️ **NEVER run full test suite** - target specific tests only
- ⚠️ **Check resources first** - abort if low memory/disk
- ⚠️ **Use screen/tmux** - allows recovery from disconnection
- ⚠️ **Be prepared to skip** - jemalloc is optional, mimalloc is backup

## Testing Requirements

### Integration Tests

- [ ] RSS stability test with jemalloc (if compilation succeeds)
- [ ] Basic smoke test with jemalloc binary
- [ ] No numerical differences in golden tests (optional, skip if unstable)

### Performance Tests (Optional)

- [ ] Run `sddp_e2e` benchmark with jemalloc (only if stable)
- [ ] Compare to baseline

## Documentation Requirements

- [ ] Create `docs/JEMALLOC_RSS_ANALYSIS.md` with results (or skip documentation)
- [ ] Document any compilation issues encountered
- [ ] Update sprint comparison table

## Effort Estimate

**Points**: 3
**Confidence**: Low (due to compilation issues)
**Rationale**: Known instability requires careful approach; may need to skip entirely

## Definition of Done

**If jemalloc works:**
- [ ] jemalloc RSS behavior measured and documented
- [ ] Comparison to glibc and mimalloc complete
- [ ] Result (PASS/FAIL) recorded for allocator selection
- [ ] All tests pass with jemalloc enabled

**If jemalloc is skipped:**
- [ ] Attempts documented with error details
- [ ] SKIP decision recorded with rationale
- [ ] T-135 updated to proceed without jemalloc data
- [ ] Recommendation: Use mimalloc as default
