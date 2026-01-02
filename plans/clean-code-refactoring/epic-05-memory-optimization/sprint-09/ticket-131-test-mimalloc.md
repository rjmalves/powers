# [T-131] Test mimalloc Allocator RSS Behavior

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Status**: ✅ Complete (NOT RECOMMENDED)
> **Dependencies**: T-130
> **Blocks**: T-135

---

## Context

### Background

mimalloc is already configured as an optional dependency in Cargo.toml. This ticket tests whether enabling mimalloc resolves the RSS growth issue identified in Sprint 8.

### Current State

```toml
# Cargo.toml
mimalloc = { version = "0.1", optional = true }

[features]
mimalloc = ["dep:mimalloc"]
```

```rust
// src/main.rs
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

## Specification

### Test Procedure

1. Build with mimalloc: `cargo build --release --features mimalloc`
2. Run RSS stability test from T-130
3. Record all snapshots and analysis
4. Compare to glibc baseline (Sprint 8 data)

### Expected Behavior

mimalloc should return freed memory to the OS more aggressively than glibc:
- RSS should decrease (or stabilize) after `finalize_iteration()` drops Models
- Inter-iteration delta should be near zero after warmup

### Metrics to Capture

| Metric | glibc (Sprint 8) | mimalloc (Target) |
|--------|------------------|-------------------|
| RSS after 20 iter | 1,045 MB | < 600 MB |
| Avg inter-iteration delta | +25 MB | ±5 MB |
| Is stable after warmup | No | **Yes** |

## Acceptance Criteria

- [ ] Build succeeds with `--features mimalloc`
- [ ] RSS stability test runs to completion
- [ ] RSS snapshots collected for all 20 iterations
- [ ] Analysis document created with comparison to glibc
- [ ] Result recorded: PASS (stable) or FAIL (still grows)

## Implementation Guide

### Suggested Approach

1. Verify mimalloc builds: `cargo build --release --features mimalloc`
2. Run RSS test: `cargo test --release --features mimalloc test_rss_stability -- --nocapture`
3. Capture output to `docs/MIMALLOC_RSS_ANALYSIS.md`
4. Parse results and update comparison table

### Commands

```bash
# Build with mimalloc
cargo build --release --features mimalloc

# Run RSS stability test
cargo test --release --features mimalloc test_rss_stability -- --nocapture 2>&1 | tee mimalloc_rss.log

# Quick sanity check
cargo run --release --features mimalloc -- run examples/05-large-scale-brazilian
```

### Analysis Template

```markdown
# mimalloc RSS Analysis

## Configuration
- Allocator: mimalloc 0.1.x
- Example: 05-large-scale-brazilian
- Iterations: 20, Forward passes: 4, Threads: 1

## Results

| Iteration | RSS Start (KB) | RSS End (KB) | Delta (KB) |
|-----------|----------------|--------------|------------|
| 1 | ... | ... | ... |
| ... | ... | ... | ... |

## Comparison to glibc

| Metric | glibc | mimalloc | Improvement |
|--------|-------|----------|-------------|
| Final RSS | 1,045 MB | X MB | Y% |
| Stable after warmup | No | ? | - |

## Conclusion

[PASS/FAIL]: mimalloc [does/does not] solve RSS growth.
```

### Pitfalls to Avoid

- ⚠️ Ensure release build (debug has different allocation patterns)
- ⚠️ Use single thread to isolate memory behavior
- ⚠️ Run multiple times to ensure consistency

## Testing Requirements

### Integration Tests

- [ ] RSS stability test passes with mimalloc feature
- [ ] All existing tests pass with mimalloc feature
- [ ] No numerical differences in golden tests

### Performance Tests

- [ ] Run `sddp_e2e` benchmark with mimalloc
- [ ] Compare to baseline (no regression > 5%)

## Documentation Requirements

- [ ] Create `docs/MIMALLOC_RSS_ANALYSIS.md` with results
- [ ] Record metrics in sprint comparison table

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Infrastructure exists, just need to run tests and analyze

## Definition of Done

- [ ] mimalloc RSS behavior measured and documented
- [ ] Comparison to glibc baseline complete
- [ ] Result (PASS/FAIL) recorded for allocator selection
- [ ] All tests pass with mimalloc enabled
