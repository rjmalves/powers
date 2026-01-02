# [T-126] Memory Regression Test

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 8 (Revised): Per-Iteration Model Architecture](./00-sprint-overview.md)
> **Dependencies**: T-125
> **Blocks**: None
> **Priority**: 6 (Production Integration)
> **Status**: ⚠️ DHAT Shows Regression (Expected)

## ⚠️ DHAT Analysis (2025-12-31)

### Comparison Results

| Metric | Post-Sprint 7 | Sprint 8 | Change |
|--------|---------------|----------|--------|
| Total Bytes Allocated | 45.43 GB | 50.32 GB | **+10.8%** |
| Total Allocation Blocks | 43.3 M | 46.6 M | **+7.6%** |
| Sum of Max Bytes | 468.4 MB | 526.6 MB | **+12.4%** |
| Allocation Sites | 12,651 | 15,238 | **+20.4%** |

### Analysis

**The DHAT regression is EXPECTED** for the following reasons:

1. **Per-iteration Model creation**: Instead of creating 1 Model kept forever, we now create N Models (one per iteration). This increases cumulative allocations.

2. **More allocation sites**: The per-iteration lifecycle code paths add allocation sites.

3. **DHAT measures cumulative allocations**: It cannot tell us about peak RSS or memory reclamation.

### What This Means

- ✅ The architecture change is working as designed
- ⚠️ DHAT is the wrong tool to validate RSS stability
- ❓ RSS stability objective needs different measurement

### Recommended Measurement Approach

1. **`/usr/bin/time -v`**: Get actual peak RSS
2. **`/proc/self/status` logging**: Track VmRSS at iteration boundaries
3. **Comparative testing**: Run same workload before/after Sprint 8, compare peak RSS

## Files to Read Before Starting

- `tests/test_iteration_lifecycle.rs` - Existing lifecycle tests (T-119, T-120)
- `docs/HIGHS_RSS_MEMORY_INVESTIGATION.md` - Memory analysis documentation

---

## Context

### Background

After integration is validated (T-125), we need a permanent regression test to ensure future changes don't reintroduce memory growth issues.

This test should:
1. Run a meaningful training workload
2. Measure RSS after each iteration
3. Assert that RSS is stable (not monotonically growing)
4. Run in CI to catch regressions

---

## Specification

### Test Design

```rust
#[test]
#[ignore] // Long-running, run explicitly in CI
fn test_memory_regression_training_loop() {
    // Use a small but meaningful problem
    let system = load_test_system("fixtures/deterministic_benchmark");
    
    // Run 10 iterations
    let config = TrainingConfig {
        max_iterations: 10,
        num_forward_passes: 5,
        ..Default::default()
    };
    
    // Capture RSS after each iteration
    let mut rss_samples = Vec::new();
    
    // Custom training with RSS sampling
    for iteration in 1..=10 {
        run_single_iteration(&mut algorithm, &config)?;
        rss_samples.push(get_current_rss_kb());
    }
    
    // Assert: RSS should not grow more than 20% from iteration 3 onwards
    // (allow warmup in first 2 iterations)
    let baseline_rss = rss_samples[2];
    for (i, rss) in rss_samples[3..].iter().enumerate() {
        let growth_pct = (*rss as f64 - baseline_rss as f64) / baseline_rss as f64 * 100.0;
        assert!(
            growth_pct < 20.0,
            "Iteration {}: RSS grew {:.1}% from baseline (regression detected)",
            i + 4, growth_pct
        );
    }
    
    // Assert: Last 3 iterations should have similar RSS (stable)
    let last_three = &rss_samples[rss_samples.len() - 3..];
    let max_rss = last_three.iter().max().unwrap();
    let min_rss = last_three.iter().min().unwrap();
    let variance_pct = (*max_rss - *min_rss) as f64 / *min_rss as f64 * 100.0;
    assert!(
        variance_pct < 10.0,
        "RSS not stable in last 3 iterations: {:.1}% variance",
        variance_pct
    );
}
```

### Test Characteristics

- **Platform**: Linux only (uses `/proc/self/statm`)
- **Duration**: ~30-60 seconds
- **Frequency**: Run in CI on main branch, not on every PR
- **Threshold**: 20% growth tolerance (accounts for allocator behavior)

---

## Acceptance Criteria

- [ ] Test implemented and passing
- [ ] Test runs on Linux CI
- [ ] Test catches intentional regression (verified by temporarily breaking lifecycle)
- [ ] Test documented in test README
- [ ] CI configuration updated (if needed)

---

## Implementation Guide

### Step 1: Create Test File

Add to existing `tests/test_iteration_lifecycle.rs` or create new file `tests/test_memory_regression.rs`.

### Step 2: Helper Functions

```rust
/// Get current process RSS in KB (Linux only)
#[cfg(target_os = "linux")]
fn get_current_rss_kb() -> usize {
    use std::fs;
    if let Ok(status) = fs::read_to_string("/proc/self/statm") {
        if let Some(pages) = status.split_whitespace().nth(1) {
            if let Ok(p) = pages.parse::<usize>() {
                return p * 4; // 4KB pages
            }
        }
    }
    0
}

#[cfg(not(target_os = "linux"))]
fn get_current_rss_kb() -> usize {
    0 // Skip on non-Linux
}
```

### Step 3: Test Fixture

Use existing benchmark fixtures or create a dedicated memory test fixture:

```rust
fn create_memory_test_algorithm() -> SddpAlgorithm {
    // Use fixtures/benchmarks.rs deterministic_single_reservoir
    // or similar small-but-meaningful problem
}
```

### Step 4: CI Configuration

Add to GitHub Actions or equivalent:

```yaml
- name: Memory Regression Test
  if: github.ref == 'refs/heads/main'
  run: cargo test --release --test test_memory_regression -- --ignored
```

---

## Testing Requirements

### Verification

- [ ] Test passes on current code (after integration)
- [ ] Test fails when lifecycle is disabled (intentional regression)
- [ ] Test skips gracefully on non-Linux

### Edge Cases

- [ ] Test handles first-iteration warmup RSS spike
- [ ] Test tolerates small variations from allocator behavior
- [ ] Test doesn't flake on CI

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Similar to existing T-120 RSS test, just more comprehensive

---

## Definition of Done

- [ ] Test implemented
- [ ] Test passes with lifecycle integration
- [ ] Test fails without lifecycle (verified)
- [ ] CI configured (if applicable)
- [ ] Documentation updated
- [ ] Code reviewed and merged
