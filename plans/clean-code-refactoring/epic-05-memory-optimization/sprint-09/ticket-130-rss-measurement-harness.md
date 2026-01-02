# [T-130] Create RSS Measurement Test Harness

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-131, T-133, T-134

---

## Context

### Background

Sprint 8 added basic RSS logging to the training loop, but we need a dedicated test harness that can:
1. Measure RSS at precise points during training
2. Compare RSS between iteration boundaries
3. Detect RSS growth patterns (monotonic vs stable)
4. Output structured data for allocator comparison

### Current State

- RSS logging exists in `src/sddp/mod.rs` at iteration start/end (debug level)
- `tests/test_rss_monitoring.rs` exists but is basic
- No structured comparison or pass/fail criteria

## Specification

### Inputs

- Training run on `examples/05-large-scale-brazilian`
- 20 iterations, 4 forward passes, 1 thread (isolate memory behavior)
- Seed: 42 (deterministic)

### Outputs

- `RssSnapshot` struct with: iteration, phase (start/end), rss_kb, timestamp
- `RssAnalysis` struct with: deltas, growth_rate, is_stable flag
- Test that passes if RSS is stable, fails otherwise

### Behavior

```rust
struct RssSnapshot {
    iteration: usize,
    phase: RssPhase,  // Start, End
    rss_kb: u64,
    active_cuts: usize,
}

struct RssAnalysis {
    snapshots: Vec<RssSnapshot>,
    iteration_deltas: Vec<i64>,  // RSS end - RSS start for each iteration
    inter_iteration_deltas: Vec<i64>,  // RSS start[n+1] - RSS end[n]
    is_stable: bool,  // true if no monotonic growth after warmup
}
```

- **Warmup**: Iterations 1-2 (allow growth for HiGHS initialization)
- **Stable**: Iterations 3+ should have near-zero inter-iteration delta
- **Tolerance**: ±5% of iteration 3 RSS

### Error Handling

- Skip test on non-Linux (RSS measurement via `/proc/self/status`)
- Fail gracefully if example files not found

## Acceptance Criteria

- [ ] `RssSnapshot` and `RssAnalysis` structs implemented
- [ ] `measure_rss()` function reads `/proc/self/status` VmRSS
- [ ] `analyze_rss_stability()` determines if RSS is stable after warmup
- [ ] Test runs 20 iterations and outputs structured analysis
- [ ] Test currently FAILS (expected - proving the problem exists)
- [ ] Output includes per-iteration deltas for debugging

## Implementation Guide

### Suggested Approach

1. Create `tests/rss_harness.rs` with structs and measurement functions
2. Implement `measure_rss() -> Option<u64>` using `/proc/self/status`
3. Implement `RssAnalysis::from_snapshots()` to compute deltas
4. Implement `RssAnalysis::is_stable()` with warmup and tolerance logic
5. Create integration test that runs training and collects snapshots
6. Assert `is_stable()` returns true (will fail with glibc, proving problem)

### Key Files to Modify

- `tests/rss_harness.rs` (new): Core measurement infrastructure
- `tests/test_rss_stability.rs` (new): Integration test using harness

### Code Skeleton

```rust
// tests/rss_harness.rs

#[derive(Debug, Clone)]
pub enum RssPhase {
    IterationStart,
    IterationEnd,
}

#[derive(Debug, Clone)]
pub struct RssSnapshot {
    pub iteration: usize,
    pub phase: RssPhase,
    pub rss_kb: u64,
    pub active_cuts: usize,
}

#[derive(Debug)]
pub struct RssAnalysis {
    pub snapshots: Vec<RssSnapshot>,
    pub warmup_iterations: usize,
}

impl RssAnalysis {
    pub fn is_stable(&self, tolerance_pct: f64) -> bool {
        // After warmup, check that RSS doesn't grow monotonically
        let post_warmup: Vec<_> = self.snapshots.iter()
            .filter(|s| s.iteration > self.warmup_iterations)
            .filter(|s| matches!(s.phase, RssPhase::IterationEnd))
            .collect();
        
        if post_warmup.len() < 2 {
            return true; // Not enough data
        }
        
        let baseline = post_warmup[0].rss_kb as f64;
        let tolerance = baseline * tolerance_pct;
        
        // Check no snapshot exceeds baseline + tolerance
        post_warmup.iter().all(|s| {
            (s.rss_kb as f64 - baseline).abs() <= tolerance
        })
    }
}

#[cfg(target_os = "linux")]
pub fn measure_rss() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse().ok();
            }
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
pub fn measure_rss() -> Option<u64> {
    None  // Not supported on this platform
}
```

### Pitfalls to Avoid

- ⚠️ Don't parse RSS from logs (use direct measurement)
- ⚠️ Don't use DHAT (measures allocations, not RSS)
- ⚠️ Ensure single-threaded to isolate memory behavior

## Testing Requirements

### Unit Tests

- [ ] `measure_rss()` returns Some on Linux, None on other platforms
- [ ] `RssAnalysis::is_stable()` returns true for flat data
- [ ] `RssAnalysis::is_stable()` returns false for monotonic growth

### Integration Tests

- [ ] Full training run captures snapshots at correct iteration boundaries
- [ ] Analysis correctly identifies current glibc behavior as unstable

## Documentation Requirements

- [ ] Doc comments on all public structs/functions
- [ ] Module doc explaining purpose and usage
- [ ] Example output format in comments

## Effort Estimate

**Points**: 3
**Confidence**: High
**Rationale**: Straightforward measurement and analysis logic, similar patterns exist in test_rss_monitoring.rs

## Definition of Done

- [ ] RSS measurement harness implemented
- [ ] Integration test runs and outputs structured analysis
- [ ] Test correctly fails with glibc (proving the problem)
- [ ] Output is human-readable for allocator comparison
- [ ] Code compiles on non-Linux (with graceful skip)
