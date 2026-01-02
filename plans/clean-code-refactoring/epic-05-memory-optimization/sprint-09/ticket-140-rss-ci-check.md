# [T-140] Add RSS Stability CI Check

> **Epic**: [Epic 5: Memory Optimization](../../00-epic-overview.md)
> **Sprint**: [Sprint 9: RSS Stabilization](./00-sprint-overview.md)
> **Dependencies**: T-137
> **Blocks**: None

---

## Context

### Background

To prevent future regressions in RSS stability, we need an automated check that runs as part of CI. This ensures that changes don't reintroduce memory growth issues.

### Current State

- RSS measurement harness exists (T-130)
- Manual testing confirms allocator works
- No automated CI check for RSS stability

## Specification

### CI Check Requirements

1. Run on every PR (or scheduled for expensive tests)
2. Execute RSS stability test with default allocator
3. Fail if RSS grows monotonically after warmup
4. Provide clear output for debugging

### Test Configuration

- Example: Small/medium example for CI speed (or use `expensive_tests` feature)
- Iterations: 10 (faster than 20, still catches growth)
- Pass criteria: RSS stable after iteration 2

### Implementation Options

**Option A: Integration Test (Recommended)**

Add to existing test suite, mark with `#[ignore]` for normal runs, enable in CI:

```rust
#[test]
#[ignore] // Run with: cargo test --features expensive_tests
fn test_rss_stability_ci() {
    // Use smaller example for CI speed
    // Assert RSS is stable
}
```

**Option B: Separate CI Job**

Add GitHub Actions workflow step:

```yaml
- name: RSS Stability Check
  run: cargo test --release --features expensive_tests test_rss_stability
```

## Acceptance Criteria

- [ ] RSS stability test integrated into test suite
- [ ] Test marked appropriately for CI execution
- [ ] Clear pass/fail output
- [ ] Test runs in reasonable time (<5 minutes)
- [ ] CI configuration updated (if separate job)

## Implementation Guide

### Suggested Approach

1. Refactor RSS test from T-130 to be CI-friendly
2. Use smaller example or fewer iterations for speed
3. Add `#[ignore]` attribute with feature flag
4. Update CI configuration to run test
5. Verify test catches regression (temporarily break, run test)

### Key Files to Modify

- `tests/test_rss_stability.rs`: Add CI-friendly test
- `.github/workflows/ci.yml` (if exists): Add test step
- `Cargo.toml`: Ensure `expensive_tests` feature exists

### Test Implementation

```rust
// tests/test_rss_stability.rs

/// CI check for RSS stability.
/// 
/// This test verifies that RSS is stable between iterations after warmup.
/// Run with: cargo test --release --features expensive_tests test_rss_stability_ci
#[test]
#[cfg_attr(not(feature = "expensive_tests"), ignore)]
fn test_rss_stability_ci() {
    use std::path::Path;
    
    // Skip on non-Linux (RSS measurement not available)
    if !cfg!(target_os = "linux") {
        eprintln!("Skipping RSS test on non-Linux platform");
        return;
    }
    
    // Use example 02 for faster CI (smaller than 05)
    let example_path = Path::new("examples/02-hydro-thermal");
    if !example_path.exists() {
        eprintln!("Example not found, skipping");
        return;
    }
    
    // Run training with RSS measurement
    let analysis = run_with_rss_measurement(example_path, 10);
    
    // Assert stability
    assert!(
        analysis.is_stable(0.05), // 5% tolerance
        "RSS is not stable after warmup:\n{:?}",
        analysis.summary()
    );
    
    println!("✓ RSS stability check passed");
    println!("  Final RSS: {} KB", analysis.final_rss_kb());
    println!("  Stable after iteration: {}", analysis.warmup_iterations);
}
```

### CI Configuration (if using GitHub Actions)

```yaml
# .github/workflows/ci.yml

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run tests
        run: cargo test --release
      
      - name: RSS Stability Check
        run: cargo test --release --features expensive_tests test_rss_stability_ci -- --nocapture
```

### Pitfalls to Avoid

- ⚠️ Don't run expensive RSS test on every `cargo test`
- ⚠️ Ensure test is deterministic (fixed seed)
- ⚠️ Provide clear failure message with debugging info

## Testing Requirements

### Verification

- [ ] Test passes with default allocator
- [ ] Test fails with system allocator (proving it catches regression)
- [ ] Test runs in reasonable time

### CI Integration

- [ ] Test runs in CI pipeline
- [ ] Failure blocks PR merge
- [ ] Clear output for debugging

## Documentation Requirements

- [ ] Document how to run RSS test locally
- [ ] Document CI behavior
- [ ] Add to CONTRIBUTING.md if exists

## Effort Estimate

**Points**: 3
**Confidence**: Medium
**Rationale**: CI integration may require debugging platform issues

## Definition of Done

- [ ] RSS stability test in test suite
- [ ] Test runs in CI
- [ ] Clear pass/fail criteria
- [ ] Documentation updated
- [ ] Verified to catch regression (manual test)
